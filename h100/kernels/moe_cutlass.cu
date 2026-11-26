#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cstdint>
#include <cmath>
#include <tuple>
#include <vector>

namespace moe_h100_impl {

constexpr int MOE_H = 2304;
constexpr int MOE_I = 1152;
constexpr int MOE_R = 16;
constexpr int MOE_E = 128;
constexpr int MOE_TOPK = 8;

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float block_max_256(float v) {
    __shared__ float sm[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) sm[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < 8) ? sm[lane] : 0.0f;
        v = warp_max(v);
        if (lane == 0) sm[0] = v;
    }
    __syncthreads();
    return sm[0];
}

__device__ __forceinline__ float block_sum_256(float v) {
    __shared__ float sm[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) sm[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < 8) ? sm[lane] : 0.0f;
        v = warp_sum(v);
        if (lane == 0) sm[0] = v;
    }
    __syncthreads();
    return sm[0];
}

__device__ __forceinline__ float fast_tanh(float x) {
    float y;
    asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 0.5f + 0.5f * fast_tanh(0.5f * x);
}

__device__ __forceinline__ uint8_t f32_to_e4m3(float v) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float dot16_s_g(const float* sZ, const __nv_bfloat16* gB) {
    uint4 v0 = *reinterpret_cast<const uint4*>(gB);
    uint4 v1 = *reinterpret_cast<const uint4*>(gB + 8);
    const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
    const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(p0[i]);
        acc += sZ[2 * i] * f.x + sZ[2 * i + 1] * f.y;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(p1[i]);
        acc += sZ[8 + 2 * i] * f.x + sZ[8 + 2 * i + 1] * f.y;
    }
    return acc;
}

// -----------------------------------------------------------------------------
// 1. Routing: Top-P filter + bincount + prefix sum + scatter indices
// -----------------------------------------------------------------------------
__global__ void moe_topp_bincount_kernel(
    const int64_t* __restrict__ top_k_index, // [N_tok, TOP_K]
    const float* __restrict__ top_k_weights, // [N_tok, TOP_K]
    int32_t* __restrict__ flat_e,            // [N_tok * TOP_K]
    float* __restrict__ w_eff,               // [N_tok, TOP_K]
    int32_t* __restrict__ counts,            // [E]
    int N_tok, float top_p, int k_min, int E, int top_k
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N_tok) return;

    int e_arr[8];
    float w_arr[8];
    float w_sum = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        e_arr[k] = static_cast<int>(top_k_index[(int64_t)t * top_k + k]);
        w_arr[k] = top_k_weights[(int64_t)t * top_k + k];
        w_sum += w_arr[k];
    }

    if (top_p >= 1.0f) {
        for (int k = 0; k < top_k; ++k) {
            flat_e[(int64_t)t * top_k + k] = e_arr[k];
            w_eff[(int64_t)t * top_k + k] = w_arr[k];
            atomicAdd(&counts[e_arr[k]], 1);
        }
        return;
    }

    // Sort descending by weight (tiny K=8 insertion/selection)
    int order[8];
    for (int k = 0; k < top_k; ++k) order[k] = k;
    for (int i = 0; i < top_k; ++i) {
        int best = i;
        for (int j = i + 1; j < top_k; ++j) {
            if (w_arr[order[j]] > w_arr[order[best]]) best = j;
        }
        int tmp = order[i]; order[i] = order[best]; order[best] = tmp;
    }

    float thresh = top_p * w_sum;
    float cum = 0.0f;
    bool active[8];
    for (int r = 0; r < top_k; ++r) {
        int idx = order[r];
        bool act = (r < k_min) || (cum < thresh);
        active[idx] = act;
        if (act) cum += w_arr[idx];
    }

    for (int k = 0; k < top_k; ++k) {
        if (active[k]) {
            flat_e[(int64_t)t * top_k + k] = e_arr[k];
            w_eff[(int64_t)t * top_k + k] = w_arr[k];
            atomicAdd(&counts[e_arr[k]], 1);
        } else {
            flat_e[(int64_t)t * top_k + k] = E;
            w_eff[(int64_t)t * top_k + k] = 0.0f;
        }
    }
}

__global__ void moe_prefix_sum_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ exp_offsets,
    int32_t* __restrict__ exp_ends,
    int32_t* __restrict__ write_ptr,
    int32_t* __restrict__ active_tokens,
    int E
) {
    // 1 block, 128 threads (E=128) — pad each expert's count to multiple of 16 for TMA/FP8 Grouped GEMM!
    __shared__ int32_t s_cnt[256];
    int tid = threadIdx.x;
    int32_t c = (tid < E) ? counts[tid] : 0;
    int32_t c_pad = (c + 15) & ~15;
    s_cnt[tid] = c_pad;
    __syncthreads();
    for (int off = 1; off < 256; off <<= 1) {
        int32_t val = (tid >= off) ? s_cnt[tid - off] : 0;
        __syncthreads();
        s_cnt[tid] += val;
        __syncthreads();
    }
    if (tid < E) {
        int32_t end_v = s_cnt[tid];
        int32_t start_v = end_v - c_pad;
        exp_offsets[tid] = start_v;
        exp_ends[tid] = end_v;
        write_ptr[tid] = start_v;
        if (tid == E - 1) active_tokens[0] = end_v;
    }
}

__global__ void moe_scatter_indices_kernel(
    const int32_t* __restrict__ flat_e,
    const float* __restrict__ w_eff,
    int32_t* __restrict__ write_ptr,
    int32_t* __restrict__ sorted_tok,
    int32_t* __restrict__ sorted_e,
    float* __restrict__ sorted_w,
    int32_t* __restrict__ inv_idx,
    int M_flat, int E, int top_k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M_flat) return;
    int e = flat_e[idx];
    if (e < E) {
        int pos = atomicAdd(&write_ptr[e], 1);
        sorted_tok[pos] = idx / top_k;
        sorted_e[pos] = e;
        sorted_w[pos] = w_eff[idx];
        inv_idx[idx] = pos;
    } else {
        inv_idx[idx] = -1;
    }
}

// -----------------------------------------------------------------------------
// 2. Quantize x [N_tok, 2304] -> FP8 + Gather to sorted [M_buf, 2304]
// -----------------------------------------------------------------------------
__global__ void moe_quant_x_fp8_2304_kernel(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ x_fp8,
    float* __restrict__ sx,
    int N_tok
) {
    int t = blockIdx.x;
    if (t >= N_tok) return;
    int tid = threadIdx.x; // 256 threads -> 9 elems/thread (256*9 = 2304)
    const __nv_bfloat16* row = x + (int64_t)t * MOE_H;
    float vals[9];
    float amax = 0.0f;
    #pragma unroll
    for (int it = 0; it < 9; ++it) {
        float v = __bfloat162float(row[tid + it * 256]);
        vals[it] = v;
        amax = fmaxf(amax, fabsf(v));
    }
    amax = block_max_256(amax);
    float scale = fmaxf(amax, 1e-8f) / 448.0f;
    float inv_s = 1.0f / scale;
    if (tid == 0) sx[t] = scale;
    uint8_t* out = x_fp8 + (int64_t)t * MOE_H;
    #pragma unroll
    for (int it = 0; it < 9; ++it) {
        out[tid + it * 256] = f32_to_e4m3(vals[it] * inv_s);
    }
}

__global__ void moe_gather_fp8_2304_kernel(
    const uint8_t* __restrict__ x_tok_fp8,
    const float* __restrict__ sx_tok,
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ active_tokens,
    uint8_t* __restrict__ x_sorted_fp8,
    float* __restrict__ sx_sorted
) {
    int m = blockIdx.x;
    if (m >= active_tokens[0]) return;
    int tid = threadIdx.x; // 144 threads (144 * 16B = 2304B exact!)
    int t = sorted_tok[m];
    if (t < 0) {
        if (tid == 0) sx_sorted[m] = 0.0f;
        if (tid < 144) reinterpret_cast<uint4*>(x_sorted_fp8 + (int64_t)m * MOE_H)[tid] = make_uint4(0, 0, 0, 0);
        return;
    }
    if (tid == 0) sx_sorted[m] = sx_tok[t];
    if (tid < 144) {
        const uint4* src = reinterpret_cast<const uint4*>(x_tok_fp8 + (int64_t)t * MOE_H);
        uint4* dst = reinterpret_cast<uint4*>(x_sorted_fp8 + (int64_t)m * MOE_H);
        dst[tid] = src[tid];
    }
}

// -----------------------------------------------------------------------------
// 3. WARP-PER-TOKEN 128-BIT VECTORIZED FWD Kernel 1:
//    8 warps (256 threads) = 8 consecutive tokens of the SAME expert e (L1 reuse of B_gu[e], A_dn!)
//    ZERO block_max __syncthreads()! Pure warp_max + 128-bit uint4 HBM transactions!
// -----------------------------------------------------------------------------
__global__ void moe_gu_lora_swiglu_fp8_fwd_kernel(
    __nv_bfloat16* __restrict__ gu,          // [M_buf, 2304]
    const __nv_bfloat16* __restrict__ z_gu,  // [N_tok, 16]
    const __nv_bfloat16* __restrict__ B_gu,  // [E, 2304, 16]
    const __nv_bfloat16* __restrict__ A_dn,  // [16, 1152]
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ sorted_e,
    const int32_t* __restrict__ active_tokens,
    uint8_t* __restrict__ act_fp8,           // [M_buf, 1152]
    float* __restrict__ s_act,               // [M_buf]
    __nv_bfloat16* __restrict__ z_dn,        // [M_buf, 16]
    float lora_scale,
    bool has_lora
) {
    const int warp = threadIdx.x >> 5; // 0..7
    const int lane = threadIdx.x & 31; // 0..31
    const int m = blockIdx.x * 8 + warp;
    if (m >= active_tokens[0]) return;

    const int t = sorted_tok[m];
    if (t < 0) {
        if (lane == 0) s_act[m] = 0.0f;
        uint8_t* out_fp8 = act_fp8 + (int64_t)m * MOE_I;
        // 1152 bytes = 72 uint4 -> 32 lanes x 3 iters
        for (int v = lane; v < 72; v += 32) {
            reinterpret_cast<uint4*>(out_fp8)[v] = make_uint4(0, 0, 0, 0);
        }
        if (has_lora && lane < MOE_R) z_dn[(int64_t)m * MOE_R + lane] = __float2bfloat16_rn(0.0f);
        return;
    }
    const int e = sorted_e[m];

    __shared__ float sZ[8][MOE_R];
    if (has_lora && lane < MOE_R) {
        sZ[warp][lane] = __bfloat162float(z_gu[(int64_t)t * MOE_R + lane]) * lora_scale;
    }
    __syncwarp();

    __nv_bfloat16* gu_row = gu + (int64_t)m * MOE_H;
    const __nv_bfloat16* B_e = has_lora ? (B_gu + (int64_t)e * MOE_H * MOE_R) : nullptr;

    // 1152 bf16 = 144 uint4 -> 32 lanes x 5 iters (max 5 uint4 = 40 floats per lane)
    float act_reg[5][8];
    float amax = 0.0f;
    float z_dn_acc[MOE_R];
    if (has_lora) {
        #pragma unroll
        for (int r = 0; r < MOE_R; ++r) z_dn_acc[r] = 0.0f;
    }

    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        int v = lane + it * 32;
        if (v < 144) {
            int i0 = v * 8;
            uint4 gv = reinterpret_cast<const uint4*>(gu_row)[v];
            uint4 uv = reinterpret_cast<const uint4*>(gu_row + MOE_I)[v];
            __nv_bfloat162* pg = reinterpret_cast<__nv_bfloat162*>(&gv);
            __nv_bfloat162* pu = reinterpret_cast<__nv_bfloat162*>(&uv);

            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 fg = __bfloat1622float2(pg[k]);
                float2 fu = __bfloat1622float2(pu[k]);
                int idx0 = i0 + 2 * k;
                int idx1 = idx0 + 1;
                if (has_lora) {
                    fg.x += dot16_s_g(sZ[warp], B_e + (int64_t)idx0 * MOE_R);
                    fg.y += dot16_s_g(sZ[warp], B_e + (int64_t)idx1 * MOE_R);
                    fu.x += dot16_s_g(sZ[warp], B_e + (int64_t)(MOE_I + idx0) * MOE_R);
                    fu.y += dot16_s_g(sZ[warp], B_e + (int64_t)(MOE_I + idx1) * MOE_R);
                    pg[k] = __float22bfloat162_rn(fg);
                    pu[k] = __float22bfloat162_rn(fu);
                }
                float a0 = (4.0f * fast_tanh(fg.x * 0.25f)) * fast_sigmoid(fg.x) * (25.0f * fast_tanh(fu.x * 0.04f));
                float a1 = (4.0f * fast_tanh(fg.y * 0.25f)) * fast_sigmoid(fg.y) * (25.0f * fast_tanh(fu.y * 0.04f));
                act_reg[it][2 * k]     = a0;
                act_reg[it][2 * k + 1] = a1;
                amax = fmaxf(amax, fmaxf(fabsf(a0), fabsf(a1)));
                if (has_lora) {
                    #pragma unroll
                    for (int r = 0; r < MOE_R; ++r) {
                        __nv_bfloat162 a2 = *reinterpret_cast<const __nv_bfloat162*>(A_dn + (int64_t)r * MOE_I + idx0);
                        float2 fa = __bfloat1622float2(a2);
                        z_dn_acc[r] += a0 * fa.x + a1 * fa.y;
                    }
                }
            }
            if (has_lora) {
                reinterpret_cast<uint4*>(gu_row)[v] = gv;
                reinterpret_cast<uint4*>(gu_row + MOE_I)[v] = uv;
            }
        }
    }

    amax = warp_max(amax);
    float scale = fmaxf(amax, 1e-8f) / 448.0f;
    float inv_s = 1.0f / scale;
    if (lane == 0) s_act[m] = scale;

    uint8_t* out_fp8 = act_fp8 + (int64_t)m * MOE_I;
    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        int v = lane + it * 32;
        if (v < 144) {
            uint2 pack;
            uint8_t* pb = reinterpret_cast<uint8_t*>(&pack);
            #pragma unroll
            for (int j = 0; j < 8; ++j) pb[j] = f32_to_e4m3(act_reg[it][j] * inv_s);
            reinterpret_cast<uint2*>(out_fp8)[v] = pack;
        }
    }

    if (has_lora) {
        #pragma unroll
        for (int r = 0; r < MOE_R; ++r) {
            float sum = warp_sum(z_dn_acc[r]);
            if (lane == 0) z_dn[(int64_t)m * MOE_R + r] = __float2bfloat16_rn(sum);
        }
    }
}

// -----------------------------------------------------------------------------
// 4. WMMA 64x64x16 Tensor-Core LoRA B add (4x reuse of B[e], 4x fewer CTAs!)
//    Grid: ((M_buf + 63) / 64, 2304 / 64 = 36), Block: 128 threads (4 warps)
//    Note: Every 16-row sub-tile belongs to 1 expert; we check e per 16-row sub-tile
//    and only reload sB[64][16] if e changes across the 4 sub-tiles (95% same expert!).
// -----------------------------------------------------------------------------
__global__ void moe_lora_b_add_wmma_kernel(
    __nv_bfloat16* __restrict__ out,         // [M_buf, 2304] updated in-place
    const __nv_bfloat16* __restrict__ z,     // [N_tok, 16] if use_sorted_tok else [M_buf, 16]
    const __nv_bfloat16* __restrict__ B_w,   // [E, 2304, 16]
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ sorted_e,
    const int32_t* __restrict__ active_tokens,
    float lora_scale,
    bool use_sorted_tok
) {
    using namespace nvcuda;
    const int m_cta = blockIdx.x * 64;
    const int act_n = active_tokens[0];
    if (m_cta >= act_n) return;

    const int n_base = blockIdx.y * 64; // 0, 64, ..., 2240
    const int tid = threadIdx.x;        // 0..127
    const int warp = tid >> 5;          // 0..3

    __shared__ __nv_bfloat16 sZ[16 * 16];
    __shared__ __nv_bfloat16 sB[64 * 16];
    __shared__ float sOut[16 * 64];
    __shared__ int sTok[16];

    int loaded_e = -1;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> frag_b;

    for (int sub = 0; sub < 4; ++sub) {
        int m_base = m_cta + sub * 16;
        if (m_base >= act_n) break;
        int e = sorted_e[m_base];
        if (e < 0) continue;

        if (tid < 16) sTok[tid] = sorted_tok[m_base + tid];
        __syncthreads();

        // Reload sB only when expert changes (usually once per 64-token CTA!)
        if (e != loaded_e) {
            int row_b = tid >> 1;
            int col_b = (tid & 1) * 8;
            const __nv_bfloat16* src = B_w + ((int64_t)e * MOE_H + (n_base + row_b)) * MOE_R + col_b;
            *reinterpret_cast<uint4*>(&sB[row_b * 16 + col_b]) = *reinterpret_cast<const uint4*>(src);
            loaded_e = e;
        }
        // Load sZ[16][16]
        {
            int idx = tid * 2;
            int u = idx >> 4;
            int r = idx & 15;
            int t_u = sTok[u];
            if (t_u >= 0) {
                int row_z = use_sorted_tok ? t_u : (m_base + u);
                *reinterpret_cast<__nv_bfloat162*>(&sZ[idx]) = *reinterpret_cast<const __nv_bfloat162*>(z + (int64_t)row_z * MOE_R + r);
            } else {
                *reinterpret_cast<__nv_bfloat162*>(&sZ[idx]) = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
            }
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_a;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;
        wmma::fill_fragment(frag_c, 0.0f);
        wmma::load_matrix_sync(frag_a, sZ, 16);
        wmma::load_matrix_sync(frag_b, sB + warp * 16 * 16, 16);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        wmma::store_matrix_sync(sOut + warp * 16, frag_c, 64, wmma::mem_row_major);
        __syncthreads();

        int u = tid >> 3;
        int c = (tid & 7) * 8;
        if (sTok[u] >= 0) {
            __nv_bfloat16* dst = out + (int64_t)(m_base + u) * MOE_H + n_base + c;
            uint4 ov = *reinterpret_cast<const uint4*>(dst);
            __nv_bfloat162* po = reinterpret_cast<__nv_bfloat162*>(&ov);
            const float* sp = &sOut[u * 64 + c];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                float2 f = __bfloat1622float2(po[k]);
                f.x += lora_scale * sp[2 * k];
                f.y += lora_scale * sp[2 * k + 1];
                po[k] = __float22bfloat162_rn(f);
            }
            *reinterpret_cast<uint4*>(dst) = ov;
        }
        __syncthreads();
    }
}

// -----------------------------------------------------------------------------
// 4b. WMMA 16x16x16 Tensor-Core LoRA dz = X_fp8_or_bf16 @ B[e] (K=2304, N=16)
//     Used for BOTH dz_dn = (dy_fp8 * s_dy) @ B_dn[e] AND dz_gu = (dgu_fp8 * s_dgu) @ B_gu[e]!
//     Grid: ((M_buf + 15) / 16), Block: 128 threads (4 warps split K=2304 into 4x576!)
// -----------------------------------------------------------------------------
__global__ void moe_lora_dz_fp8_wmma_kernel(
    const uint8_t* __restrict__ x_fp8,       // [M_buf, 2304] e4m3
    const float* __restrict__ sx,            // [M_buf]
    const __nv_bfloat16* __restrict__ B_w,   // [E, 2304, 16]
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ sorted_e,
    const int32_t* __restrict__ active_tokens,
    __nv_bfloat16* __restrict__ dz_out,      // [M_buf, 16]
    float lora_scale
) {
    using namespace nvcuda;
    const int m_base = blockIdx.x * 16;
    if (m_base >= active_tokens[0]) return;
    const int e = sorted_e[m_base];
    if (e < 0) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;   // 0..3
    const int lane = tid & 31;

    __shared__ __nv_bfloat16 sX[4][16 * 16];
    __shared__ __nv_bfloat16 sB[4][16 * 16];
    __shared__ float sRed[4][16 * 16];
    __shared__ float sScale[16];
    __shared__ int sTok[16];

    if (tid < 16) {
        sTok[tid] = sorted_tok[m_base + tid];
        sScale[tid] = (sTok[tid] >= 0) ? (sx[m_base + tid] * lora_scale) : 0.0f;
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);

    const __nv_bfloat16* B_e = B_w + (int64_t)e * MOE_H * MOE_R;

    // Each warp processes 36 K-tiles of 16 (36 * 16 = 576, 4 * 576 = 2304 exact!)
    int k_start = warp * 576;
    for (int k = k_start; k < k_start + 576; k += 16) {
        // Load 16x16 of x_fp8 -> dequant to bf16 in sX[warp] (32 lanes x 8 elems = 256)
        int r = lane >> 1;          // 0..15
        int c = (lane & 1) * 8;     // 0 or 8
        if (sTok[r] >= 0) {
            uint2 xv = *reinterpret_cast<const uint2*>(x_fp8 + (int64_t)(m_base + r) * MOE_H + k + c);
            const __nv_fp8_e4m3* px = reinterpret_cast<const __nv_fp8_e4m3*>(&xv);
            __nv_bfloat16* dst = &sX[warp][r * 16 + c];
            #pragma unroll
            for (int j = 0; j < 8; ++j) dst[j] = __float2bfloat16_rn(float(px[j]));
        } else {
            *reinterpret_cast<uint4*>(&sX[warp][r * 16 + c]) = make_uint4(0, 0, 0, 0);
        }
        // Load 16x16 of B_e[k..k+15, 0..15] -> sB[warp] (32 lanes x 8 bf16 = 1 uint4!)
        *reinterpret_cast<uint4*>(&sB[warp][r * 16 + c]) = *reinterpret_cast<const uint4*>(B_e + (int64_t)(k + r) * MOE_R + c);
        __syncwarp();

        wmma::load_matrix_sync(frag_a, sX[warp], 16);
        wmma::load_matrix_sync(frag_b, sB[warp], 16);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        __syncwarp();
    }

    wmma::store_matrix_sync(sRed[warp], frag_c, 16, wmma::mem_row_major);
    __syncthreads();

    // Reduce 4 warps -> dz_out[m_base + u, r] (256 elems -> 128 threads x 2)
    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        int idx = tid + it * 128;
        int u = idx >> 4;
        int c = idx & 15;
        if (sTok[u] >= 0) {
            float sum = sRed[0][idx] + sRed[1][idx] + sRed[2][idx] + sRed[3][idx];
            dz_out[(int64_t)(m_base + u) * MOE_R + c] = __float2bfloat16_rn(sum * sScale[u]);
        } else {
            dz_out[(int64_t)(m_base + u) * MOE_R + c] = __float2bfloat16_rn(0.0f);
        }
    }
}

// Exact BF16 WMMA dz_dn = (w * dy[t]) @ B_dn[e] * lora_scale (faster than FP8 dequant AND exact BF16!)
__global__ void moe_lora_dz_dy_bf16_wmma_kernel(
    const __nv_bfloat16* __restrict__ dy,    // [N_tok, 2304]
    const __nv_bfloat16* __restrict__ B_dn,  // [E, 2304, 16]
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ sorted_e,
    const float* __restrict__ sorted_w,
    const int32_t* __restrict__ active_tokens,
    __nv_bfloat16* __restrict__ dz_out,      // [M_buf, 16]
    float lora_scale
) {
    using namespace nvcuda;
    const int m_base = blockIdx.x * 16;
    if (m_base >= active_tokens[0]) return;
    const int e = sorted_e[m_base];
    if (e < 0) return;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ __nv_bfloat16 sX[4][16 * 16];
    __shared__ __nv_bfloat16 sB[4][16 * 16];
    __shared__ float sRed[4][16 * 16];
    __shared__ float sScale[16];
    __shared__ int sTok[16];

    if (tid < 16) {
        sTok[tid] = sorted_tok[m_base + tid];
        sScale[tid] = (sTok[tid] >= 0) ? (sorted_w[m_base + tid] * lora_scale) : 0.0f;
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);

    const __nv_bfloat16* B_e = B_dn + (int64_t)e * MOE_H * MOE_R;
    int k_start = warp * 576;
    for (int k = k_start; k < k_start + 576; k += 16) {
        int r = lane >> 1;
        int c = (lane & 1) * 8;
        int t_r = sTok[r];
        if (t_r >= 0) {
            *reinterpret_cast<uint4*>(&sX[warp][r * 16 + c]) = *reinterpret_cast<const uint4*>(dy + (int64_t)t_r * MOE_H + k + c);
        } else {
            *reinterpret_cast<uint4*>(&sX[warp][r * 16 + c]) = make_uint4(0, 0, 0, 0);
        }
        *reinterpret_cast<uint4*>(&sB[warp][r * 16 + c]) = *reinterpret_cast<const uint4*>(B_e + (int64_t)(k + r) * MOE_R + c);
        __syncwarp();

        wmma::load_matrix_sync(frag_a, sX[warp], 16);
        wmma::load_matrix_sync(frag_b, sB[warp], 16);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        __syncwarp();
    }

    wmma::store_matrix_sync(sRed[warp], frag_c, 16, wmma::mem_row_major);
    __syncthreads();

    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        int idx = tid + it * 128;
        int u = idx >> 4;
        int c = idx & 15;
        if (sTok[u] >= 0) {
            float sum = sRed[0][idx] + sRed[1][idx] + sRed[2][idx] + sRed[3][idx];
            dz_out[(int64_t)(m_base + u) * MOE_R + c] = __float2bfloat16_rn(sum * sScale[u]);
        } else {
            dz_out[(int64_t)(m_base + u) * MOE_R + c] = __float2bfloat16_rn(0.0f);
        }
    }
}

__global__ void moe_dn_lora_scatter_fwd_kernel(
    const __nv_bfloat16* __restrict__ y_exp, // [M_buf, 2304] (already includes LoRA!)
    const int32_t* __restrict__ inv_idx,     // [N_tok, TOP_K]
    const float* __restrict__ w_eff,         // [N_tok, TOP_K]
    __nv_bfloat16* __restrict__ y_out,       // [N_tok, 2304]
    int N_tok, int top_k
) {
    int t = blockIdx.x;
    if (t >= N_tok) return;
    int tid = threadIdx.x; // 288 threads? Use 256 threads with uint4 (288 uint4 = 2304 bf16)!
    // 256 threads: tid < 288 (1 or 2 uint4 per thread -> 100% coalesced 128-bit!)
    for (int v_idx = tid; v_idx < 288; v_idx += 256) {
        float acc[8] = {0.0f};
        for (int k = 0; k < top_k; ++k) {
            int pos = inv_idx[(int64_t)t * top_k + k];
            if (pos < 0) continue;
            float w = w_eff[(int64_t)t * top_k + k];
            uint4 yv = reinterpret_cast<const uint4*>(y_exp + (int64_t)pos * MOE_H)[v_idx];
            const __nv_bfloat162* py = reinterpret_cast<const __nv_bfloat162*>(&yv);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 f = __bfloat1622float2(py[i]);
                acc[2 * i]     += w * f.x;
                acc[2 * i + 1] += w * f.y;
            }
        }
        uint4 out_v;
        __nv_bfloat162* po = reinterpret_cast<__nv_bfloat162*>(&out_v);
        #pragma unroll
        for (int i = 0; i < 4; ++i) po[i] = __float22bfloat162_rn(make_float2(acc[2 * i], acc[2 * i + 1]));
        reinterpret_cast<uint4*>(y_out + (int64_t)t * MOE_H)[v_idx] = out_v;
    }
}

// -----------------------------------------------------------------------------
// 5. MEGA-FUSED BWD Kernel 1:
//    Gather dy[t] * w -> dy_fp8 [M, 2304] + s_dy [M] + d_w [N_tok, TOP_K] + dz_dn [M, 16]
// -----------------------------------------------------------------------------
__global__ void moe_bwd_gather_dw_quant_fp8_kernel(
    const __nv_bfloat16* __restrict__ dy,      // [N_tok, 2304]
    const __nv_bfloat16* __restrict__ y_exp,   // [M_buf, 2304] (already includes LoRA!)
    const __nv_bfloat16* __restrict__ B_dn,    // [E, 2304, 16]
    const int32_t* __restrict__ sorted_tok,
    const int32_t* __restrict__ sorted_e,
    const float* __restrict__ sorted_w,
    const int32_t* __restrict__ active_tokens,
    uint8_t* __restrict__ dy_fp8,              // [M_buf, 2304]
    float* __restrict__ s_dy,                  // [M_buf]
    float* __restrict__ dw_sorted,             // [M_buf]
    __nv_bfloat16* __restrict__ dz_dn,         // [M_buf, 16]
    float lora_scale, bool has_lora
) {
    int m = blockIdx.x;
    if (m >= active_tokens[0]) return;
    int tid = threadIdx.x; // 256 threads -> 9 elems/thread
    int t = sorted_tok[m];
    if (t < 0) {
        if (tid == 0) { s_dy[m] = 0.0f; dw_sorted[m] = 0.0f; }
        uint8_t* out = dy_fp8 + (int64_t)m * MOE_H;
        #pragma unroll
        for (int it = 0; it < 9; ++it) out[tid + it * 256] = 0;
        if (has_lora && tid < MOE_R) dz_dn[(int64_t)m * MOE_R + tid] = __float2bfloat16_rn(0.0f);
        return;
    }
    int e = sorted_e[m];
    float w = sorted_w[m];

    const __nv_bfloat16* dy_row = dy + (int64_t)t * MOE_H;
    const __nv_bfloat16* ye_row = y_exp + (int64_t)m * MOE_H;
    const __nv_bfloat16* B_e = has_lora ? (B_dn + (int64_t)e * MOE_H * MOE_R) : nullptr;

    float vals[9];
    float amax = 0.0f;
    float dw_acc = 0.0f;
    float dz_acc[MOE_R];
    if (has_lora) {
        #pragma unroll
        for (int r = 0; r < MOE_R; ++r) dz_acc[r] = 0.0f;
    }

    #pragma unroll
    for (int it = 0; it < 9; ++it) {
        int h = tid + it * 256;
        float g = __bfloat162float(dy_row[h]);
        float yv = __bfloat162float(ye_row[h]);
        dw_acc += g * yv;
        float gs = g * w;
        vals[it] = gs;
        amax = fmaxf(amax, fabsf(gs));
        if (has_lora) {
            uint4 b0 = *reinterpret_cast<const uint4*>(B_e + (int64_t)h * MOE_R);
            uint4 b1 = *reinterpret_cast<const uint4*>(B_e + (int64_t)h * MOE_R + 8);
            const __nv_bfloat162* p0 = reinterpret_cast<const __nv_bfloat162*>(&b0);
            const __nv_bfloat162* p1 = reinterpret_cast<const __nv_bfloat162*>(&b1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 f = __bfloat1622float2(p0[i]);
                dz_acc[2 * i] += gs * f.x; dz_acc[2 * i + 1] += gs * f.y;
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 f = __bfloat1622float2(p1[i]);
                dz_acc[8 + 2 * i] += gs * f.x; dz_acc[8 + 2 * i + 1] += gs * f.y;
            }
        }
    }

    amax = block_max_256(amax);
    float scale = fmaxf(amax, 1e-8f) / 448.0f;
    float inv_s = 1.0f / scale;
    if (tid == 0) s_dy[m] = scale;

    uint8_t* out = dy_fp8 + (int64_t)m * MOE_H;
    #pragma unroll
    for (int it = 0; it < 9; ++it) {
        out[tid + it * 256] = f32_to_e4m3(vals[it] * inv_s);
    }

    float dw_tot = block_sum_256(dw_acc);
    if (tid == 0) dw_sorted[m] = dw_tot;

    if (has_lora) {
        __shared__ float sRed[8][MOE_R];
        int warp = tid >> 5;
        int lane = tid & 31;
        #pragma unroll
        for (int r = 0; r < MOE_R; ++r) {
            float v = warp_sum(dz_acc[r]);
            if (lane == 0) sRed[warp][r] = v;
        }
        __syncthreads();
        if (tid < MOE_R) {
            float sum = 0.0f;
            #pragma unroll
            for (int w_i = 0; w_i < 8; ++w_i) sum += sRed[w_i][tid];
            dz_dn[(int64_t)m * MOE_R + tid] = __float2bfloat16_rn(sum * lora_scale);
        }
    }
}

__global__ void moe_unsort_dw_kernel(
    const float* __restrict__ dw_sorted,
    const int32_t* __restrict__ inv_idx,
    float* __restrict__ dw_out,
    int M_flat
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M_flat) return;
    int pos = inv_idx[idx];
    dw_out[idx] = (pos >= 0) ? dw_sorted[pos] : 0.0f;
}

// -----------------------------------------------------------------------------
// 6. MEGA-FUSED BWD Kernel 2:
//    d_act [M, 1152] + dz_dn[m] @ A_dn -> SoftcappedSwiGLU_bwd(gu) -> dgu_fp8 [M, 2304] + s_dgu [M] + dz_gu [M, 16]
// -----------------------------------------------------------------------------
__global__ void moe_bwd_swiglu_quant_fp8_kernel(
    const __nv_bfloat16* __restrict__ gu,     // [M_buf, 2304]
    const __nv_bfloat16* __restrict__ d_act,  // [M_buf, 1152]
    const __nv_bfloat16* __restrict__ dz_dn,  // [M_buf, 16]
    const __nv_bfloat16* __restrict__ A_dn,   // [16, 1152]
    const __nv_bfloat16* __restrict__ B_gu,   // [E, 2304, 16]
    const int32_t* __restrict__ sorted_e,
    const int32_t* __restrict__ active_tokens,
    uint8_t* __restrict__ dgu_fp8,            // [M_buf, 2304]
    float* __restrict__ s_dgu,                // [M_buf]
    __nv_bfloat16* __restrict__ dz_gu,        // [M_buf, 16]
    float lora_scale, bool has_lora
) {
    int m = blockIdx.x;
    if (m >= active_tokens[0]) return;
    int tid = threadIdx.x; // 256 threads -> 5 iters cover 1152
    int e = sorted_e[m];
    if (e < 0) {
        if (tid == 0) s_dgu[m] = 0.0f;
        uint8_t* out = dgu_fp8 + (int64_t)m * MOE_H;
        #pragma unroll
        for (int it = 0; it < 5; ++it) {
            int i = tid + it * 256;
            if (i < MOE_I) { out[i] = 0; out[MOE_I + i] = 0; }
        }
        return;
    }

    __shared__ float sDZdn[MOE_R];
    if (has_lora && tid < MOE_R) {
        sDZdn[tid] = __bfloat162float(dz_dn[(int64_t)m * MOE_R + tid]);
    }
    __syncthreads();

    const __nv_bfloat16* gu_row = gu + (int64_t)m * MOE_H;
    const __nv_bfloat16* da_row = d_act + (int64_t)m * MOE_I;
    const __nv_bfloat16* B_e = has_lora ? (B_gu + (int64_t)e * MOE_H * MOE_R) : nullptr;

    float dg_reg[5], du_reg[5];
    float amax = 0.0f;
    float dz_gu_acc[MOE_R];
    if (has_lora) {
        #pragma unroll
        for (int r = 0; r < MOE_R; ++r) dz_gu_acc[r] = 0.0f;
    }

    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        int i = tid + it * 256;
        if (i < MOE_I) {
            float g = __bfloat162float(gu_row[i]);
            float u = __bfloat162float(gu_row[MOE_I + i]);
            float da = __bfloat162float(da_row[i]);
            if (has_lora) {
                float add = 0.0f;
                #pragma unroll
                for (int r = 0; r < MOE_R; ++r) add += sDZdn[r] * __bfloat162float(A_dn[(int64_t)r * MOE_I + i]);
                da += add;
            }
            float tg = fast_tanh(g * 0.25f);
            float g_cap = 4.0f * tg;
            float sig = fast_sigmoid(g);
            float tu = fast_tanh(u * 0.04f);
            float u_cap = 25.0f * tu;
            float du = da * (g_cap * sig) * (1.0f - tu * tu);
            float dg = (da * u_cap) * (sig * (1.0f - tg * tg) + g_cap * sig * (1.0f - sig));
            dg_reg[it] = dg;
            du_reg[it] = du;
            amax = fmaxf(amax, fmaxf(fabsf(dg), fabsf(du)));

            if (has_lora) {
                uint4 bg0 = *reinterpret_cast<const uint4*>(B_e + (int64_t)i * MOE_R);
                uint4 bg1 = *reinterpret_cast<const uint4*>(B_e + (int64_t)i * MOE_R + 8);
                uint4 bu0 = *reinterpret_cast<const uint4*>(B_e + (int64_t)(MOE_I + i) * MOE_R);
                uint4 bu1 = *reinterpret_cast<const uint4*>(B_e + (int64_t)(MOE_I + i) * MOE_R + 8);
                const __nv_bfloat162* pg0 = reinterpret_cast<const __nv_bfloat162*>(&bg0);
                const __nv_bfloat162* pg1 = reinterpret_cast<const __nv_bfloat162*>(&bg1);
                const __nv_bfloat162* pu0 = reinterpret_cast<const __nv_bfloat162*>(&bu0);
                const __nv_bfloat162* pu1 = reinterpret_cast<const __nv_bfloat162*>(&bu1);
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float2 fg = __bfloat1622float2(pg0[k]);
                    float2 fu = __bfloat1622float2(pu0[k]);
                    dz_gu_acc[2 * k]     += dg * fg.x + du * fu.x;
                    dz_gu_acc[2 * k + 1] += dg * fg.y + du * fu.y;
                }
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float2 fg = __bfloat1622float2(pg1[k]);
                    float2 fu = __bfloat1622float2(pu1[k]);
                    dz_gu_acc[8 + 2 * k]     += dg * fg.x + du * fu.x;
                    dz_gu_acc[8 + 2 * k + 1] += dg * fg.y + du * fu.y;
                }
            }
        }
    }

    amax = block_max_256(amax);
    float scale = fmaxf(amax, 1e-8f) / 448.0f;
    float inv_s = 1.0f / scale;
    if (tid == 0) s_dgu[m] = scale;

    uint8_t* out = dgu_fp8 + (int64_t)m * MOE_H;
    #pragma unroll
    for (int it = 0; it < 5; ++it) {
        int i = tid + it * 256;
        if (i < MOE_I) {
            out[i] = f32_to_e4m3(dg_reg[it] * inv_s);
            out[MOE_I + i] = f32_to_e4m3(du_reg[it] * inv_s);
        }
    }

    if (has_lora) {
        __shared__ float sRed[8][MOE_R];
        int warp = tid >> 5;
        int lane = tid & 31;
        #pragma unroll
        for (int r = 0; r < MOE_R; ++r) {
            float v = warp_sum(dz_gu_acc[r]);
            if (lane == 0) sRed[warp][r] = v;
        }
        __syncthreads();
        if (tid < MOE_R) {
            float sum = 0.0f;
            #pragma unroll
            for (int w_i = 0; w_i < 8; ++w_i) sum += sRed[w_i][tid];
            dz_gu[(int64_t)m * MOE_R + tid] = __float2bfloat16_rn(sum * lora_scale);
        }
    }
}

// -----------------------------------------------------------------------------
// 7. Fused BWD Scatter dx [N_tok, 2304] + dz_gu_tok [N_tok, 16] (ZERO atomicAdd!)
// -----------------------------------------------------------------------------
__global__ void moe_bwd_scatter_dx_kernel(
    const __nv_bfloat16* __restrict__ dx_exp, // [M_buf, 2304]
    const __nv_bfloat16* __restrict__ dz_gu,  // [M_buf, 16]
    const int32_t* __restrict__ inv_idx,      // [N_tok, TOP_K]
    __nv_bfloat16* __restrict__ dx_out,       // [N_tok, 2304]
    __nv_bfloat16* __restrict__ dz_gu_tok,    // [N_tok, 16]
    int N_tok, int top_k, bool has_lora
) {
    int t = blockIdx.x;
    if (t >= N_tok) return;
    int tid = threadIdx.x; // 256 threads -> 9 elems/thread

    float acc[9] = {0.0f};
    float dz_t = 0.0f;

    for (int k = 0; k < top_k; ++k) {
        int pos = inv_idx[(int64_t)t * top_k + k];
        if (pos < 0) continue;
        const __nv_bfloat16* row = dx_exp + (int64_t)pos * MOE_H;
        #pragma unroll
        for (int it = 0; it < 9; ++it) {
            acc[it] += __bfloat162float(row[tid + it * 256]);
        }
        if (has_lora && tid < MOE_R) {
            dz_t += __bfloat162float(dz_gu[(int64_t)pos * MOE_R + tid]);
        }
    }

    __nv_bfloat16* out = dx_out + (int64_t)t * MOE_H;
    #pragma unroll
    for (int it = 0; it < 9; ++it) {
        out[tid + it * 256] = __float2bfloat16_rn(acc[it]);
    }
    if (has_lora && tid < MOE_R) {
        dz_gu_tok[(int64_t)t * MOE_R + tid] = __float2bfloat16_rn(dz_t);
    }
}

} // namespace moe_h100_impl

// -----------------------------------------------------------------------------
// PyTorch C++ Entrypoints
// -----------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_route_and_quant_fp8_cuda(
    torch::Tensor x,             // [N_tok, 2304] bf16
    torch::Tensor top_k_index,   // [N_tok, TOP_K] int64
    torch::Tensor top_k_weights, // [N_tok, TOP_K] fp32
    int64_t num_experts,
    double top_p,
    int64_t k_min
) {
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int N_tok = x.size(0);
    int top_k = top_k_index.size(1);
    int M_flat = N_tok * top_k;
    int E = static_cast<int>(num_experts);
    int M_buf = M_flat + E * 16;

    auto i32_opt = x.options().dtype(torch::kInt32);
    auto f32_opt = x.options().dtype(torch::kFloat32);
    auto u8_opt  = x.options().dtype(torch::kUInt8);

    auto counts        = torch::zeros({E}, i32_opt);
    auto flat_e        = torch::empty({M_flat}, i32_opt);
    auto w_eff         = torch::empty({N_tok, top_k}, f32_opt);
    auto exp_offsets   = torch::empty({E}, i32_opt);
    auto exp_ends      = torch::empty({E}, i32_opt);
    auto write_ptr     = torch::empty({E}, i32_opt);
    auto active_tokens = torch::empty({1}, i32_opt);
    auto sorted_tok    = torch::full({M_buf}, -1, i32_opt);
    auto sorted_e      = torch::full({M_buf}, -1, i32_opt);
    auto sorted_w      = torch::zeros({M_buf}, f32_opt);
    auto inv_idx       = torch::empty({N_tok, top_k}, i32_opt);

    auto x_tok_fp8     = torch::empty({N_tok, moe_h100_impl::MOE_H}, u8_opt);
    auto sx_tok        = torch::empty({N_tok}, f32_opt);
    auto x_sorted_fp8  = torch::empty({M_buf, moe_h100_impl::MOE_H}, u8_opt);
    auto sx_sorted     = torch::empty({M_buf}, f32_opt);

    moe_h100_impl::moe_topp_bincount_kernel<<<(N_tok + 255) / 256, 256, 0, stream>>>(
        top_k_index.data_ptr<int64_t>(), top_k_weights.data_ptr<float>(),
        flat_e.data_ptr<int32_t>(), w_eff.data_ptr<float>(), counts.data_ptr<int32_t>(),
        N_tok, static_cast<float>(top_p), static_cast<int>(k_min), E, top_k
    );
    moe_h100_impl::moe_prefix_sum_kernel<<<1, 256, 0, stream>>>(
        counts.data_ptr<int32_t>(), exp_offsets.data_ptr<int32_t>(), exp_ends.data_ptr<int32_t>(),
        write_ptr.data_ptr<int32_t>(), active_tokens.data_ptr<int32_t>(), E
    );
    moe_h100_impl::moe_scatter_indices_kernel<<<(M_flat + 255) / 256, 256, 0, stream>>>(
        flat_e.data_ptr<int32_t>(), w_eff.data_ptr<float>(), write_ptr.data_ptr<int32_t>(),
        sorted_tok.data_ptr<int32_t>(), sorted_e.data_ptr<int32_t>(), sorted_w.data_ptr<float>(),
        inv_idx.data_ptr<int32_t>(), M_flat, E, top_k
    );
    moe_h100_impl::moe_quant_x_fp8_2304_kernel<<<N_tok, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        x_tok_fp8.data_ptr<uint8_t>(), sx_tok.data_ptr<float>(), N_tok
    );
    moe_h100_impl::moe_gather_fp8_2304_kernel<<<M_buf, 160, 0, stream>>>(
        x_tok_fp8.data_ptr<uint8_t>(), sx_tok.data_ptr<float>(),
        sorted_tok.data_ptr<int32_t>(), active_tokens.data_ptr<int32_t>(),
        x_sorted_fp8.data_ptr<uint8_t>(), sx_sorted.data_ptr<float>()
    );

    return std::make_tuple(
        x_sorted_fp8.view(torch::kFloat8_e4m3fn), sx_sorted,
        exp_ends, active_tokens, sorted_tok, sorted_e, sorted_w, inv_idx
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
moe_gu_lora_swiglu_fp8_fwd_cuda(
    torch::Tensor gu,
    torch::Tensor z_gu,
    torch::Tensor B_gu,
    torch::Tensor A_dn,
    torch::Tensor sorted_tok,
    torch::Tensor sorted_e,
    torch::Tensor active_tokens,
    double lora_scale,
    bool has_lora
) {
    c10::cuda::CUDAGuard guard(gu.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int M_buf = gu.size(0);

    auto act_fp8 = torch::empty({M_buf, moe_h100_impl::MOE_I}, gu.options().dtype(torch::kUInt8));
    auto s_act   = torch::empty({M_buf}, gu.options().dtype(torch::kFloat32));
    auto z_dn    = torch::empty({M_buf, moe_h100_impl::MOE_R}, gu.options());

    if (has_lora) {
        dim3 grid((M_buf + 63) / 64, moe_h100_impl::MOE_H / 64);
        moe_h100_impl::moe_lora_b_add_wmma_kernel<<<grid, 128, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(gu.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(z_gu.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(B_gu.data_ptr<at::BFloat16>()),
            sorted_tok.data_ptr<int32_t>(), sorted_e.data_ptr<int32_t>(), active_tokens.data_ptr<int32_t>(),
            static_cast<float>(lora_scale), true
        );
    }
    moe_h100_impl::moe_gu_lora_swiglu_fp8_fwd_kernel<<<(M_buf + 7) / 8, 256, 0, stream>>>(
        reinterpret_cast<__nv_bfloat16*>(gu.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(z_gu.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B_gu.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(A_dn.data_ptr<at::BFloat16>()),
        sorted_tok.data_ptr<int32_t>(), sorted_e.data_ptr<int32_t>(), active_tokens.data_ptr<int32_t>(),
        act_fp8.data_ptr<uint8_t>(), s_act.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(z_dn.data_ptr<at::BFloat16>()),
        static_cast<float>(lora_scale), false
    );

    return std::make_tuple(act_fp8.view(torch::kFloat8_e4m3fn), s_act, z_dn);
}

torch::Tensor moe_dn_lora_scatter_fwd_cuda(
    torch::Tensor y_exp,
    torch::Tensor z_dn,
    torch::Tensor B_dn,
    torch::Tensor inv_idx,
    torch::Tensor sorted_tok,
    torch::Tensor sorted_e,
    torch::Tensor active_tokens,
    torch::Tensor w_eff,
    double lora_scale,
    bool has_lora
) {
    c10::cuda::CUDAGuard guard(y_exp.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int M_buf = y_exp.size(0);
    int N_tok = inv_idx.size(0);
    int top_k = inv_idx.size(1);

    if (has_lora) {
        dim3 grid((M_buf + 63) / 64, moe_h100_impl::MOE_H / 64);
        moe_h100_impl::moe_lora_b_add_wmma_kernel<<<grid, 128, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16*>(y_exp.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(z_dn.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(B_dn.data_ptr<at::BFloat16>()),
            sorted_tok.data_ptr<int32_t>(), sorted_e.data_ptr<int32_t>(), active_tokens.data_ptr<int32_t>(),
            static_cast<float>(lora_scale), false
        );
    }

    auto y_out = torch::empty({N_tok, moe_h100_impl::MOE_H}, y_exp.options());
    moe_h100_impl::moe_dn_lora_scatter_fwd_kernel<<<N_tok, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(y_exp.data_ptr<at::BFloat16>()),
        inv_idx.data_ptr<int32_t>(), w_eff.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(y_out.data_ptr<at::BFloat16>()),
        N_tok, top_k
    );
    return y_out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
moe_bwd_gather_dw_quant_fp8_cuda(
    torch::Tensor dy,
    torch::Tensor y_exp,
    torch::Tensor B_dn,
    torch::Tensor sorted_tok,
    torch::Tensor sorted_e,
    torch::Tensor sorted_w,
    torch::Tensor inv_idx,
    torch::Tensor active_tokens,
    double lora_scale,
    bool has_lora
) {
    c10::cuda::CUDAGuard guard(dy.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int M_buf = y_exp.size(0);
    int N_tok = inv_idx.size(0);
    int top_k = inv_idx.size(1);

    auto dy_fp8    = torch::empty({M_buf, moe_h100_impl::MOE_H}, dy.options().dtype(torch::kUInt8));
    auto s_dy      = torch::empty({M_buf}, dy.options().dtype(torch::kFloat32));
    auto dw_sorted = torch::empty({M_buf}, dy.options().dtype(torch::kFloat32));
    auto dw_out    = torch::empty({N_tok, top_k}, dy.options().dtype(torch::kFloat32));
    auto dz_dn     = torch::empty({M_buf, moe_h100_impl::MOE_R}, dy.options());

    moe_h100_impl::moe_bwd_gather_dw_quant_fp8_kernel<<<M_buf, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(y_exp.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B_dn.data_ptr<at::BFloat16>()),
        sorted_tok.data_ptr<int32_t>(), sorted_e.data_ptr<int32_t>(), sorted_w.data_ptr<float>(),
        active_tokens.data_ptr<int32_t>(),
        dy_fp8.data_ptr<uint8_t>(), s_dy.data_ptr<float>(), dw_sorted.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dz_dn.data_ptr<at::BFloat16>()),
        static_cast<float>(lora_scale), false
    );
    if (has_lora) {
        moe_h100_impl::moe_lora_dz_dy_bf16_wmma_kernel<<<(M_buf + 15) / 16, 128, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(dy.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(B_dn.data_ptr<at::BFloat16>()),
            sorted_tok.data_ptr<int32_t>(), sorted_e.data_ptr<int32_t>(), sorted_w.data_ptr<float>(),
            active_tokens.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(dz_dn.data_ptr<at::BFloat16>()),
            static_cast<float>(lora_scale)
        );
    }
    int M_flat = N_tok * top_k;
    moe_h100_impl::moe_unsort_dw_kernel<<<(M_flat + 255) / 256, 256, 0, stream>>>(
        dw_sorted.data_ptr<float>(), inv_idx.data_ptr<int32_t>(), dw_out.data_ptr<float>(), M_flat
    );
    return std::make_tuple(dy_fp8.view(torch::kFloat8_e4m3fn), s_dy, dw_out, dz_dn);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
moe_bwd_swiglu_quant_fp8_cuda(
    torch::Tensor gu,
    torch::Tensor d_act,
    torch::Tensor dz_dn,
    torch::Tensor A_dn,
    torch::Tensor B_gu,
    torch::Tensor sorted_e,
    torch::Tensor active_tokens,
    double lora_scale,
    bool has_lora
) {
    c10::cuda::CUDAGuard guard(gu.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int M_buf = gu.size(0);

    auto dgu_fp8 = torch::empty({M_buf, moe_h100_impl::MOE_H}, gu.options().dtype(torch::kUInt8));
    auto s_dgu   = torch::empty({M_buf}, gu.options().dtype(torch::kFloat32));
    auto dz_gu   = torch::empty({M_buf, moe_h100_impl::MOE_R}, gu.options());

    moe_h100_impl::moe_bwd_swiglu_quant_fp8_kernel<<<M_buf, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gu.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(d_act.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(dz_dn.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(A_dn.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(B_gu.data_ptr<at::BFloat16>()),
        sorted_e.data_ptr<int32_t>(), active_tokens.data_ptr<int32_t>(),
        dgu_fp8.data_ptr<uint8_t>(), s_dgu.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dz_gu.data_ptr<at::BFloat16>()),
        static_cast<float>(lora_scale), false
    );
    if (has_lora) {
        moe_h100_impl::moe_lora_dz_fp8_wmma_kernel<<<(M_buf + 15) / 16, 128, 0, stream>>>(
            dgu_fp8.data_ptr<uint8_t>(), s_dgu.data_ptr<float>(),
            reinterpret_cast<const __nv_bfloat16*>(B_gu.data_ptr<at::BFloat16>()),
            sorted_e.data_ptr<int32_t>(), sorted_e.data_ptr<int32_t>(), active_tokens.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(dz_gu.data_ptr<at::BFloat16>()),
            static_cast<float>(lora_scale)
        );
    }
    return std::make_tuple(dgu_fp8.view(torch::kFloat8_e4m3fn), s_dgu, dz_gu);
}

std::tuple<torch::Tensor, torch::Tensor>
moe_bwd_scatter_dx_cuda(
    torch::Tensor dx_exp,
    torch::Tensor dz_gu,
    torch::Tensor inv_idx,
    bool has_lora
) {
    c10::cuda::CUDAGuard guard(dx_exp.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int N_tok = inv_idx.size(0);
    int top_k = inv_idx.size(1);

    auto dx_out    = torch::empty({N_tok, moe_h100_impl::MOE_H}, dx_exp.options());
    auto dz_gu_tok = torch::empty({N_tok, moe_h100_impl::MOE_R}, dx_exp.options());

    moe_h100_impl::moe_bwd_scatter_dx_kernel<<<N_tok, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dx_exp.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(dz_gu.data_ptr<at::BFloat16>()),
        inv_idx.data_ptr<int32_t>(),
        reinterpret_cast<__nv_bfloat16*>(dx_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dz_gu_tok.data_ptr<at::BFloat16>()),
        N_tok, top_k, has_lora
    );
    return std::make_tuple(dx_out, dz_gu_tok);
}
