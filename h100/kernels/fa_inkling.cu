#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

using namespace cute;

namespace fa_inkling_impl {

constexpr int FA_BM = 64;
constexpr int FA_BN = 64;
constexpr int FA_D = 128;
constexpr int FA_R = 16;
constexpr int FA_REL_EXTENT = 1024;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float LN2   = 0.6931471805599453f;

// -----------------------------------------------------------------------------
// Low-level async copy & quad-shuffle helpers
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16_z(void* smem_ptr, const void* gmem_ptr, int src_bytes) {
    uint32_t dst = smem_u32(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(dst), "l"(gmem_ptr), "r"(src_bytes) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ float quad_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
    return v;
}

__device__ __forceinline__ float quad_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 1);
    v += __shfl_xor_sync(0xffffffff, v, 2);
    return v;
}

// Compute dot(a[16] in registers, b[16] in SMEM) using packed __nv_bfloat162 in FP32
__device__ __forceinline__ float dot16_reg_smem(const __nv_bfloat162 a[8], const __nv_bfloat16* b) {
    const uint4 b0 = *reinterpret_cast<const uint4*>(b);
    const uint4 b1 = *reinterpret_cast<const uint4*>(b + 8);
    const __nv_bfloat162* pb0 = reinterpret_cast<const __nv_bfloat162*>(&b0);
    const __nv_bfloat162* pb1 = reinterpret_cast<const __nv_bfloat162*>(&b1);
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 fa = __bfloat1622float2(a[i]);
        float2 fb = __bfloat1622float2(pb0[i]);
        acc += fa.x * fb.x + fa.y * fb.y;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 fa = __bfloat1622float2(a[4 + i]);
        float2 fb = __bfloat1622float2(pb1[i]);
        acc += fa.x * fb.x + fa.y * fb.y;
    }
    return acc;
}

// Compute dot(a[16], b[16]) using packed __nv_bfloat162 in FP32
__device__ __forceinline__ float dot16_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b) {
    const uint4 a0 = *reinterpret_cast<const uint4*>(a);
    const uint4 a1 = *reinterpret_cast<const uint4*>(a + 8);
    const uint4 b0 = *reinterpret_cast<const uint4*>(b);
    const uint4 b1 = *reinterpret_cast<const uint4*>(b + 8);

    const __nv_bfloat162* pa0 = reinterpret_cast<const __nv_bfloat162*>(&a0);
    const __nv_bfloat162* pa1 = reinterpret_cast<const __nv_bfloat162*>(&a1);
    const __nv_bfloat162* pb0 = reinterpret_cast<const __nv_bfloat162*>(&b0);
    const __nv_bfloat162* pb1 = reinterpret_cast<const __nv_bfloat162*>(&b1);

    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 fa = __bfloat1622float2(pa0[i]);
        float2 fb = __bfloat1622float2(pb0[i]);
        acc += fa.x * fb.x + fa.y * fb.y;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 fa = __bfloat1622float2(pa1[i]);
        float2 fb = __bfloat1622float2(pb1[i]);
        acc += fa.x * fb.x + fa.y * fb.y;
    }
    return acc;
}

// Load [64, 128] GMEM tile into K-major SMEM (Layout_K_SW128_Atom)
template <class SmemTensor>
__device__ __forceinline__ void load_tile_k_major(
    SmemTensor& sTensor,
    const __nv_bfloat16* __restrict__ g_base,
    int start_seq,
    int S,
    int stride_seq,
    int tid
) {
    int col = (tid & 15) * 8;
    int r_off = tid >> 4;
    #pragma unroll
    for (int it = 0; it < 8; ++it) {
        int r = it * 8 + r_off;
        int sg = start_seq + r;
        int bytes = (sg < S) ? 16 : 0;
        const __nv_bfloat16* src = g_base + (sg < S ? sg : 0) * stride_seq + col;
        void* dst = static_cast<void*>(&sTensor(r, col));
        cp_async_16_z(dst, src, bytes);
    }
}

// Load [64, 128] GMEM tile into MN-major SMEM (Layout_MN_SW128_Atom shaped (128, 64))
template <class SmemTensor>
__device__ __forceinline__ void load_tile_mn_major(
    SmemTensor& sTensor,
    const __nv_bfloat16* __restrict__ g_base,
    int start_seq,
    int S,
    int stride_seq,
    int tid
) {
    int col = (tid & 15) * 8;
    int r_off = tid >> 4;
    #pragma unroll
    for (int it = 0; it < 8; ++it) {
        int r = it * 8 + r_off;
        int sg = start_seq + r;
        int bytes = (sg < S) ? 16 : 0;
        const __nv_bfloat16* src = g_base + (sg < S ? sg : 0) * stride_seq + col;
        void* dst = static_cast<void*>(&sTensor(col, r));
        cp_async_16_z(dst, src, bytes);
    }
}

// Load [16, 128] slice of rel_proj[16, 1024] into transposed SMEM [128][16]
// d_base = m0 - n0 - 64 is always a multiple of 64 (hence every 8-bf16 chunk is 16B aligned).
__device__ __forceinline__ void load_rel_proj_transposed(
    __nv_bfloat16 sRelP[128][16],
    const __nv_bfloat16* __restrict__ rel_proj,
    int d_base,
    int tid
) {
    // 128 threads x 2 chunks = 256 chunks of 8 bf16 across (rank=0..15, d_chunk=0..15)
    #pragma unroll
    for (int it = 0; it < 2; ++it) {
        int idx = tid + it * 128;
        int r = idx >> 4;          // 0..15
        int d_off = (idx & 15) * 8; // 0, 8, ..., 120
        int d_g = d_base + d_off;
        if (d_g >= 0 && d_g < FA_REL_EXTENT) {
            uint4 v = *reinterpret_cast<const uint4*>(rel_proj + r * FA_REL_EXTENT + d_g);
            const __nv_bfloat16* pv = reinterpret_cast<const __nv_bfloat16*>(&v);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                sRelP[d_off + k][r] = pv[k];
            }
        } else {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                sRelP[d_off + k][r] = __float2bfloat16(0.0f);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Forward Kernel: 1 warpgroup (128 threads), BM=64, BN=64, 2-stage KV pipeline
// SMEM = 90,624 bytes -> 2 CTAs/SM fit in 227 KB on H100!
// -----------------------------------------------------------------------------
struct alignas(128) SmemFwd {
    cutlass::bfloat16_t sQ[FA_BM * FA_D];
    cutlass::bfloat16_t sK[2][FA_BN * FA_D];
    cutlass::bfloat16_t sV[2][FA_D * FA_BN];
    __nv_bfloat16 sRelP[2][128][FA_R];
    __nv_bfloat16 sRelS[FA_BM][FA_R];
    int16_t sDocQ[FA_BM];
    int16_t sDocK[2][FA_BN];
};

__global__ void __launch_bounds__(128, 2) fa_inkling_fwd_kernel(
    const cutlass::bfloat16_t* __restrict__ q,
    const cutlass::bfloat16_t* __restrict__ k,
    const cutlass::bfloat16_t* __restrict__ v,
    const __nv_bfloat16* __restrict__ rel_states, // [B, S, H_q, 16]
    const __nv_bfloat16* __restrict__ rel_proj,   // [16, 1024]
    const __nv_bfloat16* __restrict__ sinks_2d,   // [B, H_q, S]
    const int16_t* __restrict__ doc_id,           // [B, S]
    const int32_t* __restrict__ q_n_start,        // [B, S]
    cutlass::bfloat16_t* __restrict__ out,        // [B, S, H_q, 128]
    float* __restrict__ L_buf,                    // [B*H_q, S]
    int B, int H_q, int H_kv, int S,
    float sm_scale,
    int sliding_win
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ char smem_raw[];
    SmemFwd& sm = *reinterpret_cast<SmemFwd*>(smem_raw);

    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * FA_BM;
    const int bh = blockIdx.y;
    const int kv_g = bh / 3;
    const int b = bh / H_q;
    const int h_q = bh % H_q;
    const int h_kv = kv_g % H_kv;
    const int GQA = H_q / H_kv;
    (void)GQA;

    const __nv_bfloat16* gQ = reinterpret_cast<const __nv_bfloat16*>(q) + ((int64_t)b * H_q + h_q) * S * FA_D;
    const __nv_bfloat16* gK = reinterpret_cast<const __nv_bfloat16*>(k) + ((int64_t)b * H_kv + h_kv) * S * FA_D;
    const __nv_bfloat16* gV = reinterpret_cast<const __nv_bfloat16*>(v) + ((int64_t)b * H_kv + h_kv) * S * FA_D;
    const __nv_bfloat16* gRelS = rel_states + (int64_t)b * S * H_q * FA_R + (int64_t)h_q * FA_R;
    const __nv_bfloat16* gSink = sinks_2d + ((int64_t)b * H_q + h_q) * S;
    const int16_t* gDoc = doc_id + (int64_t)b * S;
    const int32_t* gQNStart = q_n_start + (int64_t)b * S;

    auto sQ_lay = tile_to_shape(GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{}, Shape<_64, _128>{});
    auto sK_lay = tile_to_shape(GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{}, Shape<_64, _128>{});
    auto sV_lay = tile_to_shape(GMMA::Layout_MN_SW128_Atom<cutlass::bfloat16_t>{}, Shape<_128, _64>{});

    auto sQ = make_tensor(make_smem_ptr(sm.sQ), sQ_lay);
    auto sK0 = make_tensor(make_smem_ptr(sm.sK[0]), sK_lay);
    auto sK1 = make_tensor(make_smem_ptr(sm.sK[1]), sK_lay);
    auto sV0 = make_tensor(make_smem_ptr(sm.sV[0]), sV_lay);
    auto sV1 = make_tensor(make_smem_ptr(sm.sV[1]), sV_lay);

    // Load Q, sRelS, sDocQ, and find min q_n_start over tile [m0, m0+64)
    load_tile_k_major(sQ, gQ, m0, S, FA_D, tid);
    cp_async_commit();

    if (tid < FA_BM) {
        int mg = m0 + tid;
        sm.sDocQ[tid] = (mg < S) ? gDoc[mg] : int16_t(-1);
        // Load sRelS[tid][0..15] (2 uint4 per thread, wait 64 threads)
        if (mg < S) {
            const __nv_bfloat16* rs_ptr = gRelS + (int64_t)mg * H_q * FA_R;
            *reinterpret_cast<uint4*>(&sm.sRelS[tid][0]) = *reinterpret_cast<const uint4*>(rs_ptr);
            *reinterpret_cast<uint4*>(&sm.sRelS[tid][8]) = *reinterpret_cast<const uint4*>(rs_ptr + 8);
        } else {
            *reinterpret_cast<uint4*>(&sm.sRelS[tid][0]) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&sm.sRelS[tid][8]) = make_uint4(0, 0, 0, 0);
        }
    }

    // Compute CTA-wide KV loop bounds [n_start, n_end)
    int n_min_val = S;
    int m_end_tile = min(m0 + FA_BM, S);
    if (tid < (m_end_tile - m0)) {
        n_min_val = gQNStart[m0 + tid];
    }
    // Warp 0 + Warp 1 across 64 threads (first 2 warps) -> share via smem or just use q_n_start[m0] ?
    // Since doc_id is blockwise monotonic, min across tile is safe via warp shuffle + 2-slot smem:
    __shared__ int s_nmin[2];
    int v_red = n_min_val;
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        v_red = min(v_red, __shfl_xor_sync(0xffffffff, v_red, off));
    }
    if ((tid & 31) == 0 && (tid >> 5) < 2) s_nmin[tid >> 5] = v_red;
    __syncthreads();
    int n_min_tile = min(s_nmin[0], s_nmin[1]);
    int n_start = (max(0, n_min_tile) / FA_BN) * FA_BN;
    int n_end = min(m0 + FA_BM, S);

    // Wait Q ready
    cp_async_wait<0>();
    __syncthreads();

    // Thread coordinates inside WGMMA C fragments (64x64 and 64x128)
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row0 = warp * 16 + (lane >> 2);
    const int row1 = row0 + 8;
    const int mg0 = m0 + row0;
    const int mg1 = m0 + row1;
    const int c0 = (lane & 3) * 2;
    const int16_t dq0 = sm.sDocQ[row0];
    const int16_t dq1 = sm.sDocQ[row1];

    const float sm_scale_log2 = sm_scale * LOG2E;
    float m_row0 = (mg0 < S) ? __bfloat162float(gSink[mg0]) * LOG2E : -1e30f;
    float m_row1 = (mg1 < S) ? __bfloat162float(gSink[mg1]) * LOG2E : -1e30f;
    float l_row0 = 1.0f;
    float l_row1 = 1.0f;

    // Hoist sRelS[row0/row1] * LOG2E into packed __nv_bfloat162[8] registers ONCE before KV loop!
    __nv_bfloat162 rRelS0[8];
    __nv_bfloat162 rRelS1[8];
    {
        const uint4 u00 = *reinterpret_cast<const uint4*>(&sm.sRelS[row0][0]);
        const uint4 u01 = *reinterpret_cast<const uint4*>(&sm.sRelS[row0][8]);
        const uint4 u10 = *reinterpret_cast<const uint4*>(&sm.sRelS[row1][0]);
        const uint4 u11 = *reinterpret_cast<const uint4*>(&sm.sRelS[row1][8]);
        const __nv_bfloat162* p00 = reinterpret_cast<const __nv_bfloat162*>(&u00);
        const __nv_bfloat162* p01 = reinterpret_cast<const __nv_bfloat162*>(&u01);
        const __nv_bfloat162* p10 = reinterpret_cast<const __nv_bfloat162*>(&u10);
        const __nv_bfloat162* p11 = reinterpret_cast<const __nv_bfloat162*>(&u11);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f0 = __bfloat1622float2(p00[i]); rRelS0[i]     = __float22bfloat162_rn(make_float2(f0.x * LOG2E, f0.y * LOG2E));
            float2 f1 = __bfloat1622float2(p01[i]); rRelS0[4 + i] = __float22bfloat162_rn(make_float2(f1.x * LOG2E, f1.y * LOG2E));
            float2 g0 = __bfloat1622float2(p10[i]); rRelS1[i]     = __float22bfloat162_rn(make_float2(g0.x * LOG2E, g0.y * LOG2E));
            float2 g1 = __bfloat1622float2(p11[i]); rRelS1[4 + i] = __float22bfloat162_rn(make_float2(g1.x * LOG2E, g1.y * LOG2E));
        }
    }
    const bool doc_q_same = (m0 + FA_BM <= S) && (sm.sDocQ[0] == sm.sDocQ[FA_BM - 1]);
    const int16_t doc_q_0 = sm.sDocQ[0];

    using MmaSS = decltype(make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
    using MmaRS = decltype(make_tiled_mma(SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));
    MmaSS mma_ss;
    MmaRS mma_rs;

    auto thr_ss = mma_ss.get_thread_slice(tid);
    auto thr_rs = mma_rs.get_thread_slice(tid);

    auto tCrQ  = thr_ss.make_fragment_A(thr_ss.partition_A(sQ));
    auto tCrK0 = thr_ss.make_fragment_B(thr_ss.partition_B(sK0));
    auto tCrK1 = thr_ss.make_fragment_B(thr_ss.partition_B(sK1));
    auto tCrV0 = thr_rs.make_fragment_B(thr_rs.partition_B(sV0));
    auto tCrV1 = thr_rs.make_fragment_B(thr_rs.partition_B(sV1));

    auto tCrS = partition_fragment_C(mma_ss, Shape<_64, _64>{});
    auto tCrO = partition_fragment_C(mma_rs, Shape<_64, _128>{});
    clear(tCrO);

    cutlass::bfloat16_t P_bf16[32];
    auto sP_lay = tile_to_shape(GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{}, Shape<_64, _64>{});
    auto tCrP_proto = thr_rs.partition_fragment_A(make_tensor(make_smem_ptr((cutlass::bfloat16_t*)nullptr), sP_lay));
    auto tCrP = make_tensor(make_rmem_ptr(P_bf16), tCrP_proto.layout());

    // Prologue: load first KV tile if any
    int stage = 0;
    if (n_start < n_end) {
        load_tile_k_major(sK0, gK, n_start, S, FA_D, tid);
        load_tile_mn_major(sV0, gV, n_start, S, FA_D, tid);
        cp_async_commit();
        if (tid < FA_BN) {
            int ng = n_start + tid;
            sm.sDocK[0][tid] = (ng < S) ? gDoc[ng] : int16_t(-2);
        }
        bool ov0 = (m0 + FA_BM - 1 >= n_start) && (m0 - (n_start + FA_BN - 1) < FA_REL_EXTENT);
        if (ov0) load_rel_proj_transposed(sm.sRelP[0], rel_proj, m0 - n_start - 64, tid);
        cp_async_wait<0>();
        __syncthreads();
    }

    for (int n0 = n_start; n0 < n_end; n0 += FA_BN) {
        int next_n0 = n0 + FA_BN;
        int next_stage = stage ^ 1;

        // Overlap next KV load with current WGMMA S = Q @ K^T
        if (next_n0 < n_end) {
            if (next_stage == 0) {
                load_tile_k_major(sK0, gK, next_n0, S, FA_D, tid);
                load_tile_mn_major(sV0, gV, next_n0, S, FA_D, tid);
            } else {
                load_tile_k_major(sK1, gK, next_n0, S, FA_D, tid);
                load_tile_mn_major(sV1, gV, next_n0, S, FA_D, tid);
            }
            cp_async_commit();
            if (tid < FA_BN) {
                int ng = next_n0 + tid;
                sm.sDocK[next_stage][tid] = (ng < S) ? gDoc[ng] : int16_t(-2);
            }
            bool ov_next = (m0 + FA_BM - 1 >= next_n0) && (m0 - (next_n0 + FA_BN - 1) < FA_REL_EXTENT);
            if (ov_next) load_rel_proj_transposed(sm.sRelP[next_stage], rel_proj, m0 - next_n0 - 64, tid);
        }

        // Compute S = Q @ K^T via SS WGMMA
        clear(tCrS);
        warpgroup_fence_operand(tCrS);
        warpgroup_arrive();
        if (stage == 0) gemm(mma_ss, tCrQ, tCrK0, tCrS);
        else            gemm(mma_ss, tCrQ, tCrK1, tCrS);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrS);

        // Apply mask + rel_bias and online softmax (with Mask-Free Fast-Path & Lazy Rescaling!)
        const int d_base = m0 - n0 - 64;
        const bool ov_cur = (m0 + FA_BM - 1 >= n0) && (m0 - (n0 + FA_BN - 1) < FA_REL_EXTENT);
        const bool full_valid = doc_q_same && (m0 >= n0 + FA_BN) && (m0 + FA_BM - 1 - n0 < sliding_win)
                                && (sm.sDocK[stage][0] == sm.sDocK[stage][FA_BN - 1])
                                && (doc_q_0 == sm.sDocK[stage][0]);
        float max_b0 = -1e30f;
        float max_b1 = -1e30f;

        if (full_valid) {
            #pragma unroll
            for (int k_grp = 0; k_grp < 8; ++k_grp) {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    int ng = n0 + k_grp * 8 + c0 + sub;
                    int idx0 = 4 * k_grp + sub;
                    int idx1 = 4 * k_grp + 2 + sub;

                    float rb0 = (ov_cur && (mg0 - ng) < FA_REL_EXTENT)
                                ? dot16_reg_smem(rRelS0, sm.sRelP[stage][(mg0 - ng) - d_base]) : 0.0f;
                    float s0 = tCrS(idx0) * sm_scale_log2 + rb0;
                    tCrS(idx0) = s0;
                    max_b0 = fmaxf(max_b0, s0);

                    float rb1 = (ov_cur && (mg1 - ng) < FA_REL_EXTENT)
                                ? dot16_reg_smem(rRelS1, sm.sRelP[stage][(mg1 - ng) - d_base]) : 0.0f;
                    float s1 = tCrS(idx1) * sm_scale_log2 + rb1;
                    tCrS(idx1) = s1;
                    max_b1 = fmaxf(max_b1, s1);
                }
            }
        } else {
            #pragma unroll
            for (int k_grp = 0; k_grp < 8; ++k_grp) {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    int n_loc = k_grp * 8 + c0 + sub;
                    int ng = n0 + n_loc;
                    int16_t dk = sm.sDocK[stage][n_loc];
                    int idx0 = 4 * k_grp + sub;
                    int idx1 = 4 * k_grp + 2 + sub;

                    bool v0 = (mg0 < S) && (ng < S) && (mg0 >= ng) && (mg0 - ng < sliding_win) && (dq0 == dk);
                    if (v0) {
                        float rb0 = (ov_cur && (mg0 - ng) < FA_REL_EXTENT)
                                    ? dot16_reg_smem(rRelS0, sm.sRelP[stage][(mg0 - ng) - d_base]) : 0.0f;
                        float s0 = tCrS(idx0) * sm_scale_log2 + rb0;
                        tCrS(idx0) = s0;
                        max_b0 = fmaxf(max_b0, s0);
                    } else {
                        tCrS(idx0) = -1e30f;
                    }

                    bool v1 = (mg1 < S) && (ng < S) && (mg1 >= ng) && (mg1 - ng < sliding_win) && (dq1 == dk);
                    if (v1) {
                        float rb1 = (ov_cur && (mg1 - ng) < FA_REL_EXTENT)
                                    ? dot16_reg_smem(rRelS1, sm.sRelP[stage][(mg1 - ng) - d_base]) : 0.0f;
                        float s1 = tCrS(idx1) * sm_scale_log2 + rb1;
                        tCrS(idx1) = s1;
                        max_b1 = fmaxf(max_b1, s1);
                    } else {
                        tCrS(idx1) = -1e30f;
                    }
                }
            }
        }

        max_b0 = quad_max(max_b0);
        max_b1 = quad_max(max_b1);
        float m_new0 = fmaxf(m_row0, max_b0);
        float m_new1 = fmaxf(m_row1, max_b1);

        // Lazy Rescaling: skip 32 fmuls per row when m_new == m_row!
        if (m_new0 > m_row0) {
            float alpha0 = exp2f(m_row0 - m_new0);
            l_row0 *= alpha0;
            #pragma unroll
            for (int k_grp = 0; k_grp < 16; ++k_grp) {
                tCrO(4 * k_grp + 0) *= alpha0;
                tCrO(4 * k_grp + 1) *= alpha0;
            }
        }
        if (m_new1 > m_row1) {
            float alpha1 = exp2f(m_row1 - m_new1);
            l_row1 *= alpha1;
            #pragma unroll
            for (int k_grp = 0; k_grp < 16; ++k_grp) {
                tCrO(4 * k_grp + 2) *= alpha1;
                tCrO(4 * k_grp + 3) *= alpha1;
            }
        }

        float p_sum0 = 0.0f;
        float p_sum1 = 0.0f;
        #pragma unroll
        for (int k_grp = 0; k_grp < 8; ++k_grp) {
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                int idx0 = 4 * k_grp + sub;
                int idx1 = 4 * k_grp + 2 + sub;
                float s0 = tCrS(idx0);
                float s1 = tCrS(idx1);
                float p0 = (s0 > -1e20f) ? exp2f(s0 - m_new0) : 0.0f;
                float p1 = (s1 > -1e20f) ? exp2f(s1 - m_new1) : 0.0f;
                p_sum0 += p0;
                p_sum1 += p1;
                P_bf16[idx0] = cutlass::bfloat16_t(__float2bfloat16_rn(p0));
                P_bf16[idx1] = cutlass::bfloat16_t(__float2bfloat16_rn(p1));
            }
        }
        l_row0 += quad_sum(p_sum0);
        l_row1 += quad_sum(p_sum1);
        m_row0 = m_new0;
        m_row1 = m_new1;

        // O += P @ V via RS WGMMA
        warpgroup_fence_operand(tCrO);
        warpgroup_arrive();
        if (stage == 0) gemm(mma_rs, tCrP, tCrV0, tCrO);
        else            gemm(mma_rs, tCrP, tCrV1, tCrO);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrO);

        if (next_n0 < n_end) {
            cp_async_wait<0>();
            __syncthreads();
        }
        stage = next_stage;
    }

    // Normalize O & write L_buf
    float inv_l0 = 1.0f / l_row0;
    float inv_l1 = 1.0f / l_row1;
    #pragma unroll
    for (int k_grp = 0; k_grp < 16; ++k_grp) {
        tCrO(4 * k_grp + 0) *= inv_l0;
        tCrO(4 * k_grp + 1) *= inv_l0;
        tCrO(4 * k_grp + 2) *= inv_l1;
        tCrO(4 * k_grp + 3) *= inv_l1;
    }

    if (c0 == 0) {
        if (mg0 < S) L_buf[((int64_t)b * H_q + h_q) * S + mg0] = (m_row0 + log2f(l_row0)) * LN2;
        if (mg1 < S) L_buf[((int64_t)b * H_q + h_q) * S + mg1] = (m_row1 + log2f(l_row1)) * LN2;
    }

    // Stage O into sQ (no longer needed!) for coalesced float4 store -> [B, S, H_q, 128]
    __syncthreads();
    __nv_bfloat16* sOut = reinterpret_cast<__nv_bfloat16*>(sm.sQ);
    #pragma unroll
    for (int k_grp = 0; k_grp < 16; ++k_grp) {
        int col = k_grp * 8 + c0;
        *reinterpret_cast<__nv_bfloat162*>(&sOut[row0 * FA_D + col]) =
            __float22bfloat162_rn(make_float2(tCrO(4 * k_grp + 0), tCrO(4 * k_grp + 1)));
        *reinterpret_cast<__nv_bfloat162*>(&sOut[row1 * FA_D + col]) =
            __float22bfloat162_rn(make_float2(tCrO(4 * k_grp + 2), tCrO(4 * k_grp + 3)));
    }
    __syncthreads();

    __nv_bfloat16* gOut = reinterpret_cast<__nv_bfloat16*>(out) + ((int64_t)b * S * H_q + (int64_t)m0 * H_q + h_q) * FA_D;
    int col_st = (tid & 15) * 8;
    int r_st_off = tid >> 4;
    #pragma unroll
    for (int it = 0; it < 8; ++it) {
        int r = it * 8 + r_st_off;
        if (m0 + r < S) {
            *reinterpret_cast<uint4*>(gOut + (int64_t)r * H_q * FA_D + col_st) =
                *reinterpret_cast<const uint4*>(&sOut[r * FA_D + col_st]);
        }
    }
#endif
}

// -----------------------------------------------------------------------------
// Backward Prep Kernel: D[b, h_q, m] = sum_d(dO * O), dsinks = -exp(sink - L) * D
// Grid: (S, B * H_q), Block: 128 threads -> 4 rows/CTA? No! Let block handle 16 rows: 8 thr/row -> 16 bf16/thr!
// -----------------------------------------------------------------------------
__global__ void fa_inkling_bwd_prep_kernel(
    const __nv_bfloat16* __restrict__ out,      // [B, S, H_q, 128]
    const __nv_bfloat16* __restrict__ grad_out, // [B, S, H_q, 128]
    const __nv_bfloat16* __restrict__ sinks_2d, // [B, H_q, S]
    const float* __restrict__ L_buf,            // [B*H_q, S]
    float* __restrict__ D_buf,                  // [B*H_q, S]
    __nv_bfloat16* __restrict__ dsinks_2d,      // [B, H_q, S]
    int B, int H_q, int S
) {
    int tid = threadIdx.x; // 0..127 -> 16 rows x 8 threads/row
    int m = blockIdx.x * 16 + (tid >> 3);
    int bh = blockIdx.y;
    if (m >= S) return;
    int b = bh / H_q;
    int h_q = bh % H_q;
    int col = (tid & 7) * 16; // 16 bf16 per thread = 2 uint4

    int64_t tok_off = ((int64_t)b * S * H_q + (int64_t)m * H_q + h_q) * FA_D + col;
    uint4 o0 = *reinterpret_cast<const uint4*>(out + tok_off);
    uint4 o1 = *reinterpret_cast<const uint4*>(out + tok_off + 8);
    uint4 g0 = *reinterpret_cast<const uint4*>(grad_out + tok_off);
    uint4 g1 = *reinterpret_cast<const uint4*>(grad_out + tok_off + 8);

    const __nv_bfloat162* po0 = reinterpret_cast<const __nv_bfloat162*>(&o0);
    const __nv_bfloat162* po1 = reinterpret_cast<const __nv_bfloat162*>(&o1);
    const __nv_bfloat162* pg0 = reinterpret_cast<const __nv_bfloat162*>(&g0);
    const __nv_bfloat162* pg1 = reinterpret_cast<const __nv_bfloat162*>(&g1);

    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 fo = __bfloat1622float2(po0[i]);
        float2 fg = __bfloat1622float2(pg0[i]);
        acc += fo.x * fg.x + fo.y * fg.y;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 fo = __bfloat1622float2(po1[i]);
        float2 fg = __bfloat1622float2(pg1[i]);
        acc += fo.x * fg.x + fo.y * fg.y;
    }

    // Reduce across 8 threads (xor 1, 2, 4)
    acc += __shfl_xor_sync(0xffffffff, acc, 1);
    acc += __shfl_xor_sync(0xffffffff, acc, 2);
    acc += __shfl_xor_sync(0xffffffff, acc, 4);

    if ((tid & 7) == 0) {
        int64_t idx = (int64_t)bh * S + m;
        D_buf[idx] = acc;
        float L_val = L_buf[idx];
        float sink_val = __bfloat162float(sinks_2d[idx]);
        float sink_p = expf(sink_val - L_val);
        dsinks_2d[idx] = __float2bfloat16_rn(-sink_p * acc);
    }
}

// -----------------------------------------------------------------------------
// Backward dQ + d_rel_states Kernel (BM=64, BN=64, 128 threads, ZERO atomics!)
// SMEM = 86,272 bytes -> 2 CTAs/SM fit in 227 KB!
// -----------------------------------------------------------------------------
struct alignas(128) SmemBwdDQ {
    cutlass::bfloat16_t sQ[FA_BM * FA_D];
    cutlass::bfloat16_t sdO[FA_BM * FA_D];
    cutlass::bfloat16_t sK_k[FA_BN * FA_D];
    cutlass::bfloat16_t sV_k[FA_BN * FA_D];
    cutlass::bfloat16_t sK_mn[FA_D * FA_BN];
    __nv_bfloat16 sRelP[128][FA_R];
    __nv_bfloat16 sRelS[FA_BM][FA_R];
    float sL[FA_BM];
    float sD[FA_BM];
    int16_t sDocQ[FA_BM];
    int16_t sDocK[FA_BN];
};

__global__ void __launch_bounds__(128, 2) fa_inkling_bwd_dq_kernel(
    const cutlass::bfloat16_t* __restrict__ q,
    const cutlass::bfloat16_t* __restrict__ k,
    const cutlass::bfloat16_t* __restrict__ v,
    const __nv_bfloat16* __restrict__ rel_states,   // [B, S, H_q, 16]
    const __nv_bfloat16* __restrict__ rel_proj,     // [16, 1024]
    const int16_t* __restrict__ doc_id,             // [B, S]
    const int32_t* __restrict__ q_n_start,          // [B, S]
    const float* __restrict__ L_buf,                // [B*H_q, S]
    const float* __restrict__ D_buf,                // [B*H_q, S]
    const cutlass::bfloat16_t* __restrict__ grad_out, // [B, S, H_q, 128]
    cutlass::bfloat16_t* __restrict__ dq,           // [B, H_q, S, 128]
    __nv_bfloat16* __restrict__ d_rel_states,       // [B, S, H_q, 16]
    int B, int H_q, int H_kv, int S,
    float sm_scale,
    int sliding_win
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ char smem_raw[];
    SmemBwdDQ& sm = *reinterpret_cast<SmemBwdDQ*>(smem_raw);

    const int tid = threadIdx.x;
    const int m0 = blockIdx.x * FA_BM;
    const int bh = blockIdx.y;
    const int b = bh / H_q;
    const int h_q = bh % H_q;
    const int h_kv = h_q / (H_q / H_kv);

    const __nv_bfloat16* gQ   = reinterpret_cast<const __nv_bfloat16*>(q) + ((int64_t)b * H_q + h_q) * S * FA_D;
    const __nv_bfloat16* gdO  = reinterpret_cast<const __nv_bfloat16*>(grad_out) + ((int64_t)b * S * H_q + h_q) * FA_D;
    const __nv_bfloat16* gK   = reinterpret_cast<const __nv_bfloat16*>(k) + ((int64_t)b * H_kv + h_kv) * S * FA_D;
    const __nv_bfloat16* gV   = reinterpret_cast<const __nv_bfloat16*>(v) + ((int64_t)b * H_kv + h_kv) * S * FA_D;
    const __nv_bfloat16* gRelS = rel_states + (int64_t)b * S * H_q * FA_R + (int64_t)h_q * FA_R;
    const int16_t* gDoc = doc_id + (int64_t)b * S;
    const int32_t* gQNStart = q_n_start + (int64_t)b * S;
    const float* gL = L_buf + (int64_t)bh * S;
    const float* gD = D_buf + (int64_t)bh * S;

    auto sQ_lay = tile_to_shape(GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{}, Shape<_64, _128>{});
    auto sK_lay = tile_to_shape(GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{}, Shape<_64, _128>{});
    auto sK_mn_lay = tile_to_shape(GMMA::Layout_MN_SW128_Atom<cutlass::bfloat16_t>{}, Shape<_128, _64>{});

    auto sQ    = make_tensor(make_smem_ptr(sm.sQ), sQ_lay);
    auto sdO   = make_tensor(make_smem_ptr(sm.sdO), sQ_lay);
    auto sK_k  = make_tensor(make_smem_ptr(sm.sK_k), sK_lay);
    auto sV_k  = make_tensor(make_smem_ptr(sm.sV_k), sK_lay);
    auto sK_mn = make_tensor(make_smem_ptr(sm.sK_mn), sK_mn_lay);

    load_tile_k_major(sQ, gQ, m0, S, FA_D, tid);
    load_tile_k_major(sdO, gdO, m0, S, H_q * FA_D, tid);
    cp_async_commit();

    if (tid < FA_BM) {
        int mg = m0 + tid;
        sm.sDocQ[tid] = (mg < S) ? gDoc[mg] : int16_t(-1);
        sm.sL[tid]    = (mg < S) ? (gL[mg] * LOG2E) : 0.0f;
        sm.sD[tid]    = (mg < S) ? gD[mg] : 0.0f;
        if (mg < S) {
            const __nv_bfloat16* rs_ptr = gRelS + (int64_t)mg * H_q * FA_R;
            *reinterpret_cast<uint4*>(&sm.sRelS[tid][0]) = *reinterpret_cast<const uint4*>(rs_ptr);
            *reinterpret_cast<uint4*>(&sm.sRelS[tid][8]) = *reinterpret_cast<const uint4*>(rs_ptr + 8);
        } else {
            *reinterpret_cast<uint4*>(&sm.sRelS[tid][0]) = make_uint4(0, 0, 0, 0);
            *reinterpret_cast<uint4*>(&sm.sRelS[tid][8]) = make_uint4(0, 0, 0, 0);
        }
    }

    int n_min_val = S;
    int m_end_tile = min(m0 + FA_BM, S);
    if (tid < (m_end_tile - m0)) {
        n_min_val = gQNStart[m0 + tid];
    }
    __shared__ int s_nmin[2];
    int v_red = n_min_val;
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        v_red = min(v_red, __shfl_xor_sync(0xffffffff, v_red, off));
    }
    if ((tid & 31) == 0 && (tid >> 5) < 2) s_nmin[tid >> 5] = v_red;
    __syncthreads();
    int n_start = (max(0, min(s_nmin[0], s_nmin[1])) / FA_BN) * FA_BN;
    int n_end = min(m0 + FA_BM, S);

    cp_async_wait<0>();
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row0 = warp * 16 + (lane >> 2);
    const int row1 = row0 + 8;
    const int mg0 = m0 + row0;
    const int mg1 = m0 + row1;
    const int c0 = (lane & 3) * 2;
    const int16_t dq0 = sm.sDocQ[row0];
    const int16_t dq1 = sm.sDocQ[row1];
    const float L0_log2 = sm.sL[row0];
    const float L1_log2 = sm.sL[row1];
    const float D0_val = sm.sD[row0];
    const float D1_val = sm.sD[row1];
    const float sm_scale_log2 = sm_scale * LOG2E;

    using MmaSS = decltype(make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
    using MmaRS = decltype(make_tiled_mma(SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));
    MmaSS mma_ss;
    MmaRS mma_rs;

    auto thr_ss = mma_ss.get_thread_slice(tid);
    auto thr_rs = mma_rs.get_thread_slice(tid);

    auto tCrS  = partition_fragment_C(mma_ss, Shape<_64, _64>{});
    auto tCrdQ = partition_fragment_C(mma_rs, Shape<_64, _128>{});
    clear(tCrdQ);

    cutlass::bfloat16_t P_bf16[32];
    cutlass::bfloat16_t dS_bf16[32];
    auto sP_lay = tile_to_shape(GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{}, Shape<_64, _64>{});
    auto tCrdS_proto = thr_rs.partition_fragment_A(make_tensor(make_smem_ptr((cutlass::bfloat16_t*)nullptr), sP_lay));
    auto tCrdS = make_tensor(make_rmem_ptr(dS_bf16), tCrdS_proto.layout());

    float drs0[FA_R] = {0.0f};
    float drs1[FA_R] = {0.0f};

    // Prologue: load first n_start
    if (n_start < n_end) {
        load_tile_k_major(sK_k, gK, n_start, S, FA_D, tid);
        load_tile_k_major(sV_k, gV, n_start, S, FA_D, tid);
        cp_async_commit();
        load_tile_mn_major(sK_mn, gK, n_start, S, FA_D, tid);
        cp_async_commit();
        if (tid < FA_BN) {
            int ng = n_start + tid;
            sm.sDocK[tid] = (ng < S) ? gDoc[ng] : int16_t(-2);
        }
        bool ov0 = (m0 + FA_BM - 1 >= n_start) && (m0 - (n_start + FA_BN - 1) < FA_REL_EXTENT);
        if (ov0) load_rel_proj_transposed(sm.sRelP, rel_proj, m0 - n_start - 64, tid);
        cp_async_wait<1>();
        __syncthreads();
    }

    for (int n0 = n_start; n0 < n_end; n0 += FA_BN) {
        int next_n0 = n0 + FA_BN;

        // 1. S = Q @ K^T (locally scoped SMEM descriptors -> ~40 fewer live regs)
        clear(tCrS);
        warpgroup_fence_operand(tCrS);
        warpgroup_arrive();
        {
            auto tCrQ   = thr_ss.make_fragment_A(thr_ss.partition_A(sQ));
            auto tCrK_k = thr_ss.make_fragment_B(thr_ss.partition_B(sK_k));
            gemm(mma_ss, tCrQ, tCrK_k, tCrS);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrS);

        const int d_base = m0 - n0 - 64;
        const bool ov_cur = (m0 + FA_BM - 1 >= n0) && (m0 - (n0 + FA_BN - 1) < FA_REL_EXTENT);

        // Compute P -> P_bf16 before reusing tCrS for dP!
        #pragma unroll
        for (int k_grp = 0; k_grp < 8; ++k_grp) {
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                int n_loc = k_grp * 8 + c0 + sub;
                int ng = n0 + n_loc;
                int16_t dk = sm.sDocK[n_loc];
                int idx0 = 4 * k_grp + sub;
                int idx1 = 4 * k_grp + 2 + sub;

                if ((mg0 < S) && (ng < S) && (mg0 >= ng) && (mg0 - ng < sliding_win) && (dq0 == dk)) {
                    float rb0 = (ov_cur && ((mg0 - ng) < FA_REL_EXTENT))
                                ? dot16_bf16(sm.sRelS[row0], sm.sRelP[(mg0 - ng) - d_base]) : 0.0f;
                    P_bf16[idx0] = cutlass::bfloat16_t(__float2bfloat16_rn(
                        exp2f(tCrS(idx0) * sm_scale_log2 + rb0 * LOG2E - L0_log2)));
                } else {
                    P_bf16[idx0] = cutlass::bfloat16_t(0.0f);
                }
                if ((mg1 < S) && (ng < S) && (mg1 >= ng) && (mg1 - ng < sliding_win) && (dq1 == dk)) {
                    float rb1 = (ov_cur && ((mg1 - ng) < FA_REL_EXTENT))
                                ? dot16_bf16(sm.sRelS[row1], sm.sRelP[(mg1 - ng) - d_base]) : 0.0f;
                    P_bf16[idx1] = cutlass::bfloat16_t(__float2bfloat16_rn(
                        exp2f(tCrS(idx1) * sm_scale_log2 + rb1 * LOG2E - L1_log2)));
                } else {
                    P_bf16[idx1] = cutlass::bfloat16_t(0.0f);
                }
            }
        }

        // 2. dP = dO @ V^T (reusing tCrS!)
        clear(tCrS);
        warpgroup_fence_operand(tCrS);
        warpgroup_arrive();
        {
            auto tCrdO  = thr_ss.make_fragment_A(thr_ss.partition_A(sdO));
            auto tCrV_k = thr_ss.make_fragment_B(thr_ss.partition_B(sV_k));
            gemm(mma_ss, tCrdO, tCrV_k, tCrS);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrS);

        #pragma unroll
        for (int k_grp = 0; k_grp < 8; ++k_grp) {
            #pragma unroll
            for (int sub = 0; sub < 2; ++sub) {
                int n_loc = k_grp * 8 + c0 + sub;
                int ng = n0 + n_loc;
                int idx0 = 4 * k_grp + sub;
                int idx1 = 4 * k_grp + 2 + sub;

                float p0 = static_cast<float>(P_bf16[idx0]);
                float ds0 = p0 * (tCrS(idx0) - D0_val);
                dS_bf16[idx0] = cutlass::bfloat16_t(__float2bfloat16_rn(ds0));
                if (ov_cur && (p0 > 0.0f) && ((mg0 - ng) < FA_REL_EXTENT)) {
                    const __nv_bfloat16* rp0 = sm.sRelP[(mg0 - ng) - d_base];
                    uint4 v0 = *reinterpret_cast<const uint4*>(rp0);
                    uint4 v1 = *reinterpret_cast<const uint4*>(rp0 + 8);
                    const __nv_bfloat162* pv0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
                    const __nv_bfloat162* pv1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float2 f = __bfloat1622float2(pv0[i]);
                        drs0[2 * i]     += ds0 * f.x;
                        drs0[2 * i + 1] += ds0 * f.y;
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float2 f = __bfloat1622float2(pv1[i]);
                        drs0[8 + 2 * i]     += ds0 * f.x;
                        drs0[8 + 2 * i + 1] += ds0 * f.y;
                    }
                }

                float p1 = static_cast<float>(P_bf16[idx1]);
                float ds1 = p1 * (tCrS(idx1) - D1_val);
                dS_bf16[idx1] = cutlass::bfloat16_t(__float2bfloat16_rn(ds1));
                if (ov_cur && (p1 > 0.0f) && ((mg1 - ng) < FA_REL_EXTENT)) {
                    const __nv_bfloat16* rp1 = sm.sRelP[(mg1 - ng) - d_base];
                    uint4 v0 = *reinterpret_cast<const uint4*>(rp1);
                    uint4 v1 = *reinterpret_cast<const uint4*>(rp1 + 8);
                    const __nv_bfloat162* pv0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
                    const __nv_bfloat162* pv1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float2 f = __bfloat1622float2(pv0[i]);
                        drs1[2 * i]     += ds1 * f.x;
                        drs1[2 * i + 1] += ds1 * f.y;
                    }
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        float2 f = __bfloat1622float2(pv1[i]);
                        drs1[8 + 2 * i]     += ds1 * f.x;
                        drs1[8 + 2 * i + 1] += ds1 * f.y;
                    }
                }
            }
        }

        // Wait for sK_mn of current tile, while launching cp.async of NEXT sK_k, sV_k
        cp_async_wait<0>();
        __syncthreads();

        if (next_n0 < n_end) {
            load_tile_k_major(sK_k, gK, next_n0, S, FA_D, tid);
            load_tile_k_major(sV_k, gV, next_n0, S, FA_D, tid);
            cp_async_commit();
            if (tid < FA_BN) {
                int ng = next_n0 + tid;
                sm.sDocK[tid] = (ng < S) ? gDoc[ng] : int16_t(-2);
            }
            bool ov_next = (m0 + FA_BM - 1 >= next_n0) && (m0 - (next_n0 + FA_BN - 1) < FA_REL_EXTENT);
            if (ov_next) load_rel_proj_transposed(sm.sRelP, rel_proj, m0 - next_n0 - 64, tid);
        }

        // dQ += dS @ K_mn (RS WGMMA)
        warpgroup_fence_operand(tCrdQ);
        warpgroup_arrive();
        {
            auto tCrK_mn = thr_rs.make_fragment_B(thr_rs.partition_B(sK_mn));
            gemm(mma_rs, tCrdS, tCrK_mn, tCrdQ);
        }
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(tCrdQ);

        if (next_n0 < n_end) {
            __syncthreads(); // ensure dQ finished reading sK_mn before overwriting it!
            load_tile_mn_major(sK_mn, gK, next_n0, S, FA_D, tid);
            cp_async_commit();
            cp_async_wait<1>(); // sK_k, sV_k of next_n0 ready; sK_mn keeps loading behind next S/dP!
            __syncthreads();
        }
    }

    // Write d_rel_states via 4-thread quad reduction
    #pragma unroll
    for (int r = 0; r < FA_R; ++r) {
        drs0[r] = quad_sum(drs0[r]);
        drs1[r] = quad_sum(drs1[r]);
    }
    if (c0 == 0) {
        __nv_bfloat16* gdRel = d_rel_states + (int64_t)b * S * H_q * FA_R + (int64_t)h_q * FA_R;
        if (mg0 < S) {
            __nv_bfloat162 out2[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                out2[i] = __float22bfloat162_rn(make_float2(drs0[2 * i], drs0[2 * i + 1]));
            }
            *reinterpret_cast<uint4*>(gdRel + (int64_t)mg0 * H_q * FA_R)     = *reinterpret_cast<const uint4*>(&out2[0]);
            *reinterpret_cast<uint4*>(gdRel + (int64_t)mg0 * H_q * FA_R + 8) = *reinterpret_cast<const uint4*>(&out2[4]);
        }
        if (mg1 < S) {
            __nv_bfloat162 out2[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                out2[i] = __float22bfloat162_rn(make_float2(drs1[2 * i], drs1[2 * i + 1]));
            }
            *reinterpret_cast<uint4*>(gdRel + (int64_t)mg1 * H_q * FA_R)     = *reinterpret_cast<const uint4*>(&out2[0]);
            *reinterpret_cast<uint4*>(gdRel + (int64_t)mg1 * H_q * FA_R + 8) = *reinterpret_cast<const uint4*>(&out2[4]);
        }
    }

    // Scale dQ by sm_scale and stage through sQ for coalesced store -> [B, H_q, S, 128]
    __syncthreads();
    __nv_bfloat16* sOut = reinterpret_cast<__nv_bfloat16*>(sm.sQ);
    #pragma unroll
    for (int k_grp = 0; k_grp < 16; ++k_grp) {
        int col = k_grp * 8 + c0;
        *reinterpret_cast<__nv_bfloat162*>(&sOut[row0 * FA_D + col]) =
            __float22bfloat162_rn(make_float2(tCrdQ(4 * k_grp + 0) * sm_scale, tCrdQ(4 * k_grp + 1) * sm_scale));
        *reinterpret_cast<__nv_bfloat162*>(&sOut[row1 * FA_D + col]) =
            __float22bfloat162_rn(make_float2(tCrdQ(4 * k_grp + 2) * sm_scale, tCrdQ(4 * k_grp + 3) * sm_scale));
    }
    __syncthreads();

    __nv_bfloat16* gdQ = reinterpret_cast<__nv_bfloat16*>(dq) + ((int64_t)b * H_q + h_q) * S * FA_D;
    int col_st = (tid & 15) * 8;
    int r_st_off = tid >> 4;
    #pragma unroll
    for (int it = 0; it < 8; ++it) {
        int r = it * 8 + r_st_off;
        if (m0 + r < S) {
            *reinterpret_cast<uint4*>(gdQ + (int64_t)(m0 + r) * FA_D + col_st) =
                *reinterpret_cast<const uint4*>(&sOut[r * FA_D + col_st]);
        }
    }
#endif
}

// -----------------------------------------------------------------------------
// Backward dK + dV Kernel (GQA=3 fused in CTA -> ZERO atomics!)
// Grid: (S/64, B*H_kv), Block: 128 threads.
// SMEM = 105,344 bytes -> 2 CTAs/SM fit in 227 KB!
// -----------------------------------------------------------------------------
struct alignas(128) SmemBwdDKV {
    cutlass::bfloat16_t sK[FA_BN * FA_D];
    cutlass::bfloat16_t sV[FA_BN * FA_D];
    cutlass::bfloat16_t sQ_k[FA_BM * FA_D];
    cutlass::bfloat16_t sDO_k[FA_BM * FA_D];
    cutlass::bfloat16_t sQ_mn[FA_D * FA_BM];
    cutlass::bfloat16_t sDO_mn[FA_D * FA_BM];
    __nv_bfloat16 sRelP[128][FA_R];
    __nv_bfloat16 sRelS_m[FA_BM][FA_R];
    float sL[FA_BM];
    float sD[FA_BM];
    int16_t sDocM[FA_BM];
    int16_t sDocN[FA_BN];
};

__global__ void __launch_bounds__(128, 2) fa_inkling_bwd_dkv_kernel(
    const cutlass::bfloat16_t* __restrict__ q,
    const cutlass::bfloat16_t* __restrict__ k,
    const cutlass::bfloat16_t* __restrict__ v,
    const __nv_bfloat16* __restrict__ rel_states,     // [B, S, H_q, 16]
    const __nv_bfloat16* __restrict__ rel_proj,       // [16, 1024]
    const int16_t* __restrict__ doc_id,               // [B, S]
    const int32_t* __restrict__ k_m_start,            // [B, S]
    const int32_t* __restrict__ k_m_end,              // [B, S]
    const float* __restrict__ L_buf,                  // [B*H_q, S]
    const float* __restrict__ D_buf,                  // [B*H_q, S]
    const cutlass::bfloat16_t* __restrict__ grad_out, // [B, S, H_q, 128]
    cutlass::bfloat16_t* __restrict__ dk,             // [B, H_kv, S, 128]
    cutlass::bfloat16_t* __restrict__ dv,             // [B, H_kv, S, 128]
    int B, int H_q, int H_kv, int S,
    float sm_scale,
    int sliding_win
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ char smem_raw[];
    SmemBwdDKV& sm = *reinterpret_cast<SmemBwdDKV*>(smem_raw);

    const int tid = threadIdx.x;
    const int n0 = blockIdx.x * FA_BN;
    const int bh_q = blockIdx.y;
    const int b = bh_q / H_q;
    const int h_q = bh_q % H_q;
    const int h_kv = h_q / (H_q / H_kv);

    const __nv_bfloat16* gK = reinterpret_cast<const __nv_bfloat16*>(k) + ((int64_t)b * H_kv + h_kv) * S * FA_D;
    const __nv_bfloat16* gV = reinterpret_cast<const __nv_bfloat16*>(v) + ((int64_t)b * H_kv + h_kv) * S * FA_D;
    const __nv_bfloat16* gQ = reinterpret_cast<const __nv_bfloat16*>(q) + (int64_t)bh_q * S * FA_D;
    const __nv_bfloat16* gdO = reinterpret_cast<const __nv_bfloat16*>(grad_out) + ((int64_t)b * S * H_q + h_q) * FA_D;
    const __nv_bfloat16* gRelS = rel_states + (int64_t)b * S * H_q * FA_R + (int64_t)h_q * FA_R;
    const int16_t* gDoc = doc_id + (int64_t)b * S;
    const int32_t* gKMEnd = k_m_end + (int64_t)b * S;
    const float* gL = L_buf + (int64_t)bh_q * S;
    const float* gD = D_buf + (int64_t)bh_q * S;

    auto sK_lay   = tile_to_shape(GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{}, Shape<_64, _128>{});
    auto sQ_mn_lay = tile_to_shape(GMMA::Layout_MN_SW128_Atom<cutlass::bfloat16_t>{}, Shape<_128, _64>{});

    auto sK     = make_tensor(make_smem_ptr(sm.sK), sK_lay);
    auto sV     = make_tensor(make_smem_ptr(sm.sV), sK_lay);
    auto sQ_k   = make_tensor(make_smem_ptr(sm.sQ_k), sK_lay);
    auto sDO_k  = make_tensor(make_smem_ptr(sm.sDO_k), sK_lay);
    auto sQ_mn  = make_tensor(make_smem_ptr(sm.sQ_mn), sQ_mn_lay);
    auto sDO_mn = make_tensor(make_smem_ptr(sm.sDO_mn), sQ_mn_lay);

    // Load K, V, docN ONCE for all GQA Q-heads
    load_tile_k_major(sK, gK, n0, S, FA_D, tid);
    load_tile_k_major(sV, gV, n0, S, FA_D, tid);
    cp_async_commit();

    if (tid < FA_BN) {
        int ng = n0 + tid;
        sm.sDocN[tid] = (ng < S) ? gDoc[ng] : int16_t(-2);
    }

    // Determine m_end_max across n in [n0, min(n0+BN, S))
    int m_max_val = n0;
    int n_end_tile = min(n0 + FA_BN, S);
    if (tid < (n_end_tile - n0)) {
        m_max_val = gKMEnd[n0 + tid];
    }
    __shared__ int s_mmax[2];
    int v_red = m_max_val;
    #pragma unroll
    for (int off = 16; off >= 1; off >>= 1) {
        v_red = max(v_red, __shfl_xor_sync(0xffffffff, v_red, off));
    }
    if ((tid & 31) == 0 && (tid >> 5) < 2) s_mmax[tid >> 5] = v_red;
    __syncthreads();
    int m_start = (n0 / FA_BM) * FA_BM;
    int m_end = min(max(s_mmax[0], s_mmax[1]), S);

    cp_async_wait<0>();
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    // Note: In S^T = K @ Q^T, row is n (n0 + row0/row1), col is m (m0 + 8*k_grp + c0 + sub)
    const int row0 = warp * 16 + (lane >> 2);
    const int row1 = row0 + 8;
    const int ng0 = n0 + row0;
    const int ng1 = n0 + row1;
    const int c0 = (lane & 3) * 2;
    const int16_t dn0 = sm.sDocN[row0];
    const int16_t dn1 = sm.sDocN[row1];
    const float sm_scale_log2 = sm_scale * LOG2E;

    using MmaSS = decltype(make_tiled_mma(SM90_64x64x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{}));
    using MmaRS = decltype(make_tiled_mma(SM90_64x128x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::MN>{}));
    MmaSS mma_ss;
    MmaRS mma_rs;

    auto thr_ss = mma_ss.get_thread_slice(tid);
    auto thr_rs = mma_rs.get_thread_slice(tid);

    auto tCrST = partition_fragment_C(mma_ss, Shape<_64, _64>{});
    auto tCrdK = partition_fragment_C(mma_rs, Shape<_64, _128>{});
    auto tCrdV = partition_fragment_C(mma_rs, Shape<_64, _128>{});
    clear(tCrdK);
    clear(tCrdV);

    cutlass::bfloat16_t PT_bf16[32];
    auto sP_lay = tile_to_shape(GMMA::Layout_K_SW128_Atom<cutlass::bfloat16_t>{}, Shape<_64, _64>{});
    auto tCr_proto = thr_rs.partition_fragment_A(make_tensor(make_smem_ptr((cutlass::bfloat16_t*)nullptr), sP_lay));
    auto tCrPT  = make_tensor(make_rmem_ptr(PT_bf16), tCr_proto.layout());

    if (m_start < m_end) {
        auto load_m_meta = [&](int m_cur) {
            if (tid < FA_BM) {
                int mg = m_cur + tid;
                sm.sDocM[tid] = (mg < S) ? gDoc[mg] : int16_t(-1);
                sm.sL[tid]    = (mg < S) ? (gL[mg] * LOG2E) : 0.0f;
                sm.sD[tid]    = (mg < S) ? gD[mg] : 0.0f;
                if (mg < S) {
                    const __nv_bfloat16* rs_ptr = gRelS + (int64_t)mg * H_q * FA_R;
                    *reinterpret_cast<uint4*>(&sm.sRelS_m[tid][0]) = *reinterpret_cast<const uint4*>(rs_ptr);
                    *reinterpret_cast<uint4*>(&sm.sRelS_m[tid][8]) = *reinterpret_cast<const uint4*>(rs_ptr + 8);
                } else {
                    *reinterpret_cast<uint4*>(&sm.sRelS_m[tid][0]) = make_uint4(0, 0, 0, 0);
                    *reinterpret_cast<uint4*>(&sm.sRelS_m[tid][8]) = make_uint4(0, 0, 0, 0);
                }
            }
            bool ov = (m_cur + FA_BM - 1 >= n0) && (m_cur - (n0 + FA_BN - 1) < FA_REL_EXTENT);
            if (ov) load_rel_proj_transposed(sm.sRelP, rel_proj, m_cur - n0 - 64, tid);
        };

        // G1: sQ_k, sDO_k | G2: sDO_mn | G3: sQ_mn
        load_tile_k_major(sQ_k,  gQ,  m_start, S, FA_D, tid);
        load_tile_k_major(sDO_k, gdO, m_start, S, H_q * FA_D, tid);
        cp_async_commit();
        load_tile_mn_major(sDO_mn, gdO, m_start, S, H_q * FA_D, tid);
        cp_async_commit();
        load_tile_mn_major(sQ_mn,  gQ,  m_start, S, FA_D, tid);
        cp_async_commit();
        load_m_meta(m_start);
        cp_async_wait<1>();
        __syncthreads();

        for (int m0 = m_start; m0 < m_end; m0 += FA_BM) {
            int next_m0 = m0 + FA_BM;

            // 1. S^T = K @ Q^T
            clear(tCrST);
            warpgroup_fence_operand(tCrST);
            warpgroup_arrive();
            {
                auto tCrK   = thr_ss.make_fragment_A(thr_ss.partition_A(sK));
                auto tCrQ_k = thr_ss.make_fragment_B(thr_ss.partition_B(sQ_k));
                gemm(mma_ss, tCrK, tCrQ_k, tCrST);
            }
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(tCrST);

            const int d_base = m0 - n0 - 64;
            const bool ov_cur = (m0 + FA_BM - 1 >= n0) && (m0 - (n0 + FA_BN - 1) < FA_REL_EXTENT);

            #pragma unroll
            for (int k_grp = 0; k_grp < 8; ++k_grp) {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    int m_loc = k_grp * 8 + c0 + sub;
                    int mg = m0 + m_loc;
                    int16_t dm = sm.sDocM[m_loc];
                    float Lm_log2 = sm.sL[m_loc];
                    const __nv_bfloat16* rel_s_m = sm.sRelS_m[m_loc];
                    int idx0 = 4 * k_grp + sub;
                    int idx1 = 4 * k_grp + 2 + sub;

                    if ((mg < S) && (ng0 < S) && (mg >= ng0) && (mg - ng0 < sliding_win) && (dm == dn0)) {
                        float rb0 = (ov_cur && (mg - ng0) < FA_REL_EXTENT)
                                    ? dot16_bf16(rel_s_m, sm.sRelP[(mg - ng0) - d_base]) : 0.0f;
                        PT_bf16[idx0] = cutlass::bfloat16_t(__float2bfloat16_rn(
                            exp2f(tCrST(idx0) * sm_scale_log2 + rb0 * LOG2E - Lm_log2)));
                    } else {
                        PT_bf16[idx0] = cutlass::bfloat16_t(0.0f);
                    }
                    if ((mg < S) && (ng1 < S) && (mg >= ng1) && (mg - ng1 < sliding_win) && (dm == dn1)) {
                        float rb1 = (ov_cur && (mg - ng1) < FA_REL_EXTENT)
                                    ? dot16_bf16(rel_s_m, sm.sRelP[(mg - ng1) - d_base]) : 0.0f;
                        PT_bf16[idx1] = cutlass::bfloat16_t(__float2bfloat16_rn(
                            exp2f(tCrST(idx1) * sm_scale_log2 + rb1 * LOG2E - Lm_log2)));
                    } else {
                        PT_bf16[idx1] = cutlass::bfloat16_t(0.0f);
                    }
                }
            }

            // 2. Overlap dV += PT @ dO_mn AND dP^T = V @ dO^T!
            clear(tCrST);
            warpgroup_fence_operand(tCrdV);
            warpgroup_fence_operand(tCrST);
            warpgroup_arrive();
            {
                auto tCrDO_mn = thr_rs.make_fragment_B(thr_rs.partition_B(sDO_mn));
                auto tCrV     = thr_ss.make_fragment_A(thr_ss.partition_A(sV));
                auto tCrDO_k  = thr_ss.make_fragment_B(thr_ss.partition_B(sDO_k));
                gemm(mma_rs, tCrPT, tCrDO_mn, tCrdV);
                gemm(mma_ss, tCrV,  tCrDO_k,  tCrST);
            }
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(tCrdV);
            warpgroup_fence_operand(tCrST);

            // 3. Compute dS^T in-place inside PT_bf16 (eliminating dST_bf16[32] -> -16 regs -> ZERO spill!)
            #pragma unroll
            for (int k_grp = 0; k_grp < 8; ++k_grp) {
                #pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    int m_loc = k_grp * 8 + c0 + sub;
                    float Dm_val = sm.sD[m_loc];
                    int idx0 = 4 * k_grp + sub;
                    int idx1 = 4 * k_grp + 2 + sub;
                    PT_bf16[idx0] = cutlass::bfloat16_t(__float2bfloat16_rn(
                        static_cast<float>(PT_bf16[idx0]) * (tCrST(idx0) - Dm_val)));
                    PT_bf16[idx1] = cutlass::bfloat16_t(__float2bfloat16_rn(
                        static_cast<float>(PT_bf16[idx1]) * (tCrST(idx1) - Dm_val)));
                }
            }

            // Wait for sQ_mn of m0, then overlap NEXT sQ_k, sDO_k, sDO_mn behind dK WGMMA!
            cp_async_wait<0>();
            __syncthreads();

            if (next_m0 < m_end) {
                load_tile_k_major(sQ_k,  gQ,  next_m0, S, FA_D, tid);
                load_tile_k_major(sDO_k, gdO, next_m0, S, H_q * FA_D, tid);
                cp_async_commit();
                load_tile_mn_major(sDO_mn, gdO, next_m0, S, H_q * FA_D, tid);
                cp_async_commit();
                load_m_meta(next_m0);
            }

            // 4. dK += dS^T (in PT_bf16) @ Q_mn
            warpgroup_fence_operand(tCrdK);
            warpgroup_arrive();
            {
                auto tCrQ_mn = thr_rs.make_fragment_B(thr_rs.partition_B(sQ_mn));
                gemm(mma_rs, tCrPT, tCrQ_mn, tCrdK);
            }
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            warpgroup_fence_operand(tCrdK);

            if (next_m0 < m_end) {
                __syncthreads();
                load_tile_mn_major(sQ_mn, gQ, next_m0, S, FA_D, tid);
                cp_async_commit();
                cp_async_wait<1>();
                __syncthreads();
            }
        }
        __syncthreads();
    }

    // Store dV and (dK * sm_scale) via sK and sV buffers cleanly!
    __nv_bfloat16* sOutV = reinterpret_cast<__nv_bfloat16*>(sm.sK);
    __nv_bfloat16* sOutK = reinterpret_cast<__nv_bfloat16*>(sm.sV);
    #pragma unroll
    for (int k_grp = 0; k_grp < 16; ++k_grp) {
        int col = k_grp * 8 + c0;
        *reinterpret_cast<__nv_bfloat162*>(&sOutV[row0 * FA_D + col]) =
            __float22bfloat162_rn(make_float2(tCrdV(4 * k_grp + 0), tCrdV(4 * k_grp + 1)));
        *reinterpret_cast<__nv_bfloat162*>(&sOutV[row1 * FA_D + col]) =
            __float22bfloat162_rn(make_float2(tCrdV(4 * k_grp + 2), tCrdV(4 * k_grp + 3)));

        *reinterpret_cast<__nv_bfloat162*>(&sOutK[row0 * FA_D + col]) =
            __float22bfloat162_rn(make_float2(tCrdK(4 * k_grp + 0) * sm_scale, tCrdK(4 * k_grp + 1) * sm_scale));
        *reinterpret_cast<__nv_bfloat162*>(&sOutK[row1 * FA_D + col]) =
            __float22bfloat162_rn(make_float2(tCrdK(4 * k_grp + 2) * sm_scale, tCrdK(4 * k_grp + 3) * sm_scale));
    }
    __syncthreads();

    __nv_bfloat16* gdV = reinterpret_cast<__nv_bfloat16*>(dv) + (int64_t)bh_q * S * FA_D;
    __nv_bfloat16* gdK = reinterpret_cast<__nv_bfloat16*>(dk) + (int64_t)bh_q * S * FA_D;
    int col_st = (tid & 15) * 8;
    int r_st_off = tid >> 4;
    #pragma unroll
    for (int it = 0; it < 8; ++it) {
        int r = it * 8 + r_st_off;
        if (n0 + r < S) {
            *reinterpret_cast<uint4*>(gdV + (int64_t)(n0 + r) * FA_D + col_st) =
                *reinterpret_cast<const uint4*>(&sOutV[r * FA_D + col_st]);
            *reinterpret_cast<uint4*>(gdK + (int64_t)(n0 + r) * FA_D + col_st) =
                *reinterpret_cast<const uint4*>(&sOutK[r * FA_D + col_st]);
        }
    }
#endif
}

// Reduce [B, H_q, S, 128] -> [B, H_kv, S, 128] for both dK and dV in 1 coalesced vectorized pass
__global__ void reduce_gqa_dkv_kernel(
    const __nv_bfloat16* __restrict__ dk_tmp,
    const __nv_bfloat16* __restrict__ dv_tmp,
    __nv_bfloat16* __restrict__ dk,
    __nv_bfloat16* __restrict__ dv,
    int B, int H_q, int H_kv, int S
) {
    const int GQA = H_q / H_kv;
    const int64_t total_vec = (int64_t)B * H_kv * S * (FA_D / 8); // 8 bf16 per uint4
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_vec) return;

    const int64_t vec_per_head = (int64_t)S * (FA_D / 8);
    const int64_t bkv = idx / vec_per_head;
    const int64_t off = idx % vec_per_head;
    const int b = bkv / H_kv;
    const int h_kv = bkv % H_kv;
    const int64_t base_q = ((int64_t)b * H_q + (int64_t)h_kv * GQA) * vec_per_head + off;

    float ak[8] = {0.0f};
    float av[8] = {0.0f};
    #pragma unroll
    for (int g = 0; g < GQA; ++g) {
        uint4 vk = reinterpret_cast<const uint4*>(dk_tmp)[base_q + g * vec_per_head];
        uint4 vv = reinterpret_cast<const uint4*>(dv_tmp)[base_q + g * vec_per_head];
        const __nv_bfloat162* pk = reinterpret_cast<const __nv_bfloat162*>(&vk);
        const __nv_bfloat162* pv = reinterpret_cast<const __nv_bfloat162*>(&vv);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 fk = __bfloat1622float2(pk[i]);
            float2 fv = __bfloat1622float2(pv[i]);
            ak[2 * i] += fk.x; ak[2 * i + 1] += fk.y;
            av[2 * i] += fv.x; av[2 * i + 1] += fv.y;
        }
    }
    uint4 ok, ov;
    __nv_bfloat162* pok = reinterpret_cast<__nv_bfloat162*>(&ok);
    __nv_bfloat162* pov = reinterpret_cast<__nv_bfloat162*>(&ov);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        pok[i] = __float22bfloat162_rn(make_float2(ak[2 * i], ak[2 * i + 1]));
        pov[i] = __float22bfloat162_rn(make_float2(av[2 * i], av[2 * i + 1]));
    }
    reinterpret_cast<uint4*>(dk)[idx] = ok;
    reinterpret_cast<uint4*>(dv)[idx] = ov;
}

static cudaStream_t get_fa_aux_stream() {
    static cudaStream_t s2 = nullptr;
    if (!s2) cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    return s2;
}

static void get_fa_sync_events(cudaEvent_t& ev1, cudaEvent_t& ev2) {
    static cudaEvent_t e1 = nullptr, e2 = nullptr;
    if (!e1) {
        cudaEventCreateWithFlags(&e1, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&e2, cudaEventDisableTiming);
    }
    ev1 = e1; ev2 = e2;
}

static void configure_smem_once() {
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(fa_inkling_fwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SmemFwd));
        cudaFuncSetAttribute(fa_inkling_bwd_dq_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SmemBwdDQ));
        cudaFuncSetAttribute(fa_inkling_bwd_dkv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SmemBwdDKV));
        configured = true;
    }
}

} // namespace fa_inkling_impl

// -----------------------------------------------------------------------------
// PyTorch C++ Entrypoints
// -----------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> fa_inkling_fwd_cuda(
    torch::Tensor q,          // [B, H_q, S, 128]
    torch::Tensor k,          // [B, H_kv, S, 128]
    torch::Tensor v,          // [B, H_kv, S, 128]
    torch::Tensor rel_states, // [B, S, H_q, 16]
    torch::Tensor rel_proj,   // [16, 1024]
    torch::Tensor sinks_2d,   // [B, H_q, S]
    torch::Tensor doc_id,     // [B, S] int16
    torch::Tensor q_n_start,  // [B, S] int32
    double sm_scale,
    int64_t sliding_win
) {
    c10::cuda::CUDAGuard guard(q.device());
    fa_inkling_impl::configure_smem_once();
    auto stream = at::cuda::getCurrentCUDAStream();

    int B    = q.size(0);
    int H_q  = q.size(1);
    int S    = q.size(2);
    int H_kv = k.size(1);

    auto out   = torch::empty({B, S, H_q, fa_inkling_impl::FA_D}, q.options());
    auto L_buf = torch::empty({B * H_q, S}, q.options().dtype(torch::kFloat32));

    dim3 grid((S + fa_inkling_impl::FA_BM - 1) / fa_inkling_impl::FA_BM, B * H_q);
    dim3 block(128);

    fa_inkling_impl::fa_inkling_fwd_kernel<<<grid, block, sizeof(fa_inkling_impl::SmemFwd), stream>>>(
        reinterpret_cast<const cutlass::bfloat16_t*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const cutlass::bfloat16_t*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const cutlass::bfloat16_t*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(rel_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(rel_proj.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sinks_2d.data_ptr<at::BFloat16>()),
        doc_id.data_ptr<int16_t>(),
        q_n_start.data_ptr<int32_t>(),
        reinterpret_cast<cutlass::bfloat16_t*>(out.data_ptr<at::BFloat16>()),
        L_buf.data_ptr<float>(),
        B, H_q, H_kv, S,
        static_cast<float>(sm_scale),
        static_cast<int>(sliding_win)
    );

    return std::make_tuple(out, L_buf);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> fa_inkling_bwd_cuda(
    torch::Tensor q,          // [B, H_q, S, 128]
    torch::Tensor k,          // [B, H_kv, S, 128]
    torch::Tensor v,          // [B, H_kv, S, 128]
    torch::Tensor rel_states, // [B, S, H_q, 16]
    torch::Tensor rel_proj,   // [16, 1024]
    torch::Tensor sinks_2d,   // [B, H_q, S]
    torch::Tensor doc_id,     // [B, S]
    torch::Tensor q_n_start,  // [B, S]
    torch::Tensor k_m_start,  // [B, S]
    torch::Tensor k_m_end,    // [B, S]
    torch::Tensor out,        // [B, S, H_q, 128]
    torch::Tensor L_buf,      // [B*H_q, S]
    torch::Tensor grad_out,   // [B, S, H_q, 128]
    double sm_scale,
    int64_t sliding_win
) {
    c10::cuda::CUDAGuard guard(q.device());
    fa_inkling_impl::configure_smem_once();
    auto stream = at::cuda::getCurrentCUDAStream();

    int B    = q.size(0);
    int H_q  = q.size(1);
    int S    = q.size(2);
    int H_kv = k.size(1);

    auto dq          = torch::empty_like(q);
    auto dk          = torch::empty_like(k);
    auto dv          = torch::empty_like(v);
    auto dk_tmp      = torch::empty_like(q);
    auto dv_tmp      = torch::empty_like(q);
    auto d_rel       = torch::empty_like(rel_states);
    auto dsinks      = torch::empty_like(sinks_2d);
    auto D_buf       = torch::empty({B * H_q, S}, L_buf.options());

    cudaStream_t stream2 = fa_inkling_impl::get_fa_aux_stream();
    cudaEvent_t ev_prep, ev_dkv;
    fa_inkling_impl::get_fa_sync_events(ev_prep, ev_dkv);

    // 1. Prep: D_buf + dsinks on main stream
    {
        dim3 grid((S + 15) / 16, B * H_q);
        dim3 block(128);
        fa_inkling_impl::fa_inkling_bwd_prep_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(sinks_2d.data_ptr<at::BFloat16>()),
            L_buf.data_ptr<float>(),
            D_buf.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(dsinks.data_ptr<at::BFloat16>()),
            B, H_q, S
        );
    }
    cudaEventRecord(ev_prep, stream);
    cudaStreamWaitEvent(stream2, ev_prep, 0);

    // 2. dQ + d_rel_states on main stream (6144 CTAs, uint4 vectorized drs)
    {
        dim3 grid((S + fa_inkling_impl::FA_BM - 1) / fa_inkling_impl::FA_BM, B * H_q);
        dim3 block(128);
        fa_inkling_impl::fa_inkling_bwd_dq_kernel<<<grid, block, sizeof(fa_inkling_impl::SmemBwdDQ), stream>>>(
            reinterpret_cast<const cutlass::bfloat16_t*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const cutlass::bfloat16_t*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const cutlass::bfloat16_t*>(v.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(rel_states.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(rel_proj.data_ptr<at::BFloat16>()),
            doc_id.data_ptr<int16_t>(),
            q_n_start.data_ptr<int32_t>(),
            L_buf.data_ptr<float>(),
            D_buf.data_ptr<float>(),
            reinterpret_cast<const cutlass::bfloat16_t*>(grad_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<cutlass::bfloat16_t*>(dq.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(d_rel.data_ptr<at::BFloat16>()),
            B, H_q, H_kv, S,
            static_cast<float>(sm_scale),
            static_cast<int>(sliding_win)
        );
    }

    // 3. dK + dV parallelized over all B*H_q (6144 CTAs!) + reduce_gqa on stream2
    {
        dim3 grid((S + fa_inkling_impl::FA_BN - 1) / fa_inkling_impl::FA_BN, B * H_q);
        dim3 block(128);
        fa_inkling_impl::fa_inkling_bwd_dkv_kernel<<<grid, block, sizeof(fa_inkling_impl::SmemBwdDKV), stream2>>>(
            reinterpret_cast<const cutlass::bfloat16_t*>(q.data_ptr<at::BFloat16>()),
            reinterpret_cast<const cutlass::bfloat16_t*>(k.data_ptr<at::BFloat16>()),
            reinterpret_cast<const cutlass::bfloat16_t*>(v.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(rel_states.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(rel_proj.data_ptr<at::BFloat16>()),
            doc_id.data_ptr<int16_t>(),
            k_m_start.data_ptr<int32_t>(),
            k_m_end.data_ptr<int32_t>(),
            L_buf.data_ptr<float>(),
            D_buf.data_ptr<float>(),
            reinterpret_cast<const cutlass::bfloat16_t*>(grad_out.data_ptr<at::BFloat16>()),
            reinterpret_cast<cutlass::bfloat16_t*>(dk_tmp.data_ptr<at::BFloat16>()),
            reinterpret_cast<cutlass::bfloat16_t*>(dv_tmp.data_ptr<at::BFloat16>()),
            B, H_q, H_kv, S,
            static_cast<float>(sm_scale),
            static_cast<int>(sliding_win)
        );
        int64_t total_vec = (int64_t)B * H_kv * S * (fa_inkling_impl::FA_D / 8);
        int threads = 256;
        int blocks = static_cast<int>((total_vec + threads - 1) / threads);
        fa_inkling_impl::reduce_gqa_dkv_kernel<<<blocks, threads, 0, stream2>>>(
            reinterpret_cast<const __nv_bfloat16*>(dk_tmp.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(dv_tmp.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(dk.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(dv.data_ptr<at::BFloat16>()),
            B, H_q, H_kv, S
        );
    }
    cudaEventRecord(ev_dkv, stream2);
    cudaStreamWaitEvent(stream, ev_dkv, 0);

    return std::make_tuple(dq, dk, dv, d_rel, dsinks);
}

std::vector<double> fa_inkling_bwd_profile_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor rel_states, torch::Tensor rel_proj, torch::Tensor sinks_2d,
    torch::Tensor doc_id, torch::Tensor q_n_start, torch::Tensor k_m_start, torch::Tensor k_m_end,
    torch::Tensor out, torch::Tensor L_buf, torch::Tensor grad_out,
    double sm_scale, int64_t sliding_win
) {
    c10::cuda::CUDAGuard guard(q.device());
    fa_inkling_impl::configure_smem_once();
    auto stream = at::cuda::getCurrentCUDAStream();
    int B = q.size(0), H_q = q.size(1), S = q.size(2), H_kv = k.size(1);

    auto dq = torch::empty_like(q), dk = torch::empty_like(k), dv = torch::empty_like(v);
    auto dk_tmp = torch::empty_like(q), dv_tmp = torch::empty_like(q);
    auto d_rel = torch::empty_like(rel_states), dsinks = torch::empty_like(sinks_2d);
    auto D_buf = torch::empty({B * H_q, S}, L_buf.options());

    cudaEvent_t e0, e1, e2, e3, e4;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3); cudaEventCreate(&e4);

    cudaEventRecord(e0, stream);
    fa_inkling_impl::fa_inkling_bwd_prep_kernel<<<dim3((S + 15) / 16, B * H_q), dim3(128), 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(sinks_2d.data_ptr<at::BFloat16>()),
        L_buf.data_ptr<float>(), D_buf.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(dsinks.data_ptr<at::BFloat16>()), B, H_q, S);
    cudaEventRecord(e1, stream);

    fa_inkling_impl::fa_inkling_bwd_dq_kernel<<<dim3((S + fa_inkling_impl::FA_BM - 1) / fa_inkling_impl::FA_BM, B * H_q), dim3(128), sizeof(fa_inkling_impl::SmemBwdDQ), stream>>>(
        reinterpret_cast<const cutlass::bfloat16_t*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const cutlass::bfloat16_t*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const cutlass::bfloat16_t*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(rel_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(rel_proj.data_ptr<at::BFloat16>()),
        doc_id.data_ptr<int16_t>(), q_n_start.data_ptr<int32_t>(), L_buf.data_ptr<float>(), D_buf.data_ptr<float>(),
        reinterpret_cast<const cutlass::bfloat16_t*>(grad_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::bfloat16_t*>(dq.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(d_rel.data_ptr<at::BFloat16>()),
        B, H_q, H_kv, S, static_cast<float>(sm_scale), static_cast<int>(sliding_win));
    cudaEventRecord(e2, stream);

    fa_inkling_impl::fa_inkling_bwd_dkv_kernel<<<dim3((S + fa_inkling_impl::FA_BN - 1) / fa_inkling_impl::FA_BN, B * H_q), dim3(128), sizeof(fa_inkling_impl::SmemBwdDKV), stream>>>(
        reinterpret_cast<const cutlass::bfloat16_t*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const cutlass::bfloat16_t*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const cutlass::bfloat16_t*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(rel_states.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(rel_proj.data_ptr<at::BFloat16>()),
        doc_id.data_ptr<int16_t>(), k_m_start.data_ptr<int32_t>(), k_m_end.data_ptr<int32_t>(), L_buf.data_ptr<float>(), D_buf.data_ptr<float>(),
        reinterpret_cast<const cutlass::bfloat16_t*>(grad_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::bfloat16_t*>(dk_tmp.data_ptr<at::BFloat16>()),
        reinterpret_cast<cutlass::bfloat16_t*>(dv_tmp.data_ptr<at::BFloat16>()),
        B, H_q, H_kv, S, static_cast<float>(sm_scale), static_cast<int>(sliding_win));
    cudaEventRecord(e3, stream);

    int64_t total_vec = (int64_t)B * H_kv * S * (fa_inkling_impl::FA_D / 8);
    fa_inkling_impl::reduce_gqa_dkv_kernel<<<static_cast<int>((total_vec + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(dk_tmp.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(dv_tmp.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dk.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(dv.data_ptr<at::BFloat16>()), B, H_q, H_kv, S);
    cudaEventRecord(e4, stream);
    cudaEventSynchronize(e4);

    float t_prep = 0, t_dq = 0, t_dkv = 0, t_red = 0;
    cudaEventElapsedTime(&t_prep, e0, e1);
    cudaEventElapsedTime(&t_dq, e1, e2);
    cudaEventElapsedTime(&t_dkv, e2, e3);
    cudaEventElapsedTime(&t_red, e3, e4);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3); cudaEventDestroy(e4);
    return {t_prep, t_dq, t_dkv, t_red};
}
