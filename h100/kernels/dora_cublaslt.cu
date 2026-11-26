#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <mma.h>

#include <unordered_map>
#include <mutex>
#include <tuple>
#include <stdexcept>
#include <cstring>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st = (call);                                            \
        TORCH_CHECK(st == CUBLAS_STATUS_SUCCESS, "cuBLASLt error: ", (int)st); \
    } while (0)

namespace {

struct AlgoCacheKey {
    int m, n, k, fast_accum;
    bool operator==(const AlgoCacheKey& o) const {
        return m == o.m && n == o.n && k == o.k && fast_accum == o.fast_accum;
    }
};

struct AlgoCacheHash {
    size_t operator()(const AlgoCacheKey& x) const {
        size_t h = std::hash<int>()(x.m);
        h ^= std::hash<int>()(x.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(x.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(x.fast_accum) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct LtContext {
    cublasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    void* aux_workspace = nullptr;
    size_t workspace_size = 32ULL * 1024ULL * 1024ULL;
    size_t aux_workspace_size = 4ULL * 1024ULL * 1024ULL;
    cudaStream_t aux_stream = nullptr;
    cudaEvent_t ev_fork = nullptr;
    cudaEvent_t ev_join = nullptr;
    std::mutex mu;
    std::unordered_map<AlgoCacheKey, cublasLtMatmulAlgo_t, AlgoCacheHash> algo_cache;

    void init_once() {
        std::lock_guard<std::mutex> g(mu);
        if (handle == nullptr) {
            CUBLAS_CHECK(cublasLtCreate(&handle));
            CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
            CUDA_CHECK(cudaMalloc(&aux_workspace, aux_workspace_size));
            CUDA_CHECK(cudaStreamCreateWithFlags(&aux_stream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_fork, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&ev_join, cudaEventDisableTiming));
        }
    }
};

static LtContext g_lt;

cublasLtMatmulAlgo_t select_best_fp8_algo(
    int m, int n, int k,
    const void* A_fp8, const void* B_fp8, void* D_bf16,
    const float* scale_a, const float* scale_b,
    bool fast_accum, cudaStream_t stream
) {
    g_lt.init_once();
    AlgoCacheKey key{m, n, k, fast_accum ? 1 : 0};
    {
        std::lock_guard<std::mutex> g(g_lt.mu);
        auto it = g_lt.algo_cache.find(key);
        if (it != g_lt.algo_cache.end()) {
            return it->second;
        }
    }

    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr, layD = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opT = CUBLAS_OP_T;
    cublasOperation_t opN = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    int8_t fa = fast_accum ? 1 : 0;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fa, sizeof(fa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(scale_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(scale_b)));

    // Row-major Y[M, N] = B_row[M, K] @ A_row[N, K]^T => Col-major D[N, M] = opT(A_col[K, N]) @ opN(B_col[K, M])
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layA, CUDA_R_8F_E4M3, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layB, CUDA_R_8F_E4M3, k, m, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layC, CUDA_R_16BF, n, m, n));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layD, CUDA_R_16BF, n, m, n));

    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g_lt.workspace_size, sizeof(g_lt.workspace_size)));

    constexpr int MAX_ALGOS = 12;
    cublasLtMatmulHeuristicResult_t results[MAX_ALGOS];
    int n_found = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        g_lt.handle, desc, layA, layB, layC, layD, pref, MAX_ALGOS, results, &n_found
    ));
    TORCH_CHECK(n_found > 0, "No cuBLASLt FP8 algo found for M=", m, " N=", n, " K=", k);

    cublasLtMatmulAlgo_t best_algo = results[0].algo;
    if (n_found > 1) {
        float alpha = 1.0f, beta = 0.0f;
        cudaEvent_t ev0, ev1;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        float best_ms = 1e30f;
        for (int i = 0; i < n_found; ++i) {
            if (results[i].state != CUBLAS_STATUS_SUCCESS) continue;
            cublasStatus_t st = cublasLtMatmul(
                g_lt.handle, desc, &alpha, A_fp8, layA, B_fp8, layB, &beta, D_bf16, layC, D_bf16, layD,
                &results[i].algo, g_lt.workspace, g_lt.workspace_size, stream
            );
            if (st != CUBLAS_STATUS_SUCCESS) continue;
            CUDA_CHECK(cudaEventRecord(ev0, stream));
            for (int r = 0; r < 5; ++r) {
                cublasLtMatmul(
                    g_lt.handle, desc, &alpha, A_fp8, layA, B_fp8, layB, &beta, D_bf16, layC, D_bf16, layD,
                    &results[i].algo, g_lt.workspace, g_lt.workspace_size, stream
                );
            }
            CUDA_CHECK(cudaEventRecord(ev1, stream));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            if (ms < best_ms) {
                best_ms = ms;
                best_algo = results[i].algo;
            }
        }
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layD);
    cublasLtMatrixLayoutDestroy(layC);
    cublasLtMatrixLayoutDestroy(layB);
    cublasLtMatrixLayoutDestroy(layA);
    cublasLtMatmulDescDestroy(desc);

    {
        std::lock_guard<std::mutex> g(g_lt.mu);
        g_lt.algo_cache[key] = best_algo;
    }
    return best_algo;
}

void run_cublaslt_fp8(
    int m, int n, int k,
    const void* A_fp8, const void* B_fp8, void* D_bf16,
    const float* scale_a, const float* scale_b,
    const cublasLtMatmulAlgo_t& algo, cudaStream_t stream
) {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t layA = nullptr, layB = nullptr, layC = nullptr, layD = nullptr;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    int8_t fa = 1;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fa, sizeof(fa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scale_a, sizeof(scale_a)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scale_b, sizeof(scale_b)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layA, CUDA_R_8F_E4M3, k, n, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layB, CUDA_R_8F_E4M3, k, m, k));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layC, CUDA_R_16BF, n, m, n));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layD, CUDA_R_16BF, n, m, n));

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(
        g_lt.handle, desc, &alpha, A_fp8, layA, B_fp8, layB, &beta, D_bf16, layC, D_bf16, layD,
        &algo, g_lt.workspace, g_lt.workspace_size, stream
    ));

    cublasLtMatrixLayoutDestroy(layD);
    cublasLtMatrixLayoutDestroy(layC);
    cublasLtMatrixLayoutDestroy(layB);
    cublasLtMatrixLayoutDestroy(layA);
    cublasLtMatmulDescDestroy(desc);
}

// 128-bit vectorized BF16 -> FP8 E4M3 quantization (8 elements per thread)
__global__ void quant_bf16_to_fp8_vec8_kernel(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ InvScale,
    __nv_fp8_e4m3* __restrict__ Out,
    int64_t n_vec8
) {
    float inv_s = __ldg(InvScale);
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec8) return;
    uint4 v = reinterpret_cast<const uint4*>(X)[idx];
    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
    uint16_t packed[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 f = __bfloat1622float2(p[i]);
        f.x *= inv_s;
        f.y *= inv_s;
        __nv_fp8x2_e4m3 fp8_2(f);
        packed[i] = *reinterpret_cast<const uint16_t*>(&fp8_2);
    }
    uint2 out_v = make_uint2(
        (uint32_t)packed[0] | ((uint32_t)packed[1] << 16),
        (uint32_t)packed[2] | ((uint32_t)packed[3] << 16)
    );
    reinterpret_cast<uint2*>(Out)[idx] = out_v;
}

__global__ void quant_bf16_to_fp8_scalar_kernel(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ InvScale,
    __nv_fp8_e4m3* __restrict__ Out,
    int64_t n_elems
) {
    float inv_s = __ldg(InvScale);
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems) return;
    float f = __bfloat162float(X[idx]) * inv_s;
    Out[idx] = __nv_fp8_e4m3(f);
}

// In-place Fused 3x DoRA Forward Epilogue using WMMA Tensor Cores (64x128 tile, 8 warps, 128-bit uint4 loads/stores)
// SMEM = 64*24*2 (3KB) + 128*24*2 (6KB) + 64*136*4 (34.8KB) + 128*4 (0.5KB) = 44.5 KB <= 48 KB
__global__ void dora_3x_lora_scale_fwd_wmma_inplace_kernel(
    __nv_bfloat16* __restrict__ Out,
    const __nv_bfloat16* __restrict__ H_a,
    const __nv_bfloat16* __restrict__ B_cat,
    const __nv_bfloat16* __restrict__ M_vec,
    const __nv_bfloat16* __restrict__ InvN_vec,
    int M_dim, int N_sub, int R_dim, float lora_scale, int n_blks_sub
) {
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int A_STRIDE = 24;   // 16 + 8 bf16 pad
    constexpr int B_STRIDE = 24;   // 16 + 8 bf16 pad
    constexpr int C_STRIDE = 136;  // 128 + 8 float pad

    int pid_m = blockIdx.x;
    int pid_nb = blockIdx.y;
    int k_sl = pid_nb / n_blks_sub;
    int n_loc_blk = pid_nb % n_blks_sub;

    int m0 = pid_m * BM;
    int n0_loc = n_loc_blk * BN;
    int n0_tot = k_sl * N_sub + n0_loc;
    int N_tot = 3 * N_sub;

    int tid = threadIdx.x;     // 0..255 (8 warps)
    int warp_id = tid >> 5;
    int warp_m = warp_id >> 1; // 0..3 -> 16 rows
    int warp_n = warp_id & 1;  // 0..1 -> 64 cols (4x 16x16 wmma)

    __shared__ __nv_bfloat16 smem_A[BM * A_STRIDE];
    __shared__ __nv_bfloat16 smem_B[BN * B_STRIDE];
    __shared__ float smem_C[BM * C_STRIDE];
    __shared__ float smem_S[BN];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag0, b_frag1, b_frag2, b_frag3;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1, acc2, acc3;
    wmma::fill_fragment(acc0, 0.0f);
    wmma::fill_fragment(acc1, 0.0f);
    wmma::fill_fragment(acc2, 0.0f);
    wmma::fill_fragment(acc3, 0.0f);

    const __nv_bfloat16* H_ptr = H_a + (int64_t)m0 * (3 * R_dim) + k_sl * R_dim;
    const __nv_bfloat16* B_ptr = B_cat + (int64_t)k_sl * N_sub * R_dim + (int64_t)n0_loc * R_dim;

    // Preload scale s[128] once per CTA (coalesced)
    if (tid < BN) {
        int gn = n0_tot + tid;
        smem_S[tid] = (n0_loc + tid < N_sub)
            ? (__bfloat162float(M_vec[gn]) * __bfloat162float(InvN_vec[gn]))
            : 1.0f;
    }

    const bool vec_ab = ((R_dim & 7) == 0);
    for (int kr = 0; kr < R_dim; kr += 16) {
        if (vec_ab) {
            // smem_A: 64 rows x 2 uint4 = 128 uint4 ops -> threads 0..127 load 1 uint4 each
            if (tid < 128) {
                int rm = tid >> 1;
                int rk = (tid & 1) << 3;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (m0 + rm < M_dim && kr + rk < R_dim) {
                    v = *reinterpret_cast<const uint4*>(H_ptr + (int64_t)rm * (3 * R_dim) + kr + rk);
                }
                *reinterpret_cast<uint4*>(smem_A + rm * A_STRIDE + rk) = v;
            }
            // smem_B: 128 rows x 2 uint4 = 256 uint4 ops -> all 256 threads load 1 uint4 each
            {
                int rn = tid >> 1;
                int rk = (tid & 1) << 3;
                uint4 v = make_uint4(0, 0, 0, 0);
                if (n0_loc + rn < N_sub && kr + rk < R_dim) {
                    v = *reinterpret_cast<const uint4*>(B_ptr + (int64_t)rn * R_dim + kr + rk);
                }
                *reinterpret_cast<uint4*>(smem_B + rn * B_STRIDE + rk) = v;
            }
        } else {
            for (int idx = tid; idx < BM * 16; idx += 256) {
                int rm = idx >> 4, rk = idx & 15;
                smem_A[rm * A_STRIDE + rk] = (m0 + rm < M_dim && kr + rk < R_dim) ? H_ptr[(int64_t)rm * (3 * R_dim) + kr + rk] : __float2bfloat16(0.0f);
            }
            for (int idx = tid; idx < BN * 16; idx += 256) {
                int rn = idx >> 4, rk = idx & 15;
                smem_B[rn * B_STRIDE + rk] = (n0_loc + rn < N_sub && kr + rk < R_dim) ? B_ptr[(int64_t)rn * R_dim + kr + rk] : __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, smem_A + (warp_m * 16) * A_STRIDE, A_STRIDE);
        wmma::load_matrix_sync(b_frag0, smem_B + (warp_n * 64 + 0) * B_STRIDE, B_STRIDE);
        wmma::load_matrix_sync(b_frag1, smem_B + (warp_n * 64 + 16) * B_STRIDE, B_STRIDE);
        wmma::load_matrix_sync(b_frag2, smem_B + (warp_n * 64 + 32) * B_STRIDE, B_STRIDE);
        wmma::load_matrix_sync(b_frag3, smem_B + (warp_n * 64 + 48) * B_STRIDE, B_STRIDE);
        wmma::mma_sync(acc0, a_frag, b_frag0, acc0);
        wmma::mma_sync(acc1, a_frag, b_frag1, acc1);
        wmma::mma_sync(acc2, a_frag, b_frag2, acc2);
        wmma::mma_sync(acc3, a_frag, b_frag3, acc3);
        __syncthreads();
    }

    float* c_base = smem_C + (warp_m * 16) * C_STRIDE + warp_n * 64;
    wmma::store_matrix_sync(c_base + 0, acc0, C_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(c_base + 16, acc1, C_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(c_base + 32, acc2, C_STRIDE, wmma::mem_row_major);
    wmma::store_matrix_sync(c_base + 48, acc3, C_STRIDE, wmma::mem_row_major);
    __syncthreads();

    // Update Out[64, 128] in-place: 64 * 16 = 1024 uint4 ops = 4 uint4/thread!
    if ((N_sub & 7) == 0 && (n0_tot & 7) == 0) {
        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int vidx = tid + step * 256;
            int rm = vidx >> 4;
            int rn = (vidx & 15) << 3;
            int gm = m0 + rm;
            if (gm < M_dim && n0_loc + rn < N_sub) {
                __nv_bfloat16* ptr = Out + (int64_t)gm * N_tot + n0_tot + rn;
                uint4 yv = *reinterpret_cast<const uint4*>(ptr);
                __nv_bfloat16* yp = reinterpret_cast<__nv_bfloat16*>(&yv);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float val = __bfloat162float(yp[j]) + lora_scale * smem_C[rm * C_STRIDE + rn + j];
                    yp[j] = __float2bfloat16(val * smem_S[rn + j]);
                }
                *reinterpret_cast<uint4*>(ptr) = yv;
            }
        }
    } else {
        for (int idx = tid; idx < BM * BN; idx += 256) {
            int rm = idx >> 7;
            int rn = idx & 127;
            int gm = m0 + rm;
            if (gm < M_dim && n0_loc + rn < N_sub) {
                __nv_bfloat16* ptr = Out + (int64_t)gm * N_tot + n0_tot + rn;
                float val = __bfloat162float(*ptr) + lora_scale * smem_C[rm * C_STRIDE + rn];
                *ptr = __float2bfloat16(val * smem_S[rn]);
            }
        }
    }
}

// Backward Stage 1 (Vectorized 128-bit over cols):
//   Reads go[m, n] and out[m, n]; computes part_dm[cm, n], dy_pre[m, n], dy_fp8[m, n]
//   Grid: (n_chunks, N_tot / 1024), Block: 128 threads -> 8 cols/thread = 128-bit uint4!
__global__ void dora_3x_bwd_scale_dm_fp8_vec8_kernel(
    const __nv_bfloat16* __restrict__ GradOut,
    const __nv_bfloat16* __restrict__ Out,
    const __nv_bfloat16* __restrict__ M_vec,
    const __nv_bfloat16* __restrict__ InvN_vec,
    const float* __restrict__ InvScaleDY,
    __nv_bfloat16* __restrict__ dY_pre,
    __nv_fp8_e4m3* __restrict__ dY_fp8,
    float* __restrict__ Part_dM,
    int M_dim, int N_tot, int chunk_m
) {
    int cm = blockIdx.x;
    int n_vec_base = blockIdx.y * blockDim.x + threadIdx.x; // 1 unit = 8 elements
    int n0 = n_vec_base * 8;
    if (n0 >= N_tot) return;

    uint4 m_u4 = *reinterpret_cast<const uint4*>(M_vec + n0);
    uint4 in_u4 = *reinterpret_cast<const uint4*>(InvN_vec + n0);
    const __nv_bfloat16* mp = reinterpret_cast<const __nv_bfloat16*>(&m_u4);
    const __nv_bfloat16* inp = reinterpret_cast<const __nv_bfloat16*>(&in_u4);

    float s[8], inv_m[8], dm_acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        float mv = __bfloat162float(mp[j]);
        float invn = __bfloat162float(inp[j]);
        s[j] = mv * invn;
        inv_m[j] = 1.0f / fmaxf(mv, 1e-8f);
        dm_acc[j] = 0.0f;
    }
    float inv_sdy = __ldg(InvScaleDY);

    int m_start = cm * chunk_m;
    int m_end = min(m_start + chunk_m, M_dim);
    int n_vec_tot = N_tot >> 3;

    for (int m = m_start; m < m_end; ++m) {
        int64_t row_vec_off = (int64_t)m * n_vec_tot + n_vec_base;
        uint4 go_u4 = reinterpret_cast<const uint4*>(GradOut)[row_vec_off];
        uint4 ov_u4 = reinterpret_cast<const uint4*>(Out)[row_vec_off];
        const __nv_bfloat16* gop = reinterpret_cast<const __nv_bfloat16*>(&go_u4);
        const __nv_bfloat16* ovp = reinterpret_cast<const __nv_bfloat16*>(&ov_u4);

        uint4 dy_pre_u4;
        __nv_bfloat16* dypp = reinterpret_cast<__nv_bfloat16*>(&dy_pre_u4);
        uint16_t packed[4];

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float gf = __bfloat162float(gop[j]);
            float of = __bfloat162float(ovp[j]);
            dm_acc[j] += gf * of;
            float dyp = gf * s[j];
            dypp[j] = __float2bfloat16(dyp);
        }
        reinterpret_cast<uint4*>(dY_pre)[row_vec_off] = dy_pre_u4;

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float gf0 = __bfloat162float(gop[2 * i]) * s[2 * i] * inv_sdy;
            float gf1 = __bfloat162float(gop[2 * i + 1]) * s[2 * i + 1] * inv_sdy;
            __nv_fp8x2_e4m3 fp8_2(make_float2(gf0, gf1));
            packed[i] = *reinterpret_cast<const uint16_t*>(&fp8_2);
        }
        reinterpret_cast<uint2*>(dY_fp8)[row_vec_off] = make_uint2(
            (uint32_t)packed[0] | ((uint32_t)packed[1] << 16),
            (uint32_t)packed[2] | ((uint32_t)packed[3] << 16)
        );
    }

    float* p_dst = Part_dM + (int64_t)cm * N_tot + n0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        p_dst[j] = dm_acc[j] * inv_m[j];
    }
}

__global__ void reduce_dm_chunks_kernel(
    const float* __restrict__ Part_dM,
    __nv_bfloat16* __restrict__ dM_out,
    int n_chunks, int N_tot
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N_tot) return;
    float acc = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        acc += Part_dM[(int64_t)c * N_tot + n];
    }
    dM_out[n] = __float2bfloat16(acc);
}

} // namespace

// Helper: 2D BF16 Matmul using cublasLt/cublas on arbitrary stream (row-major out = A[M,K] @ B^T[N,K] or flags)
static void cublaslt_bf16_gemm(
    cublasOperation_t opA_rm, cublasOperation_t opB_rm,
    int m, int n, int k,
    const void* A, int lda, int64_t strideA,
    const void* B, int ldb, int64_t strideB,
    void* C_out, int ldc, int64_t strideC,
    int batch_count, float alpha, float beta,
    cudaStream_t stream,
    void* ws, size_t ws_bytes
) {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t layB = nullptr, layA = nullptr, layC = nullptr;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opB_rm, sizeof(opB_rm)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opA_rm, sizeof(opA_rm)));

    int b_rows_cm = (opB_rm == CUBLAS_OP_N) ? n : k;
    int b_cols_cm = (opB_rm == CUBLAS_OP_N) ? k : n;
    int a_rows_cm = (opA_rm == CUBLAS_OP_N) ? k : m;
    int a_cols_cm = (opA_rm == CUBLAS_OP_N) ? m : k;

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layB, CUDA_R_16BF, b_rows_cm, b_cols_cm, ldb));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layA, CUDA_R_16BF, a_rows_cm, a_cols_cm, lda));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layC, CUDA_R_16BF, n, m, ldc));
    if (batch_count > 1) {
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layB, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layB, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideB, sizeof(strideB)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layA, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layA, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideA, sizeof(strideA)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layC, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
        CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(layC, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strideC, sizeof(strideC)));
    }

    cublasLtMatmulPreference_t pref = nullptr;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
    cublasLtMatmulHeuristicResult_t res;
    int found = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(g_lt.handle, desc, layB, layA, layC, layC, pref, 1, &res, &found));
    TORCH_CHECK(found > 0, "BF16 cuBLASLt heuristic failed m=", m, " n=", n, " k=", k, " batch=", batch_count);

    CUBLAS_CHECK(cublasLtMatmul(
        g_lt.handle, desc, &alpha, B, layB, A, layA, &beta, C_out, layC, C_out, layC,
        &res.algo, ws, ws_bytes, stream
    ));

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layC);
    cublasLtMatrixLayoutDestroy(layA);
    cublasLtMatrixLayoutDestroy(layB);
    cublasLtMatmulDescDestroy(desc);
}

std::tuple<torch::Tensor, torch::Tensor> dora_3x_fp8_fwd_cuda(
    torch::Tensor x2,        // [M, K] bf16 contiguous
    torch::Tensor w0_fp8,    // [N_tot, K] float8_e4m3fn contiguous
    torch::Tensor sw,        // [1] float32
    torch::Tensor sx,        // [1] float32
    torch::Tensor inv_sx,    // [1] float32
    torch::Tensor la_cat,    // [3*r, K] bf16 contiguous
    torch::Tensor b_cat,     // [3, N_sub, r] bf16 contiguous
    torch::Tensor m_cat,     // [N_tot] bf16 contiguous
    torch::Tensor inv_n_cat, // [N_tot] bf16 contiguous
    double lora_scale
) {
    g_lt.init_once();
    int M = (int)x2.size(0);
    int K = (int)x2.size(1);
    int N_tot = (int)w0_fp8.size(0);
    int N_sub = N_tot / 3;
    int r = (int)b_cat.size(2);
    auto opts_bf16 = x2.options();
    auto opts_fp8 = x2.options().dtype(torch::kFloat8_e4m3fn);

    cudaStream_t main_stream = at::cuda::getCurrentCUDAStream().stream();

    torch::Tensor h_a_cat = torch::empty({M, 3 * r}, opts_bf16);
    torch::Tensor x_fp8 = torch::empty({M, K}, opts_fp8);
    torch::Tensor out = torch::empty({M, N_tot}, opts_bf16);

    // Run sequential on main_stream (avoid aux_stream contention/overhead for tiny h_a GEMM)
    cublaslt_bf16_gemm(
        CUBLAS_OP_N, CUBLAS_OP_T,
        M, 3 * r, K,
        x2.data_ptr(), K, 0, la_cat.data_ptr(), K, 0,
        h_a_cat.data_ptr(), 3 * r, 0,
        1, 1.0f, 0.0f, main_stream, g_lt.workspace, g_lt.workspace_size
    );

    int64_t n_elems = (int64_t)M * K;
    if ((n_elems & 7) == 0) {
        int64_t n_vec8 = n_elems >> 3;
        quant_bf16_to_fp8_vec8_kernel<<<(int)((n_vec8 + 255) / 256), 256, 0, main_stream>>>(
            (const __nv_bfloat16*)x2.data_ptr(), (const float*)inv_sx.data_ptr(),
            (__nv_fp8_e4m3*)x_fp8.data_ptr(), n_vec8
        );
    } else {
        quant_bf16_to_fp8_scalar_kernel<<<(int)((n_elems + 255) / 256), 256, 0, main_stream>>>(
            (const __nv_bfloat16*)x2.data_ptr(), (const float*)inv_sx.data_ptr(),
            (__nv_fp8_e4m3*)x_fp8.data_ptr(), n_elems
        );
    }

    cublasLtMatmulAlgo_t algo = select_best_fp8_algo(
        M, N_tot, K,
        w0_fp8.data_ptr(), x_fp8.data_ptr(), out.data_ptr(),
        (const float*)sw.data_ptr(), (const float*)sx.data_ptr(),
        true, main_stream
    );
    run_cublaslt_fp8(
        M, N_tot, K,
        w0_fp8.data_ptr(), x_fp8.data_ptr(), out.data_ptr(),
        (const float*)sw.data_ptr(), (const float*)sx.data_ptr(),
        algo, main_stream
    );

    int n_blks_sub = (N_sub + 127) / 128;
    dim3 grid((M + 63) / 64, 3 * n_blks_sub);
    dora_3x_lora_scale_fwd_wmma_inplace_kernel<<<grid, 256, 0, main_stream>>>(
        (__nv_bfloat16*)out.data_ptr(),
        (const __nv_bfloat16*)h_a_cat.data_ptr(),
        (const __nv_bfloat16*)b_cat.data_ptr(),
        (const __nv_bfloat16*)m_cat.data_ptr(),
        (const __nv_bfloat16*)inv_n_cat.data_ptr(),
        M, N_sub, r, (float)lora_scale, n_blks_sub
    );

    return std::make_tuple(out, h_a_cat);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> dora_3x_fp8_bwd_cuda(
    torch::Tensor go2,       // [M, N_tot] bf16 contiguous
    torch::Tensor out,       // [M, N_tot] bf16 contiguous
    torch::Tensor x2,        // [M, K] bf16 contiguous
    torch::Tensor w0_T_fp8,  // [K, N_tot] float8_e4m3fn contiguous
    torch::Tensor sw,        // [1] float32
    torch::Tensor sdy,       // [1] float32
    torch::Tensor inv_sdy,   // [1] float32
    torch::Tensor la_cat,    // [3*r, K] bf16 contiguous
    torch::Tensor b_cat,     // [3, N_sub, r] bf16 contiguous
    torch::Tensor m_cat,     // [N_tot] bf16 contiguous
    torch::Tensor inv_n_cat, // [N_tot] bf16 contiguous
    torch::Tensor h_a_cat,   // [M, 3*r] bf16 contiguous
    double lora_scale
) {
    g_lt.init_once();
    int M = (int)go2.size(0);
    int N_tot = (int)go2.size(1);
    int N_sub = N_tot / 3;
    int K = (int)x2.size(1);
    int r = (int)b_cat.size(2);
    auto opts_bf16 = go2.options();
    auto opts_fp8 = go2.options().dtype(torch::kFloat8_e4m3fn);
    auto opts_f32 = go2.options().dtype(torch::kFloat32);

    cudaStream_t main_stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int CHUNK_M = 128;
    int n_chunks = (M + CHUNK_M - 1) / CHUNK_M;

    torch::Tensor dy_pre = torch::empty({M, N_tot}, opts_bf16);
    torch::Tensor dy_fp8 = torch::empty({M, N_tot}, opts_fp8);
    torch::Tensor part_dm = torch::empty({n_chunks, N_tot}, opts_f32);
    torch::Tensor dm = torch::empty({N_tot}, opts_bf16);
    torch::Tensor dx = torch::empty({M, K}, opts_bf16);

    TORCH_CHECK((N_tot & 7) == 0, "N_tot must be divisible by 8");
    int n_vec_tot = N_tot >> 3;
    dim3 dm_grid(n_chunks, (n_vec_tot + 127) / 128);
    dora_3x_bwd_scale_dm_fp8_vec8_kernel<<<dm_grid, 128, 0, main_stream>>>(
        (const __nv_bfloat16*)go2.data_ptr(),
        (const __nv_bfloat16*)out.data_ptr(),
        (const __nv_bfloat16*)m_cat.data_ptr(),
        (const __nv_bfloat16*)inv_n_cat.data_ptr(),
        (const float*)inv_sdy.data_ptr(),
        (__nv_bfloat16*)dy_pre.data_ptr(),
        (__nv_fp8_e4m3*)dy_fp8.data_ptr(),
        (float*)part_dm.data_ptr(),
        M, N_tot, CHUNK_M
    );

    CUDA_CHECK(cudaEventRecord(g_lt.ev_fork, main_stream));
    CUDA_CHECK(cudaStreamWaitEvent(g_lt.aux_stream, g_lt.ev_fork, 0));

    cublasLtMatmulAlgo_t algo = select_best_fp8_algo(
        M, K, N_tot,
        w0_T_fp8.data_ptr(), dy_fp8.data_ptr(), dx.data_ptr(),
        (const float*)sw.data_ptr(), (const float*)sdy.data_ptr(),
        true, main_stream
    );
    run_cublaslt_fp8(
        M, K, N_tot,
        w0_T_fp8.data_ptr(), dy_fp8.data_ptr(), dx.data_ptr(),
        (const float*)sw.data_ptr(), (const float*)sdy.data_ptr(),
        algo, main_stream
    );

    reduce_dm_chunks_kernel<<<(N_tot + 255) / 256, 256, 0, g_lt.aux_stream>>>(
        (const float*)part_dm.data_ptr(), (__nv_bfloat16*)dm.data_ptr(), n_chunks, N_tot
    );

    torch::Tensor d_b_cat = torch::empty({3, N_sub, r}, opts_bf16);
    torch::Tensor dh_a_cat = torch::empty({M, 3 * r}, opts_bf16);
    torch::Tensor d_la_cat = torch::empty({3 * r, K}, opts_bf16);

    float lsc = (float)lora_scale;
    // Batched Strided GEMMs (batch=3 in ONE cuBLASLt call instead of 3!)
    cublaslt_bf16_gemm(
        CUBLAS_OP_T, CUBLAS_OP_N,
        N_sub, r, M,
        dy_pre.data_ptr(), N_tot, N_sub,
        h_a_cat.data_ptr(), 3 * r, r,
        d_b_cat.data_ptr(), r, (int64_t)N_sub * r,
        3, lsc, 0.0f, g_lt.aux_stream, g_lt.aux_workspace, g_lt.aux_workspace_size
    );
    cublaslt_bf16_gemm(
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, r, N_sub,
        dy_pre.data_ptr(), N_tot, N_sub,
        b_cat.data_ptr(), r, (int64_t)N_sub * r,
        dh_a_cat.data_ptr(), 3 * r, r,
        3, lsc, 0.0f, g_lt.aux_stream, g_lt.aux_workspace, g_lt.aux_workspace_size
    );
    cublaslt_bf16_gemm(
        CUBLAS_OP_T, CUBLAS_OP_N,
        3 * r, K, M,
        dh_a_cat.data_ptr(), 3 * r, 0, x2.data_ptr(), K, 0,
        d_la_cat.data_ptr(), K, 0,
        1, 1.0f, 0.0f, g_lt.aux_stream, g_lt.aux_workspace, g_lt.aux_workspace_size
    );

    CUDA_CHECK(cudaEventRecord(g_lt.ev_join, g_lt.aux_stream));
    CUDA_CHECK(cudaStreamWaitEvent(main_stream, g_lt.ev_join, 0));

    cublaslt_bf16_gemm(
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, K, 3 * r,
        dh_a_cat.data_ptr(), 3 * r, 0, la_cat.data_ptr(), K, 0,
        dx.data_ptr(), K, 0,
        1, 1.0f, 1.0f, main_stream, g_lt.workspace, g_lt.workspace_size
    );

    return std::make_tuple(dx, d_la_cat, d_b_cat, dm);
}

// Lightweight breakdown helper for H100 profiling (returns [t_ha, t_qx, t_fp8, t_epilogue] ms)
std::vector<double> dora_3x_fp8_fwd_profile_cuda(
    torch::Tensor x2, torch::Tensor w0_fp8, torch::Tensor sw, torch::Tensor sx, torch::Tensor inv_sx,
    torch::Tensor la_cat, torch::Tensor b_cat, torch::Tensor m_cat, torch::Tensor inv_n_cat, double lora_scale
) {
    g_lt.init_once();
    int M = (int)x2.size(0), K = (int)x2.size(1), N_tot = (int)w0_fp8.size(0), N_sub = N_tot / 3, r = (int)b_cat.size(2);
    cudaStream_t st = at::cuda::getCurrentCUDAStream().stream();
    auto ha = torch::empty({M, 3 * r}, x2.options());
    auto xfp8 = torch::empty({M, K}, x2.options().dtype(torch::kFloat8_e4m3fn));
    auto out = torch::empty({M, N_tot}, x2.options());
    auto algo = select_best_fp8_algo(M, N_tot, K, w0_fp8.data_ptr(), xfp8.data_ptr(), out.data_ptr(), (const float*)sw.data_ptr(), (const float*)sx.data_ptr(), true, st);
    cudaEvent_t e0, e1, e2, e3, e4;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3); cudaEventCreate(&e4);
    cudaEventRecord(e0, st);
    cublaslt_bf16_gemm(CUBLAS_OP_N, CUBLAS_OP_T, M, 3 * r, K, x2.data_ptr(), K, 0, la_cat.data_ptr(), K, 0, ha.data_ptr(), 3 * r, 0, 1, 1.0f, 0.0f, st, g_lt.workspace, g_lt.workspace_size);
    cudaEventRecord(e1, st);
    quant_bf16_to_fp8_vec8_kernel<<<(int)(((int64_t)M * K / 8 + 255) / 256), 256, 0, st>>>((const __nv_bfloat16*)x2.data_ptr(), (const float*)inv_sx.data_ptr(), (__nv_fp8_e4m3*)xfp8.data_ptr(), (int64_t)M * K / 8);
    cudaEventRecord(e2, st);
    run_cublaslt_fp8(M, N_tot, K, w0_fp8.data_ptr(), xfp8.data_ptr(), out.data_ptr(), (const float*)sw.data_ptr(), (const float*)sx.data_ptr(), algo, st);
    cudaEventRecord(e3, st);
    int n_blks_sub = (N_sub + 127) / 128;
    dora_3x_lora_scale_fwd_wmma_inplace_kernel<<<dim3((M + 63) / 64, 3 * n_blks_sub), 256, 0, st>>>((__nv_bfloat16*)out.data_ptr(), (const __nv_bfloat16*)ha.data_ptr(), (const __nv_bfloat16*)b_cat.data_ptr(), (const __nv_bfloat16*)m_cat.data_ptr(), (const __nv_bfloat16*)inv_n_cat.data_ptr(), M, N_sub, r, (float)lora_scale, n_blks_sub);
    cudaEventRecord(e4, st);
    cudaEventSynchronize(e4);
    float t0=0, t1=0, t2=0, t3=0;
    cudaEventElapsedTime(&t0, e0, e1);
    cudaEventElapsedTime(&t1, e1, e2);
    cudaEventElapsedTime(&t2, e2, e3);
    cudaEventElapsedTime(&t3, e3, e4);
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2); cudaEventDestroy(e3); cudaEventDestroy(e4);
    return {t0, t1, t2, t3};
}
