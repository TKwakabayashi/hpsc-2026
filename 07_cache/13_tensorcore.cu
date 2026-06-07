#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
#include <cuda_fp16.h>
using namespace std;
using namespace nvcuda;

#define WARP_SIZE 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 128
#define BN 128
#define BK 64

#define WARP_M_COUNT 4
#define WARP_N_COUNT 2
#define WARPS_PER_BLOCK (WARP_M_COUNT * WARP_N_COUNT) // 8
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * WARP_SIZE) // 256

#define M_TILES (BM / WMMA_M) // 8
#define N_TILES (BN / WMMA_N) // 8
#define WARP_ROW_TILES (M_TILES / WARP_M_COUNT) // 2
#define WARP_COL_TILES (N_TILES / WARP_N_COUNT) // 4

#define VEC 8

#define PAD_A 8
#define PAD_B 8
#define LDA_S (BM + PAD_A) // 136
#define LDB_S (BK + PAD_B) // 72
#define A_STAGE (BK * LDA_S)
#define B_STAGE (BN * LDB_S)
#define A_TOTAL (2 * A_STAGE)

#define NUM_ITER 10
#define NUM_WARMUP 2

__device__ __forceinline__ void cp_async_cg(void *smem, const void *gmem) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void cp_async_wait_all() {asm volatile("cp.async.wait_all;\n"); }

__global__ void f2h(const float *src, half *dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

__device__ __forceinline__ void load_tile(
    half a_sh[BK][BM + PAD_A], half b_sh[BN][BK + PAD_B],
    const half *Ah, const half *Bh,
    int dim_m, int dim_k, int offset_a_m, int offset_b_n, int k0, int tid) {

    for (int idx = tid; idx < BK * (BM / VEC); idx += THREADS_PER_BLOCK) {
        int k_local = idx / (BM / VEC);
        int m_chunk = idx % (BM / VEC);
        cp_async_cg(&a_sh[k_local][m_chunk * VEC], &Ah[(size_t)(k0 + k_local) * dim_m + offset_a_m + m_chunk * VEC]);
    }

    for (int idx = tid; idx < (BK / VEC) * BN; idx += THREADS_PER_BLOCK) {
        int n_local = idx / (BK / VEC);
        int k_chunk = idx % (BK / VEC);
        cp_async_cg(&b_sh[n_local][k_chunk * VEC], &Bh[(size_t)(offset_b_n + n_local) * dim_k + (k0 + k_chunk * VEC)]);
    }
}

__device__ __forceinline__ void load_tile_dyn(
    half *As, half *Bs, const half *Ah, const half *Bh,
    int dim_m, int dim_k, int offset_a_m, int offset_b_n, int k0, int tid) {

    for (int idx = tid; idx < BK * (BM / VEC); idx += THREADS_PER_BLOCK) {
        int kl = idx / (BM / VEC);
        int mc = idx % (BM / VEC);
        cp_async_cg(As + kl * LDA_S + mc * VEC, &Ah[(size_t)(k0 + kl) * dim_m + offset_a_m + mc * VEC]);
    }

    for (int idx = tid; idx < (BK / VEC) * BN; idx += THREADS_PER_BLOCK) {
        int nl = idx / (BK / VEC);
        int kc = idx % (BK / VEC);
        cp_async_cg(Bs + nl * LDB_S + kc * VEC, &Bh[(size_t)(offset_b_n + nl) * dim_k + (k0 + kc * VEC)]);
    }
}

__global__ void kernel(int dim_m, int dim_n, int dim_k,
		       const half *Ah, const half *Bh, float *d_c) {

    extern __shared__ half smem[];
    half *Asbuf = smem;
    half *Bsbuf = smem + A_TOTAL;

    int offset_a_m = BM * blockIdx.x;
    int offset_b_n = BN * blockIdx.y;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int warp_m = warp_id % WARP_M_COUNT;
    int warp_n = warp_id / WARP_M_COUNT;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WARP_ROW_TILES][WARP_COL_TILES];
    for (int r = 0; r < WARP_ROW_TILES; r++)
        for (int c = 0; c < WARP_COL_TILES; c++)
            wmma::fill_fragment(acc[r][c], 0.0f);

    const int NK = dim_k / BK;

    load_tile_dyn(Asbuf, Bsbuf, Ah, Bh, dim_m, dim_k, offset_a_m, offset_b_n, 0, tid);
    cp_async_commit();

    for (int kt = 0; kt < NK; kt++) {
        cp_async_wait_all();
        __syncthreads();

        const int cur = kt & 1;
        const int nxt = (kt + 1) & 1;

        if (kt + 1 < NK) {
            load_tile_dyn(Asbuf + nxt * A_STAGE, Bsbuf + nxt * B_STAGE, Ah, Bh, dim_m, dim_k, offset_a_m, offset_b_n, (kt + 1) * BK, tid);
            cp_async_commit();
        }

        half *As = Asbuf + cur * A_STAGE;
        half *Bs = Bsbuf + cur * B_STAGE;
        
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            for (int r = 0; r < WARP_ROW_TILES; r++) {
                int row_tile = warp_m * WARP_ROW_TILES + r;
                wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
                wmma::load_matrix_sync(a_frag, As + kk * LDA_S + row_tile * WMMA_M, LDA_S);
                for (int c = 0; c <  WARP_COL_TILES; c++) {
                    int col_tile = warp_n * WARP_COL_TILES + c;
                    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
                    wmma::load_matrix_sync(b_frag, Bs + (col_tile * WMMA_N) * LDB_S + kk, LDB_S);
                    wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
                }
            }
        }
        __syncthreads();
    }
    for (int r = 0; r < WARP_ROW_TILES; r++) {
        for (int c = 0; c < WARP_COL_TILES; c++) {
            int row_tile = warp_m * WARP_ROW_TILES + r;
            int col_tile = warp_n * WARP_COL_TILES + c;
            int c_m = offset_a_m + row_tile * WMMA_M;
            int c_n = offset_b_n + col_tile * WMMA_N;
            if (c_n < dim_n && c_m < dim_m)
                wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
        }
    }
}

int main(int argc, const char **argv) {
    int m = 10240;
    int k = 4096;
    int n = 8192;
    float alpha = 1.0;
    float beta = 0.0;
    float *A, *B, *C, *C2;
    half *Ah, *Bh;
    cudaMallocManaged(&A, m * k * sizeof(float));
    cudaMallocManaged(&B, k * n * sizeof(float));
    cudaMallocManaged(&C, m * n * sizeof(float));
    cudaMallocManaged(&C2, m * n * sizeof(float));
    cudaMalloc(&Ah, m * k * sizeof(half));
    cudaMalloc(&Bh, k * n * sizeof(half));
    for (int i=0; i<m; i++)
        for (int j=0; j<k; j++)
            A[k*i+j] = drand48();
    for (int i=0; i<k; i++)
        for (int j=0; j<n; j++)
            B[n*i+j] = drand48();
    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
            C[m*i+j] = C2[m*i+j] = 0;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    auto tic = chrono::steady_clock::now();
    for (int i = 0; i < NUM_ITER+NUM_WARMUP; i++) {
        if (i == NUM_WARMUP) tic = chrono::steady_clock::now();
            cublasGemmEx(cublas_handle,
            	CUBLAS_OP_N,
                CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                A, CUDA_R_32F, m,
                B, CUDA_R_32F, k,
                &beta,
                C, CUDA_R_32F, m,
                CUBLAS_COMPUTE_32F_FAST_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
    }
    auto toc = chrono::steady_clock::now();
    int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
    double tcublas = chrono::duration<double>(toc - tic).count() / NUM_ITER;
    double cublas_flops = double(num_flops) / tcublas / 1.0e9;

    dim3 block(THREADS_PER_BLOCK);
    dim3 grid((m + BM - 1) / BM, (n + BN -1) / BN);
    size_t shmem = (size_t)(A_TOTAL + 2 * B_STAGE) * sizeof(half);
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    for (int i = 0; i < NUM_ITER+NUM_WARMUP; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
            f2h<<<((size_t)m*k + 255)/256, 256>>>(A, Ah, (size_t)m*k);
            f2h<<<((size_t)k*n + 255)/256, 256>>>(B, Bh, (size_t)k*n);
            kernel<<< grid, block, shmem>>>(m,
                n,
                k,
                Ah,
                Bh,
                C2);
            cudaDeviceSynchronize();
    }
    toc = chrono::steady_clock::now();
    double tcutlass = chrono::duration<double>(toc - tic).count() / NUM_ITER;
    double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;
    printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);
    double err = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            err += fabs(C[m*i+j] - C2[m*i+j]);
        }
    }
    printf("error: %lf\n", err/n/m);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C2);
    cudaFree(Ah);
    cudaFree(Bh);
    cublasDestroy(cublas_handle);
}