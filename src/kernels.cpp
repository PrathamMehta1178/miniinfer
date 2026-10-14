#include "miniinfer/kernels.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

// Throws so tests report "not implemented" instead of silently passing garbage.
#define MINIINFER_TODO(what) throw std::logic_error(std::string("TODO: implement ") + what)

namespace miniinfer {

// ---------------------------------------------------------------------------
// YOUR CODE: elementwise helpers (warm-up)
// ---------------------------------------------------------------------------

void add_bias_rows(float* Y, const float* bias, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            Y[r * cols + c] += bias[c];
        }
    }
}

void relu_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) {
        if (x[i] < 0) {
            x[i] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// YOUR CODE: matmul kernels. C[M x N] = A[M x K] * B[K x N], overwrite C.
// Guide: section 5. Run ./build/test_kernels after each one.
// ---------------------------------------------------------------------------

void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {          // each row of A (each image)
        for (int j = 0; j < N; j++) {      // each column of B (each output neuron)
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {  // walk along the shared dimension
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void matmul_reordered(const float* A, const float* B, float* C, int M, int N, int K) {
    std::fill(C, C + (size_t)M * N, 0.0f);   // start C at zero
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a = A[i * K + k];          // one number from A, reused across the whole row
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

void matmul_blocked(const float* A, const float* B, float* C, int M, int N, int K, int tile) {
    std::fill(C, C + (size_t)M * N, 0.0f);
    for (int i0 = 0; i0 < M; i0 += tile) {
        for (int k0 = 0; k0 < K; k0 += tile) {
            for (int j0 = 0; j0 < N; j0 += tile) {
                // Work on one tile: rows i0.., inner dim k0.., columns j0..
                int i_end = std::min(i0 + tile, M);
                int k_end = std::min(k0 + tile, K);
                int j_end = std::min(j0 + tile, N);
                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a = A[i * K + k];
                        for (int j = j0; j < j_end; j++) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

void matmul_parallel(const float* A, const float* B, float* C, int M, int N, int K, int tile) {
    std::fill(C, C + (size_t)M * N, 0.0f);
    #pragma omp parallel for
    for (int i0 = 0; i0 < M; i0 += tile) {
        for (int k0 = 0; k0 < K; k0 += tile) {
            for (int j0 = 0; j0 < N; j0 += tile) {
                int i_end = std::min(i0 + tile, M);
                int k_end = std::min(k0 + tile, K);
                int j_end = std::min(j0 + tile, N);
                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a = A[i * K + k];
                        for (int j = j0; j < j_end; j++) {
                            C[i * N + j] += a * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Provided: kernel dispatch
// ---------------------------------------------------------------------------

const char* kernel_name(Kernel k) {
    switch (k) {
        case Kernel::Naive: return "naive";
        case Kernel::Reordered: return "reordered";
        case Kernel::Blocked: return "blocked";
        case Kernel::Parallel: return "parallel";
    }
    return "unknown";
}

void matmul(Kernel k, const float* A, const float* B, float* C, int M, int N, int K) {
    switch (k) {
        case Kernel::Naive: matmul_naive(A, B, C, M, N, K); return;
        case Kernel::Reordered: matmul_reordered(A, B, C, M, N, K); return;
        case Kernel::Blocked: matmul_blocked(A, B, C, M, N, K); return;
        case Kernel::Parallel: matmul_parallel(A, B, C, M, N, K); return;
    }
}

}  // namespace miniinfer
