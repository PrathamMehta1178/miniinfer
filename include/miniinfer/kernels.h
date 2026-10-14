#pragma once
// Matrix-multiply kernels. All matrices are row-major float arrays.
//
//   C[M x N] = A[M x K] * B[K x N]
//
// Every kernel OVERWRITES C (it does not add to what was there).
// Element (i, j) of a matrix with `cols` columns lives at index i * cols + j.

namespace miniinfer {

// Step 1: the textbook triple loop, i -> j -> k. Slow but easy to trust.
// Every other kernel is tested against this one.
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K);

// Step 2: same math, loops reordered to i -> k -> j so the inner loop
// walks B and C in memory order (cache friendly).
void matmul_reordered(const float* A, const float* B, float* C, int M, int N, int K);

// Step 3: cache blocking (tiling). Work on tile x tile sub-blocks that fit in cache.
// Must handle sizes that are not multiples of `tile`.
void matmul_blocked(const float* A, const float* B, float* C, int M, int N, int K, int tile = 64);

// Step 4: blocked + OpenMP threads over the outer loop.
void matmul_parallel(const float* A, const float* B, float* C, int M, int N, int K, int tile = 64);

// Which kernel the model uses. Lets the benchmark and tests swap backends.
enum class Kernel { Naive, Reordered, Blocked, Parallel };
const char* kernel_name(Kernel k);
void matmul(Kernel k, const float* A, const float* B, float* C, int M, int N, int K);

// Elementwise helpers.
void add_bias_rows(float* Y, const float* bias, int rows, int cols);  // Y[r][c] += bias[c]
void relu_inplace(float* x, int n);                                   // x = max(0, x)

}  // namespace miniinfer
