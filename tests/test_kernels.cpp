// Checks every matmul kernel and helper against a simple double-precision reference.
// Odd sizes are deliberate: they catch tile-edge bugs in the blocked kernels.
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "miniinfer/kernels.h"

using namespace miniinfer;

namespace {

int failures = 0;

void report(const char* name, bool ok, const char* detail) {
    std::printf("  [%s] %-28s %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (!ok) ++failures;
}

// Reference in double precision. Independent of your code, so it can check matmul_naive too.
void reference_matmul(const std::vector<float>& A, const std::vector<float>& B, std::vector<double>& C,
                      int M, int N, int K) {
    C.assign(static_cast<size_t>(M) * N, 0.0);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            double s = 0;
            for (int k = 0; k < K; ++k) s += double(A[i * K + k]) * B[k * N + j];
            C[i * N + j] = s;
        }
}

void test_matmul(Kernel kern, int M, int N, int K, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> A(static_cast<size_t>(M) * K), B(static_cast<size_t>(K) * N);
    for (auto& v : A) v = dist(rng);
    for (auto& v : B) v = dist(rng);
    // Fill C with junk to catch kernels that forget to zero it.
    std::vector<float> C(static_cast<size_t>(M) * N, 12345.f);
    std::vector<double> ref;
    reference_matmul(A, B, ref, M, N, K);

    char name[64], detail[96];
    std::snprintf(name, sizeof(name), "%s %dx%dx%d", kernel_name(kern), M, N, K);
    try {
        matmul(kern, A.data(), B.data(), C.data(), M, N, K);
    } catch (const std::logic_error& e) {
        report(name, false, e.what());
        return;
    }
    // float32 sums of K terms: allow error that grows with sqrt(K).
    const double tol = 1e-5 * std::sqrt(double(K)) * 4;
    double max_err = 0;
    for (size_t i = 0; i < C.size(); ++i) max_err = std::max(max_err, std::fabs(C[i] - ref[i]));
    std::snprintf(detail, sizeof(detail), "max err %.2e (tol %.1e)", max_err, tol);
    report(name, max_err <= tol, detail);
}

void test_helpers() {
    std::vector<float> x = {-2.f, -0.f, 0.5f, 3.f};
    try {
        relu_inplace(x.data(), 4);
        bool ok = x[0] == 0.f && x[1] == 0.f && x[2] == 0.5f && x[3] == 3.f;
        report("relu_inplace", ok, ok ? "" : "wrong values");
    } catch (const std::logic_error& e) {
        report("relu_inplace", false, e.what());
    }

    std::vector<float> Y = {1, 2, 3, 4, 5, 6};  // 2 rows x 3 cols
    std::vector<float> b = {10, 20, 30};
    try {
        add_bias_rows(Y.data(), b.data(), 2, 3);
        std::vector<float> want = {11, 22, 33, 14, 25, 36};
        bool ok = Y == want;
        report("add_bias_rows", ok, ok ? "" : "wrong values");
    } catch (const std::logic_error& e) {
        report("add_bias_rows", false, e.what());
    }
}

}  // namespace

int main() {
    std::printf("helpers\n");
    test_helpers();

    std::mt19937 rng(42);
    const int sizes[][3] = {
        {1, 1, 1}, {3, 5, 7}, {37, 101, 53}, {64, 64, 64}, {65, 63, 129}, {32, 128, 784}, {200, 10, 128},
    };
    for (Kernel k : {Kernel::Naive, Kernel::Reordered, Kernel::Blocked, Kernel::Parallel}) {
        std::printf("%s\n", kernel_name(k));
        for (const auto& s : sizes) test_matmul(k, s[0], s[1], s[2], rng);
    }

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
