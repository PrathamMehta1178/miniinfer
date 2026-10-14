// Benchmarks: raw matmul kernels, then end-to-end model inference.
// Usage: ./build/bench [--csv] [--data DIR]
// Set the thread count with: OMP_NUM_THREADS=4 ./build/bench
// Guide: section 7 (warm-up, median, p95, one change at a time).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "miniinfer/model.h"

using namespace miniinfer;
using Clock = std::chrono::steady_clock;

namespace {

bool g_csv = false;

struct Stats {
    double median_ms = 0, p95_ms = 0;
    int reps = 0;
};

// Warm up, then time repeated runs until the time budget or max reps is reached.
template <typename F>
Stats measure(F&& fn, int warmup = 3, int min_reps = 5, int max_reps = 100, double budget_s = 1.0) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> ms;
    const auto start = Clock::now();
    while (int(ms.size()) < max_reps) {
        const auto t0 = Clock::now();
        fn();
        ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        if (int(ms.size()) >= min_reps && elapsed > budget_s) break;
    }
    std::sort(ms.begin(), ms.end());
    Stats s;
    s.reps = int(ms.size());
    s.median_ms = ms[ms.size() / 2];
    s.p95_ms = ms[std::min(ms.size() - 1, size_t(ms.size() * 0.95))];
    return s;
}

int threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void bench_matmul() {
    const int sizes[] = {64, 128, 256, 512, 1024};
    if (g_csv) std::printf("section,kernel,M,N,K,median_ms,p95_ms,gflops,reps\n");
    else std::printf("\nMATMUL  C[n x n] = A[n x n] * B[n x n]\n%-10s %6s %12s %12s %10s %6s\n",
                     "kernel", "n", "median ms", "p95 ms", "GFLOP/s", "reps");

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (int n : sizes) {
        std::vector<float> A(size_t(n) * n), B(size_t(n) * n), C(size_t(n) * n);
        for (auto& v : A) v = dist(rng);
        for (auto& v : B) v = dist(rng);
        const double flops = 2.0 * n * n * n;

        for (Kernel k : {Kernel::Naive, Kernel::Reordered, Kernel::Blocked, Kernel::Parallel}) {
            // The naive kernel gets very slow at large n; keep its run short.
            const bool slow = k == Kernel::Naive && n >= 1024;
            Stats s;
            try {
                s = measure([&] { matmul(k, A.data(), B.data(), C.data(), n, n, n); },
                            slow ? 1 : 3, slow ? 3 : 5);
            } catch (const std::logic_error&) {
                if (!g_csv) std::printf("%-10s %6d   (not implemented yet)\n", kernel_name(k), n);
                continue;
            }
            const double gflops = flops / (s.median_ms * 1e6);
            if (g_csv)
                std::printf("matmul,%s,%d,%d,%d,%.4f,%.4f,%.3f,%d\n", kernel_name(k), n, n, n, s.median_ms,
                            s.p95_ms, gflops, s.reps);
            else
                std::printf("%-10s %6d %12.3f %12.3f %10.2f %6d\n", kernel_name(k), n, s.median_ms, s.p95_ms,
                            gflops, s.reps);
        }
    }
}

void bench_model(const std::string& dir) {
    Model model;
    try {
        model.load(dir + "/model.bin");
    } catch (const std::exception& e) {
        if (!g_csv) std::printf("\nMODEL  skipped: %s\n", e.what());
        return;
    }
    const int batches[] = {1, 8, 64, 256};
    model.prepare(256);
    std::vector<float> input(size_t(256) * model.input_dim());
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (auto& v : input) v = dist(rng);

    if (g_csv) std::printf("section,kernel,batch,median_ms,p95_ms,images_per_s,reps\n");
    else std::printf("\nMODEL  end-to-end forward pass\n%-10s %6s %12s %12s %14s %6s\n", "kernel", "batch",
                     "median ms", "p95 ms", "images/s", "reps");

    for (Kernel k : {Kernel::Naive, Kernel::Reordered, Kernel::Blocked, Kernel::Parallel}) {
        model.kernel = k;
        for (int b : batches) {
            Stats s;
            try {
                s = measure([&] { model.forward(input.data(), b); });
            } catch (const std::logic_error&) {
                if (!g_csv) std::printf("%-10s %6d   (not implemented yet)\n", kernel_name(k), b);
                break;
            }
            const double ips = b / (s.median_ms / 1000.0);
            if (g_csv)
                std::printf("model,%s,%d,%.4f,%.4f,%.1f,%d\n", kernel_name(k), b, s.median_ms, s.p95_ms, ips, s.reps);
            else
                std::printf("%-10s %6d %12.4f %12.4f %14.0f %6d\n", kernel_name(k), b, s.median_ms, s.p95_ms, ips,
                            s.reps);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = "data";
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--csv")) g_csv = true;
        else if (!std::strcmp(argv[i], "--data") && i + 1 < argc) dir = argv[++i];
    }
    if (!g_csv) std::printf("MiniInfer benchmark | OpenMP threads: %d\n", threads());
    bench_matmul();
    bench_model(dir);
    return 0;
}
