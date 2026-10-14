"""PyTorch baseline with the same sizes and method as bench.cpp, for a fair comparison.

Usage:  python bench/bench_pytorch.py --threads 4
Use the same thread count as OMP_NUM_THREADS for the C++ run.
"""
import argparse
import time

import torch


def measure(fn, warmup=3, min_reps=5, max_reps=100, budget_s=1.0):
    for _ in range(warmup):
        fn()
    times = []
    start = time.perf_counter()
    while len(times) < max_reps:
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
        if len(times) >= min_reps and time.perf_counter() - start > budget_s:
            break
    times.sort()
    return times[len(times) // 2], times[min(len(times) - 1, int(len(times) * 0.95))], len(times)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    print(f"PyTorch {torch.__version__} | threads: {torch.get_num_threads()}")

    print("\nMATMUL")
    print(f"{'n':>6} {'median ms':>12} {'p95 ms':>12} {'GFLOP/s':>10}")
    for n in [64, 128, 256, 512, 1024]:
        a, b = torch.rand(n, n) * 2 - 1, torch.rand(n, n) * 2 - 1
        med, p95, _ = measure(lambda: a @ b)
        print(f"{n:>6} {med:>12.3f} {p95:>12.3f} {2 * n**3 / (med * 1e6):>10.2f}")

    print("\nMODEL (same 784-128-10 shape, random weights; timing doesn't depend on values)")
    model = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(), torch.nn.Linear(128, 10)).eval()
    print(f"{'batch':>6} {'median ms':>12} {'p95 ms':>12} {'images/s':>14}")
    for batch in [1, 8, 64, 256]:
        x = torch.rand(batch, 784)
        med, p95, _ = measure(lambda: model(x))
        print(f"{batch:>6} {med:>12.4f} {p95:>12.4f} {batch / (med / 1000):>14.0f}")


if __name__ == "__main__":
    main()
