# MiniInfer

A small neural-network inference engine in C++. A 784 → 128 → 10 MLP is trained on MNIST in PyTorch, its weights are exported to a documented binary format, and the forward pass runs on hand-written matrix-multiply kernels. Every kernel is tested against a reference, and every optimization is benchmarked against the previous one and against PyTorch.

**Highlights**

- Matches PyTorch on all 500 exported test images (100% prediction agreement, max logit error 7.6e-6).
- Matmul goes from 0.86 GFLOP/s (naive) to 67–111 GFLOP/s (blocked + OpenMP) on an 8-core laptop CPU.
- At batch size 1, the C++ engine has ~3× lower latency than calling the same model through PyTorch, because it skips framework overhead. From batch 8 up, PyTorch wins.
- Cache blocking, as implemented, was *slower* than simple loop reordering. That result is reported below, not hidden.

## Quick start (Linux / WSL)

```bash
sudo apt install build-essential cmake python3-venv
python3 -m venv ~/mi-env && source ~/mi-env/bin/activate
pip install torch torchvision numpy

python python/train_export.py          # trains the MLP, writes data/model.bin + data/testdata.bin
cmake -B build && cmake --build build -j
./build/test_kernels                   # every kernel vs a double-precision reference
./build/test_model                     # full forward pass vs PyTorch's outputs
OMP_NUM_THREADS=8 ./build/bench        # benchmarks (add --csv for machine-readable output)
python bench/bench_pytorch.py --threads 8
```

## How it works

1. **Train and export.** `python/train_export.py` trains the MLP (96.86% test accuracy), then writes the weights to `model.bin` and an answer key (`testdata.bin`) with 500 test images, PyTorch's hidden-layer activations, and PyTorch's logits.
2. **Load.** The C++ loader reads `model.bin` and transposes each weight matrix from PyTorch's (out, in) layout to (in, out), so every layer is a plain matmul: `Y = X · Wᵀ + b`.
3. **Forward pass.** Dense → bias → ReLU → dense → bias. Activation buffers are allocated once in `prepare()` and reused, so inference does no memory allocation.
4. **Kernels.** Four interchangeable matmul implementations, selected at runtime:

| Kernel | Idea |
| --- | --- |
| `naive` | Textbook i → j → k triple loop. The inner loop strides through B by a whole row per step. |
| `reordered` | Same math, i → k → j order. The inner loop reads B and writes C sequentially. |
| `blocked` | Reordered loops inside 64 × 64 tiles, intended to keep reused data in cache. |
| `parallel` | Blocked, with OpenMP splitting the row tiles across cores. Safe because each tile writes distinct rows of C. |

## Correctness

- `test_kernels` compares every kernel against a double-precision reference on 7 shapes, including awkward ones (37×101×53, 65×63×129) that catch tile-edge bugs. Output buffers are pre-filled with junk to catch kernels that forget to zero them. Tolerance scales with √K, since float32 summation error grows with the number of terms.
- `test_model` runs all 500 exported images through each kernel at batch sizes 1, 7, 64 and 256, and checks layer-1 activations, final logits and predicted digits against PyTorch.

All four kernels pass at every batch size. Measured errors for the naive kernel:

| Check | Result |
| --- | --- |
| Prediction agreement with PyTorch | 500 / 500 |
| Max layer-1 activation error | 6.7e-6 |
| Max logit error | 7.6e-6 (tolerance 1e-4) |
| Accuracy on these 500 images | 98.40% (the first 500 test images are slightly easier than the full set) |

Outputs don't match PyTorch bit-for-bit because floating-point addition isn't associative: different loop orders round differently.

## Benchmarks

**Setup:** Intel Core Ultra 9 288V (8 cores, no SMT), Ubuntu on WSL2, g++ 13 with `-O3 -march=native`, `OMP_NUM_THREADS=8`. PyTorch 2.14.1 with `torch.set_num_threads(8)`.

**Method:** warm-up runs, then repeated timing (5–100 reps within a ~1 s budget per configuration), reporting the median. The full suite was run twice; ranges below show both runs. PyTorch was run once. Raw output is in `results.txt`.

### Matmul throughput, GFLOP/s (higher is better)

C[n × n] = A[n × n] · B[n × n], counted as 2n³ operations.

| n | naive | reordered | blocked | parallel | PyTorch |
| --- | --- | --- | --- | --- | --- |
| 64 | 7.7–7.8 | 35.7–36.0 | 29.9–30.0 | 26.0–28.3 | n/a* |
| 128 | 6.4–6.6 | 30.4–31.0 | 24.5–24.7 | 1.2–1.3 | 331 |
| 256 | 3.5–3.7 | 22.9–24.5 | 17.1–18.5 | 9.5–11.6 | 495 |
| 512 | 2.6–2.7 | 24.6–25.7 | 16.5–18.3 | 67–72 | 498 |
| 1024 | 0.84–0.86 | 19.3–22.7 | 14.9–16.0 | 73–111 | 446 |

\*PyTorch's n=64 result (0.16 GFLOP/s) looks like a thread-pool start-up artifact and is excluded.

### End-to-end model throughput, images/s (higher is better)

| Batch | naive | reordered | blocked | parallel | PyTorch |
| --- | --- | --- | --- | --- | --- |
| 1 | 7.5K–15K | 123K–144K | 90K–103K | 0.5K–23K | 48K |
| 8 | 7.4K–16K | 108K–135K | 102K–113K | 103K–106K | 303K |
| 64 | 8.4K–15K | 107K–131K | 103K–116K | 107K | 1.05M |
| 256 | 11K–15K | 107K–128K | 103K–113K | 380K–389K | 1.33M |

## Findings

**1. Memory access order matters more than arithmetic.** Swapping two loops (naive → reordered) made matmul 22–27× faster at n = 1024 with identical math. The naive inner loop reads B down a column, jumping 4 KB per step, so nearly every read misses cache. The reordered inner loop reads consecutive floats, using all 16 floats in each 64-byte cache line, and is simple enough for the compiler to vectorize.

**2. The naive kernel gets worse as matrices grow.** Its throughput drops from 7.8 GFLOP/s at n = 64 to 0.86 at n = 1024. Small matrices fit in cache, so the bad access pattern is hidden; large ones don't.

**3. Cache blocking was a regression here.** Blocked was 20–45% slower than reordered at every size, in every run. Likely causes: the reordered loop already streams memory sequentially, which the hardware prefetcher handles well, and 64-wide tiles shorten the inner loop and add loop overhead. The tile size was not tuned.

**4. Threads help only when there is enough work.** Parallel reached 67–111 GFLOP/s at n ≥ 512, but was slower than single-threaded below that: at n = 128 it ran ~25× slower than reordered. Starting and synchronizing 8 threads costs milliseconds of overhead, and parallelism is over row tiles, so a 128-row matrix has only 2 tiles for 8 threads. In the model, this makes the parallel kernel the best choice at batch 256 (3× reordered) and the worst at batch 1.

**5. Parallel results are noisy.** Parallel medians varied by up to ~50% between runs, and p95 latencies were often several times the median. Thread scheduling under WSL contributes, and the CPU combines performance and efficiency cores, so an even split of work across 8 threads finishes at the speed of the slowest cores.

**6. Framework overhead dominates small inference.** At batch 1, MiniInfer's reordered kernel processed 123K–144K images/s versus PyTorch's 48K, because each PyTorch call pays fixed Python and dispatcher overhead. This is a comparison of call overhead, not of kernel speed: from batch 8 up, PyTorch's vendor-tuned kernels win, reaching ~4–7× MiniInfer's best matmul throughput.

## Limitations and next steps

- **Kernel selection by size.** The data supports a simple rule: reordered for small batches, parallel for large ones.
- **Tune or rethink blocking.** Sweep tile sizes, try parallelizing over column tiles as well as row tiles, and pack B into contiguous tiles.
- **Explicit SIMD.** Write the inner loop with AVX intrinsics instead of relying on auto-vectorization.
- **Load-aware scheduling.** Try OpenMP dynamic scheduling to handle cores of different speeds.
- **Phase 2:** a small transformer (attention, layer norm, KV cache) on the same runtime, then a CUDA backend.

## Model file format (`model.bin`)

All values little-endian. Matrices are row-major.

| Field | Type | Notes |
| --- | --- | --- |
| magic | uint32 | `0x464E494D` ("MINF") |
| version | uint32 | `1` |
| num_layers | uint32 | |
| per layer: out_dim, in_dim | int32, int32 | |
| per layer: weight | float32 × out·in | shape (out, in), as PyTorch stores it |
| per layer: bias | float32 × out | |

## Layout

| Path | What it is |
| --- | --- |
| `python/train_export.py` | Trains the MLP, exports weights and the answer key |
| `include/miniinfer/`, `src/` | Kernels, model loader and forward pass |
| `tests/` | Kernel and end-to-end correctness tests |
| `bench/` | C++ benchmark and the matching PyTorch baseline |
| `results.txt` | Raw benchmark output used in this README |
