// End-to-end check: your forward pass vs PyTorch's outputs on the exported test images.
// Usage: ./build/test_model [data_dir]   (default: data)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "miniinfer/model.h"

using namespace miniinfer;

namespace {

constexpr double kTol = 1e-4;  // max allowed |yours - PyTorch| for any logit or activation

int argmax(const float* v, int n) { return static_cast<int>(std::max_element(v, v + n) - v); }

double max_abs_diff(const float* a, const float* b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, double(std::fabs(a[i] - b[i])));
    return m;
}

// Runs the whole test set through the model in chunks of `batch`. Returns true on pass.
bool check(Model& model, const TestData& t, Kernel kern, int batch) {
    model.kernel = kern;
    double hidden_err = 0, logit_err = 0;
    int agree = 0, correct = 0;

    for (int start = 0; start < t.n; start += batch) {
        const int b = std::min(batch, t.n - start);
        const float* logits = model.forward(&t.images[size_t(start) * t.in_dim], b);
        hidden_err = std::max(hidden_err,
                              max_abs_diff(model.activation(0), &t.hidden_acts[size_t(start) * t.hidden],
                                           size_t(b) * t.hidden));
        logit_err = std::max(logit_err,
                             max_abs_diff(logits, &t.logits[size_t(start) * t.out_dim], size_t(b) * t.out_dim));
        for (int i = 0; i < b; ++i) {
            const int mine = argmax(logits + i * t.out_dim, t.out_dim);
            agree += mine == argmax(&t.logits[size_t(start + i) * t.out_dim], t.out_dim);
            correct += mine == t.labels[start + i];
        }
    }

    const bool ok = hidden_err <= kTol && logit_err <= kTol && agree == t.n;
    std::printf("  [%s] %-9s batch %-4d layer1 err %.2e | logit err %.2e | agrees %d/%d | accuracy %.2f%%\n",
                ok ? "PASS" : "FAIL", kernel_name(kern), batch, hidden_err, logit_err, agree, t.n,
                100.0 * correct / t.n);
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "data";
    Model model;
    TestData t;
    try {
        model.load(dir + "/model.bin");
        t = load_testdata(dir + "/testdata.bin");
    } catch (const std::exception& e) {
        std::printf("setup failed: %s\nRun `python python/train_export.py` first.\n", e.what());
        return 2;
    }
    std::printf("loaded %d layers, %d test images (tolerance %.0e)\n", model.num_layers(), t.n, kTol);

    const int batches[] = {1, 7, 64, 256};
    model.prepare(256);
    int failures = 0;
    for (Kernel k : {Kernel::Naive, Kernel::Reordered, Kernel::Blocked, Kernel::Parallel}) {
        for (int b : batches) {
            try {
                failures += !check(model, t, k, b);
            } catch (const std::logic_error& e) {
                std::printf("  [FAIL] %-9s batch %-4d %s\n", kernel_name(k), b, e.what());
                ++failures;
                break;  // no point running other batch sizes for an unimplemented kernel
            }
        }
    }
    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "ALL PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
