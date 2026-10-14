#pragma once
#include <string>
#include <vector>

#include "miniinfer/kernels.h"

namespace miniinfer {

// One dense layer: y = x * W^T + b
struct DenseLayer {
    int in_dim = 0;
    int out_dim = 0;
    std::vector<float> weight_t;  // stored TRANSPOSED: shape (in_dim, out_dim), row-major
    std::vector<float> bias;      // shape (out_dim)
};

class Model {
public:
    // Reads model.bin (format in README.md). Throws std::runtime_error on a bad file.
    void load(const std::string& path);

    // Allocate activation buffers for batches up to max_batch. Call once after load().
    void prepare(int max_batch);

    // Forward pass on `batch` images.
    //   input:  batch x in_dim floats (row-major)
    //   returns a pointer to batch x out_dim logits, valid until the next forward().
    // Layers: dense -> ReLU -> dense -> ... (no ReLU after the last layer).
    const float* forward(const float* input, int batch);

    // Output of hidden layer `i` (after ReLU) from the most recent forward(). For debugging/tests.
    const float* activation(int i) const { return acts_[i].data(); }

    int input_dim() const { return layers_.front().in_dim; }
    int output_dim() const { return layers_.back().out_dim; }
    int num_layers() const { return static_cast<int>(layers_.size()); }

    Kernel kernel = Kernel::Naive;  // which matmul backend forward() uses

private:
    std::vector<DenseLayer> layers_;
    std::vector<std::vector<float>> acts_;  // one reusable output buffer per layer
    int max_batch_ = 0;
};

// Answer key exported by python/train_export.py.
struct TestData {
    int n = 0, in_dim = 0, hidden = 0, out_dim = 0;
    std::vector<float> images;  // n x in_dim
    std::vector<float> hidden_acts;  // n x hidden (layer 1 after ReLU)
    std::vector<float> logits;  // n x out_dim
    std::vector<int> labels;    // n
};
TestData load_testdata(const std::string& path);

}  // namespace miniinfer
