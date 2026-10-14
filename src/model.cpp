#include "miniinfer/model.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace miniinfer {

namespace {

constexpr uint32_t kModelMagic = 0x464E494D;  // "MINF"
constexpr uint32_t kTestMagic = 0x5453494D;   // "MIST"
constexpr uint32_t kVersion = 1;

template <typename T>
T read_scalar(std::ifstream& f, const std::string& path) {
    T v{};
    f.read(reinterpret_cast<char*>(&v), sizeof(T));
    if (!f) throw std::runtime_error("unexpected end of file: " + path);
    return v;
}

void read_floats(std::ifstream& f, std::vector<float>& out, size_t count, const std::string& path) {
    out.resize(count);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) throw std::runtime_error("unexpected end of file: " + path);
}

}  // namespace

// ---------------------------------------------------------------------------
// Provided: loading
// ---------------------------------------------------------------------------

void Model::load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    if (read_scalar<uint32_t>(f, path) != kModelMagic) throw std::runtime_error("not a MiniInfer model: " + path);
    if (read_scalar<uint32_t>(f, path) != kVersion) throw std::runtime_error("unsupported model version: " + path);
    const uint32_t n_layers = read_scalar<uint32_t>(f, path);

    layers_.clear();
    for (uint32_t l = 0; l < n_layers; ++l) {
        DenseLayer layer;
        layer.out_dim = read_scalar<int32_t>(f, path);
        layer.in_dim = read_scalar<int32_t>(f, path);
        if (!layers_.empty() && layers_.back().out_dim != layer.in_dim)
            throw std::runtime_error("layer shapes don't chain in " + path);

        // File stores W as (out, in) like PyTorch. Transpose to (in, out) so a layer is
        // a plain matmul:  Y[B x out] = X[B x in] * Wt[in x out].
        std::vector<float> w;
        read_floats(f, w, static_cast<size_t>(layer.out_dim) * layer.in_dim, path);
        layer.weight_t.resize(w.size());
        for (int o = 0; o < layer.out_dim; ++o)
            for (int i = 0; i < layer.in_dim; ++i)
                layer.weight_t[i * layer.out_dim + o] = w[o * layer.in_dim + i];

        read_floats(f, layer.bias, layer.out_dim, path);
        layers_.push_back(std::move(layer));
    }
    if (layers_.empty()) throw std::runtime_error("model has no layers: " + path);
}

void Model::prepare(int max_batch) {
    max_batch_ = max_batch;
    acts_.clear();
    for (const auto& layer : layers_)
        acts_.emplace_back(static_cast<size_t>(max_batch) * layer.out_dim);
}

// ---------------------------------------------------------------------------
// YOUR CODE: the forward pass. Guide: section 2.
// ---------------------------------------------------------------------------

const float* Model::forward(const float* input, int batch) {
    if (batch > max_batch_) throw std::runtime_error("batch larger than prepare() size");

    // TODO: for each layer l:
    //   1. in  = (l == 0) ? input : acts_[l - 1].data()
    //   2. out = acts_[l].data()
    //   3. matmul(kernel, in, layer.weight_t.data(), out, batch, layer.out_dim, layer.in_dim)
    //   4. add_bias_rows(out, layer.bias.data(), batch, layer.out_dim)
    //   5. relu_inplace on out, EXCEPT for the last layer (those are the logits)
    // Return acts_.back().data().
    // Notice: no memory is allocated here. That's what prepare() is for.
        for (size_t l = 0; l < layers_.size(); l++) {
        const DenseLayer& layer = layers_[l];
        const float* in = (l == 0) ? input : acts_[l - 1].data();
        float* out = acts_[l].data();

        matmul(kernel, in, layer.weight_t.data(), out, batch, layer.out_dim, layer.in_dim);
        add_bias_rows(out, layer.bias.data(), batch, layer.out_dim);

        bool is_last = (l + 1 == layers_.size());
        if (!is_last) {
            relu_inplace(out, batch * layer.out_dim);
        }
    }
    return acts_.back().data();
}

// ---------------------------------------------------------------------------
// Provided: answer-key loader
// ---------------------------------------------------------------------------

TestData load_testdata(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    if (read_scalar<uint32_t>(f, path) != kTestMagic) throw std::runtime_error("not a test-data file: " + path);
    if (read_scalar<uint32_t>(f, path) != kVersion) throw std::runtime_error("unsupported test-data version: " + path);

    TestData t;
    t.n = read_scalar<int32_t>(f, path);
    t.in_dim = read_scalar<int32_t>(f, path);
    t.hidden = read_scalar<int32_t>(f, path);
    t.out_dim = read_scalar<int32_t>(f, path);
    read_floats(f, t.images, static_cast<size_t>(t.n) * t.in_dim, path);
    read_floats(f, t.hidden_acts, static_cast<size_t>(t.n) * t.hidden, path);
    read_floats(f, t.logits, static_cast<size_t>(t.n) * t.out_dim, path);
    t.labels.resize(t.n);
    f.read(reinterpret_cast<char*>(t.labels.data()), static_cast<std::streamsize>(t.n * sizeof(int32_t)));
    if (!f) throw std::runtime_error("unexpected end of file: " + path);
    return t;
}

}  // namespace miniinfer
