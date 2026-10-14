"""Train a small MLP on MNIST in PyTorch and export it for MiniInfer.

Writes two files into ../data/:
  model.bin     - the trained weights (format documented in README.md)
  testdata.bin  - test images plus PyTorch's own outputs, used as the answer key

Run from the repo root:  python python/train_export.py
"""
import os
import struct

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

MODEL_MAGIC = 0x464E494D  # bytes "MINF" in little-endian
TEST_MAGIC = 0x5453494D   # bytes "MIST"
VERSION = 1

HIDDEN = 128
EPOCHS = 3
N_TEST_EXPORT = 500  # how many test images go into the answer key

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, HIDDEN)  # weight shape (128, 784) = (out, in)
        self.fc2 = nn.Linear(HIDDEN, 10)   # weight shape (10, 128)

    def forward(self, x):
        x = x.view(x.size(0), -1)          # (B, 1, 28, 28) -> (B, 784)
        h = torch.relu(self.fc1(x))
        return self.fc2(h)                 # logits, no softmax (argmax is the prediction)


def load_mnist():
    # ToTensor scales pixels to [0, 1]. No normalization, to keep the C++ side simple.
    tf = transforms.ToTensor()
    train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tf)
    return train, test


def train(model, train_set):
    loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{EPOCHS}  avg loss {total / len(loader):.4f}")


@torch.no_grad()
def accuracy(model, test_set):
    loader = torch.utils.data.DataLoader(test_set, batch_size=1000)
    model.eval()
    correct = 0
    for x, y in loader:
        correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(test_set)


def f32_bytes(t):
    """Tensor -> raw float32 bytes in row-major (C) order."""
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype=np.float32).tobytes()


def export_model(model, path):
    layers = [model.fc1, model.fc2]
    with open(path, "wb") as f:
        f.write(struct.pack("<III", MODEL_MAGIC, VERSION, len(layers)))
        for layer in layers:
            out_dim, in_dim = layer.weight.shape
            f.write(struct.pack("<ii", out_dim, in_dim))
            f.write(f32_bytes(layer.weight))  # out*in floats, row-major (out, in)
            f.write(f32_bytes(layer.bias))    # out floats
    print(f"wrote {path}")


@torch.no_grad()
def export_testdata(model, test_set, path):
    model.eval()
    x = torch.stack([test_set[i][0] for i in range(N_TEST_EXPORT)]).view(N_TEST_EXPORT, -1)
    labels = torch.tensor([test_set[i][1] for i in range(N_TEST_EXPORT)], dtype=torch.int32)
    hidden = torch.relu(model.fc1(x))  # layer-1 output after ReLU, for debugging
    logits = model.fc2(hidden)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIiiii", TEST_MAGIC, VERSION, N_TEST_EXPORT, 784, HIDDEN, 10))
        f.write(f32_bytes(x))
        f.write(f32_bytes(hidden))
        f.write(f32_bytes(logits))
        f.write(labels.numpy().astype("<i4").tobytes())
    print(f"wrote {path}")


def main():
    torch.manual_seed(0)
    os.makedirs(DATA_DIR, exist_ok=True)
    train_set, test_set = load_mnist()
    model = MLP()
    train(model, train_set)
    print(f"PyTorch test accuracy: {accuracy(model, test_set) * 100:.2f}%")
    export_model(model, os.path.join(DATA_DIR, "model.bin"))
    export_testdata(model, test_set, os.path.join(DATA_DIR, "testdata.bin"))


if __name__ == "__main__":
    main()
