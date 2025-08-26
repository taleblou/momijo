# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Minimal ufunc-like elementwise operations for Tensor[Float64]

from momijo.tensor.tensor import Tensor

fn add(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    assert(a.shape == b.shape, "Shapes must match for elementwise add")
    var out = Tensor[Float64](shape=a.shape)
    let n = a.size()
    for i in range(0, n):
        out.data[i] = a.data[a.offset + i] + b.data[b.offset + i]
    return out

fn mul(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    assert(a.shape == b.shape, "Shapes must match for elementwise mul")
    var out = Tensor[Float64](shape=a.shape)
    let n = a.size()
    for i in range(0, n):
        out.data[i] = a.data[a.offset + i] * b.data[b.offset + i]
    return out

fn scalar_add(a: Tensor[Float64], c: Float64) -> Tensor[Float64]:
    var out = Tensor[Float64](shape=a.shape)
    let n = a.size()
    for i in range(0, n):
        out.data[i] = a.data[a.offset + i] + c
    return out
