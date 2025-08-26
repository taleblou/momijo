# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.


# Reductions: sum/mean for 1D tensors (demo)

from momijo.tensor.tensor import Tensor

fn sum1d(a: Tensor[Float64]) -> Float64:
    assert(len(a.shape) == 1, "sum1d expects rank-1 tensor")
    var s: Float64 = 0.0
    let n = a.size()
    for i in range(0, n):
        s += a.data[a.offset + i]
    return s

fn mean1d(a: Tensor[Float64]) -> Float64:
    return sum1d(a) / Float64(a.size())
