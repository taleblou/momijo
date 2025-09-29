# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.tensor.ops
# File: src/momijo/tensor/ops/reduce.mojo

 
 
from momijo.tensor.tensor import Tensor

 
# ---------- list reducers (work now, no Tensor dependency) ----------
fn sum_list_f32(xs: List[Float32]) -> Float32:
    var acc: Float32 = Float32(0.0)
    var i = 0
    var n = len(xs)
    while i < n:
        acc = acc + xs[i]
        i += 1
    return acc

fn mean_list_f32(xs: List[Float32]) -> Float32:
    var n = len(xs)
    if n == 0:
        return Float32(0.0)
    return sum_list_f32(xs) / Float32(n)

# ---------- Tensor entry points (Float32) ----------
# Keep these concrete to avoid generic parsing issues.
# They are safe stubs until your Tensor APIs (shape/strides/element access) are finalized.

alias F32Tensor = Tensor[Float32]

fn sum_f32(x: F32Tensor) -> F32Tensor:
    # TODO: replace with real reduction over Tensor once APIs are stable
    return x

fn mean_f32(x: F32Tensor) -> F32Tensor:
    # TODO: replace with real reduction over Tensor once APIs are stable
    return x