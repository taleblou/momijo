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
# Project: momijo.kernels.common
# File: src/momijo/kernels/common/numerics.mojo

from math import exp

struct NumericConsts:
    const EPSILON: Float64 = 1e-12
    const PI: Float64 = 3.141592653589793
    const E: Float64 = 2.718281828459045
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# Clamp a value to [lo, hi]
fn clamp[T: Comparable & Copyable](x: T, lo: T, hi: T) -> T:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

# Safe division with epsilon to avoid NaN/Inf
fn safe_div(x: Float64, y: Float64, eps: Float64 = NumericConsts.EPSILON) -> Float64:
    if abs(y) < eps:
        return x / (y + eps)
    return x / y

# Sigmoid function
fn sigmoid(x: Float64) -> Float64:
    return 1.0 / (1.0 + exp(-x))

# ReLU function
fn relu(x: Float64) -> Float64:
    if x > 0.0:
        return x
    return 0.0

# Tanh function
fn tanh_fn(x: Float64) -> Float64:
    var e1 = exp(x)
    var e2 = exp(-x)
    return (e1 - e2) / (e1 + e2)

# Numerically stable softmax for a list of floats
fn softmax(xs: List[Float64]) -> List[Float64]:
    if len(xs) == 0:
        return []
    var max_val = xs[0]
    for v in xs:
        if v > max_val:
            max_val = v
    var exps = List[Float64]()
    var sum_val: Float64 = 0.0
    for v in xs:
        var e = exp(v - max_val)
        exps.append(e)
        sum_val += e
    var out = List[Float64]()
    for e in exps:
        out.append(e / sum_val)
    return out

# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True
    var x = 2.0
    var y = safe_div(1.0, 0.0)
    if sigmoid(0.0) <= 0.0:
        ok = False
    if relu(-1.0) != 0.0:
        ok = False
    var s = softmax([1.0, 2.0, 3.0])
    if len(s) != 3:
        ok = False
    return ok