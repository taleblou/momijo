# Project:      Momijo
# Module:       src.momijo.tensor.conversion
# File:         conversion.mojo
# Path:         src/momijo/tensor/conversion.mojo
#
# Description:  Core tensor/ndarray components: shapes/strides, broadcasting rules,
#               element-wise ops, and foundational kernels.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Key functions: __module_name__, __self_test__, argmax_index, argmin_index, astype


from momijo.tensor.tensor import Tensor

fn __module_name__() -> String:
    return String("momijo/tensor/conversion.mojo")
fn __self_test__() -> Bool:
    # Extend with real checks later
    return True

# Lightweight helpers so tests can import them
fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]
            idx = i
        i += 1
    return idx
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]
            idx = i
        i += 1
    return idx

# Public API (stub): keep signature simple so it compiles today.
# Later you can generalize to support multiple dtypes and real casting.
fn astype(
    x: Tensor[Float32],
    dtype_code: Int  # placeholder (e.g., 0=f32,1=f64,2=i32,...) — evolve to your DType later
) -> Tensor[Float32]:
    # TODO: when Tensor exposes data+dtype, return a new Tensor with casted data.
    # For now, pass-through so tests that only import this symbol compile and run.
    return x