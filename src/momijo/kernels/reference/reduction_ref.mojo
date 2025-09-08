# Project:      Momijo
# Module:       src.momijo.kernels.reference.reduction_ref
# File:         reduction_ref.mojo
# Path:         src/momijo/kernels/reference/reduction_ref.mojo
#
# Description:  src.momijo.kernels.reference.reduction_ref — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Key functions: sum_ref, max_ref, min_ref, dot_ref, _self_test


from momijo.tensor.tensor import Tensor

fn sum_ref(A: Tensor) -> Float64:
    var n = A.numel()
    var acc: Float64 = 0.0
    for i in range(n):
        acc += A.get_item(i)
    return acc

# Reference max reduction
fn max_ref(A: Tensor) -> Float64:
    assert A.numel() > 0, "Empty tensor in max_ref"
    var n = A.numel()
    var m = A.get_item(0)
    for i in range(1, n):
        var v = A.get_item(i)
        if v > m:
            m = v
    return m

# Reference min reduction
fn min_ref(A: Tensor) -> Float64:
    assert A.numel() > 0, "Empty tensor in min_ref"
    var n = A.numel()
    var m = A.get_item(0)
    for i in range(1, n):
        var v = A.get_item(i)
        if v < m:
            m = v
    return m

# Reference dot product
fn dot_ref(A: Tensor, B: Tensor) -> Float64:
    assert A.numel() == B.numel(), "Size mismatch in dot_ref"
    var n = A.numel()
    var acc: Float64 = 0.0
    for i in range(n):
        acc += A.get_item(i) * B.get_item(i)
    return acc

# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True

    var A = Tensor([1.0, 2.0, 3.0, 4.0], shape=[4])
    var B = Tensor([2.0, 2.0, 2.0, 2.0], shape=[4])

    if sum_ref(A) != 10.0:
        ok = False
    if max_ref(A) != 4.0:
        ok = False
    if min_ref(A) != 1.0:
        ok = False
    if dot_ref(A, B) != 20.0:
        ok = False

    return ok