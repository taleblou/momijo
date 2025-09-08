# Project:      Momijo
# Module:       src.momijo.kernels.reference.elemwise_ref
# File:         elemwise_ref.mojo
# Path:         src/momijo/kernels/reference/elemwise_ref.mojo
#
# Description:  src.momijo.kernels.reference.elemwise_ref — focused Momijo functionality with a stable public API.
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
#   - Key functions: elemwise_add_ref, elemwise_mul_ref, elemwise_div_ref, elemwise_relu_ref, elemwise_sigmoid_ref, _self_test, main
#   - Error paths explicitly marked with 'raises'.


from math import exp
from momijo.core.error import Error
from momijo.tensor.tensor import Tensor

fn elemwise_add_ref(A: Tensor, B: Tensor, mut C: Tensor) raises -> Error:
    assert A.numel() == B.numel() and B.numel() == C.numel(), "Size mismatch in elemwise_add_ref"
    var n = A.numel()
    for i in range(n):
        C.set_item(i, A.get_item(i) + B.get_item(i))
    return Error.ok()

# Elementwise multiply: C = A * B
fn elemwise_mul_ref(A: Tensor, B: Tensor, mut C: Tensor) raises -> Error:
    assert A.numel() == B.numel() and B.numel() == C.numel(), "Size mismatch in elemwise_mul_ref"
    var n = A.numel()
    for i in range(n):
        C.set_item(i, A.get_item(i) * B.get_item(i))
    return Error.ok()

# Elementwise divide: C = A / B
fn elemwise_div_ref(A: Tensor, B: Tensor, mut C: Tensor) raises -> Error:
    assert A.numel() == B.numel() and B.numel() == C.numel(), "Size mismatch in elemwise_div_ref"
    var n = A.numel()
    for i in range(n):
        var denom = B.get_item(i)
        C.set_item(i, A.get_item(i) / denom)
    return Error.ok()

# Elementwise ReLU: C = max(0, A)
fn elemwise_relu_ref(A: Tensor, mut C: Tensor) raises -> Error:
    assert A.numel() == C.numel(), "Size mismatch in elemwise_relu_ref"
    var n = A.numel()
    for i in range(n):
        var val = A.get_item(i)
        if val > 0.0:
            C.set_item(i, val)
        else:
            C.set_item(i, 0.0)
    return Error.ok()

# Elementwise Sigmoid: C = 1 / (1 + exp(-A))
fn elemwise_sigmoid_ref(A: Tensor, mut C: Tensor) raises -> Error:
    assert A.numel() == C.numel(), "Size mismatch in elemwise_sigmoid_ref"
    var n = A.numel()
    for i in range(n):
        var val = A.get_item(i)
        C.set_item(i, 1.0 / (1.0 + exp(-val)))
    return Error.ok()

# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True

    var A = Tensor([1.0, -2.0, 3.0], shape=[3])
    var B = Tensor([4.0, 5.0, 6.0], shape=[3])
    var C = Tensor([0.0, 0.0, 0.0], shape=[3])

    try:
        var err1 = elemwise_add_ref(A, B, C)
    except e:
        return False
    if C.get_item(0) != 5.0 or C.get_item(2) != 9.0:
        ok = False

    try:
        var err2 = elemwise_mul_ref(A, B, C)
    except e:
        return False
    if C.get_item(0) != 4.0 or C.get_item(2) != 18.0:
        ok = False

    try:
        var err3 = elemwise_relu_ref(A, C)
    except e:
        return False
    if C.get_item(1) != 0.0:
        ok = False

    try:
        var err4 = elemwise_sigmoid_ref(A, C)
    except e:
        return False

    if abs(C.get_item(0) - 0.731) > 1e-2:
        ok = False

    return ok
fn main() -> None:
    var ok = _self_test()
    if ok:
        print("kernels/reference/elemwise_ref.mojo self-test: OK")
    else:
        print("kernels/reference/elemwise_ref.mojo self-test: FAILED")