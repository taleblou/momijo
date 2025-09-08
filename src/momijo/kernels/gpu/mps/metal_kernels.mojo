# Project:      Momijo
# Module:       src.momijo.kernels.gpu.mps.metal_kernels
# File:         metal_kernels.mojo
# Path:         src/momijo/kernels/gpu/mps/metal_kernels.mojo
#
# Description:  src.momijo.kernels.gpu.mps.metal_kernels — focused Momijo functionality with a stable public API.
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
#   - Key functions: mps_elemwise_add, mps_elemwise_mul, mps_relu, _self_test
#   - Error paths explicitly marked with 'raises'.
#   - GPU/device utilities present; validate backend assumptions.


from momijo.core.device import Device
from momijo.core.error import Error
from momijo.tensor.tensor import Tensor

fn mps_elemwise_add(alpha: Float64, A: Tensor, beta: Float64, B: Tensor, mut C: Tensor, device: Device) raises -> Error:
    assert A.numel() == B.numel() and B.numel() == C.numel(), "Size mismatch in mps_elemwise_add"
    var n = A.numel()

    if device.is_cpu():
        for i in range(n):
            var val = alpha * A.get_item(i) + beta * B.get_item(i)
            C.set_item(i, val)
    else:
        # TODO: Replace with real Metal kernel dispatch
        # encode_add(alpha, A, beta, B, C, device)  # pseudo-call
        for i in range(n):  # fallback
            var val = alpha * A.get_item(i) + beta * B.get_item(i)
            C.set_item(i, val)

    return Error.ok()

# Elementwise multiply: C = A * B
fn mps_elemwise_mul(A: Tensor, B: Tensor, mut C: Tensor, device: Device) raises -> Error:
    assert A.numel() == B.numel() and B.numel() == C.numel(), "Size mismatch in mps_elemwise_mul"
    var n = A.numel()

    if device.is_cpu():
        for i in range(n):
            C.set_item(i, A.get_item(i) * B.get_item(i))
    else:
        # TODO: Replace with real Metal kernel dispatch
        for i in range(n):  # fallback
            C.set_item(i, A.get_item(i) * B.get_item(i))

    return Error.ok()

# Elementwise ReLU: C = max(0, A)
fn mps_relu(A: Tensor, mut C: Tensor, device: Device) raises -> Error:
    assert A.numel() == C.numel(), "Size mismatch in mps_relu"
    var n = A.numel()

    if device.is_cpu():
        for i in range(n):
            var v = A.get_item(i)
            if v > 0.0:
                C.set_item(i, v)
            else:
                C.set_item(i, 0.0)
    else:
        # TODO: Replace with real Metal kernel dispatch
        for i in range(n):  # fallback
            var v = A.get_item(i)
            if v > 0.0:
                C.set_item(i, v)
            else:
                C.set_item(i, 0.0)

    return Error.ok()

# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True
    var dev = Device("cpu")

    var A = Tensor([1.0, -2.0, 3.0, 4.0], shape=[4])
    var B = Tensor([4.0, 5.0, 6.0, 7.0], shape=[4])
    var C = Tensor([0.0, 0.0, 0.0, 0.0], shape=[4])

    # add
    try:
        var err1 = mps_elemwise_add(1.0, A, 1.0, B, C, dev)
    except e:
        return False
    if C.get_item(0) != 5.0 or C.get_item(3) != 11.0:
        ok = False

    # mul
    try:
        var err2 = mps_elemwise_mul(A, B, C, dev)
    except e:
        return False
    if C.get_item(0) != 4.0 or C.get_item(2) != 18.0:
        ok = False

    # relu
    try:
        var err3 = mps_relu(A, C, dev)
    except e:
        return False
    if C.get_item(1) != 0.0 or C.get_item(0) != 1.0:
        ok = False

    return ok