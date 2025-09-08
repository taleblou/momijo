# Project:      Momijo
# Module:       src.momijo.kernels.gpu.cuda.cuda_kernels
# File:         cuda_kernels.mojo
# Path:         src/momijo/kernels/gpu/cuda/cuda_kernels.mojo
#
# Description:  src.momijo.kernels.gpu.cuda.cuda_kernels — focused Momijo functionality with a stable public API.
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
#   - Key functions: cuda_elemwise_add, cuda_elemwise_mul, _self_test
#   - Error paths explicitly marked with 'raises'.
#   - GPU/device utilities present; validate backend assumptions.


from momijo.core.device import Device
from momijo.core.error import Error
from momijo.tensor.tensor import Tensor

fn cuda_elemwise_add(alpha: Float64, A: Tensor, beta: Float64, B: Tensor, mut C: Tensor, device: Device) raises -> Error:
    assert A.numel() == B.numel() and B.numel() == C.numel(), "Size mismatch in cuda_elemwise_add"
    var n = A.numel()
    if device.is_cpu():
        for i in range(n):
            var val = alpha * A.get_item(i) + beta * B.get_item(i)
            C.set_item(i, val)
    else:
        # Here would be a real CUDA kernel launch
        # For now, fallback to CPU implementation
        for i in range(n):
            var val = alpha * A.get_item(i) + beta * B.get_item(i)
            C.set_item(i, val)
    return Error.ok()

# Placeholder for CUDA elementwise multiply kernel (with CPU fallback)
fn cuda_elemwise_mul(A: Tensor, B: Tensor, mut C: Tensor, device: Device) raises -> Error:
    assert A.numel() == B.numel() and B.numel() == C.numel(), "Size mismatch in cuda_elemwise_mul"
    var n = A.numel()
    if device.is_cpu():
        for i in range(n):
            var val = A.get_item(i) * B.get_item(i)
            C.set_item(i, val)
    else:
        # Here would be a real CUDA kernel launch
        # For now, fallback to CPU implementation
        for i in range(n):
            var val = A.get_item(i) * B.get_item(i)
            C.set_item(i, val)
    return Error.ok()

# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True
    var dev = Device("cpu")
    var A = Tensor([1.0, 2.0, 3.0], shape=[3])
    var B = Tensor([4.0, 5.0, 6.0], shape=[3])
    var C = Tensor([0.0, 0.0, 0.0], shape=[3])
    try:
        var err1 = cuda_elemwise_add(1.0, A, 1.0, B, C, dev)
    except e:
        return False
    if C.get_item(0) != 5.0 or C.get_item(2) != 9.0:
        ok = False

    try:
        var err2 = cuda_elemwise_mul(A, B, C, dev)
    except e:
        return False
    if C.get_item(0) != 4.0 or C.get_item(2) != 18.0:
        ok = False

    return ok