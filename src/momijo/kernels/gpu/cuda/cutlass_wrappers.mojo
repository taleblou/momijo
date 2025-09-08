# Project:      Momijo
# Module:       src.momijo.kernels.gpu.cuda.cutlass_wrappers
# File:         cutlass_wrappers.mojo
# Path:         src/momijo/kernels/gpu/cuda/cutlass_wrappers.mojo
#
# Description:  src.momijo.kernels.gpu.cuda.cutlass_wrappers — focused Momijo functionality with a stable public API.
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
#   - Key functions: cutlass_gemm, cutlass_conv2d, _self_test
#   - Error paths explicitly marked with 'raises'.
#   - GPU/device utilities present; validate backend assumptions.


from momijo.core.device import Device
from momijo.core.error import Error
from momijo.tensor.tensor import Tensor

fn cutlass_gemm(alpha: Float64, A: Tensor, B: Tensor, beta: Float64, mut C: Tensor, device: Device) raises -> Error:
    assert A.shape().rank == 2 and B.shape().rank == 2 and C.shape().rank == 2, "Matrices must be 2D"
    var m = A.shape()[0]
    var n = B.shape()[1]
    var k = A.shape()[1]
    assert B.shape()[0] == k, "Inner dimension mismatch"
    assert C.shape()[0] == m and C.shape()[1] == n, "Output shape mismatch"

    if device.is_cpu():
        for i in range(m):
            for j in range(n):
                var sum_val: Float64 = 0.0
                for p in range(k):
                    var a_val = A.get_item(i * k + p)
                    var b_val = B.get_item(p * n + j)
                    sum_val += a_val * b_val
                var c_val = C.get_item(i * n + j)
                var new_val = alpha * sum_val + beta * c_val
                C.set_item(i * n + j, new_val)
    else:
        # Real implementation would call CUTLASS GEMM API
        # For now, fallback to CPU path
        return cutlass_gemm(alpha, A, B, beta, C, Device("cpu"))

    return Error.ok()

# Placeholder for CUTLASS Conv2D (fallback to CPU naive implementation)
fn cutlass_conv2d(input: Tensor, weight: Tensor, mut output: Tensor, device: Device) raises -> Error:
    # input: [N, C_in, H_in, W_in], weight: [C_out, C_in, K_h, K_w]
    var N = input.shape()[0]
    var C_in = input.shape()[1]
    var H_in = input.shape()[2]
    var W_in = input.shape()[3]
    var C_out = weight.shape()[0]
    var K_h = weight.shape()[2]
    var K_w = weight.shape()[3]
    var H_out = H_in - K_h + 1
    var W_out = W_in - K_w + 1

    assert output.shape()[0] == N and output.shape()[1] == C_out, "Output shape mismatch"

    if device.is_cpu():
        for n in range(N):
            for co in range(C_out):
                for ho in range(H_out):
                    for wo in range(W_out):
                        var sum_val: Float64 = 0.0
                        for ci in range(C_in):
                            for kh in range(K_h):
                                for kw in range(K_w):
                                    var in_h = ho + kh
                                    var in_w = wo + kw
                                    var in_idx = ((n * C_in + ci) * H_in + in_h) * W_in + in_w
                                    var w_idx = ((co * C_in + ci) * K_h + kh) * K_w + kw
                                    sum_val += input.get_item(in_idx) * weight.get_item(w_idx)
                        var out_idx = ((n * C_out + co) * H_out + ho) * W_out + wo
                        output.set_item(out_idx, sum_val)
    else:
        # Real implementation would call CUTLASS Conv2D API
        # For now, fallback to CPU path
        return cutlass_conv2d(input, weight, output, Device("cpu"))

    return Error.ok()

# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True
    var dev = Device("cpu")

    # Test GEMM
    var A = Tensor([1.0, 2.0,
                    3.0, 4.0], shape=[2,2])
    var B = Tensor([5.0, 6.0,
                    7.0, 8.0], shape=[2,2])
    var C = Tensor([0.0, 0.0,
                    0.0, 0.0], shape=[2,2])
    try:
        var err1 = cutlass_gemm(1.0, A, B, 0.0, C, dev)
    except e:
        return False
    if C.get_item(0) != 19.0 or C.get_item(3) != 50.0:
        ok = False

    # Test Conv2D 1x1
    var input = Tensor([1.0], shape=[1,1,1,1])
    var weight = Tensor([2.0], shape=[1,1,1,1])
    var output = Tensor([0.0], shape=[1,1,1,1])
    try:
        var err2 = cutlass_conv2d(input, weight, output, dev)
    except e:
        return False
    if output.get_item(0) != 2.0:
        ok = False

    return ok