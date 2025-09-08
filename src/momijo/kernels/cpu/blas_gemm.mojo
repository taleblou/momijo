# Project:      Momijo
# Module:       src.momijo.kernels.cpu.blas_gemm
# File:         blas_gemm.mojo
# Path:         src/momijo/kernels/cpu/blas_gemm.mojo
#
# Description:  src.momijo.kernels.cpu.blas_gemm — focused Momijo functionality with a stable public API.
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
#   - Key functions: gemm, _self_test
#   - Error paths explicitly marked with 'raises'.


from momijo.tensor.tensor import Tensor

fn gemm(alpha: Float64, A: Tensor, B: Tensor, beta: Float64, mut C: Tensor) raises:
    assert A.shape().rank == 2, "Matrix A must be 2D"
    assert B.shape().rank == 2, "Matrix B must be 2D"
    assert C.shape().rank == 2, "Matrix C must be 2D"
    assert A.shape()[1] == B.shape()[0], "Inner dimensions must match"
    assert A.shape()[0] == C.shape()[0], "Output rows mismatch"
    assert B.shape()[1] == C.shape()[1], "Output cols mismatch"

    var m = A.shape()[0]
    var n = B.shape()[1]
    var k = A.shape()[1]

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

# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True
    # A (2x2), B (2x2)
    var A = Tensor([1.0, 2.0,
                    3.0, 4.0], shape=[2, 2])
    var B = Tensor([5.0, 6.0,
                    7.0, 8.0], shape=[2, 2])
    var C = Tensor([0.0, 0.0,
                    0.0, 0.0], shape=[2, 2])

    try:
        gemm(1.0, A, B, 0.0, C)
    except e:
        return False

    # Expected result: [[19, 22], [43, 50]]
    if C.get_item(0) != 19.0 or C.get_item(1) != 22.0:
        ok = False
    if C.get_item(2) != 43.0 or C.get_item(3) != 50.0:
        ok = False

    return ok