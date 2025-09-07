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
# Project: momijo.kernels.reference
# File: src/momijo/kernels/reference/conv_ref.mojo

from momijo.core.error import Error
from momijo.tensor.tensor import Tensor

fn conv2d_ref(input: Tensor, weight: Tensor, mut output: Tensor) raises -> Error:
    # input: [N, C_in, H_in, W_in]
    # weight: [C_out, C_in, K_h, K_w]
    # output: [N, C_out, H_out, W_out]

    var N = input.shape()[0]
    var C_in = input.shape()[1]
    var H_in = input.shape()[2]
    var W_in = input.shape()[3]

    var C_out = weight.shape()[0]
    var K_h = weight.shape()[2]
    var K_w = weight.shape()[3]

    var H_out = H_in - K_h + 1
    var W_out = W_in - K_w + 1

    assert output.shape()[0] == N, "Output batch mismatch"
    assert output.shape()[1] == C_out, "Output channel mismatch"
    assert output.shape()[2] == H_out, "Output height mismatch"
    assert output.shape()[3] == W_out, "Output width mismatch"

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

    return Error.ok()

# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True
    # Test 1x1 conv on 1x1 input
    var input = Tensor([1.0], shape=[1,1,1,1])
    var weight = Tensor([2.0], shape=[1,1,1,1])
    var output = Tensor([0.0], shape=[1,1,1,1])
    try:
        var err = conv2d_ref(input, weight, output)
    except e:
        return False
    if output.get_item(0) != 2.0:
        ok = False
    return ok
fn main() -> None:
    var ok = _self_test()
    if ok:
        print("kernels/reference/conv_ref.mojo self-test: OK")
    else:
        print("kernels/reference/conv_ref.mojo self-test: FAILED")