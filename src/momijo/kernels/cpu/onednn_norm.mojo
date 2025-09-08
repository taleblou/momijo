# Project:      Momijo
# Module:       src.momijo.kernels.cpu.onednn_norm
# File:         onednn_norm.mojo
# Path:         src/momijo/kernels/cpu/onednn_norm.mojo
#
# Description:  src.momijo.kernels.cpu.onednn_norm — focused Momijo functionality with a stable public API.
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
#   - Key functions: batchnorm_inference, batchnorm_training, batchnorm_onednn, _self_test
#   - Error paths explicitly marked with 'raises'.


from math import sqrt
from momijo.tensor.tensor import Tensor

fn batchnorm_inference(input: Tensor, mean: Tensor, var: Tensor, gamma: Tensor, beta: Tensor, mut output: Tensor, eps: Float64 = 1e-5) raises:
    # input/output: [N, C, H, W]
    var N = input.shape()[0]
    var C = input.shape()[1]
    var H = input.shape()[2]
    var W = input.shape()[3]

    assert output.shape() == input.shape(), "Output shape mismatch in batchnorm_inference"

    for n in range(N):
        for c in range(C):
            var mean_val = mean.get_item(c)
            var var_val = var.get_item(c)
            var gamma_val = gamma.get_item(c)
            var beta_val = beta.get_item(c)
            for h in range(H):
                for w in range(W):
                    var idx = ((n * C + c) * H + h) * W + w
                    var norm_val = (input.get_item(idx) - mean_val) / sqrt(var_val + eps)
                    var out_val = gamma_val * norm_val + beta_val
                    output.set_item(idx, out_val)

# Training batch normalization (computes mean/var on the fly)
fn batchnorm_training(input: Tensor, gamma: Tensor, beta: Tensor, mut output: Tensor, mut running_mean: Tensor, mut running_var: Tensor, momentum: Float64 = 0.1, eps: Float64 = 1e-5) raises:
    var N = input.shape()[0]
    var C = input.shape()[1]
    var H = input.shape()[2]
    var W = input.shape()[3]
    var spatial_dim = N * H * W

    for c in range(C):
        var mean_val: Float64 = 0.0
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    var idx = ((n * C + c) * H + h) * W + w
                    mean_val += input.get_item(idx)
        mean_val /= spatial_dim

        var var_val: Float64 = 0.0
        for n in range(N):
            for h in range(H):
                for w in range(W):
                    var idx = ((n * C + c) * H + h) * W + w
                    var diff = input.get_item(idx) - mean_val
                    var_val += diff * diff
        var_val /= spatial_dim

        running_mean.set_item(c, momentum * mean_val + (1.0 - momentum) * running_mean.get_item(c))
        running_var.set_item(c, momentum * var_val + (1.0 - momentum) * running_var.get_item(c))

        for n in range(N):
            for h in range(H):
                for w in range(W):
                    var idx = ((n * C + c) * H + h) * W + w
                    var norm_val = (input.get_item(idx) - mean_val) / sqrt(var_val + eps)
                    var out_val = gamma.get_item(c) * norm_val + beta.get_item(c)
                    output.set_item(idx, out_val)

# Placeholder for OneDNN accelerated normalization (not implemented)
fn batchnorm_onednn(input: Tensor, gamma: Tensor, beta: Tensor, mut output: Tensor, mut running_mean: Tensor, mut running_var: Tensor, eps: Float64 = 1e-5) raises:
    # Ideally this would call into OneDNN primitives if available
    # For now, fallback to training mode naive implementation
    batchnorm_training(input, gamma, beta, output, running_mean, running_var, 0.1, eps)

# --- Minimal self-test for smoke testing ---
fn _self_test() -> Bool:
    var ok = True
    var input = Tensor([1.0, 2.0, 3.0, 4.0], shape=[1,1,2,2])
    var gamma = Tensor([1.0], shape=[1])
    var beta = Tensor([0.0], shape=[1])
    var mean = Tensor([2.5], shape=[1])
    var var = Tensor([1.25], shape=[1])
    var output = Tensor([0.0,0.0,0.0,0.0], shape=[1,1,2,2])
    try:
        batchnorm_inference(input, mean, var, gamma, beta, output)
    except e:
        return False
    # Check normalized mean ~ 0
    var val_sum: Float64 = 0.0
    for i in range(4):
        val_sum += output.get_item(i)
    if abs(val_sum) > 1e-6:
        ok = False
    return ok