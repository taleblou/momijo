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
# Project: momijo.nn.layers
# File: src/momijo/nn/layers/batchnorm.mojo

from momijo.core.error import module
from momijo.core.traits import zero
from momijo.dataframe.expr import single
from momijo.dataframe.helpers import m, sqrt
from momijo.io.datasets.datapipe import batch
from momijo.nn.training.metrics import update
from momijo.utils.random import sample
from momijo.utils.timer import start
from pathlib import Path
from pathlib.path import Path

# NOTE: Removed duplicate definition of `_zeros1d`; use `from momijo.nn.losses.mse import _zeros1d`

fn _ones1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(1.0)
    return y
fn _zeros2d(r: Int, c: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(r):
        var row = List[Float64]()
        for j in range(c): row.push(0.0)
        y.push(row)
    return y
fn _mean_channelwise(x: List[List[Float64]]) -> List[Float64]:
    var N = len(x)
    if N == 0: return List[Float64]()
    var C = len(x[0])
    var m = _zeros1d(C)
    for i in range(N):
        for c in range(C):
            m[c] += x[i][c]
    var invN = 1.0 / Float64(N)
    for c in range(C): m[c] *= invN
    return m
fn _var_channelwise(x: List[List[Float64]], mean: List[Float64]) -> List[Float64]:
    var N = len(x)
    if N == 0: return List[Float64]()
    var C = len(x[0])
    var v = _zeros1d(C)
    for i in range(N):
        for c in range(C):
            var d = x[i][c] - mean[c]
            v[c] += d * d
    var invN = 1.0 / Float64(N)
    for c in range(C): v[c] *= invN
    return v
fn _norm_apply_affine(sample: List[Float64], mean: List[Float64], var: List[Float64], gamma: List[Float64], beta: List[Float64], eps: Float64) -> List[Float64]:
    var C = len(sample)
    var out = _zeros1d(C)
    for c in range(C):
        var denom = (var[c] + eps)
        # crude sqrt using Newton steps to avoid heavy deps
        var s = 1.0
        if denom > 0.0:
            # initial guess
            s = denom
            # two Newton iterations for sqrt
            s = 0.5 * (s + denom / s)
            s = 0.5 * (s + denom / s)
        var xhat = (sample[c] - mean[c]) / s
        out[c] = gamma[c] * xhat + beta[c]
    return out
fn _update_running(mut running: List[Float64], batch: List[Float64], momentum: Float64) -> None:
    var C = len(batch)
    for c in range(C):
        running[c] = momentum * running[c] + (1.0 - momentum) * batch[c]

# --- BatchNorm1d ---
struct BatchNorm1d:
    var num_features: Int
    var eps: Float64
    var momentum: Float64
    var affine: Bool
    var track_running_stats: Bool
    var training: Bool

    var running_mean: List[Float64]
    var running_var: List[Float64]
    var gamma: List[Float64]
    var beta: List[Float64]
fn __init__(out self, num_features: Int, eps: Float64 = 1e-5, momentum: Float64 = 0.1, affine: Bool = True, track_running_stats: Bool = True, training: Bool = True) -> None:
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.training = training

        self.running_mean = _zeros1d(num_features)
        self.running_var = _ones1d(num_features)  # start with 1
        self.gamma = _ones1d(num_features) if affine else _ones1d(num_features)
        self.beta = _zeros1d(num_features) if affine else _zeros1d(num_features)
# NOTE: Removed duplicate definition of `train_mode`; use `from momijo.nn.layers.dropout import train_mode`
fn eval_mode(mut self) -> None:
        self.training = False

    # Forward for batch input [N, C]
fn forward_batch(mut self, x: List[List[Float64]]) -> List[List[Float64]]:
        var N = len(x)
        if N == 0: return x
        var C = len(x[0])
        # compute per-channel stats
        var mean_b = _mean_channelwise(x)
        var var_b = _var_channelwise(x, mean_b)

        # update running stats if configured and in training
        if self.training and self.track_running_stats:
            _update_running(self.running_mean, mean_b, self.momentum)
            _update_running(self.running_var, var_b, self.momentum)

        var out = _zeros2d(N, C)
        var mean_used = mean_b if self.training else self.running_mean
        var var_used = var_b if self.training else self.running_var

        for i in range(N):
            out[i] = _norm_apply_affine(x[i], mean_used, var_used, self.gamma, self.beta, self.eps)
        return out

    # Forward for single sample [C]
fn forward_sample(self, x: List[Float64]) -> List[Float64]:
        # In training mode, we use running stats even for single sample to avoid div by zero.
        var mean_used = self.running_mean
        var var_used = self.running_var
        return _norm_apply_affine(x, mean_used, var_used, self.gamma, self.beta, self.eps)

    # Convenience wrapper: detect rank by element type
fn forward(mut self, x_2d: List[List[Float64]]) -> List[List[Float64]]:
        return self.forward_batch(x_2d)
fn __copyinit__(out self, other: Self) -> None:
        self.num_features = other.num_features
        self.eps = other.eps
        self.momentum = other.momentum
        self.affine = other.affine
        self.track_running_stats = other.track_running_stats
        self.training = other.training
        self.running_mean = other.running_mean
        self.running_var = other.running_var
        self.gamma = other.gamma
        self.beta = other.beta
fn __moveinit__(out self, deinit other: Self) -> None:
        self.num_features = other.num_features
        self.eps = other.eps
        self.momentum = other.momentum
        self.affine = other.affine
        self.track_running_stats = other.track_running_stats
        self.training = other.training
        self.running_mean = other.running_mean
        self.running_var = other.running_var
        self.gamma = other.gamma
        self.beta = other.beta
# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    var N = 4
    var C = 3
    var x = _zeros2d(N, C)
    for i in range(N):
        for c in range(C):
            x[i][c] = Float64(i + 1) * 0.5 + Float64(c)  # simple pattern

    var bn = BatchNorm1d(C, 1e-5, 0.1, True, True, True)
    var y_train = bn.forward_batch(x)
    ok = ok and (len(y_train) == N) and (len(y_train[0]) == C)

    # Switch to eval and run again on same input
    bn.eval_mode()
    var y_eval = bn.forward_batch(x)
    ok = ok and (len(y_eval) == N) and (len(y_eval[0]) == C)

    # Single sample forward (uses running stats)
    var sample = _zeros1d(C)
    for c in range(C): sample[c] = 0.3 * Float64(c + 1)
    var y1 = bn.forward_sample(sample)
    ok = ok and (len(y1) == C)

    return ok