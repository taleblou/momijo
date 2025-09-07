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
# Project: momijo.nn.losses
# File: src/momijo/nn/losses/cross_entropy.mojo

from momijo.core.error import module, underflow
from momijo.core.traits import one
from momijo.dataframe.expr import single
from momijo.dataframe.helpers import m, t
from momijo.io.datasets.datapipe import batch
from momijo.tensor.tensor import index
from pathlib import Path
from pathlib.path import Path

fn _exp(x: Float64) -> Float64:
    # truncated series (sufficient for pedagogy/smoke tests)
    var term = 1.0
    var sum = 1.0
    var n = 1
    var k = 1.0
    while n <= 20:
        term *= x / k
        sum += term
        n += 1
        k += 1.0
    return sum

# log via Newton/Halley-like iterations on e^y=x
fn _log(x: Float64) -> Float64:
    if x <= 0.0:
        return -745.0  # clamp for underflow cases
    var y = 0.0
    for it in range(8):
        var ey = _exp(y)
        y = y + 2.0 * (x - ey) / (x + ey)
    return y
fn _logsumexp_row(logits: List[Float64]) -> Float64:
    # log(sum_j exp(l_j)) in stable fashion
    var m = -1.7976931348623157e308
    var n = len(logits)
    for i in range(n):
        if logits[i] > m: m = logits[i]
    var s = 0.0
    for i in range(n):
        s += _exp(logits[i] - m)
    return m + _log(s)
fn _zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y

# --- log_softmax and NLL ---
fn log_softmax_row(logits: List[Float64]) -> List[Float64]:
    var lse = _logsumexp_row(logits)
    var n = len(logits)
    var y = _zeros1d(n)
    for i in range(n): y[i] = logits[i] - lse
    return y

# Negative log-likelihood for a single row (class index target).
# If weight provided and length==C, multiply by weight[target].
# If label_smoothing>0: target distribution is
#   (1-eps) on true class + eps/C on all classes.
fn nll_loss_row(logits: List[Float64], target_idx: Int, class_weight: List[Float64], ignore_index: Int, label_smoothing: Float64) -> (Float64, Float64):
    var C = len(logits)
    if C == 0:
        return (0.0, 0.0)
    if target_idx == ignore_index:
        return (0.0, 0.0)  # (loss, denom_weight)
    var ls = label_smoothing
    if ls < 0.0: ls = 0.0
    if ls > 0.999999: ls = 0.999999
    var logp = log_softmax_row(logits)  # [C]
    var loss = 0.0
    if ls == 0.0:
        var w = 1.0
        if len(class_weight) == C:
            w = class_weight[target_idx]
        loss = -logp[target_idx] * w
        return (loss, w)
    else:
        # label smoothing: -sum_j q_j log p_j ; q_j = (1-eps) 1[j=t] + eps/C
        var w = 1.0
        if len(class_weight) == C:
            w = class_weight[target_idx]
        var uniform = ls / Float64(C)
        for j in range(C):
            var q = uniform
            if j == target_idx:
                q = (1.0 - ls) + uniform
            loss += -q * logp[j]
        loss *= w
        return (loss, w)

# --- 1D API ---
fn cross_entropy1d(logits: List[Float64], target: Int, reduction: String = "mean", class_weight: List[Float64] = List[Float64](), ignore_index: Int = -100, label_smoothing: Float64 = 0.0) -> List[Float64]:
    var (l, w) = nll_loss_row(logits, target, class_weight, ignore_index, label_smoothing)
    var out = List[Float64]()
    if reduction == "none":
        out.push(l)
        return out
    if reduction == "sum":
        out.push(l)
        return out
    # mean
    var denom = w
    if denom == 0.0: denom = 1.0
    out.push(l / denom)
    return out

# --- 2D API: logits [N,C], targets indices [N] ---
fn cross_entropy2d(logits: List[List[Float64]], targets: List[Int], reduction: String = "mean", class_weight: List[Float64] = List[Float64](), ignore_index: Int = -100, label_smoothing: Float64 = 0.0) -> List[Float64]:
    var N = len(logits)
    var out = List[Float64]()
    if reduction == "none":
        for n in range(N):
            var (ln, _) = nll_loss_row(logits[n], targets[n], class_weight, ignore_index, label_smoothing)
            out.push(ln)
        return out
    var sumv = 0.0
    var denom = 0.0
    for n in range(N):
        var (ln, wn) = nll_loss_row(logits[n], targets[n], class_weight, ignore_index, label_smoothing)
        sumv += ln
        denom += wn
    if reduction == "sum":
        out.push(sumv)
    else:
        if denom == 0.0: denom = 1.0
        out.push(sumv / denom)
    return out

# --- Optional: one-hot targets [N,C] ---
# If label_smoothing>0 it is treated in addition to given one-hot distribution;
# i.e., we first normalize provided targets row to sum=1, then mix with uniform.
fn cross_entropy2d_onehot(logits: List[List[Float64]], targets: List[List[Float64]], reduction: String = "mean", class_weight: List[Float64] = List[Float64](), label_smoothing: Float64 = 0.0) -> List[Float64]:
    var N = len(logits)
    var C = 0
    if N > 0: C = len(logits[0])
    var out = List[Float64]()
    var ls = label_smoothing
    if ls < 0.0: ls = 0.0
    if ls > 0.999999: ls = 0.999999

    if reduction == "none":
        for n in range(N):
            var logp = log_softmax_row(logits[n])
            # normalize targets row to sum=1
            var s = 0.0
            for j in range(C): s += targets[n][j]
            if s == 0.0: s = 1.0
            var loss = 0.0
            var w = 1.0
            # optional per-class weight as expected value under q
            for j in range(C):
                var q = targets[n][j] / s
                if ls > 0.0:
                    q = (1.0 - ls) * q + ls / Float64(C)
                var cw = 1.0
                if len(class_weight) == C:
                    cw = class_weight[j]
                loss += -q * logp[j] * cw
            out.push(loss)
        return out

    var sumv = 0.0
    var denom = 0.0
    for n in range(N):
        var logp = log_softmax_row(logits[n])
        var s = 0.0
        for j in range(C): s += targets[n][j]
        if s == 0.0: s = 1.0
        var loss = 0.0
        var wsum = 0.0
        for j in range(C):
            var q = targets[n][j] / s
            if ls > 0.0:
                q = (1.0 - ls) * q + ls / Float64(C)
            var cw = 1.0
            if len(class_weight) == C:
                cw = class_weight[j]
            loss += -q * logp[j] * cw
            wsum += cw * q   # denom as expected class-weight
        sumv += loss
        denom += (wsum if wsum > 0.0 else 1.0)
    if reduction == "sum":
        out.push(sumv)
    else:
        out.push(sumv / denom)
    return out

# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # 1) 1D simple 3-class
    var z = List[Float64](); z.push(0.1); z.push(0.3); z.push(-0.2)
    var l1 = cross_entropy1d(z, 1, "mean", List[Float64](), -100, 0.0)[0]
    ok = ok and (l1 == l1)  # not NaN

    # 2) 2D batch with ignore_index and class weights
    var Z = List[List[Float64]]()
    var r0 = List[Float64](); r0.push(1.0); r0.push(0.0); r0.push(-1.0)
    var r1 = List[Float64](); r1.push(0.5); r1.push(0.2); r1.push(-0.7)
    Z.push(r0); Z.push(r1)
    var T = List[Int](); T.push(0); T.push(-100)  # second is ignored
    var W = List[Float64](); W.push(1.0); W.push(2.0); W.push(3.0)
    var l2 = cross_entropy2d(Z, T, "mean", W, -100, 0.1)[0]
    ok = ok and (l2 == l2)

    # 3) one-hot path
    var To = List[List[Float64]]()
    var t0 = List[Float64](); t0.push(0.0); t0.push(1.0); t0.push(0.0)
    var t1 = List[Float64](); t1.push(1.0); t1.push(0.0); t1.push(0.0)
    To.push(t0); To.push(t1)
    var l3 = cross_entropy2d_onehot(Z, To, "mean", W, 0.05)[0]
    ok = ok and (l3 == l3)

    # 4) none and sum reductions
    var vnone = cross_entropy2d(Z, List[Int]([0,1]), "none", List[Float64](), -100, 0.0)
    ok = ok and (len(vnone) == 2)
    var vsum = cross_entropy2d(Z, List[Int]([0,1]), "sum", List[Float64](), -100, 0.0)[0]
    ok = ok and (vsum >= 0.0)

    return ok