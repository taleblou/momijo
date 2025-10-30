# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.losses.classification
# File:         src/momijo/learn/losses/classification.mojo
#
# Description:
#   Classification losses with numerically stable implementations:
#   - Cross-Entropy (multi-class) for index or one-hot targets
#   - Binary Cross-Entropy (with logits)
#   - Focal Loss (binary & multi-class via logits)
#   Primary implementations are backend-agnostic (List[Float64]/List[List[Float64]]).
#   This file ALSO includes thin adapters for momijo.tensor.Tensor[Float64] so you
#   can use the same APIs with tensors without touching the core logic.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor   # used by the optional tensor adapters below

# =============================================================================
# Backend-agnostic numeric helpers (stable fallbacks)
# =============================================================================

@always_inline
fn _eps() -> Float64:
    return 1e-12

@always_inline
fn _abs(x: Float64) -> Float64:
    return -x if x < 0.0 else x

@always_inline
fn _max(a: Float64, b: Float64) -> Float64:
    return a if a > b else b

@always_inline
fn _clamp(x: Float64, lo: Float64, hi: Float64) -> Float64:
    var v = x
    if v < lo: v = lo
    if v > hi: v = hi
    return v

# Crude exp/log fallbacks (range-limited) — good enough for demos/tests.
# For production, wire to a high-quality math/tensor implementation.
fn _exp(x_in: Float64) -> Float64:
    var x = _clamp(x_in, -40.0, 40.0)   # prevent overflow
    # 5th-order Taylor around 0 with e^x = (e^(x/2))^2 trick
    var h = x * 0.5
    var h2 = h * h
    var h3 = h2 * h
    var h4 = h3 * h
    var h5 = h4 * h
    var t = 1.0 + h + (h2 * 0.5) + (h3 * (1.0 / 6.0)) + (h4 * 0.0416666666667) + (h5 * 0.0083333333333)
    return t * t

fn _log(x_in: Float64) -> Float64:
    # Newton iteration on f(y)=exp(y)-x: y_{k+1} = y_k - 1 + x/exp(y_k)
    var x = _max(x_in, _eps())
    var y = x - 1.0  # initial guess near ln(1+u) ~ u
    var k = 0
    while k < 12:
        var ey = _exp(y)
        y = y - 1.0 + (x / ey)
        k = k + 1
    return y

# =============================================================================
# Common helpers
# =============================================================================

# logsumexp over one row
fn _log_sum_exp(row: List[Float64]) -> Float64:
    var m = row[0]
    var i = 1
    while i < Int(row.size()):
        if row[i] > m:
            m = row[i]
        i = i + 1
    var s: Float64 = 0.0
    i = 0
    while i < Int(row.size()):
        s = s + _exp(row[i] - m)
        i = i + 1
    return m + _log(s + _eps())

# Softmax probabilities (row-wise)
fn _softmax_row(row: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    out.reserve(Int(row.size()))
    var lse = _log_sum_exp(row)
    var i = 0
    while i < Int(row.size()):
        out.push_back(_exp(row[i] - lse))
        i = i + 1
    return out

# (1 - p)^gamma using log/exp with clamping
fn _pow1m(p: Float64, gamma: Float64) -> Float64:
    var q = _clamp(1.0 - p, _eps(), 1.0)
    return _exp(gamma * _log(q))

# Sigmoid(x) = 1 / (1 + exp(-x))
@always_inline
fn _sigmoid(x: Float64) -> Float64:
    if x >= 0.0:
        var z = _exp(-x)
        return 1.0 / (1.0 + z)
    else:
        var z = _exp(x)
        return z / (1.0 + z)

# =============================================================================
# Cross-Entropy (multi-class)
# =============================================================================

# Targets are indices (0..C-1) — logits: [N][C], target_index: [N]
fn cross_entropy(logits: List[List[Float64]], target_index: List[Int]) -> Float64:
    var n = Int(logits.size())
    if n == 0: return 0.0
    var total: Float64 = 0.0
    var i = 0
    while i < n:
        var row = logits[i]
        var t = target_index[i]
        var lse = _log_sum_exp(row)
        total = total + (lse - row[t])   # -log softmax = logsumexp - logit[t]
        i = i + 1
    return total / Float64(n)

# Targets are one-hot rows — logits: [N][C], target_one_hot: [N][C]
fn cross_entropy(logits: List[List[Float64]], target_one_hot: List[List[Float64]]) -> Float64:
    var n = Int(logits.size())
    if n == 0: return 0.0
    var total: Float64 = 0.0
    var i = 0
    while i < n:
        var row = logits[i]
        var trow = target_one_hot[i]
        var lse = _log_sum_exp(row)
        var ce_i: Float64 = 0.0
        var j = 0
        while j < Int(row.size()):
            var y = trow[j]
            if y != 0.0:
                ce_i = ce_i - (y * (row[j] - lse))
            j = j + 1
        total = total + ce_i
        i = i + 1
    return total / Float64(n)

# =============================================================================
# Binary Cross-Entropy (with logits)
# =============================================================================
# Stable per-element formula:
#   BCE(x,y) = max(x,0) - x*y + log(1 + exp(-|x|))

@always_inline
fn _bce_logit_scalar(logit: Float64, target: Float64) -> Float64:
    var x = logit
    var y = _clamp(target, 0.0, 1.0)
    return _max(x, 0.0) - (x * y) + _log(1.0 + _exp(-_abs(x)) + _eps())

# Vector: logits/targets are length-N
fn binary_cross_entropy(logits: List[Float64], targets: List[Float64]) -> Float64:
    var n = Int(logits.size())
    if n == 0: return 0.0
    var total: Float64 = 0.0
    var i = 0
    while i < n:
        total = total + _bce_logit_scalar(logits[i], targets[i])
        i = i + 1
    return total / Float64(n)

# Batched: logits/targets are [N][K] (K independent binary tasks per sample)
fn binary_cross_entropy(logits: List[List[Float64]], targets: List[List[Float64]]) -> Float64:
    var n = Int(logits.size())
    if n == 0: return 0.0
    var total: Float64 = 0.0
    var i = 0
    while i < n:
        total = total + binary_cross_entropy(logits[i], targets[i])
        i = i + 1
    return total / Float64(n)

# =============================================================================
# Focal Loss (with logits)
# =============================================================================

# Binary focal loss: logits/targets are length-N
fn focal_loss_binary_with_logits(
    logits: List[Float64],
    targets: List[Float64],
    gamma: Float64 = 2.0,
    alpha: Float64 = 0.25
) -> Float64:
    var n = Int(logits.size())
    if n == 0: return 0.0
    var total: Float64 = 0.0
    var i = 0
    while i < n:
        var x = logits[i]
        var y = _clamp(targets[i], 0.0, 1.0)
        var p = _sigmoid(x)                 # probability for class=1
        var pt = p if y == 1.0 else (1.0 - p)
        var w = alpha if y == 1.0 else (1.0 - alpha)
        var ce = _bce_logit_scalar(x, y)
        var mod = _pow1m(pt, gamma)         # (1 - pt)^gamma
        total = total + (w * mod * ce)
        i = i + 1
    return total / Float64(n)

# Multi-class focal (targets are indices) — logits: [N][C], target_index: [N]
fn focal_loss_multiclass_with_logits(
    logits: List[List[Float64]],
    target_index: List[Int],
    gamma: Float64 = 2.0,
    alpha: Float64 = 0.25
) -> Float64:
    var n = Int(logits.size())
    if n == 0: return 0.0
    var total: Float64 = 0.0
    var i = 0
    while i < n:
        var probs = _softmax_row(logits[i])
        var t = target_index[i]
        var pt = probs[t]
        var ce_i = -_log(_clamp(pt, _eps(), 1.0))  # CE_i = -log p_t
        var mod = _pow1m(pt, gamma)
        total = total + (alpha * mod * ce_i)       # class-balanced with single alpha
        i = i + 1
    return total / Float64(n)

# Multi-class focal (one-hot targets) — logits: [N][C], target_one_hot: [N][C]
fn focal_loss_multiclass_with_logits(
    logits: List[List[Float64]],
    target_one_hot: List[List[Float64]],
    gamma: Float64 = 2.0,
    alpha: Float64 = 0.25
) -> Float64:
    var n = Int(logits.size())
    if n == 0: return 0.0
    var total: Float64 = 0.0
    var i = 0
    while i < n:
        var probs = _softmax_row(logits[i])
        var trow = target_one_hot[i]
        var j = 0
        while j < Int(probs.size()):
            var y = trow[j]
            if y != 0.0:
                var pt = probs[j]
                var ce_j = -_log(_clamp(pt, _eps(), 1.0))
                var mod = _pow1m(pt, gamma)
                total = total + (alpha * mod * ce_j)
            j = j + 1
        i = i + 1
    return total / Float64(n)

# =============================================================================
# momijo.tensor ADAPTERS (optional)
# =============================================================================
# These overloads allow calling the same losses with tensor.Tensor[Float64].
# We keep this section small and isolated; if your tensor API names differ,
# edit only the four adapter hooks _t_shape1/_t_shape2/_t_get1/_t_get2 below.

# ---- Adapter hooks: change these bodies to match your real Tensor API ----

@always_inline
fn _t_shape1(t: tensor.Tensor[Float64]) -> Int:
    # Replace with your real accessor, e.g., t.shape()[0]
    return tensor.shape1(t)

@always_inline
fn _t_shape2(t: tensor.Tensor[Float64]) -> (Int, Int):
    # Replace with your real accessor, e.g., (t.shape()[0], t.shape()[1])
    return tensor.shape2(t)

@always_inline
fn _t_get1(t: tensor.Tensor[Float64], i: Int) -> Float64:
    # Replace with your real accessor, e.g., t.get([i])
    return tensor.get1_f64(t, i)

@always_inline
fn _t_get2(t: tensor.Tensor[Float64], i: Int, j: Int) -> Float64:
    # Replace with your real accessor, e.g., t.get([i, j])
    return tensor.get2_f64(t, i, j)

# ---- Converters: Tensor -> List / List[List] ----

fn _t_to_list1(t: tensor.Tensor[Float64]) -> List[Float64]:
    var n = _t_shape1(t)
    var out = List[Float64]()
    out.reserve(n)
    var i = 0
    while i < n:
        out.push_back(_t_get1(t, i))
        i = i + 1
    return out

fn _t_to_list2(t: tensor.Tensor[Float64]) -> List[List[Float64]]:
    var dims = _t_shape2(t)
    var n = dims[0]; var c = dims[1]
    var rows = List[List[Float64]]()
    rows.reserve(n)
    var i = 0
    while i < n:
        var row = List[Float64]()
        row.reserve(c)
        var j = 0
        while j < c:
            row.push_back(_t_get2(t, i, j))
            j = j + 1
        rows.push_back(row)
        i = i + 1
    return rows

# ---- Tensor overloads calling the list implementations ----

# CE: logits NxC (Tensor), targets index List[Int]
fn cross_entropy(logits: tensor.Tensor[Float64], target_index: List[Int]) -> Float64:
    return cross_entropy(_t_to_list2(logits), target_index)

# CE: logits NxC (Tensor), targets one-hot NxC (Tensor)
fn cross_entropy(logits: tensor.Tensor[Float64], target_one_hot: tensor.Tensor[Float64]) -> Float64:
    return cross_entropy(_t_to_list2(logits), _t_to_list2(target_one_hot))

# BCE (vector): logits len-N (Tensor), targets len-N (Tensor)
fn binary_cross_entropy(logits: tensor.Tensor[Float64], targets: tensor.Tensor[Float64]) -> Float64:
    return binary_cross_entropy(_t_to_list1(logits), _t_to_list1(targets))

# BCE (batched): logits NxK (Tensor), targets NxK (Tensor)
fn binary_cross_entropy(logits: tensor.Tensor[Float64], targets: tensor.Tensor[Float64]) -> Float64:
    return binary_cross_entropy(_t_to_list2(logits), _t_to_list2(targets))

# Focal (binary): logits len-N (Tensor), targets len-N (Tensor)
fn focal_loss_binary_with_logits(
    logits: tensor.Tensor[Float64],
    targets: tensor.Tensor[Float64],
    gamma: Float64 = 2.0,
    alpha: Float64 = 0.25
) -> Float64:
    return focal_loss_binary_with_logits(_t_to_list1(logits), _t_to_list1(targets), gamma, alpha)

# Focal (multiclass, index): logits NxC (Tensor), target index List[Int]
fn focal_loss_multiclass_with_logits(
    logits: tensor.Tensor[Float64],
    target_index: List[Int],
    gamma: Float64 = 2.0,
    alpha: Float64 = 0.25
) -> Float64:
    return focal_loss_multiclass_with_logits(_t_to_list2(logits), target_index, gamma, alpha)

# Focal (multiclass, one-hot): logits NxC (Tensor), targets NxC (Tensor)
fn focal_loss_multiclass_with_logits(
    logits: tensor.Tensor[Float64],
    target_one_hot: tensor.Tensor[Float64],
    gamma: Float64 = 2.0,
    alpha: Float64 = 0.25
) -> Float64:
    return focal_loss_multiclass_with_logits(_t_to_list2(logits), _t_to_list2(target_one_hot), gamma, alpha)

 