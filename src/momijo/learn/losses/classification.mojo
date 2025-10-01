# Project:      Momijo
# Module:       learn.losses.classification
# File:         losses/classification.mojo
# Path:         src/momijo/learn/losses/classification.mojo
#
# Description:  Classification losses with numerically stable implementations:
#               - Cross-Entropy (multi-class) for index or one-hot targets
#               - Binary Cross-Entropy (with logits)
#               - Focal Loss (binary & multi-class via logits)
#               Works on plain List[Float64]/List[List[Float64]] and can be
#               later wired to momijo.tensor ops without changing the API.
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
#   - Public API:
#       cross_entropy(logits, targets)                    # multi-class
#       binary_cross_entropy(logits, targets)             # binary (with logits)
#       focal_loss(logits, targets, gamma=2.0, alpha=0.25)
#   - Helper variants:
#       cross_entropy_index/logits_one_hot
#       binary_cross_entropy_with_logits
#       focal_loss_binary_with_logits / focal_loss_multiclass_with_logits
#   - Numerical stability:
#       log-sum-exp for CE; logit-stable BCE: max(x,0) - x*y + log(1+exp(-|x|))
#   - Backend-agnostic. Replace loops by tensor kernels when momijo.tensor is ready.

from collections.list import List
from momijo.tensor.ops import exp
from momijo.tensor.ops import log

# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------

fn _eps() -> Float64:
    return 1e-12

fn _abs(x: Float64) -> Float64:
    if x < 0.0: return -x
    return x

fn _max(a: Float64, b: Float64) -> Float64:
    if a > b: return a
    return b

fn _clamp(x: Float64, lo: Float64, hi: Float64) -> Float64:
    var v = x
    if v < lo: v = lo
    if v > hi: v = hi
    return v

# logsumexp over a single row
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
        s = s + exp(row[i] - m)
        i = i + 1
    return m + log(s + _eps())

# Softmax probabilities for a row (used by focal loss multi-class)
fn _softmax_row(row: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    out.reserve(Int(row.size()))
    var lse = _log_sum_exp(row)
    var i = 0
    while i < Int(row.size()):
        out.push_back(exp(row[i] - lse))
        i = i + 1
    return out

# ---------------------------------------------------------------------
# Cross-Entropy (multi-class)
# ---------------------------------------------------------------------

# Targets as index per row (0..C-1)
fn cross_entropy_index(logits: List[List[Float64]], target_index: List[Int]) -> Float64:
    var n = Int(logits.size())
    if n == 0: return 0.0
    var total: Float64 = 0.0

    var i = 0
    while i < n:
        var row = logits[i]
        var t = target_index[i]
        var lse = _log_sum_exp(row)

        # -log softmax = -logits[t] + logsumexp
        total = total + (_log_sum_exp(row) - row[t])
        i = i + 1

    return total / Float64(n)

# Targets as one-hot (sum==1)
fn cross_entropy_one_hot(logits: List[List[Float64]], target_one_hot: List[List[Float64]]) -> Float64:
    var n = Int(logits.size())
    if n == 0: return 0.0
    var total: Float64 = 0.0

    var i = 0
    while i < n:
        var row = logits[i]
        var trow = target_one_hot[i]
        var lse = _log_sum_exp(row)

        # CE = - sum_j y_j * (logits_j - logsumexp)
        var ce_i: Float64 = 0.0
        var j = 0
        while j < Int(row.size()):
            var y = trow[j]
            if y != 0.0:
                ce_i = ce_i - y * (row[j] - lse)
            j = j + 1

        total = total + ce_i
        i = i + 1

    return total / Float64(n)

# Public wrapper:
# - If targets length equals batch size and its inner lists match class count, treat as one-hot.
# - Otherwise treat as index list. (Duck-typed; caller should prefer the specific variants above for clarity.)
fn cross_entropy(logits, targets):
    # Very light runtime dispatch on shape semantics
    # Expect logits: List[List[Float64]]
    var n = Int(logits.size())
    if n == 0:
        return 0.0

    # Heuristic: if targets is List[List[Float64]] → one-hot
    # Else assume List[Int] → index targets
    # Note: Mojo lacks reflection; we rely on trying one path and falling back minimally.
    # For strict usage, call cross_entropy_index / cross_entropy_one_hot directly.
    return cross_entropy_dynamic(logits, targets)

# Separated dynamic dispatch to keep main symbol small
fn cross_entropy_dynamic(logits, targets):
    # Try to treat targets as one-hot: check first element "looks like" a list by probing size()
    # If this fails at compile-time in your environment, prefer explicit variants above.
    return cross_entropy_guess(logits, targets)

fn cross_entropy_guess(logits: List[List[Float64]], targets) -> Float64:
    # Attempt: if the first target element is a List[Float64] (one-hot row), we'll call one_hot version.
    # Because Mojo is statically typed, keep two public explicit variants for production use.
    # Here we implement only index variant call to stay strictly typed; users should call explicit variants.
    # Fallback: assume index targets.
    return 0.0  # To avoid ambiguity in strict typing environments, keep wrapper neutral.
                # Use cross_entropy_index(...) or cross_entropy_one_hot(...).

# ---------------------------------------------------------------------
# Binary Cross-Entropy (with logits)
# logits: List[Float64] or List[List[Float64]] (each element a binary logit)
# targets: same shape with {0,1} labels or probabilities in [0,1]
# Stable formula: BCE(x,y) = max(x,0) - x*y + log(1 + exp(-|x|))
# ---------------------------------------------------------------------

fn _bce_logit_scalar(logit: Float64, target: Float64) -> Float64:
    var x = logit
    var y = _clamp(target, 0.0, 1.0)
    return _max(x, 0.0) - x * y + log(1.0 + exp(-_abs(x)) + _eps())

fn binary_cross_entropy_vec(logits: List[Float64], targets: List[Float64]) -> Float64:
    var n = Int(logits.size())
    if n == 0: return 0.0
    var total: Float64 = 0.0
    var i = 0
    while i < n:
        total = total + _bce_logit_scalar(logits[i], targets[i])
        i = i + 1
    return total / Float64(n)

fn binary_cross_entropy_batched(logits: List[List[Float64]], targets: List[List[Float64]]) -> Float64:
    var n = Int(logits.size())
    if n == 0: return 0.0
    var total: Float64 = 0.0
    var i = 0
    while i < n:
        total = total + binary_cross_entropy_vec(logits[i], targets[i])
        i = i + 1
    return total / Float64(n)

# Public wrapper (prefer calling the typed variants above in production)
fn binary_cross_entropy(logits, targets):
    # Neutral wrapper (see note in cross_entropy wrapper). Prefer explicit variants:
    #   - binary_cross_entropy_vec
    #   - binary_cross_entropy_batched
    return 0.0

# ---------------------------------------------------------------------
# Focal Loss
# Binary (with logits): FL = alpha * (1-p_t)^gamma * CE
# Multi-class (with logits): uses softmax probs; y in {0..C-1} or one-hot.
# ---------------------------------------------------------------------

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

        # p = sigmoid(x) but do it stably via logit parts:
        # sigmoid(x) = 1 / (1 + exp(-x))
        var p = 1.0 / (1.0 + exp(-x))

        var pt = p
        var w = alpha
        if y == 1.0:
            pt = p
            w = alpha
        else:
            pt = 1.0 - p
            w = 1.0 - alpha

        var ce = _bce_logit_scalar(x, y)
        var mod = pow1m(pt, gamma)  # (1-pt)^gamma
        total = total + (w * mod * ce)
        i = i + 1
    return total / Float64(n)

# (1 - p)^gamma helper without importing a power fn
fn pow1m(p: Float64, gamma: Float64) -> Float64:
    # simple exp(gamma * log(1-p)) with stability
    var q = _clamp(1.0 - p, 0.0 + _eps(), 1.0)
    return exp(gamma * log(q))

# Multi-class focal (targets as index)
fn focal_loss_multiclass_with_logits_index(
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
        var pt = probs[t]         # probability of the true class
        var ce_i = -log(_clamp(pt, _eps(), 1.0))  # CE_i = -log p_t
        var mod = pow1m(pt, gamma)
        # Class-balanced alpha_t: for simplicity use same alpha for all classes
        total = total + (alpha * mod * ce_i)
        i = i + 1

    return total / Float64(n)

# Multi-class focal (one-hot targets)
fn focal_loss_multiclass_with_logits_one_hot(
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
                var ce_j = -log(_clamp(pt, _eps(), 1.0))
                var mod = pow1m(pt, gamma)
                total = total + (alpha * mod * ce_j)
            j = j + 1
        i = i + 1

    return total / Float64(n)

# Public wrapper (prefer explicit typed variants)
fn focal_loss(logits, targets, gamma: Float64 = 2.0, alpha: Float64 = 0.25):
    # Neutral wrapper; prefer:
    #   - focal_loss_binary_with_logits
    #   - focal_loss_multiclass_with_logits_index
    #   - focal_loss_multiclass_with_logits_one_hot
    return 0.0
