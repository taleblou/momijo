# Project:      Momijo
# Module:       learn.losses.regression
# File:         losses/regression.mojo
# Path:         src/momijo/learn/losses/regression.mojo
#
# Description:  Regression losses (MSE/MAE/Huber) for Momijo Learn.
#               Provides scalar and vectorized variants with optional sample
#               weights and configurable reduction ("mean" | "sum").
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
#   - Functions: mse_loss, mae_loss, huber_loss (scalar & vector forms)
#   - Reduction: "mean" (default) or "sum"
#   - Optional sample_weight (same length as targets)
#   - Backend-agnostic: operates on Float64 and List[Float64];
#     can be wired to momijo.tensor later.

from collections.list import List

# ----------------------------
# Internal helpers
# ----------------------------

fn _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x

fn _validate_lengths(y_pred: List[Float64], y_true: List[Float64]):
    assert(Int(y_pred.size()) == Int(y_true.size()))

fn _validate_weights(weights_opt: Optional[List[Float64]], n: Int):
    if weights_opt is None:
        return
    var w = weights_opt.value()
    assert(Int(w.size()) == n)

fn _is_mean(reduction: String) -> Bool:
    return reduction == String("mean")

fn _is_sum(reduction: String) -> Bool:
    return reduction == String("sum")

fn _assert_valid_reduction(reduction: String):
    assert(_is_mean(reduction) or _is_sum(reduction))

# ----------------------------
# Mean Squared Error (MSE)
# ----------------------------

# Scalar variant
fn mse_loss(y_pred: Float64, y_true: Float64) -> Float64:
    var diff = y_pred - y_true
    return diff * diff

# Vector variant
fn mse_loss(
    y_pred: List[Float64],
    y_true: List[Float64],
    reduction: String = String("mean"),
    sample_weight: Optional[List[Float64]] = None
) -> Float64:
    _assert_valid_reduction(reduction)
    _validate_lengths(y_pred, y_true)
    var n = Int(y_true.size())
    if n == 0:
        return 0.0
    _validate_weights(sample_weight, n)

    var sum_val: Float64 = 0.0
    var weight_sum: Float64 = 0.0
    var i = 0
    if sample_weight is None:
        while i < n:
            var d = y_pred[i] - y_true[i]
            sum_val = sum_val + (d * d)
            i = i + 1
        if _is_sum(reduction):
            return sum_val
        # mean
        return sum_val / Float64(n)
    else:
        var w = sample_weight.value()
        while i < n:
            var d = y_pred[i] - y_true[i]
            var wi = w[i]
            sum_val = sum_val + wi * (d * d)
            weight_sum = weight_sum + wi
            i = i + 1
        if _is_sum(reduction):
            return sum_val
        # mean with weights: divide by sum of weights (guard zero)
        if weight_sum == 0.0:
            return 0.0
        return sum_val / weight_sum

# ----------------------------
# Mean Absolute Error (MAE)
# ----------------------------

# Scalar variant
fn mae_loss(y_pred: Float64, y_true: Float64) -> Float64:
    var diff = y_pred - y_true
    return _abs(diff)

# Vector variant
fn mae_loss(
    y_pred: List[Float64],
    y_true: List[Float64],
    reduction: String = String("mean"),
    sample_weight: Optional[List[Float64]] = None
) -> Float64:
    _assert_valid_reduction(reduction)
    _validate_lengths(y_pred, y_true)
    var n = Int(y_true.size())
    if n == 0:
        return 0.0
    _validate_weights(sample_weight, n)

    var sum_val: Float64 = 0.0
    var weight_sum: Float64 = 0.0
    var i = 0
    if sample_weight is None:
        while i < n:
            var d = _abs(y_pred[i] - y_true[i])
            sum_val = sum_val + d
            i = i + 1
        if _is_sum(reduction):
            return sum_val
        return sum_val / Float64(n)
    else:
        var w = sample_weight.value()
        while i < n:
            var d = _abs(y_pred[i] - y_true[i])
            var wi = w[i]
            sum_val = sum_val + wi * d
            weight_sum = weight_sum + wi
            i = i + 1
        if _is_sum(reduction):
            return sum_val
        if weight_sum == 0.0:
            return 0.0
        return sum_val / weight_sum

# ----------------------------
# Huber loss (a.k.a. Smooth L1)
# ----------------------------
# δ-controls the transition between L2 (quadratic) and L1 (linear) regions:
# huber(e) = 0.5 * e^2              if |e| <= δ
#          = δ * (|e| - 0.5 * δ)    if |e| >  δ

# Scalar variant
fn huber_loss(y_pred: Float64, y_true: Float64, delta: Float64 = 1.0) -> Float64:
    var e = y_pred - y_true
    var ae = _abs(e)
    if ae <= delta:
        return 0.5 * e * e
    return delta * (ae - 0.5 * delta)

# Vector variant
fn huber_loss(
    y_pred: List[Float64],
    y_true: List[Float64],
    delta: Float64 = 1.0,
    reduction: String = String("mean"),
    sample_weight: Optional[List[Float64]] = None
) -> Float64:
    _assert_valid_reduction(reduction)
    _validate_lengths(y_pred, y_true)
    var n = Int(y_true.size())
    if n == 0:
        return 0.0
    _validate_weights(sample_weight, n)

    var sum_val: Float64 = 0.0
    var weight_sum: Float64 = 0.0
    var i = 0
    if sample_weight is None:
        while i < n:
            var e = y_pred[i] - y_true[i]
            var ae = _abs(e)
            if ae <= delta:
                sum_val = sum_val + (0.5 * e * e)
            else:
                sum_val = sum_val + (delta * (ae - 0.5 * delta))
            i = i + 1
        if _is_sum(reduction):
            return sum_val
        return sum_val / Float64(n)
    else:
        var w = sample_weight.value()
        while i < n:
            var e = y_pred[i] - y_true[i]
            var ae = _abs(e)
            var li: Float64
            if ae <= delta:
                li = 0.5 * e * e
            else:
                li = delta * (ae - 0.5 * delta)
            var wi = w[i]
            sum_val = sum_val + wi * li
            weight_sum = weight_sum + wi
            i = i + 1
        if _is_sum(reduction):
            return sum_val
        if weight_sum == 0.0:
            return 0.0
        return sum_val / weight_sum
