# Project:      Momijo
# Module:       learn.metrics.regression
# File:         metrics/regression.mojo
# Path:         src/momijo/learn/metrics/regression.mojo
#
# Description:  Regression metrics for Momijo Learn. Provides common metrics
#               such as MSE, RMSE, MAE, MAPE, R2 score, and explained variance.
#               Implemented backend-agnostic on List[Float64] for portability;
#               later can be overloaded for Tensor types in momijo.tensor.
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
#   - Functions: mse, rmse, mae, mape, r2_score, explained_variance
#   - Helpers: _mean, _variance, _safe_div, _abs64, _sqrt64
#   - Defensive behavior on edge cases (empty inputs, zero-variance targets).

from collections.list import List

# -----------------------------
# Internal numeric helpers
# -----------------------------

fn _abs64(x: Float64) -> Float64:
    if x < 0.0:
        return 0.0 - x
    return x

# Newton-Raphson sqrt with a few iterations; stable for non-negative inputs.
fn _sqrt64(x: Float64) -> Float64:
    if x <= 0.0:
        return 0.0
    var g = x
    var i = 0
    # 8 iterations are enough for our purposes.
    while i < 8:
        g = 0.5 * (g + x / g)
        i = i + 1
    return g

fn _safe_div(num: Float64, den: Float64, eps: Float64 = 1e-12) -> Float64:
    var d = den
    if d < 0.0:
        d = 0.0 - d
    if d <= eps and d >= 0.0 - eps:
        # Avoid division by zero: return 0 if numerator ~ 0; else a large sentinel.
        if num == 0.0:
            return 0.0
        # Scale by eps to keep bounded but informative
        return num / (eps if den >= 0.0 else 0.0 - eps)
    return num / den

fn _mean(xs: List[Float64]) -> Float64:
    var n = len(xs)
    if n == 0:
        return 0.0
    var s = 0.0
    var i = 0
    while i < n:
        s = s + xs[i]
        i = i + 1
    return s / Float64(n)

# Unbiased = False (population variance) since metrics usually use population form here.
fn _variance(xs: List[Float64], mean_val: Float64) -> Float64:
    var n = len(xs)
    if n == 0:
        return 0.0
    var s2 = 0.0
    var i = 0
    while i < n:
        var d = xs[i] - mean_val
        s2 = s2 + d * d
        i = i + 1
    return s2 / Float64(n)

# -----------------------------
# Public regression metrics
# -----------------------------

# Mean Squared Error
fn mse(y_pred: List[Float64], y_true: List[Float64]) -> Float64:
    var n_pred = len(y_pred)
    var n_true = len(y_true)
    if n_pred == 0 or n_true == 0:
        return 0.0
    # Require equal length; if not, use the min length (defensive).
    var n = n_pred
    if n_true < n:
        n = n_true

    var s = 0.0
    var i = 0
    while i < n:
        var e = y_pred[i] - y_true[i]
        s = s + e * e
        i = i + 1
    return s / Float64(n)

# Root Mean Squared Error
fn rmse(y_pred: List[Float64], y_true: List[Float64]) -> Float64:
    return _sqrt64(mse(y_pred, y_true))

# Mean Absolute Error
fn mae(y_pred: List[Float64], y_true: List[Float64]) -> Float64:
    var n_pred = len(y_pred)
    var n_true = len(y_true)
    if n_pred == 0 or n_true == 0:
        return 0.0
    var n = n_pred
    if n_true < n:
        n = n_true

    var s = 0.0
    var i = 0
    while i < n:
        s = s + _abs64(y_pred[i] - y_true[i])
        i = i + 1
    return s / Float64(n)

# Mean Absolute Percentage Error (in percent if scale=100.0).
# Zeros in y_true are handled with eps to avoid division by zero.
fn mape(y_pred: List[Float64], y_true: List[Float64], scale: Float64 = 100.0, eps: Float64 = 1e-8) -> Float64:
    var n_pred = len(y_pred)
    var n_true = len(y_true)
    if n_pred == 0 or n_true == 0:
        return 0.0
    var n = n_pred
    if n_true < n:
        n = n_true

    var s = 0.0
    var i = 0
    while i < n:
        var denom = y_true[i]
        if denom == 0.0:
            # fallback to eps preserving signless magnitude
            denom = eps
        s = s + _abs64((y_pred[i] - y_true[i]) / denom)
        i = i + 1
    return (s / Float64(n)) * scale

# Coefficient of Determination (R^2).
# R^2 = 1 - SS_res / SS_tot
# If SS_tot == 0: return 1.0 if SS_res==0 else 0.0 (degenerate target variance).
fn r2_score(y_pred: List[Float64], y_true: List[Float64]) -> Float64:
    var n_pred = len(y_pred)
    var n_true = len(y_true)
    if n_pred == 0 or n_true == 0:
        return 0.0
    var n = n_pred
    if n_true < n:
        n = n_true

    # mean of y_true over the matched length
    var sum_true = 0.0
    var i = 0
    while i < n:
        sum_true = sum_true + y_true[i]
        i = i + 1
    var mean_true = sum_true / Float64(n)

    var ss_res = 0.0
    var ss_tot = 0.0
    i = 0
    while i < n:
        var diff = y_pred[i] - y_true[i]
        ss_res = ss_res + diff * diff
        var dmean = y_true[i] - mean_true
        ss_tot = ss_tot + dmean * dmean
        i = i + 1

    if ss_tot == 0.0:
        if ss_res == 0.0:
            return 1.0
        return 0.0

    return 1.0 - (ss_res / ss_tot)

# Explained Variance Score:
# 1 - Var(y - y_pred) / Var(y)
# If Var(y) == 0: return 1.0 if Var(error)==0 else 0.0
fn explained_variance(y_pred: List[Float64], y_true: List[Float64]) -> Float64:
    var n_pred = len(y_pred)
    var n_true = len(y_true)
    if n_pred == 0 or n_true == 0:
        return 0.0
    var n = n_pred
    if n_true < n:
        n = n_true

    # Collect trimmed views into temporary lists to reuse helpers
    var t_true = List[Float64]()
    var t_err  = List[Float64]()

    var i = 0
    while i < n:
        t_true.append(y_true[i])
        t_err.append(y_true[i] - y_pred[i])
        i = i + 1

    var mu_true = _mean(t_true)
    var var_true = _variance(t_true, mu_true)

    var mu_err = _mean(t_err)
    var var_err = _variance(t_err, mu_err)

    if var_true == 0.0:
        if var_err == 0.0:
            return 1.0
        return 0.0

    return 1.0 - (var_err / var_true)
