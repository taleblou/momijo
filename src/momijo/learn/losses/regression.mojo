# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.losses.regression
# File:         src/momijo/learn/losses/regression.mojo
#
# Description:
#   Regression losses (MSE / MAE / Huber) for Momijo Learn.
#   - Scalar and vectorized variants
#   - Optional sample weights
#   - Reductions: "mean" (default) or "sum"
#   Backend-agnostic (List[Float64]); can be wired to momijo.tensor later.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List

# --- Optional Tensor integration (adapters only; safe no-op if unused) -------
# Import exact facades; adjust paths if your project structure differs.
from momijo.tensor.tensor import Tensor
from momijo.tensor.dtypes import Float32, Float64  # dtype facades  

# Convert Tensor to flat List[Float64].
# Assumptions (pick whichever exists in your tensor API):
#   - Preferred: t.to_list_f64() / t.to_list_f32()
#   - Fallback:  t.flatten().numel(), and get1d(i) or get(i)
# Edit inside these helpers if your actual API differs.

fn _tensor_to_list_f64(t: Tensor[Float64]) raises -> List[Float64]:
    # Fast path: direct list conversion if available
    try:
        # If your Tensor exposes a direct converter:
        return t.to_list_f64()
    except _:
        pass

    # Generic fallback: flatten + indexed read
    var flat = t.flatten()       # expects a view contiguous in logical order
    var n = flat.numel()
    var out = List[Float64]()
    out.reserve(n)
    var i = 0
    while i < n:
        # Try common indexer names; adjust 
        try:
            out.append(flat.get1d(i))
        except _:
            out.append(Float64(flat.get(i)))
        i = i + 1
    return out

fn _tensor_to_list_f64_from_f32(t: Tensor[Float32]) raises -> List[Float64]:
    try:
        var xs32 = t.to_list_f32()
        var out = List[Float64]()
        out.reserve(len(xs32))
        var i = 0
        while i < len(xs32):
            out.append(Float64(xs32[i]))
            i = i + 1
        return out
    except _:
        pass

    var flat = t.flatten()
    var n = flat.numel()
    var out2 = List[Float64]()
    out2.reserve(n)
    var j = 0
    while j < n:
        try:
            out2.append(Float64(flat.get1d(j)))
        except _:
            out2.append(Float64(flat.get(j)))
        j = j + 1
    return out2

# -----------------------------------------------------------------------------
# Internal helpers (pure)
# -----------------------------------------------------------------------------

@always_inline
fn _abs(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x

@always_inline
fn _is_mean(reduction: String) -> Bool:
    return reduction == String("mean")

@always_inline
fn _is_sum(reduction: String) -> Bool:
    return reduction == String("sum")

# -----------------------------------------------------------------------------
# Internal validators (raising)
# -----------------------------------------------------------------------------

fn _check_valid_reduction(reduction: String) raises:
    if not (_is_mean(reduction) or _is_sum(reduction)):
        raise String("Invalid reduction: expected 'mean' or 'sum'.")

fn _check_equal_lengths(y_pred: List[Float64], y_true: List[Float64]) raises:
    if len(y_pred) != len(y_true):
        raise String("Length mismatch: y_pred and y_true must have equal lengths.")

fn _check_weights_len(weights_opt: Optional[List[Float64]], n: Int) raises:
    if weights_opt is None:
        return
    var w = weights_opt.value()
    if len(w) != n:
        raise String("Length mismatch: sample_weight must have the same length as targets.")

# -----------------------------------------------------------------------------
# MSE (scalar)
# -----------------------------------------------------------------------------

@always_inline
fn mse_loss(y_pred: Float64, y_true: Float64) -> Float64:
    var diff = y_pred - y_true
    return diff * diff

# -----------------------------------------------------------------------------
# MSE (vector) — public wrapper (non-raising)
# -----------------------------------------------------------------------------

fn mse_loss(
    y_pred: List[Float64],
    y_true: List[Float64],
    reduction: String = String("mean"),
    sample_weight: Optional[List[Float64]] = None
) -> Float64:
    try:
        return _mse_loss_impl(y_pred, y_true, reduction, sample_weight)
    except _:
        # Safe fallback on invalid input
        return 0.0

# Internal raising implementation
fn _mse_loss_impl(
    y_pred: List[Float64],
    y_true: List[Float64],
    reduction: String,
    sample_weight: Optional[List[Float64]]
) raises -> Float64:
    _check_valid_reduction(reduction)
    _check_equal_lengths(y_pred, y_true)
    var n = len(y_true)
    if n == 0:
        return 0.0
    _check_weights_len(sample_weight, n)

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
        if weight_sum == 0.0:
            return 0.0
        return sum_val / weight_sum

# -----------------------------------------------------------------------------
# MAE (scalar)
# -----------------------------------------------------------------------------

@always_inline
fn mae_loss(y_pred: Float64, y_true: Float64) -> Float64:
    var diff = y_pred - y_true
    return _abs(diff)

# -----------------------------------------------------------------------------
# MAE (vector) — public wrapper (non-raising)
# -----------------------------------------------------------------------------

fn mae_loss(
    y_pred: List[Float64],
    y_true: List[Float64],
    reduction: String = String("mean"),
    sample_weight: Optional[List[Float64]] = None
) -> Float64:
    try:
        return _mae_loss_impl(y_pred, y_true, reduction, sample_weight)
    except _:
        return 0.0

# Internal raising implementation
fn _mae_loss_impl(
    y_pred: List[Float64],
    y_true: List[Float64],
    reduction: String,
    sample_weight: Optional[List[Float64]]
) raises -> Float64:
    _check_valid_reduction(reduction)
    _check_equal_lengths(y_pred, y_true)
    var n = len(y_true)
    if n == 0:
        return 0.0
    _check_weights_len(sample_weight, n)

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

# -----------------------------------------------------------------------------
# Huber (scalar)
# -----------------------------------------------------------------------------

@always_inline
fn huber_loss(y_pred: Float64, y_true: Float64, delta: Float64 = 1.0) -> Float64:
    var e = y_pred - y_true
    var ae = _abs(e)
    if ae <= delta:
        return 0.5 * e * e
    return delta * (ae - 0.5 * delta)

# -----------------------------------------------------------------------------
# Huber (vector) — public wrapper (non-raising)
# -----------------------------------------------------------------------------

fn huber_loss(
    y_pred: List[Float64],
    y_true: List[Float64],
    delta: Float64 = 1.0,
    reduction: String = String("mean"),
    sample_weight: Optional[List[Float64]] = None
) -> Float64:
    try:
        return _huber_loss_impl(y_pred, y_true, delta, reduction, sample_weight)
    except _:
        return 0.0

# Internal raising implementation
fn _huber_loss_impl(
    y_pred: List[Float64],
    y_true: List[Float64],
    delta: Float64,
    reduction: String,
    sample_weight: Optional[List[Float64]]
) raises -> Float64:
    _check_valid_reduction(reduction)
    _check_equal_lengths(y_pred, y_true)
    var n = len(y_true)
    if n == 0:
        return 0.0
    _check_weights_len(sample_weight, n)

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

# -----------------------------------------------------------------------------
# Tensor overloads (thin adapters -> List[Float64] core)
# -----------------------------------------------------------------------------
# These wrappers keep compile surface stable even if tensor internals evolve.
# Adjust only the two adapter functions above for API differences.

fn mse_loss(
    y_pred: Tensor[Float64],
    y_true: Tensor[Float64],
    reduction: String = String("mean"),
    sample_weight: Optional[Tensor[Float64]] = None
) -> Float64:
    try:
        var yp = _tensor_to_list_f64(y_pred)
        var yt = _tensor_to_list_f64(y_true)
        var sw_opt: Optional[List[Float64]] = None
        if sample_weight is not None:
            sw_opt = _tensor_to_list_f64(sample_weight.value())
        return _mse_loss_impl(yp, yt, reduction, sw_opt)
    except _:
        return 0.0

fn mse_loss(
    y_pred: Tensor[Float32],
    y_true: Tensor[Float32],
    reduction: String = String("mean"),
    sample_weight: Optional[Tensor[Float32]] = None
) -> Float64:
    try:
        var yp = _tensor_to_list_f64_from_f32(y_pred)
        var yt = _tensor_to_list_f64_from_f32(y_true)
        var sw_opt: Optional[List[Float64]] = None
        if sample_weight is not None:
            sw_opt = _tensor_to_list_f64_from_f32(sample_weight.value())
        return _mse_loss_impl(yp, yt, reduction, sw_opt)
    except _:
        return 0.0

fn mae_loss(
    y_pred: Tensor[Float64],
    y_true: Tensor[Float64],
    reduction: String = String("mean"),
    sample_weight: Optional[Tensor[Float64]] = None
) -> Float64:
    try:
        var yp = _tensor_to_list_f64(y_pred)
        var yt = _tensor_to_list_f64(y_true)
        var sw_opt: Optional[List[Float64]] = None
        if sample_weight is not None:
            sw_opt = _tensor_to_list_f64(sample_weight.value())
        return _mae_loss_impl(yp, yt, reduction, sw_opt)
    except _:
        return 0.0

fn mae_loss(
    y_pred: Tensor[Float32],
    y_true: Tensor[Float32],
    reduction: String = String("mean"),
    sample_weight: Optional[Tensor[Float32]] = None
) -> Float64:
    try:
        var yp = _tensor_to_list_f64_from_f32(y_pred)
        var yt = _tensor_to_list_f64_from_f32(y_true)
        var sw_opt: Optional[List[Float64]] = None
        if sample_weight is not None:
            sw_opt = _tensor_to_list_f64_from_f32(sample_weight.value())
        return _mae_loss_impl(yp, yt, reduction, sw_opt)
    except _:
        return 0.0

fn huber_loss(
    y_pred: Tensor[Float64],
    y_true: Tensor[Float64],
    delta: Float64 = 1.0,
    reduction: String = String("mean"),
    sample_weight: Optional[Tensor[Float64]] = None
) -> Float64:
    try:
        var yp = _tensor_to_list_f64(y_pred)
        var yt = _tensor_to_list_f64(y_true)
        var sw_opt: Optional[List[Float64]] = None
        if sample_weight is not None:
            sw_opt = _tensor_to_list_f64(sample_weight.value())
        return _huber_loss_impl(yp, yt, delta, reduction, sw_opt)
    except _:
        return 0.0

fn huber_loss(
    y_pred: Tensor[Float32],
    y_true: Tensor[Float32],
    delta: Float64 = 1.0,
    reduction: String = String("mean"),
    sample_weight: Optional[Tensor[Float32]] = None
) -> Float64:
    try:
        var yp = _tensor_to_list_f64_from_f32(y_pred)
        var yt = _tensor_to_list_f64_from_f32(y_true)
        var sw_opt: Optional[List[Float64]] = None
        if sample_weight is not None:
            sw_opt = _tensor_to_list_f64_from_f32(sample_weight.value())
        return _huber_loss_impl(yp, yt, delta, reduction, sw_opt)
    except _:
        return 0.0
