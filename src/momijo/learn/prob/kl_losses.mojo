# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.prob.kl_losses
# File:         src/momijo/learn/prob/kl_losses.mojo
#
# Description:
#   Kullback–Leibler divergence (KL) utilities for probabilistic losses.
#   - Normal vs Normal (scalar / diagonal vector / tensor variants)
#   - Bernoulli (scalar / vector / tensor variants)
#   - Categorical from probability lists/tensors
#   - Results are in nats. Backend-agnostic and numerically guarded.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.learn.prob.distributions import Normal
from momijo.learn.prob.distributions import Bernoulli
from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# Numeric helpers (local approximations to avoid hard math deps)
# -----------------------------------------------------------------------------

@always_inline
fn _clamp(x: Float64, lo: Float64, hi: Float64) -> Float64:
    var v = x
    if v < lo: v = lo
    if v > hi: v = hi
    return v

# Exponential approximation on a bounded range using (1 + x/n)^n with n=64
fn _exp64(x: Float64) -> Float64:
    var xv = _clamp(x, -50.0, 50.0)
    var n = 64.0
    var base = 1.0 + (xv / n)
    var y = base
    y = y * y   # ^2
    y = y * y   # ^4
    y = y * y   # ^8
    y = y * y   # ^16
    y = y * y   # ^32
    y = y * y   # ^64
    return y

# Natural log approximation via range reduction and atanh-series.
# ln(x) = ln(m) + k*ln2, reduce x = m*2^k, with m in ~[0.75, 1.5]
fn _log64(x: Float64) -> Float64:
    if x <= 0.0:
        # represent -inf by a large negative number
        return -1.7976931348623157e308
    var ln2 = 0.6931471805599453
    var k = 0
    var m = x
    while m > 1.5:
        m = m * 0.5
        k = k + 1
    while m < 0.75:
        m = m * 2.0
        k = k - 1
    var t = (m - 1.0) / (m + 1.0)
    var t2 = t * t
    var term = t
    var sum = term
    var i = 1
    while i <= 6:       # 6 odd terms are enough for losses
        term = term * t2
        var denom = Float64(2 * i + 1)
        sum = sum + (term / denom)
        i = i + 1
    return 2.0 * sum + Float64(k) * ln2

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

fn _eps() -> Float64:
    # Minimum positive for probability/variance clamping
    return 1e-12

fn _clamp_prob(x: Float64) -> Float64:
    var e = _eps()
    var one = 1.0 - e
    if x < e: return e
    if x > one: return one
    return x

fn _assert_same_len(a: List[Float64], b: List[Float64]) -> None:
    assert(Int(a.size()) == Int(b.size()))

@always_inline
fn _assert_same_shape_t(a: tensor.Tensor[Float64], b: tensor.Tensor[Float64]) -> None:
    var sa = a.shape(); var sb = b.shape()
    assert(len(sa) == len(sb))
    var i = 0
    while i < len(sa):
        assert(sa[i] == sb[i])
        i = i + 1

@always_inline
fn _numel(shape: List[Int]) -> Int:
    var p = 1
    var i = 0
    var n = len(shape)
    while i < n:
        p = p * shape[i]
        i = i + 1
    return p

# Reduce over last dimension for 2D tensors: [N, D] -> [N]
fn _reduce_lastdim_sum_2d(x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
    var shp = x.shape()
    assert(len(shp) == 2)
    var n = shp[0]; var d = shp[1]
    var out = tensor.Tensor[Float64]([n], 0.0)
    var xd = x._data; var yd = out._data
    var i = 0
    while i < n:
        var acc = 0.0
        var j = 0
        var base = i * d
        while j < d:
            acc = acc + xd[base + j]
            j = j + 1
        yd[i] = acc
        i = i + 1
    return out

# -----------------------------------------------------------------------------
# KL for Normal distributions (scalar/list)
# -----------------------------------------------------------------------------
# KL( N(mu_p, var_p) || N(mu_q, var_q) ) =
#   0.5 * [ log(var_q / var_p) + (var_p + (mu_p - mu_q)^2) / var_q - 1 ]
# We use log-variance (logvar = log(var)) for stability.

fn kl_normal_normal_scalar(mu_p: Float64, logvar_p: Float64,
                           mu_q: Float64, logvar_q: Float64) -> Float64:
    var var_p = _exp64(logvar_p)
    var var_q = _exp64(logvar_q)
    var e = _eps()
    if var_p < e: var_p = e
    if var_q < e: var_q = e

    var term_log = logvar_q - logvar_p
    var mean_diff = mu_p - mu_q
    var term_frac = (var_p + mean_diff * mean_diff) / var_q
    var kl = 0.5 * (term_log + term_frac - 1.0)
    if kl < 0.0: return 0.0
    return kl

fn kl_normal_normal_diag(mu_p: List[Float64], logvar_p: List[Float64],
                         mu_q: List[Float64], logvar_q: List[Float64]) -> Float64:
    _assert_same_len(mu_p, mu_q)
    _assert_same_len(logvar_p, logvar_q)
    _assert_same_len(mu_p, logvar_p)

    var n = Int(mu_p.size())
    var acc = 0.0
    var i = 0
    while i < n:
        acc = acc + kl_normal_normal_scalar(mu_p[i], logvar_p[i], mu_q[i], logvar_q[i])
        i = i + 1
    if acc < 0.0: return 0.0
    return acc

# Typed overload using Normal structs (expects fields mean: Float64, std: Float64)
fn kl_divergence(p: Normal, q: Normal) -> Float64:
    var mu_p = p.mean
    var mu_q = q.mean

    var std_p = p.std
    var std_q = q.std

    var e = _eps()
    if std_p < e: std_p = e
    if std_q < e: std_q = e

    var logvar_p = 2.0 * _log64(std_p)
    var logvar_q = 2.0 * _log64(std_q)
    return kl_normal_normal_scalar(mu_p, logvar_p, mu_q, logvar_q)

# -----------------------------------------------------------------------------
# KL for Bernoulli (scalar/list)
# -----------------------------------------------------------------------------
# KL( Ber(p) || Ber(q) ) = p*log(p/q) + (1-p)*log((1-p)/(1-q))

fn kl_bernoulli_scalar(p: Float64, q: Float64) -> Float64:
    var pc = _clamp_prob(p)
    var qc = _clamp_prob(q)

    var one = 1.0
    var term1 = pc * (_log64(pc) - _log64(qc))
    var term2 = (one - pc) * (_log64(one - pc) - _log64(one - qc))
    var kl = term1 + term2
    if kl < 0.0: return 0.0
    return kl

fn kl_bernoulli_vector(p: List[Float64], q: List[Float64]) -> Float64:
    _assert_same_len(p, q)
    var n = Int(p.size())
    var acc = 0.0
    var i = 0
    while i < n:
        acc = acc + kl_bernoulli_scalar(p[i], q[i])
        i = i + 1
    if acc < 0.0: return 0.0
    return acc

# Typed overload using Bernoulli structs (expects field p: Float64)
fn kl_divergence(p: Bernoulli, q: Bernoulli) -> Float64:
    return kl_bernoulli_scalar(p.p, q.p)

# -----------------------------------------------------------------------------
# KL for Categorical (list)
# -----------------------------------------------------------------------------
# KL( Cat(p) || Cat(q) ) = sum_i p_i * log(p_i / q_i)

fn kl_categorical(p: List[Float64], q: List[Float64]) -> Float64:
    _assert_same_len(p, q)
    var n = Int(p.size())
    var acc = 0.0
    var i = 0
    while i < n:
        var pi = _clamp_prob(p[i])
        var qi = _clamp_prob(q[i])
        acc = acc + (pi * (_log64(pi) - _log64(qi)))
        i = i + 1
    if acc < 0.0: return 0.0
    return acc

# -----------------------------------------------------------------------------
# Generic shim (kept for backward compatibility)
# -----------------------------------------------------------------------------
# NOTE: Keep a distinct name to avoid overload ambiguity.
fn kl_divergence_any(p, q) -> Float64:
    # Prefer calling one of the typed or explicit helpers.
    return 0.0

# -----------------------------------------------------------------------------
#                             TENSOR VARIANTS
# -----------------------------------------------------------------------------
# All tensor variants use Float64 tensors and return either a scalar sum or a
# 1D tensor for 2D inputs (batch-wise reduction over the last dimension).
# -----------------------------------------------------------------------------

# --- Normal vs Normal (diagonal) ---

# Sum over all elements (any shape, shapes must match)
fn kl_normal_normal_diag_sum(
    mu_p: tensor.Tensor[Float64],
    logvar_p: tensor.Tensor[Float64],
    mu_q: tensor.Tensor[Float64],
    logvar_q: tensor.Tensor[Float64]
) -> Float64:
    _assert_same_shape_t(mu_p, mu_q)
    _assert_same_shape_t(logvar_p, logvar_q)
    _assert_same_shape_t(mu_p, logvar_p)

    var shp = mu_p.shape()
    var n = _numel(shp)

    var mup = mu_p._data
    var lvp = logvar_p._data
    var muq = mu_q._data
    var lvq = logvar_q._data

    var acc = 0.0
    var i = 0
    while i < n:
        var var_p = _exp64(lvp[i])
        var var_q = _exp64(lvq[i])
        var e = _eps()
        if var_p < e: var_p = e
        if var_q < e: var_q = e
        var term_log = lvq[i] - lvp[i]
        var md = mup[i] - muq[i]
        var term_frac = (var_p + md * md) / var_q
        var kl = 0.5 * (term_log + term_frac - 1.0)
        if kl > 0.0:
            acc = acc + kl
        i = i + 1
    return acc

# Batch-wise for 2D: inputs [N, D] -> returns [N], reduce over D
fn kl_normal_normal_diag_batched(
    mu_p: tensor.Tensor[Float64],
    logvar_p: tensor.Tensor[Float64],
    mu_q: tensor.Tensor[Float64],
    logvar_q: tensor.Tensor[Float64]
) -> tensor.Tensor[Float64]:
    _assert_same_shape_t(mu_p, mu_q)
    _assert_same_shape_t(logvar_p, logvar_q)
    _assert_same_shape_t(mu_p, logvar_p)

    var shp = mu_p.shape()
    assert(len(shp) == 2)
    var n = shp[0]; var d = shp[1]

    var row_kl = tensor.Tensor[Float64]([n, d], 0.0)
    var rdata = row_kl._data
    var mup = mu_p._data
    var lvp = logvar_p._data
    var muq = mu_q._data
    var lvq = logvar_q._data

    var i = 0
    var total = n * d
    while i < total:
        var var_p = _exp64(lvp[i])
        var var_q = _exp64(lvq[i])
        var e = _eps()
        if var_p < e: var_p = e
        if var_q < e: var_q = e
        var term_log = lvq[i] - lvp[i]
        var md = mup[i] - muq[i]
        var term_frac = (var_p + md * md) / var_q
        var kl = 0.5 * (term_log + term_frac - 1.0)
        if kl > 0.0:
            rdata[i] = kl
        else:
            rdata[i] = 0.0
        i = i + 1
    return _reduce_lastdim_sum_2d(row_kl)

# --- Bernoulli ---

# Sum over all elements
fn kl_bernoulli_sum(
    p: tensor.Tensor[Float64],
    q: tensor.Tensor[Float64]
) -> Float64:
    _assert_same_shape_t(p, q)
    var shp = p.shape()
    var n = _numel(shp)
    var pd = p._data; var qd = q._data
    var acc = 0.0
    var i = 0
    while i < n:
        var pc = _clamp_prob(pd[i])
        var qc = _clamp_prob(qd[i])
        var one = 1.0
        var term1 = pc * (_log64(pc) - _log64(qc))
        var term2 = (one - pc) * (_log64(one - pc) - _log64(one - qc))
        var kl = term1 + term2
        if kl > 0.0:
            acc = acc + kl
        i = i + 1
    return acc

# Batch-wise for 2D: [N, D] -> [N]
fn kl_bernoulli_batched(
    p2d: tensor.Tensor[Float64],
    q2d: tensor.Tensor[Float64]
) -> tensor.Tensor[Float64]:
    _assert_same_shape_t(p2d, q2d)
    var shp = p2d.shape()
    assert(len(shp) == 2)
    var n = shp[0]; var d = shp[1]

    var buf = tensor.Tensor[Float64]([n, d], 0.0)
    var bd = buf._data
    var pd = p2d._data; var qd = q2d._data

    var i = 0
    var total = n * d
    while i < total:
        var pc = _clamp_prob(pd[i])
        var qc = _clamp_prob(qd[i])
        var one = 1.0
        var term1 = pc * (_log64(pc) - _log64(qc))
        var term2 = (one - pc) * (_log64(one - pc) - _log64(one - qc))
        var kl = term1 + term2
        if kl > 0.0:
            bd[i] = kl
        else:
            bd[i] = 0.0
        i = i + 1
    return _reduce_lastdim_sum_2d(buf)

# --- Categorical ---

# Sum over all elements
fn kl_categorical_sum(
    p: tensor.Tensor[Float64],
    q: tensor.Tensor[Float64]
) -> Float64:
    _assert_same_shape_t(p, q)
    var shp = p.shape()
    var n = _numel(shp)
    var pd = p._data; var qd = q._data
    var acc = 0.0
    var i = 0
    while i < n:
        var pi = _clamp_prob(pd[i])
        var qi = _clamp_prob(qd[i])
        var kl = pi * (_log64(pi) - _log64(qi))
        if kl > 0.0:
            acc = acc + kl
        i = i + 1
    return acc

# Batch-wise for 2D: [N, D] -> [N]
fn kl_categorical_batched(
    p2d: tensor.Tensor[Float64],
    q2d: tensor.Tensor[Float64]
) -> tensor.Tensor[Float64]:
    _assert_same_shape_t(p2d, q2d)
    var shp = p2d.shape()
    assert(len(shp) == 2)
    var n = shp[0]; var d = shp[1]

    var buf = tensor.Tensor[Float64]([n, d], 0.0)
    var bd = buf._data
    var pd = p2d._data; var qd = q2d._data

    var i = 0
    var total = n * d
    while i < total:
        var pi = _clamp_prob(pd[i])
        var qi = _clamp_prob(qd[i])
        var kl = pi * (_log64(pi) - _log64(qi))
        if kl > 0.0:
            bd[i] = kl
        else:
            bd[i] = 0.0
        i = i + 1
    return _reduce_lastdim_sum_2d(buf)
