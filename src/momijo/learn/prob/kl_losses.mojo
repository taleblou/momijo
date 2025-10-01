# Project:      Momijo
# Module:       learn.prob.kl_losses
# File:         prob/kl_losses.mojo
# Path:         src/momijo/learn/prob/kl_losses.mojo
#
# Description:  Kullback–Leibler divergence (KL) utilities for probabilistic losses.
#               Provides KL for Normal (scalar/vector/diagonal), Bernoulli (scalar/vector),
#               and a generic categorical KL from probability lists. Results are in nats.
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
#   - Functions (all return Float64 unless noted):
#     * kl_divergence(p: Normal, q: Normal)
#     * kl_normal_normal_scalar(mu_p, logvar_p, mu_q, logvar_q)
#     * kl_normal_normal_diag(mu_p: List[Float64], logvar_p: List[Float64],
#                             mu_q: List[Float64], logvar_q: List[Float64]) -> Float64
#     * kl_bernoulli_scalar(p: Float64, q: Float64)
#     * kl_bernoulli_vector(p: List[Float64], q: List[Float64]) -> Float64
#     * kl_categorical(p: List[Float64], q: List[Float64]) -> Float64
#   - Helpers: _eps(), _clamp_prob(x), _assert_same_len(a,b)

from collections.list import List
from momijo.learn.prob.distributions import Normal
from momijo.learn.prob.distributions import Bernoulli

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@staticmethod
fn _eps() -> Float64:
    # Minimum positive for probability clamping
    return 1e-12

fn _clamp_prob(x: Float64) -> Float64:
    var e = _eps()
    var one = 1.0 - e
    if x < e:
        return e
    if x > one:
        return one
    return x

fn _assert_same_len(a: List[Float64], b: List[Float64]):
    assert(Int(a.size()) == Int(b.size()))

# -----------------------------------------------------------------------------
# KL for Normal distributions
# -----------------------------------------------------------------------------
# KL( N(mu_p, var_p) || N(mu_q, var_q) ) =
#   0.5 * [ log(var_q / var_p) + (var_p + (mu_p - mu_q)^2) / var_q - 1 ]
# We use log-variance (logvar = log(var)) to stay numerically stable.

fn kl_normal_normal_scalar(mu_p: Float64, logvar_p: Float64,
                           mu_q: Float64, logvar_q: Float64) -> Float64:
    var var_p = exp(logvar_p)
    var var_q = exp(logvar_q)

    # Avoid degenerate variance
    var e = _eps()
    if var_p < e:
        var_p = e
    if var_q < e:
        var_q = e

    var term_log = logvar_q - logvar_p
    var mean_diff = mu_p - mu_q
    var term_frac = (var_p + mean_diff * mean_diff) / var_q
    var kl = 0.5 * (term_log + term_frac - 1.0)
    if kl < 0.0:
        # Numerical guard: KL should be >= 0
        return 0.0
    return kl

# Diagonal multivariate case: sum over dimensions
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
    return acc

# Overload using distribution structs
fn kl_divergence(p: Normal, q: Normal) -> Float64:
    # Expected Normal struct interface:
    #   fields (or accessors) mean: Float64, std: Float64
    # Convert to (mu, logvar)
    var mu_p = p.mean
    var mu_q = q.mean

    var std_p = p.std
    var std_q = q.std

    # Guard std -> logvar
    var e = _eps()
    if std_p < e:
        std_p = e
    if std_q < e:
        std_q = e

    var logvar_p = 2.0 * log(std_p)
    var logvar_q = 2.0 * log(std_q)
    return kl_normal_normal_scalar(mu_p, logvar_p, mu_q, logvar_q)

# -----------------------------------------------------------------------------
# KL for Bernoulli distributions
# -----------------------------------------------------------------------------
# KL( Ber(p) || Ber(q) ) = p*log(p/q) + (1-p)*log((1-p)/(1-q))

fn kl_bernoulli_scalar(p: Float64, q: Float64) -> Float64:
    var pc = _clamp_prob(p)
    var qc = _clamp_prob(q)

    var one = 1.0
    var term1 = pc * log(pc / qc)
    var term2 = (one - pc) * log((one - pc) / (one - qc))
    var kl = term1 + term2
    if kl < 0.0:
        return 0.0
    return kl

fn kl_bernoulli_vector(p: List[Float64], q: List[Float64]) -> Float64:
    _assert_same_len(p, q)
    var n = Int(p.size())
    var acc = 0.0
    var i = 0
    while i < n:
        acc = acc + kl_bernoulli_scalar(p[i], q[i])
        i = i + 1
    return acc

# Overload using distribution structs
fn kl_divergence(p: Bernoulli, q: Bernoulli) -> Float64:
    # Expected Bernoulli struct interface:
    #   field p: Float64 (success probability)
    return kl_bernoulli_scalar(p.p, q.p)

# -----------------------------------------------------------------------------
# KL for Categorical distributions (generic, via probability lists)
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
        acc = acc + (pi * log(pi / qi))
        i = i + 1
    if acc < 0.0:
        return 0.0
    return acc

# -----------------------------------------------------------------------------
# Generic shim (kept for backward compatibility with skeleton):
# This fallback returns 0.0; prefer the typed overloads above.
# -----------------------------------------------------------------------------

fn kl_divergence(p, q) -> Float64:
    # Prefer calling one of:
    #   - kl_normal_normal_scalar / kl_normal_normal_diag
    #   - kl_bernoulli_scalar / kl_bernoulli_vector
    #   - kl_categorical
    # or typed overloads: kl_divergence(Normal, Normal), kl_divergence(Bernoulli, Bernoulli)
    return 0.0
