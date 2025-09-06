# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.mi_losses
# Path:   src/momijo/nn/mi_losses.mojo
#
# Mutual-information related utility losses for educational GANs (InfoGAN, etc.).
# Dependency-light (List[Float64]), suitable for smoke tests and teaching.
#
# Style:
# - No global vars, no `export`. Use `var` (not `let`).
# - Constructors (if any) use `fn __init__(out self, ...)`.
# - Prefer `mut/out` over `inout`.

# --- Helpers ---
fn _sum1d(xs: List[Float64]) -> Float64:
    var s = 0.0
    for v in xs: s += v
    return s

fn _max1d(xs: List[Float64]) -> Float64:
    var m = -1.7976931348623157e308
    for v in xs:
        if v > m: m = v
    return m

fn _exp(x: Float64) -> Float64:
    var term = 1.0
    var sum = 1.0
    var n = 1
    var k = 1.0
    while n <= 12:
        term *= x / k
        sum += term
        n += 1
        k += 1.0
    return sum

fn _log1p(y: Float64) -> Float64:
    var x = y
    if x <= -0.999999999999: x = -0.999999999999
    var term = x
    var res = term
    var n = 2.0
    var sign = -1.0
    for i in range(9):
        term = term * x
        res += sign * term / n
        n += 1.0
        sign = -sign
    return res

fn _log(x: Float64) -> Float64:
    return _log1p(x - 1.0)

fn softmax1d(logits: List[Float64]) -> List[Float64]:
    var m = _max1d(logits)
    var exps = List[Float64]()
    for v in logits: exps.push(_exp(v - m))
    var s = _sum1d(exps)
    var out = List[Float64]()
    if s == 0.0:
        var n = len(exps)
        var p = 1.0 / Float64(n)
        for i in range(n): out.push(p)
        return out
    for e in exps: out.push(e / s)
    return out

# --- Losses ---
# Cross entropy for a single categorical target index against logits
fn cross_entropy_logits1d(logits: List[Float64], target_index: Int) -> Float64:
    var n = len(logits)
    if n == 0: return 0.0
    var m = _max1d(logits)
    var exps = List[Float64]()
    for v in logits: exps.push(_exp(v - m))
    var s = _sum1d(exps)
    if s == 0.0: return 0.0
    var logZ = _log(s) + m
    var idx = target_index
    if idx < 0: idx = 0
    if idx >= n: idx = n - 1
    return -(logits[idx] - logZ)

# Categorical cross entropy for probability vector p (target probs) vs q_logits (pred logits)
# H(p, q) = - sum_i p_i * log softmax(q)_i
fn cross_entropy_prob_p_vs_logits_q(p: List[Float64], q_logits: List[Float64]) -> Float64:
    var q_logZ: Float64
    var m = _max1d(q_logits)
    var exps = List[Float64]()
    for v in q_logits: exps.push(_exp(v - m))
    var s = _sum1d(exps)
    if s == 0.0: return 0.0
    q_logZ = _log(s) + m
    var ce = 0.0
    var n = len(p)
    for i in range(n):
        ce += p[i] * (q_logZ - q_logits[i])  # -p_i * log softmax_i
    return ce

# Gaussian negative log-likelihood for scalar c ~ N(mu, sigma^2) with logvar=log sigma^2
fn gaussian_nll(mu: Float64, logvar: Float64, c: Float64) -> Float64:
    var inv_var = _exp(-logvar)
    var diff = c - mu
    # Drop constant term 0.5*log(2*pi) for pedagogy
    return 0.5 * (logvar + diff * diff * inv_var)

# Batch mean of Gaussian NLL over vectors
fn gaussian_nll_vec(mu: List[Float64], logvar: List[Float64], c: List[Float64]) -> Float64:
    var n = len(c)
    if n == 0 or n != len(mu) or n != len(logvar): return 0.0
    var s = 0.0
    for i in range(n):
        s += gaussian_nll(mu[i], logvar[i], c[i])
    return s / Float64(n)

# Mutual information lower bound (InfoGAN-style) per sample:
# Given cat target index y, predicted cat logits q_cat_logits,
# and continuous codes c with predicted (mu, logvar),
# L_mi = CE(y, q_cat_logits) + mean GaussianNLL(c | mu, logvar)
fn mi_lower_bound(q_cat_logits: List[Float64], y_index: Int, mu: List[Float64], logvar: List[Float64], c: List[Float64]) -> Float64:
    var ce = cross_entropy_logits1d(q_cat_logits, y_index)
    var nll = gaussian_nll_vec(mu, logvar, c)
    return ce + nll

# Optional: KL divergence between categorical p and q (where q provided as logits)
# KL(p || q) = sum_i p_i * (log p_i - log softmax(q)_i)
fn kl_categorical_p_vs_logits_q(p: List[Float64], q_logits: List[Float64]) -> Float64:
    var n = len(p)
    if n == 0 or n != len(q_logits): return 0.0
    var m = _max1d(q_logits)
    var exps = List[Float64]()
    for v in q_logits: exps.push(_exp(v - m))
    var s = _sum1d(exps)
    if s == 0.0: return 0.0
    var logZ = _log(s) + m
    var kl = 0.0
    for i in range(n):
        var pi = p[i]
        if pi <= 0.0: 
            continue
        kl += pi * (_log(pi) - (q_logits[i] - logZ))
    return kl

# --- Smoke tests ---
fn _approx(a: Float64, b: Float64, tol: Float64 = 1e-6) -> Bool:
    var d = a - b
    if d < 0.0: d = -d
    return d <= tol

fn _self_test() -> Bool:
    var ok = True

    # CE with one-hot (index 2)
    var logits = List[Float64](); logits.push(0.1); logits.push(0.2); logits.push(1.7)
    var ce = cross_entropy_logits1d(logits, 2)
    ok = ok and (ce >= 0.0)

    # MI lower bound
    var mu = List[Float64](); mu.push(0.0); mu.push(0.5)
    var lv = List[Float64](); lv.push(0.0); lv.push(0.0)
    var c = List[Float64](); c.push(0.0); c.push(1.0)
    var mi = mi_lower_bound(logits, 1, mu, lv, c)
    ok = ok and (mi >= 0.0)

    # KL categorical (p vs logits q)
    var p = List[Float64](); p.push(0.0); p.push(1.0); p.push(0.0)
    var kl = kl_categorical_p_vs_logits_q(p, logits)
    ok = ok and (kl >= 0.0)

    return ok 
