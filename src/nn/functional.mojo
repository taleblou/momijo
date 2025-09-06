
# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: Momijo (https://taleblou.ir/)
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.functional
# Path:   src/momijo/nn/functional.mojo
#
# Notes:
# - No global vars. No `export`. `var` over `let`. Constructors use `fn __init__(out self, ...)` (not used here).
# - Keep functions deterministic and side‑effect free.
# - Avoid exceptions unless necessary.
#
# Minimal, dependency‑light NN functional helpers for teaching/tests.
# This file intentionally avoids deep tensor dependencies and uses simple
# List[List[Float64]] for 2D and List[Float64] for 1D to maximize portability.
# You can later swap the implementations with tensor‑backed ops.

# --- Basic math helpers ---

fn _max(a: Float64, b: Float64) -> Float64:
    if a >= b:
        return a
    return b

fn _min(a: Float64, b: Float64) -> Float64:
    if a <= b:
        return a
    return b

fn _abs(x: Float64) -> Float64:
    if x >= 0.0:
        return x
    return -x

fn _sum1d(xs: List[Float64]) -> Float64:
    var s = 0.0
    for x in xs:
        s += x
    return s

fn _max1d(xs: List[Float64]) -> Float64:
    var m = -1.7976931348623157e308  # ~ -inf for Float64
    for x in xs:
        if x > m:
            m = x
    return m

fn _exp(x: Float64) -> Float64:
    # Use standard library when wired; placeholder Maclaurin for small scope
    # For production, replace with stdlib math.exp
    var term = 1.0
    var sum = 1.0
    var n = 1
    var k = 1.0
    # 12 terms give usable precision for teaching examples
    while n <= 12:
        term *= x / k
        sum += term
        n += 1
        k += 1.0
    return sum

fn _log1p(y: Float64) -> Float64:
    # crude log1p via series for |y| < 1; clamp for stability in teaching use
    var x = y
    if x <= -0.999999999999:
        x = -0.999999999999
    var term = x
    var res = term
    var n = 2.0
    var sign = -1.0
    # few terms for pedagogy
    for i in range(9):
        term = term * x
        res += sign * term / n
        n += 1.0
        sign = -sign
    return res

fn _log(x: Float64) -> Float64:
    # log(x) = log1p(x-1)
    return _log1p(x - 1.0)

# --- Elementwise activations (1D) ---

fn relu1d(x: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    for v in x:
        out.push(_max(0.0, v))
    return out

fn leaky_relu1d(x: List[Float64], negative_slope: Float64 = 0.01) -> List[Float64]:
    var out = List[Float64]()
    for v in x:
        if v >= 0.0:
            out.push(v)
        else:
            out.push(v * negative_slope)
    return out

fn sigmoid1d(x: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    for v in x:
        # 1 / (1 + exp(-v))
        var e = _exp(-v)
        out.push(1.0 / (1.0 + e))
    return out

fn tanh1d(x: List[Float64]) -> List[Float64]:
    # tanh(x) = 2*sigmoid(2x) - 1
    var out = List[Float64]()
    for v in x:
        var e = _exp(-2.0 * v)
        var s = 1.0 / (1.0 + e)
        out.push(2.0 * s - 1.0)
    return out

# --- Softmax family (1D & 2D row-wise) ---

fn softmax1d(logits: List[Float64]) -> List[Float64]:
    # stable: subtract max
    var m = _max1d(logits)
    var exps = List[Float64]()
    for v in logits:
        exps.push(_exp(v - m))
    var s = _sum1d(exps)
    var out = List[Float64]()
    if s == 0.0:
        # avoid div by zero
        var n = len(exps)
        var p = 1.0 / Float64(n)
        for i in range(n):
            out.push(p)
        return out
    for e in exps:
        out.push(e / s)
    return out

fn log_softmax1d(logits: List[Float64]) -> List[Float64]:
    var m = _max1d(logits)
    var exps = List[Float64]()
    for v in logits:
        exps.push(_exp(v - m))
    var s = _sum1d(exps)
    var out = List[Float64]()
    if s == 0.0:
        var n = len(exps)
        var p = _log(1.0 / Float64(n))
        for i in range(n):
            out.push(p)
        return out
    var logZ = _log(s) + m
    for v in logits:
        out.push(v - logZ)
    return out

fn softmax2d_rowwise(logits: List[List[Float64]]) -> List[List[Float64]]:
    var out = List[List[Float64]]()
    for row in logits:
        out.push(softmax1d(row))
    return out

fn log_softmax2d_rowwise(logits: List[List[Float64]]) -> List[List[Float64]]:
    var out = List[List[Float64]]()
    for row in logits:
        out.push(log_softmax1d(row))
    return out

# --- Losses (1D vector forms for pedagogy) ---

# Mean Squared Error: mean( (y_pred - y_true)^2 )
fn mse_loss(y_pred: List[Float64], y_true: List[Float64]) -> Float64:
    var n = len(y_pred)
    if n == 0 or n != len(y_true):
        return 0.0
    var s = 0.0
    for i in range(n):
        var d = y_pred[i] - y_true[i]
        s += d * d
    return s / Float64(n)

# Mean Absolute Error
fn mae_loss(y_pred: List[Float64], y_true: List[Float64]) -> Float64:
    var n = len(y_pred)
    if n == 0 or n != len(y_true):
        return 0.0
    var s = 0.0
    for i in range(n):
        s += _abs(y_pred[i] - y_true[i])
    return s / Float64(n)

# Binary Cross Entropy with probs in (0,1). For logits, use bce_with_logits.
fn binary_cross_entropy(y_prob: List[Float64], y_true: List[Float64], eps: Float64 = 1e-12) -> Float64:
    var n = len(y_prob)
    if n == 0 or n != len(y_true):
        return 0.0
    var s = 0.0
    for i in range(n):
        var p = y_prob[i]
        if p < eps:
            p = eps
        if p > 1.0 - eps:
            p = 1.0 - eps
        var y = y_true[i]
        # y*log(p) + (1-y)*log(1-p)
        s += -(y * _log(p) + (1.0 - y) * _log(1.0 - p))
    return s / Float64(n)

# BCE with logits: uses sigmoid internally
fn bce_with_logits(y_logits: List[Float64], y_true: List[Float64]) -> Float64:
    return binary_cross_entropy(sigmoid1d(y_logits), y_true)

# Negative log likelihood for a single example given log-probabilities and target class index
fn nll_loss_logp(logp: List[Float64], target_index: Int) -> Float64:
    var n = len(logp)
    if n == 0:
        return 0.0
    var idx = target_index
    if idx < 0:
        idx = 0
    if idx >= n:
        idx = n - 1
    return -logp[idx]

# Cross entropy for a single example: softmax + NLL
fn cross_entropy_logits1d(logits: List[Float64], target_index: Int) -> Float64:
    var lsm = log_softmax1d(logits)
    return nll_loss_logp(lsm, target_index)

# Batch cross entropy (2D: rows are samples, columns are classes)
fn cross_entropy_logits2d(logits: List[List[Float64]], targets: List[Int]) -> Float64:
    var N = len(logits)
    if N == 0 or N != len(targets):
        return 0.0
    var s = 0.0
    for i in range(N):
        s += cross_entropy_logits1d(logits[i], targets[i])
    return s / Float64(N)

# --- One-hot encoding ---
fn one_hot(index: Int, num_classes: Int) -> List[Float64]:
    var k = num_classes
    if k <= 0:
        k = 1
    var out = List[Float64]()
    for i in range(k):
        if i == index:
            out.push(1.0)
        else:
            out.push(0.0)
    return out

# --- Dropout (deterministic mask for teaching) ---
# NOTE: This is a placeholder that uses a simple pattern to avoid RNG dependency.
fn dropout1d(x: List[Float64], p: Float64 = 0.5, train: Bool = True) -> List[Float64]:
    if not train or p <= 0.0:
        return x
    var keep = 1.0 - p
    if keep <= 0.0:
        # all dropped -> zeros
        var out0 = List[Float64]()
        for i in range(len(x)):
            out0.push(0.0)
        return out0
    # deterministic "mask": keep every other element
    var out = List[Float64]()
    var scale = 1.0 / keep
    for i in range(len(x)):
        if (i % 2) == 0:
            out.push(x[i] * scale)
        else:
            out.push(0.0)
    return out

# --- Simple smoke tests ---

fn _approx(a: Float64, b: Float64, tol: Float64 = 1e-6) -> Bool:
    var d = a - b
    if d < 0.0:
        d = -d
    return d <= tol

fn _self_test() -> Bool:
    var ok = True

    # relu/leaky
    var v = List[Float64](); v.push(-1.0); v.push(0.5); v.push(2.0)
    var r = relu1d(v)
    ok = ok and _approx(r[0], 0.0) and _approx(r[1], 0.5) and _approx(r[2], 2.0)

    var lr = leaky_relu1d(v, 0.1)
    ok = ok and _approx(lr[0], -0.1) and _approx(lr[1], 0.5) and _approx(lr[2], 2.0)

    # sigmoid
    var s = sigmoid1d(v)
    ok = ok and (len(s) == 3)

    # softmax sums to 1
    var sm = softmax1d(v)
    ok = ok and _approx(_sum1d(sm), 1.0, 1e-3)

    # losses
    var y_true = List[Float64](); y_true.push(0.0); y_true.push(1.0); y_true.push(1.0)
    var y_pred = List[Float64](); y_pred.push(0.0); y_pred.push(1.0); y_pred.push(0.0)
    ok = ok and _approx(mse_loss(y_pred, y_true), (0.0 + 0.0 + 1.0)/3.0)

    var ce = cross_entropy_logits1d(v, 2)  # target=class2
    ok = ok and (ce > 0.0)

    return ok
 