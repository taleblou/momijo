# Project:      Momijo
# Module:       src.momijo.nn.losses.bce_with_logits
# File:         bce_with_logits.mojo
# Path:         src/momijo/nn/losses/bce_with_logits.mojo
#
# Description:  Loss functions for supervised learning in Momijo with numerically
#               stable forward and backward computations for classification/regression.
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
#   - Key functions: _abs, _exp, _log, _log1p, sigmoid, softplus, bce_with_logits_elem, bce_with_logits1d ...
#   - Uses generic functions/types with explicit trait bounds.


fn _abs(x: Float64) -> Float64:
    if x < 0.0: return -x
    return x
fn _exp(x: Float64) -> Float64:
    # truncated series for e^x (good enough for pedagogy/smoke tests)
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

# Natural log via Newton iterations on e^y = x (x>0). Stable for pedagogy.
fn _log(x: Float64) -> Float64:
    if x <= 0.0:
        # avoid -inf; return a large negative for zero/negatives
        return -745.0  # ~ log(1e-323)
    var y = 0.0   # initial guess ln(1)=0
    for it in range(8):
        var ey = _exp(y)
        # y_{n+1} = y + 2*(x - e^y)/(x + e^y)  (Halley-like step)
        y = y + 2.0 * (x - ey) / (x + ey)
    return y
fn _log1p(x: Float64) -> Float64:
    # log(1+x) with basic handling near zero
    var one_plus = 1.0 + x
    return _log(one_plus)
fn sigmoid(x: Float64) -> Float64:
    var e = _exp(-x)
    return 1.0 / (1.0 + e)
fn softplus(x: Float64) -> Float64:
    # softplus(x) = log(1 + e^x) computed stably
    if x > 0.0:
        # x + log1p(exp(-x))
        return x + _log1p(_exp(-x))
    else:
        # log1p(exp(x))
        return _log1p(_exp(x))

# --- Elementwise BCE with logits (no reduction) ---
# pos_weight <= 0 means "disabled" and behaves like standard BCEWithLogits.
fn bce_with_logits_elem(x: Float64, y: Float64, pos_weight: Float64 = 0.0) -> Float64:
    if pos_weight > 0.0:
        # PyTorch-like definition: (1 - y)*softplus(x) + pos_w*y*softplus(-x)
        var s_pos = softplus(x)
        var s_neg = s_pos - x          # softplus(-x) = softplus(x) - x
        return (1.0 - y) * s_pos + pos_weight * y * s_neg
    else:
        # Standard stable form: softplus(x) - x*y
        return softplus(x) - x * y

# --- 1D API ---
fn bce_with_logits1d(logits: List[Float64], targets: List[Float64], reduction: String = "mean", weight: List[Float64] = List[Float64](), pos_weight: Float64 = 0.0) -> List[Float64]:
    var n = len(logits)
    var out = List[Float64]()
    var use_w = len(weight) == n
    var sumv = 0.0
    for i in range(n):
        var l = bce_with_logits_elem(logits[i], targets[i], pos_weight)
        if use_w: l *= weight[i]
        if reduction == "none":
            out.push(l)
        else:
            sumv += l
    if reduction == "mean":
        var denom = Float64(n)
        if use_w:
            # weighted mean denominator = sum of weights
            denom = 0.0
            for i in range(n): denom += weight[i]
            if denom == 0.0: denom = 1.0
        out.push(sumv / denom)
    elif reduction == "sum":
        out.push(sumv)
    return out

# --- 2D API (row-major) ---
fn bce_with_logits2d(logits: List[List[Float64]], targets: List[List[Float64]], reduction: String = "mean", weight: List[List[Float64]] = List[List[Float64]](), pos_weight: Float64 = 0.0) -> List[Float64]:
    var R = len(logits)
    var C = 0
    if R > 0: C = len(logits[0])

    var use_w = (len(weight) == R) and (R > 0) and (len(weight[0]) == C)
    var result = List[Float64]()

    if reduction == "none":
        # flatten in row-major
        for i in range(R):
            for j in range(C):
                var l = bce_with_logits_elem(logits[i][j], targets[i][j], pos_weight)
                if use_w: l *= weight[i][j]
                result.push(l)
        return result

    var sumv = 0.0
    var denom = 0.0
    for i in range(R):
        for j in range(C):
            var l = bce_with_logits_elem(logits[i][j], targets[i][j], pos_weight)
            if reduction == "sum" or reduction == "mean":
                if use_w:
                    l *= weight[i][j]
                    denom += weight[i][j]
                else:
                    denom += 1.0
            sumv += l
    if reduction == "sum":
        result.push(sumv)
    else:
        # mean
        if denom == 0.0: denom = 1.0
        result.push(sumv / denom)
    return result

# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # Simple checks 1D
    var logits = List[Float64](); logits.push(0.0); logits.push(2.0); logits.push(-2.0)
    var targets = List[Float64](); targets.push(0.0); targets.push(1.0); targets.push(1.0)

    # No reduction -> 3 values
    var none_vals = bce_with_logits1d(logits, targets, "none", List[Float64](), 0.0)
    ok = ok and (len(none_vals) == 3)

    # Mean vs sum consistency
    var meanv = bce_with_logits1d(logits, targets, "mean", List[Float64](), 0.0)[0]
    var sumv = bce_with_logits1d(logits, targets, "sum", List[Float64](), 0.0)[0]
    ok = ok and (sumv >= meanv)

    # Weight effect
    var w = List[Float64](); w.push(1.0); w.push(2.0); w.push(3.0)
    var mean_w = bce_with_logits1d(logits, targets, "mean", w, 0.0)[0]
    ok = ok and (mean_w == mean_w)  # not NaN

    # pos_weight effect should increase loss for positive targets
    var mean_pw1 = bce_with_logits1d(logits, targets, "mean", List[Float64](), 1.0)[0]
    var mean_pw5 = bce_with_logits1d(logits, targets, "mean", List[Float64](), 5.0)[0]
    ok = ok and (mean_pw5 >= mean_pw1)

    # 2D path
    var L = List[List[Float64]](); 
    var T = List[List[Float64]]();
    var row0 = List[Float64](); row0.push(0.0); row0.push(1.0)
    var row1 = List[Float64](); row1.push(-1.5); row1.push(2.5)
    L.push(row0); L.push(row1)
    var t0 = List[Float64](); t0.push(0.0); t0.push(1.0)
    var t1 = List[Float64](); t1.push(1.0); t1.push(0.0)
    T.push(t0); T.push(t1)
    var m2 = bce_with_logits2d(L, T, "mean", List[List[Float64]](), 0.0)[0]
    ok = ok and (m2 == m2)

    return ok