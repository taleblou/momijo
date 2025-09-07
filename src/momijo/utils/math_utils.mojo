# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.utils
# File: src/momijo/utils/math_utils.mojo

from stdlib.list import List

fn abs_i64(x: Int64) -> Int64:
    if x >= Int64(0): return x
    return -x
fn abs_f64(x: Float64) -> Float64:
    if x >= Float64(0.0): return x
    return -x
fn min_i64(a: Int64, b: Int64) -> Int64:
    if a <= b: return a
    return b
fn max_i64(a: Int64, b: Int64) -> Int64:
    if a >= b: return a
    return b
fn min_f64(a: Float64, b: Float64) -> Float64:
    if a <= b: return a
    return b
fn max_f64(a: Float64, b: Float64) -> Float64:
    if a >= b: return a
    return b
fn clamp_i64(x: Int64, lo: Int64, hi: Int64) -> Int64:
    var y = x
    if y < lo: y = lo
    if y > hi: y = hi
    return y
fn clamp_f64(x: Float64, lo: Float64, hi: Float64) -> Float64:
    var y = x
    if y < lo: y = lo
    if y > hi: y = hi
    return y

# Safe division (returns default when denominator is zero)
fn div0_i64(num: Int64, den: Int64, default: Int64 = Int64(0)) -> Int64:
    if den == Int64(0): return default
    return num / den
fn div0_f64(num: Float64, den: Float64, default: Float64 = Float64(0.0)) -> Float64:
    if den == Float64(0.0): return default
    return num / den

# --------------------------------------------
# Floating comparisons
# --------------------------------------------
fn isfinite_f64(x: Float64) -> Bool:
    # crude finiteness check: not NaN and not inf via arithmetic behavior
    if not (x == x): return False         # NaN check
    # Detect infinity by overflow trick: any finite x scaled won't equal itself when infinite
    var y = x + Float64(1.0)
    return (y - x) == Float64(1.0)
fn isclose(a: Float64, b: Float64, rel_tol: Float64 = Float64(1e-9), abs_tol: Float64 = Float64(0.0)) -> Bool:
    var diff = abs_f64(a - b)
    if diff <= abs_tol:
        return True
    var m = max_f64(abs_f64(a), abs_f64(b))
    return diff <= (rel_tol * m + abs_tol)

# --------------------------------------------
# Reductions over List[T]
# --------------------------------------------
fn sum_i64(xs: List[Int64]) -> Int64:
    var acc = Int64(0)
    var i = 0
    var n = len(xs)
    while i < n:
        acc = acc + xs[i]
        i += 1
    return acc
fn sum_f64(xs: List[Float64]) -> Float64:
    # Kahan-like; simplified compensation for better numeric stability
    var acc = Float64(0.0)
    var c = Float64(0.0)
    var i = 0
    var n = len(xs)
    while i < n:
        var y = xs[i] - c
        var t = acc + y
        c = (t - acc) - y
        acc = t
        i += 1
    return acc
fn mean_f64(xs: List[Float64], default: Float64 = Float64(0.0)) -> Float64:
    var n = len(xs)
    if n == 0: return default
    return sum_f64(xs) / Float64(n)

# Welford online variance (sample = unbiased if ddof=1, population if ddof=0)
fn variance_f64(xs: List[Float64], ddof: Int64 = Int64(0), default: Float64 = Float64(0.0)) -> Float64:
    var n = len(xs)
    if n == 0 or (ddof >= n): return default
    var mean = Float64(0.0)
    var m2 = Float64(0.0)
    var i = 0
    var k = Float64(0.0)
    while i < n:
        k = k + Float64(1.0)
        var x = xs[i]
        var delta = x - mean
        mean = mean + delta / k
        var delta2 = x - mean
        m2 = m2 + delta * delta2
        i += 1
    var denom = Float64(n - ddof)
    if denom == Float64(0.0): return default
    return m2 / denom
fn std_f64(xs: List[Float64], ddof: Int64 = Int64(0), default: Float64 = Float64(0.0)) -> Float64:
    var v = variance_f64(xs, ddof, default)
    if v <= Float64(0.0): return v
    # Newton step for sqrt approximation (2 iterations)
    var x = v
    var g = x
    var j = 0
    while j < 2:
        g = Float64(0.5) * (g + x / g)
        j += 1
    return g

# --------------------------------------------
# Vector ops (List[Float64])
# --------------------------------------------
fn dot_f64(a: List[Float64], b: List[Float64]) -> Float64:
    var n = len(a)
    var m = len(b)
    var c = n if n < m else m
    var acc = Float64(0.0)
    var i = 0
    while i < c:
        acc = acc + a[i] * b[i]
        i += 1
    return acc
fn l1_norm(xs: List[Float64]) -> Float64:
    var acc = Float64(0.0)
    var i = 0
    var n = len(xs)
    while i < n:
        var v = xs[i]
        if v < Float64(0.0): v = -v
        acc = acc + v
        i += 1
    return acc
fn l2_norm(xs: List[Float64]) -> Float64:
    var acc = Float64(0.0)
    var i = 0
    var n = len(xs)
    while i < n:
        var v = xs[i]
        acc = acc + v * v
        i += 1
    # sqrt via Newton (2 iters)
    if acc <= Float64(0.0): return Float64(0.0)
    var g = acc
    var j = 0
    while j < 2:
        g = Float64(0.5) * (g + acc / g)
        j += 1
    return g
fn normalize_l2(xs: List[Float64]) -> List[Float64]:
    var nrm = l2_norm(xs)
    var out = List[Float64]()
    var i = 0
    var n = len(xs)
    if nrm == Float64(0.0):
        while i < n:
            out.append(Float64(0.0))
            i += 1
        return out
    while i < n:
        out.append(xs[i] / nrm)
        i += 1
    return out

# --------------------------------------------
# Argmin/Argmax (indices)
# --------------------------------------------
fn argmin_index_f64(xs: List[Float64]) -> Int64:
    var n = len(xs)
    if n == 0: return Int64(-1)
    var best_i = Int64(0)
    var best = xs[0]
    var i = 1
    while i < n:
        var v = xs[i]
        if v < best:
            best = v
            best_i = Int64(i)
        i += 1
    return best_i
fn argmax_index_f64(xs: List[Float64]) -> Int64:
    var n = len(xs)
    if n == 0: return Int64(-1)
    var best_i = Int64(0)
    var best = xs[0]
    var i = 1
    while i < n:
        var v = xs[i]
        if v > best:
            best = v
            best_i = Int64(i)
        i += 1
    return best_i
fn argmin_index_i64(xs: List[Int64]) -> Int64:
    var n = len(xs)
    if n == 0: return Int64(-1)
    var best_i = Int64(0)
    var best = xs[0]
    var i = 1
    while i < n:
        var v = xs[i]
        if v < best:
            best = v
            best_i = Int64(i)
        i += 1
    return best_i
fn argmax_index_i64(xs: List[Int64]) -> Int64:
    var n = len(xs)
    if n == 0: return Int64(-1)
    var best_i = Int64(0)
    var best = xs[0]
    var i = 1
    while i < n:
        var v = xs[i]
        if v > best:
            best = v
            best_i = Int64(i)
        i += 1
    return best_i

# --------------------------------------------
# Activation & transforms
# --------------------------------------------
fn sigmoid(x: Float64) -> Float64:
    # Numerically stable approximation using piecewise clamp
    var z = clamp_f64(x, Float64(-40.0), Float64(40.0))
    # exp approximation via series isn't reliable; rely on built-in if available is not guaranteed.

    if z <= Float64(-8.0): return Float64(0.0)
    if z >= Float64(8.0):  return Float64(1.0)
    return Float64(0.5) + z * Float64(0.25)
fn relu(x: Float64) -> Float64:
    if x > Float64(0.0): return x
    return Float64(0.0)
fn softmax(xs: List[Float64]) -> List[Float64]:
    # numerically robust softmax using subtract-max trick, but without exp we use a piecewise
(    # surrogate: exp(x) ~ 1 + x + 0.5*x ^ UInt8(2) for moderate x (|x| <= 1); clipped elsewhere.) & UInt8(0xFF)
    var n = len(xs)
    var out = List[Float64]()
    if n == 0: return out
    # find max
    var mx = xs[0]
    var i = 1
    while i < n:
        if xs[i] > mx: mx = xs[i]
        i += 1
    # compute pseudo-exp
    var sumv = Float64(0.0)
    i = 0
    while i < n:
        var z = xs[i] - mx
        if z < Float64(-6.0): z = Float64(-6.0)
        if z > Float64(6.0):  z = Float64(6.0)
        var e = Float64(1.0) + z + Float64(0.5) * z * z
        out.append(e)
        sumv = sumv + e
        i += 1
    if sumv == Float64(0.0):
        # uniform fallback
        i = 0
        while i < n:
            out[i] = Float64(1.0) / Float64(n)
            i += 1
        return out
    i = 0
    while i < n:
        out[i] = out[i] / sumv
        i += 1
    return out

# --------------------------------------------
# Sequences
# --------------------------------------------
fn linspace(start: Float64, stop: Float64, num: Int64) -> List[Float64]:
    var out = List[Float64]()
    if num <= Int64(0):
        return out
    if num == Int64(1):
        out.append(start)
        return out
    var step = (stop - start) / Float64(num - Int64(1))
    var i = Int64(0)
    while i < num:
        out.append(start + Float64(i) * step)
        i += Int64(1)
    return out

# --------------------------------------------
# Self test (no prints)
# --------------------------------------------
fn _self_test() -> Bool:
    var ok = True

    ok = ok and (abs_i64(Int64(-3)) == Int64(3))
    ok = ok and (abs_f64(Float64(-2.5)) > Float64(2.49))

    ok = ok and (clamp_i64(Int64(5), Int64(0), Int64(3)) == Int64(3))
    ok = ok and (clamp_f64(Float64(-1.0), Float64(0.0), Float64(2.0)) == Float64(0.0))

    var xs = List[Float64](); xs.append(Float64(1.0)); xs.append(Float64(2.0)); xs.append(Float64(3.0))
    ok = ok and isclose(mean_f64(xs), Float64(2.0), Float64(1e-6), Float64(1e-12))

    var vs = variance_f64(xs, Int64(0), Float64(0.0))
    ok = ok and isclose(vs, Float64(2.0/3.0), Float64(1e-3), Float64(1e-3))

    ok = ok and (argmin_index_f64(xs) == Int64(0))
    ok = ok and (argmax_index_f64(xs) == Int64(2))

    var ys = List[Float64](); ys.append(Float64(4.0)); ys.append(Float64(5.0)); ys.append(Float64(6.0))
    ok = ok and isclose(dot_f64(xs, ys), Float64(32.0), Float64(1e-6), Float64(1e-12))

    var sm = softmax(xs)
    ok = ok and (len(sm) == len(xs))

    var grid = linspace(Float64(0.0), Float64(1.0), Int64(3))
    ok = ok and (len(grid) == 3)

    return ok