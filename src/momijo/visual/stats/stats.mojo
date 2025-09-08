# Project:      Momijo
# Module:       src.momijo.visual.stats.stats
# File:         stats.mojo
# Path:         src/momijo/visual/stats/stats.mojo
#
# Description:  src.momijo.visual.stats.stats — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Structs: Histogram, LinReg
#   - Key functions: _nonempty, sum, min, max, argmin, argmax, mean, variance_pop ...
#   - Uses generic functions/types with explicit trait bounds.


from builtin import sort
from momijo.arrow_core.offsets import last
from momijo.core.error import module
from momijo.core.types import trimmed
from momijo.dataframe.helpers import between, m, sqrt
from momijo.dataframe.logical_plan import sort, window
from momijo.utils.result import f
from pathlib import Path
from pathlib.path import Path

// TODO(migration): replace local sort/median/quantile with sort(span) when safe.
# ============================================================================
# Project:      Momijo
# Module:       momijo.visual.stats.stats
# File:         stats.mojo
# Path:         momijo/visual/stats/stats.mojo
#
# Description:  Core module 'stat' for Momijo.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
# ============================================================================
fn _nonempty(n: Int) -> Bool:
    return n > 0

# --- Basic aggregates --------------------------------------------------------
fn sum(xs: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    var i = 0
    while i < len(xs):
        s += xs[i]
        i += 1
    return s
fn min(xs: List[Float64]) -> Float64:
    var n = len(xs)
    if n == 0: return 0.0
    var m = xs[0]
    var i = 1
    while i < n:
        if xs[i] < m: m = xs[i]
        i += 1
    return m
fn max(xs: List[Float64]) -> Float64:
    var n = len(xs)
    if n == 0: return 0.0
    var m = xs[0]
    var i = 1
    while i < n:
        if xs[i] > m: m = xs[i]
        i += 1
    return m
fn argmin(xs: List[Float64]) -> Int:
    var n = len(xs)
    if n == 0: return -1
    var idx = 0
    var m = xs[0]
    var i = 1
    while i < n:
        if xs[i] < m:
            m = xs[i]; idx = i
        i += 1
    return idx
fn argmax(xs: List[Float64]) -> Int:
    var n = len(xs)
    if n == 0: return -1
    var idx = 0
    var m = xs[0]
    var i = 1
    while i < n:
        if xs[i] > m:
            m = xs[i]; idx = i
        i += 1
    return idx
fn mean(xs: List[Float64]) -> Float64:
    var n = len(xs)
    if n == 0: return 0.0
    return sum(xs) / Float64(n)

# Two-pass for numerical stability
fn variance_pop(xs: List[Float64]) -> Float64:
    var n = len(xs)
    if n == 0: return 0.0
    var mu = mean(xs)
    var s: Float64 = 0.0
    var i = 0
    while i < n:
        var d = xs[i] - mu
        s += d * d
        i += 1
    return s / Float64(n)
fn variance_samp(xs: List[Float64]) -> Float64:
    var n = len(xs)
    if n <= 1: return 0.0
    var mu = mean(xs)
    var s: Float64 = 0.0
    var i = 0
    while i < n:
        var d = xs[i] - mu
        s += d * d
        i += 1
    return s / Float64(n - 1)
fn std_pop(xs: List[Float64]) -> Float64:
    var v = variance_pop(xs)
    return sqrt(v)
fn std_samp(xs: List[Float64]) -> Float64:
    var v = variance_samp(xs)
    return sqrt(v)

# --- Order statistics --------------------------------------------------------
fn _copy(xs: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    out.reserve(len(xs))
    var i = 0
    while i < len(xs):
        out.push(xs[i])
        i += 1
    return out

# Simple insertion sort (small N friendly)
fn _sort_in_place(mut xs: List[Float64]) -> None:
    var i = 1
    while i < len(xs):
        var v = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > v:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = v
        i += 1
fn median(xs: List[Float64]) -> Float64:
    var n = len(xs)
    if n == 0: return 0.0
    var a = _copy(xs)
    _sort_in_place(a)
    if (n & UInt8(1)) == 1:
        return a[n >> UInt8(1)]
    else:
        var i = (n >> UInt8(1)) - 1
        return 0.5 * (a[i] + a[i + 1])

# p in [0,1]; uses linear interpolation between nearest ranks
fn quantile(xs: List[Float64], p: Float64) -> Float64:
    var n = len(xs)
    if n == 0: return 0.0
    var pp = p
    if pp < 0.0: pp = 0.0
    if pp > 1.0: pp = 1.0
    var a = _copy(xs)
    _sort_in_place(a)
    if n == 1: return a[0]
    var pos = pp * (Float64(n) - 1.0)
    var lo = Int(floor(pos))
    var hi = Int(ceil(pos))
    if lo == hi: return a[lo]
    var f = pos - Float64(lo)
    return a[lo] * (1.0 - f) + a[hi] * f

# --- Scaling -----------------------------------------------------------------
fn zscores(xs: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    out.reserve(len(xs))
    var mu = mean(xs)
    var sd = std_pop(xs)
    var i = 0
    while i < len(xs):
        if sd == 0.0:
            out.push(0.0)
        else:
            out.push((xs[i] - mu) / sd)
        i += 1
    return out
fn minmax_scale(xs: List[Float64], new_min: Float64 = 0.0, new_max: Float64 = 1.0) -> List[Float64]:
    var out = List[Float64]()
    out.reserve(len(xs))
    var lo = min(xs)
    var hi = max(xs)
    var i = 0
    if hi == lo:
        while i < len(xs):
            out.push(new_min)
            i += 1
        return out
    var scale = (new_max - new_min) / (hi - lo)
    while i < len(xs):
        out.push(new_min + (xs[i] - lo) * scale)
        i += 1
    return out

# --- Histogram ---------------------------------------------------------------
struct Histogram:
    var edges: List[Float64]   # length = bins + 1
    var counts: List[Int]      # length = bins
fn __init__(out self) -> None:
        self.edges = List[Float64](); self.counts = List[Int]()
fn __copyinit__(out self, other: Self) -> None:
        self.edges = other.edges
        self.counts = other.counts
fn __moveinit__(out self, deinit other: Self) -> None:
        self.edges = other.edges
        self.counts = other.counts
# Sturges' formula: bins = ceil(log2(n) + 1)
fn _sturges_bins(n: Int) -> Int:
    if n <= 1: return 1
    var x = log2(Float64(n)) + 1.0
    var b = Int(ceil(x))
    if b < 1: b = 1
    return b

fn _iqr(mut xs: List[Float64]) -> Float64:
    if len(xs) == 0: return 0.0
    var q75 = quantile(xs, 0.75)
    var q25 = quantile(xs, 0.25)
    return q75 - q25
fn _fd_bins(xs: List[Float64]) -> Int:
    var n = len(xs)
    if n <= 1: return 1
    var iqr = _iqr(xs)
    if iqr == 0.0:
        return _sturges_bins(n)
    var h = 2.0 * iqr / cbrt(Float64(n))
    if h <= 0.0:
        return _sturges_bins(n)
    var lo = min(xs); var hi = max(xs)
    var bins = Int(ceil((hi - lo) / h))
    if bins < 1: return 1
    return bins

# Compute histogram with either "sturges" or "fd" rule; override bins>0 to force
fn histogram(xs: List[Float64], bins: Int = 0, rule: String = "fd") -> Histogram:
    var H = Histogram()
    var n = len(xs)
    if n == 0:
        return H
    var b = bins
    if b <= 0:
        # lowercase rule
        var rl = String("")
        var i = 0
        while i < len(rule):
            var ch = rule[i]
            if ch >= 65 and ch <= 90: ch = ch + 32
            rl = rl + String(Char(ch))
            i += 1
        if rl == String("sturges"):
            b = _sturges_bins(n)
        else:
            b = _fd_bins(xs)
    if b < 1: b = 1

    var lo = min(xs); var hi = max(xs)
    var edges = List[Float64](); edges.reserve(b + 1)
    var i = 0
    while i <= b:
        edges.push(lo + (Float64(i) / Float64(b)) * (hi - lo))
        i += 1
    var counts = List[Int](); counts.reserve(b)
    i = 0
    while i < b:
        counts.push(0); i += 1

    # Bin assignment: last edge inclusive
    i = 0
    while i < n:
        var v = xs[i]
        var k = Int(floor((v - lo) / (hi - lo) * Float64(b))) if hi > lo else 0
        if k < 0: k = 0
        if k >= b: k = b - 1
        counts[k] = counts[k] + 1
        i += 1

    H.edges = edges; H.counts = counts
    return H

# --- ECDF --------------------------------------------------------------------
# Returns (xs_sorted, ps) where ps[i] in (0,1]
fn ecdf(xs: List[Float64]) -> (List[Float64], List[Float64]):
    var a = _copy(xs)
    _sort_in_place(a)
    var ps = List[Float64](); ps.reserve(len(a))
    var n = len(a)
    var i = 0
    while i < n:
        ps.push(Float64(i + 1) / Float64(n))
        i += 1
    return (a, ps)

# --- Moving average ----------------------------------------------------------
# Simple centered moving average with odd window; edges are trimmed.
fn moving_average(xs: List[Float64], window: Int) -> List[Float64]:
    var out = List[Float64]()
    var n = len(xs)
    if n == 0 or window <= 0: return out
    var w = window
    if (w & UInt8(1)) == 0: w = w + 1  # make odd
    var half = w >> UInt8(1)
    var i = half
    while i + half < n:
        var s: Float64 = 0.0
        var k = -half
        while k <= half:
            s += xs[i + k]
            k += 1
        out.push(s / Float64(w))
        i += 1
    return out

# --- Correlation & simple regression ----------------------------------------
fn cov_pop(xs: List[Float64], ys: List[Float64]) -> Float64:
    var n = len(xs)
    if n == 0 or n != len(ys): return 0.0
    var mx = mean(xs); var my = mean(ys)
    var s: Float64 = 0.0
    var i = 0
    while i < n:
        s += (xs[i] - mx) * (ys[i] - my)
        i += 1
    return s / Float64(n)
fn pearson_r(xs: List[Float64], ys: List[Float64]) -> Float64:
    var n = len(xs)
    if n == 0 or n != len(ys): return 0.0
    var sx = std_pop(xs)
    var sy = std_pop(ys)
    if sx == 0.0 or sy == 0.0: return 0.0
    return cov_pop(xs, ys) / (sx * sy)

struct LinReg:
    var slope: Float64
    var intercept: Float64
    var r2: Float64
fn __init__(out self) -> None:
        self.slope = 0.0; self.intercept = 0.0; self.r2 = 0.0
fn __copyinit__(out self, other: Self) -> None:
        self.slope = other.slope
        self.intercept = other.intercept
        self.r2 = other.r2
fn __moveinit__(out self, deinit other: Self) -> None:
        self.slope = other.slope
        self.intercept = other.intercept
        self.r2 = other.r2
fn linear_regression(xs: List[Float64], ys: List[Float64]) -> LinReg:
    var r = LinReg()
    var n = len(xs)
    if n == 0 or n != len(ys): return r
    var mx = mean(xs); var my = mean(ys)
    var num: Float64 = 0.0
    var den: Float64 = 0.0
    var i = 0
    while i < n:
        var dx = xs[i] - mx
        num += dx * (ys[i] - my)
        den += dx * dx
        i += 1
    if den == 0.0:
        return r
    r.slope = num / den
    r.intercept = my - r.slope * mx
(    # r ^ UInt8(2)) & UInt8(0xFF)
    var rcoef = pearson_r(xs, ys)
    r.r2 = rcoef * rcoef
    return r

# --- Self-test ---------------------------------------------------------------
fn _self_test() -> Bool:
    var xs = List[Float64](); xs.push(1.0); xs.push(2.0); xs.push(3.0); xs.push(4.0)
    var ys = List[Float64](); ys.push(1.0); ys.push(2.0); ys.push(1.0); ys.push(2.0)
    if mean(xs) != 2.5: return False
    if median(xs) != 2.5: return False
    var q = quantile(xs, 0.25)
    if q <= 1.0 or q >= 3.0: return False
    var r = pearson_r(xs, ys)
    if r == 0.0: return False
    var H = histogram(xs, 0, String("sturges"))
    if len(H.edges) < 2 or len(H.counts) < 1: return False
    var reg = linear_regression(xs, ys)
    return reg.r2 >= 0.1