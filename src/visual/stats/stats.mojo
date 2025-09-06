# ============================================================================
#  Momijo Visualization - stats/stats.mojo
#  Copyright (c) 2025  Morteza Talebou
#  MIT License - https://taleblou.ir/
#  This file follows the user's Mojo checklist: no global/export; __init__(out self,...); var only.
# ============================================================================

struct HistBin:
    var left: Float64
    var right: Float64
    var count: Int
    fn __init__(out self, left: Float64, right: Float64, count: Int):
        self.left = left; self.right = right; self.count = count

fn min_max(xs: List[Float64]) -> (Float64, Float64):
    var n = len(xs)
    if n == 0: return (0.0,1.0)
    var mn = xs[0]; var mx = xs[0]
    var i = 1
    while i < n:
        var v = xs[i]
        if v < mn: mn = v
        if v > mx: mx = v
        i += 1
    if mn == mx: mx = mn + 1.0
    return (mn,mx)

fn histogram(xs: List[Float64], bins: Int) -> List[HistBin]:
    var (mn,mx) = min_max(xs)
    var out = List[HistBin]()
    if bins <= 0: bins = 10
    var width = (mx - mn) / Float64(bins)
    var counts = List[Int]()
    counts.reserve(bins)
    var i = 0
    while i < bins:
        counts.append(0)
        i += 1
    i = 0
    while i < len(xs):
        var v = xs[i]
        var b = Int((v - mn) / width)
        if b < 0: b = 0
        if b >= bins: b = bins - 1
        counts[b] = counts[b] + 1
        i += 1
    i = 0
    while i < bins:
        var l = mn + Float64(i) * width
        var r = l + width
        out.append(HistBin(l,r,counts[i]))
        i += 1
    return out

fn gaussian_kde(xs: List[Float64], x: List[Float64], bw: Float64) -> List[Float64]:
    var n = len(xs)
    var out = List[Float64]()
    var inv = 1.0 / (bw * 2.5066282746310002)  # sqrt(2π)
    var i = 0
    while i < len(x):
        var xi = x[i]
        var s = 0.0
        var j = 0
        while j < n:
            var z = (xi - xs[j]) / bw
            s += exp(-0.5 * z * z)
            j += 1
        out.append(s * inv / Float64(n))
        i += 1
    return out

fn linear_regression(xs: List[Float64], ys: List[Float64]) -> (Float64, Float64):
    var n = len(xs)
    if n == 0: return (0.0, 0.0)
    var sx = 0.0; var sy = 0.0; var sxx = 0.0; var sxy = 0.0
    var i = 0
    while i < n:
        var x = xs[i]; var y = ys[i]
        sx += x; sy += y; sxx += x*x; sxy += x*y
        i += 1
    var denom = Float64(n) * sxx - sx * sx
    if denom == 0.0: return (0.0, sy / Float64(n))
    var m = (Float64(n) * sxy - sx * sy) / denom
    var b = (sy - m * sx) / Float64(n)
    return (m, b)

fn error_bars(mean: List[Float64], std: List[Float64]) -> (List[Float64], List[Float64]):
    var lo = List[Float64](); var hi = List[Float64]()
    var n = len(mean); var i = 0
    while i < n:
        lo.append(mean[i] - std[i])
        hi.append(mean[i] + std[i])
        i += 1
    return (lo, hi)
