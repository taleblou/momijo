# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.visual
# File: src/momijo/visual/stats/violin_utils.mojo
# Description: Utilities for building violin curves (histogram-based) and simple list passthrough.

# Returns the same list (handy to keep a uniform callsite).
fn _ints(xs: List[Int]) -> List[Int]:
    return xs

# Build a histogram-smoothed violin curve from integer samples.
# - samples: raw values
# - bin_count: number of bins to aggregate along the value axis (>=2 recommended)
# Returns a JSON string of pairs: [[y0, widthPct0], [y1, widthPct1], ...]
# Where:
#   - yN is the bin-center in the original value space (Int)
#   - widthPctN is 0..100 (Int), normalized relative width for that y
fn _compute_violin_curve_json(samples: List[Int], bin_count: Int) -> String:
    var n = len(samples)
    if n == 0:
        return String("[]")

    # Min/Max
    var minv = samples[0]
    var maxv = samples[0]
    var i = 1
    while i < n:
        var v = samples[i]
        if v < minv: minv = v
        if v > maxv: maxv = v
        i += 1
    if maxv == minv:
        maxv = minv + 1  # avoid zero range

    # Bins
    var bc = bin_count if bin_count > 1 else 10
    var bins = List[Int]()
    var j = 0
    while j < bc:
        bins.append(0)
        j += 1

    # Assign samples to bins
    i = 0
    while i < n:
        var v2 = samples[i]
        var idx = ((v2 - minv) * bc) // (maxv - minv + 1)
        if idx >= bc: idx = bc - 1
        if idx < 0:   idx = 0
        bins[idx] = bins[idx] + 1
        i += 1

    # 3-point moving average smoothing (single pass)
    var smooth = List[Int]()
    j = 0
    while j < bc:
        var s = bins[j]
        if j > 0:     s += bins[j-1]
        if j+1 < bc:  s += bins[j+1]
        var denom = 1
        if j > 0:     denom += 1
        if j+1 < bc:  denom += 1
        smooth.append(s // denom)
        j += 1

    # Normalize to width percent (0..100)
    var maxc = 1
    j = 0
    while j < bc:
        if smooth[j] > maxc: maxc = smooth[j]
        j += 1
    if maxc == 0: maxc = 1

    # Build pairs: [y_center, widthPercent]
    var out = String("[")
    j = 0
    while j < bc:
        var y_center = minv + (((j * 2 + 1) * (maxv - minv)) // (2 * bc))
        var w_pct    = (smooth[j] * 100) // maxc
        out += String("[") + String(y_center) + String(",") + String(w_pct) + String("]")
        if j + 1 < bc: out += String(",")
        j += 1
    out += String("]")
    return out
