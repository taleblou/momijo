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
# Project: momijo.nn.training
# File: src/momijo/nn/training/visualize.mojo

from momijo.core.error import module
from momijo.dataframe.bitmap import invert
from momijo.dataframe.helpers import t
from momijo.utils.result import g
from momijo.visual.stats.stats import Histogram, histogram
from pathlib import Path
from pathlib.path import Path

fn _minf(a: Float64, b: Float64) -> Float64:
    if a <= b: return a
    return b
fn _maxf(a: Float64, b: Float64) -> Float64:
    if a >= b: return a
    return b
fn _clampf(x: Float64, lo: Float64, hi: Float64) -> Float64:
    var y = x
    if y < lo: y = lo
    if y > hi: y = hi
    return y
fn _minmax(values: List[Float64]) -> (Float64, Float64):
    var n = len(values)
    if n == 0: return (0.0, 0.0)
    var mn = values[0]
    var mx = values[0]
    for i in range(1, n):
        if values[i] < mn: mn = values[i]
        if values[i] > mx: mx = values[i]
    return (mn, mx)
fn _repeat(ch: String, count: Int) -> String:
    var s = String("")
    var c = (count if count >= 0 else 0)
    for i in range(c): s = s + ch
    return s
fn _to_int(x: Float64) -> Int:
    var i = Int(x)
    return i

# --------- Sparkline ---------
fn sparkline(values: List[Float64]) -> String:
    var n = len(values)
    if n == 0: return String("")
    var (mn, mx) = _minmax(values)
    var rng = mx - mn
    if rng == 0.0: rng = 1.0
    # levels (8): space + 7 blocks
    var levels = List[String]([String("▁"), String("▂"), String("▃"), String("▄"), String("▅"), String("▆"), String("▇"), String("█")])
    var s = String("")
    for i in range(n):
        var t = (values[i] - mn) / rng
        var idx = _to_int(t * 7.0)
        if idx < 0: idx = 0
        if idx > 7: idx = 7
        s = s + levels[idx]
    return s

# --------- Line chart (ASCII grid, origin top-left) ---------
fn line_chart_ascii(values: List[Float64], height: Int = 10, fill: Bool = False) -> String:
    var n = len(values)
    if n == 0: return String("")
    var H = (height if height >= 2 else 2)
    var (mn, mx) = _minmax(values)
    var rng = mx - mn
    if rng == 0.0: rng = 1.0
    # grid H x n filled with spaces
    var grid = List[List[String]]()
    for r in range(H):
        var row = List[String]()
        for c in range(n): row.push(String(" "))
        grid.push(row)
    # draw axis (bottom row)
    for c in range(n): grid[H-1][c] = String("·")
    # plot
    for c in range(n):
        var t = (values[c] - mn) / rng
        var y = H - 1 - _to_int(t * Float64(H - 1))  # invert for top-origin
        if y < 0: y = 0
        if y >= H: y = H - 1
        grid[y][c] = String("•")
        if fill:
            for r in range(y+1, H-1):
                grid[r][c] = String("│")
    # render
    var out = String("")
    for r in range(H):
        var line = String("")
        for c in range(n): line = line + grid[r][c]
        out = out + line + String("\n")
    return out

# --------- Histogram (vertical bars) ---------
fn hist_ascii(values: List[Float64], bins: Int = 10, height: Int = 10) -> String:
    var N = len(values)
    if N == 0: return String("")
    var B = (bins if bins >= 1 else 1)
    var H = (height if height >= 2 else 2)
    var (mn, mx) = _minmax(values)
    var rng = mx - mn
    if rng == 0.0: rng = 1.0
    var counts = List[Int]()
    for b in range(B): counts.push(0)
    for i in range(N):
        var t = (values[i] - mn) / rng
        var idx = _to_int(t * Float64(B))
        if idx == B: idx = B - 1
        if idx < 0: idx = 0
        if idx >= B: idx = B - 1
        counts[idx] = counts[idx] + 1
    # scale to height
    var cmax = 1
    for b in range(B):
        if counts[b] > cmax: cmax = counts[b]
    var grid = List[List[String]]()
    for r in range(H):
        var row = List[String]()
        for b in range(B): row.push(String(" "))
        grid.push(row)
    for b in range(B):
        var h = _to_int(Float64(counts[b]) / Float64(cmax) * Float64(H - 1))
        for r in range(H-1, H-1-h, -1):
            if r >= 0 and r < H: grid[r][b] = String("█")
    # baseline
    for b in range(B): grid[H-1][b] = String("·")
    # render
    var out = String("")
    for r in range(H):
        var line = String("")
        for b in range(B): line = line + grid[r][b]
        out = out + line + String("\n")
    return out

# --------- Confusion-matrix ASCII heatmap ---------
fn cm_ascii_heatmap(cm: List[List[Int]]) -> String:
    var R = len(cm)
    if R == 0: return String("")
    var C = 0
    if R > 0: C = len(cm[0])
    var vmax = 0
    for i in range(R):
        for j in range(C):
            if cm[i][j] > vmax: vmax = cm[i][j]
    if vmax <= 0: vmax = 1
    var shades = List[String]([String(" "), String("."), String(":"), String("-"), String("="), String("+"), String("*"), String("#"), String("%"), String("@")])
    var out = String("")
    for i in range(R):
        var line = String("")
        for j in range(C):
            var t = Float64(cm[i][j]) / Float64(vmax)
            var idx = _to_int(t * 9.0)
            if idx < 0: idx = 0
            if idx > 9: idx = 9
            line = line + shades[idx]
        out = out + line + String("\n")
    return out

# --------- PPM (ASCII P3) outputs ---------
fn _to_byte(x: Float64) -> Int:
    var y = _clampf(x, 0.0, 1.0) * 255.0
    var i = _to_int(y)
    if i < 0: i = 0
    if i > 255: i = 255
    return i
fn ppm_from_gray(img: List[List[Float64]]) -> String:
    var H = len(img)
    var W = 0
    if H > 0: W = len(img[0])
    var s = String("P3\n") + String(W) + String(" ") + String(H) + String("\n255\n")
    for i in range(H):
        var line = String("")
        for j in range(W):
            var v = _to_byte(img[i][j])
            line = line + String(v) + String(" ") + String(v) + String(" ") + String(v) + String(" ")
        s = s + line + String("\n")
    return s

# blue -> red gradient: (r,g,b) = (t, 0, 1-t)
fn ppm_from_heatmap(mat: List[List[Float64]], vmin: Float64, vmax: Float64) -> String:
    var H = len(mat)
    var W = 0
    if H > 0: W = len(mat[0])
    var s = String("P3\n") + String(W) + String(" ") + String(H) + String("\n255\n")
    var rng = vmax - vmin
    if rng == 0.0: rng = 1.0
    for i in range(H):
        var line = String("")
        for j in range(W):
            var t = (mat[i][j] - vmin) / rng
            t = _clampf(t, 0.0, 1.0)
            var r = _to_byte(t)
            var g = 0
            var b = _to_byte(1.0 - t)
            line = line + String(r) + String(" ") + String(g) + String(" ") + String(b) + String(" ")
        s = s + line + String("\n")
    return s

# --------- Self-test ---------
fn _self_test() -> Bool:
    var ok = True

    # sparkline
    var vals = List[Float64]([0.0, 1.0, 2.0, 1.0, 0.5, 1.5, 2.0, 2.5, 2.0])
    var sp = sparkline(vals)
    ok = ok and (len(sp) == len(vals))

    # line chart
    var lc = line_chart_ascii(vals, 8, True)
    ok = ok and (len(lc) > 0)

    # histogram
    var hs = hist_ascii(vals, 6, 8)
    ok = ok and (len(hs) > 0)

    # confusion heatmap
    var cm = List[List[Int]]()
    cm.push(List[Int]([5,1,0]))
    cm.push(List[Int]([0,7,2]))
    cm.push(List[Int]([1,0,4]))
    var cmh = cm_ascii_heatmap(cm)
    ok = ok and (len(cmh) > 0)

    # ppm gray
    var img = List[List[Float64]]()
    for i in range(4):
        var row = List[Float64]()
        for j in range(5):
            row.push(Float64(i * 5 + j) / 20.0)
        img.push(row)
    var pgray = ppm_from_gray(img)
    ok = ok and (len(pgray) > 0)

    # ppm heatmap
    var mat = List[List[Float64]]()
    for i in range(4):
        var row = List[Float64]()
        for j in range(5):
            row.push(Float64(j))
        mat.push(row)
    var pheat = ppm_from_heatmap(mat, 0.0, 4.0)
    ok = ok and (len(pheat) > 0)

    return ok