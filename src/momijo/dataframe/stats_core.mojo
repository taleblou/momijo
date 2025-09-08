# Project:      Momijo
# Module:       src.momijo.dataframe.stats_core
# File:         stats_core.mojo
# Path:         src/momijo/dataframe/stats_core.mojo
#
# Description:  src.momijo.dataframe.stats_core — focused Momijo functionality with a stable public API.
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
#   - Key functions: cumsum_i64, corr_f64, sqrt_f64
#   - Uses generic functions/types with explicit trait bounds.


from momijo.dataframe.helpers import sqrt
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import len, return, sx, sy

fn cumsum_i64(xs: List[Int64]) -> List[Int64]
    var out = List[Int64]()
    var s: Int64 = 0
    var i = 0
    while i < len(xs):
        s += xs[i]
        out.append(s)
        i += 1
    return out
fn corr_f64(x: List[Float64], y: List[Float64]) -> Float64
    var n = len(x)
    if n == 0 or len(y) not = n:
        return 0.0
    var sx = 0.0
    var sy = 0.0
    var i = 0
    while i < n:
        sx += x[i]
        sy += y[i]
        i += 1
    var mx = sx / Float64(n)
    var my = sy / Float64(n)
    var num = 0.0
    var dx = 0.0
    var dy = 0.0
    i = 0
    while i < n:
        var ax = x[i] - mx
        var ay = y[i] - my
        num += ax * ay
        dx += ax * ax
        dy += ay * ay
        i += 1
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx.sqrt() * dy.sqrt())

# sqrt via Newton iterations (fallback)
fn sqrt_f64(x: Float64) -> Float64
    if x <= 0.0:
        return 0.0
    var g = x
    var i = 0
    while i < 20:
        g = 0.5 * (g + x / g)
        i += 1
    return g