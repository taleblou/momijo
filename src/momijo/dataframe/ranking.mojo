# Project:      Momijo
# Module:       src.momijo.dataframe.ranking
# File:         ranking.mojo
# Path:         src/momijo/dataframe/ranking.mojo
#
# Description:  src.momijo.dataframe.ranking — focused Momijo functionality with a stable public API.
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
#   - Key functions: nlargest_f64, nsmallest_f64, clip_f64
#   - Uses generic functions/types with explicit trait bounds.


from momijo.dataframe.series_bool import append
from momijo.dataframe.sorting import argsort_f64
from momijo.extras.stubs import hi, if, len

fn nlargest_f64(xs: List[Float64], n: Int) -> List[Int]
    var idx = argsort_f64(xs, False)
    var k = n
    if k > len(idx):
        k = len(idx)
    var out = List[Int]()
    var i = 0
    while i < k:
        out.append(idx[i])
        i += 1
    return out
fn nsmallest_f64(xs: List[Float64], n: Int) -> List[Int]
    var idx = argsort_f64(xs, True)
    var k = n
    if k > len(idx):
        k = len(idx)
    var out = List[Int]()
    var i = 0
    while i < k:
        out.append(idx[i])
        i += 1
    return out
fn clip_f64(xs: List[Float64], lo: Float64, hi: Float64) -> List[Float64]
    var out = List[Float64]()
    var i = 0
    while i < len(xs):
        var v = xs[i]
        if v < lo:
            v = lo
        if v > hi:
            v = hi
        out.append(v)
        i += 1
    return out