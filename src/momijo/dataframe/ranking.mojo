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
# Project: momijo.dataframe
# File: src/momijo/dataframe/ranking.mojo

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