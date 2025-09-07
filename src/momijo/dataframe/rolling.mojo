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
# File: src/momijo/dataframe/rolling.mojo

from momijo.dataframe.series_bool import append
from momijo.extras.stubs import cnt, len

fn rolling_mean(xs: List[Float64], win: Int) -> List[Float64]
    var n = len(xs)
    var out = List[Float64]()
    var i = 0
    while i < n:
        var s = 0.0
        var cnt = 0
        var k = i - win + 1
        if k < 0:
            k = 0
        while k <= i:
            s += xs[k]
            cnt += 1
            k += 1
        out.append(s / Float64(cnt))
        i += 1
    return out
fn rolling_apply_abs(xs: List[Float64], win: Int) -> List[Float64]
    # rolling mean of absolute values
    var n = len(xs)
    var out = List[Float64]()
    var i = 0
    while i < n:
        var s = 0.0
        var cnt = 0
        var k = i - win + 1
        if k < 0:
            k = 0
        while k <= i:
            var v = xs[k]
            s += (v if v >= 0.0 else -v)
            cnt += 1
            k += 1
        out.append(s / Float64(cnt))
        i += 1
    return out