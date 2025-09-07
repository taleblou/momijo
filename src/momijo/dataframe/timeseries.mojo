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
# File: src/momijo/dataframe/timeseries.mojo

from momijo.dataframe.series_bool import append
from momijo.extras.stubs import len

fn resample_sum_min(series: List[Float64], freq: Int) -> List[Float64]
    var out = List[Float64]()
    var i = 0
    while i < len(series):
        var s = 0.0
        var k = 0
        while k < freq and i + k < len(series):
            s += series[i + k]
            k += 1
        out.append(s)
        i += freq
    return out