# Project:      Momijo
# Module:       src.momijo.dataframe.timeseries
# File:         timeseries.mojo
# Path:         src/momijo/dataframe/timeseries.mojo
#
# Description:  src.momijo.dataframe.timeseries — focused Momijo functionality with a stable public API.
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
#   - Key functions: resample_sum_min
#   - Uses generic functions/types with explicit trait bounds.


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