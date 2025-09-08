# Project:      Momijo
# Module:       src.momijo.dataframe.expanding
# File:         expanding.mojo
# Path:         src/momijo/dataframe/expanding.mojo
#
# Description:  src.momijo.dataframe.expanding — focused Momijo functionality with a stable public API.
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
#   - Key functions: expanding_corr_last
#   - Uses generic functions/types with explicit trait bounds.


from momijo.dataframe.series_bool import append
from momijo.dataframe.stats_core import corr_f64
from momijo.extras.stubs import len
from momijo.tensor.indexing import slice

fn expanding_corr_last(x: List[Float64], y: List[Float64]) -> List[Float64]
    var out = List[Float64]()
    var i = 1
    while i <= len(x):
        out.append(corr_f64(x.slice(0, i), y.slice(0, i)))
        i += 1
    return out