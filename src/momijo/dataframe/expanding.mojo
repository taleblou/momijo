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
# File: src/momijo/dataframe/expanding.mojo

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