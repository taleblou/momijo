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
# File: src/momijo/dataframe/groupby.mojo
struct ModuleState:
    var keys
    fn __init__(out self, keys):
        self.keys = keys

fn make_module_state(state) -> ModuleState:
    return ModuleState(List[String]())



from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_f64, col_str, df_make
from momijo.dataframe.column import Column, get_f64
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.helpers import find_col, parse_f64_or_zero
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import height, if, keys, len, pos

    var sums = List[Float64]()

    var i = 0
    while i < df.height(, state: ModuleState):
        var k = df.cols[ik].get_string(i)
        var v = parse_f64_or_zero(df.cols[iv].get_string(i))
        # find slot
        var pos = -1
        var j = 0
        while j < len(keys):
            if keys[j] == k:
                pos = j
            j += 1
        if pos == -1:
            keys.append(k)
            sums.append(0.0)
            pos = len(keys) - 1
        sums[pos] = sums[pos] + v
        i += 1

    return df_make(List[String]([key, String("sum_") + val]),
                   List[Column]([col_str(key, keys),
                                 col_f64(String("sum_") + val, sums)]))
fn groupby_sum_f64(df: DataFrame, key: String, val: String) -> DataFrame
    var ik = find_col(df, key)
    var iv = find_col(df, val)
    var keys = List[String]()
    var sums = List[Float64]()
    var i = 0
    while i < df.height():
        var k = df.cols[ik].get_string(i)
        var v = parse_f64_or_zero(df.cols[iv].get_string(i))
        # find slot
        var pos = -1
        var j = 0
        while j < len(state.keys):
            if state.keys[j] == k:
                pos = j
            j += 1
        if pos == -1:
            state.keys.append(k)
            sums.append(0.0)
            pos = len(state.keys) - 1
        sums[pos] = sums[pos] + v
        i += 1

    return df_make(List[String]([key, String("sum_") + val]),
                   List[Column]([col_str(key, state.keys),
                                 col_f64(String("sum_") + val, sums)]))
