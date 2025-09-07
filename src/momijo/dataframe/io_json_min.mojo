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
# File: src/momijo/dataframe/io_json_min.mojo

from momijo.dataframe.api import col_str, df_make
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.helpers import contains_string
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import cols, cols_vals, for, if, keys, len

fn json_normalize_min(rows: List[Map[String, String]]) -> DataFrame
    var keys = List[String]()
    var i = 0
    while i < len(rows):
        for (k, v) in rows[i]:
            if not contains_string(keys, k):
                keys.append(k)
        i += 1
    var cols_vals = List[List[String]]()
    var j = 0
    while j < len(keys):
        cols_vals.append(List[String]())
        j += 1
    var r = 0
    while r < len(rows):
        var k2 = 0
        while k2 < len(keys):
            var key = keys[k2]
            var v = String("")
            if key in rows[r]:
                v = rows[r][key]
            cols_vals[k2].append(v)
            k2 += 1
        r += 1
    var cols = List[Column]()
    var t = 0
    while t < len(keys):
        cols.append(col_str(keys[t], cols_vals[t]))
        t += 1
    return df_make(keys, cols)