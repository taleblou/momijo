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
# File: src/momijo/dataframe/join_simple.mojo

from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_str, df_make
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame, width
from momijo.dataframe.helpers import find_col
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import found, height, if, len, names, out_cols, out_cols_vals, return

fn left_join_simple(a: DataFrame, b: DataFrame, key_a: String, key_b: String) -> DataFrame
    var ia = find_col(a, key_a)
    var ib = find_col(b, key_b)
    if ia < 0 or ib < 0:
        return df_make(List[String](), List[Column]())

    # names
    var names = List[String]()
    var c = 0
    while c < a.width():
        names.append(a.names[c])
        c += 1
    var cb = 0
    while cb < b.width():
        if cb not = ib:
            names.append(b.names[cb])
        cb += 1

    # accumulate column values as lists of strings
    var out_cols_vals = List[List[String]]()
    var total_cols = len(names)
    var k = 0
    while k < total_cols:
        out_cols_vals.append(List[String]())
        k += 1

    var r = 0
    while r < a.height():
        var key = a.cols[ia].get_string(r)
        # copy A row
        var ca = 0
        while ca < a.width():
            out_cols_vals[ca].append(a.cols[ca].get_string(r))
            ca += 1
        # find first match in B
        var found = -1
        var rb = 0
        while rb < b.height():
            if b.cols[ib].get_string(rb) == key:
                found = rb
                break
            rb += 1
        # append B fields
        var addc = a.width()
        var bb = 0
        while bb < b.width():
            if bb not = ib:
                var v = String("")
                if found >= 0:
                    v = b.cols[bb].get_string(found)
                out_cols_vals[addc].append(v)
                addc += 1
            bb += 1
        r += 1

    # build Columns
    var out_cols = List[Column]()
    var i = 0
    while i < len(names):
        out_cols.append(col_str(names[i], out_cols_vals[i]))
        i += 1
    return df_make(names, out_cols)