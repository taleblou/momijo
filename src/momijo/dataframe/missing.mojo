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
# File: src/momijo/dataframe/missing.mojo

from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_str, df_make
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame, width
from momijo.dataframe.helpers import isna_str
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import height, if, keep, len, ok, vals

fn fillna_str_col(c: Column, fill: String) -> Column
    var out = List[String]()
    var r = 0
    while r < c.len():
        var v = c.get_string(r)
        if isna_str(v): out.append(fill)
        else: out.append(v)
        r += 1
    return col_str(String("filled"), out)

# Drop any row that has an NA-like string cell in any column.
fn dropna_rows(df: DataFrame) -> DataFrame
    var keep = List[Int]()
    var r = 0
    while r < df.height():
        var ok = True
        var c = 0
        while c < df.width():
            if isna_str(df.cols[c].get_string(r)):
                ok = False
            c += 1
        if ok:
            keep.append(r)
        r += 1
    # build
    var names = df.names
    var cols = List[Column]()
    var cc = 0
    while cc < df.width():
        var vals = List[String]()
        var i = 0
        while i < len(keep):
            vals.append(df.cols[cc].get_string(keep[i]))
            i += 1
        cols.append(col_str(names[cc], vals))
        cc += 1
    return df_make(names, cols)