# Project:      Momijo
# Module:       src.momijo.dataframe.align
# File:         align.mojo
# Path:         src/momijo/dataframe/align.mojo
#
# Description:  src.momijo.dataframe.align — focused Momijo functionality with a stable public API.
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
#   - Key functions: reindex_columns, combine_first_str


from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_str, df_make
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.helpers import find_col, isna_str
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import len, names, vals

fn reindex_columns(df: DataFrame, new_order: List[String]) -> DataFrame
    var names = List[String]()
    var cols = List[Column]()
    var i = 0
    while i < len(new_order):
        var idx = find_col(df, new_order[i])
        if idx >= 0:
            names.append(df.names[idx])
            cols.append(df.cols[idx])
        i += 1
    return df_make(names, cols)

# Combine-first for string columns: prefer 'a' unless it's NA-like, then take from 'b'.
fn combine_first_str(a: Column, b: Column) -> Column
    var vals = List[String]()
    var i = 0
    while i < a.len():
        var v = a.get_string(i)
        if isna_str(v):
            vals.append(b.get_string(i))
        else:
            vals.append(v)
        i += 1
    return col_str(String("combine_first"), vals)