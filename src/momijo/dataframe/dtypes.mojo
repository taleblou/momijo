# Project:      Momijo
# Module:       src.momijo.dataframe.dtypes
# File:         dtypes.mojo
# Path:         src/momijo/dataframe/dtypes.mojo
#
# Description:  src.momijo.dataframe.dtypes — focused Momijo functionality with a stable public API.
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
#   - Key functions: is_numeric_col, select_numeric, drop_duplicates_rows


from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_str, df_make
from momijo.dataframe.column import Column, ColumnTag, F64, I64
from momijo.dataframe.frame import DataFrame, width
from momijo.dataframe.helpers import contains_string, find_col
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import height, if, len, names, return, seen, tag, vals

fn is_numeric_col(c: Column) -> Bool
    return c.tag() == ColumnTag.I64() or c.tag() == ColumnTag.F64()
fn select_numeric(df: DataFrame) -> DataFrame
    var names = List[String]()
    var cols = List[Column]()
    var i = 0
    while i < df.width():
        if is_numeric_col(df.cols[i]):
            names.append(df.names[i])
            cols.append(df.cols[i])
        i += 1
    return df_make(names, cols)
fn drop_duplicates_rows(df: DataFrame, subset: String) -> DataFrame
    var idx = find_col(df, subset)
    if idx < 0:
        return df_make(List[String](), List[Column]())
    var seen = List[String]()
    var keep = List[Int]()
    var r = 0
    while r < df.height():
        var v = df.cols[idx].get_string(r)
        if not contains_string(seen, v):
            seen.append(v)
            keep.append(r)
        r += 1
    var names = df.names
    var cols = List[Column]()
    var c = 0
    while c < df.width():
        var vals = List[String]()
        var i = 0
        while i < len(keep):
            vals.append(df.cols[c].get_string(keep[i]))
            i += 1
        cols.append(col_str(names[c], vals))
        c += 1
    return df_make(names, cols)