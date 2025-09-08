# Project:      Momijo
# Module:       src.momijo.dataframe.query
# File:         query.mojo
# Path:         src/momijo/dataframe/query.mojo
#
# Description:  src.momijo.dataframe.query — focused Momijo functionality with a stable public API.
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
#   - Key functions: df_query


from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.api import col_str, df_make
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame, width
from momijo.dataframe.helpers import find_col
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import height, if, keep, len, return, vals

fn df_query(df: DataFrame, col: String, eq_val: String) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0:
        return df_make(List[String](), List[Column]())
    var keep = List[Int]()
    var r = 0
    while r < df.height():
        if df.cols[idx].get_string(r) == eq_val:
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