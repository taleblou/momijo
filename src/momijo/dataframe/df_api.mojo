# Project:      Momijo
# Module:       src.momijo.dataframe.df_api
# File:         df_api.mojo
# Path:         src/momijo/dataframe/df_api.mojo
#
# Description:  src.momijo.dataframe.df_api — focused Momijo functionality with a stable public API.
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
#   - Key functions: df_nlargest, df_nsmallest, df_clip, df_fillna_i64, df_fillna_f64, df_fillna_bool


from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.column import Column, from_f64, from_str, get_f64
from momijo.dataframe.frame import DataFrame, get_column_at, width
from momijo.dataframe.helpers import find_col
from momijo.dataframe.na import fillna_col_bool, fillna_col_f64, fillna_col_i64
from momijo.dataframe.ranking import clip_f64
from momijo.dataframe.ranking import nlargest_f64, nsmallest_f64
from momijo.dataframe.series_bool import append
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_str import SeriesStr
from momijo.extras.stubs import cols, height, if, len, return, scores, vals

fn df_nlargest(df: DataFrame, col: String, n: Int) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0:
        return df
    var scores = List[Float64]()
    var i = 0
    while i < df.height():
        scores.append(df.cols[idx].get_f64(i))
        i += 1
    var top_idx = nlargest_f64(scores, n)
    var names = df.names
    var cols = List[Column]()
    var c = 0
    while c < df.width():
        var vals = List[String]()
        var j = 0
        while j < len(top_idx):
            vals.append(df.cols[c].get_string(top_idx[j]))
            j += 1
        cols.append(Column.from_str(SeriesStr(names[c], vals)))
        c += 1
    return DataFrame(names, cols)
fn df_nsmallest(df: DataFrame, col: String, n: Int) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0:
        return df
    var scores = List[Float64]()
    var i = 0
    while i < df.height():
        scores.append(df.cols[idx].get_f64(i))
        i += 1
    var bot_idx = nsmallest_f64(scores, n)
    var names = df.names
    var cols = List[Column]()
    var c = 0
    while c < df.width():
        var vals = List[String]()
        var j = 0
        while j < len(bot_idx):
            vals.append(df.cols[c].get_string(bot_idx[j]))
            j += 1
        cols.append(Column.from_str(SeriesStr(names[c], vals)))
        c += 1
    return DataFrame(names, cols)

fn df_clip(df: DataFrame, col: String, lo: Float64, hi: Float64) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0: return df
    var vals = List[Float64]()
    var r = 0
    while r < df.height():
        vals.append(df.cols[idx].get_f64(r))
        r += 1
    var clipped = clip_f64(vals, lo, hi)
    var names = df.names
    var cols = List[Column]()
    var c = 0
    while c < df.width():
        if c == idx:
            cols.append(Column.from_f64(SeriesF64(names[c], clipped)))
        else:
            cols.append(df.get_column_at(c))
        c += 1
    return DataFrame(names, cols)
fn df_fillna_i64(df: DataFrame, col: String, fill: Int64) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0: return df
    var names = df.names
    var cols = List[Column]()
    var c = 0
    while c < df.width():
        if c == idx: cols.append(fillna_col_i64(df.get_column_at(c), fill)) else: cols.append(df.get_column_at(c))
        c += 1
    return DataFrame(names, cols)
fn df_fillna_f64(df: DataFrame, col: String, fill: Float64) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0: return df
    var names = df.names
    var cols = List[Column]()
    var c = 0
    while c < df.width():
        if c == idx: cols.append(fillna_col_f64(df.get_column_at(c), fill)) else: cols.append(df.get_column_at(c))
        c += 1
    return DataFrame(names, cols)
fn df_fillna_bool(df: DataFrame, col: String, fill: Bool) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0: return df
    var names = df.names
    var cols = List[Column]()
    var c = 0
    while c < df.width():
        if c == idx: cols.append(fillna_col_bool(df.get_column_at(c), fill)) else: cols.append(df.get_column_at(c))
        c += 1
    return DataFrame(names, cols)