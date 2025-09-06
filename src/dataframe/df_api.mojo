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
# File: src/momijo/dataframe/df_api.mojo

from momijo.extras.stubs import Copyright, MIT, SUGGEST, cols, column, from, height, https, if, len, momijo, return, scores, src, vals
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.column import Column  # chosen by proximity
#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
#   from momijo.dataframe.aliases import col
#   from momijo.dataframe.api import col
#   from momijo.dataframe.column import col
# SUGGEST (alpha): from momijo.dataframe.aliases import col
#   from momijo.arrow_core.buffer import fill
#   from momijo.core.ndarray import fill
# SUGGEST (alpha): from momijo.arrow_core.buffer import fill
from momijo.dataframe.column import from_f64  # chosen by proximity
from momijo.dataframe.column import get_f64  # chosen by proximity
#   from momijo.dataframe.frame import height
#   from momijo.dataframe.index import height
# SUGGEST (alpha): from momijo.dataframe.frame import height
#   from momijo.arrow_core.array import len
#   from momijo.arrow_core.array_base import len
#   from momijo.arrow_core.arrays.boolean_array import len
#   from momijo.arrow_core.arrays.list_array import len
#   from momijo.arrow_core.arrays.primitive_array import len
#   from momijo.arrow_core.arrays.string_array import len
#   from momijo.arrow_core.bitmap import len
#   from momijo.arrow_core.buffer import len
#   from momijo.arrow_core.buffer_slice import len
#   from momijo.arrow_core.byte_string_array import len
#   from momijo.arrow_core.column import len
#   from momijo.arrow_core.offsets import len
#   from momijo.arrow_core.poly_column import len
#   from momijo.arrow_core.string_array import len
#   from momijo.core.types import len
#   from momijo.dataframe.column import len
#   from momijo.dataframe.index import len
#   from momijo.dataframe.series_bool import len
#   from momijo.dataframe.series_f64 import len
#   from momijo.dataframe.series_i64 import len
#   from momijo.dataframe.series_str import len
# SUGGEST (alpha): from momijo.arrow_core.array import len
from momijo.dataframe.series_str import SeriesStr
from momijo.dataframe.column import from_str
from momijo.dataframe.frame import get_column_at
from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.frame import width
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.api import df_make, col_f64
from momijo.dataframe.helpers import find_col
from momijo.dataframe.ranking import nlargest_f64, nsmallest_f64

# Return top-n rows by a Float64 column (copy-out)
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

from momijo.dataframe.ranking import clip_f64
from momijo.dataframe.na import fillna_col_i64, fillna_col_f64, fillna_col_bool
from momijo.dataframe.api import col_i64, col_f64, col_bool, df_make
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_bool import SeriesBool, append

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
