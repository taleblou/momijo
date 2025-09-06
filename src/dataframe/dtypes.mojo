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
# File: src/momijo/dataframe/dtypes.mojo

#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
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
from momijo.extras.stubs import Copyright, MIT, SUGGEST, from, height, https, if, len, momijo, names, return, seen, src, tag, vals
from momijo.dataframe.series_bool import append
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.column import F64
from momijo.dataframe.column import I64
from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.frame import width
from momijo.dataframe.column import Column, ColumnTag
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.api import df_make, col_str, col_i64
from momijo.dataframe.helpers import contains_string, find_col

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
