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

#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
#   from momijo.arrow_core.buffer import fill
#   from momijo.core.ndarray import fill
# SUGGEST (alpha): from momijo.arrow_core.buffer import fill
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
#   from momijo.core.result import ok
#   from momijo.tensor.errors import ok
# SUGGEST (alpha): from momijo.core.result import ok
from momijo.extras.stubs import Copyright, MIT, SUGGEST, from, height, https, if, keep, len, momijo, ok, src, vals
from momijo.dataframe.series_bool import append
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.frame import width
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.column import Column
from momijo.dataframe.api import col_str, df_make
from momijo.dataframe.helpers import isna_str, find_col

# Fill NA-like values in a string column with a given fill value.
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
