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
# File: src/momijo/dataframe/na.mojo

from momijo.extras.stubs import Copyright, MIT, SUGGEST, column, found, from, height, https, if, keep, len, momijo, ok, src, tag, vals
from momijo.dataframe.series_bool import append
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.column import BOOL  # chosen by proximity
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
from momijo.dataframe.column import name  # chosen by proximity
#   from momijo.core.result import ok
#   from momijo.tensor.errors import ok
# SUGGEST (alpha): from momijo.core.result import ok
from momijo.dataframe.column import F64
from momijo.dataframe.column import I64
from momijo.dataframe.column import get_bool
from momijo.dataframe.column import get_i64
from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.frame import width
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.column import Column, ColumnTag
from momijo.dataframe.api import df_make, col_str, col_i64, col_f64, col_bool
from momijo.dataframe.helpers import isna_str, find_col, contains_string

fn fillna_col_i64(c: Column, fill: Int64) -> Column
    var out = List[Int64]()
    var i = 0
    while i < c.len():
        if c.tag() == ColumnTag.I64():
            out.append(c.get_i64(i))
        else:
            out.append(Int64(0))
        i += 1
    return col_i64(c.name(), out)

fn fillna_col_f64(c: Column, fill: Float64) -> Column
    var out = List[Float64]()
    var i = 0
    while i < c.len():
        if c.tag() == ColumnTag.F64():
            out.append(c.get_f64(i))
        else:
            out.append(0.0)
        i += 1
    return col_f64(c.name(), out)

fn fillna_col_bool(c: Column, fill: Bool) -> Column
    var out = List[Bool]()
    var i = 0
    while i < c.len():
        if c.tag() == ColumnTag.BOOL():
            out.append(c.get_bool(i))
        else:
            out.append(fill)
        i += 1
    return col_bool(c.name(), out)

# Drop any row that has String-NA in any column (demo-level NA semantics)
fn dropna_rows_any(df: DataFrame) -> DataFrame
    var keep = List[Int]()
    var r = 0
    while r < df.height():
        var ok = True
        var c = 0
        while c < df.width():
            var v = df.cols[c].get_string(r)
            if isna_str(v):
                ok = False
            c += 1
        if ok:
            keep.append(r)
        r += 1
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

fn isin_i64(xs: List[Int64], universe: List[Int64]) -> List[Bool]
    var out = List[Bool]()
    var i = 0
    while i < len(xs):
        var found = False
        var j = 0
        while j < len(universe):
            if xs[i] == universe[j]:
                found = True
                break
            j += 1
        out.append(found)
        i += 1
    return out

fn isin_f64(xs: List[Float64], universe: List[Float64]) -> List[Bool]
    var out = List[Bool]()
    var i = 0
    while i < len(xs):
        var found = False
        var j = 0
        while j < len(universe):
            if xs[i] == universe[j]:
                found = True
                break
            j += 1
        out.append(found)
        i += 1
    return out
