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
# File: src/momijo/dataframe/frame_ops.mojo

#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
from momijo.extras.stubs import Copyright, MIT, SUGGEST, _to_string, arr, arr_b, arr_i, arr_s, from, https, len, momijo, names, src, value
from momijo.dataframe.io_csv import is_bool
from momijo.dataframe.column import head
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.column import from_bool  # chosen by proximity
from momijo.dataframe.column import from_f64  # chosen by proximity
from momijo.dataframe.column import from_i64  # chosen by proximity
from momijo.dataframe.column import get_f64  # chosen by proximity
#   from momijo.dataframe.frame import height
#   from momijo.dataframe.index import height
# SUGGEST (alpha): from momijo.dataframe.frame import height
#   from momijo.dataframe.column import is_bool
#   from momijo.dataframe.io_csv import is_bool
#   from momijo.tensor.dtype import is_bool
# SUGGEST (alpha): from momijo.dataframe.column import is_bool
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
#   from momijo.dataframe.aliases import take
#   from momijo.dataframe.column import take
#   from momijo.dataframe.frame import take
#   from momijo.dataframe.helpers import take
#   from momijo.dataframe.series_bool import take
#   from momijo.dataframe.series_f64 import take
#   from momijo.dataframe.series_i64 import take
#   from momijo.dataframe.series_str import take
# SUGGEST (alpha): from momijo.dataframe.aliases import take
#   from momijo.arrow_core.arrays.boolean_array import value
#   from momijo.arrow_core.arrays.primitive_array import value
#   from momijo.arrow_core.arrays.string_array import value
#   from momijo.core.types import value
# SUGGEST (alpha): from momijo.arrow_core.arrays.boolean_array import value
from momijo.dataframe.column import from_str
from momijo.dataframe.column import get_bool
from momijo.dataframe.frame import get_column_at
from momijo.dataframe.column import get_i64
from momijo.dataframe.column import is_f64
from momijo.dataframe.column import is_i64
from momijo.dataframe.frame import width
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.column import Column
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_str import SeriesStr
from momijo.dataframe.series_bool import SeriesBool, append
from momijo.dataframe.api import df_make
from momijo.dataframe.helpers import find_col

var names = List[String]()
    var cols = List[Column]()

    var cidx = 0
    while cidx < df.width():
        var name = df.names[cidx]
        names.append(name)

        var src = df.get_column_at(cidx)

        if src.is_f64():
            var arr = List[Float64]()
            var r = 0
            while r < take:
                arr.append(src.get_f64(r))
                r += 1
            cols.append(Column.from_f64(SeriesF64(name, arr)))

        elif src.is_i64():
            var arr_i = List[Int64]()
            var r2 = 0
            while r2 < take:
                arr_i.append(src.get_i64(r2))
                r2 += 1
            cols.append(Column.from_i64(SeriesI64(name, arr_i)))

        elif src.is_bool():
            var arr_b = List[Bool]()
            var r3 = 0
            while r3 < take:
                arr_b.append(src.get_bool(r3))
                r3 += 1
            cols.append(Column.from_bool(SeriesBool(name, arr_b)))

        else:
            var arr_s = List[String]()
            var r4 = 0
            while r4 < take:
                arr_s.append(src.value()_to_string(r4))
                r4 += 1
            cols.append(Column.from_str(SeriesStr(name, arr_s)))

        cidx += 1

    return DataFrame(names, cols)

# Select a subset of columns by name, in the given order. Missing names are skipped.
fn select_columns_safe(df: DataFrame, want: List[String]) -> DataFrame
    var names = List[String]()
    var cols = List[Column]()
    var i = 0
    while i < len(want):
        var idx = find_col(df, want[i])
        if idx >= 0:
            names.append(df.names[idx])
            cols.append(df.get_column_at(idx))
        i += 1
    return df_make(names, cols)
