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
# File: src/momijo/dataframe/groupby.mojo

from momijo.extras.stubs import Copyright, MIT, SUGGEST, from, height, https, if, keys, len, momijo, numeric, pos, src
from momijo.dataframe.series_bool import append
from momijo.dataframe.series import groupby_sum
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
from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.api import col_str, col_f64, df_make
from momijo.dataframe.helpers import find_col, parse_f64_or_zero

# GroupBy sum over a string key and a numeric (string/float) value column.

var keys = List[String]()
    var sums = List[Float64]()

    var i = 0
    while i < df.height():
        var k = df.cols[ik].get_string(i)
        var v = parse_f64_or_zero(df.cols[iv].get_string(i))
        # find slot
        var pos = -1
        var j = 0
        while j < len(keys):
            if keys[j] == k:
                pos = j
            j += 1
        if pos == -1:
            keys.append(k)
            sums.append(0.0)
            pos = len(keys) - 1
        sums[pos] = sums[pos] + v
        i += 1

    return df_make(List[String]([key, String("sum_") + val]),
                   List[Column]([col_str(key, keys),
                                 col_f64(String("sum_") + val, sums)]))

fn groupby_sum_f64(df: DataFrame, key: String, val: String) -> DataFrame
    var ik = find_col(df, key)
    var iv = find_col(df, val)
    var keys = List[String]()
    var sums = List[Float64]()
    var i = 0
    while i < df.height():
        var k = df.cols[ik].get_string(i)
        var v = df.cols[iv].get_f64(i)
        var pos = -1
        var t = 0
        while t < len(keys):
            if keys[t] == k:
                pos = t
            t += 1
        if pos < 0:
            keys.append(k)
            sums.append(0.0)
            pos = len(keys) - 1
        sums[pos] = sums[pos] + v
        i += 1
    return df_make(List[String]([key, String("sum_") + val]),
                   List[Column]([col_str(key, keys), col_f64(String("sum_") + val, sums)]))
