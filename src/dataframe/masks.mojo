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
# File: src/momijo/dataframe/masks.mojo

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
from momijo.extras.stubs import Copyright, MIT, SUGGEST, acc, from, height, https, keep_idx, len, momijo, return, src
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
from momijo.dataframe.series_str import SeriesStr
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.api import col_str, col_i64, col_f64, df_make
from momijo.dataframe.helpers import find_col, contains_string, parse_i64_or_zero

fn between_i64(x: Int64, a: Int64, b: Int64) -> Bool
    return x >= a and x <= b

fn isin_string(s: String, universe: List[String]) -> Bool
    return contains_string(universe, s)

# Filter rows where column 'qty' is between [a,b], inclusive. String-based materialization.
fn df_where_qty_between(df: DataFrame, a: Int64, b: Int64) -> DataFrame
    var i_qty = find_col(df, String("qty"))
    if i_qty < 0:
        return df_make(List[String](), List[Column]())
    var keep_idx = List[Int]()
    var r = 0
    while r < df.height():
        var q = parse_i64_or_zero(df.cols[i_qty].get_string(r))
        if between_i64(q, a, b):
            keep_idx.append(r)
        r += 1
    # materialize filtered copy
    var names = df.names
    var cols = List[Column]()
    var c = 0
    while c < df.width():
        var acc = List[String]()
        var rr = 0
        while rr < len(keep_idx):
            acc.append(df.cols[c].get_string(keep_idx[rr]))
            rr += 1
        cols.append(col_str(df.names[c], acc))
        c += 1
    return df_make(names, cols)
