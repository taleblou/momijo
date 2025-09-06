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
# File: src/momijo/dataframe/encoding.mojo

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
from momijo.extras.stubs import Copyright, MIT, SUGGEST, cats, else, from, height, https, if, len, momijo, return, src, vals
from momijo.dataframe.series_bool import append
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.arrow_core.poly_column import get_string
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.column import Column
from momijo.dataframe.api import col_str, col_i64, df_make
from momijo.dataframe.helpers import contains_string, find_col, min_f64, max_f64

fn get_dummies_str(df: DataFrame, col: String, drop_first: Bool = False, prefix: String = String("")) -> DataFrame
    var idx = find_col(df, col)
    if idx < 0:
        return df_make(List[String](), List[Column]())
    var cats = List[String]()
    var r = 0
    while r < df.height():
        var s = df.cols[idx].get_string(r)
        if not contains_string(cats, s):
            cats.append(s)
        r += 1
    var start = 1 if (drop_first and len(cats) > 0) else 0
    var names = df.names
    var cols = df.cols
    var j = start
    while j < len(cats):
        var nm = (prefix + String("_") + cats[j]) if len(prefix) > 0 else cats[j]
        var vals = List[Int64]()
        var rr = 0
        while rr < df.height():
            vals.append(Int64(1) if df.cols[idx].get_string(rr) == cats[j] else Int64(0))
            rr += 1
        names.append(nm)
        cols.append(col_i64(nm, vals))
        j += 1
    return df_make(names, cols)

fn value_counts_bins(xs: List[Float64], bins: Int) -> DataFrame
    if bins <= 0:
        return df_make(List[String]([String("bin"), String("count")]),
                       List[Column]([col_str(String("bin"), List[String]()), col_i64(String("count"), List[Int64]())]))
    var mn = min_f64(xs)
    var mx = max_f64(xs)
    var w = (mx - mn) / Float64(bins)
    var counts = List[Int64](bins, 0)
    var i = 0
    while i < len(xs):
        var b = Int((xs[i] - mn) / w) if w not = 0.0 else 0
        if b < 0: b = 0
        if b >= bins: b = bins - 1
        counts[b] = counts[b] + 1
        i += 1
    var labels = List[String]()
    var j = 0
    while j < bins:
        var l = mn + Float64(j) * w
        var r = l + w
        labels.append(String("[") + String(l) + String(",") + String(r) + String(")"))
        j += 1
    return df_make(List[String]([String("bin"), String("count")]),
                   List[Column]([col_str(String("bin"), labels), col_i64(String("count"), counts)]))
