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
# File: src/momijo/dataframe/join_simple.mojo

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
from momijo.extras.stubs import Copyright, MIT, SUGGEST, found, from, height, https, if, len, momijo, names, out_cols, out_cols_vals, return, src
from momijo.dataframe.logical_plan import join
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
from momijo.dataframe.helpers import find_col

# Left join (demo) using string materialization; avoids conflicts with core join.mojo
fn left_join_simple(a: DataFrame, b: DataFrame, key_a: String, key_b: String) -> DataFrame
    var ia = find_col(a, key_a)
    var ib = find_col(b, key_b)
    if ia < 0 or ib < 0:
        return df_make(List[String](), List[Column]())

    # names
    var names = List[String]()
    var c = 0
    while c < a.width():
        names.append(a.names[c])
        c += 1
    var cb = 0
    while cb < b.width():
        if cb not = ib:
            names.append(b.names[cb])
        cb += 1

    # accumulate column values as lists of strings
    var out_cols_vals = List[List[String]]()
    var total_cols = len(names)
    var k = 0
    while k < total_cols:
        out_cols_vals.append(List[String]())
        k += 1

    var r = 0
    while r < a.height():
        var key = a.cols[ia].get_string(r)
        # copy A row
        var ca = 0
        while ca < a.width():
            out_cols_vals[ca].append(a.cols[ca].get_string(r))
            ca += 1
        # find first match in B
        var found = -1
        var rb = 0
        while rb < b.height():
            if b.cols[ib].get_string(rb) == key:
                found = rb
                break
            rb += 1
        # append B fields
        var addc = a.width()
        var bb = 0
        while bb < b.width():
            if bb not = ib:
                var v = String("")
                if found >= 0:
                    v = b.cols[bb].get_string(found)
                out_cols_vals[addc].append(v)
                addc += 1
            bb += 1
        r += 1

    # build Columns
    var out_cols = List[Column]()
    var i = 0
    while i < len(names):
        out_cols.append(col_str(names[i], out_cols_vals[i]))
        i += 1
    return df_make(names, out_cols)
