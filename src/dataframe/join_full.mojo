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
# File: src/momijo/dataframe/join_full.mojo

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
from momijo.extras.stubs import Copyright, MIT, SUGGEST, append_df, append_str, cols, dup, from, height, https, if, key, keys, len, matched, momijo, names, pos, row, row2, seen, src, vals
from momijo.dataframe.logical_plan import JOIN
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
from momijo.dataframe.api import df_make, col_str
from momijo.dataframe.helpers import find_col

# Materialize a pairwise matcher for single key (String-based)
fn _build_map(df: DataFrame, key: String) -> List[List[Int]]
    var idx = find_col(df, key)
    var keys = List[String]()
    var poslist = List[List[Int]]()
    var r = 0
    while r < df.height():
        var k = df.cols[idx].get_string(r)
        var pos = -1
        var i = 0
        while i < len(keys):
            if keys[i] == k: pos = i
            i += 1
        if pos < 0:
            keys.append(k); poslist.append(List[Int]())
            pos = len(keys) - 1
        poslist[pos].append(r)
        r += 1
    return poslist  # index paired with an implicit 'keys'

fn _keys_of(df: DataFrame, key: String) -> List[String]
    var idx = find_col(df, key)
    var seen = List[String]()
    var r = 0
    while r < df.height():
        var k = df.cols[idx].get_string(r)
        var dup = False
        var i = 0
        while i < len(seen):
            if seen[i] == k: dup = True
            i += 1
        if not dup: seen.append(k)
        r += 1
    return seen

fn _row_to_strings(df: DataFrame, r: Int) -> List[String]
    var vals = List[String]()
    var c = 0
    while c < df.width():
        vals.append(df.cols[c].get_string(r))
        c += 1
    return vals

fn _append_row(cols: List[Column], row: List[String])
    var i = 0
    while i < len(cols):
        cols[i].append_str(row[i])
        i += 1

fn _empty_row(n: Int) -> List[String]
    var vals = List[String]()
    var i = 0
    while i < n:
        vals.append(String(""))
        i += 1
    return vals

# INNER JOIN (single key)
fn inner_join(a: DataFrame, b: DataFrame, key_a: String, key_b: String) -> DataFrame
    var ia = find_col(a, key_a); var ib = find_col(b, key_b)
    var names = List[String]()
    var cols = List[Column]()
    var c = 0
    while c < a.width():
        names.append(a.names[c]); cols.append(col_str(a.names[c], List[String]())); c += 1
    var cb = 0
    while cb < b.width():
        if cb not = ib:
            names.append(b.names[cb]); cols.append(col_str(b.names[cb], List[String]()))
        cb += 1

    var rb = 0
    while rb < b.height():
        var k = b.cols[ib].get_string(rb)
        var ra = 0
        while ra < a.height():
            if a.cols[ia].get_string(ra) == k:
                var row = _row_to_strings(a, ra)
                var c2 = 0
                while c2 < b.width():
                    if c2 not = ib: row.append(b.cols[c2].get_string(rb))
                    c2 += 1
                _append_row(cols, row)
            ra += 1
        rb += 1
    return df_make(names, cols)

# LEFT/RIGHT/OUTER
fn left_join_full(a: DataFrame, b: DataFrame, key_a: String, key_b: String) -> DataFrame
    var ia = find_col(a, key_a); var ib = find_col(b, key_b)
    var names = List[String]()
    var cols = List[Column]()
    var c = 0
    while c < a.width():
        names.append(a.names[c]); cols.append(col_str(a.names[c], List[String]())); c += 1
    var cb = 0
    while cb < b.width():
        if cb not = ib:
            names.append(b.names[cb]); cols.append(col_str(b.names[cb], List[String]()))
        cb += 1

    var ra = 0
    while ra < a.height():
        var ka = a.cols[ia].get_string(ra)
        var matched = False
        var rb = 0
        while rb < b.height():
            if b.cols[ib].get_string(rb) == ka:
                matched = True
                var row = _row_to_strings(a, ra)
                var c2 = 0
                while c2 < b.width():
                    if c2 not = ib: row.append(b.cols[c2].get_string(rb))
                    c2 += 1
                _append_row(cols, row)
            rb += 1
        if not matched:
            var row2 = _row_to_strings(a, ra)
            var gaps = _empty_row(b.width() - 1)
            var i = 0
            while i < len(gaps): row2.append(gaps[i]); i += 1
            _append_row(cols, row2)
        ra += 1
    return df_make(names, cols)

fn right_join_full(a: DataFrame, b: DataFrame, key_a: String, key_b: String) -> DataFrame
    return left_join_full(b, a, key_b, key_a)

fn outer_join_full(a: DataFrame, b: DataFrame, key_a: String, key_b: String) -> DataFrame
    var left = left_join_full(a, b, key_a, key_b)
    var right_only = right_join_full(a, b, key_a, key_b)
    # naive union by rows-as-strings uniqueness
    var names = left.names
    var cols = List[Column]()
    var c = 0
    while c < left.width():
        cols.append(col_str(names[c], List[String]()))
        c += 1

    fn append_df(df: DataFrame)
        var r = 0
        while r < df.height():
            var row = List[String]()
            var cc = 0
            while cc < df.width():
                row.append(df.cols[cc].get_string(r))
                cc += 1
            _append_row(cols, row)
            r += 1

    append_df(left)
    append_df(right_only)
    return df_make(names, cols)
