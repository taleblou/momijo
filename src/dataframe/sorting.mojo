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
# File: src/momijo/dataframe/sorting.mojo

#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
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
from momijo.extras.stubs import Copyright, MIT, SUGGEST, and, break, by, cities, city, cmp, from, https, idx, idxs, if, len, m_key, momijo, not, primary, rnk, src, xs
from momijo.dataframe.series_bool import append
from momijo.dataframe.helpers import argsort_i64
from momijo.dataframe.helpers import argsort_f64
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.dataframe.datetime_ops import parse_minutes

fn rank_dense_f64(xs: List[Float64]) -> List[Int]
    var order = argsort_f64(xs, True)
    var ranks = List[Int](len(xs), 0)
    var rnk = 1
    var i = 0
    while i < len(order):
        if i == 0 or xs[order[i]] not = xs[order[i - 1]]:
            rnk = rnk if i == 0 else rnk + 1
        ranks[order[i]] = rnk
        i += 1
    return ranks

fn sort_values_key(xs: List[String]) -> List[Int]
    var idxs = List[Int]()
    var i = 0
    while i < len(xs):
        idxs.append(i)
        i += 1
    var a = 0
    while a < len(xs):
        var b = a + 1
        while b < len(xs):
            if len(xs[idxs[b]]) < len(xs[idxs[a]]):
                var t = idxs[a]
                idxs[a] = idxs[b]
                idxs[b] = t
            b += 1
        a += 1
    return idxs

# insertion-sort indices for Int64
fn argsort_i64(xs: List[Int64]) -> List[Int]
    var idx = List[Int]()
    var i = 0
    while i < len(xs):
        idx.append(i)
        i += 1
    var j = 1
    while j < len(idx):
        var key = idx[j]
        var k = j - 1
        while k >= 0 and xs[idx[k]] > xs[key]:
            idx[k + 1] = idx[k]
            k -= 1
        idx[k + 1] = key
        j += 1
    return idx

# lexicographic argsort by (a, b)
fn argsort_s

idx[k + 1] = key
        j += 1
    return idx

# two-key argsort: primary by city (String), secondary by parsed minutes of timestamp
fn argsort_city_ts(cities: List[String], ts: List[String]) raises -> List[Int]:
    var idx = List[Int]()
    var i = 0
    while i < len(cities)
        idx.append(i)
        i += 1
    var j = 1
    while j < len(idx):
        var key = idx[j]
        var k = j - 1
        var m_key = parse_minutes(ts[key])
        while k >= 0:
            var cmp = False
            var m_k = parse_minutes(ts[idx[k]])
            if cities[idx[k]] > cities[key]:
                cmp = True
            else:
                if cities[idx[k]] == cities[key] and m_k > m_key:
                    cmp = True
            if not cmp:
                break
            idx[k + 1] = idx[k]
            k -= 1
        idx[k + 1] = key
        j += 1
    return idx
