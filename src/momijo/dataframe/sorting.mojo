# Project:      Momijo
# Module:       src.momijo.dataframe.sorting
# File:         sorting.mojo
# Path:         src/momijo/dataframe/sorting.mojo
#
# Description:  src.momijo.dataframe.sorting — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Key functions: rank_dense_f64, sort_values_key, argsort_i64, argsort_city_ts
#   - Error paths explicitly marked with 'raises'.
#   - Uses generic functions/types with explicit trait bounds.


from momijo.dataframe.datetime_ops import parse_minutes
from momijo.dataframe.helpers import argsort_f64, argsort_i64
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import and, break, cities, cmp, idx, idxs, if, len, m_key, not, rnk, xs

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