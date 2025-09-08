# Project:      Momijo
# Module:       src.momijo.arrow_core.array_stats
# File:         array_stats.mojo
# Path:         src/momijo/arrow_core/array_stats.mojo
#
# Description:  Arrow-inspired columnar primitives (offsets, buffers, list/struct arrays)
#               supporting zero-copy slicing and predictable memory semantics.
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
#   - Key functions: sum, mean, sum_f64, mean_f64, min, max, min_f64, max_f64 ...
#   - Uses generic functions/types with explicit trait bounds.


from math import sqrt
from momijo.arrow_core.array import Array

fn count[T: Copyable & Movable](arr: Array[T]) -> Int:
    return arr.len()

fn valid_count[T: Copyable & Movable](arr: Array[T]) -> Int:
    return len(arr.valid_indices())

fn null_count[T: Copyable & Movable](arr: Array[T]) -> Int:
    return arr.null_count()

# ---------- Sum / Mean (Int) ----------
fn sum(arr: Array[Int]) -> Int:
    var s: Int = 0
    for v in arr.compact_values():
        s += v
    return s
fn mean(arr: Array[Int]) -> Float64:
    var vals = arr.compact_values()
    var n = len(vals)
    if n == 0:
        return 0.0
    var s: Int = 0
    for v in vals:
        s += v
    return Float64(s) / Float64(n)

# ---------- Sum / Mean (Float64) ----------
fn sum_f64(arr: Array[Float64]) -> Float64:
    var s: Float64 = 0.0
    for v in arr.compact_values():
        s += v
    return s
fn mean_f64(arr: Array[Float64]) -> Float64:
    var vals = arr.compact_values()
    var n = len(vals)
    if n == 0:
        return 0.0
    var s: Float64 = 0.0
    for v in vals:
        s += v
    return s / Float64(n)

# ---------- Min / Max (Int) ----------
fn min(arr: Array[Int]) -> Int:
    var vals = arr.compact_values()
    if len(vals) == 0:
        return 0
    var m = vals[0]
    for v in vals:
        if v < m:
            m = v
    return m
fn max(arr: Array[Int]) -> Int:
    var vals = arr.compact_values()
    if len(vals) == 0:
        return 0
    var m = vals[0]
    for v in vals:
        if v > m:
            m = v
    return m

# ---------- Min / Max (Float64) ----------
fn min_f64(arr: Array[Float64]) -> Float64:
    var vals = arr.compact_values()
    if len(vals) == 0:
        return 0.0
    var m = vals[0]
    for v in vals:
        if v < m:
            m = v
    return m
fn max_f64(arr: Array[Float64]) -> Float64:
    var vals = arr.compact_values()
    if len(vals) == 0:
        return 0.0
    var m = vals[0]
    for v in vals:
        if v > m:
            m = v
    return m

# ---------- Range ----------
fn range(arr: Array[Int]) -> Int:
    if valid_count(arr) == 0:
        return 0
    return max(arr) - min(arr)
fn range_f64(arr: Array[Float64]) -> Float64:
    if valid_count(arr) == 0:
        return 0.0
    return max_f64(arr) - min_f64(arr)

# ---------- Variance / Stddev (population) ----------
fn variance(arr: Array[Int]) -> Float64:
    var vals = arr.compact_values()
    var n = len(vals)
    if n == 0:
        return 0.0
    var mu = mean(arr)
    var s: Float64 = 0.0
    for v in vals:
        var d = Float64(v) - mu
        s += d * d
    return s / Float64(n)
fn stddev(arr: Array[Int]) -> Float64:
    return sqrt(variance(arr))
fn variance_f64(arr: Array[Float64]) -> Float64:
    var vals = arr.compact_values()
    var n = len(vals)
    if n == 0:
        return 0.0
    var mu = mean_f64(arr)
    var s: Float64 = 0.0
    for v in vals:
        var d = v - mu
        s += d * d
    return s / Float64(n)
fn stddev_f64(arr: Array[Float64]) -> Float64:
    return sqrt(variance_f64(arr))