# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.arrow_core
# File: momijo/arrow_core/array_stats.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

from momijo.arrow_core.array import Array
from momijo.arrow_core.bitmap import Bitmap

# ---------- Counts ----------

fn count[T: Copyable & Movable](arr: Array[T]) -> Int:
    return arr.len()

fn valid_count[T: Copyable & Movable](arr: Array[T]) -> Int:
    return len(arr.valid_indices())

fn null_count[T: Copyable & Movable](arr: Array[T]) -> Int:
    return arr.null_count()

# ---------- Sum / Mean ----------

fn sum(arr: Array[Int]) -> Int:
    var s: Int = 0
    for v in arr.compact_values():
        s += v
    return s

fn sum_f64(arr: Array[Float64]) -> Float64:
    var s: Float64 = 0.0
    for v in arr.compact_values():
        s += v
    return s

fn mean(arr: Array[Int]) -> Float64:
    let vals = arr.compact_values()
    if len(vals) == 0:
        return 0.0
    var s: Int = 0
    for v in vals:
        s += v
    return Float64(s) / Float64(len(vals))

fn mean_f64(arr: Array[Float64]) -> Float64:
    let vals = arr.compact_values()
    if len(vals) == 0:
        return 0.0
    var s: Float64 = 0.0
    for v in vals:
        s += v
    return s / Float64(len(vals))

# ---------- Min / Max / Range ----------

fn min(arr: Array[Int]) -> Int:
    let vals = arr.compact_values()
    if len(vals) == 0:
        return 0
    var m = vals[0]
    for v in vals:
        if v < m:
            m = v
    return m

fn max(arr: Array[Int]) -> Int:
    let vals = arr.compact_values()
    if len(vals) == 0:
        return 0
    var m = vals[0]
    for v in vals:
        if v > m:
            m = v
    return m

fn min_f64(arr: Array[Float64]) -> Float64:
    let vals = arr.compact_values()
    if len(vals) == 0:
        return 0.0
    var m = vals[0]
    for v in vals:
        if v < m:
            m = v
    return m

fn max_f64(arr: Array[Float64]) -> Float64:
    let vals = arr.compact_values()
    if len(vals) == 0:
        return 0.0
    var m = vals[0]
    for v in vals:
        if v > m:
            m = v
    return m

fn range(arr: Array[Int]) -> Int:
    if valid_count(arr) == 0:
        return 0
    return max(arr) - min(arr)

fn range_f64(arr: Array[Float64]) -> Float64:
    if valid_count(arr) == 0:
        return 0.0
    return max_f64(arr) - min_f64(arr)

# ---------- Variance / Stddev ----------

fn variance(arr: Array[Int]) -> Float64:
    let vals = arr.compact_values()
    let n = len(vals)
    if n == 0:
        return 0.0
    var mean_val = mean(arr)
    var s: Float64 = 0.0
    for v in vals:
        let diff = Float64(v) - mean_val
        s += diff * diff
    return s / Float64(n)

fn stddev(arr: Array[Int]) -> Float64:
    let var_val = variance(arr)
    return sqrt(var_val)

fn variance_f64(arr: Array[Float64]) -> Float64:
    let vals = arr.compact_values()
    let n = len(vals)
    if n == 0:
        return 0.0
    var mean_val = mean_f64(arr)
    var s: Float64 = 0.0
    for v in vals:
        let diff = v - mean_val
        s += diff * diff
    return s / Float64(n)

fn stddev_f64(arr: Array[Float64]) -> Float64:
    let var_val = variance_f64(arr)
    return sqrt(var_val)

