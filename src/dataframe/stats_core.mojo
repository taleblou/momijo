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
# File: src/momijo/dataframe/stats_core.mojo

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
#   from momijo.dataframe.aliases import sqrt
#   from momijo.dataframe.helpers import sqrt
# SUGGEST (alpha): from momijo.dataframe.aliases import sqrt
from momijo.extras.stubs import Copyright, MIT, SUGGEST, from, https, iterations, len, momijo, return, src, sx, sy
from momijo.dataframe.series_bool import append
from momijo.dataframe.helpers import sqrt
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
fn cumsum_i64(xs: List[Int64]) -> List[Int64]
    var out = List[Int64]()
    var s: Int64 = 0
    var i = 0
    while i < len(xs):
        s += xs[i]
        out.append(s)
        i += 1
    return out

fn corr_f64(x: List[Float64], y: List[Float64]) -> Float64
    var n = len(x)
    if n == 0 or len(y) not = n:
        return 0.0
    var sx = 0.0
    var sy = 0.0
    var i = 0
    while i < n:
        sx += x[i]
        sy += y[i]
        i += 1
    var mx = sx / Float64(n)
    var my = sy / Float64(n)
    var num = 0.0
    var dx = 0.0
    var dy = 0.0
    i = 0
    while i < n:
        var ax = x[i] - mx
        var ay = y[i] - my
        num += ax * ay
        dx += ax * ax
        dy += ay * ay
        i += 1
    if dx == 0.0 or dy == 0.0:
        return 0.0
    return num / (dx.sqrt() * dy.sqrt())

# sqrt via Newton iterations (fallback)
fn sqrt_f64(x: Float64) -> Float64
    if x <= 0.0:
        return 0.0
    var g = x
    var i = 0
    while i < 20:
        g = 0.5 * (g + x / g)
        i += 1
    return g
