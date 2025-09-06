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
# File: src/momijo/dataframe/rolling.mojo

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
from momijo.extras.stubs import Copyright, MIT, SUGGEST, cnt, from, https, len, momijo, src
from momijo.dataframe.series_bool import append
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
fn rolling_mean(xs: List[Float64], win: Int) -> List[Float64]
    var n = len(xs)
    var out = List[Float64]()
    var i = 0
    while i < n:
        var s = 0.0
        var cnt = 0
        var k = i - win + 1
        if k < 0:
            k = 0
        while k <= i:
            s += xs[k]
            cnt += 1
            k += 1
        out.append(s / Float64(cnt))
        i += 1
    return out

fn rolling_apply_abs(xs: List[Float64], win: Int) -> List[Float64]
    # rolling mean of absolute values
    var n = len(xs)
    var out = List[Float64]()
    var i = 0
    while i < n:
        var s = 0.0
        var cnt = 0
        var k = i - win + 1
        if k < 0:
            k = 0
        while k <= i:
            var v = xs[k]
            s += (v if v >= 0.0 else -v)
            cnt += 1
            k += 1
        out.append(s / Float64(cnt))
        i += 1
    return out
