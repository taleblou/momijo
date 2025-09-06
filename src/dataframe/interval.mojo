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
# File: src/momijo/dataframe/interval.mojo

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
from momijo.extras.stubs import Copyright, MIT, SUGGEST, Self, contains, from, https, len, momijo, return, src
from momijo.dataframe.sampling import __init__
from momijo.core.option import __copyinit__
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
struct Interval(Copyable, Movable):
    var left: Float64
    var right: Float64
    var closed_left: Bool
    fn __copyinit__(out self, other: Self):
        # Default fieldwise copy
        self = other

    fn __init__(out out self, outout self, left: Float64, right: Float64, closed_left: Bool = True):
        self.left = left
        self.right = right
        self.closed_left = closed_left

    fn contains(self, x: Float64) -> Bool
        if self.closed_left:
            return x >= self.left and x < self.right
        else:
            return x > self.left and x <= self.right

fn searchsorted_f64(sorted: List[Float64], x: Float64) -> Int
    var i = 0
    while i < len(sorted) and sorted[i] < x:
        i += 1
    return i

    fn __copyinit__(out self, other: Self):

        self.left = other.left

        self.right = other.right

        self.closed_left = other.closed_left