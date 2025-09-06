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
# File: src/momijo/dataframe/categorical.mojo

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
from momijo.extras.stubs import Copyright, MIT, SUGGEST, Self, from, https, idx, if, in, len, momijo, remove_unused, reorder, src
from momijo.dataframe.sampling import __init__
from momijo.core.option import __copyinit__
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
struct Categorical(Copyable, Movable):
    var categories: List[String]
    var codes: List[Int]
    fn __copyinit__(out self, other: Self):
        # Default fieldwise copy
        self = other

    fn __init__(out out self, outout self, categories: List[String], codes: List[Int]):
        self.categories = categories
        self.codes = codes

    fn remove_unused(out self)
        # noop demo: in a full impl, rebuild categories/codes
        pass

    fn reorder(out self, new_order: List[String])
        # rebuild codes to match new_order
        var i = 0
        while i < len(self.codes):
            var cat = self.categories[self.codes[i]]
            # find index in new_order
            var k = 0
            var idx = 0
            while k < len(new_order):
                if new_order[k] == cat:
                    idx = k
                k += 1
            self.codes[i] = idx
            i += 1
        self.categories = new_order

    fn __copyinit__(out self, other: Self):

        self.categories = other.categories

        self.codes = other.codes