# Project:      Momijo
# Module:       src.momijo.dataframe.categorical
# File:         categorical.mojo
# Path:         src/momijo/dataframe/categorical.mojo
#
# Description:  src.momijo.dataframe.categorical — focused Momijo functionality with a stable public API.
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
#   - Structs: Categorical
#   - Key functions: __copyinit__, __init__, remove_unused, reorder, __copyinit__
#   - Uses generic functions/types with explicit trait bounds.


from momijo.core.option import __copyinit__
from momijo.dataframe.sampling import __init__
from momijo.extras.stubs import Self, idx, if, len, remove_unused, reorder

struct Categorical(Copyable, Movable):
    var categories: List[String]
    var codes: List[Int]
fn __copyinit__(out self, other: Self) -> None:
        # Default fieldwise copy
        self = other
fn __init__(out out self, outout self, categories: List[String], codes: List[Int]) -> None:
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
fn __copyinit__(out self, other: Self) -> None:

        self.categories = other.categories

        self.codes = other.codes