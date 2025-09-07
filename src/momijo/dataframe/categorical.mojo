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