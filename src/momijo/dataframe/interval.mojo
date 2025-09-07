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

from momijo.core.option import __copyinit__
from momijo.dataframe.sampling import __init__
from momijo.extras.stubs import Self, contains, len, return

struct Interval(Copyable, Movable):
    var left: Float64
    var right: Float64
    var closed_left: Bool
fn __copyinit__(out self, other: Self) -> None:
        # Default fieldwise copy
        self = other
fn __init__(out out self, outout self, left: Float64, right: Float64, closed_left: Bool = True) -> None:
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
fn __copyinit__(out self, other: Self) -> None:

        self.left = other.left

        self.right = other.right

        self.closed_left = other.closed_left