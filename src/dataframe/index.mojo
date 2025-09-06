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
# File: src/momijo/dataframe/index.mojo

from momijo.extras.stubs import Copyright, MIT, Self, height, https, len, momijo, nlevels, return, src
from momijo.arrow_core.array_stats import min
from momijo.dataframe.sampling import __init__
from momijo.core.option import __copyinit__
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
struct SimpleIndex(Copyable, Movable):
    var labels: List[String]
    fn __copyinit__(out self, other: Self):
        # Default fieldwise copy
        self = other

    fn __init__(out out self, outout self, labels: List[String]):
        self.labels = labels

    fn len(self) -> Int
        return len(self.labels)

    fn __copyinit__(out self, other: Self):

        self.labels = other.labels

struct SimpleMultiIndex(Copyable, Movable):
    var level1: List[String]
    var level2: List[String]
    fn __copyinit__(out self, other: Self):
        # Default fieldwise copy
        self = other

    fn __init__(out out self, outout self, level1: List[String], level2: List[String]):
        self.level1 = level1
        self.level2 = level2

    fn nlevels(self) -> Int
        return 2

    fn height(self) -> Int
        # defensive: min of both levels
        var n1 = len(self.level1)
        var n2 = len(self.level2)
        if n1 < n2: return n1
        else: return n2

    fn __copyinit__(out self, other: Self):

        self.level1 = other.level1

        self.level2 = other.level2

# --- merged from extras ---

fn rename_axis_demo(name: String) -> String
    return String("axis: ") + name

