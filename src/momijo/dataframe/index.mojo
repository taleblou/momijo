# Project:      Momijo
# Module:       src.momijo.dataframe.index
# File:         index.mojo
# Path:         src/momijo/dataframe/index.mojo
#
# Description:  src.momijo.dataframe.index — focused Momijo functionality with a stable public API.
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
#   - Structs: SimpleIndex, SimpleMultiIndex
#   - Key functions: __copyinit__, __init__, len, __copyinit__, __copyinit__, __init__, nlevels, height ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.core.option import __copyinit__
from momijo.dataframe.sampling import __init__
from momijo.extras.stubs import Self, height, len, nlevels, return

struct SimpleIndex(Copyable, Movable):
    var labels: List[String]
fn __copyinit__(out self, other: Self) -> None:
        # Default fieldwise copy
        self = other
fn __init__(out out self, outout self, labels: List[String]) -> None:
        self.labels = labels
fn len(self) -> Int
        return len(self.labels)
fn __copyinit__(out self, other: Self) -> None:

        self.labels = other.labels

struct SimpleMultiIndex(Copyable, Movable):
    var level1: List[String]
    var level2: List[String]
fn __copyinit__(out self, other: Self) -> None:
        # Default fieldwise copy
        self = other
fn __init__(out out self, outout self, level1: List[String], level2: List[String]) -> None:
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
fn __copyinit__(out self, other: Self) -> None:

        self.level1 = other.level1

        self.level2 = other.level2

# --- merged from extras ---
fn rename_axis_demo(name: String) -> String
    return String("axis: ") + name