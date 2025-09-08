# Project:      Momijo
# Module:       src.momijo.enum.match
# File:         match.mojo
# Path:         src/momijo/enum/match.mojo
#
# Description:  src.momijo.enum.match — focused Momijo functionality with a stable public API.
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
#   - Structs: Case, RangeCase
#   - Key functions: __init__, __copyinit__, __init__, __copyinit__


from momijo.core.error import module
from momijo.utils.timer import start
from momijo.vision.schedule.schedule import end
from pathlib import Path
from pathlib.path import Path

struct Case(ExplicitlyCopyable, Movable):
    var tag: Int
    var result: Int  
    var arm: UInt64
fn __init__(out self, tag: Int, result: Int, arm: UInt64)):
        self.tag = tag
        self.result = result
        self.arm = arm
fn __copyinit__(out self, other: Self) -> None:
        self.tag = other.tag
        self.result = other.result
        self.arm =  other.arm

# Range case inclusive
struct RangeCase(ExplicitlyCopyable, Movable):
    var start: Int
    var end: Int
    var result: Int
fn __init__(out self, start: Int, end: Int, result: Int) -> None:
        self.start = start
        self.end = end
        self.result = result
fn __copyinit__(out self, other: Self) -> None:
        self.start = other.start
        self.end = other.end
        self.result = other.result