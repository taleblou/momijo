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
# Project: momijo.enum
# File: momijo/enum/match.mojo


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