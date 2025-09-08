# Project:      Momijo
# Module:       src.momijo.enum.repr
# File:         repr.mojo
# Path:         src/momijo/enum/repr.mojo
#
# Description:  src.momijo.enum.repr — focused Momijo functionality with a stable public API.
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
#   - Structs: EnumRepr
#   - Key functions: __init__, __copyinit__


struct EnumRepr(ExplicitlyCopyable, Movable):
    var strategy: Int
fn __init__(out self, strategy: Int) -> None:
        self.strategy = strategy
fn __copyinit__(out self, other: Self) -> None:
        self.strategy = other.strategy