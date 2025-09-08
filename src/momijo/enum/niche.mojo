# Project:      Momijo
# Module:       src.momijo.enum.niche
# File:         niche.mojo
# Path:         src/momijo/enum/niche.mojo
#
# Description:  src.momijo.enum.niche — focused Momijo functionality with a stable public API.
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
#   - Structs: OptionPtr
#   - Key functions: __init__, __copyinit__, some_ptr, none_ptr, is_none


struct OptionPtr(
    Copyable,
    Movable,
):
    var raw: UInt64
fn __init__(out self, raw: UInt64) -> None:
        self.raw = raw
fn __copyinit__(out self, other: Self) -> None:
        self.raw = other.raw
fn some_ptr(p: UInt64) -> OptionPtr:
    return OptionPtr(raw=p if p != UInt64(0) else UInt64(1))
fn none_ptr() -> OptionPtr:
    return OptionPtr(raw=UInt64(0))
fn is_none(x: OptionPtr) -> Bool:
    return x.raw == UInt64(0)