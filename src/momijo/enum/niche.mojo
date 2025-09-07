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
# File: momijo/enum/niche.mojo


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