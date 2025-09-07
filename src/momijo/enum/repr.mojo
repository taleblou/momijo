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
# File: momijo/enum/repr.mojo


struct EnumRepr(ExplicitlyCopyable, Movable):
    var strategy: Int
fn __init__(out self, strategy: Int) -> None:
        self.strategy = strategy
fn __copyinit__(out self, other: Self) -> None:
        self.strategy = other.strategy