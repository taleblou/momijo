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
# File: momijo/enum/flag.mojo


from momijo.core.error import module
from momijo.utils.result import f
from pathlib import Path
from pathlib.path import Path

return FlagSet()
fn flags_has(f: FlagSet, bit: UInt64) -> Bool:
    return (f.bits & bit) != 0
fn flags_set(mut f: FlagSet, bit: UInt64) -> None:
    f.bits = f.bits | bit
fn flags_count(f: FlagSet) -> Int:
    var b = f.bits
    var c = 0
    while b != 0:
        c += 1
        b = b & (b - UInt64(1))
    return c