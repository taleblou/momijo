# Project:      Momijo
# Module:       src.momijo.enum.flag
# File:         flag.mojo
# Path:         src/momijo/enum/flag.mojo
#
# Description:  src.momijo.enum.flag — focused Momijo functionality with a stable public API.
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
#   - Key functions: flags_has, flags_set, flags_count


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