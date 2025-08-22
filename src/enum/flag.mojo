# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
return FlagSet()
fn flags_has(f: FlagSet, bit: UInt64) -> Bool:
    return (f.bits & bit) != 0
fn flags_set(mut f: FlagSet, bit: UInt64):
    f.bits = f.bits | bit
fn flags_count(f: FlagSet) -> Int:
    var b = f.bits
    var c = 0
    while b != 0:
        c += 1
        b = b & (b - UInt64(1))
    return c