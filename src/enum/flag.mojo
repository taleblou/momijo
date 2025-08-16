#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
from bit import pop_count

struct FlagSet:
    var bits: UInt64

fn flags_empty() -> FlagSet:
    return FlagSet(bits=0)

fn flags_all(mask: UInt64) -> FlagSet:
    return FlagSet(bits=mask)

fn flags_has(f: FlagSet, bit: UInt64) -> Bool:
    return (f.bits & bit) == bit

fn flags_set(inout f: FlagSet, bit: UInt64):
    f.bits = f.bits | bit

fn flags_clear(inout f: FlagSet, bit: UInt64):
    f.bits = f.bits & (~bit)

fn flags_toggle(inout f: FlagSet, bit: UInt64):
    f.bits = f.bits ^ bit

fn flags_count(f: FlagSet) -> Int:
    return Int(pop_count(f.bits))

fn flags_intersect(a: FlagSet, b: FlagSet) -> FlagSet:
    return FlagSet(bits=(a.bits & b.bits))

fn flags_union(a: FlagSet, b: FlagSet) -> FlagSet:
    return FlagSet(bits=(a.bits | b.bits))

fn flags_minus(a: FlagSet, b: FlagSet) -> FlagSet:
    return FlagSet(bits=(a.bits & (~b.bits)))
