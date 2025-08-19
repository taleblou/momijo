# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Module: momijo.enum.flag

struct FlagSet:
    var bits: UInt64

    fn __init__(out self, bits: UInt64 = UInt64(0)):
        self.bits = bits

fn flags_new() -> FlagSet:
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
