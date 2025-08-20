# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Module: momijo.enum.union

# Compute number of 64-bit words needed to store `bits` bits (ceiling division by 64)
fn words_needed(bits: Int) -> Int:
    if bits <= 0:
        return 0
    var b = UInt64(bits)
    var q = Int((b + UInt64(63)) >> UInt64(6))
    return q
