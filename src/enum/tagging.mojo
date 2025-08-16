#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
from runtime.util import ctz_u64

fn tag_bits_from_alignment(align: UInt64) -> UInt64:
    return UInt64(ctz_u64(align))

fn tag_pack_ptr(ptr: UInt64, tag: UInt64, bits: UInt64) -> UInt64:
    var mask = (1 << bits) - 1
    return (ptr & ~mask) | (tag & mask)

fn tag_unpack_ptr(packed: UInt64, bits: UInt64) -> (UInt64, UInt64):
    var mask = (1 << bits) - 1
    var tag = packed & mask
    var ptr = packed & ~mask
    return (ptr, tag)
