# Module: momijo.enum.tagging
# Minimal enum utilities implemented in Mojo.
# Project: momijo.enum
# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Momijo Enum
# This file is part of the Momijo project. See the LICENSE file at the repository root.

#
# Copyright (c) 2025 Morteza Taleblou (https:#taleblou.ir/)
# All rights reserved.
#
from runtime.util import ctz_u64

# Does: utility function in enum module.
# Inputs: align.
# Returns: result value or status.
fn tag_bits_from_alignment(align: UInt64) -> UInt64:
    return UInt64(ctz_u64(align))

# Does: utility function in enum module.
# Inputs: ptr, tag, bits.
# Returns: result value or status.
fn tag_pack_ptr(ptr: UInt64, tag: UInt64, bits: UInt64) -> UInt64:
    var mask = (1 << bits) - 1
    return (ptr & ~mask) | (tag & mask)

# Does: utility function in enum module.
# Inputs: packed, bits.
# Returns: result value or status.
fn tag_unpack_ptr(packed: UInt64, bits: UInt64) -> (UInt64, UInt64):
    var mask = (1 << bits) - 1
    var tag = packed & mask
    var ptr = packed & ~mask
    return (ptr, tag)