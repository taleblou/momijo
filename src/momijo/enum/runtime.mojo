# Project:      Momijo
# Module:       src.momijo.enum.runtime
# File:         runtime.mojo
# Path:         src/momijo/enum/runtime.mojo
#
# Description:  src.momijo.enum.runtime — focused Momijo functionality with a stable public API.
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
#   - Key functions: enum_payload_word, tag_bits_from_alignment, tag_pack_ptr, tag_unpack_ptr


from momijo.core.error import module
from momijo.tensor.storage import ptr
from pathlib import Path
from pathlib.path import Path

fn enum_payload_word(packed: UInt64) -> UInt64:
    return packed & UInt64(0xFFFFFFFF)
fn tag_bits_from_alignment(alignment: UInt64) -> UInt64:
    var bits = UInt64(0)
    var a = alignment
    while (a & UInt64(1)) == UInt64(0) and a != UInt64(0):
        bits += UInt64(1)
        a = a >> UInt64(1)
    return bits
fn tag_pack_ptr(ptr: UInt64, tag: UInt64, tag_bits: UInt64) -> UInt64:
    var mask = (UInt64(1) << tag_bits) - UInt64(1)
    return (ptr & ~mask) | (tag & mask)
fn tag_unpack_ptr(packed: UInt64, tag_bits: UInt64) -> (UInt64, UInt64):
    var mask = (UInt64(1) << tag_bits) - UInt64(1)
    var ptr = packed & ~mask
    var tag = packed & mask
    return (ptr, tag)