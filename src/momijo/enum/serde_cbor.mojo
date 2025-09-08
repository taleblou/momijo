# Project:      Momijo
# Module:       src.momijo.enum.serde_cbor
# File:         serde_cbor.mojo
# Path:         src/momijo/enum/serde_cbor.mojo
#
# Description:  src.momijo.enum.serde_cbor — focused Momijo functionality with a stable public API.
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
#   - Key functions: _u64_to_le, _le_to_u64, enum_to_cbor, enum_from_cbor


from momijo.enum import EnumValue

fn _u64_to_le(x: UInt64) -> List[UInt8]:
    var b = List[UInt8](8)
    for i in range(0, 8):
        b[i] = UInt8((x >> UInt64(8*i)) & 0xFF)
    return b

# Does: utility function in enum module.
# Inputs: b, off.
# Returns: result value or status.
fn _le_to_u64(b: List[UInt8], off: Int) -> UInt64:
    var x: UInt64 = 0
    for i in range(0, 8):
        x = x | (UInt64(b[off+i]) << UInt64(8*i))
    return x

# Does: utility function in enum module.
# Inputs: v.
# Returns: result value or status.
fn enum_to_cbor(v: EnumValue) -> List[UInt8]:
    var out = List[UInt8](1 + 8*5)
    out[0] = 0xE1
    var off = 1
    var parts = [v.tag, v.w0, v.w1, v.w2, v.w3]
    for i in range(0, 5):
        var le = _u64_to_le(parts[i])
        for j in range(0, 8):
            out[off] = le[j]; off += 1
    return out

# Does: utility function in enum module.
# Inputs: buf.
# Returns: result value or status.
fn enum_from_cbor(buf: List[UInt8]) -> (Bool, EnumValue):
    if len(buf) < 1 + 8*5: return (False, EnumValue(tag=0,w0=0,w1=0,w2=0,w3=0))
    if buf[0] != 0xE1: return (False, EnumValue(tag=0,w0=0,w1=0,w2=0,w3=0))
    var off = 1
    var parts = List[UInt64](5)
    for i in range(0, 5):
        parts[i] = _le_to_u64(buf, off); off += 8
    return (True, EnumValue(tag=parts[0], w0=parts[1], w1=parts[2], w2=parts[3], w3=parts[4]))