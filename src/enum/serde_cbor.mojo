#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
from .enum import EnumValue

fn _u64_to_le(x: UInt64) -> List[UInt8]:
    var b = List[UInt8](8)
    for i in range(0, 8):
        b[i] = UInt8((x >> UInt64(8*i)) & 0xFF)
    return b

fn _le_to_u64(b: List[UInt8], off: Int) -> UInt64:
    var x: UInt64 = 0
    for i in range(0, 8):
        x = x | (UInt64(b[off+i]) << UInt64(8*i))
    return x

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

fn enum_from_cbor(buf: List[UInt8]) -> (Bool, EnumValue):
    if len(buf) < 1 + 8*5: return (False, EnumValue(tag=0,w0=0,w1=0,w2=0,w3=0))
    if buf[0] != 0xE1: return (False, EnumValue(tag=0,w0=0,w1=0,w2=0,w3=0))
    var off = 1
    var parts = List[UInt64](5)
    for i in range(0, 5):
        parts[i] = _le_to_u64(buf, off); off += 8
    return (True, EnumValue(tag=parts[0], w0=parts[1], w1=parts[2], w2=parts[3], w3=parts[4]))
