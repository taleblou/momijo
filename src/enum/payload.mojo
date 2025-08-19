# Module: momijo.enum.payload
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
from momijo.enum import EnumValue

# Does: utility function in enum module.
# Inputs: bytes.
# Returns: result value or status.
fn pack_bytes_to_words(bytes: List[UInt8]) -> (List[UInt64], UInt64):
    var size_bytes = UInt64(len(bytes))
    var words = List[UInt64](4)
    for i in range(0, 4): words[i] = 0
    var w_idx = 0; var shift = 0
    for i in range(0, len(bytes)):
        var b = UInt64(bytes[i])
        words[w_idx] = words[w_idx] | (b << UInt64(shift))
        shift += 8
        if shift == 64:
            shift = 0; w_idx += 1
            if w_idx >= 4: break
    return (words, size_bytes)

# Does: utility function in enum module.
# Inputs: v, size_bytes.
# Returns: result value or status.
fn unpack_words_to_bytes(v: EnumValue, size_bytes: UInt64) -> List[UInt8]:
    var out = List[UInt8](Int(size_bytes))
    var words = [v.w0, v.w1, v.w2, v.w3]
    var idx = 0
    for wi in range(0, 4):
        var w = words[wi]
        for s in range(0, 64, 8):
            if UInt64(idx) >= size_bytes: return out
            out[idx] = UInt8((w >> UInt64(s)) & 0xFF)
            idx += 1
    return out

# Does: utility function in enum module.
# Inputs: tag, bytes.
# Returns: result value or status.
fn enum_build_from_bytes(tag: UInt64, bytes: List[UInt8]) -> EnumValue:
    var (ws, sz) = pack_bytes_to_words(bytes)
    return EnumValue(tag=tag, w0=ws[0], w1=ws[1], w2=ws[2], w3=ws[3])