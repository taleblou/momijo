# Project:      Momijo
# Module:       src.momijo.enum.payload
# File:         payload.mojo
# Path:         src/momijo/enum/payload.mojo
#
# Description:  src.momijo.enum.payload — focused Momijo functionality with a stable public API.
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
#   - Key functions: pack_bytes_to_words, unpack_words_to_bytes, enum_build_from_bytes


from momijo.enum import EnumValue

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