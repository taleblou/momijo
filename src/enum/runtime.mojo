# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Module: momijo.enum.runtime
fn enum_desc(tag: Int) -> String:
    return String("tag=") + String(tag)
fn enum_build_explicit(tag: Int, payload: UInt64) -> UInt64:
    var t = UInt64(tag) << UInt64(32)
    return t | (payload & UInt64(0xFFFFFFFF))
fn enum_tag(packed: UInt64) -> Int:
    return Int((packed >> UInt64(32)) & UInt64(0xFFFFFFFF))
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
