# Module: momijo.enum.enum
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
from .union import words_needed
from .repr import EnumRepr

# Data structure representing a concept in the enum library.
struct EnumValue:
    var tag: UInt64
    var w0: UInt64
    var w1: UInt64
    var w2: UInt64
    var w3: UInt64

# Does: utility function in enum module.
# Inputs: desc, tag, words, size_bytes.
# Returns: result value or status.
fn enum_build_explicit(desc: EnumRepr, tag: UInt64, words: List[UInt64], size_bytes: UInt64) -> EnumValue:
    var v = EnumValue(tag=tag, w0=0, w1=0, w2=0, w3=0)
    var used = Int(words_needed(size_bytes))
    if used > 0: v.w0 = words[0]
    if used > 1: v.w1 = words[1]
    if used > 2: v.w2 = words[2]
    if used > 3: v.w3 = words[3]
    return v

# Does: utility function in enum module.
# Inputs: v.
# Returns: result value or status.
fn enum_tag(v: EnumValue) -> UInt64:
    return v.tag

# Does: utility function in enum module.
# Inputs: v, idx.
# Returns: result value or status.
fn enum_payload_word(v: EnumValue, idx: Int) -> UInt64:
    if idx == 0: return v.w0
    if idx == 1: return v.w1
    if idx == 2: return v.w2
    if idx == 3: return v.w3
    return 0