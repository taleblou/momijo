# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Module: momijo.enum.jumptable
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
from .match import Case

const JT_U64: UInt64 = 0
const JT_U32: UInt64 = 1
const JT_U16: UInt64 = 2

# Data structure representing a concept in the enum library.
struct JumpTable:
    var base: UInt64
    var span: Int
    var default_arm: UInt64
    var kind: UInt64
    var data64: List[UInt64]
    var data32: List[UInt32]
    var data16: List[UInt16]

# Does: utility function in enum module.
# Inputs: cases.
# Returns: result value or status.
fn _max_arm(cases: List[Case]) -> UInt64:
    var m: UInt64 = 0
    for i in range(0, len(cases)):
        if cases[i].arm > m: m = cases[i].arm
    return m

# Does: utility function in enum module.
# Inputs: cases, default_arm.
# Returns: result value or status.
fn build_jump_table(cases: List[Case], default_arm: UInt64) -> JumpTable:
    if len(cases) == 0:
        return JumpTable(base=0, span=0, default_arm=default_arm, kind=JT_U64, data64=List[UInt64](0), data32=List[UInt32](0), data16=List[UInt16](0))
    var lo = cases[0].tag; var hi = cases[0].tag
    for i in range(1, len(cases)):
        var t = cases[i].tag
        if t < lo: lo = t
        if t > hi: hi = t
    var span = Int((hi - lo) + 1)
    var maxa = _max_arm(cases)
    var kind: UInt64 = JT_U64
    if maxa <= 0xFFFF: kind = JT_U16
    elif maxa <= 0xFFFFFFFF: kind = JT_U32
    var dt64 = List[UInt64](0); var dt32 = List[UInt32](0); var dt16 = List[UInt16](0)
    if kind == JT_U64:
        dt64 = List[UInt64](span)
        for i in range(0, span): dt64[i] = default_arm
        for i in range(0, len(cases)): dt64[Int(cases[i].tag - lo)] = cases[i].arm
    elif kind == JT_U32:
        dt32 = List[UInt32](span)
        for i in range(0, span): dt32[i] = UInt32(default_arm)
        for i in range(0, len(cases)): dt32[Int(cases[i].tag - lo)] = UInt32(cases[i].arm)
    else:
        dt16 = List[UInt16](span)
        for i in range(0, span): dt16[i] = UInt16(default_arm)
        for i in range(0, len(cases)): dt16[Int(cases[i].tag - lo)] = UInt16(cases[i].arm)
    return JumpTable(base=lo, span=span, default_arm=default_arm, kind=kind, data64=dt64, data32=dt32, data16=dt16)

# Does: utility function in enum module.
# Inputs: tag, jt.
# Returns: result value or status.
fn jump_lookup(tag: UInt64, jt: JumpTable) -> UInt64:
    if jt.span == 0: return jt.default_arm
    if tag < jt.base: return jt.default_arm
    var idx = Int(tag - jt.base)
    if idx < 0 or idx >= jt.span: return jt.default_arm
    if jt.kind == JT_U64: return jt.data64[idx]
    if jt.kind == JT_U32: return UInt64(jt.data32[idx])
    return UInt64(jt.data16[idx])