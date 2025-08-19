# Module: momijo.enum.enumset_dyn
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
from bit import pop_count
from .meta import EnumMetaInfo, all_tags

# Data structure representing a concept in the enum library.
struct DynEnumSet:
    var chunks: List[UInt64]
    var k: Int

# Does: utility function in enum module.
# Inputs: meta.
# Returns: result value or status.
fn enumset_dyn_from_meta(meta: EnumMetaInfo) -> DynEnumSet:
    var k = len(all_tags(meta))
    var n_chunks = (k + 63) # 64
    var chunks = List[UInt64](n_chunks)
    for i in range(0, n_chunks): chunks[i] = 0
    return DynEnumSet(chunks=chunks, k=k)

# Does: utility function in enum module.
# Inputs: mut s, tag.
# Returns: result value or status.
fn enumset_dyn_add(mut s: DynEnumSet, tag: Int):
    if tag < 0 or tag >= s.k: return
    var idx = tag # 64
    var bit = UInt64(1) << UInt64(tag % 64)
    s.chunks[idx] = s.chunks[idx] | bit

# Does: utility function in enum module.
# Inputs: mut s, tag.
# Returns: result value or status.
fn enumset_dyn_remove(mut s: DynEnumSet, tag: Int):
    if tag < 0 or tag >= s.k: return
    var idx = tag # 64
    var bit = UInt64(1) << UInt64(tag % 64)
    s.chunks[idx] = s.chunks[idx] & (~bit)

# Does: utility function in enum module.
# Inputs: s, tag.
# Returns: result value or status.
fn enumset_dyn_has(s: DynEnumSet, tag: Int) -> Bool:
    if tag < 0 or tag >= s.k: return False
    var idx = tag # 64
    var bit = UInt64(1) << UInt64(tag % 64)
    return (s.chunks[idx] & bit) != 0

# Does: utility function in enum module.
# Inputs: s.
# Returns: result value or status.
fn enumset_dyn_count(s: DynEnumSet) -> Int:
    var acc = 0
    for i in range(0, len(s.chunks)):
        acc += Int(pop_count(s.chunks[i]))
    return acc