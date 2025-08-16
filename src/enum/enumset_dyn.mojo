#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
from bit import pop_count
from .meta import EnumMetaInfo, all_tags

struct DynEnumSet:
    var chunks: List[UInt64]
    var k: Int

fn enumset_dyn_from_meta(meta: EnumMetaInfo) -> DynEnumSet:
    var k = len(all_tags(meta))
    var n_chunks = (k + 63) // 64
    var chunks = List[UInt64](n_chunks)
    for i in range(0, n_chunks): chunks[i] = 0
    return DynEnumSet(chunks=chunks, k=k)

fn enumset_dyn_add(inout s: DynEnumSet, tag: Int):
    if tag < 0 or tag >= s.k: return
    var idx = tag // 64
    var bit = UInt64(1) << UInt64(tag % 64)
    s.chunks[idx] = s.chunks[idx] | bit

fn enumset_dyn_remove(inout s: DynEnumSet, tag: Int):
    if tag < 0 or tag >= s.k: return
    var idx = tag // 64
    var bit = UInt64(1) << UInt64(tag % 64)
    s.chunks[idx] = s.chunks[idx] & (~bit)

fn enumset_dyn_has(s: DynEnumSet, tag: Int) -> Bool:
    if tag < 0 or tag >= s.k: return False
    var idx = tag // 64
    var bit = UInt64(1) << UInt64(tag % 64)
    return (s.chunks[idx] & bit) != 0

fn enumset_dyn_count(s: DynEnumSet) -> Int:
    var acc = 0
    for i in range(0, len(s.chunks)):
        acc += Int(pop_count(s.chunks[i]))
    return acc
