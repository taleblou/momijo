#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
from bit import pop_count
from .meta import EnumMetaInfo, all_tags, name_of

struct EnumSet:
    var bits: UInt64
    var k: Int

fn enumset_empty_k(k: Int) -> EnumSet:
    return EnumSet(bits=0, k=k)

fn enumset_from_meta(meta: EnumMetaInfo) -> EnumSet:
    return EnumSet(bits=0, k=len(all_tags(meta)))

fn enumset_single(tag: UInt64, k: Int) -> EnumSet:
    if Int(tag) >= k: return EnumSet(bits=0, k=k)
    return EnumSet(bits=(1 << tag), k=k)

fn enumset_add(inout s: EnumSet, tag: UInt64):
    if Int(tag) < s.k: s.bits = s.bits | (1 << tag)

fn enumset_remove(inout s: EnumSet, tag: UInt64):
    if Int(tag) < s.k: s.bits = s.bits & (~(1 << tag))

fn enumset_has(s: EnumSet, tag: UInt64) -> Bool:
    if Int(tag) >= s.k: return False
    return (s.bits & (1 << tag)) != 0

fn enumset_count(s: EnumSet) -> Int:
    return Int(pop_count(s.bits))

fn enumset_union(a: EnumSet, b: EnumSet) -> EnumSet:
    var k = a.k if a.k >= b.k else b.k
    return EnumSet(bits=(a.bits | b.bits), k=k)

fn enumset_intersect(a: EnumSet, b: EnumSet) -> EnumSet:
    var k = a.k if a.k >= b.k else b.k
    return EnumSet(bits=(a.bits & b.bits), k=k)

fn enumset_minus(a: EnumSet, b: EnumSet) -> EnumSet:
    return EnumSet(bits=(a.bits & (~b.bits)), k=a.k)

fn enumset_iter_tags(s: EnumSet) -> List[UInt64]:
    var out = List[UInt64](0)
    for t in range(0, s.k):
        if (s.bits & (1 << UInt64(t))) != 0:
            out.append(UInt64(t))
    return out

fn enumset_iter_names(s: EnumSet, meta: EnumMetaInfo) -> List[String]:
    var out = List[String](0)
    var tags = enumset_iter_tags(s)
    for i in range(0, len(tags)):
        out.append(name_of(meta, tags[i]))
    return out
