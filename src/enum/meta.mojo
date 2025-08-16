#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
struct EnumMetaInfo:
    var names: List[String]
    var tags:  List[UInt64]

fn make_meta(names: List[String], tags: List[UInt64]) -> EnumMetaInfo:
    return EnumMetaInfo(names=names, tags=tags)

fn name_of(meta: EnumMetaInfo, tag: UInt64) -> String:
    for i in range(0, len(meta.tags)):
        if meta.tags[i] == tag:
            return meta.names[i]
    return String("<unknown>")

fn tag_of(meta: EnumMetaInfo, name: String) -> (Bool, UInt64):
    for i in range(0, len(meta.names)):
        if meta.names[i] == name:
            return (True, meta.tags[i])
    return (False, 0)

fn all_names(meta: EnumMetaInfo) -> List[String]:
    return meta.names

fn all_tags(meta: EnumMetaInfo) -> List[UInt64]:
    return meta.tags

# Auto generators
fn make_meta_from_names(names: List[String]) -> EnumMetaInfo:
    var tags = List[UInt64](len(names))
    for i in range(0, len(names)):
        tags[i] = UInt64(i)
    return make_meta(names, tags)

fn make_meta_dense(k: Int, prefix: String = String("V")) -> EnumMetaInfo:
    var names = List[String](k)
    var tags = List[UInt64](k)
    for i in range(0, k):
        names[i] = prefix + String(i)
        tags[i] = UInt64(i)
    return make_meta(names, tags)

fn make_meta_from_pairs(pairs: List[(String, UInt64)]) -> EnumMetaInfo:
    var n = len(pairs)
    var names = List[String](n)
    var tags  = List[UInt64](n)
    for i in range(0, n):
        names[i] = pairs[i].0
        tags[i]  = pairs[i].1
    return make_meta(names, tags)
