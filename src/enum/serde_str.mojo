#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
from .meta import EnumMetaInfo, name_of, tag_of
from .enumset import EnumSet, enumset_from_meta, enumset_add, enumset_iter_names

fn enum_tag_to_string(meta: EnumMetaInfo, tag: UInt64) -> String:
    return name_of(meta, tag)

fn enum_tag_from_string(meta: EnumMetaInfo, name: String) -> (Bool, UInt64):
    return tag_of(meta, name)

fn enumset_to_string(meta: EnumMetaInfo, s: EnumSet) -> String:
    var parts = enumset_iter_names(s, meta)
    var out = String("")
    for i in range(0, len(parts)):
        out = out + (String(",") if i>0 else String("")) + parts[i]
    return out

fn enumset_from_string(meta: EnumMetaInfo, csv: String) -> EnumSet:
    var s = enumset_from_meta(meta)
    var name = String("")
    for i in range(0, len(csv)):
        var ch = csv[i]
        if ch == ',':
            var (ok, tag) = tag_of(meta, name)
            if ok: enumset_add(s, tag)
            name = String("")
        else:
            name = name + String(ch)
    if len(name) > 0:
        var (ok, tag) = tag_of(meta, name)
        if ok: enumset_add(s, tag)
    return s
