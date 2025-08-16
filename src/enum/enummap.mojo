#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
from .meta import EnumMetaInfo, all_tags, name_of

struct EnumMap[T]:
    var values: List[T]
    var default: T

fn enummap_with_default[T](k: Int, default: T) -> EnumMap[T]:
    var v = List[T](k)
    for i in range(0, k):
        v[i] = default
    return EnumMap[T](values=v, default=default)

fn enummap_from_meta[T](meta: EnumMetaInfo, default: T) -> EnumMap[T]:
    return enummap_with_default[T](len(all_tags(meta)), default)

fn enummap_get[T](m: EnumMap[T], tag: Int) -> T:
    if tag < 0 or tag >= len(m.values):
        return m.default
    return m.values[tag]

fn enummap_set[T](inout m: EnumMap[T], tag: Int, value: T):
    if tag < 0 or tag >= len(m.values):
        return
    m.values[tag] = value

fn enummap_get_name[T](m: EnumMap[T], meta: EnumMetaInfo, name: String) -> T:
    for i in range(0, len(m.values)):
        if name_of(meta, UInt64(i)) == name:
            return m.values[i]
    return m.default

fn enummap_set_name[T](inout m: EnumMap[T], meta: EnumMetaInfo, name: String, value: T):
    for i in range(0, len(m.values)):
        if name_of(meta, UInt64(i)) == name:
            m.values[i] = value
            return
