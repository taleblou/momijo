#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
from .meta import EnumMetaInfo, tag_of, name_of

fn strenum_to_string(meta: EnumMetaInfo, tag: UInt64) -> String:
    return name_of(meta, tag)

fn strenum_from_string(meta: EnumMetaInfo, name: String) -> (Bool, UInt64):
    return tag_of(meta, name)
