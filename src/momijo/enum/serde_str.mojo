# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.enum
# File: momijo/enum/serde_str.mojo


from .enumset import EnumSet, enumset_add, enumset_from_meta, enumset_iter_names
from momijo.core.error import module
from momijo.enum.abi import enumset_add
from momijo.enum.enumset import EnumSet
from momijo.enum.meta import EnumMetaInfo, name_of, tag_of
from pathlib import Path
from pathlib.path import Path

# Does: utility function in enum module.
# Inputs: meta, tag.
# Returns: result value or status.
fn enum_tag_to_string(meta: EnumMetaInfo, tag: UInt64) -> String:
    return name_of(meta, tag)

# Does: utility function in enum module.
# Inputs: meta, name.
# Returns: result value or status.
fn enum_tag_from_string(meta: EnumMetaInfo, name: String) -> (Bool, UInt64):
    return tag_of(meta, name)

# Does: utility function in enum module.
# Inputs: meta, s.
# Returns: result value or status.
fn enumset_to_string(meta: EnumMetaInfo, s: EnumSet) -> String:
    var parts = enumset_iter_names(s, meta)
    var out = String("")
    for i in range(0, len(parts)):
        out = out + (String(",") if i>0 else String("")) + parts[i]
    return out

# Does: utility function in enum module.
# Inputs: meta, csv.
# Returns: result value or status.
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