# Project:      Momijo
# Module:       src.momijo.enum.strenum
# File:         strenum.mojo
# Path:         src/momijo/enum/strenum.mojo
#
# Description:  src.momijo.enum.strenum — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Key functions: strenum_to_string, strenum_from_string


from .meta import EnumMetaInfo, tag_of, name_of

# Does: utility function in enum module.
# Inputs: meta, tag.
# Returns: result value or status.
fn strenum_to_string(meta: EnumMetaInfo, tag: UInt64) -> String:
    return name_of(meta, tag)

# Does: utility function in enum module.
# Inputs: meta, name.
# Returns: result value or status.
fn strenum_from_string(meta: EnumMetaInfo, name: String) -> (Bool, UInt64):
    return tag_of(meta, name)