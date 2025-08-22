# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Module: momijo.enum.strenum
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