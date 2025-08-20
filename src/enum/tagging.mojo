# Module: momijo.enum.tagging
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
from momijo.enum.runtime import tag_bits_from_alignment, tag_pack_ptr, tag_unpack_ptr# Does: utility function in enum module.
# Inputs: align.
# Returns: result value or status.

# Does: utility function in enum module.
# Inputs: ptr, tag, bits.
# Returns: result value or status.
    return (ptr & ~mask) | (tag & mask)

# Does: utility function in enum module.
# Inputs: packed, bits.
# Returns: result value or status.
    var tag = packed & mask
    var ptr = packed & ~mask
    return (ptr, tag)