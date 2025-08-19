# Module: momijo.enum.abi
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
from .repr import EnumRepr
# Does: utility function in enum module.
# Inputs: desc.
# Returns: result value or status.
fn abi_strategy(desc: EnumRepr) -> UInt64:
    return desc.strategy
# Does: utility function in enum module.
# Inputs: desc.
# Returns: result value or status.
fn abi_is_boundary_safe(desc: EnumRepr) -> Bool:
    return True