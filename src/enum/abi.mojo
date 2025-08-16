#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
from .repr import EnumRepr
fn abi_strategy(desc: EnumRepr) -> UInt64:
    return desc.strategy
fn abi_is_boundary_safe(desc: EnumRepr) -> Bool:
    return True
