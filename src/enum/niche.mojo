#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
struct OptionPtr:
    var raw: UInt64

fn some_ptr(p: UInt64) -> OptionPtr:
    return OptionPtr(raw=p if p != 0 else 1)

fn none_ptr() -> OptionPtr:
    return OptionPtr(raw=0)

fn is_none(o: OptionPtr) -> Bool:
    return o.raw == 0
