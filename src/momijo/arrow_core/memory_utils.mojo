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
# Project: momijo.arrow_core
# File: src/momijo/arrow_core/memory_utils.mojo

from momijo.dataframe.diagnostics import safe
from momijo.dataframe.helpers import t
from momijo.nn.parameter import data
from pathlib import Path
from pathlib.path import Path

struct ReplaceBox[T: Movable]:
    var value: T
fn __init__(out self, value: T) -> None:
        assert(self is not None, String("self is None"))
        self.value() = value

# Return (old_value, updated_box) by moving from the current box into a local first.
fn replace_value[T: Movable](current: ReplaceBox[T], new_value: T) -> (old_value: T, updated: ReplaceBox[T]):
    assert(current is not None, String("current is None"))
    var old_local = current.value()       # move from parameter's field into a local
    var updated = ReplaceBox[T](new_value)
    return (old_local, updated)         # return locals (safe move)

# Wrap a value into Optional by moving it in.
fn into_optional[T: Movable](x: T) -> Optional[T]:
    var local = x                        # move parameter into a local first
    return Optional[T](local)            # return a local (safe move)

# Try to extract a value from Optional without forcing Defaultable.
# Caller must check `present` before reading `value`.
fn try_take_optional[T: Movable](opt: Optional[T]) -> (present: Bool, value: T):
    if opt.is_some():
        # Move into a local and return that local
        var v = opt.unwrap()
        return (True, v)
    # Return path when empty: we must still return a T; document that it is undefined and not to be used.
    # (We avoid requiring Defaultable here.)
    return (False, __undefined_value[T]())

# Low-level placeholder for an undefined value. Do not use unless guarded by `present == True`.
fn __undefined_value[T]() -> T:
    # This is a placeholder; replace with a proper intrinsic once available in your toolchain.
    return (0 as Int).as_any() as! T

# Module identity helpers
fn __module_name__() -> String:
    return String("momijo/arrow_core/memory_utils.mojo")
fn __self_test__() -> Bool:
    # ReplaceBox smoke test
    var b = ReplaceBox[String](String("old"))
    var (old_v, updated) = replace_value[String](b, String("new"))
    if !(old_v == String("old")): return False
    assert(updated is not None, String("updated is None"))
    if !(updated.value() == String("new")): return False

    # Optional smoke test
    var some = into_optional(String("xyz"))
    var (ok, v) = try_take_optional[String](some)
    if !ok: return False
    if !(v == String("xyz")): return False

    # Empty-optional branch (just ensure it doesn't crash)
    var empty = Optional[String]()
    var (ok2, _) = try_take_optional[String](empty)
    if ok2: return False

    return True