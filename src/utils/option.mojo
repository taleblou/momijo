# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.utils
# File: src/momijo/utils/option.mojo
#
# Checklist compliance:
# - Only 'var' (no 'let'); no 'export'; no global mutable state
# - Explicit imports; English comments
# - Prefer mut/out over inout
# - Struct constructors use `fn __init__(out self, ...)`
# - Provide __copyinit__ to satisfy Copyable semantics where needed
# - Minimal, predictable API (no exceptions here)
#
# A lightweight, generic Option[T] with ergonomic helpers.
# T must be Copyable & Movable & Defaultable so that we can safely store/return values.

from stdlib.string import String
from stdlib.list import List

# -------------------------------------
# Option[T]
# -------------------------------------
struct Option[T: Copyable & Movable & Defaultable]:
    has: Bool
    value: T

    fn __init__(out self):
        self.has = False
        self.value = T()

    fn __init__(out self, v: T):
        self.has = True
        self.value = v

    fn __copyinit__(out self, other: Self):
        self.has = other.has
        self.value = other.value

    # Basic predicates
    fn is_some(self) -> Bool:
        return self.has

    fn is_none(self) -> Bool:
        return not self.has

    # Mutators
    fn set(mut self, v: T):
        self.has = True
        self.value = v

    fn clear(mut self):
        self.has = False
        self.value = T()

    # Take the value, leaving None
    fn take(mut self) -> Option[T]:
        if self.has:
            var out = Option[T](self.value)
            self.clear()
            return out
        return Option[T]()

    # Observers
    fn get(self) -> T:
        if self.has: return self.value
        return T()

    fn get_or(self, default: T) -> T:
        if self.has: return self.value
        return default

    fn to_bool(self) -> Bool:
        return self.has

    # Combinators
    fn or(self, other: Option[T]) -> Option[T]:
        if self.has: return self
        return other

    fn and(self, other: Option[T]) -> Option[T]:
        if self.has: return other
        return Option[T]()

# -------------------------------------
# Helpers & factories
# -------------------------------------
fn some[T: Copyable & Movable & Defaultable](v: T) -> Option[T]:
    var o = Option[T](v)
    return o

fn none[T: Copyable & Movable & Defaultable]() -> Option[T]:
    var o = Option[T]()
    return o

# -------------------------------------
# Option over String shortcuts
# -------------------------------------
fn some_str(v: String) -> Option[String]:
    var o = Option[String](v)
    return o

fn none_str() -> Option[String]:
    var o = Option[String]()
    return o

# -------------------------------------
# Self test (no prints)
# -------------------------------------
fn _self_test() -> Bool:
    var ok = True

    var a = none[Int64]()
    ok = ok and a.is_none()
    ok = ok and (a.get_or(Int64(7)) == Int64(7))

    var b = some[Int64](Int64(42))
    ok = ok and b.is_some()
    ok = ok and (b.get() == Int64(42))

    var c = b.and(some[Int64](Int64(9)))
    ok = ok and (c.get() == Int64(9))

    var d = a.or(some[Int64](Int64(5)))
    ok = ok and (d.get() == Int64(5))

    var t = d.take()
    ok = ok and t.is_some()
    ok = ok and d.is_none()

    var s = some_str(String("hi"))
    ok = ok and s.is_some()

    return ok
