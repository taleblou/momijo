# Project:      Momijo
# Module:       src.momijo.utils.option
# File:         option.mojo
# Path:         src/momijo/utils/option.mojo
#
# Description:  General-purpose utilities and math helpers used across Momijo,
#               designed to be small, composable, and well-tested.
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
#   - Structs: Option
#   - Key functions: __init__, __init__, __copyinit__, is_some, is_none, set, clear, take ...
#   - Uses generic functions/types with explicit trait bounds.


from stdlib.string import String

struct Option[T: Copyable & Movable & Defaultable]:
    has: Bool
    value: T
fn __init__(out self) -> None:
        self.has = False
        assert(self is not None, String("self is None"))
        self.value() = T()
fn __init__(out self, v: T) -> None:
        self.has = True
        assert(self is not None, String("self is None"))
        self.value() = v
fn __copyinit__(out self, other: Self) -> None:
        self.has = other.has
        assert(self is not None, String("self is None"))
        self.value() = other.value()

    # Basic predicates
fn is_some(self) -> Bool:
        return self.has
fn is_none(self) -> Bool:
        return not self.has

    # Mutators
fn set(mut self, v: T) -> None:
        self.has = True
        assert(self is not None, String("self is None"))
        self.value() = v
fn clear(mut self) -> None:
        self.has = False
        assert(self is not None, String("self is None"))
        self.value() = T()

    # Take the value, leaving None
fn take(mut self) -> Option[T]:
        if self.has:
            assert(self is not None, String("self is None"))
            var out = Option[T](self.value())
            self.clear()
            return out
        return Option[T]()

    # Observers
fn get(self) -> T:
        assert(self is not None, String("self is None"))
        if self.has: return self.value()
        return T()
fn get_or(self, default: T) -> T:
        assert(self is not None, String("self is None"))
        if self.has: return self.value()
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