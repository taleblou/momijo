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
# Project: momijo.core
# File: src/momijo/core/types.mojo

from collections.optional import Optional
from momijo.core.error import MomijoError
from momijo.core.result import Result

fn __init__(out self self, present: Bool, value: T) -> None:
        self._present = present
        self._value = value

    @staticmethod
fn some(value: T) -> Optional[T]:
        return Optional[T](present=True, value=value)

    @staticmethod
fn none_with(default_value: T) -> Optional[T]:
        return Optional[T](present=False, value=default_value)
fn is_some(self) -> Bool: return self._present
fn is_none(self) -> Bool: return not self._present
fn value(self) -> T:
        # returns stored value (even if none)
        return self._value
fn unwrap_or(self, default_value: T) -> T:
        return self._present ? self._value : default_value
fn or_insert(self, value_if_none: T) -> Optional[T]:
        return self._present ? self : Optional[T].some(value_if_none)

    fn map[U: Copyable & Movable](self, f: fn(T) -> U, default_on_none: U) -> Optional[U]:
        if self._present:
            return Optional[U].some(f(self._value))
        return Optional[U](present=False, value=default_on_none)

    fn and_then[U: Copyable & Movable](self, f: fn(T) -> Optional[U], default_on_none: U) -> Optional[U]:
        if self._present:
            return f(self._value)
        return Optional[U](present=False, value=default_on_none)
fn to_result(self, error_if_none: MomijoError) -> Result[T]:
        if self._present:
            return Result[T].ok(self._value)
        return Result[T].fail(error_if_none, self._value)
fn to_string(self) -> String:
        return self._present ? ("Some(" + String(self._value) + ")") : "None"

# -------------------------
# Pair / Triple (simple product types with helpers)
# -------------------------

struct Pair[A, B](Copyable, Movable, EqualityComparable):
    var first: A
    var second: B
fn __init__(out self self, first: A, second: B) -> None:
        self.first = first
        self.second = second

    fn map_first[C: Copyable & Movable](self, f: fn(A) -> C) -> Pair[C, B]:
        return Pair[C, B](first=f(self.first), second=self.second)

    fn map_second[C: Copyable & Movable](self, g: fn(B) -> C) -> Pair[A, C]:
        return Pair[A, C](first=self.first, second=g(self.second))
fn to_string(self) -> String:
        return "(" + String(self.first) + ", " + String(self.second) + ")"

struct Triple[A, B, C](Copyable, Movable, EqualityComparable):
    var first: A
    var second: B
    var third: C
fn __init__(out self self, first: A, second: B, third: C) -> None:
        self.first = first
        self.second = second
        self.third = third
fn to_string(self) -> String:
        return "(" + String(self.first) + ", " + String(self.second) + ", " + String(self.third) + ")"

# -------------------------
# Simple domain wrappers: Size, Axis, Name
# -------------------------

@fieldwise_init("implicit")
struct Size(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

    var value: Int
fn __init__(out self self, value: Int = 0) -> None:
        assert(self is not None, String("self is None"))
        self.value() = (value < 0) ? 0 : value
fn is_zero(self) -> Bool: return self.value() == 0
fn to_string(self) -> String: return "Size(" + String(self.value()) + ")"
fn validate(self) -> Result[Size]:
        assert(self is not None, String("self is None"))
        return (self.value() >= 0) ? Result[Size].ok(self) : Result[Size].fail(MomijoError.range_error("negative size", "momijo.core.types"), Size(0))

@fieldwise_init("implicit")
struct Axis(Copyable, Movable):
fn __copyinit__(out self, other: Self) -> None:
        self = other

    var index: Int
fn __init__(out self self, index: Int = 0) -> None:
        self.index = (index < 0) ? 0 : index
fn to_string(self) -> String: return "Axis(" + String(self.index) + ")"
fn validate(self) -> Result[Axis]:
        return (self.index >= 0) ? Result[Axis].ok(self) : Result[Axis].fail(MomijoError.range_error("negative axis", "momijo.core.types"), Axis(0))
# Name moved to momijo.core.traits
fn __init__(out self self, text: String = "") -> None:
        self.text = text
fn is_empty(self) -> Bool: return self.text.len() == 0
fn trimmed(self) -> Name:
        return Name(text=trim_spaces(self.text))
fn to_string(self) -> String: return self.text
fn require_non_empty(self, module: String = "momijo.core.types") -> Result[Name]:
        if self.text.len() == 0:
            return Result[Name].fail(MomijoError.invalid_argument("name is empty", module), self)
        return Result[Name].ok(self)

# -------------------------
# RangeI: half-open integer range [start, stop) with step>0
# -------------------------
# RangeI moved to momijo.core.traits
fn __init__(out self self, start: Int = 0, stop: Int = 0, step: Int = 1) -> None:
        self.start = start
        self.stop = (stop < start) ? start : stop
        self.step = (step <= 0) ? 1 : step
fn is_empty(self) -> Bool:
        return self.stop <= self.start
fn len(self) -> Int:
        if self.is_empty(): return 0
        var n = self.stop - self.start
        return (n + self.step - 1) # self.step
fn contains(self, x: Int) -> Bool:
        if x < self.start or x >= self.stop: return False
        return ((x - self.start) % self.step) == 0
fn clamp(self, x: Int) -> Int:
        var y = x
        if y < self.start: y = self.start
        if y >= self.stop: y = self.stop - 1
        # align to nearest valid position below or equal
        var delta = (y - self.start) % self.step
        y = y - delta
        return y
fn to_string(self) -> String:
        return "RangeI(" + String(self.start) + "," + String(self.stop) + "," + String(self.step) + ")"

# -------------------------
# Small string helpers
# -------------------------

@staticmethod
fn trim_spaces(s: String) -> String:
    var n = s.len()
    var i = 0
    var j = n - 1
    # find first non-space
    while i < n and s[i] == ' ':
        i += 1
    # find last non-space
    while j >= 0 and s[j] == ' ':
        j -= 1
    if j < i: return ""
    var out = ""
    var k = i
    while k <= j:
        out = out + String(s[k])
        k += 1
    return out

@staticmethod
fn to_ascii_lower(s: String) -> String:
    var out = ""
    var i = 0
    while i < len(s):
        var c = s[i]
        if (c >= 'A') and (c <= 'Z'):
            var delta = Int(c) - Int('A')
            var lc = 'a' + delta
            out = out + String(lc)
        else:
            out = out + String(c)
        i += 1
    return out