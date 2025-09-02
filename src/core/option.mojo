# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.core
# File: momijo/core/option.mojo

# Simple Option<T> with static factories: some(v) and none().
# Conforms to your checklist: fn __init__(out self,...), no inout, uses var, String(...) formatting.

struct Option[T: Copyable & Movable & Defaultable](Copyable, Movable):
    var _has: Bool
    var _val: T

    fn __copyinit__(out self, other: Self):
        self._has = other._has
        self._val = other._val

    fn __init__(out self, has: Bool = False, value: T = T()):
        self._has = has
        self._val = value

    @staticmethod
    fn some(v: T) -> Option[T]:
        return Option[T](has=True, value=v)

    @staticmethod
    fn none() -> Option[T]:
        return Option[T](has=False)

    fn is_some(self) -> Bool: return self._has
    fn is_none(self) -> Bool: return not self._has

    fn unwrap(self) -> T: return self._val

    fn unwrap_or(self, default_value: T) -> T:
        if self._has: return self._val
        return default_value

    fn map[U: Copyable & Movable & Defaultable](self, f: fn(T) -> U) -> Option[U]:
        if self._has: return Option[U].some(f(self._val))
        return Option[U].none()

    fn and_then[U: Copyable & Movable & Defaultable](self, f: fn(T) -> Option[U]) -> Option[U]:
        if self._has: return f(self._val)
        return Option[U].none()

    fn to_string(self) -> String:
        if self._has: return "Some(...)"
        return "None"

