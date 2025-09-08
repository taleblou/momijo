# Project:      Momijo
# Module:       src.momijo.core.option
# File:         option.mojo
# Path:         src/momijo/core/option.mojo
#
# Description:  src.momijo.core.option — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Key functions: __copyinit__, __init__, some, none, is_some, is_none, unwrap, unwrap_or ...
#   - Static methods present.
#   - Uses generic functions/types with explicit trait bounds.


struct Option[T: Copyable & Movable & Defaultable](Copyable, Movable):
    var _has: Bool
    var _val: T
fn __copyinit__(out self, other: Self) -> None:
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