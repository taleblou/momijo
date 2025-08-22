# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.core
# File: momijo/core/result.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 
 
from momijo.core.error import MomijoError

struct Result[T](Copyable, Movable):
    var _error: MomijoError
    var _value: T

    fn __init__(out self, value: T, error: MomijoError = MomijoError.none()):
        self._value = value
        self._error = error

    # ---------- factories ----------
    @staticmethod
    fn ok(value: T) -> Result[T]:
        return Result[T](value=value, error=MomijoError.none())

    @staticmethod
    fn fail(error: MomijoError, fallback: T) -> Result[T]:
        return Result[T](value=fallback, error=error)

    @staticmethod
    fn from_error(error: MomijoError, default_supplier: fn() -> T) -> Result[T]:
        return Result[T](value=default_supplier(), error=error)

    @staticmethod
    fn ok_if(cond: Bool, value: T, error: MomijoError, fallback: T) -> Result[T]:
        return cond ? Result[T].ok(value) : Result[T].fail(error, fallback)

    # ---------- predicates ----------
    fn is_ok(self) -> Bool:
        return self._error.is_ok()

    fn is_err(self) -> Bool:
        return not self._error.is_ok()

    # ---------- accessors ----------
    fn value(self) -> T:
        # Returns the stored value (even if error present).
        return self._value

    fn error(self) -> MomijoError:
        return self._error

    fn value_or(self, default_value: T) -> T:
        return self.is_ok() ? self._value : default_value

    fn replace_value(self, v: T) -> Result[T]:
        return Result[T](value=v, error=self._error)

    fn replace_error(self, e: MomijoError) -> Result[T]:
        return Result[T](value=self._value, error=e)

    # ---------- transformations ----------
    fn map[U](self, f: fn(T) -> U, fallback_on_error: U) -> Result[U]:
        if self.is_ok():
            return Result[U].ok(f(self._value))
        else:
            return Result[U].fail(self._error, fallback_on_error)

    fn map_err(self, g: fn(MomijoError) -> MomijoError) -> Result[T]:
        if self.is_err():
            return Result[T](value=self._value, error=g(self._error))
        return self

    fn and_then[U](self, f: fn(T) -> Result[U], fallback_on_error: U) -> Result[U]:
        if self.is_ok():
            return f(self._value)
        else:
            return Result[U].fail(self._error, fallback_on_error)

    fn or_else(self, g: fn(MomijoError) -> Result[T]) -> Result[T]:
        if self.is_ok():
            return self
        else:
            return g(self._error)

    fn tap(self, f: fn(T) -> None) -> Result[T]:
        if self.is_ok():
            f(self._value)
        return self

    fn tap_err(self, f: fn(MomijoError) -> None) -> Result[T]:
        if self.is_err():
            f(self._error)
        return self

    # ---------- combinators ----------
    fn zip[U, R](self, other: Result[U], f: fn(T, U) -> R, fallback_on_error: R) -> Result[R]:
        if self.is_ok() and other.is_ok():
            return Result[R].ok(f(self._value, other._value))
        # prefer self error if present
        var err = self.is_err() ? self._error : other._error
        return Result[R].fail(err, fallback_on_error)

    # ---------- utilities ----------
    fn match[U](self, on_ok: fn(T) -> U, on_err: fn(MomijoError, T) -> U) -> U:
        return self.is_ok() ? on_ok(self._value) : on_err(self._error, self._value)

    fn to_string(self) -> String:
        if self.is_ok():
            return "Result{Ok}"
        else:
            return "Result{Err " + self._error.describe() + "}"

# ------------- Free helpers -------------

@staticmethod
fn result_ok[T](value: T) -> Result[T]:
    return Result[T].ok(value)

@staticmethod
fn result_fail[T](error: MomijoError, fallback: T) -> Result[T]:
    return Result[T].fail(error, fallback)

@staticmethod
fn result_from_error[T](error: MomijoError, default_supplier: fn() -> T) -> Result[T]:
    return Result[T].from_error(error, default_supplier)

@staticmethod
fn result_zip[A, B, R](a: Result[A], b: Result[B], f: fn(A, B) -> R, fallback_on_error: R) -> Result[R]:
    return a.zip[B, R](b, f, fallback_on_error)

@staticmethod
fn collect_results[T](items: List[Result[T]], fallback_on_error: T) -> Result[List[T]]:
    var out = List[T]()
    out.reserve(len(items))
    var first_err = MomijoError.none()
    var has_err = False
    var i = 0
    while i < len(items):
        var r = items[i]
        if r.is_ok():
            out.append(r.value())
        else:
            if not has_err:
                first_err = r.error()
                has_err = True
            out.append(fallback_on_error)
        i += 1
    if has_err:
        return Result[List[T]].fail(first_err, out)
    return Result[List[T]].ok(out)