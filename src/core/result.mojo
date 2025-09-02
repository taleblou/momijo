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
# File: momijo/core/result.mojo

# Result[T, E] = Ok(T) | Err(E)
# This implementation does NOT require Defaultable for T or E.
# Internally stores the active payload in a small List to avoid default construction.

struct Result[
    T: Copyable & Movable,
    E: Copyable & Movable
](Copyable, Movable):
    var _ok: Bool
    var _val: List[T]   # empty or length-1 when _ok == True
    var _err: List[E]   # empty or length-1 when _ok == False

    fn __copyinit__(out self, other: Self):
        self._ok = other._ok
        self._val = other._val
        self._err = other._err

    fn __init__(out self, ok: Bool = False):
        self._ok = ok
        self._val = List[T]()
        self._err = List[E]()

    # --- factories ---
    @staticmethod
    fn ok(v: T) -> Result[T, E]:
        var r = Result[T, E](ok=True)
        r._val.append(v)
        return r

    @staticmethod
    fn err(e: E) -> Result[T, E]:
        var r = Result[T, E](ok=False)
        r._err.append(e)
        return r

    # --- queries ---
    fn is_ok(self) -> Bool:  return self._ok
    fn is_err(self) -> Bool: return not self._ok

    # --- getters (valid only when the variant matches) ---
    fn unwrap(self) -> T:
        return self._val[0]

    fn unwrap_err(self) -> E:
        return self._err[0]

    fn unwrap_or(self, default_value: T) -> T:
        if self._ok: return self._val[0]
        return default_value

    fn unwrap_or_else(self, recover: fn(E) -> T) -> T:
        if self._ok: return self._val[0]
        return recover(self._err[0])

    # --- transforms ---
    fn map[U: Copyable & Movable](self, f: fn(T) -> U) -> Result[U, E]:
        if self._ok:
            return Result[U, E].ok(f(self._val[0]))
        return Result[U, E].err(self._err[0])

    fn map_err[F: Copyable & Movable](self, g: fn(E) -> F) -> Result[T, F]:
        if self._ok:
            return Result[T, F].ok(self._val[0])
        return Result[T, F].err(g(self._err[0]))

    fn and_then[U: Copyable & Movable](self, f: fn(T) -> Result[U, E]) -> Result[U, E]:
        if self._ok:
            return f(self._val[0])
        return Result[U, E].err(self._err[0])

    fn or_else[F: Copyable & Movable](self, g: fn(E) -> Result[T, F]) -> Result[T, F]:
        if self._ok:
            return Result[T, F].ok(self._val[0])
        return g(self._err[0])

    fn map_or[U: Copyable & Movable](self, default_value: U, f: fn(T) -> U) -> U:
        if self._ok: return f(self._val[0])
        return default_value

    fn map_or_else[U: Copyable & Movable](self, g: fn(E) -> U, f: fn(T) -> U) -> U:
        if self._ok: return f(self._val[0])
        return g(self._err[0])

    fn to_string(self) -> String:
        if self._ok: return "Ok(...)"
        return "Err(...)"
