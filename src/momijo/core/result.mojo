# Project:      Momijo
# Module:       src.momijo.core.result
# File:         result.mojo
# Path:         src/momijo/core/result.mojo
#
# Description:  src.momijo.core.result — focused Momijo functionality with a stable public API.
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
#   - Structs: Result
#   - Key functions: __copyinit__, __init__, ok, err, is_ok, is_err, unwrap, unwrap_err ...
#   - Static methods present.
#   - Uses generic functions/types with explicit trait bounds.


from momijo.core.error import module
from momijo.tensor.broadcast import valid
from momijo.utils.result import f, g
from pathlib import Path
from pathlib.path import Path

struct Result[T: Copyable & Movable, E: Copyable & Movable](Copyable, Movable):
    var _ok: Bool
    var _val: List[T]   # empty or length-1 when _ok == True
    var _err: List[E]   # empty or length-1 when _ok == False
fn __copyinit__(out self, other: Self) -> None:
        self._ok = other._ok
        self._val = other._val
        self._err = other._err
fn __init__(out self, ok: Bool = False) -> None:
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