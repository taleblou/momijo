# Project:      Momijo
# Module:       src.momijo.utils.result
# File:         result.mojo
# Path:         src/momijo/utils/result.mojo
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
#   - Structs: Result, Option
#   - Key functions: __init__, __init__, __copyinit__, is_ok, is_err, Ok, Err, get_ok ...
#   - Static methods present.
#   - Uses generic functions/types with explicit trait bounds.


from stdlib.string import String

struct Result[T: Copyable & Movable & Defaultable, E: Copyable & Movable & Defaultable]:
    ok: Bool
    val: T
    err: E
fn __init__(out self) -> None:
        self.ok = False
        self.val = T()
        self.err = E()
fn __init__(out self, ok: Bool, val: T, err: E) -> None:
        self.ok = ok
        self.val = val
        self.err = err
fn __copyinit__(out self, other: Self) -> None:
        self.ok = other.ok
        self.val = other.val
        self.err = other.err

    # Predicates
fn is_ok(self) -> Bool: return self.ok
fn is_err(self) -> Bool: return not self.ok

    # Constructors
    @staticmethod
fn Ok(v: T) -> Self:
        var r = Result[T, E](True, v, E())
        return r

    @staticmethod
fn Err(e: E) -> Self:
        var r = Result[T, E](False, T(), e)
        return r

    # Observers
fn get_ok(self) -> T:
        if self.ok: return self.val
        return T()
fn get_err(self) -> E:
        if not self.ok: return self.err
        return E()

    # Fallbacks
fn unwrap_or(self, default: T) -> T:
        if self.ok: return self.val
        return default
fn unwrap_err_or(self, default: E) -> E:
        if not self.ok: return self.err
        return default

    # Combinators
    fn map[U: Copyable & Movable & Defaultable](self, f: fn(T) -> U) -> Result[U, E]:
        if self.ok:
            var u = f(self.val)
            return Result[U, E].Ok(u)
        return Result[U, E].Err(self.err)

    fn map_err[F: Copyable & Movable & Defaultable](self, g: fn(E) -> F) -> Result[T, F]:
        if self.ok:
            return Result[T, F].Ok(self.val)
        var e2 = g(self.err)
        return Result[T, F].Err(e2)

    fn and_then[U: Copyable & Movable & Defaultable](self, f: fn(T) -> Result[U, E]) -> Result[U, E]:
        if self.ok:
            return f(self.val)
        return Result[U, E].Err(self.err)

    fn or_else[F: Copyable & Movable & Defaultable](self, g: fn(E) -> Result[T, F]) -> Result[T, F]:
        if self.ok:
            return Result[T, F].Ok(self.val)
        return g(self.err)

# -------------------------------------
# Helpers / factories
# -------------------------------------
fn ok[T: Copyable & Movable & Defaultable, E: Copyable & Movable & Defaultable](v: T) -> Result[T, E]:
    return Result[T, E].Ok(v)

fn err[T: Copyable & Movable & Defaultable, E: Copyable & Movable & Defaultable](e: E) -> Result[T, E]:
    return Result[T, E].Err(e)

# Convert Option[T] to Result[T, E] with provided error
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
fn is_some(self) -> Bool: return self.has
fn is_none(self) -> Bool: return not self.has
fn get(self) -> T:
        assert(self is not None, String("self is None"))
        if self.has: return self.value()
        return T()

fn result_from_option[T: Copyable & Movable & Defaultable, E: Copyable & Movable & Defaultable](o: Option[T], e: E) -> Result[T, E]:
    if o.is_some():
        return Result[T, E].Ok(o.get())
    return Result[T, E].Err(e)

# -------------------------------------

# -------------------------------------
fn result_debug[T: Copyable & Movable & Defaultable, E: Copyable & Movable & Defaultable](r: Result[T, E]) -> String:
    if r.is_ok():
        return String("Ok(") + String(String(r.get_ok())) + String(")")
    return String("Err(") + String(String(r.get_err())) + String(")")

# -------------------------------------
# Self-test (no prints)
# -------------------------------------
fn _self_test() -> Bool:
    var ok_all = True

    var a = ok[Int64, String](Int64(5))
    ok_all = ok_all and a.is_ok() and (a.get_ok() == Int64(5))

    var b = err[Int64, String](String("e"))
    ok_all = ok_all and b.is_err() and (b.get_err() == String("e"))

    # map
fn f(x: Int64) -> Int64: return x + Int64(1)
    var c = a.map[Int64](f)
    ok_all = ok_all and (c.get_ok() == Int64(6))

    # and_then
fn g(x: Int64) -> Result[Int64, String]:
        if x > Int64(0): return ok[Int64, String](x * Int64(2))
        return err[Int64, String](String("neg"))
    var d = c.and_then[Int64](g)
    ok_all = ok_all and d.is_ok() and (d.get_ok() == Int64(12))

    # map_err
fn geh(e: String) -> String: return e + String("!")
    var e1 = b.map_err[String](geh)
    ok_all = ok_all and e1.is_err() and (e1.get_err() == String("e!"))

    # option -> result
    var o = Option[Int64](Int64(7))
    var r = result_from_option[Int64, String](o, String("none"))
    ok_all = ok_all and r.is_ok() and (r.get_ok() == Int64(7))

    # unwrap_or
    ok_all = ok_all and (b.unwrap_or(Int64(9)) == Int64(9))

    return ok_all