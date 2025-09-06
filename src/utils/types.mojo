# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.utils
# File: src/momijo/utils/types.mojo
#
# Checklist compliance:
# - Only 'var' (no 'let'); no 'export'; no global mutable state
# - Explicit imports; English comments
# - Prefer mut/out over inout
# - Struct constructors via `fn __init__(out self, ...)` (+ __copyinit__ where needed)
# - Deterministic behavior; no exceptions; no prints in library
#
# Overview
# --------
# Foundational generic utility types to avoid repetition across Momijo:
#   - Pair[T, U], Triple[A, B, C]
#   - Either[L, R] (sum type with simple combinators)
#   - RangeI64: simple half-open integer range [start, end) with step=+1
#   - NonEmptyList[T]: wrapper that tracks non-emptiness, with safe accessors
#
# These are intentionally lightweight and purely data-oriented with minimal helpers.
# No operator overloading or implicit conversions to keep call-sites explicit and predictable.

from stdlib.list import List
from stdlib.string import String

# ---------------------------------------------
# Pair / Triple
# ---------------------------------------------

struct Pair[T: Copyable & Movable, U: Copyable & Movable]:
    first: T
    second: U

    fn __init__(out self):
        self.first = T()
        self.second = U()

    fn __init__(out self, a: T, b: U):
        self.first = a
        self.second = b

    fn __copyinit__(out self, other: Self):
        self.first = other.first
        self.second = other.second

    fn swap(mut self):
        var tmp = self.first
        self.first = self.second
        self.second = tmp

    fn map_first[V: Copyable & Movable](self, f: fn(T) -> V) -> Pair[V, U]:
        return Pair[V, U](f(self.first), self.second)

    fn map_second[V: Copyable & Movable](self, g: fn(U) -> V) -> Pair[T, V]:
        return Pair[T, V](self.first, g(self.second))

    fn to_string(self) -> String:
        return String("(") + String(String(self.first)) + String(", ") + String(String(self.second)) + String(")")

struct Triple[A: Copyable & Movable, B: Copyable & Movable, C: Copyable & Movable]:
    a: A
    b: B
    c: C

    fn __init__(out self):
        self.a = A()
        self.b = B()
        self.c = C()

    fn __init__(out self, a: A, b: B, c: C):
        self.a = a
        self.b = b
        self.c = c

    fn __copyinit__(out self, other: Self):
        self.a = other.a
        self.b = other.b
        self.c = other.c

    fn to_string(self) -> String:
        return String("(") + String(String(self.a)) + String(", ") + String(String(self.b)) + String(", ") + String(String(self.c)) + String(")")

# ---------------------------------------------
# Either (sum type)
# ---------------------------------------------

struct Either[L: Copyable & Movable & Defaultable, R: Copyable & Movable & Defaultable]:
    is_left: Bool
    left: L
    right: R

    fn __init__(out self):
        self.is_left = True
        self.left = L()
        self.right = R()

    fn __init__(out self, left: L, right: R, is_left: Bool):
        self.is_left = is_left
        self.left = left
        self.right = right

    fn __copyinit__(out self, other: Self):
        self.is_left = other.is_left
        self.left = other.left
        self.right = other.right

    @staticmethod
    fn Left(v: L) -> Self:
        return Either[L, R](v, R(), True)

    @staticmethod
    fn Right(v: R) -> Self:
        return Either[L, R](L(), v, False)

    fn is_right(self) -> Bool:
        return not self.is_left

    fn get_left(self) -> L:
        if self.is_left: return self.left
        return L()

    fn get_right(self) -> R:
        if not self.is_left: return self.right
        return R()

    fn map_left[L2: Copyable & Movable & Defaultable](self, f: fn(L) -> L2) -> Either[L2, R]:
        if self.is_left:
            return Either[L2, R].Left(f(self.left))
        return Either[L2, R].Right(self.right)

    fn map_right[R2: Copyable & Movable & Defaultable](self, g: fn(R) -> R2) -> Either[L, R2]:
        if self.is_left:
            return Either[L, R2].Left(self.left)
        return Either[L, R2].Right(g(self.right))

    fn to_string(self) -> String:
        if self.is_left:
            return String("Left(") + String(String(self.left)) + String(")")
        return String("Right(") + String(String(self.right)) + String(")")

# ---------------------------------------------
# RangeI64: half-open [start, end), step = +1
# ---------------------------------------------

struct RangeI64:
    start: Int64
    end: Int64

    fn __init__(out self):
        self.start = Int64(0)
        self.end = Int64(0)

    fn __init__(out self, start: Int64, end: Int64):
        if end < start:
            self.start = end
            self.end = start
        else:
            self.start = start
            self.end = end

    fn __copyinit__(out self, other: Self):
        self.start = other.start
        self.end = other.end

    fn is_empty(self) -> Bool:
        return self.start >= self.end

    fn contains(self, x: Int64) -> Bool:
        return x >= self.start and x < self.end

    fn len(self) -> Int64:
        if self.end <= self.start: return Int64(0)
        return self.end - self.start

    fn clamp_to(self, other: RangeI64) -> RangeI64:
        var a = self.start
        var b = self.end
        var c = other.start
        var d = other.end
        var s = a
        if c > s: s = c
        var e = b
        if d < e: e = d
        return RangeI64(s, e)

    fn to_string(self) -> String:
        return String("[") + String(String(self.start)) + String(", ") + String(String(self.end)) + String(")")

# ---------------------------------------------
# NonEmptyList[T]
# ---------------------------------------------

struct NonEmptyList[T: Copyable & Movable & Defaultable]:
    xs: List[T]
    nonempty: Bool

    fn __init__(out self):
        self.xs = List[T]()
        self.nonempty = False

    fn __init__(out self, xs: List[T]):
        self.xs = xs
        self.nonempty = len(xs) > 0

    fn __copyinit__(out self, other: Self):
        self.xs = other.xs
        self.nonempty = other.nonempty

    fn is_nonempty(self) -> Bool: return self.nonempty
    fn is_empty(self) -> Bool: return not self.nonempty

    fn head(self) -> T:
        if self.nonempty:
            return self.xs[0]
        return T()

    fn tail(self) -> List[T]:
        var out = List[T]()
        if not self.nonempty: return out
        var i = 1
        var n = len(self.xs)
        while i < n:
            out.append(self.xs[i])
            i += 1
        return out

    fn to_string(self) -> String:
        # simple representation: "NonEmpty(len=n)"
        return String("NonEmpty(len=") + String(String(Int64(len(self.xs)))) + String(")")

# ---------------------------------------------
# Self-test
# ---------------------------------------------

fn _self_test() -> Bool:
    var ok = True

    # Pair/Triple
    var p = Pair[Int64, String](Int64(1), String("a"))
    ok = ok and (p.first == Int64(1))
    var p2 = p.map_second[String](fn (s: String) -> String: return s + String("b"))
    ok = ok and (p2.second == String("ab"))

    var t = Triple[Int64, Int64, Int64](Int64(1), Int64(2), Int64(3))
    ok = ok and (t.b == Int64(2))

    # Either
    var e1 = Either[String, Int64].Left(String("oops"))
    ok = ok and e1.is_left
    var e2 = e1.map_left[String](fn (x: String) -> String: return x + String("!"))
    ok = ok and (e2.get_left() == String("oops!"))
    var r = Either[String, Int64].Right(Int64(42))
    ok = ok and r.is_right() and (r.get_right() == Int64(42))

    # RangeI64
    var rg = RangeI64(Int64(3), Int64(7))
    ok = ok and rg.contains(Int64(3)) and not rg.contains(Int64(7))
    ok = ok and (rg.len() == Int64(4))
    var rg2 = RangeI64(Int64(5), Int64(10))
    var rc = rg.clamp_to(rg2)  # [5,7)
    ok = ok and (rc.start == Int64(5) and rc.end == Int64(7))

    # NonEmptyList
    var nel = NonEmptyList[Int64](List[Int64]())
    ok = ok and nel.is_empty()
    var xs = List[Int64](); xs.append(Int64(9)); xs.append(Int64(8))
    var nel2 = NonEmptyList[Int64](xs)
    ok = ok and nel2.is_nonempty() and (nel2.head() == Int64(9))

    return ok
