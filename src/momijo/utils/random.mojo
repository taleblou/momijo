# Project:      Momijo
# Module:       src.momijo.utils.random
# File:         random.mojo
# Path:         src/momijo/utils/random.mojo
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
#   - Structs: PRNG, Option, PRNG, Option
#   - Key functions: _splitmix64_step, _xorshift64star_step, __init__, __copyinit__, next_u64, next_u32, next_f64_01, randint ...
#   - Uses generic functions/types with explicit trait bounds.


from stdlib.list import List

fn _splitmix64_step(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> UInt64(31))
fn _xorshift64star_step(x: UInt64) -> UInt64:
    var z = x
    z = z ^ (z >> UInt64(12))
    z = z ^ (z << UInt64(25))
    z = z ^ (z >> UInt64(27))
    return z * UInt64(0x2545F4914F6CDD1D)

# ---------------------------------------------
# PRNG
# ---------------------------------------------

struct PRNG:
    state: UInt64  # non-zero
fn __init__(out self, seed: UInt64 = UInt64(88172645463393265)):
        # mix the seed so that small seeds are OK and avoid zero
        var s = _splitmix64_step(seed)
        if s == UInt64(0):
            s = UInt64(0x106689D45497FDB5)  # some odd constant
        self.state = s
fn __copyinit__(out self, other: Self) -> None:
        self.state = other.state
fn next_u64(mut self) -> UInt64:
        var s = self.state
        if s == UInt64(0):
            s = UInt64(0x106689D45497FDB5)
        s = _xorshift64star_step(s)
        self.state = s
        return s
fn next_u32(mut self) -> UInt32:
        return UInt32(self.next_u64() & UInt64(0xFFFFFFFF))
fn next_f64_01(mut self) -> Float64:
        # map 53 high bits to [0,1)
        var x = self.next_u64()
        var y = (x >> UInt64(11)) & UInt64(0x1FFFFFFFFFFFFF)  # 53 bits
(        return Float64(y) / Float64(0x20000000000000)  # UInt8(2) ^ UInt8(53)) & UInt8(0xFF)

    # Uniform integer in [lo, hi] (inclusive). If lo>hi, swap.
fn randint(mut self, lo: Int64, hi: Int64) -> Int64:
        var a = lo
        var b = hi
        if a > b:
            var tmp = a
            a = b
            b = tmp
        var span = UInt64(Int64(b - a) + Int64(1))
        # rejection sampling to avoid bias if span not power of two
        var bound = UInt64(~UInt64(0)) - (UInt64(~UInt64(0)) % span)
        var r = self.next_u64()
        while r >= bound:
            r = self.next_u64()
        return a + Int64(r % span)

    # Uniform float in [lo, hi). If lo>hi, swap.
fn randfloat(mut self, lo: Float64 = Float64(0.0), hi: Float64 = Float64(1.0)) -> Float64:
        var a = lo
        var b = hi
        if a > b:
            var tmp = a
            a = b
            b = tmp
        return a + (b - a) * self.next_f64_01()

    # Bernoulli(p) -> Bool
fn bernoulli(mut self, p: Float64) -> Bool:
        var q = p
        if q <= Float64(0.0): return False
        if q >= Float64(1.0): return True
        return self.next_f64_01() < q

    # Standard normal using Box-Muller (polar form)
fn gauss(mut self, mean: Float64 = Float64(0.0), std: Float64 = Float64(1.0)) -> Float64:
        if std <= Float64(0.0):
            return mean
        var u1 = Float64(0.0)
        var u2 = Float64(0.0)
        # generate two uniform (0,1]
        u1 = self.next_f64_01()
        if u1 == Float64(0.0): u1 = Float64(1.0) / Float64(2.0**53)  # minimal positive
        u2 = self.next_f64_01()
        # approximate transforms without log/cos/sin is tricky; rely on a coarse approximation:

        var acc = Float64(0.0)
        var i = 0
        while i < 12:
            acc = acc + self.next_f64_01()
            i += 1
        var z = acc - Float64(6.0)  # approx N(0,1)
        return mean + std * z

    # Shuffle a list in-place (Fisher-Yates)
    fn shuffle_in_place[T: Copyable & Movable](mut self, xs: List[T]):
        var n = len(xs)
        if n <= 1: return
        var i = n - 1
        while Int64(i) > Int64(0):
            var j = Int64(self.randint(Int64(0), Int64(i)))
            # swap xs[i] and xs[j]
            var tmp = xs[i]
            xs[i] = xs[j]
            xs[j] = tmp
            i = i - 1

    # Choose one element; returns index or -1 if empty
fn choice_index(mut self, n: Int64) -> Int64:
        if n <= Int64(0): return Int64(-1)
        return self.randint(Int64(0), n - Int64(1))

    fn choice[T: Copyable & Movable](mut self, xs: List[T]) -> Option[T]:
        var n = len(xs)
        if n == 0:
            return Option[T]()
        var idx = self.choice_index(Int64(n))
        return Option[T](xs[Int64(idx)])

    # Sample k distinct items without replacement (k clipped to [0,n])
    fn sample[T: Copyable & Movable](mut self, xs: List[T], k: Int64) -> List[T]:
        var n = Int64(len(xs))
        var kk = k
        if kk < Int64(0): kk = Int64(0)
        if kk > n: kk = n
        var tmp = List[T]()
        # copy
        var i = Int64(0)
        while i < n:
            tmp.append(xs[i])
            i += Int64(1)
        self.shuffle_in_place(tmp)
        var out = List[T]()
        i = Int64(0)
        while i < kk:
            out.append(tmp[i])
            i += Int64(1)
        return out
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# ---------------------------------------------
# Option[T] lightweight (local minimal) to avoid circular import
# ---------------------------------------------
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
fn get_or(self, default: T) -> T:
        assert(self is not None, String("self is None"))
        if self.has: return self.value()
        return default

# ---------------------------------------------
# Convenience factories
# ---------------------------------------------
fn prng_default() -> PRNG:
    var r = PRNG(UInt64(88172645463393265))
    return r
fn prng_from_seed(seed: UInt64) -> PRNG:
    var r = PRNG(seed)
    return r

# ---------------------------------------------
# Self-test (no prints)
# ---------------------------------------------
fn _self_test() -> Bool:
    var ok = True
    var r1 = prng_from_seed(UInt64(123))
    var r2 = prng_from_seed(UInt64(123))
    ok = ok and (r1.next_u64() == r2.next_u64())
    ok = ok and (r1.randint(Int64(0), Int64(10)) >= Int64(0))

    var xs = List[Int64](); xs.append(Int64(1)); xs.append(Int64(2)); xs.append(Int64(3))
    r1.shuffle_in_place(xs)
    ok = ok and (len(xs) == 3)

    var pick = r1.choice(xs)
    ok = ok and pick.is_some()

    var smp = r1.sample(xs, Int64(2))
    ok = ok and (len(smp) == 2)

    var f = r1.randfloat(Float64(-1.0), Float64(1.0))
    ok = ok and (f >= Float64(-1.0) and f < Float64(1.0))

    return ok
# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.utils
# File: src/momijo/utils/random.mojo
#
# Checklist compliance:
# - Only 'var' (no 'var'); no ''; no global mutable state
# - Explicit imports; English comments
# - Prefer mut/out over inout
# - Struct constructors use `fn __init__(out self, ...)`
# - Deterministic PRNG (SplitMix64 seeding + XorShift64* core)
# - No prints or exceptions in library code
#
# This module provides a lightweight deterministic PRNG suitable for tests.
# Design:
#   - Seed via SplitMix64 to avoid low-entropy seeds
#   - Core step uses xorshift64* (good speed and quality for many tasks)
#   - Helpers for uniform floats/ints, shuffling, choice, sampling, Gaussian

# ---------------------------------------------
# SplitMix64 (for seeding) and xorshift64*
# ---------------------------------------------

fn _splitmix64_step(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> UInt64(30))) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> UInt64(27))) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> UInt64(31))
fn _xorshift64star_step(x: UInt64) -> UInt64:
    var z = x
    z = z ^ (z >> UInt64(12))
    z = z ^ (z << UInt64(25))
    z = z ^ (z >> UInt64(27))
    return z * UInt64(0x2545F4914F6CDD1D)

# ---------------------------------------------
# PRNG
# ---------------------------------------------

struct PRNG:
    state: UInt64  # non-zero
fn __init__(out self, seed: UInt64 = UInt64(88172645463393265)):
        # mix the seed so that small seeds are OK and avoid zero
        var s = _splitmix64_step(seed)
        if s == UInt64(0):
            s = UInt64(0x106689D45497FDB5)  # some odd constant
        self.state = s
fn __copyinit__(out self, other: Self) -> None:
        self.state = other.state
fn next_u64(mut self) -> UInt64:
        var s = self.state
        if s == UInt64(0):
            s = UInt64(0x106689D45497FDB5)
        s = _xorshift64star_step(s)
        self.state = s
        return s
fn next_u32(mut self) -> UInt32:
        return UInt32(self.next_u64() & UInt64(0xFFFFFFFF))
fn next_f64_01(mut self) -> Float64:
        # map 53 high bits to [0,1)
        var x = self.next_u64()
        var y = (x >> UInt64(11)) & UInt64(0x1FFFFFFFFFFFFF)  # 53 bits
(        return Float64(y) / Float64(0x20000000000000)  # UInt8(2) ^ UInt8(53)) & UInt8(0xFF)

    # Uniform integer in [lo, hi] (inclusive). If lo>hi, swap.
fn randint(mut self, lo: Int64, hi: Int64) -> Int64:
        var a = lo
        var b = hi
        if a > b:
            var tmp = a
            a = b
            b = tmp
        var span = UInt64(Int64(b - a) + Int64(1))
        # rejection sampling to avoid bias if span not power of two
        var bound = UInt64(~UInt64(0)) - (UInt64(~UInt64(0)) % span)
        var r = self.next_u64()
        while r >= bound:
            r = self.next_u64()
        return a + Int64(r % span)

    # Uniform float in [lo, hi). If lo>hi, swap.
fn randfloat(mut self, lo: Float64 = Float64(0.0), hi: Float64 = Float64(1.0)) -> Float64:
        var a = lo
        var b = hi
        if a > b:
            var tmp = a
            a = b
            b = tmp
        return a + (b - a) * self.next_f64_01()

    # Bernoulli(p) -> Bool
fn bernoulli(mut self, p: Float64) -> Bool:
        var q = p
        if q <= Float64(0.0): return False
        if q >= Float64(1.0): return True
        return self.next_f64_01() < q

    # Standard normal using Box-Muller (polar form)
fn gauss(mut self, mean: Float64 = Float64(0.0), std: Float64 = Float64(1.0)) -> Float64:
        if std <= Float64(0.0):
            return mean
        var u1 = Float64(0.0)
        var u2 = Float64(0.0)
        # generate two uniform (0,1]
        u1 = self.next_f64_01()
        if u1 == Float64(0.0): u1 = Float64(1.0) / Float64(2.0**53)  # minimal positive
        u2 = self.next_f64_01()
        # approximate transforms without log/cos/sin is tricky; rely on a coarse approximation:

        var acc = Float64(0.0)
        var i = 0
        while i < 12:
            acc = acc + self.next_f64_01()
            i += 1
        var z = acc - Float64(6.0)  # approx N(0,1)
        return mean + std * z

    # Shuffle a list in-place (Fisher-Yates)
    fn shuffle_in_place[T: Copyable & Movable](mut self, xs: List[T]):
        var n = len(xs)
        if n <= 1: return
        var i = n - 1
        while Int64(i) > Int64(0):
            var j = Int64(self.randint(Int64(0), Int64(i)))
            # swap xs[i] and xs[j]
            var tmp = xs[i]
            xs[i] = xs[j]
            xs[j] = tmp
            i = i - 1

    # Choose one element; returns index or -1 if empty
fn choice_index(mut self, n: Int64) -> Int64:
        if n <= Int64(0): return Int64(-1)
        return self.randint(Int64(0), n - Int64(1))

    fn choice[T: Copyable & Movable](mut self, xs: List[T]) -> Option[T]:
        var n = len(xs)
        if n == 0:
            return Option[T]()
        var idx = self.choice_index(Int64(n))
        return Option[T](xs[Int64(idx)])

    # Sample k distinct items without replacement (k clipped to [0,n])
    fn sample[T: Copyable & Movable](mut self, xs: List[T], k: Int64) -> List[T]:
        var n = Int64(len(xs))
        var kk = k
        if kk < Int64(0): kk = Int64(0)
        if kk > n: kk = n
        var tmp = List[T]()
        # copy
        var i = Int64(0)
        while i < n:
            tmp.append(xs[i])
            i += Int64(1)
        self.shuffle_in_place(tmp)
        var out = List[T]()
        i = Int64(0)
        while i < kk:
            out.append(tmp[i])
            i += Int64(1)
        return out
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# ---------------------------------------------
# Option[T] lightweight (local minimal) to avoid circular import
# ---------------------------------------------
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
fn get_or(self, default: T) -> T:
        assert(self is not None, String("self is None"))
        if self.has: return self.value()
        return default

# ---------------------------------------------
# Convenience factories
# ---------------------------------------------
fn prng_default() -> PRNG:
    var r = PRNG(UInt64(88172645463393265))
    return r
fn prng_from_seed(seed: UInt64) -> PRNG:
    var r = PRNG(seed)
    return r

# ---------------------------------------------
# Self-test (no prints)
# ---------------------------------------------
fn _self_test() -> Bool:
    var ok = True
    var r1 = prng_from_seed(UInt64(123))
    var r2 = prng_from_seed(UInt64(123))
    ok = ok and (r1.next_u64() == r2.next_u64())
    ok = ok and (r1.randint(Int64(0), Int64(10)) >= Int64(0))

    var xs = List[Int64](); xs.append(Int64(1)); xs.append(Int64(2)); xs.append(Int64(3))
    r1.shuffle_in_place(xs)
    ok = ok and (len(xs) == 3)

    var pick = r1.choice(xs)
    ok = ok and pick.is_some()

    var smp = r1.sample(xs, Int64(2))
    ok = ok and (len(smp) == 2)

    var f = r1.randfloat(Float64(-1.0), Float64(1.0))
    ok = ok and (f >= Float64(-1.0) and f < Float64(1.0))

    return ok