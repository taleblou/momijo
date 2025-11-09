# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.tensor.creation
# File:         src/momijo/tensor/creation.mojo
#
# Description:
#   Generic tensor creation and random utilities.
#   Var-only, no asserts, hot paths unrolled.
#   No global mutable state. If seed is None, a fixed per-call seed is used
#   (deterministic across calls).

from collections.list import List
from momijo.tensor.tensor import Tensor
from momijo.tensor.helpers import (
    compute_row_major_strides,
    astype_with,
    copy_list_int,
    numel,
    # Identity
    id_i8, id_i16, id_i32, id_i64, id_int,
    id_u8, id_u16, id_u32, id_u64,
    id_f16, id_f32, id_f64,
    # Int -> *
    to_i8_from_int, to_i16_from_int, to_i32_from_int, to_i64_from_int,
    to_u8_from_int, to_u16_from_int, to_u32_from_int, to_u64_from_int,
    to_f16_from_int, to_f32_from_int, to_f64_from_int,
    # f64 -> *
    to_i8_from_f64, to_i16_from_f64, to_i32_from_f64, to_i64_from_f64, to_int_from_f64,
    to_u8_from_f64, to_u16_from_f64, to_u32_from_f64, to_u64_from_f64,
    to_f16_from_f64, to_f32_from_f64, to_f64_from_f64,
)
from momijo.tensor.cast import *
from math import *
import random
# ------------------------------ small local helpers ------------------------------ #

@always_inline
fn clamp01(x: Float64) -> Float64:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

@always_inline
fn _arange_len(a: Int, b: Int, st: Int) -> Int:
    var n = 0
    if st > 0:
        if a < b:
            n = (b - a + st - 1) // st
    else:
        if a > b:
            n = (a - b + (-st) - 1) // (-st)
    return n

# ------------------------------ RNGs ------------------------------ #
# SplitMix64 seed expander + xoroshiro128** core.
struct SplitMix64(Copyable, Movable):
    var state: UInt64

    fn __init__(out self, seed: UInt64):
        self.state = seed

    fn next(mut self) -> UInt64:
        self.state = self.state &+ 0x9E3779B97F4A7C15  # keep &+ for add
        var z = self.state
        z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
        return z ^ (z >> 31)

@always_inline
fn rotl64(x: UInt64, k: Int32) -> UInt64:
    var kk = UInt64(Int(k) & 63)
    if kk == 0:
        return x
    return (x << kk) | (x >> (UInt64(64) - kk))

struct RNG64(Copyable, Movable):
    var s0: UInt64
    var s1: UInt64

    fn __init__(out self, seed: UInt64):
        var s = seed
        if s == 0:
            s = 0x9E3779B97F4A7C15
        var sm = SplitMix64(s)
        self.s0 = sm.next()
        self.s1 = sm.next()
        if (self.s0 | self.s1) == 0:
            self.s1 = 0xFFFFFFFFFFFFFFFF

    @always_inline
    fn next_u64(mut self) -> UInt64:
        var s0 = self.s0
        var s1 = self.s1
        var r = rotl64(s0 * UInt64(5), 7) * UInt64(9)
        s1 ^= s0
        self.s0 = rotl64(s0, 24) ^ s1 ^ (s1 << 16)
        self.s1 = rotl64(s1, 37)
        return r

    @always_inline
    fn next_f64(mut self) -> Float64:
        var u = self.next_u64()
        var hi = u >> 11
        return Float64(hi) / 9007199254740992.0  # 2^53
    @always_inline
    fn next_f32(mut self) -> Float32:
        var u = self.next_u64()
        var hi = u >> 11
        return Float32(hi) / 9007199254740992.0  # 2^53

    # Unbiased Int in [low, high)
    @always_inline
    fn next_i32(mut self, low: Int, high: Int) -> Int:
        var lo = low
        var span = high - low
        if span <= 0:
            return lo
        var s = UInt64(span)
        var limit = 0xFFFFFFFFFFFFFFFF - (0xFFFFFFFFFFFFFFFF % s)
        var u = self.next_u64()
        while u >= limit:
            u = self.next_u64()
        var r = Int(u % s)
        return lo + r

@always_inline
fn init_rng(seed: Optional[Int]) -> RNG64:
    if seed is None:
        return RNG64(0xD1B54A32D192ED03)
    return RNG64(UInt64(seed.value()))



# -------------------- generic builders / fillers -------------------- #

# -------------------------------- zeros --------------------------------
@always_inline
fn zero_scalar_of[T: ImplicitlyCopyable & Copyable & Movable](
    from_f64: fn (Float64) -> T
) -> T:
    return from_f64(0.0)

# Core: build a zero tensor with given shape using a Float64->T converter
@always_inline
fn zeros_with_shape[T: ImplicitlyCopyable & Copyable & Movable](
    shape: List[Int],
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    # Defensive copy (per project rules)
    var sh = shape.copy()
    var n = numel(sh)

    # Build zero-filled data buffer
    var data = List[T]()
    data.reserve(n)

    var z = from_f64(0.0)

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        data.append(z); data.append(z); data.append(z); data.append(z)
        data.append(z); data.append(z); data.append(z); data.append(z)
        i = i + 8
    while i < n:
        data.append(z)
        i = i + 1

    # Row-major strides
    var strides = compute_row_major_strides(sh)

    # Use positional args; most Mojo ctors aren't named. Include offset=0.
    return Tensor[T](data, sh, strides, 0)


# Generic wrapper: prefer this instead of calling full(...)
@always_inline
fn zeros[T: ImplicitlyCopyable & Copyable & Movable](
    shape: List[Int],
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    return zeros_with_shape[T](shape, from_f64)

# ------------------------------ dtype-specific ------------------------------

@always_inline
fn zeros(shape: List[Int]) -> Tensor[Float32]:
    return zeros_with_shape[Float32](shape, to_f32_from_f64)

@always_inline
fn zeros_f64(shape: List[Int]) -> Tensor[Float64]:
    return zeros_with_shape[Float64](shape, to_f64_from_f64)

@always_inline
fn zeros_f32(shape: List[Int]) -> Tensor[Float32]:
    return zeros_with_shape[Float32](shape, to_f32_from_f64)

@always_inline
fn zeros_int(shape: List[Int]) -> Tensor[Int]:
    return zeros_with_shape[Int](shape, to_int_from_f64)
@always_inline
fn zeros_u8(shape: List[Int]) -> Tensor[UInt8]:
    return zeros_with_shape[UInt8](shape, to_u8_from_f64)

@always_inline
fn zeros_i8(shape: List[Int]) -> Tensor[Int8]:
    return zeros_with_shape[Int8](shape, to_i8_from_f64)
@always_inline
fn zeros_i32(shape: List[Int]) -> Tensor[Int32]:
    return zeros_with_shape[Int32](shape, to_i32_from_f64)

# ------------------------------ like-helpers ------------------------------

@always_inline
fn zeros_like(x: Tensor[Float64]) -> Tensor[Float64]:
    return zeros_with_shape[Float64](x.shape(), to_f64_from_f64)

@always_inline
fn zeros_like(x: Tensor[Float32]) -> Tensor[Float32]:
    return zeros_with_shape[Float32](x.shape(), to_f32_from_f64)

@always_inline
fn zeros_like(x: Tensor[Int]) -> Tensor[Int]:
    return zeros_with_shape[Int](x.shape(), to_int_from_f64)




# ------------------------------ full ---------------------------------

@always_inline
fn full[T: ImplicitlyCopyable & Copyable & Movable](shape: List[Int], fill: T) -> Tensor[T]:
    # Defensive copy of shape
    var sh = shape.copy()

    # Build data buffer
    var n = numel(sh)
    var data = List[T]()
    data.reserve(n)

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        data.append(fill); data.append(fill); data.append(fill); data.append(fill)
        data.append(fill); data.append(fill); data.append(fill); data.append(fill)
        i = i + 8
    while i < n:
        data.append(fill)
        i = i + 1

    # Row-major strides
    var strides = compute_row_major_strides(sh)

    # Use the 4-arg ctor: (data, shape, strides, offset)
    return Tensor[T](data, sh, strides, 0)

@always_inline
fn full(shape: List[Int], fill: Float64) -> Tensor[Float64]:
    return full[Float64](shape, fill)

@always_inline
fn full(shape: List[Int], fill: Float32) -> Tensor[Float32]:
    return full[Float32](shape, fill)

@always_inline
fn full(shape: List[Int], fill: Int) -> Tensor[Int]:
    return full[Int](shape, fill)

@always_inline
fn full_like[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], fill: T) -> Tensor[T]:
    return full[T](x.shape(), fill)


# ------------------------------ ones ---------------------------------

@always_inline
fn ones_with_shape[T: ImplicitlyCopyable & Copyable & Movable](
    shape: List[Int],
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    # 1 as T
    var one_t = from_f64(1.0)
    return full[T](shape, one_t)

@always_inline
fn ones(shape: List[Int]) -> Tensor[Float64]:
    return ones_with_shape[Float64](shape, to_f64_from_f64)

@always_inline
fn ones_f64(shape: List[Int]) -> Tensor[Float64]:
    return ones_with_shape[Float64](shape, to_f64_from_f64)

@always_inline
fn ones_f32(shape: List[Int]) -> Tensor[Float32]:
    return ones_with_shape[Float32](shape, to_f32_from_f64)

@always_inline
fn ones_int(shape: List[Int]) -> Tensor[Int]:
    return ones_with_shape[Int](shape, to_int_from_f64)

# Like-helpers (dtype-preserving)
@always_inline
fn ones_like(x: Tensor[Float64]) -> Tensor[Float64]:
    return ones_f64(x.shape())

@always_inline
fn ones_like(x: Tensor[Float32]) -> Tensor[Float32]:
    return ones_f32(x.shape())

@always_inline
fn ones_like(x: Tensor[Int]) -> Tensor[Int]:
    return ones_int(x.shape())


# ------------------------------ eye (identity) ------------------------------

# Identity core: rows x cols, build via converter (no T())
fn eye_with_shape[T: ImplicitlyCopyable & Copyable & Movable](
    rows: Int,
    cols: Int,
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    var r = rows
    var c = cols
    if r < 0:
        r = 0
    if c < 0:
        c = 0

    var count = r * c
    var xs = List[T]()
    xs.reserve(count)

    # zero and one via converter
    var zero_t = from_f64(0.0)
    var one_t  = from_f64(1.0)

    # fill zeros (unrolled)
    var i = 0
    var lim = (count // 8) * 8
    while i < lim:
        xs.append(zero_t); xs.append(zero_t); xs.append(zero_t); xs.append(zero_t)
        xs.append(zero_t); xs.append(zero_t); xs.append(zero_t); xs.append(zero_t)
        i = i + 8
    while i < count:
        xs.append(zero_t)
        i = i + 1

    # set diagonal to 1
    var d = r if r < c else c
    var k = 0
    while k < d:
        xs[k * c + k] = one_t
        k = k + 1

    var shp = List[Int]()
    shp.append(r); shp.append(c)
    return Tensor[T](xs, shp)

# Public API: put required arg before the optional one
fn eye[T: ImplicitlyCopyable & Copyable & Movable](
    n: Int,
    from_f64: fn (Float64) -> T,
    m: Optional[Int] = None
) -> Tensor[T]:
    var cols = n
    if not (m is None):
        cols = m.value()
    return eye_with_shape[T](n, cols, from_f64)

# Dtype wrappers
@always_inline
fn eye_f64(n: Int, m: Optional[Int] = None) -> Tensor[Float64]:
    return eye[Float64](n, to_f64_from_f64, m)

@always_inline
fn eye_f32(n: Int, m: Optional[Int] = None) -> Tensor[Float32]:
    return eye[Float32](n, to_f32_from_f64, m)

@always_inline
fn eye_int(n: Int, m: Optional[Int] = None) -> Tensor[Int]:
    return eye[Int](n, to_int_from_f64, m)



# --------------------------- range / arange (all numeric) ---------------------------

# Generic arange with converter from Int -> T
fn arange_with[T: ImplicitlyCopyable & Copyable & Movable](
    start: Int,
    f: fn (Int) -> T,
    stop: Int = -1,
    step: Int = 1
) -> Tensor[T]:
    var a = start
    var b = stop
    var st = step
    if st == 0:
        st = 1
    if b == -1:
        b = a
        a = 0

    var n = 0
    if st > 0:
        if a < b:
            n = (b - a + st - 1) // st
    else:
        if a > b:
            n = (a - b + (-st) - 1) // (-st)

    var xs = List[T]()
    xs.reserve(n)

    var cur = a
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        xs.append(f(cur));        cur = cur + st
        xs.append(f(cur));        cur = cur + st
        xs.append(f(cur));        cur = cur + st
        xs.append(f(cur));        cur = cur + st
        xs.append(f(cur));        cur = cur + st
        xs.append(f(cur));        cur = cur + st
        xs.append(f(cur));        cur = cur + st
        xs.append(f(cur));        cur = cur + st
        i = i + 8

    while i < n:
        xs.append(f(cur))
        cur = cur + st
        i = i + 1

    var shp = List[Int]()
    shp.append(n)

    # Constructor expects (data, shape)
    return Tensor[T](xs, shp)


# Convenience wrappers
@always_inline
fn arange_f32(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[Float32]:
    return arange_with[Float32](start, to_f32_from_int, stop, step)

@always_inline
fn arange_f64(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[Float64]:
    return arange_with[Float64](start, to_f64_from_int, stop, step)

@always_inline
fn arange_int(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[Int]:
    return arange_with[Int](start, id_int, stop, step)
@always_inline
fn arange_i8(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[Int8]:
    return arange_with[Int8](start, to_i8_from_int, stop, step)

@always_inline
fn arange_i16(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[Int16]:
    return arange_with[Int16](start, to_i16_from_int, stop, step)

@always_inline
fn arange_i32(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[Int32]:
    return arange_with[Int32](start, to_i32_from_int, stop, step)

@always_inline
fn arange_i64(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[Int64]:
    return arange_with[Int64](start, to_i64_from_int, stop, step)

@always_inline
fn arange_u8(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[UInt8]:
    return arange_with[UInt8](start, to_u8_from_int, stop, step)

@always_inline
fn arange_u16(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[UInt16]:
    return arange_with[UInt16](start, to_u16_from_int, stop, step)

@always_inline
fn arange_u32(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[UInt32]:
    return arange_with[UInt32](start, to_u32_from_int, stop, step)

@always_inline
fn arange_u64(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[UInt64]:
    return arange_with[UInt64](start, to_u64_from_int, stop, step)

@always_inline
fn arange_f16(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[Float16]:
    return arange_with[Float16](start, to_f16_from_int, stop, step)


# ------------------------------
# Uniform: randu_with and wrappers
# ------------------------------

fn randu_with[T: ImplicitlyCopyable & Copyable & Movable](
    shape: List[Int],
    f: fn (Float64) -> T,
    low: Float64 = 0.0,
    high: Float64 = 1.0,
    seed: Optional[Int] = None
) -> Tensor[T]:
    var sh = copy_list_int(shape)
    var n = numel(sh)
    var xs = List[T]()
    xs.reserve(n)
    var rng = init_rng(seed)
    var scale = high - low

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        xs.append(f(low + scale * rng.next_f64()))
        xs.append(f(low + scale * rng.next_f64()))
        xs.append(f(low + scale * rng.next_f64()))
        xs.append(f(low + scale * rng.next_f64()))
        xs.append(f(low + scale * rng.next_f64()))
        xs.append(f(low + scale * rng.next_f64()))
        xs.append(f(low + scale * rng.next_f64()))
        xs.append(f(low + scale * rng.next_f64()))
        i = i + 8
    while i < n:
        xs.append(f(low + scale * rng.next_f64()))
        i = i + 1

    # Tensor ctor: (data, shape)
    return Tensor[T](xs, sh)

@always_inline
fn randu_f64(shape: List[Int], low: Float64 = 0.0, high: Float64 = 1.0, seed: Optional[Int] = None) -> Tensor[Float64]:
    return randu_with[Float64](shape, to_f64_from_f64, low, high, seed)

@always_inline
fn randu_f32(shape: List[Int], low: Float64 = 0.0, high: Float64 = 1.0, seed: Optional[Int] = None) -> Tensor[Float32]:
    return randu_with[Float32](shape, to_f32_from_f64, low, high, seed)

# Like-helpers
@always_inline
fn rand_like_f64(x: Tensor[Float64], seed: Optional[Int] = None) -> Tensor[Float64]:
    return randu_f64(x.shape(), 0.0, 1.0, seed)

@always_inline
fn rand_like_f32(x: Tensor[Float32], seed: Optional[Int] = None) -> Tensor[Float32]:
    return randu_f32(x.shape(), 0.0, 1.0, seed)

# ------------------------------
# Normal(0,1): Box–Muller (Float64 core)
# ------------------------------

# Tensor-like (uses x.shape())
fn _randn_like_with_f64core[T: ImplicitlyCopyable & Copyable & Movable](
    shape: List[Int],
    map64: fn (Float64) -> T,
    seed: Optional[Int] = None
) -> Tensor[T]:
    var sh = copy_list_int(shape)
    var n = numel(sh)

    var out = List[T]()
    out.reserve(n)

    var rng = init_rng(seed)

    var i = 0
    while i < n:
        var u1: Float64 = rng.next_f64()
        var u2: Float64 = rng.next_f64()
        if u1 < Float64(1e-12):
            u1 = Float64(1e-12)

        var r: Float64 = sqrt(Float64(-2.0) * log(u1))
        var theta: Float64 = Float64(6.283185307179586) * u2  # 2π
        var z0: Float64 = r * cos(theta)
        var z1: Float64 = r * sin(theta)

        out.append(map64(z0))
        if i + 1 < n:
            out.append(map64(z1))
        i = i + 2
    return Tensor[T](out, sh)

# Tensor-like (Float32 core math, useful for pure f32 path)
fn _randn_like_with_f32core[T: ImplicitlyCopyable & Copyable & Movable](
    shape: List[Int],
    map32: fn (Float32) -> T,
    seed: Optional[Int] = None
) -> Tensor[T]:
    var sh = copy_list_int(shape)
    var n = numel(sh)

    var out = List[T]()
    out.reserve(n)

    var rng = init_rng(seed)

    var i = 0
    while i < n:
        var u1: Float32 = rng.next_f32()
        var u2: Float32 = rng.next_f32()
        if u1 < Float32(1e-12):
            u1 = Float32(1e-12)

        var r: Float32 = sqrt(Float32(-2.0) * log(u1))
        var theta: Float32 = Float32(6.283185307179586) * u2  # 2π
        var z0: Float32 = r * cos(theta)
        var z1: Float32 = r * sin(theta)

        out.append(map32(z0))
        if i + 1 < n:
            out.append(map32(z1))
        i = i + 2
    return Tensor[T](out, sh)

# ------------------------------
# Public randn overloads (exact signatures requested)
# ------------------------------

# 1) randn for Int "shape" ⇒ Int tensor (via Float64 core, then cast)
fn randn(shape: List[Int], seed: Optional[Int] = None) -> Tensor[Int]:
    return _randn_like_with_f64core[Int](shape, to_int_from_f64, seed)

# # 2) randn where "shape" is List[Float32] ⇒ Float32 tensor
# fn randn(shape: List[Float32], seed: Optional[Int] = None) -> Tensor[Float32]:
#     var sh = _shape_from_f32(shape)
#     return _randn_like_with_f32core[Float32](sh, to_f32_from_f32, seed)

# # 3) randn where "shape" is List[Float64] ⇒ Float64 tensor
# fn randn(shape: List[Float64], seed: Optional[Int] = None) -> Tensor[Float64]:
#     var sh = _shape_from_f64(shape)
#     return _randn_like_with_f64core[Float64](sh, to_f64_from_f64, seed)

# 4) randn like(tensor) overloads
fn randn(x: Tensor[Float64], seed: Optional[Int] = None) -> Tensor[Float64]:
    return _randn_like_with_f64core[Float64](x.shape(), to_f64_from_f64, seed)

fn randn(x: Tensor[Float32], seed: Optional[Int] = None) -> Tensor[Float32]:
    return _randn_like_with_f32core[Float32](x.shape(), to_f32_from_f32, seed)

fn randn(x: Tensor[Int], seed: Optional[Int] = None) -> Tensor[Int]:
    return _randn_like_with_f64core[Int](x.shape(), to_int_from_f64, seed)

# ------------------------------
# Optional convenience: explicit dtype helpers
# ------------------------------

@always_inline
fn randn_f64(shape: List[Int], seed: Optional[Int] = None) -> Tensor[Float64]:
    return _randn_like_with_f64core[Float64](shape, to_f64_from_f64, seed)

@always_inline
fn randn_f32(shape: List[Int], seed: Optional[Int] = None) -> Tensor[Float32]:
    return _randn_like_with_f32core[Float32](shape, to_f32_from_f32, seed)

@always_inline
fn randn_int(shape: List[Int], seed: Optional[Int] = None) -> Tensor[Int]:
    return _randn_like_with_f64core[Int](shape, to_int_from_f64, seed)

# ============================== NEW: rand (uniform) like randn ==============================

# Like-based (use x.shape())
fn rand_like_with[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    f: fn (Float64) -> T,
    low: Float64 = 0.0,
    high: Float64 = 1.0,
    seed: Optional[Int] = None
) -> Tensor[T]:
    var shape = x.shape()
    var n = numel(shape)
    var out = List[T]()
    out.reserve(n)

    var rng = init_rng(seed)
    var scale = high - low

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(f(low + scale * rng.next_f64()))
        out.append(f(low + scale * rng.next_f64()))
        out.append(f(low + scale * rng.next_f64()))
        out.append(f(low + scale * rng.next_f64()))
        out.append(f(low + scale * rng.next_f64()))
        out.append(f(low + scale * rng.next_f64()))
        out.append(f(low + scale * rng.next_f64()))
        out.append(f(low + scale * rng.next_f64()))
        i = i + 8
    while i < n:
        out.append(f(low + scale * rng.next_f64()))
        i = i + 1

    # Match the constructor form used in randn_like_with
    return Tensor[T](shape, out)

# Shape-based helper
fn rand_with_shape[T: ImplicitlyCopyable & Copyable & Movable](
    shape: List[Int],
    f: fn (Float64) -> T,
    low: Float64 = 0.0,
    high: Float64 = 1.0,
    seed: Optional[Int] = None
) -> Tensor[T]:
    # Reuse the uniform core via randu_with to avoid code duplication
    return randu_with[T](shape, f, low, high, seed)

# ------------------------------ rand overloads (Tensor-based) ------------------------------ #

@always_inline
fn rand(x: Tensor[Float64],
        low: Float64 = 0.0,
        high: Float64 = 1.0,
        seed: Optional[Int] = None) -> Tensor[Float64]:
    return rand_like_with[Float64](x, to_f64_from_f64, low, high, seed)

@always_inline
fn rand(x: Tensor[Float32],
        low: Float64 = 0.0,
        high: Float64 = 1.0,
        seed: Optional[Int] = None) -> Tensor[Float32]:
    return rand_like_with[Float32](x, to_f32_from_f64, low, high, seed)

@always_inline
fn rand(x: Tensor[Int],
        low: Int = 0,
        high: Int = 2,
        seed: Optional[Int] = None) -> Tensor[Int]:
    # Integer range [low, high): mapping via to_int_from_f64
    return rand_like_with[Int](x, to_int_from_f64, Float64(low), Float64(high), seed)

@always_inline
fn rand(
    shape: List[Int],
    low: Float64 = 0.0,
    high: Float64 = 1.0,
    seed: Optional[Int] = None
) -> Tensor[Float64]:
    return rand_with_shape[Float64](shape, to_f64_from_f64, low, high, seed)



# Bernoulli(p); converter Int -> T for 0/1 mapping
fn bernoulli_with[T: ImplicitlyCopyable & Copyable & Movable](
    shape: List[Int],
    p: Float64,
    f: fn (Int) -> T,
    seed: Optional[Int] = None
) -> Tensor[T]:
    var sh = copy_list_int(shape)
    var n = numel(sh)
    var xs = List[T]()
    xs.reserve(n)
    var rng = init_rng(seed)
    var pp = clamp01(p)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        xs.append(if rng.next_f64() < pp { f(1) } else { f(0) })
        xs.append(if rng.next_f64() < pp { f(1) } else { f(0) })
        xs.append(if rng.next_f64() < pp { f(1) } else { f(0) })
        xs.append(if rng.next_f64() < pp { f(1) } else { f(0) })
        xs.append(if rng.next_f64() < pp { f(1) } else { f(0) })
        xs.append(if rng.next_f64() < pp { f(1) } else { f(0) })
        xs.append(if rng.next_f64() < pp { f(1) } else { f(0) })
        xs.append(if rng.next_f64() < pp { f(1) } else { f(0) })
        i += 8
    while i < n:
        xs.append(if rng.next_f64() < pp { f(1) } else { f(0) })
        i += 1
    return Tensor[T](xs, sh)

@always_inline
fn bernoulli_f64(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[Float64]:
    return bernoulli_with[Float64](shape, p, to_f64_from_int, seed)

@always_inline
fn bernoulli_f32(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[Float32]:
    return bernoulli_with[Float32](shape, p, to_f32_from_int, seed)

fn randperm_int(n: Int, seed: Optional[Int] = None) -> Tensor[Int]:
    var nn =n
    if n < 0: nn=0

    # build [0, 1, 2, ..., nn-1]
    var xs = List[Int]()
    xs.reserve(nn)
    var i = 0
    while i < nn:
        xs.append(i)
        i += 1

    # shuffle in-place if length > 1
    if nn > 1:
        var rng = init_rng(seed)
        var j = nn - 1
        while j > 0:
            # next_i32(lo, hi) ⇒ [lo, hi)
            var k = rng.next_i32(0, j + 1)
            var tmp = xs[j]; xs[j] = xs[k]; xs[k] = tmp
            j -= 1

    # unambiguous 4-arg constructor
    var shp = List[Int](); shp.append(nn)
    var strides = compute_row_major_strides(shp)
    return Tensor[Int](xs, shp, strides, 0)

# Same as above, but cast indices to T via converter Int -> T
fn randperm_with[T: ImplicitlyCopyable & Copyable & Movable](
    n: Int,
    f: fn (Int) -> T,
    seed: Optional[Int] = None
) -> Tensor[T]:
    var base = randperm_int(n, seed)
    return astype_with[Int, T](base, f)

@always_inline
fn randperm_f64(n: Int, seed: Optional[Int] = None) -> Tensor[Float64]:
    return randperm_with[Float64](n, to_f64_from_int, seed)

@always_inline
fn randperm_f32(n: Int, seed: Optional[Int] = None) -> Tensor[Float32]:
    return randperm_with[Float32](n, to_f32_from_int, seed)

# ---------------------- Float64 convenience API ---------------------- #


@always_inline
fn arange_f64_alias(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[Float64]:
    return arange_f64(start, stop, step)

@always_inline
fn rand_like_f64_alias(x: Tensor[Float64], seed: Optional[Int] = None) -> Tensor[Float64]:
    return rand_like_f64(x, seed)

@always_inline
fn randn_like_f64_alias(x: Tensor[Float64], seed: Optional[Int] = None) -> Tensor[Float64]:
    return randn(x, seed)

@always_inline
fn bernoulli_f64_alias(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[Float64]:
    return bernoulli_f64(shape, p, seed)

# ---------------------- Float64 convenience API ---------------------- #




@always_inline
fn arange_f32_alias(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[Float32]:
    return arange_f32(start, stop, step)

@always_inline
fn rand_like_f32_alias(x: Tensor[Float32], seed: Optional[Int] = None) -> Tensor[Float32]:
    return rand_like_f32(x, seed)

@always_inline
fn randn_like_f32_alias(x: Tensor[Float64], seed: Optional[Int] = None) -> Tensor[Float32]:
    return randn(x, seed)

@always_inline
fn bernoulli_f32_alias(shape: List[Int], p: Float32, seed: Optional[Int] = None) -> Tensor[Float32]:
    return bernoulli_f32(shape, p, seed)


# ---------------------- Float64 convenience API ---------------------- #






@always_inline
fn arange_int_alias(start: Int, stop: Int = -1, step: Int = 1) -> Tensor[Int]:
    return arange_int(start, stop, step)

@always_inline
fn rand_like_int_alias(x: Tensor[Int], seed: Optional[Int] = None) -> Tensor[Int]:
    return rand_like_int(x, seed)

@always_inline
fn randn_like_int_alias(x: Tensor[Int], seed: Optional[Int] = None) -> Tensor[Int]:
    return randn(x, seed)

@always_inline
fn bernoulli_f64_alias(shape: List[Int], p: Int, seed: Optional[Int] = None) -> Tensor[Int]:
    return bernoulli_int(shape, p, seed)


# ---------------------------- *_like aliases ----------------------------



# ------------------------------ random (all numeric) ------------------------------

@always_inline
fn randu_f16(shape: List[Int], low: Float64 = 0.0, high: Float64 = 1.0, seed: Optional[Int] = None) -> Tensor[Float16]:
    return randu_with[Float16](shape, to_f16_from_f64, low, high, seed)

@always_inline
fn rand_like_f16(x: Tensor[Float16], seed: Optional[Int] = None) -> Tensor[Float16]:
    return randu_f16(x.shape(), 0.0, 1.0, seed)

@always_inline
fn randn_like_f16(x: Tensor[Float16], seed: Optional[Int] = None) -> Tensor[Float16]:
    return randn_like_with[Float16](x, to_f16_from_f64, seed)

# Bernoulli: Int(0/1) -> T
@always_inline
fn bernoulli_f16(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[Float16]:
    return bernoulli_with[Float16](shape, p, to_f16_from_int, seed)

@always_inline
fn bernoulli_f32(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[Float32]:
    return bernoulli_with[Float32](shape, p, to_f32_from_int, seed)

@always_inline
fn bernoulli_f64(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[Float64]:
    return bernoulli_with[Float64](shape, p, to_f64_from_int, seed)

@always_inline
fn bernoulli_i8(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[Int8]:
    return bernoulli_with[Int8](shape, p, to_i8_from_int, seed)

@always_inline
fn bernoulli_i16(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[Int16]:
    return bernoulli_with[Int16](shape, p, to_i16_from_int, seed)

@always_inline
fn bernoulli_i32(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[Int32]:
    return bernoulli_with[Int32](shape, p, to_i32_from_int, seed)

@always_inline
fn bernoulli_i64(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[Int64]:
    return bernoulli_with[Int64](shape, p, to_i64_from_int, seed)

@always_inline
fn bernoulli_int(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[Int]:
    return bernoulli_with[Int](shape, p, id_int, seed)

@always_inline
fn bernoulli_u8(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[UInt8]:
    return bernoulli_with[UInt8](shape, p, to_u8_from_int, seed)

@always_inline
fn bernoulli_u16(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[UInt16]:
    return bernoulli_with[UInt16](shape, p, to_u16_from_int, seed)

@always_inline
fn bernoulli_u32(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[UInt32]:
    return bernoulli_with[UInt32](shape, p, to_u32_from_int, seed)

@always_inline
fn bernoulli_u64(shape: List[Int], p: Float64, seed: Optional[Int] = None) -> Tensor[UInt64]:
    return bernoulli_with[UInt64](shape, p, to_u64_from_int, seed)

# ------------------------------ randperm variants ------------------------------

@always_inline
fn randperm_i8(n: Int, seed: Optional[Int] = None) -> Tensor[Int8]:
    return randperm_with[Int8](n, to_i8_from_int, seed)

@always_inline
fn randperm_i16(n: Int, seed: Optional[Int] = None) -> Tensor[Int16]:
    return randperm_with[Int16](n, to_i16_from_int, seed)

@always_inline
fn randperm_i32(n: Int, seed: Optional[Int] = None) -> Tensor[Int32]:
    return randperm_with[Int32](n, to_i32_from_int, seed)

@always_inline
fn randperm_i64(n: Int, seed: Optional[Int] = None) -> Tensor[Int64]:
    return randperm_with[Int64](n, to_i64_from_int, seed)

@always_inline
fn randperm_u8(n: Int, seed: Optional[Int] = None) -> Tensor[UInt8]:
    return randperm_with[UInt8](n, to_u8_from_int, seed)

@always_inline
fn randperm_u16(n: Int, seed: Optional[Int] = None) -> Tensor[UInt16]:
    return randperm_with[UInt16](n, to_u16_from_int, seed)

@always_inline
fn randperm_u32(n: Int, seed: Optional[Int] = None) -> Tensor[UInt32]:
    return randperm_with[UInt32](n, to_u32_from_int, seed)

@always_inline
fn randperm_u64(n: Int, seed: Optional[Int] = None) -> Tensor[UInt64]:
    return randperm_with[UInt64](n, to_u64_from_int, seed)

@always_inline
fn randperm_f16(n: Int, seed: Optional[Int] = None) -> Tensor[Float16]:
    return randperm_with[Float16](n, to_f16_from_int, seed)

@always_inline
fn randperm_f32(n: Int, seed: Optional[Int] = None) -> Tensor[Float32]:
    return randperm_with[Float32](n, to_f32_from_int, seed)

@always_inline
fn randperm_f64(n: Int, seed: Optional[Int] = None) -> Tensor[Float64]:
    return randperm_with[Float64](n, to_f64_from_int, seed)



# ---------------------------- linspace (generic) ----------------------------

fn linspace_with[T: ImplicitlyCopyable & Copyable & Movable](
    a: Float64,
    b: Float64,
    n: Int,
    f: fn (Float64) -> T,
    endpoint: Bool = True
) -> Tensor[T]:
    var nn = n
    if nn <= 0:
        var xs = List[T]()
        var shp = List[Int](); shp.append(0)
        return Tensor[T](xs, shp)

    if nn == 1:
        var xs = List[T]()
        xs.append(f(a))
        var shp = List[Int](); shp.append(1)
        return Tensor[T](xs, shp)

    var denom = nn - 1
    if not endpoint:
        denom = nn
    var step = (b - a) / Float64(denom)

    var xs = List[T]()
    xs.reserve(nn)

    var i = 0
    var lim = (nn // 8) * 8
    while i < lim:
        xs.append(f(a + step * Float64(i    )))
        xs.append(f(a + step * Float64(i + 1)))
        xs.append(f(a + step * Float64(i + 2)))
        xs.append(f(a + step * Float64(i + 3)))
        xs.append(f(a + step * Float64(i + 4)))
        xs.append(f(a + step * Float64(i + 5)))
        xs.append(f(a + step * Float64(i + 6)))
        xs.append(f(a + step * Float64(i + 7)))
        i += 8
    while i < nn:
        xs.append(f(a + step * Float64(i)))
        i += 1

    if endpoint:
        # Force exact b at the end to minimize rounding drift.
        xs[nn - 1] = f(b)

    var shp = List[Int](); shp.append(nn)
    return Tensor[T](xs, shp)

# ------------------------- typed convenience wrappers -------------------------

@always_inline
fn linspace_f64(a: Float64, b: Float64, n: Int, endpoint: Bool = True) -> Tensor[Float64]:
    return linspace_with[Float64](a, b, n, to_f64_from_f64, endpoint)

@always_inline
fn linspace_f32(a: Float32, b: Float32, n: Int, endpoint: Bool = True) -> Tensor[Float32]:
    return linspace_with[Float32](a, b, n, to_f32_from_f64, endpoint)

@always_inline
fn linspace_Int(a: Int, b: Int, n: Int, endpoint: Bool = True) -> Tensor[Int]:
    return linspace_with[Int](a, b, n, to_f16_from_f64, endpoint)

# ------------------------------ list variant ------------------------------

# For cases where a plain List[Float64] is desired (non-Tensor API).
fn linspace_list_f64(a: Float64, b: Float64, n: Int, endpoint: Bool = True) -> List[Float64]:
    var nn = n
    if nn <= 0:
        return List[Float64]()
    if nn == 1:
        var out1 = List[Float64]()
        out1.append(a)
        return out1.copy()

    var denom = nn - 1
    if not endpoint:
        denom = nn
    var step = (b - a) / Float64(denom)

    var out = List[Float64]()
    out.reserve(nn)

    var i = 0
    var lim = (nn // 8) * 8
    while i < lim:
        out.append(a + step * Float64(i    ))
        out.append(a + step * Float64(i + 1))
        out.append(a + step * Float64(i + 2))
        out.append(a + step * Float64(i + 3))
        out.append(a + step * Float64(i + 4))
        out.append(a + step * Float64(i + 5))
        out.append(a + step * Float64(i + 6))
        out.append(a + step * Float64(i + 7))
        i += 8
    while i < nn:
        out.append(a + step * Float64(i))
        i += 1

    if endpoint:
        out[nn - 1] = b

    return out.copy()

# ------------------------------- empty -------------------------------
# Zero-initialized tensor. 'from_f64' must come BEFORE the optional 'shape'.
@always_inline
fn empty_tensor_with[T: ImplicitlyCopyable & Copyable & Movable](
    from_f64: fn (Float64) -> T,
    shape: Optional[List[Int]] = None
) -> Tensor[T]:
    # Resolve shape (default to [0])
    var shp = List[Int]()
    if shape is None:
        shp.append(0)
    else:
        shp = shape.value().copy()

    # Compute element count
    var n = 1
    var i = 0
    var r = len(shp)
    while i < r:
        n = n * shp[i]
        i += 1

    # Allocate and fill with zeros (unrolled)
    var data = List[T]()
    data.reserve(n)
    var z = from_f64(0.0)

    var k = 0
    var lim = (n // 8) * 8
    while k < lim:
        data.append(z); data.append(z); data.append(z); data.append(z)
        data.append(z); data.append(z); data.append(z); data.append(z)
        k += 8
    while k < n:
        data.append(z)
        k += 1

    # Row-major strides and construct tensor (offset = 0)
    var strides = compute_row_major_strides(shp)
    return Tensor[T](data, shp, strides, 0)

@always_inline
fn empty_tensor_with[T: ImplicitlyCopyable & Copyable & Movable](
    from_f32: fn (Float32) -> T,
    shape: Optional[List[Int]] = None
) -> Tensor[T]:
    # Resolve shape (default to [0])
    var shp = List[Int]()
    if shape is None:
        shp.append(0)
    else:
        shp = shape.value().copy()

    # Compute element count
    var n = 1
    var i = 0
    var r = len(shp)
    while i < r:
        n = n * shp[i]
        i += 1

    # Allocate and fill with zeros (unrolled)
    var data = List[T]()
    data.reserve(n)
    var z = from_f32(0.0)

    var k = 0
    var lim = (n // 8) * 8
    while k < lim:
        data.append(z); data.append(z); data.append(z); data.append(z)
        data.append(z); data.append(z); data.append(z); data.append(z)
        k += 8
    while k < n:
        data.append(z)
        k += 1

    # Row-major strides and construct tensor (offset = 0)
    var strides = compute_row_major_strides(shp)
    return Tensor[T](data, shp, strides, 0)

fn empty_tensor[T: ImplicitlyCopyable & Copyable & Movable]() -> Tensor[T]:
    var data = List[T]()        # no elements
    var shape = List[Int]()     # 1D with zero length
    shape.append(0)
    return Tensor[T](data, shape)
# -------------------------- convenient wrappers --------------------------
@always_inline
fn empty_f64() -> Tensor[Float64]:
    return empty_tensor_with[Float64](to_f64_from_f64, None)
@always_inline
fn empty_f32() -> Tensor[Float32]:
    return empty_tensor_with[Float32](to_f32_from_f32, None)
@always_inline
fn empty(shape: List[Int]) -> Tensor[Float64]:
    return empty_tensor_with[Float64](to_f64_from_f64, shape.copy())

@always_inline
fn empty_f64(shape: List[Int]) -> Tensor[Float64]:
    return empty_tensor_with[Float64](to_f64_from_f64, shape.copy())

@always_inline
fn empty_f32(shape: List[Int]) -> Tensor[Float32]:
    return empty_tensor_with[Float32](to_f32_from_f64, shape.copy())

@always_inline
fn empty_int(shape: List[Int]) -> Tensor[Int]:
    return empty_tensor_with[Int](to_int_from_f64, shape.copy())

@always_inline
fn empty_like(x: Tensor[Float64]) -> Tensor[Float64]:
    return empty_tensor_with[Float64](to_f64_from_f64, x.shape())

@always_inline
fn empty_like(x: Tensor[Float32]) -> Tensor[Float32]:
    return empty_tensor_with[Float32](to_f32_from_f64, x.shape())

@always_inline
fn empty_like(x: Tensor[Int]) -> Tensor[Int]:
    return empty_tensor_with[Int](to_int_from_f64, x.shape())



@always_inline
fn from_list_float64(data: List[Float64]) -> Tensor[Float64]:
    # 1D tensor from a flat list (row-major)
    var shape = List[Int]()
    shape.append(len(data))
    var strides = compute_row_major_strides(shape)
    return Tensor[Float64](data.copy(), shape, strides, 0)


@always_inline
fn from_list_float32(data: List[Float32]) -> Tensor[Float32]:
    # 1D tensor from a flat list (row-major)
    var shape = List[Int]()
    shape.append(len(data))
    var strides = compute_row_major_strides(shape)
    return Tensor[Float32](data.copy(), shape, strides, 0)

@always_inline
fn from_list_int(data: List[Int]) -> Tensor[Int]:
    var n = len(data)
    var shape = List[Int]()
    shape.append(n)
    var strides = compute_row_major_strides(shape)
    return Tensor[Int](data.copy(), shape, strides, 0)


@always_inline
fn from_list_int32(data: List[Int32]) -> Tensor[Int32]:
    var n = len(data)
    var shape = List[Int]()
    shape.append(n)
    var strides = compute_row_major_strides(shape)
    return Tensor[Int32](data.copy(), shape, strides, 0)

@always_inline
fn from_list_int16(data: List[Int16]) -> Tensor[Int16]:
    var n = len(data)
    var shape = List[Int]()
    shape.append(n)
    var strides = compute_row_major_strides(shape)
    return Tensor[Int16](data.copy(), shape, strides, 0)


@always_inline
fn from_list_bool(data: List[Bool]) -> Tensor[Bool]:
    var n = len(data)
    var shape = List[Int]()
    shape.append(n)
    var strides = compute_row_major_strides(shape)
    return Tensor[Bool](data.copy(), shape, strides, 0)


# -----------------------------------------------------------------------------
# Generic 2D & 3D builders from nested lists (row-major, safe, no asserts)
# -----------------------------------------------------------------------------

@always_inline
fn from_2d_list[T: ImplicitlyCopyable & Copyable & Movable](
    rows: List[List[T]]
) -> Tensor[T]:
    # Determine shape [r, c] with safe min-length across rows
    var r = len(rows)
    var c = 0
    if r > 0:
        c = len(rows[0])
        var i = 1
        while i < r:
            var li = len(rows[i])
            if li < c:
                c = li
            i += 1

    # Flatten row-major
    var flat = List[T]()
    var i = 0
    while i < r:
        var j = 0
        while j < c:
            flat.append(rows[i][j])
            j += 1
        i += 1

    # Shape & strides
    var shape = List[Int]()
    shape.append(r)
    shape.append(c)
    var strides = compute_row_major_strides(shape)
    return Tensor[T](flat, shape, strides, 0)


@always_inline
fn from_3d_list[T: ImplicitlyCopyable & Copyable & Movable](
    blocks: List[List[List[T]]]
) -> Tensor[T]:
    # Determine shape [a, b, c] with safe min-lengths
    var a = len(blocks)
    var b = 0
    var c = 0

    if a > 0:
        b = len(blocks[0])
        var i = 1
        while i < a:
            var lb = len(blocks[i])
            if lb < b:
                b = lb
            i += 1

        if b > 0:
            c = len(blocks[0][0])
            var ii = 0
            while ii < a:
                var jj = 0
                while jj < len(blocks[ii]):
                    var lc = len(blocks[ii][jj])
                    if lc < c:
                        c = lc
                    jj += 1
                ii += 1

    # Flatten row-major: iterate a, then b, then c
    var flat = List[T]()
    var i = 0
    while i < a:
        var j = 0
        while j < b:
            var k = 0
            while k < c:
                flat.append(blocks[i][j][k])
                k += 1
            j += 1
        i += 1

    # Shape & strides
    var shape = List[Int]()
    shape.append(a)
    shape.append(b)
    shape.append(c)
    var strides = compute_row_major_strides(shape)
    return Tensor[T](flat, shape, strides, 0)


# -----------------------------------------------------------------------------
# Type-specific thin wrappers (match your existing API naming style)
# -----------------------------------------------------------------------------

# Float64
@always_inline
fn from_2d_list_float64(rows: List[List[Float64]]) -> Tensor[Float64]:
    return from_2d_list[Float64](rows)

@always_inline
fn from_3d_list_float64(blocks: List[List[List[Float64]]]) -> Tensor[Float64]:
    return from_3d_list[Float64](blocks)

# Float32
@always_inline
fn from_2d_list_float32(rows: List[List[Float32]]) -> Tensor[Float32]:
    return from_2d_list[Float32](rows)

@always_inline
fn from_3d_list_float32(blocks: List[List[List[Float32]]]) -> Tensor[Float32]:
    return from_3d_list[Float32](blocks)

# Int
@always_inline
fn from_2d_list_int(rows: List[List[Int]]) -> Tensor[Int]:
    return from_2d_list[Int](rows)

@always_inline
fn from_3d_list_int(blocks: List[List[List[Int]]]) -> Tensor[Int]:
    return from_3d_list[Int](blocks)

# Int32
@always_inline
fn from_2d_list_int32(rows: List[List[Int32]]) -> Tensor[Int32]:
    return from_2d_list[Int32](rows)

@always_inline
fn from_3d_list_int32(blocks: List[List[List[Int32]]]) -> Tensor[Int32]:
    return from_3d_list[Int32](blocks)

# Int16
@always_inline
fn from_2d_list_int16(rows: List[List[Int16]]) -> Tensor[Int16]:
    return from_2d_list[Int16](rows)

@always_inline
fn from_3d_list_int16(blocks: List[List[List[Int16]]]) -> Tensor[Int16]:
    return from_3d_list[Int16](blocks)

# Bool
@always_inline
fn from_2d_list_bool(rows: List[List[Bool]]) -> Tensor[Bool]:
    return from_2d_list[Bool](rows)

@always_inline
fn from_3d_list_bool(blocks: List[List[List[Bool]]]) -> Tensor[Bool]:
    return from_3d_list[Bool](blocks)


@always_inline
fn scalar_zero_tensor[T: ImplicitlyCopyable & Copyable & Movable](
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    # Rank-0 tensor (shape == [], numel == 1)
    var shape = List[Int]()              # []
    var strides = List[Int]()            # row-major for rank-0 is []
    var data = List[T]()
    data.reserve(1)
    data.append(from_f64(0.0))
    return Tensor[T](data, shape, strides, 0)








@always_inline
fn scalar_f64(x: Float64) -> Tensor[Float64]:
    return full[Float64](List[Int](), Float64(x))
@always_inline
fn scalar_f64(x: Float32) -> Tensor[Float64]:
    return full[Float64](List[Int](), Float64(x))
@always_inline
fn scalar_f64(x: Int) -> Tensor[Float64]:
    return full[Float64](List[Int](), Float64(x))


@always_inline
fn scalar_f32(x: Float64) -> Tensor[Float32]:
    return full[Float32](List[Int](), Float32(x))
@always_inline
fn scalar_f32(x: Float32) -> Tensor[Float32]:
    return full[Float32](List[Int](), Float32(x))
@always_inline
fn scalar_f32(x: Int) -> Tensor[Float32]:
    return full[Float32](List[Int](), Float32(x))

@always_inline
fn scalar_int(x: Float64) -> Tensor[Int]:
    return full[Int](List[Int](), Int(x))
@always_inline
fn scalar_int(x: Float32) -> Tensor[Int]:
    return full[Int](List[Int](), Int(x))
@always_inline
fn scalar_int(x: Int) -> Tensor[Int]:
    return full[Int](List[Int](), Int(x))

# -----------------------------------------------------------------------------
# Normal distribution samplers
# -----------------------------------------------------------------------------

@always_inline
fn _randn_shape_f64(shape: List[Int], seed: Optional[Int] = None) -> Tensor[Float64]:
    var like = empty_tensor_with[Float64](to_f64_from_f64, Optional[List[Int]](shape.copy()))
    return randn_like_with[Float64](like, to_f64_from_f64, seed)

# Main API (Float64 default) — matches: tensor.normal(3.0, 0.5, [2])
@always_inline
fn normal(mean: Float64, std: Float64, shape: List[Int], seed: Optional[Int] = None) -> Tensor[Float64]:
    var z = _randn_shape_f64(shape, seed)     # ~ N(0,1)
    z = z.mul_scalar(std)                     # scale by std
    z = z.add_scalar(mean)                    # shift by mean
    return z.copy()

@always_inline
fn normal_f32(mean: Float32, std: Float32, shape: List[Int], seed: Optional[Int] = None) -> Tensor[Float32]:
    var z64 = _randn_shape_f64(shape, seed)        # sample in f64 for stability
    var y64 = z64.mul_scalar(Float64(std)).add_scalar(Float64(mean))
    return y64.to_float32()


# @always_inline
# fn zeros_like(x: tensor.GradTensor) -> tensor.GradTensor:
#     var z = tensor.zeros_like(x.value)
#     return tensor.GradTensor(z, x.ctx)
