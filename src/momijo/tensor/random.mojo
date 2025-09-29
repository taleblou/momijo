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
# Project: momijo.tensor
# File: src/momijo/tensor/random.mojo

from momijo.tensor.tensor_base import Tensor
from momijo.tensor.dtype import DType, float64 as dtype_f64, int32 as dtype_i32 
 
from momijo.core.parameter import state
 
# ---------- RNG ---------------------------------------------------------------

struct RNG(Copyable, Movable):
    var state: UInt64

    fn __init__(out self, seed: UInt64):
        self.state = seed

    fn __copyinit__(out self, other: RNG):
        self.state = other.state

    # XorShift32-like on low 32 bits.
    fn next_u32(mut self) -> UInt32:
        var x: UInt32 = UInt32(self.state & UInt64(0xFFFF_FFFF))
        x = x ^ (x << 13)
        x = x ^ (x >> 17)
        x = x ^ (x << 5)
        self.state = (self.state >> 32) | (UInt64(x) << 32)
        return x

    fn next_f32(mut self) -> Float32:
        var v = self.next_u32()
        return Float32(v) / Float32(4294967296.0)  # 2^32

# convenience wrappers (some auto examples import these)
fn RNG_next_u32(mut x: RNG) -> UInt32:
    return x.next_u32()

fn RNG_next_f32(mut x: RNG) -> Float32:
    return x.next_f32()

# ---------- list fillers (no Tensor dependency) --------------------------------

fn uniform_list_f32(mut xs: List[Float32], mut rng: RNG) -> None:
    var i = 0
    while i < len(xs):
        xs[i] = rng.next_f32()
        i += 1

fn uniform_list_f64(mut xs: List[Float64], mut rng: RNG) -> None:
    var i = 0
    while i < len(xs):
        # compose two f32 draws for a basic f64
        var a = Float64(rng.next_f32())
        var b = Float64(rng.next_f32())
        xs[i] = a + b * Float64(1e-6)
        i += 1

# Expose the name expected by your tests: overloaded uniform_ for lists.
fn uniform_(mut xs: List[Float32], mut rng: RNG) -> None:
    uniform_list_f32(xs, rng)

fn uniform_(mut xs: List[Float64], mut rng: RNG) -> None:
    uniform_list_f64(xs, rng)

    fn __copyinit__(out self, other: Self):

        self.state = other.state

        self.x = other.x



struct RandomState(ExplicitlyCopyable, Movable):
    var _state: UInt64

    fn __init__(out self, seed: UInt64):
        if seed == 0:
            seed = 0x9E3779B97F4A7C15  # golden ratio seed
        self._state = seed

    fn __copyinit__(out self, other: Self):
        self._state = other._state

    fn copy(self) -> Self:
        var out = RandomState(self._state)
        return out

    fn _next_u64(mut self) -> UInt64:
        # LCG parameters
        var a: UInt64 = 6364136223846793005
        var c: UInt64 = 1442695040888963407
        self._state = a &* self._state &+ c
        return self._state

    fn next_f64(mut self) -> Float64:
        # Map to [0,1)
        var x = self._next_u64() >> 11  # keep top 53 bits
        return Float64(x) / Float64(1 << 53)

    fn next_i32(mut self, low: Int, high: Int) -> Int:
        var r = self.next_f64()
        var span = high - low
        if span <= 0: span = 1
        return low + Int(Float64(span) * r)

# Global convenience RNG
var _global_rng = RandomState(0xD1F3E9B97C4A1A2B)

fn seed(s: UInt64) -> None:
    _global_rng = RandomState(s)

fn rand(shape: List[Int]) -> Tensor[Float64]:
    var out = Tensor[Float64](shape, 0.0, dtype_f64())
    var n = out.size()
    var i = 0
    while i < n:
        out.set_flat(i, _global_rng.next_f64())
        i += 1
    return out

fn rand_uniform(low: Float64, high: Float64, shape: List[Int]) -> Tensor[Float64]:
    var out = Tensor[Float64](shape, 0.0, dtype_f64())
    var n = out.size()
    var i = 0
    var span = high - low
    while i < n:
        out.set_flat(i, low + span * _global_rng.next_f64())
        i += 1
    return out

fn randn(shape: List[Int]) -> Tensor[Float64]:
    # Box-Muller transform
    var out = Tensor[Float64](shape, 0.0, dtype_f64())
    var n = out.size()
    var i = 0
    while i < n:
        var u1 = _global_rng.next_f64()
        var u2 = _global_rng.next_f64()
        if u1 <= 1e-12: u1 = 1e-12
        var r = builtin.sqrt(-2.0 * builtin.log(u1))
        var theta = 2.0 * 3.141592653589793 * u2
        var z0 = r * builtin.cos(theta)
        out.set_flat(i, z0)
        i += 1
        if i < n:
            var z1 = r * builtin.sin(theta)
            out.set_flat(i, z1)
            i += 1
    return out

fn randint(low: Int, high: Int, shape: List[Int]) -> Tensor[Int]:
    var out = Tensor[Int](shape, 0, dtype_i32())
    var n = out.size()
    var i = 0
    while i < n:
        out.set_flat(i, _global_rng.next_i32(low, high))
        i += 1
    return out
