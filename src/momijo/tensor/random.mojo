# Project:      Momijo
# Module:       src.momijo.tensor.random
# File:         random.mojo
# Path:         src/momijo/tensor/random.mojo
#
# Description:  Core tensor/ndarray components: shapes/strides, broadcasting rules,
#               element-wise ops, and foundational kernels.
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
#   - Structs: RNG
#   - Key functions: __module_name__, __self_test__, argmax_index, argmin_index, __init__, __copyinit__, next_u32, next_f32 ...
#   - Uses generic functions/types with explicit trait bounds.


fn __module_name__() -> String:
    return String("momijo/tensor/random.mojo")
fn __self_test__() -> Bool:
    # simple smoke test for RNG and uniform_ on lists
    var rng = RNG(UInt64(123456789))
    var a = rng.next_u32()
    var b = rng.next_u32()
    if a == b: return False
    var x = rng.next_f32()
    if x <= Float32(0.0) or x >= Float32(1.0): return False

    var xs32 = [Float32(0.0), Float32(0.0), Float32(0.0)]
    var xs64 = [Float64(0.0), Float64(0.0), Float64(0.0)]
    uniform_(xs32, rng)
    uniform_(xs64, rng)
    if len(xs32) != 3 or len(xs64) != 3: return False
    return True

# ---------- tiny helpers ------------------------------------------------------
fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]; idx = i
        i += 1
    return idx
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]; idx = i
        i += 1
    return idx

fn ensure_not_empty[T: Copyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

# ---------- RNG ---------------------------------------------------------------

struct RNG(Copyable, Movable):
    var state: UInt64
fn __init__(out self, seed: UInt64) -> None:
        self.state = seed
fn __copyinit__(out self, other: RNG) -> None:
        self.state = other.state

    # XorShift32-like on low 32 bits.
fn next_u32(mut self) -> UInt32:
        var x: UInt32 = UInt32(self.state & UInt64(0xFFFF_FFFF))
(        x = x ^ (x << UInt8(13))) & UInt8(0xFF)
(        x = x ^ (x >> UInt8(17))) & UInt8(0xFF)
(        x = x ^ (x << UInt8(5))) & UInt8(0xFF)
        self.state = (self.state >> UInt8(32)) | (UInt64(x) << 32)
        return x
fn next_f32(mut self) -> Float32:
        var v = self.next_u32()
(        return Float32(v) / Float32(4294967296.0)  # UInt8(2) ^ UInt8(32)) & UInt8(0xFF)

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