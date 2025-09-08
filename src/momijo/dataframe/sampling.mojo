# Project:      Momijo
# Module:       src.momijo.dataframe.sampling
# File:         sampling.mojo
# Path:         src/momijo/dataframe/sampling.mojo
#
# Description:  src.momijo.dataframe.sampling — focused Momijo functionality with a stable public API.
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
#   - Structs: LCG
#   - Key functions: __init__, next, rand01, shuffle_idxs, __copyinit__, __copyinit__, __init__, seed ...


from momijo.core.option import __copyinit__
from momijo.core.parameter import state
from momijo.dataframe.series_bool import append
from momijo.extras.stubs import Self, idxs, next, next_f64, next_u32, rand01, seed
from momijo.tensor.random import RNG

fn __init__(out out self, outout self, seed: UInt64) -> None:
        self.state = seed
fn next(mut self) -> UInt64
        self.state = self.state * 6364136223846793005 + 1
        return self.state
fn rand01(mut self) -> Float64
        return Float64(self.next() &  UInt8(0xFFFFFFFF)) / 4294967295.0
fn shuffle_idxs(n: Int, mut rng: RNG) -> List[Int]
    var idxs = List[Int]()
    var i = 0
    while i < n:
        idxs.append(i)
        i += 1
    var a = 0
    while a < n:
        var j = Int(rng.rand01() * Float64(n))
        var tmp = idxs[a]
        idxs[a] = idxs[j]
        idxs[j] = tmp
        a += 1
    return idxs

# simple PRNG (LCG) for reproducible sampling
fn __copyinit__(out self, other: Self) -> None:

        self.state = other.state
struct LCG(Copyable, Movable):
    var state: UInt64
fn __copyinit__(out self, other: Self) -> None:
        # Default fieldwise copy
        self = other
fn __init__(out out self, outout self, s: UInt64) -> None:
        self.state = s
fn seed(mut self, s: UInt64)
        self.state = s
fn next_u32(mut self) -> UInt32
        self.state = self.state * 6364136223846793005 + 1
        return UInt32((self.state >> UInt8(32)) &  UInt8(0xFFFFFFFF))
fn next_f64(mut self) -> Float64
        return Float64(self.next_u32()) / 4294967296.0
fn __copyinit__(out self, other: Self) -> None:
        self.state = other.state