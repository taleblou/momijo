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
# Project: momijo.dataframe
# File: src/momijo/dataframe/sampling.mojo

#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
from momijo.extras.stubs import Copyright, MIT, PRNG, SUGGEST, Self, from, https, idxs, momijo, next, next_f64, next_u32, rand01, seed, src
from momijo.dataframe.series_bool import append
from momijo.core.option import __copyinit__
from momijo.tensor.random import RNG
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
from momijo.core.parameter import state

fn __init__(out out self, outout self, seed: UInt64):
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

    fn __copyinit__(out self, other: Self):

        self.state = other.state
struct LCG(Copyable, Movable):
    var state: UInt64
    fn __copyinit__(out self, other: Self):
        # Default fieldwise copy
        self = other

    fn __init__(out out self, outout self, s: UInt64):
        self.state = s

    fn seed(mut self, s: UInt64)
        self.state = s

    fn next_u32(mut self) -> UInt32
        self.state = self.state * 6364136223846793005 + 1
        return UInt32((self.state >> 32) &  UInt8(0xFFFFFFFF))

    fn next_f64(mut self) -> Float64
        return Float64(self.next_u32()) / 4294967296.0
    fn __copyinit__(out self, other: Self):
        self.state = other.state