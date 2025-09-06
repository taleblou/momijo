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
# File: src/momijo/dataframe/generic_ops.mojo

#   from momijo.core.shape import append
#   from momijo.core.traits import append
#   from momijo.dataframe.series_bool import append
#   from momijo.dataframe.series_f64 import append
#   from momijo.dataframe.series_i64 import append
#   from momijo.dataframe.series_str import append
# SUGGEST (alpha): from momijo.core.shape import append
#   from momijo.arrow_core.array import len
#   from momijo.arrow_core.array_base import len
#   from momijo.arrow_core.arrays.boolean_array import len
#   from momijo.arrow_core.arrays.list_array import len
#   from momijo.arrow_core.arrays.primitive_array import len
#   from momijo.arrow_core.arrays.string_array import len
#   from momijo.arrow_core.bitmap import len
#   from momijo.arrow_core.buffer import len
#   from momijo.arrow_core.buffer_slice import len
#   from momijo.arrow_core.byte_string_array import len
#   from momijo.arrow_core.column import len
#   from momijo.arrow_core.offsets import len
#   from momijo.arrow_core.poly_column import len
#   from momijo.arrow_core.string_array import len
#   from momijo.core.types import len
#   from momijo.dataframe.column import len
#   from momijo.dataframe.index import len
#   from momijo.dataframe.series_bool import len
#   from momijo.dataframe.series_f64 import len
#   from momijo.dataframe.series_i64 import len
#   from momijo.dataframe.series_str import len
# SUGGEST (alpha): from momijo.arrow_core.array import len
from momijo.extras.stubs import Copyright, MIT, SUGGEST, eq, found, from, https, if, in_b, len, momijo, pass, return, seen, src
from momijo.dataframe.series_bool import append
from algorithm.reduction import map_reduce
from algorithm.reduction import any_true
from algorithm.reduction import all_true
from algorithm.reduction import cumsum
from algorithm.reduction import reduce
from algorithm.reduction import reduce_boolean
from algorithm.functional import vectorize
trait Equality[T]:
    fn eq(a: T, b: T) -> Bool

# Instances
struct EqString: pass
fn (self: EqString) eq(a: String, b: String) -> Bool return a == b

struct EqI64: pass
fn (self: EqI64) eq(a: Int64, b: Int64) -> Bool: return a == b

struct EqF64: pass
fn (self: EqF64) eq(a: Float64, b: Float64) -> Bool: return a == b

struct EqBool: pass
fn (self: EqBool) eq(a: Bool, b: Bool) -> Bool: return a == b

# unique
fn unique_g[T, E: Equality[T]](xs: List[T], eqer: E) -> List[T]:
    var out = List[T]()
    var i = 0
    while i < len(xs):
        var seen = False; var j = 0
        while j < len(out):
            if eqer.eq(out[j], xs[i]): seen = True
            j += 1
        if not seen: out.append(xs[i])
        i += 1
    return out

# isin
fn isin_g[T, E: Equality[T]](xs: List[T], univ: List[T], eqer: E) -> List[Bool]:
    var out = List[Bool]()
    var i = 0
    while i < len(xs):
        var found = False; var j = 0
        while j < len(univ):
            if eqer.eq(xs[i], univ[j]): found = True
            j += 1
        out.append(found); i += 1
    return out

# union
fn union_g[T, E: Equality[T]](a: List[T], b: List[T], eqer: E) -> List[T]:
    var out = unique_g(a, eqer); var i = 0
    while i < len(b):
        var seen = False; var j = 0
        while j < len(out):
            if eqer.eq(out[j], b[i]): seen = True
            j += 1
        if not seen: out.append(b[i])
        i += 1
    return out

# diff
fn diff_g[T, E: Equality[T]](a: List[T], b: List[T], eqer: E) -> List[T]:
    var out = List[T](); var i = 0
    while i < len(a):
        var in_b = False; var j = 0
        while j < len(b):
            if eqer.eq(a[i], b[j]): in_b = True
            j += 1
        if not in_b: out.append(a[i])
        i += 1
    return out

# symdiff
fn symdiff_g[T, E: Equality[T]](a: List[T], b: List[T], eqer: E) -> List[T]:
    var ab = diff_g(a, b, eqer); var ba = diff_g(b, a, eqer)
    return union_g(ab, ba, eqer)
