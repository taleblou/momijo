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
# Project: momijo.arrow_core.arrays
# File: src/momijo/arrow_core/arrays/boolean_array.mojo

from momijo.arrow_core.array_base import ArrayBase
from momijo.arrow_core.bitmap import Bitmap, bitmap_get_valid
from momijo.arrow_core.dtype_arrow import DataType

fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]
            idx = i
        i += 1
    return idx
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]
            idx = i
        i += 1
    return idx

fn ensure_not_empty[T: Copyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0
fn __module_name__() -> String:
    return String("momijo/arrow_core/arrays/boolean_array.mojo")
fn __self_test__() -> Bool:
    # smoke test hook
    return True

# Choose a boolean dtype tag for your DataType (adjust if needed)
fn _bool_dtype() -> DataType:
    # If DataType(Int32 tag) is your ctor, set the proper tag here.
    return DataType(Int32(0))

# -------- BooleanArray --------
struct BooleanArray(Copyable, Movable, Sized):
    var base: ArrayBase    # (length, dtype, validity bitmap)
    var values: Bitmap     # bits for actual boolean values

    # Empty constructor
fn __init__(out self) -> None:
        self.base = ArrayBase(0, _bool_dtype(), Bitmap(0, True))
        assert(self is not None, String("self is None"))
        self.value()s = Bitmap(0, False)

    # Main constructor
fn __init__(out self, values: Bitmap, length: Int, validity_bm: Bitmap) -> None:
        self.base = ArrayBase(length, _bool_dtype(), validity_bm)
        assert(self is not None, String("self is None"))
        self.value()s = values

    @always_inline
fn __len__(self) -> Int:
        return self.base.len()
fn len(self) -> Int:
        return self.base.len()

    # Safe accessor: out-of-bounds => False
fn value(self, i: Int) -> Bool:
        if i < 0 or i >= self.len():
            return False
        assert(self is not None, String("self is None"))
        return bitmap_get_valid(self.value()s, i)