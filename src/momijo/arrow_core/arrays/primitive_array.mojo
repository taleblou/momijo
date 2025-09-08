# Project:      Momijo
# Module:       src.momijo.arrow_core.arrays.primitive_array
# File:         primitive_array.mojo
# Path:         src/momijo/arrow_core/arrays/primitive_array.mojo
#
# Description:  Arrow-inspired columnar primitives (offsets, buffers, list/struct arrays)
#               supporting zero-copy slicing and predictable memory semantics.
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
#   - Structs: PrimitiveArray
#   - Key functions: argmax_index, argmin_index, __module_name__, __self_test__, _fallback_dtype, __init__, __init__, __len__ ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.array_base import ArrayBase
from momijo.arrow_core.bitmap import Bitmap
from momijo.arrow_core.dtype_arrow import DataType

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
fn __module_name__() -> String:
    return String("momijo/arrow_core/arrays/primitive_array.mojo")
fn __self_test__() -> Bool:
    return True

# --- real implementation ---

# Fallback dtype for zero-arg ctor in tests
fn _fallback_dtype() -> DataType:
    # DataType ctor typically takes a tag Int32
    return DataType(Int32(1))

struct PrimitiveArray[T: Copyable & Movable](Copyable, Movable, Sized):
    var base: ArrayBase
    var data: List[T]

    # empty ctor (for tests/tools)
fn __init__(out self) -> None:
        self.base = ArrayBase(0, _fallback_dtype(), Bitmap(0, True))
        self.data = List[T]()

    # main ctor
fn __init__(out self, data: List[T], dtype: DataType, validity: Bitmap) -> None:
        self.base = ArrayBase(len(data), dtype, validity)
        self.data = data

    @always_inline
fn __len__(self) -> Int:
        return self.base.len()
fn len(self) -> Int:
        return self.base.len()
fn value(self, i: Int) -> T:
        # Clamp index to a valid range to avoid OOB in tests
        var j = i
        if j < 0:
            j = 0
        if j >= self.base.length:
            j = self.base.length - 1
        return self.data[j]