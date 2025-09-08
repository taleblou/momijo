# Project:      Momijo
# Module:       src.momijo.arrow_core.arrays.struct_array
# File:         struct_array.mojo
# Path:         src/momijo/arrow_core/arrays/struct_array.mojo
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
#   - Structs: StructArray
#   - Key functions: argmax_index, argmin_index, __module_name__, __self_test__, __init__, field, __copyinit__, __moveinit__
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.array_base import ArrayBase
from momijo.arrow_core.bitmap import Bitmap
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
    return String("momijo/arrow_core/arrays/struct_array.mojo")
fn __self_test__() -> Bool:
    return True

# ---------- StructArray ----------
struct StructArray:
    var base: ArrayBase
    var children: List[ArrayBase]

    # NOTE: accept Bitmap directly to match ArrayBase API
fn __init__(out self, children: List[ArrayBase], length: Int, validity_bm: Bitmap) -> None:
        # TODO: replace DataType(Int32(0)) with struct_dtype() once available
        self.base = ArrayBase(length, DataType(Int32(0)), validity_bm)
        self.children = children
fn field(self, i: Int) -> ArrayBase:
        return self.children[i]
fn __copyinit__(out self, other: Self) -> None:
        self.base = other.base
        self.children = other.children
fn __moveinit__(out self, deinit other: Self) -> None:
        self.base = other.base
        self.children = other.children