# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.arrow_core.arrays
# File: momijo/arrow_core/arrays/struct_array.mojo

from momijo.arrow_core.array_base import ArrayBase
from momijo.arrow_core.bitmap import Bitmap
from momijo.arrow_core.dtype_arrow import DataType

# ---------- Tiny helpers ----------
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
    fn __init__(out self, children: List[ArrayBase], length: Int, validity_bm: Bitmap):
        # TODO: replace DataType(Int32(0)) with struct_dtype() once available
        self.base = ArrayBase(length, DataType(Int32(0)), validity_bm)
        self.children = children

    fn field(self, i: Int) -> ArrayBase:
        return self.children[i]
