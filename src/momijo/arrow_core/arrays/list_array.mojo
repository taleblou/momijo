# Project:      Momijo
# Module:       src.momijo.arrow_core.arrays.list_array
# File:         list_array.mojo
# Path:         src/momijo/arrow_core/arrays/list_array.mojo
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
#   - Structs: ListArray
#   - Key functions: argmax_index, argmin_index, __module_name__, __self_test__, _list_dtype, __init__, __init__, __len__ ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.array_base import ArrayBase
from momijo.arrow_core.bitmap import Bitmap
from momijo.arrow_core.dtype_arrow import DataType
from momijo.arrow_core.offsets import Offsets

fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
    var best = xs[0]; var idx = 0; var i = 1
    while i < len(xs):
        if xs[i] > best: best = xs[i]; idx = i
        i += 1
    return idx
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
    var best = xs[0]; var idx = 0; var i = 1
    while i < len(xs):
        if xs[i] < best: best = xs[i]; idx = i
        i += 1
    return idx

fn ensure_not_empty[T: Copyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0
fn __module_name__() -> String:
    return String("momijo/arrow_core/arrays/list_array.mojo")
fn __self_test__() -> Bool:
    return True

# Pick a stable tag for "list" dtype (adjust if your DataType uses other tags)
fn _list_dtype() -> DataType:
    # If DataType takes an Int32 tag, choose one consistently (e.g., 2)
    return DataType(Int32(2))

# ---------- ListArray ----------
struct ListArray(Copyable, Movable, Sized):
    var base: ArrayBase
    var offsets: Offsets
    var child: ArrayBase

    # default ctor (empty)
fn __init__(out self) -> None:
        self.base = ArrayBase(0, _list_dtype(), Bitmap(0, True))
        self.offsets = Offsets()
        # placeholder child; if you have a real child dtype, set it later
        self.child = ArrayBase(0, _list_dtype(), Bitmap(0, True))

    # main ctor
fn __init__(out self, offsets: Offsets, child: ArrayBase, length: Int, validity: Bitmap) -> None:
        self.base = ArrayBase(length, _list_dtype(), validity)
        self.offsets = offsets
        self.child = child

    @always_inline
fn __len__(self) -> Int:
        return self.base.len()
fn len(self) -> Int:
        return self.base.len()