# Project:      Momijo
# Module:       src.momijo.arrow_core.arrays.string_array
# File:         string_array.mojo
# Path:         src/momijo/arrow_core/arrays/string_array.mojo
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
#   - Structs: StringArray
#   - Key functions: argmax_index, argmin_index, __module_name__, __self_test__, _string_dtype, __init__, __init__, len ...


from momijo.arrow_core.array_base import ArrayBase
from momijo.arrow_core.bitmap import Bitmap
from momijo.arrow_core.dtype_arrow import DataType
from momijo.arrow_core.offsets import Offsets

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
fn __module_name__() -> String:
    return String("momijo/arrow_core/arrays/string_array.mojo")
fn __self_test__() -> Bool:
    return True

fn _string_dtype() -> DataType:
    return DataType(Int32(3))

# --------- StringArray ---------
struct StringArray:
    var base: ArrayBase
    var offsets: Offsets          # length = N+1
    var data_bytes: List[UInt8]   # contiguous UTF-8 bytes

    # default ctor (for tests)
fn __init__(out self) -> None:
        self.base = ArrayBase(0, _string_dtype(), Bitmap(0, True))
        self.offsets = Offsets()
        self.data_bytes = List[UInt8]()

    # main ctor
fn __init__(out self,
                offsets: Offsets,
                data_bytes: List[UInt8],
                length: Int,
                validity: Bitmap) -> None:
        self.base = ArrayBase(length, _string_dtype(), validity)
        self.offsets = offsets
        self.data_bytes = data_bytes
fn len(self) -> Int:
        return self.base.len()
fn value(self, i: Int) -> String:
        # Guard only; keep implementation minimal to compile everywhere.
        if i < 0 or i >= self.base.length:
            return String("")
        return String("")
fn __copyinit__(out self, other: Self) -> None:
        self.base = other.base
        self.offsets = other.offsets
        self.data_bytes = other.data_bytes
fn __moveinit__(out self, deinit other: Self) -> None:
        self.base = other.base
        self.offsets = other.offsets
        self.data_bytes = other.data_bytes