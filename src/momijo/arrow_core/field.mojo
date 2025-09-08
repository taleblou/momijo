# Project:      Momijo
# Module:       src.momijo.arrow_core.field
# File:         field.mojo
# Path:         src/momijo/arrow_core/field.mojo
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
#   - Structs: Field
#   - Key functions: __init__, __copyinit__, argmax_index, argmin_index, __module_name__, __self_test__
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.dtype_arrow import DataType

struct Field(ExplicitlyCopyable, Movable):
    var name: String
    var dtype: DataType
    var nullable: Bool
fn __init__(out self, name: String, dtype: DataType, nullable: Bool) -> None:
        self.name = name
        self.dtype = dtype
        self.nullable = nullable
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.dtype = other.dtype
        self.nullable = other.nullable

# --- tiny local utils kept if you need them; ok to remove if duplicated elsewhere ---
fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0: return -1
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
    if len(xs) == 0: return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]
            idx = i
        i += 1
    return idx

fn ensure_not_empty[T: ExplicitlyCopyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0
fn __module_name__() -> String:
    return String("momijo/arrow_core/field.mojo")
fn __self_test__() -> Bool:
    # Cheap smoke test hook
    var f = Field(String("a"), DataType.int32(), True)
    return f.name == String("a")