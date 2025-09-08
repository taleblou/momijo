# Project:      Momijo
# Module:       src.momijo.arrow_core.schema
# File:         schema.mojo
# Path:         src/momijo/arrow_core/schema.mojo
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
#   - Structs: Schema
#   - Key functions: __module_name__, __self_test__, argmax_index, argmin_index, __init__, __copyinit__, __len__, num_fields ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.field import Field

fn __module_name__() -> String:
    return String("momijo/arrow_core/schema.mojo")
fn __self_test__() -> Bool:
    # Cheap smoke test; extend later
    var xs = List[Float64]()
    xs.append(1.0); xs.append(3.0); xs.append(2.0)
    if argmax_index(xs) != 1: return False
    if argmin_index(xs) != 0: return False
    return True

# ---------------------------
# Tiny local helpers
# ---------------------------
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

fn ensure_not_empty[T: ExplicitlyCopyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

# ---------------------------
# Schema
# ---------------------------
struct Schema(ExplicitlyCopyable, Movable, Sized):
    var fields: List[Field]
fn __init__(out self, fields: List[Field]) -> None:
        self.fields = fields

    # Required for ExplicitlyCopyable
fn __copyinit__(out self, other: Self) -> None:
        self.fields = other.fields

    # Required for Sized
fn __len__(self) -> Int:
        return len(self.fields)

    # Convenience: number of fields
fn num_fields(self) -> Int:
        return len(self.fields)

    # Access field by index (no bounds check here; add if you prefer)
fn field(self, i: Int) -> Field:
        return self.fields[i]