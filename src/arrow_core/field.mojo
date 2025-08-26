# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.arrow_core
# File: momijo/arrow_core/field.mojo

from momijo.arrow_core.dtype_arrow import DataType

struct Field(ExplicitlyCopyable, Movable):
    var name: String
    var dtype: DataType
    var nullable: Bool

    fn __init__(out self, name: String, dtype: DataType, nullable: Bool):
        self.name = name
        self.dtype = dtype
        self.nullable = nullable

    fn __copyinit__(out self, other: Self):
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
