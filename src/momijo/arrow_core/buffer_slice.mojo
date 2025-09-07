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
# Project: momijo.arrow_core
# File: src/momijo/arrow_core/buffer_slice.mojo

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
    return String("momijo/arrow_core/buffer_slice.mojo")
fn __self_test__() -> Bool:
    return True

# -----------------------------------------
# BufferSlice: safe view over List[UInt8]
# -----------------------------------------
struct BufferSlice:
    var data: List[UInt8]   # backing storage (owned for now)
    var start: Int          # offset into data
    var nbytes: Int         # view length
fn __init__(out self, data: List[UInt8], start: Int, nbytes: Int) -> None:
        # Defensive normalization (no asserts/exceptions for wide compatibility)
        var s = start
        var n = nbytes
        var max_len = len(data)

        if s < 0:
            s = 0
        if s > max_len:
            s = max_len
        if n < 0:
            n = 0
        if s + n > max_len:
            n = max_len - s

        self.data = data
        self.start = s
        self.nbytes = n
fn len(self) -> Int:
        return self.nbytes
fn at(self, i: Int) -> UInt8:
        # Bounds-safe: returns 0 if out-of-range
        if i < 0:
            return 0
        if i >= self.nbytes:
            return 0
        return self.data[self.start + i]
fn to_list(self) -> List[UInt8]:
        var out_list = List[UInt8]()
        out_list.reserve(self.nbytes)
        var i = 0
        while i < self.nbytes:
            out_list.push_back(self.data[self.start + i])
            i += 1
        return out_list
fn __copyinit__(out self, other: Self) -> None:
        self.data = other.data
        self.start = other.start
        self.nbytes = other.nbytes
fn __moveinit__(out self, deinit other: Self) -> None:
        self.data = other.data
        self.start = other.start
        self.nbytes = other.nbytes
# Free-function factory (decorator-free)
fn buffer_slice_from_all(data: List[UInt8]) -> BufferSlice:
    return BufferSlice(data, 0, len(data))