# Project:      Momijo
# Module:       src.momijo.arrow_core.offsets
# File:         offsets.mojo
# Path:         src/momijo/arrow_core/offsets.mojo
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
#   - Structs: Offsets
#   - Key functions: __module_name__, __self_test__, argmax_index, argmin_index, __init__, __copyinit__, __len__, __eq__ ...
#   - Uses generic functions/types with explicit trait bounds.


fn __module_name__() -> String:
    return String("momijo/arrow_core/offsets.mojo")
fn __self_test__() -> Bool:
    # extend with real checks as needed
    return True

# --- Lightweight helpers ---
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

# ---------------- Offsets ----------------
struct Offsets(Copyable, Movable, EqualityComparable, Sized):
    var data: List[Int]
fn __init__(out self) -> None:
        self.data = List[Int]()
        self.data.append(0)
fn __copyinit__(out self, other: Offsets) -> None:
        self.data = List[Int]()
        var i = 0
        var n = len(other.data)
        while i < n:
            self.data.append(other.data[i])
            i += 1
fn __len__(self) -> Int:
        return len(self.data)
fn __eq__(self, other: Offsets) -> Bool:
        var n = len(self.data)
        if n != len(other.data):
            return False
        var i = 0
        while i < n:
            if self.data[i] != other.data[i]:
                return False
            i += 1
        return True
fn __ne__(self, other: Offsets) -> Bool:
        return not self.__eq__(other)
fn append_offset(mut self, o: Int) -> None:
        self.data.append(o)
fn len(self) -> Int:
        return len(self.data)
fn last(self) -> Int:
        # assumes at least one element exists (constructor seeds 0)
        return self.data[len(self.data) - 1]
fn add_length(mut self, length: Int) -> None:
        var next = self.last() + length
        self.data.append(next)
fn is_valid(self) -> Bool:
        var i = 1
        var n = len(self.data)
        while i < n:
            if self.data[i] < self.data[i - 1]:
                return False
            i += 1
        return True

# Build Offsets from a list of lengths
fn from_lengths(lengths: List[Int]) -> Offsets:
    var o = Offsets()
    var i = 0
    var n = len(lengths)
    while i < n:
        o.add_length(lengths[i])
        i += 1
    return o

# --- Module-level convenience to match legacy imports elsewhere ---
fn last(o: Offsets) -> Int:
    return o.last()