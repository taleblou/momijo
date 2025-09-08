# Project:      Momijo
# Module:       src.momijo.arrow_core.ipc.metadata
# File:         metadata.mojo
# Path:         src/momijo/arrow_core/ipc/metadata.mojo
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
#   - Structs: IPCFieldNode
#   - Key functions: __init__, __copyinit__, __moveinit__, argmax_index, argmin_index, __module_name__, __self_test__
#   - Uses generic functions/types with explicit trait bounds.


struct IPCFieldNode:
    var length: Int
    var null_count: Int
fn __init__(out self, length: Int, null_count: Int) -> None:
        self.length = length
        self.null_count = null_count
fn __copyinit__(out self, other: Self) -> None:
        self.length = other.length
        self.null_count = other.null_count
fn __moveinit__(out self, deinit other: Self) -> None:
        self.length = other.length
        self.null_count = other.null_count
# ---------------------------
# Helpers used by examples
# ---------------------------
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

fn ensure_not_empty[T: Copyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0
fn __module_name__() -> String:
    return String("momijo/arrow_core/ipc/metadata.mojo")
fn __self_test__() -> Bool:
    # minimal sanity checks
    var n = IPCFieldNode(10, 2)
    if n.length != 10 or n.null_count != 2: return False
    var xs: List[Float64] = [0.1, -3.2, 5.0, 2.2]
    if argmax_index(xs) != 2: return False
    if argmin_index(xs) != 1: return False
    return True