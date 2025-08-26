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
# Project: momijo.arrow_core.ipc
# File: momijo/arrow_core/ipc/metadata.mojo


struct IPCFieldNode:
    var length: Int
    var null_count: Int

    fn __init__(out self, length: Int, null_count: Int):
        self.length = length
        self.null_count = null_count

    fn __copyinit__(out self, other: Self):
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