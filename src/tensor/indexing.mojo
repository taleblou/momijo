# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
#
# Project: momijo.tensor
# File: momijo/tensor/indexing.mojo

fn __module_name__() -> String:
    return String("momijo/tensor/indexing.mojo")

fn __self_test__() -> Bool:
    # Lightweight smoke test – extend with real checks later.
    return True

# ---- small helpers (no external deps) ---------------------------------------

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

# ---- indexing over plain lists (temporary shim) -----------------------------
# These signatures deliberately avoid importing Tensor/Views so the module compiles.
# Swap these with real Tensor-based versions once those types are stable in your tree.

# select: pick a single index along "dim" (for lists, dim is ignored; we just return the element repeated as a 1-item list)
fn select[T: Copyable & Movable](xs: List[T], dim: Int, index: Int) -> List[T]:
    if len(xs) == 0:
        return List[T]()
    var idx = index
    if idx < 0:
        idx = 0
    if idx >= len(xs):
        idx = len(xs) - 1
    var out = List[T]()
    out.append(xs[idx])
    return out

# slice: Python-like start:end:step over a 1D list.
fn slice[T: Copyable & Movable](xs: List[T], dim: Int, start: Int, end: Int, step: Int = 1) -> List[T]:
    # dim is ignored for 1D shim.
    var n = len(xs)
    if n == 0 or step == 0:
        return List[T]()
    var s = start
    var e = end
    if s < 0: s = 0
    if e > n: e = n
    if e < s:
        return List[T]()
    var out = List[T]()
    var i = s
    while i < e:
        out.append(xs[i])
        i += step
    return out
