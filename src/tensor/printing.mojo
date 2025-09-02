# MIT License
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
#
# Project: momijo.tensor
# File: momijo/tensor/printing.mojo

# ---- tiny utils kept consistent with other modules ---------------------------

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
    return String("momijo/tensor/printing.mojo")

fn __self_test__() -> Bool:
    return True

# ---- summarize (stub) --------------------------------------------------------
# Dependency-free placeholder so tests can import this symbol without pulling in
# your Tensor implementation. Replace with a real Tensor-aware version later.

fn summarize[T: Copyable & Movable](_t: T) -> String:
    # Minimal, always-valid string. Swap this with a real formatter once your
    # Tensor API is finalized (shape/dtype/device/contiguity accessors).
    return String("Tensor(shape=?, dtype=?, device=?, contiguous=?)")
