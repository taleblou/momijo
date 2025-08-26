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
# File: momijo/arrow_core/tensor_bridge.mojo

from momijo.arrow_core.array_base import ArrayBase
from momijo.arrow_core.dtype_arrow import DataType

# ---------------- Module meta ----------------
fn __module_name__() -> String:
    return String("momijo/arrow_core/tensor_bridge.mojo")

fn __self_test__() -> Bool:
    var xs = List[Float64]()
    xs.append(1.0); xs.append(3.0); xs.append(2.0)
    if argmax_index(xs) != 1: return False
    if argmin_index(xs) != 0: return False
    return True

# ---------------- Small helpers ----------------
fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]; idx = i
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
            best = xs[i]; idx = i
        i += 1
    return idx

fn ensure_not_empty[T: ExplicitlyCopyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

# ---------------- TensorHandle ----------------
struct TensorHandle(ExplicitlyCopyable, Movable):
    var addr: UInt64     # raw address (0 means "no data")
    var nbytes: Int

    fn __init__(out self, addr: UInt64, nbytes: Int):
        self.addr = addr
        self.nbytes = nbytes

    fn __copyinit__(out self, other: Self):
        self.addr = other.addr
        self.nbytes = other.nbytes

# ---------------- Bridge functions ----------------

fn array_to_tensor(a: ArrayBase) -> TensorHandle:
    # TODO: map ArrayBase buffer to a raw address/size
    return TensorHandle(0, 0)

fn tensor_to_array(t: TensorHandle) raises -> ArrayBase:
    # TODO: zero-copy PrimitiveArray view over tensor memory
    raise String("tensor_to_array: not implemented yet")
