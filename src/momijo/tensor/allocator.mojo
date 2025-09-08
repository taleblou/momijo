# Project:      Momijo
# Module:       src.momijo.tensor.allocator
# File:         allocator.mojo
# Path:         src/momijo/tensor/allocator.mojo
#
# Description:  Core tensor/ndarray components: shapes/strides, broadcasting rules,
#               element-wise ops, and foundational kernels.
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
#   - Structs: Allocation, CPUAllocator
#   - Key functions: __module_name__, __self_test__, argmax_index, argmin_index, __init__, __init__, __copyinit__, is_null ...
#   - Low-level memory (Pointer/UnsafePointer) used; observe safety invariants.


from memory import Pointer
from momijo.tensor.tensor_base import nbytes
from pathlib import Path
from pathlib.path import Path

fn __module_name__() -> String:
    return String("momijo.tensor.allocator")
fn __self_test__() -> Int:
    var alloc = Allocation()
    if not alloc.is_null():
        return -1
    return 0
fn argmax_index(a: Int, b: Int) -> Int:
    if a >= b:
        return 0
    return 1
fn argmin_index(a: Int, b: Int) -> Int:
    if a <= b:
        return 0
    return 1

# ---------- Allocation ----------
# NOTE: Use a raw address instead of a parametric Pointer[...] to avoid 'mut' inference.
struct Allocation(Copyable, Movable):
    var addr: UInt64
    var nbytes: Int
    var alignment: Int
fn __init__(out self) -> None:
        self.addr = 0
        self.nbytes = 0
        self.alignment = 1
fn __init__(out self, addr: UInt64, nbytes: Int, alignment: Int = 1) -> None:
        self.addr = addr
        self.nbytes = nbytes
        self.alignment = alignment
fn __copyinit__(out self, other: Self) -> None:
        self.addr = other.addr
        self.nbytes = other.nbytes
        self.alignment = other.alignment
fn is_null(self) -> Bool:
        return self.addr == 0

# ---------- CPUAllocator ----------
struct CPUAllocator(Movable):
fn __init__(out self) -> None:
        pass

    # Stub: returns a "null" allocation with metadata only.
fn allocate(self, nbytes: Int, alignment: Int = 64) -> Allocation:
        return Allocation(0, nbytes, alignment)
fn free(self, alloc: Allocation) -> None:
        # No-op in the stub.
        pass