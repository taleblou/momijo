# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.tensor
# File: momijo/tensor/allocator.mojo

# ---------- Module helpers ----------
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

    fn __init__(out self):
        self.addr = 0
        self.nbytes = 0
        self.alignment = 1

    fn __init__(out self, addr: UInt64, nbytes: Int, alignment: Int = 1):
        self.addr = addr
        self.nbytes = nbytes
        self.alignment = alignment

    fn __copyinit__(out self, other: Self):
        self.addr = other.addr
        self.nbytes = other.nbytes
        self.alignment = other.alignment

    fn is_null(self) -> Bool:
        return self.addr == 0

# ---------- CPUAllocator ----------
struct CPUAllocator(Movable):
    fn __init__(out self):
        pass

    # Stub: returns a "null" allocation with metadata only.
    fn allocate(self, nbytes: Int, alignment: Int = 64) -> Allocation:
        return Allocation(0, nbytes, alignment)

    fn free(self, alloc: Allocation):
        # No-op in the stub.
        pass
