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
# File: momijo/tensor/storage.mojo

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
    return String("momijo/tensor/storage.mojo")

fn __self_test__() -> Bool:
    # This is a cheap smoke-test hook; extend with real checks as needed.
    return True

# ---- Minimal Storage header (to satisfy tensor_base imports) ----
from momijo.tensor.device import Device

struct Storage:
    var device: Device
    var nbytes: Int

    fn __init__(out self, nbytes: Int, device: Device):
        self.device = device
        self.nbytes = nbytes

    # NOTE: Minimal stub; returns a nil pointer.
    # This is enough for header-level ops (pointer arithmetic ok if not dereferenced).
    # If you need actual readable/writable backing memory, we can extend this.
    fn data_ptr(self) -> Pointer[UInt8]:
        return Pointer[UInt8].nil()

    fn size_in_bytes(self) -> Int:
        return self.nbytes

# ---- Simple typed buffers (kept from your draft; useful for tests/examples) ----

struct BufferF32:
    var data: List[Float32]
    fn __init__(out self, n: Int):
        # Safe manual fill in case list-repetition isn't supported on your toolchain
        self.data = List[Float32]()
        var i = 0
        while i < n:
            self.data.append(0.0)
            i += 1
    fn ptr_mut(self) -> List[Float32]:
        return self.data
    fn ptr(self) -> List[Float32]:
        return self.data

struct BufferF64:
    var data: List[Float64]
    fn __init__(out self, n: Int):
        self.data = List[Float64]()
        var i = 0
        while i < n:
            self.data.append(0.0)
            i += 1
    fn ptr_mut(self) -> List[Float64]:
        return self.data
    fn ptr(self) -> List[Float64]:
        return self.data

struct BufferI32:
    var data: List[Int32]
    fn __init__(out self, n: Int):
        self.data = List[Int32]()
        var i = 0
        while i < n:
            self.data.append(0)
            i += 1
    fn ptr_mut(self) -> List[Int32]:
        return self.data
    fn ptr(self) -> List[Int32]:
        return self.data

struct BufferI64:
    var data: List[Int64]
    fn __init__(out self, n: Int):
        self.data = List[Int64]()
        var i = 0
        while i < n:
            self.data.append(0)
            i += 1
    fn ptr_mut(self) -> List[Int64]:
        return self.data
    fn ptr(self) -> List[Int64]:
        return self.data
