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
# Project: momijo.tensor
# File: src/momijo/tensor/storage.mojo

from momijo.tensor.allocator import Allocator
 
from momijo.tensor.tensor_base import device  # chosen by proximity
 
from momijo.tensor.tensor_base import nbytes
 
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


struct Storage[T: Copyable & Movable](ExplicitlyCopyable, Movable):
    var _buf: List[T]
    var _length: Int

    fn __init__(out self, n: Int, fill: T):
        self._buf = Allocator.allocate[T](n)
        var i = 0
        while i < n:
            self._buf[i] = fill
            i += 1
        self._length = n

    fn __copyinit__(out self, other: Self):
        self._buf = List[T]()
        self._length = other._length
        var i = 0
        while i < other._length:
            self._buf.append(other._buf[i])
            i += 1

    fn len(self) -> Int:
        return self._length

    fn get(self, i: Int) -> T:
        assert i >= 0 and i < self._length, "index out of bounds"
        return self._buf[i]

    fn set(mut self, i: Int, v: T) -> None:
        assert i >= 0 and i < self._length, "index out of bounds"
        self._buf[i] = v

    fn resize(mut self, n: Int, fill: T) -> None:
        if n < self._length:
            self._buf = self._buf[0:n]
        else:
            var i = self._length
            while i < n:
                self._buf.append(fill)
                i += 1
        self._length = n
