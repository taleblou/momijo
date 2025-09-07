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
# Project: momijo.arrow_core
# File: src/momijo/arrow_core/buffer.mojo

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
    return String("momijo/arrow_core/buffer.mojo")
fn __self_test__() -> Bool:
    return True

# ---------- Buffer (allocator-free, pointer-free) ----------
struct Buffer:
    var data: List[UInt8]
    var nbytes: Int

    # ---- Constructors ---------------------------------------------------------
fn __init__(out self, nbytes: Int) -> None:
        self.nbytes = nbytes
        self.data = List[UInt8]()
        var i = 0
        while i < nbytes:
            self.data.append(0)
            i += 1

    @staticmethod
fn zeros(nbytes: Int) -> Buffer:
        var b = Buffer(nbytes)
        return b

    @staticmethod
fn from_bytes(src: List[UInt8]) -> Buffer:
        var b = Buffer(len(src))
        var i = 0
        var L = len(src)
        while i < L:
            b.data[i] = src[i]
            i += 1
        return b

    # ---- Basic queries --------------------------------------------------------
fn len(self) -> Int:
        return self.nbytes
fn is_empty(self) -> Bool:
        return self.nbytes == 0

    # ---- Element access -------------------------------------------------------
fn get(self, index: Int) -> UInt8:
        if index < 0 or index >= self.nbytes:
            return 0  # out-of-range -> safe default
        return self.data[index]
fn set(mut self, index: Int, value: UInt8):
        if index < 0 or index >= self.nbytes:
            return  # silently ignore OOB writes (no exceptions)
        self.data[index] = value

    # ---- Bulk operations ------------------------------------------------------
fn clear(mut self) -> None:
        var i = 0
        while i < self.nbytes:
            self.data[i] = 0
            i += 1
fn fill(mut self, value: UInt8) -> None:
        var i = 0
        while i < self.nbytes:
            self.data[i] = value
            i += 1
fn slice(self, start: Int, length: Int) -> List[UInt8]:
        # Returns a copy of [start, start+length); OOB -> empty list.
        if start < 0 or length < 0:
            return List[UInt8]()
        if start + length > self.nbytes:
            return List[UInt8]()
        var out = List[UInt8]()
        var j = 0
        while j < length:
            out.append(self.data[start + j])
            j += 1
        return out
fn write(mut self, start: Int, src: List[UInt8]) -> Int:
        # Writes src into self at start; returns bytes written (0 if invalid start).
        if start < 0 or start >= self.nbytes:
            return 0
        var max_can_write = self.nbytes - start
        var to_write = len(src)
        if to_write > max_can_write:
            to_write = max_can_write
        var i = 0
        while i < to_write:
            self.data[start + i] = src[i]
            i += 1
        return to_write
fn copy_from(mut self,
                 src: Buffer,
                 src_offset: Int,
                 dst_offset: Int,
                 count: Int) -> Int:
        # Copies count bytes; invalid ranges -> 0.
        if count <= 0:
            return 0
        if src_offset < 0 or dst_offset < 0:
            return 0
        if src_offset + count > src.nbytes:
            return 0
        if dst_offset + count > self.nbytes:
            return 0
        var i = 0
        while i < count:
            self.data[dst_offset + i] = src.data[src_offset + i]
            i += 1
        return count

    # ---- Convenience utilities ------------------------------------------------
fn equals(self, other: Buffer) -> Bool:
        if self.nbytes != other.nbytes:
            return False
        var i = 0
        while i < self.nbytes:
            if self.data[i] != other.data[i]:
                return False
            i += 1
        return True
fn to_hex(self, max_bytes: Int = -1) -> String:
        var buf = String("")
        var n = self.nbytes
        if max_bytes >= 0 and max_bytes < n:
            n = max_bytes
        var i = 0
        while i < n:
            var v = self.data[i]
            var hi = (v >> UInt8(4)) & 0xF
            var lo = v & UInt8(0xF)
            var c_hi = UInt8(48 + hi)
            if hi >= 10:
                c_hi = UInt8(97 + (hi - 10))
            var c_lo = UInt8(48 + lo)
            if lo >= 10:
                c_lo = UInt8(97 + (lo - 10))
            buf = buf + String(Char(c_hi)) + String(Char(c_lo))
            i += 1
        if max_bytes >= 0 and self.nbytes > max_bytes:
            buf = buf + String("…")
        return buf
fn debug_string(self) -> String:
        return String("Buffer(len=") + String(self.nbytes) + String(", hex=") + self.to_hex(32) + String(")")
fn __copyinit__(out self, other: Self) -> None:
        self.data = other.data
        self.nbytes = other.nbytes
fn __moveinit__(out self, deinit other: Self) -> None:
        self.data = other.data
        self.nbytes = other.nbytes