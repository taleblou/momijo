# Project:      Momijo
# Module:       src.momijo.vision.memory
# File:         memory.mojo
# Path:         src/momijo/vision/memory.mojo
#
# Description:  src.momijo.vision.memory — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Structs: U8Buffer
#   - Key functions: __init__, is_empty, buf_get, buf_set, buf_fill, buf_copy_into, buf_clone_n, buf_realloc ...
#   - Static methods present.


from gpu import memory
from momijo.core.config import off
from momijo.core.ndarray import offset
from momijo.core.traits import zero
from momijo.core.version import major
from momijo.dataframe.diagnostics import safe
from momijo.dataframe.expr import single
from momijo.dataframe.helpers import m
from momijo.nn.parameter import data
from pathlib import Path
from pathlib.path import Path
from sys import version

struct U8Buffer(Copyable, Movable):
    var data: List[UInt8]
fn __init__(out self, size: Int) -> None:
        var buf: List[UInt8] = List[UInt8]()
        var i = 0
        while i < size:
            buf.append(0)
            i += 1
        self.data = buf
# NOTE: Removed duplicate definition of `len`; use `from momijo.arrow_core.array import len`
fn is_empty(self) -> Bool:
        return len(self.data) == 0

# Read element
@staticmethod
fn buf_get(buf: U8Buffer, idx: Int) -> UInt8:
    return buf.data[idx]

# Set element
@staticmethod
fn buf_set(mut buf: U8Buffer, idx: Int, v: UInt8) -> U8Buffer:
    buf.data[idx] = v
    return buf

# Fill with a constant value
@staticmethod
fn buf_fill(mut buf: U8Buffer, v: UInt8) -> U8Buffer:
    var i = 0
    var n = len(buf.data)
    while i < n:
        buf.data[i] = v
        i += 1
    return buf

# Copy from src into dst (up to min length)
@staticmethod
fn buf_copy_into(mut dst: U8Buffer, src: U8Buffer) -> U8Buffer:
    var n = len(dst.data)
    var m = len(src.data)
    var k = n
    if m < k: k = m
    var i = 0
    while i < k:
        dst.data[i] = src.data[i]
        i += 1
    return dst

# Create a new buffer by copying [0..n) from src (or zero-pad if n > src.len)
@staticmethod
fn buf_clone_n(src: U8Buffer, n: Int) -> U8Buffer:
    var out = U8Buffer(n)
    var m = len(src.data)
    var k = n
    if m < k: k = m
    var i = 0
    while i < k:
        out = buf_set(out, i, src.data[i])
        i += 1
    # if n > m, remaining bytes already zero
    return out

# Reallocate (returns a new buffer preserving contents up to min(old, new))
@staticmethod
fn buf_realloc(buf: U8Buffer, new_size: Int) -> U8Buffer:
    return buf_clone_n(buf, new_size)

# Append a single byte (returns a new buffer). Not optimal, but safe and simple.
@staticmethod
fn buf_append(buf: U8Buffer, v: UInt8) -> U8Buffer:
    var out = U8Buffer(len(buf.data) + 1)
    out = buf_copy_into(out, buf)
    out = buf_set(out, len(buf.data), v)
    return out

# Concatenate two buffers
@staticmethod
fn buf_concat(a: U8Buffer, b: U8Buffer) -> U8Buffer:
    var out = U8Buffer(len(a.data) + len(b.data))
    out = buf_copy_into(out, a)
    var i = 0
    while i < len(b.data):
        out = buf_set(out, len(a.data) + i, b.data[i])
        i += 1
    return out

# Safe clamp add: out[i] = clamp(a[i] + b[i], 0..255)
@staticmethod
fn buf_add_clamp(a: U8Buffer, b: U8Buffer) -> U8Buffer:
    var n = len(a.data)
    var m = len(b.data)
    var k = n
    if m < k: k = m
    var out = U8Buffer(k)
    var i = 0
    while i < k:
        var s = UInt16(a.data[i]) + UInt16(b.data[i])
        if s > UInt16(255):
            out = buf_set(out, i, UInt8(255))
        else:
            out = buf_set(out, i, UInt8(s & UInt16(0xFF)))
        i += 1
    return out

# -------------------------
# Stride helpers (row-major HWC convention)
# -------------------------
@staticmethod
fn packed_hwc_strides(h: Int, w: Int, c: Int) -> (Int, Int, Int):
    # (s0, s1, s2) for (y, x, ch)
    var s2 = 1
    var s1 = c
    var s0 = w * c
    return (s0, s1, s2)

# Flat offset helper for HWC
@staticmethod
fn hwc_offset(w: Int, c: Int, x: Int, y: Int, ch: Int) -> Int:
    return ((y * w) + x) * c + ch

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    var a = U8Buffer(4)
    a = buf_fill(a, 10)
    if buf_get(a, 0) != 10: return False

    var b = buf_append(a, 5)  # now len=5
    if b.len() != 5: return False
    if buf_get(b, 4) != 5: return False

    var c = buf_realloc(b, 3)
    if c.len() != 3: return False
    if buf_get(c, 0) != 10: return False

    var d = buf_add_clamp(a, a)  # 10+10=20
    if d.len() != 4 or buf_get(d, 2) != 20: return False

    var (s0, s1, s2) = packed_hwc_strides(2, 3, 4)
    if not (s0 == 12 and s1 == 4 and s2 == 1): return False

    var off = hwc_offset(3, 4, 1, 0, 2)
    if off != ((0 * 3) + 1) * 4 + 2: return False

    return True