# Project:      Momijo
# Module:       src.momijo.arrow_core.bitmap_ops
# File:         bitmap_ops.mojo
# Path:         src/momijo/arrow_core/bitmap_ops.mojo
#
# Description:  Arrow-style bitmap (validity mask) utilities for Momijo enabling
#               bitwise logical ops, popcount, and safe nullable indexing.
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
#   - Final-byte masking (& UInt8(0xFF)) on NOT/XOR/resize paths.
#   - Key functions: _mask_last_byte_inplace, _same_byte_len, bitmap_and, bitmap_or, bitmap_xor, bitmap_not, argmax_index, argmin_index ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.bitmap import Bitmap

fn _mask_last_byte_inplace(mut bm: Bitmap):
    var r = bm.nbits % 8
    if r == 0:
        return
    var mask: UInt8 = (UInt8(1) << UInt8(r)) - UInt8(1)
    var last_idx = len(bm.bytes) - 1
    bm.bytes[last_idx] = bm.bytes[last_idx] & mask

# --- Internal helper: verify equal byte lengths quickly ---
fn _same_byte_len(a: Bitmap, b: Bitmap, c: Bitmap) -> Bool:
    var la = len(a.bytes)
    return la == len(b.bytes) and la == len(c.bytes)

# Bitwise AND: out_dst = lhs & rhs  (element-wise on bytes)
fn bitmap_and(mut out_dst: Bitmap, lhs: Bitmap, rhs: Bitmap):
    if not _same_byte_len(lhs, rhs, out_dst):
        return
    var n = len(lhs.bytes)
    var i = 0
    while i < n:
        out_dst.bytes[i] = lhs.bytes[i] & rhs.bytes[i]
        i += 1
    _mask_last_byte_inplace(out_dst)

# Bitwise OR: out_dst = lhs | rhs
fn bitmap_or(mut out_dst: Bitmap, lhs: Bitmap, rhs: Bitmap):
    if not _same_byte_len(lhs, rhs, out_dst):
        return
    var n = len(lhs.bytes)
    var i = 0
    while i < n:
        out_dst.bytes[i] = lhs.bytes[i] | rhs.bytes[i]
        i += 1
    _mask_last_byte_inplace(out_dst)

# Bitwise XOR: out_dst = lhs ^ rhs
fn bitmap_xor(mut out_dst: Bitmap, lhs: Bitmap, rhs: Bitmap):
    if not _same_byte_len(lhs, rhs, out_dst):
        return
    var n = len(lhs.bytes)
    var i = 0
    while i < n:
        out_dst.bytes[i] = lhs.bytes[i] ^ rhs.bytes[i]
        i += 1
    _mask_last_byte_inplace(out_dst)

# Bitwise NOT: out_dst = ~src   (only bits within nbits are kept)
fn bitmap_not(mut out_dst: Bitmap, src: Bitmap):
    if len(src.bytes) != len(out_dst.bytes):
        return
    var n = len(src.bytes)
    var i = 0
    while i < n:
        # UInt8-safe NOT via XOR with 0xFF
(        out_dst.bytes[i] = UInt8(0xFF) ^ src.bytes[i]) & UInt8(0xFF)
        i += 1
    _mask_last_byte_inplace(out_dst)

# --- Optional tiny utilities (kept for parity with your project style) ---
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
    return String("momijo/arrow_core/bitmap_ops.mojo")
fn __self_test__() -> Bool:
    var a = Bitmap(nbits=10, all_valid=False)
    var b = Bitmap(nbits=10, all_valid=True)
    var out = Bitmap(nbits=10, all_valid=False)
    bitmap_and(out, a, b)
    bitmap_or(out, a, b)
    bitmap_xor(out, a, b)
    bitmap_not(out, a)
    return True