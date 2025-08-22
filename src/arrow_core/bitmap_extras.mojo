# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.arrow_core
# File: momijo/arrow_core/bitmap_extras.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

from momijo.arrow_core.bitmap import Bitmap, bitmap_get_valid, bitmap_set_valid

# ---------- Bitwise logical ops ----------

fn bitmap_and(a: Bitmap, b: Bitmap) -> Bitmap:
    let n = a.nbits if a.nbits < b.nbits else b.nbits
    var out = Bitmap(n, True)
    let nbytes = len(out.bytes)
    var i = 0
    while i < nbytes:
        out.bytes[i] = a.bytes[i] & b.bytes[i]
        i += 1
    return out

fn bitmap_or(a: Bitmap, b: Bitmap) -> Bitmap:
    let n = a.nbits if a.nbits < b.nbits else b.nbits
    var out = Bitmap(n, True)
    let nbytes = len(out.bytes)
    var i = 0
    while i < nbytes:
        out.bytes[i] = a.bytes[i] | b.bytes[i]
        i += 1
    return out

fn bitmap_xor(a: Bitmap, b: Bitmap) -> Bitmap:
    let n = a.nbits if a.nbits < b.nbits else b.nbits
    var out = Bitmap(n, True)
    let nbytes = len(out.bytes)
    var i = 0
    while i < nbytes:
        out.bytes[i] = a.bytes[i] ^ b.bytes[i]
        i += 1
    return out

fn bitmap_not(bm: Bitmap) -> Bitmap:
    var out = Bitmap(bm.nbits, True)
    let nbytes = len(bm.bytes)
    var i = 0
    while i < nbytes:
        out.bytes[i] = ~bm.bytes[i]
        i += 1
    # Mask off unused bits in the last byte
    let r = bm.nbits % 8
    if r != 0:
        let mask: UInt8 = (UInt8(1) << UInt8(r)) - UInt8(1)
        let last_idx = len(out.bytes) - 1
        out.bytes[last_idx] = out.bytes[last_idx] & mask
    return out

# ---------- Queries ----------

fn bitmap_count_set_bits(bm: Bitmap) -> Int:
    var c: Int = 0
    for i in range(bm.nbits):
        if bitmap_get_valid(bm, i):
            c += 1
    return c

fn bitmap_any(bm: Bitmap) -> Bool:
    for i in range(bm.nbits):
        if bitmap_get_valid(bm, i):
            return True
    return False

fn bitmap_all(bm: Bitmap) -> Bool:
    for i in range(bm.nbits):
        if not bitmap_get_valid(bm, i):
            return False
    return True

# ---------- Copy / Slice ----------

fn bitmap_copy_slice(bm: Bitmap, start: Int, count: Int) -> Bitmap:
    if start < 0 or count <= 0 or start >= bm.nbits:
        return Bitmap(0, True)
    let end = (start + count) if (start + count) <= bm.nbits else bm.nbits
    var out = Bitmap(end - start, True)
    var i = start
    var j = 0
    while i < end:
        let bit = bitmap_get_valid(bm, i)
        bitmap_set_valid(out, j, bit)
        i += 1
        j += 1
    return out
