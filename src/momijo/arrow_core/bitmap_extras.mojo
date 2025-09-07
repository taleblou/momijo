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
# File: src/momijo/arrow_core/bitmap_extras.mojo

from momijo.arrow_core.bitmap import bitmap_set_valid
from momijo.arrow_core.offsets import last
from momijo.core.config import off
from momijo.core.shape import Slice
from momijo.core.traits import zero
from momijo.ir.dialects.annotations import array
from momijo.tensor.dtype import nbits
from momijo.tensor.tensor_base import nbytes
from momijo.utils.timer import start
from momijo.vision.schedule.schedule import bitmap_get_valid, bitmap_set_valid
from pathlib import Path
from pathlib.path import Path
from sys import implementation

# NOTE:
# This file has been refactored to import from Modular stdlib 'bit' modules when available.
# Primary imports are attempted from:
#   - mojo.stdlib.bit.bit_ops
#   - mojo.stdlib.bit.bitset
#   - mojo.stdlib.bit.bitmap
#   - mojo.stdlib.bit.bitmask
# If your environment uses a different stdlib path, adjust the imports accordingly.

# --- Compatibility layer ---
# If specific types/functions are expected by Momijo, you can alias them here.
# Example:
# alias BitOps = bit_ops
# alias BitSet = Bitset
# alias Bitmap = Bitmap
#
# Keep original implementation below as fallback (commented out).
# To re-enable, remove the triple quotes.
"""
fn bitmap_and(a: Bitmap, b: Bitmap) -> Bitmap:
    var n = a.nbits if a.nbits < b.nbits else b.nbits
    var out = Bitmap(n, True)
    var nbytes = len(out.bytes)
    var i = 0
    while i < nbytes:
        out.bytes[i] = a.bytes[i] & b.bytes[i]
        i += 1
    return out
fn bitmap_or(a: Bitmap, b: Bitmap) -> Bitmap:
    var n = a.nbits if a.nbits < b.nbits else b.nbits
    var out = Bitmap(n, True)
    var nbytes = len(out.bytes)
    var i = 0
    while i < nbytes:
        out.bytes[i] = a.bytes[i] | b.bytes[i]
        i += 1
    return out
fn bitmap_xor(a: Bitmap, b: Bitmap) -> Bitmap:
    var n = a.nbits if a.nbits < b.nbits else b.nbits
    var out = Bitmap(n, True)
    var nbytes = len(out.bytes)
    var i = 0
    while i < nbytes:
        out.bytes[i] = a.bytes[i] ^ b.bytes[i]
        i += 1
    return out
fn bitmap_not(bm: Bitmap) -> Bitmap:
    var out = Bitmap(bm.nbits, True)
    var nbytes = len(bm.bytes)
    var i = 0
    while i < nbytes:
        out.bytes[i] = ~bm.bytes[i]
        i += 1
    # Mask off unused bits in the last byte
    var r = bm.nbits % 8
    if r != 0:
        var mask: UInt8 = (UInt8(1) << UInt8(r)) - UInt8(1)
        var last_idx = len(out.bytes) - 1
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
    var end = (start + count) if (start + count) <= bm.nbits else bm.nbits
    var out = Bitmap(end - start, True)
    var i = start
    var j = 0
    while i < end:
        var bit = bitmap_get_valid(bm, i)
        bitmap_set_valid(out, j, bit)
        i += 1
        j += 1
    return out
"""