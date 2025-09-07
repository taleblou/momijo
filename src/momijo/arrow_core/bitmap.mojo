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
# File: src/momijo/arrow_core/bitmap.mojo

from momijo.arrow_core.array import truncate
from momijo.autograd.hook import call
from momijo.core.traits import zero
from momijo.dataframe.diagnostics import safe
from momijo.ir.dialects.annotations import array
from momijo.tensor.broadcast import valid
from momijo.tensor.dtype import nbits
from momijo.tensor.tensor_base import nbytes
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

struct Bitmap(Movable, Sized):
    var bytes: List[UInt8]
    var nbits: Int

    # ---------- Copy initializer (enables safe copies when needed) ----------
fn __copyinit__(out self, other: Self) -> None:
        self.nbits = other.nbits
        var dup = List[UInt8]()
        var i = 0
        var n = len(other.bytes)
        while i < n:
            dup.append(other.bytes[i])
            i += 1
        self.bytes = dup

    # ---------- Constructors ----------
fn __init__(out self, nbits: Int = 0, all_valid: Bool = True) -> None:
        self.nbits = nbits
        var nbytes = _bytes_len(nbits)
        self.bytes = List[UInt8]()
        for _ in range(nbytes):
            if all_valid:
                self.bytes.append(UInt8(0xFF))
            else:
                self.bytes.append(UInt8(0x00))
        _mask_last_byte(self)

    # Build from an existing byte buffer (truncate/pad to 'nbits') and return a new Bitmap.
    @staticmethod
fn from_bytes(buf: List[UInt8], nbits: Int) -> Bitmap:
        var nbytes = _bytes_len(nbits)
        var out_bytes = List[UInt8]()
        var i = 0
        while i < nbytes:
            if i < len(buf):
                out_bytes.append(buf[i])
            else:
                out_bytes.append(UInt8(0))
            i += 1

        var bm = Bitmap(nbits=0, all_valid=False)
        bm.nbits = nbits
        bm.bytes = out_bytes
        _mask_last_byte(bm)
        return bm

    # ---------- Core queries ----------

    @always_inline
fn __len__(self) -> Int:
        return self.nbits
fn len(self) -> Int:
        return self.nbits
fn get_valid(self, i: Int) -> Bool:
        if i < 0 or i >= self.nbits:
            return False
        var byte_idx = i // 8
        var bit_off  = i % 8
        var b = self.bytes[byte_idx]
        var mask: UInt8 = (UInt8(1)) << UInt8(bit_off)
        return (b & mask) != UInt8(0)
fn count_valid(self) -> Int:
        var s: Int = 0
        var nbytes = len(self.bytes)
        var i = 0
        while i < nbytes:
            s += _popcount(self.bytes[i])
            i += 1
        var extra_bits = (nbytes * 8) - self.nbits
        if extra_bits > 0 and nbytes > 0:
            var mask: UInt8 = (UInt8(1) << UInt8(8 - extra_bits)) - UInt8(1)
            var last_masked = self.bytes[nbytes - 1] & mask
            s -= _popcount(self.bytes[nbytes - 1])
            s += _popcount(last_masked)
        return s
fn count_invalid(self) -> Int:
        return self.nbits - self.count_valid()

    # ---------- Mutation ----------
fn set_valid(mut self, i: Int, valid: Bool):
        if i < 0 or i >= self.nbits:
            return
        var byte_idx = i // 8
        var bit_off  = i % 8
        var mask: UInt8 = (UInt8(1)) << UInt8(bit_off)
        if valid:
            self.bytes[byte_idx] = self.bytes[byte_idx] | mask
        else:
(            var inv_mask: UInt8 = UInt8(0xFF) ^ mask) & UInt8(0xFF)
            self.bytes[byte_idx] = self.bytes[byte_idx] & inv_mask
        _mask_last_byte(self)
fn resize(mut self, new_nbits: Int, default_valid: Bool = True) -> None:
        var new_nbytes = _bytes_len(new_nbits)
        var old_nbytes = len(self.bytes)
        if new_nbytes > old_nbytes:
            for _ in range(new_nbytes - old_nbytes):
                if default_valid:
                    self.bytes.append(UInt8(0xFF))
                else:
                    self.bytes.append(UInt8(0x00))
        elif new_nbytes < old_nbytes:
            var new_bytes = List[UInt8]()
            var i = 0
            while i < new_nbytes:
                new_bytes.append(self.bytes[i])
                i += 1
            self.bytes = new_bytes
        self.nbits = new_nbits
        _mask_last_byte(self)

# ---------- Free functions ----------
fn bitmap_set_valid(mut bm: Bitmap, i: Int, valid: Bool) -> None:
    # NOTE: mutates a copy under current call semantics (demo-only)
    bm.set_valid(i, valid)
fn bitmap_get_valid(bm: Bitmap, i: Int) -> Bool:
    return bm.get_valid(i)

# ---------- Private helpers ----------
fn _bytes_len(nbits: Int) -> Int:
    var q = nbits // 8
    var r = nbits % 8
    return q if r == 0 else (q + 1)
fn _popcount(b: UInt8) -> Int:
    var x = b
    var c: Int = 0
    while x != UInt8(0):
        c += 1
        x = x & (x - UInt8(1))
    return c
fn _mask_last_byte(mut bm: Bitmap):
    var r = bm.nbits % 8
    if r == 0:
        return
    var mask: UInt8 = (UInt8(1) << UInt8(r)) - UInt8(1)
    var last_idx = len(bm.bytes) - 1
    bm.bytes[last_idx] = bm.bytes[last_idx] & mask
"""