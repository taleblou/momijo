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
# File: momijo/arrow_core/bitmap.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 
 

struct Bitmap(Copyable, Movable, Sized):
    var bytes: List[UInt8]
    var nbits: Int

    # ---------- Constructors ----------

    fn __init__(out self, nbits: Int = 0, all_valid: Bool = True):
        self.nbits = nbits
        let nbytes = _bytes_len(nbits)
        self.bytes = List[UInt8]()
        for _ in range(nbytes):
            if all_valid:
                self.bytes.append(UInt8(0xFF))
            else:
                self.bytes.append(UInt8(0x00))
        # Mask off extra bits in last byte if not full
        _mask_last_byte(mut self)

    # Build from existing byte buffer (truncated/padded to nbits).
    fn from_bytes(out self, buf: List[UInt8], nbits: Int):
        self.nbits = nbits
        let nbytes = _bytes_len(nbits)
        self.bytes = List[UInt8]()
        var i = 0
        while i < nbytes:
            if i < len(buf):
                self.bytes.append(buf[i])
            else:
                self.bytes.append(UInt8(0))
            i += 1
        _mask_last_byte(mut self)

    # ---------- Core queries ----------

    fn __len__(self) -> Int:
        return self.nbits

    fn len(self) -> Int:
        return self.nbits

    fn get_valid(self, i: Int) -> Bool:
        if i < 0 or i >= self.nbits:
            return False
        let byte_idx = i // 8
        let bit_off = i % 8
        let b = self.bytes[byte_idx]
        let mask: UInt8 = (UInt8(1)) << UInt8(bit_off)
        return (b & mask) != UInt8(0)

    fn count_valid(self) -> Int:
        var s: Int = 0
        let nbytes = len(self.bytes)
        var i = 0
        while i < nbytes:
            s += _popcount(self.bytes[i])
            i += 1
        # If last byte is partially used, mask out extras
        let extra = (nbytes * 8) - self.nbits
        if extra > 0 and nbytes > 0:
            let mask: UInt8 = (UInt8(1) << UInt8(8 - extra)) - UInt8(1)
            let last = self.bytes[nbytes - 1] & mask
            s -= _popcount(self.bytes[nbytes - 1])
            s += _popcount(last)
        return s

    fn count_invalid(self) -> Int:
        return self.nbits - self.count_valid()

    # ---------- Mutation ----------

    fn set_valid(mut self, i: Int, valid: Bool):
        if i < 0 or i >= self.nbits:
            return
        let byte_idx = i // 8
        let bit_off = i % 8
        let mask: UInt8 = (UInt8(1)) << UInt8(bit_off)
        if valid:
            self.bytes[byte_idx] = self.bytes[byte_idx] | mask
        else:
            self.bytes[byte_idx] = self.bytes[byte_idx] & (~mask)
        _mask_last_byte(mut self)

    fn resize(mut self, new_nbits: Int, default_valid: Bool = True):
        let new_nbytes = _bytes_len(new_nbits)
        let old_nbytes = len(self.bytes)
        if new_nbytes > old_nbytes:
            # Extend
            for _ in range(new_nbytes - old_nbytes):
                if default_valid:
                    self.bytes.append(UInt8(0xFF))
                else:
                    self.bytes.append(UInt8(0x00))
        elif new_nbytes < old_nbytes:
            # Shrink
            var new_bytes = List[UInt8]()
            var i = 0
            while i < new_nbytes:
                new_bytes.append(self.bytes[i])
                i += 1
            self.bytes = new_bytes
        self.nbits = new_nbits
        _mask_last_byte(mut self)

# ---------- Free functions ----------

fn bitmap_set_valid(mut bm: Bitmap, i: Int, valid: Bool):
    bm.set_valid(i, valid)

fn bitmap_get_valid(bm: Bitmap, i: Int) -> Bool:
    return bm.get_valid(i)

# ---------- Private helpers ----------

fn _bytes_len(nbits: Int) -> Int:
    let q = nbits // 8
    let r = nbits % 8
    return q if r == 0 else (q + 1)

fn _popcount(b: UInt8) -> Int:
    var x = b
    var c: Int = 0
    while x != UInt8(0):
        c += 1
        x = x & (x - UInt8(1))
    return c

fn _mask_last_byte(mut bm: Bitmap):
    let r = bm.nbits % 8
    if r == 0:
        return
    let mask: UInt8 = (UInt8(1) << UInt8(r)) - UInt8(1)
    let last_idx = len(bm.bytes) - 1
    bm.bytes[last_idx] = bm.bytes[last_idx] & mask
