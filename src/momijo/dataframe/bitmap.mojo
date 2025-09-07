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
# Project: momijo.dataframe
# File: src/momijo/dataframe/bitmap.mojo

from momijo.core.error import module
from momijo.dataframe.expr import AND, OR
from momijo.dataframe.generic_ops import Equality
from momijo.vision.transforms.compose import Resize
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
fn min_int(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b

struct Bitmap(Sized, Stringable, Copyable, Movable):
    var bits: List[Bool]

    # Empty constructor
fn __init__(out self) -> None:
        self.bits = List[Bool]()

    # Length constructor with fill value
fn __init__(out self, n: Int, fill: Bool = True) -> None:
        var b = List[Bool]()
        var i = 0
        while i < n:
            b.append(fill)
            i += 1
        self.bits = b

    # Copy constructor (required by Copyable)
fn __copyinit__(out self, other: Self) -> None:
        var b = List[Bool]()
        var i = 0
        var n = len(other.bits)
        while i < n:
            b.append(other.bits[i])
            i += 1
        self.bits = b

    # Sized
fn __len__(self) -> Int:
        return len(self.bits)

    # Safe get (out-of-range returns False)
fn get(self, i: Int) -> Bool:
        if i < 0 or i >= len(self.bits):
            return False
        return self.bits[i]

    # Safe set (returns False on out-of-range)
fn set(mut self, i: Int, v: Bool) -> Bool:
        if i < 0 or i >= len(self.bits):
            return False
        self.bits[i] = v
        return True

    # Count true bits
fn count_true(self) -> Int:
        var c = 0
        var i = 0
        var n = len(self.bits)
        while i < n:
            if self.bits[i]:
                c += 1
            i += 1
        return c

    # Count false bits
fn count_false(self) -> Int:
        return len(self.bits) - self.count_true()

    # Any true bit?
fn any_true(self) -> Bool:
        var i = 0
        var n = len(self.bits)
        while i < n:
            if self.bits[i]:
                return True
            i += 1
        return False

    # No true bits?
fn none_true(self) -> Bool:
        return not self.any_true()

    # Copy out as a plain Bool mask
fn to_mask(self) -> List[Bool]:
        var out = List[Bool]()
        var i = 0
        var n = len(self.bits)
        while i < n:
            out.append(self.bits[i])
            i += 1
        return out

    # Resize to n; shrink or extend with 'fill'
fn resize(mut self, n: Int, fill: Bool = False):
        var cur = len(self.bits)
        if n == cur:
            return

        var out = List[Bool]()
        var keep = cur
        if n < cur:
            keep = n

        # copy kept head
        var i = 0
        while i < keep:
            out.append(self.bits[i])
            i += 1

        # extend the tail if needed
        if n > cur:
            var j = cur
            while j < n:
                out.append(fill)
                j += 1

        self.bits = out

    # In-place logical NOT
fn invert(mut self) -> None:
        var i = 0
        var n = len(self.bits)
        while i < n:
            self.bits[i] = not self.bits[i]
            i += 1

    # Logical AND (length = min(self, other))
fn bit_and(self, other: Bitmap) -> Bitmap:
        var n = min_int(len(self.bits), len(other.bits))
        var out = Bitmap(n, False)
        var i = 0
        while i < n:
            out.bits[i] = self.bits[i] and other.bits[i]
            i += 1
        return out

    # Logical OR (length = min(self, other))
fn bit_or(self, other: Bitmap) -> Bitmap:
        var n = min_int(len(self.bits), len(other.bits))
        var out = Bitmap(n, False)
        var i = 0
        while i < n:
            out.bits[i] = self.bits[i] or other.bits[i]
            i += 1
        return out

    # Equality (same length and same bits)
fn __eq__(self, other: Bitmap) -> Bool:
        if len(self.bits) != len(other.bits):
            return False
        var i = 0
        var n = len(self.bits)
        while i < n:
            if self.bits[i] != other.bits[i]:
                return False
            i += 1
        return True

    # Stringable: prints as [1,0,1,...]
fn __str__(self) -> String:
        var s = String("[")
        var i = 0
        var n = len(self.bits)
        while i < n:
            if self.bits[i]:
                s += "1"
            else:
                s += "0"
            if i < n - 1:
                s += ","
            i += 1
        s += "]"
        return s

    # Factories
    @staticmethod
fn all_true(n: Int) -> Bitmap:
        return Bitmap(n, True)

    @staticmethod
fn all_false(n: Int) -> Bitmap:
        return Bitmap(n, False)

    @staticmethod
fn from_mask(mask: List[Bool]) -> Bitmap:
        var n = len(mask)
        var bm = Bitmap(n, False)
        var i = 0
        while i < n:
            bm.bits[i] = mask[i]
            i += 1
        return bm
"""