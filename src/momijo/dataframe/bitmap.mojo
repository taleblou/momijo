# Project:      Momijo
# Module:       dataframe.bitmap
# File:         bitmap.mojo
# Path:         dataframe/bitmap.mojo
#
# Description:  dataframe.bitmap — Bitmap module for Momijo DataFrame.
#               Implements core data structures, algorithms, and convenience APIs for production use.
#               Designed as a stable, composable building block within the Momijo public API.
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
#   - Structs: Bitmap
#   - Key functions: min_int, __init__, __copyinit__, __len__, len, __str__, get, set, count_true, count_false, any_true, none_true, to_mask, resize, invert, bit_and, bit_or, all_true
#   - Static methods present.



fn min_int(a: Int, b: Int) -> Int:
    if a < b:
        return a
    return b

# Bool-backed Bitmap
struct Bitmap(Copyable, Movable):
    var bits: List[Bool]

    fn __init__(out self):
        self.bits = List[Bool]()

    fn __init__(out self, n: Int, fill: Bool = True):
        var b = List[Bool]()
        var i = 0
        while i < n:
            b.append(fill)
            i += 1
        self.bits = b

    # Copyable ctor
    fn __copyinit__(out self, other: Self):
        var b = List[Bool]()
        var i = 0
        var n = len(other.bits)
        while i < n:
            b.append(other.bits[i])
            i += 1
        self.bits = b

    fn __len__(self) -> Int:
        return len(self.bits)

    fn len(self) -> Int:
        return len(self.bits)

    fn __str__(self) -> String:
        var s = String("[")
        var i = 0
        var n = len(self.bits)
        while i < n:
            if self.bits[i]:
                s += String("1")
            else:
                s += String("0")
            if i < n - 1:
                s += String(",")
            i += 1
        s += String("]")
        return s

    fn get(self, i: Int) -> Bool:
        if i < 0 or i >= len(self.bits):
            return False
        return self.bits[i]

    fn set(mut self, i: Int, v: Bool) -> Bool:
        if i < 0 or i >= len(self.bits):
            return False
        self.bits[i] = v
        return True

    # Check if a bit is set (alias for get)
    fn is_set(self, i: Int) -> Bool:
        return self.get(i)

    fn count_true(self) -> Int:
        var c = 0
        var i = 0
        var n = len(self.bits)
        while i < n:
            if self.bits[i]:
                c += 1
            i += 1
        return c

    fn count_false(self) -> Int:
        return len(self.bits) - self.count_true()

    fn any_true(self) -> Bool:
        var i = 0
        var n = len(self.bits)
        while i < n:
            if self.bits[i]:
                return True
            i += 1
        return False

    fn none_true(self) -> Bool:
        return not self.any_true()

    fn to_mask(self) -> List[Bool]:
        var out = List[Bool]()
        var i = 0
        var n = len(self.bits)
        while i < n:
            out.append(self.bits[i])
            i += 1
        return out

    fn resize(mut self, n: Int, fill: Bool = False):
        var cur = len(self.bits)
        if n == cur:
            return

        var out = List[Bool]()
        var keep = 0
        if n < cur:
            keep = n
        else:
            keep = cur

        var i = 0
        while i < keep:
            out.append(self.bits[i])
            i += 1

        if n > cur:
            var j = cur
            while j < n:
                out.append(fill)
                j += 1

        self.bits = out

    fn invert(mut self):
        var i = 0
        var n = len(self.bits)
        while i < n:
            self.bits[i] = not self.bits[i]
            i += 1

    fn bit_and(self, other: Bitmap) -> Bitmap:
        var n = min_int(len(self.bits), len(other.bits))
        var out = Bitmap(n, False)
        var i = 0
        while i < n:
            out.bits[i] = self.bits[i] and other.bits[i]
            i += 1
        return out

    fn bit_or(self, other: Bitmap) -> Bitmap:
        var n = min_int(len(self.bits), len(other.bits))
        var out = Bitmap(n, False)
        var i = 0
        while i < n:
            out.bits[i] = self.bits[i] or other.bits[i]
            i += 1
        return out

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
