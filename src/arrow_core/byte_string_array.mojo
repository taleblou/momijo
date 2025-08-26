# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.arrow_core
# File: momijo/arrow_core/byte_string_array.mojo

from momijo.arrow_core.bitmap import Bitmap, bitmap_set_valid, bitmap_get_valid

# Simplified & robust implementation:
# Store List[Optional[String]] plus a validity Bitmap. We only call Optional.value()
# (as a method) when it is not None.

struct ByteStringArray(Copyable, Movable, Sized):
    var values: List[Optional[String]]  # Some(String) or None
    var validity: Bitmap                # mirrors presence (Some -> true, None -> false)

    # ---------- Constructors ----------
    fn __init__(out self):
        self.values = List[Optional[String]]()
        self.validity = Bitmap(0, True)

    # ---------- Properties ----------
    @always_inline
    fn __len__(self) -> Int:
        return len(self.values)

    fn len(self) -> Int:
        return len(self.values)

    # ---------- Element access ----------
    fn is_valid(self, i: Int) -> Bool:
        if i < 0 or i >= len(self.values):
            return False
        return bitmap_get_valid(self.validity, i)

    fn is_null(self, i: Int) -> Bool:
        return not self.is_valid(i)

    fn get(self, i: Int) -> String:
        if i < 0 or i >= len(self.values):
            return ""
        if not self.is_valid(i):
            return ""
        var opt = self.values[i]
        if opt is None:
            return ""
        return opt.value()  # method call

    fn get_or(self, i: Int, default: String) -> String:
        if i < 0 or i >= len(self.values):
            return default
        if not self.is_valid(i):
            return default
        var opt = self.values[i]
        if opt is None:
            return default
        return opt.value()  # method call

    # ---------- Mutation ----------
    fn push(mut self, s: String, valid: Bool = True):
        if valid:
            self.values.append(Optional[String](s))
        else:
            self.values.append(Optional[String]())  # None
        var n = len(self.values)
        if self.validity.nbits != n:
            var new_bm = Bitmap(n, True)
            var old_n = self.validity.nbits
            var i: Int = 0
            while i < old_n:
                bitmap_set_valid(new_bm, i, bitmap_get_valid(self.validity, i))
                i += 1
            self.validity = new_bm
        bitmap_set_valid(self.validity, n - 1, valid)

    fn push_null(mut self):
        self.values.append(Optional[String]())  # None
        var n = len(self.values)
        if self.validity.nbits != n:
            var new_bm = Bitmap(n, True)
            var old_n = self.validity.nbits
            var i: Int = 0
            while i < old_n:
                bitmap_set_valid(new_bm, i, bitmap_get_valid(self.validity, i))
                i += 1
            self.validity = new_bm
        bitmap_set_valid(self.validity, n - 1, False)

    # ---------- Conversion ----------
    fn to_strings(self) -> List[String]:
        var out = List[String]()
        var n = len(self.values)
        var i = 0
        while i < n:
            if self.is_valid(i):
                var opt = self.values[i]
                if opt is None:
                    out.append("")
                else:
                    out.append(opt.value())  # method call
            else:
                out.append("")
            i += 1
        return out

    fn to_optional_strings(self) -> List[Optional[String]]:
        var out = List[Optional[String]]()
        var n = len(self.values)
        var i = 0
        while i < n:
            if self.is_valid(i):
                var opt = self.values[i]
                if opt is None:
                    out.append(Optional[String]())
                else:
                    out.append(Optional[String](opt.value()))  # method call
            else:
                out.append(Optional[String]())
            i += 1
        return out

    # ---------- Utility ----------
    fn clear(mut self):
        self.values = List[Optional[String]]()
        self.validity = Bitmap(0, True)

    fn slice(self, start: Int, count: Int) -> ByteStringArray:
        var out = ByteStringArray()
        var n = len(self.values)
        if start < 0 or start >= n or count <= 0:
            return out

        var end = (start + count) if (start + count) <= n else n

        out.values = List[Optional[String]]()
        var i = start
        while i < end:
            out.values.append(self.values[i])
            i += 1

        out.validity = Bitmap(end - start, True)
        var k: Int = 0
        i = start
        while i < end:
            bitmap_set_valid(out.validity, k, self.is_valid(i))
            i += 1
            k += 1

        return out

# ---------- Public factories ----------
fn byte_string_array_from_strings(strings: List[String]) -> ByteStringArray:
    var arr = ByteStringArray()
    var i: Int = 0
    while i < len(strings):
        arr.push(strings[i], True)
        i += 1
    return arr

fn byte_string_array_from_optional_strings(strings: List[Optional[String]]) -> ByteStringArray:
    var arr = ByteStringArray()
    var i: Int = 0
    while i < len(strings):
        var opt = strings[i]
        if opt is None:
            arr.push_null()
        else:
            arr.push(opt.value(), True)  # method call
        i += 1
    return arr
