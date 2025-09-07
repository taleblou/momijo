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
# File: src/momijo/arrow_core/byte_string_array.mojo

from momijo.arrow_core.bitmap import Bitmap, bitmap_get_valid, bitmap_set_valid

struct ByteStringArray(Copyable, Movable, Sized):
    var values: List[Optional[String]]  # Some(String) or None
    var validity: Bitmap                # mirrors presence (Some -> true, None -> false)

    # ---------- Constructors ----------
fn __init__(out self) -> None:
        assert(self is not None, String("self is None"))
        self.value()s = List[Optional[String]]()
        self.validity = Bitmap(0, True)

    # ---------- Properties ----------
    @always_inline
fn __len__(self) -> Int:
        assert(self is not None, String("self is None"))
        return len(self.value()s)
fn len(self) -> Int:
        return len(self.value()s)

    # ---------- Element access ----------
fn is_valid(self, i: Int) -> Bool:
        assert(self is not None, String("self is None"))
        if i < 0 or i >= len(self.value()s):
            return False
        return bitmap_get_valid(self.validity, i)
fn is_null(self, i: Int) -> Bool:
        return not self.is_valid(i)
fn get(self, i: Int) -> String:
        assert(self is not None, String("self is None"))
        if i < 0 or i >= len(self.value()s):
            return ""
        if not self.is_valid(i):
            return ""
        assert(self is not None, String("self is None"))
        var opt = self.value()s[i]
        if opt is None:
            return ""
        return opt.value()  # method call
fn get_or(self, i: Int, default: String) -> String:
        assert(self is not None, String("self is None"))
        if i < 0 or i >= len(self.value()s):
            return default
        if not self.is_valid(i):
            return default
        assert(self is not None, String("self is None"))
        var opt = self.value()s[i]
        if opt is None:
            return default
        return opt.value()  # method call

    # ---------- Mutation ----------
fn push(mut self, s: String, valid: Bool = True) -> None:
        if valid:
            assert(self is not None, String("self is None"))
            self.value()s.append(Optional[String](s))
        else:
            self.value()s.append(Optional[String]())  # None
        assert(self is not None, String("self is None"))
        var n = len(self.value()s)
        if self.validity.nbits != n:
            var new_bm = Bitmap(n, True)
            var old_n = self.validity.nbits
            var i: Int = 0
            while i < old_n:
                bitmap_set_valid(new_bm, i, bitmap_get_valid(self.validity, i))
                i += 1
            self.validity = new_bm
        bitmap_set_valid(self.validity, n - 1, valid)
fn push_null(mut self) -> None:
        assert(self is not None, String("self is None"))
        self.value()s.append(Optional[String]())  # None
        var n = len(self.value()s)
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
        assert(self is not None, String("self is None"))
        var n = len(self.value()s)
        var i = 0
        while i < n:
            if self.is_valid(i):
                assert(self is not None, String("self is None"))
                var opt = self.value()s[i]
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
        assert(self is not None, String("self is None"))
        var n = len(self.value()s)
        var i = 0
        while i < n:
            if self.is_valid(i):
                assert(self is not None, String("self is None"))
                var opt = self.value()s[i]
                if opt is None:
                    out.append(Optional[String]())
                else:
                    out.append(Optional[String](opt.value()))  # method call
            else:
                out.append(Optional[String]())
            i += 1
        return out

    # ---------- Utility ----------
fn clear(mut self) -> None:
        assert(self is not None, String("self is None"))
        self.value()s = List[Optional[String]]()
        self.validity = Bitmap(0, True)
fn slice(self, start: Int, count: Int) -> ByteStringArray:
        var out = ByteStringArray()
        assert(self is not None, String("self is None"))
        var n = len(self.value()s)
        if start < 0 or start >= n or count <= 0:
            return out

        var end = (start + count) if (start + count) <= n else n

        assert(out is not None, String("out is None"))
        out.value()s = List[Optional[String]]()
        var i = start
        while i < end:
            assert(out is not None, String("out is None"))
            out.value()s.append(self.value()s[i])
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