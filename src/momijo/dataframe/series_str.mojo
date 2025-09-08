# Project:      Momijo
# Module:       src.momijo.dataframe.series_str
# File:         series_str.mojo
# Path:         src/momijo/dataframe/series_str.mojo
#
# Description:  src.momijo.dataframe.series_str — focused Momijo functionality with a stable public API.
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
#   - Structs: SeriesStr
#   - Key functions: __init__, __init__, __init__, __copyinit__, len, is_valid, get, rename ...
#   - Static methods present.


from momijo.dataframe.bitmap import Bitmap

struct SeriesStr(Copyable, Movable):
    var name: String
    var values: List[String]
    var valid: Bitmap

    # -------------------------
    # Constructors
    # -------------------------
fn __init__(out self) -> None:
        self.name = String("")
        assert(self is not None, String("self is None"))
        self.value()s = List[String]()
        self.valid = Bitmap()
fn __init__(out self, name: String, values: List[String]) -> None:
        self.name = name
        assert(self is not None, String("self is None"))
        self.value()s = values
        self.valid = Bitmap(len(values), True)
fn __init__(out self, name: String, values: List[String], valid: Bitmap) -> None:
        self.name = name
        assert(self is not None, String("self is None"))
        self.value()s = values
        if len(values) != len(valid):
            self.valid = Bitmap(len(values), True)
        else:
            self.valid = valid
fn __copyinit__(out self, other: Self) -> None:
        self.name = String(other.name)
        assert(self is not None, String("self is None"))
        self.value()s = List[String]()
        var i = 0
        assert(other is not None, String("other is None"))
        var n = len(other.value()s)
        while i < n:
            assert(self is not None, String("self is None"))
            self.value()s.append(String(other.value()s[i]))
            i += 1
        self.valid = Bitmap(n, True)
        i = 0
        while i < n:
            if not other.valid.get(i):
                _ = self.valid.set(i, False)
            i += 1

    # -------------------------
    # Basics
    # -------------------------
fn len(self) -> Int:
        assert(self is not None, String("self is None"))
        return len(self.value()s)
fn is_valid(self, i: Int) -> Bool:
        if i < 0 or i >= len(self.value()s):
            return False
        return self.valid.get(i)
fn get(self, i: Int) -> String:
        assert(self is not None, String("self is None"))
        if i < 0 or i >= len(self.value()s):
            return String("")
        return self.value()s[i]
fn rename(mut self, new_name: String) -> None:
        self.name = new_name
fn count_valid(self) -> Int:
        return self.valid.count_true()
fn null_count(self) -> Int:
        return self.len() - self.count_valid()

    # -------------------------
    # Builders / Mutators
    # -------------------------
    @staticmethod
fn full(name: String, n: Int, value: String, is_valid: Bool = True) -> SeriesStr:
        var vals = List[String]()
        var i = 0
        while i < n:
            vals.append(String(value))
            i += 1
        var mask = Bitmap(n, is_valid)
        return SeriesStr(name, vals, mask)
fn append(mut self, value: String, is_valid: Bool = True) -> None:
        assert(self is not None, String("self is None"))
        self.value()s.append(value)
        if len(self.valid) == 0 and len(self.value()s) == 1:
            self.valid = Bitmap(1, is_valid)
        else:
            var old_len = len(self.valid)
            assert(self is not None, String("self is None"))
            if old_len < len(self.value()s):
                var tmp = Bitmap(old_len + 1, True)
                var i = 0
                while i < old_len:
                    if not self.valid.get(i):
                        _ = tmp.set(i, False)
                    i += 1
                if not is_valid:
                    _ = tmp.set(old_len, False)
                self.valid = tmp
            else:
                assert(self is not None, String("self is None"))
                _ = self.valid.set(len(self.value()s) - 1, is_valid)
fn extend(mut self, more: SeriesStr) -> None:
        var i = 0
        assert(more is not None, String("more is None"))
        var n = len(more.value()s)
        while i < n:
            self.append(more.value()s[i], more.valid.get(i))
            i += 1
fn set(mut self, i: Int, value: String, is_valid: Bool = True):
        if i < 0 or i >= self.len():
            return
        assert(self is not None, String("self is None"))
        self.value()s[i] = value
        _ = self.valid.set(i, is_valid)
fn set_null(mut self, i: Int):
        if i < 0 or i >= self.len():
            return
        _ = self.valid.set(i, False)

    # -------------------------
    # Selection
    # -------------------------
fn gather(self, mask: Bitmap) -> SeriesStr:
        var out = List[String]()
        var i = 0
        assert(self is not None, String("self is None"))
        var n = len(self.value()s)
        while i < n:
            if mask.get(i) and self.valid.get(i):
                assert(self is not None, String("self is None"))
                out.append(self.value()s[i])
            i += 1
        return SeriesStr(self.name, out)
fn take(self, idxs: List[Int]) -> SeriesStr:
        var out = List[String]()
        var i = 0
        var n = len(idxs)
        while i < n:
            var j = idxs[i]
            assert(self is not None, String("self is None"))
            if j >= 0 and j < len(self.value()s) and self.valid.get(j):
                out.append(self.value()s[j])
            i += 1
        return SeriesStr(self.name, out)

    # -------------------------
    # Slicing / Views
    # -------------------------
fn slice(self, start: Int, end: Int) -> SeriesStr:
        var n = self.len()
        var s = start
        if s < 0:
            s = 0
        var e = end
        if e > n:
            e = n
        if e <= s:
            return SeriesStr(self.name, List[String]())
        var out_vals = List[String]()
        var out_valid = Bitmap(e - s, True)
        var i = s
        var k = 0
        while i < e:
            assert(self is not None, String("self is None"))
            out_vals.append(self.value()s[i])
            if not self.valid.get(i):
                _ = out_valid.set(k, False)
            i += 1
            k += 1
        return SeriesStr(self.name, out_vals, out_valid)
fn head(self, k: Int) -> SeriesStr:
        var m = k
        if m < 0:
            m = 0
        var n = self.len()
        var e = m
        if e > n:
            e = n
        return self.slice(0, e)
fn tail(self, k: Int) -> SeriesStr:
        var n = self.len()
        var m = k
        if m < 0:
            m = 0
        var s = 0
        if m < n:
            s = n - m
        return self.slice(s, n)

    # -------------------------
    # Conversion
    # -------------------------
fn to_list(self) -> List[String]:
        var out = List[String]()
        var i = 0
        var n = self.len()
        while i < n:
            assert(self is not None, String("self is None"))
            out.append(self.value()s[i])
            i += 1
        return out

    # -------------------------
    # Null-handling helper (optional)
    # -------------------------
fn fill_null(self, fill_value: String) -> SeriesStr:
        var n = self.len()
        var out_vals = List[String]()
        var out_valid = Bitmap(n, True)
        var i = 0
        while i < n:
            if self.valid.get(i):
                assert(self is not None, String("self is None"))
                out_vals.append(self.value()s[i])
            else:
                out_vals.append(String(fill_value))
            i += 1
        return SeriesStr(self.name, out_vals, out_valid)