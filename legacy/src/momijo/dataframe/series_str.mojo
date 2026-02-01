# Project:      Momijo 
# Module:       dataframe.series_str
# File:         series_str.mojo
# Path:         dataframe/series_str.mojo
#
# Description:  dataframe.series_str — String Series utilities for Momijo DataFrame.
#               Constructors, slicing, mapping, contains/find, replace, where/label,
#               and small helpers for Mask/Bool interop.
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
#   - Structs: SeriesStr (data/name), Mask (bool vector wrapper) if referenced.
#   - Key functions: from_list, to_list, map_upper, contains, replace_all,
#                    df_label_where (Bool & Mask overloads), get.
#   - Static methods present: N/A.

from momijo.dataframe.bitmap import Bitmap
from momijo.dataframe.selection import Mask as Mask
from collections.list import List

# SeriesStr: string series with validity bitmap and deep-copy semantics.
struct SeriesStr(Copyable, Movable):
    var name:  String
    var data:  List[String]
    var valid: Bitmap

    # ---------------------------------------------------------------------
    # Constructors
    # ---------------------------------------------------------------------
    fn __init__(out self):
        self.name  = String("")
        self.data  = List[String]()
        self.valid = Bitmap(0, True)

    # (data, name) — column-major friendly signature
    fn __init__(out self, data: List[String], name: String = String("")):
        self.name  = String(name)
        self.data  = data.copy()
        self.valid = Bitmap(len(self.data), True)

    # Deep-copy constructor for Copyable (no aliasing)
    fn __copyinit__(out self, other: Self):
        self.name  = String(other.name)
        self.data  = other.data.copy()
        self.valid = other.valid.copy()

    # Clone / Copy (no mutual recursion)
    fn clone(self) -> Self:
        var out = SeriesStr()
        out.name  = String(self.name)
        out.data  = self.data.copy()
        out.valid = self.valid.copy()
        return out.copy()

    fn copy(self) -> Self:
        return self.clone()

    # ---------------------------------------------------------------------
    # Introspection
    # ---------------------------------------------------------------------
    fn len(self) -> Int:
        return len(self.data)

    fn is_empty(self) -> Bool:
        return len(self.data) == 0

    fn null_count(self) -> Int:
        return self.valid.len() - self.valid.count_true()

    fn get_name(self) -> String:
        return self.name

    fn set_name(mut self, name: String) -> None:
        self.name = String(name)

    fn validity(self) -> Bitmap:
        return self.valid.copy()

    # ---------------------------------------------------------------------
    # Element access
    # ---------------------------------------------------------------------
    fn is_valid(self, i: Int) -> Bool:
        if i < 0 or i >= len(self.data):
            return False
        return self.valid.get(i)

    fn set_null(mut self, i: Int) -> None:
        if i >= 0 and i < self.valid.len():
            _ = self.valid.set(i, False)

    fn get(self, i: Int) -> String:
        # Caller should check validity; we still guard the index.
        if i < 0 or i >= len(self.data):
            return String("")
        return self.data[i]

    fn set(mut self, i: Int, value: String) -> None:
        if i >= 0 and i < len(self.data):
            self.data[i] = String(value)
            _ = self.valid.set(i, True)

    # ---------------------------------------------------------------------
    # Resize / mutation
    # ---------------------------------------------------------------------
    # if logical length = n (truncates or pads with empty strings).
    fn resize(mut self, n: Int) -> None:
        var cur = len(self.data)
        if n <= cur:
            # Truncate data and validity
            var trimmed = List[String]()
            var i = 0
            while i < n:
                trimmed.append(self.data[i])
                i += 1
            self.data = trimmed
            self.valid.resize(n, True)
        else:
            # Grow with empty strings; mark them valid by default
            var i = 0
            while i < (n - cur):
                self.data.append(String(""))
                i += 1
            self.valid.resize(n, True)

    fn clear(mut self) -> None:
        self.data  = List[String]()
        self.valid = Bitmap(0, True)

    # ---------------------------------------------------------------------
    # Appends / bulk ops
    # ---------------------------------------------------------------------
    fn append(mut self, value: String) -> None:
        self.data.append(String(value))
        self.valid.resize(len(self.data), True)

    fn extend(mut self, more: List[String]) -> None:
        var j = 0
        var m = len(more)
        while j < m:
            self.data.append(String(more[j]))
            j += 1
        self.valid.resize(len(self.data), True)

    # ---------------------------------------------------------------------
    # Slicing / selection
    # ---------------------------------------------------------------------
    fn head(self, k: Int) -> Self:
        var out_data = List[String]()
        var n = len(self.data)
        var t = k
        if t < 0:
            t = 0
        if t > n:
            t = n
        var i = 0
        while i < t:
            out_data.append(self.data[i])
            i += 1
        var out = SeriesStr(out_data, self.name)
        out.valid = Bitmap(t, True)
        # Preserve existing nulls where applicable
        i = 0
        while i < t:
            _ = out.valid.set(i, self.valid.get(i))
            i += 1
        return out

    fn take(self, idxs: List[Int]) -> Self:
        var out_data  = List[String]()
        var out_valid = Bitmap(0, True)
        var i = 0
        var m = len(idxs)
        var n = len(self.data)
        while i < m:
            var j = idxs[i]
            if j >= 0 and j < n:
                out_data.append(self.data[j])
                out_valid.resize(len(out_data), True)
                _ = out_valid.set(len(out_data) - 1, self.valid.get(j))
            else:
                out_data.append(String(""))
                out_valid.resize(len(out_data), True)
                _ = out_valid.set(len(out_data) - 1, False)
            i += 1
        var out = SeriesStr(out_data, self.name)
        out.valid = out_valid.copy()
        return out.copy()

    fn gather(self, mask: Bitmap) -> Self:
        var out_data  = List[String]()
        var out_valid = Bitmap(0, True)
        var i = 0
        var n = len(self.data)
        var m = mask.len()
        # Only iterate over intersecting range
        var limit = n
        if m < limit:
            limit = m
        while i < limit:
            if mask.get(i):
                out_data.append(self.data[i])
                out_valid.resize(len(out_data), True)
                _ = out_valid.set(len(out_data) - 1, self.valid.get(i))
            i += 1
        var out = SeriesStr(out_data, self.name)
        out.valid = out_valid.copy()
        return out.copy()

    # ---------------------------------------------------------------------
    # Conversions / utilities
    # ---------------------------------------------------------------------
    fn values(self) -> List[String]:
        return self.data.copy()

    fn to_list(self) -> List[String]:
        return self.data.copy()

    fn index(self) -> List[Int]:
        var idx = List[Int]()
        var i = 0
        var n = len(self.data)
        while i < n:
            idx.append(i)
            i += 1
        return idx

    fn __str__(self) -> String:
        var s = String("SeriesStr(name=") + self.name + String(", n=") + String(len(self.data)) + String(", data=[")
        var i = 0
        var n = len(self.data)
        while i < n:
            s = s + self.data[i]
            if i + 1 < n:
                s = s + String(", ")
            i += 1
        s = s + String("])")
        return s


 




# ---- Moved from __init__.mojo (facade helpers) ----

# [moved] where
fn where(mask: List[Bool], then: String, else_: String, name: String = "label") -> List[String]:
    var vals = List[String]()
    var i = 0
    while i < len(mask):
        if mask[i]:
            vals.append(String(then))
        else:
            vals.append(String(else_))
        i += 1
    return vals


# Facade: label a boolean mask with two strings (then/else_)
fn df_label_where(mask: List[Bool], then: String, else_: String, name: String = "label") -> List[String]:
    var out = List[String]()
    var i = 0
    var n = len(mask)
    while i < n:
        if mask[i]:
            out.append(then)
        else:
            out.append(else_)
        i += 1
    return out

fn get(self, i: Int) -> String:
    return self.data[i]


# Overload to accept Mask directly (compat with All.mojo using df.where(mask,…))
fn df_label_where(mask: Mask, then: String, else_: String, name: String = "label") -> List[String]:
    var bools = List[Bool]()
    var i = 0
    while i < len(mask.vals):
        bools.append(mask.vals[i])
        i += 1
    return df_label_where(bools, then, else_, name)
