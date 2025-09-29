# Project:      Momijo
# Module:       dataframe.series_str
# File:         series_str.mojo
# Path:         dataframe/series_str.mojo
#
# Description:  dataframe.series_str — String (UTF-8) series for Momijo DataFrame.
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
#   - Structs: SeriesStr
#   - Key functions: __init__, len, head, get, set_name, values, index, where, df_label_where

from momijo.dataframe.bitmap import Bitmap
from momijo.dataframe.selection import Mask as Mask

# Project:      Momijo
# Module:       momijo.dataframe.series_str
# File:         series_str.mojo
# Path:         src/momijo/dataframe/series_str.mojo
# 
# Description:  Minimal String-series stub to satisfy head() signature.


# String series, aligned with SeriesI64 API style
 
struct SeriesStr(Copyable, Movable):
    var name: String
    var data: List[String]
    var valid: Bitmap

    # Default constructor
    fn __init__(out self):
        self.name = String("")
        self.data = List[String]()
        self.valid = Bitmap(0, True)

    # Constructor with data
    fn __init__(out self, data: List[String], name: String = String("")):
        self.name = name
        self.data = data
        self.valid = Bitmap(len(data), True)

    # Clone / copy
    fn clone(self) -> SeriesStr:
        var out_data = List[String]()
        var i = 0
        while i < len(self.data):
            out_data.append(self.data[i])
            i += 1
        var out = SeriesStr(out_data, self.name)
        # copy validity bitmap
        out.valid = Bitmap(self.valid.len(), True)
        var j = 0
        while j < self.valid.len():
            _ = out.valid.set(j, self.valid.get(j))
            j += 1
        return out

    fn copy(self) -> SeriesStr:
        return self.clone()

    # Basic API
    fn len(self) -> Int:
        return len(self.data)

    fn get(self, i: Int) -> String:
        return self.data[i]

    fn set(mut self, i: Int, value: String):
        if i >= 0 and i < len(self.data):
            self.data[i] = value
            _ = self.valid.set(i, True)

    fn is_valid(self, i: Int) -> Bool:
        if i >= 0 and i < len(self.data):
            return self.valid.get(i)
        return False

    fn set_name(mut self, name: String):
        self.name = name

    fn get_name(self) -> String:
        return self.name

    fn head(self, k: Int) -> SeriesStr:
        var out_data = List[String]()
        var i = 0
        while i < k and i < len(self.data):
            out_data.append(self.data[i])
            i += 1
        var out = SeriesStr(out_data, self.name)
        out.valid = Bitmap(len(out_data), True)
        return out

    fn values(self) -> List[String]:
        return self.data

    fn index(self) -> List[Int]:
        var idx = List[Int]()
        var i = 0
        while i < len(self.data):
            idx.append(i)
            i += 1
        return idx

    fn append(mut self, value: String):
        self.data.append(value)
        self.valid.resize(len(self.data), True)

    fn take(self, idxs: List[Int]) -> SeriesStr:
        var out_data = List[String]()
        var out_valid = Bitmap(0, True)
        var i = 0
        while i < len(idxs):
            var j = idxs[i]
            if j >= 0 and j < len(self.data):
                out_data.append(self.data[j])
                out_valid.resize(len(out_data), True)
                _ = out_valid.set(len(out_data) - 1, self.valid.get(j))
            else:
                out_data.append(String(""))
                out_valid.resize(len(out_data), True)
                _ = out_valid.set(len(out_data) - 1, False)
            i += 1
        var out = SeriesStr(out_data, self.name)
        out.valid = out_valid
        return out

    fn gather(self, mask: Bitmap) -> SeriesStr:
        var out_data = List[String]()
        var out_valid = Bitmap(0, True)
        var i = 0
        while i < mask.len():
            if mask.get(i):
                out_data.append(self.data[i])
                out_valid.resize(len(out_data), True)
                _ = out_valid.set(len(out_data) - 1, self.valid.get(i))
            i += 1
        var out = SeriesStr(out_data, self.name)
        out.valid = out_valid
        return out

 




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


# Overload to accept Mask directly (compat with All.mojo using df.where(mask,...))
fn df_label_where(mask: Mask, then: String, else_: String, name: String = "label") -> List[String]:
    var bools = List[Bool]()
    var i = 0
    while i < len(mask.vals):
        bools.append(mask.vals[i])
        i += 1
    return df_label_where(bools, then, else_, name)