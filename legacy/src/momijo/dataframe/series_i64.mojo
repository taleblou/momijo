# Project:      Momijo 
# Module:       dataframe.series_i64
# File:         series_i64.mojo
# Path:         dataframe/series_i64.mojo
#
# Description:  dataframe.series_i64 — Int64 Series utilities for Momijo DataFrame.
#               Storage (values + validity), cloning/copying, slicing/head,
#               selection (take/gather), and simple reductions (sum/mean).
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
#   - Structs: SeriesI64 (data: List[Int64], valid: Bitmap, name: String).
#   - Key functions: clone, copy, len, get, is_valid, get_name, head,
#                    take, gather, count_valid, sum, mean_as_f64.
#   - Static methods present: N/A.

from momijo.dataframe.bitmap import Bitmap

# Integer series (64-bit semantic), aligned with SeriesStr API style
struct SeriesI64(Copyable, Movable):
    var name: String
    var data: List[Int]
    var valid: Bitmap

    # Constructors
    fn __init__(out self):
        self.name = String("")
        self.data = List[Int]()
        self.valid = Bitmap(0, True)

    fn __init__(out self, data: List[Int], name: String = String("")):
        self.name = name
        self.data = data.copy()
        self.valid = Bitmap(len(data), True)

    fn __init__(out self, name: String, values: List[Int]):
        self.name = name
        self.data = values.copy()
        self.valid = Bitmap(len(values), True)

    # Clone / copy
    fn clone(self) -> SeriesI64:
        var out_data = List[Int]()
        var i = 0
        while i < len(self.data):
            out_data.append(self.data[i])
            i += 1
        # create new Series with data and name
        var out = SeriesI64(out_data, self.name)
        # copy validity bitmap
        out.valid = Bitmap(self.valid.len(), True)
        var j = 0
        while j < self.valid.len():
            _ = out.valid.set(j, self.valid.get(j))
            j += 1
        return out.copy()

    fn copy(self) -> SeriesI64:
        return self.clone()

    # Basic API
    fn len(self) -> Int:
        return len(self.data)

    fn get(self, i: Int) -> Int:
        return self.data[i]

    fn set(mut self, i: Int, value: Int):
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


    fn values(self) -> List[Int]:
        return self.data.copy()
    fn validity(self) -> Bitmap:
        return self.valid.copy()
        
    fn head(self, k: Int) -> SeriesI64:
        var out = SeriesI64(List[Int](), self.name)
        var i = 0
        var n = len(self.data)
        while i < k and i < n:
            out.data.append(self.data[i])
            i += 1
        out.valid = Bitmap(len(out.data), True)
        return out

    fn append(mut self, value: Int):
        self.data.append(value)
        self.valid.resize(len(self.data), True)

    fn take(self, idxs: List[Int]) -> SeriesI64:
        var out_data = List[Int]()
        var out_valid = Bitmap(0, True)
        var i = 0
        while i < len(idxs):
            var j = idxs[i]
            if j >= 0 and j < len(self.data):
                out_data.append(self.data[j])
                out_valid.resize(len(out_data), True)
                _ = out_valid.set(len(out_data) - 1, self.valid.get(j))
            else:
                out_data.append(0)  # default for invalid index
                out_valid.resize(len(out_data), True)
                _ = out_valid.set(len(out_data) - 1, False)
            i += 1
        var out = SeriesI64(out_data, self.name)
        out.valid = out_valid.copy()
        return out.copy()

    fn gather(self, mask: Bitmap) -> SeriesI64:
        var out_data = List[Int]()
        var out_valid = Bitmap(0, True)
        var i = 0
        while i < len(mask):
            if mask.get(i):
                out_data.append(self.data[i])
                out_valid.resize(len(out_data), True)
                _ = out_valid.set(len(out_data) - 1, self.valid.get(i))
            i += 1
        var out = SeriesI64(out_data, self.name)
        out.valid = out_valid.copy()
        return out.copy()

    # Aggregations
    fn count_valid(self) -> Int:
        return self.valid.count_true()

    fn sum(self) -> Int:
        var total = 0
        var i = 0
        var n = len(self.data)
        while i < n:
            if self.valid.get(i):
                total += self.data[i]
            i += 1
        return total

    fn mean_as_f64(self) -> Float64:
        var c = self.count_valid()
        if c == 0:
            return 0.0
        return Float64(self.sum()) / Float64(c)

