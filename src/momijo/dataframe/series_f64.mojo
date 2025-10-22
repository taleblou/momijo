# Project:      Momijo 
# Module:       dataframe.series_f64
# File:         series_f64.mojo
# Path:         dataframe/series_f64.mojo
#
# Description:  dataframe.series_f64 — Float64 Series utilities for Momijo DataFrame.
#               Storage (values + validity), cloning/copying, slicing/head,
#               selection (take/gather), and reductions (sum/mean).
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
#   - Structs: SeriesF64 (data: List[Float64], valid: Bitmap, name: String).
#   - Key functions: clone, copy, len, get, is_valid, get_name, values,
#                    head, take, gather, count_valid, sum, mean.
#   - Static methods present: N/A.

from momijo.dataframe.bitmap import Bitmap

# Float64 series, aligned with SeriesStr/SeriesI64 style
struct SeriesF64(Copyable, Movable):
    var name: String
    var data: List[Float64]
    var valid: Bitmap

    # Constructors
    fn __init__(out self): 
        self.name = String("")
        self.data = List[Float64]()
        self.valid = Bitmap(0, True)

    fn __init__(out self, data: List[Float64], name: String = String("")):
        self.name = name
        self.data = data.copy()
        self.valid = Bitmap(len(data), True)

    fn __init__(out self, name: String, values: List[Float64]):
        self.name = name
        self.data = values.copy()
        self.valid = Bitmap(len(values), True)

    # Clone / copy
    fn clone(self) -> SeriesF64:
        var out_data = List[Float64]()
        var i = 0
        while i < len(self.data):
            out_data.append(self.data[i])
            i += 1
        var out = SeriesF64(out_data, self.name)
        out.valid = Bitmap(self.valid.len(), True)
        var j = 0
        while j < self.valid.len():
            _ = out.valid.set(j, self.valid.get(j))
            j += 1
        return out.copy()



    fn copy(self) -> SeriesF64:
        return self.clone()

    # Basic API
    fn len(self) -> Int:
        return len(self.data)

    fn get(self, i: Int) -> Float64:
        return self.data[i]

    fn set(mut self, i: Int, value: Float64):
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

    fn values(self) -> List[Float64]:
        return self.data.copy()

    fn validity(self) -> Bitmap:
        return self.valid.copy()

    fn head(self, k: Int) -> SeriesF64:
        var out_data = List[Float64]()
        var i = 0
        var n = len(self.data)
        while i < k and i < n:
            out_data.append(self.data[i])
            i += 1
        var out = SeriesF64(out_data, self.name)
        out.valid = Bitmap(len(out_data), True)
        return out

    fn append(mut self, value: Float64):
        self.data.append(value)
        self.valid.resize(len(self.data), True)

    fn take(self, idxs: List[Int]) -> SeriesF64:
        var out_data = List[Float64]()
        var i = 0
        while i < len(idxs):
            var j = idxs[i]
            if j >= 0 and j < len(self.data):
                out_data.append(self.data[j])
            else:
                out_data.append(0.0)  # default for invalid index
            i += 1
        var out = SeriesF64(out_data, self.name)
        out.valid = Bitmap(len(out_data), True)
        return out

    fn gather(self, mask: Bitmap) -> SeriesF64:
        var out_data = List[Float64]()
        var i = 0
        while i < len(mask):
            if mask.get(i):
                out_data.append(self.data[i])
            i += 1
        var out = SeriesF64(out_data, self.name)
        out.valid = Bitmap(len(out_data), True)
        return out

    # Aggregations
    fn count_valid(self) -> Int:
        return self.valid.count_true()

    fn sum(self) -> Float64:
        var total = 0.0
        var i = 0
        var n = len(self.data)
        while i < n:
            if self.valid.get(i):
                total += self.data[i]
            i += 1
        return total

    fn mean(self) -> Float64:
        var c = self.count_valid()
        if c == 0:
            return 0.0
        return self.sum() / Float64(c)
