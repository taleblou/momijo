# Project:      Momijo
# Module:       dataframe.series_bool
# File:         series_bool.mojo
# Path:         dataframe/series_bool.mojo
#
# Description:  dataframe.series_bool — Boolean series for Momijo DataFrame.
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
#   - Structs: SeriesBool
#   - Key functions: __init__, len, get, set, set_name, values, head, count_valid, sum_true, logical_not, _binary_op, logical_and, logical_or, logical_xor

from momijo.dataframe.bitmap import Bitmap

struct SeriesBool(Copyable, Movable):
    var name: String
    var data: List[Bool]
    var valid: Bitmap

    # Constructors
    fn __init__(out self):
        self.name = String("")
        self.data = List[Bool]()
        self.valid = Bitmap(0, True)

    fn __init__(out self, data: List[Bool], name: String = String("")):
        self.name = name
        self.data = data
        self.valid = Bitmap(len(data), True)

    fn __init__(out self, data: List[Bool], name: String, valid: Bitmap):
        self.name = name
        self.data = data
        var n = len(data)
        if valid.len() == n:
            self.valid = valid
        else:
            var v = Bitmap(n, True)
            var i = 0
            var m = valid.len()
            while i < n:
                var bit = False
                if i < m:
                    bit = valid.get(i)
                _ = v.set(i, bit)
                i += 1
            self.valid = v

    # Clone / copy
    fn clone(self) -> SeriesBool:
        var out_data = List[Bool]()
        var i = 0
        while i < len(self.data):
            out_data.append(self.data[i])
            i += 1
        var out_valid = Bitmap(self.valid.len(), True)
        var j = 0
        while j < self.valid.len():
            _ = out_valid.set(j, self.valid.get(j))
            j += 1
        return SeriesBool(out_data, self.name, out_valid)

    fn copy(self) -> SeriesBool:
        return self.clone()

    # Basic API
    fn len(self) -> Int:
        return len(self.data)

    fn get(self, i: Int) -> Bool:
        return self.data[i]

    fn set(mut self, i: Int, value: Bool, is_valid: Bool = True):
        if i < 0 or i >= self.len():
            return
        self.data[i] = value
        _ = self.valid.set(i, is_valid)

    fn is_valid(self, i: Int) -> Bool:
        if i >= 0 and i < self.len():
            return self.valid.get(i)
        return False

    fn set_name(mut self, name: String):
        self.name = name

    fn get_name(self) -> String:
        return self.name

    fn values(self) -> List[Bool]:
        return self.data

    fn head(self, k: Int) -> SeriesBool:
        var out_vals = List[Bool]()
        var i = 0
        var n = self.len()
        while i < k and i < n:
            out_vals.append(self.data[i])
            i += 1
        var out_valid = Bitmap(len(out_vals), True)
        return SeriesBool(out_vals, self.name, out_valid)

    fn append(mut self, value: Bool, is_valid: Bool = True):
        self.data.append(value)
        self.valid.resize(len(self.data), True)
        _ = self.valid.set(len(self.data) - 1, is_valid)

    fn take(self, idxs: List[Int]) -> SeriesBool:
        var out_vals = List[Bool]()
        var i = 0
        while i < len(idxs):
            var j = idxs[i]
            if j >= 0 and j < len(self.data):
                out_vals.append(self.data[j])
            else:
                out_vals.append(False)
            i += 1
        var out_valid = Bitmap(len(out_vals), True)
        return SeriesBool(out_vals, self.name, out_valid)

    fn gather(self, mask: Bitmap) -> SeriesBool:
        var out_vals = List[Bool]()
        var out_valid = Bitmap(0, True)
        var i = 0
        while i < len(mask):
            if mask.get(i):
                out_vals.append(self.data[i])
                out_valid.resize(len(out_vals), True)
            i += 1
        return SeriesBool(out_vals, self.name, out_valid)

    # Aggregations / counts
    fn count_valid(self) -> Int:
        var i = 0
        var n = self.len()
        var c = 0
        while i < n:
            if self.valid.get(i):
                c += 1
            i += 1
        return c

    fn sum_true(self) -> Int:
        var i = 0
        var n = self.len()
        var c = 0
        while i < n:
            if self.valid.get(i) and self.data[i]:
                c += 1
            i += 1
        return c

    # Logical ops
    fn logical_not(self) -> SeriesBool:
        var n = self.len()
        var out_vals = List[Bool]()
        var out_valid = Bitmap(n, True)
        var i = 0
        while i < n:
            if self.valid.get(i):
                out_vals.append(not self.data[i])
            else:
                out_vals.append(False)
                _ = out_valid.set(i, False)
            i += 1
        return SeriesBool(out_vals, self.name, out_valid)

    fn _binary_op(self, other: SeriesBool, op_id: Int) -> SeriesBool:
        var n = self.len()
        var out_vals = List[Bool]()
        var out_valid = Bitmap(n, True)
        var i = 0
        while i < n:
            var valid_i = self.valid.get(i) and other.valid.get(i)
            if valid_i:
                var a = self.data[i]
                var b = other.data[i]
                var v = False
                if op_id == 0:
                    v = (a and b)
                elif op_id == 1:
                    v = (a or b)
                else:
                    v = ((a and not b) or (not a and b))
                out_vals.append(v)
            else:
                out_vals.append(False)
                _ = out_valid.set(i, False)
            i += 1
        return SeriesBool(out_vals, self.name, out_valid)

    fn logical_and(self, other: SeriesBool) -> SeriesBool:
        return self._binary_op(other, 0)

    fn logical_or(self, other: SeriesBool) -> SeriesBool:
        return self._binary_op(other, 1)

    fn logical_xor(self, other: SeriesBool) -> SeriesBool:
        return self._binary_op(other, 2)

