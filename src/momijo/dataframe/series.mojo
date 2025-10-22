# Project:      Momijo 
# Module:       dataframe.series
# File:         series.mojo
# Path:         dataframe/series.mojo
#
# Description:  dataframe.series — Generic Series utilities for Momijo DataFrame.
#               Construction/printing, value/index accessors, and simple resampling helpers.
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
#   - Structs: Series (index: List[String], values: List[String], name: String) — if present.
#   - Key functions: values, index, to_string, series_from_list, series_values, series_index,
#                    resample_sum_min.
#   - Static methods present: N/A.
 
from momijo.dataframe.series_bool import append
struct Series(Copyable, Movable):
    var name: String
    var index_vals: List[String]
    var values_: List[String]

    fn __init__(out self, name: String, index: List[String], values: List[String]):
        self.name = String(name)
        self.index_vals = List[String]()
        self.values_ = List[String]()
        var i = 0
        while i < len(index):
            self.index_vals.append(String(index[i]))
            i += 1
        i = 0
        while i < len(values):
            self.values_.append(String(values[i]))
            i += 1

    fn values(self) -> List[String]:
        return self.values_

    fn index(self) -> List[String]:
        return self.index_vals

    fn __copyinit__(out self, other: Series):
        self.name = String(other.name)
        self.index_vals = List[String]()
        self.values_ = List[String]()
        var i = 0
        while i < len(other.index_vals):
            self.index_vals.append(String(other.index_vals[i]))
            i += 1
        i = 0
        while i < len(other.values_):
            self.values_.append(String(other.values_[i]))
            i += 1

    fn to_string(self) -> String:
        var s = String("Series(name=") + self.name + ", len=" + String(len(self.values_)) + ")\n"
        var n = len(self.values_)
        var show = n
        if show > 8:
            show = 8
        var k = 0
        while k < show:
            s = s + self.index_vals[k] + ": " + self.values_[k] + "\n"
            k += 1
        if n > show:
            s = s + "…\n"
        return s

   

# Build a Series from a string index and integer values.
# NOTE: explicit return type is important.
fn series_from_list(index: List[String], values: List[Int], name: String) -> Series:
    # Convert Int values to String payload to match Series[String] storage
    var vs = List[String]()
    var i = 0
    while i < len(values):
        vs.append(String(values[i]))
        i += 1
    # Series constructor order: (name, index, values)
    var s = Series(name, index.copy(), vs)
    return s.copy()

# Pretty-print the index as "[a, b, c]".
# NOTE: explicit return type is important.
fn series_index(s: Series) -> String:
    var out = String("[")
    var i = 0
    var n = len(s.index_vals)
    while i < n:
        out += s.index_vals[i]
        if i + 1 < n:
            out += String(", ")
        i += 1
    out += "]"
    return out 


# Sum-resample over fixed-size windows (freq)
fn resample_sum_min(series: List[Float64], freq: Int) -> List[Float64]:
    if freq <= 0:
        return List[Float64]()
    var out = List[Float64]()
    var i = 0
    var n = len(series)
    while i < n:
        var s = 0.0
        var k = 0
        while k < freq and i + k < n:
            s += series[i + k]
            k += 1
        out.append(s)
        i += freq
    return out
