# Project:      Momijo
# Module:       dataframe.series
# File:         series.mojo
# Path:         dataframe/series.mojo
#
# Description:  dataframe.series — Series module for Momijo DataFrame.
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
#   - Structs: Series
#   - Key functions: __init__, values, index, __copyinit__, to_string, series_from_list, series_values, series_index, resample_sum_min

from momijo.dataframe.series_bool import append
struct Series:
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
            s = s + "...\n"
        return s

fn series_from_list(index: List[String], values: List[Int], name: String) -> Series:
    var vs = List[String]()
    var i = 0
    while i < len(values):
        vs.append(String(values[i]))
        i += 1
    var idx = List[String]()
    i = 0
    while i < len(index):
        idx.append(String(index[i]))
        i += 1
    return Series(name, idx, vs)

# -- Added: print-friendly accessors --
fn series_values(s: Series) -> String:
    var xs = s.values_
    var out = String("[")
    var i = 0
    var n = len(xs)
    while i < n:
        out += xs[i]
        if i + 1 < n:
            out += ", "
        i += 1
    out += "]"
    return out

fn series_index(s: Series) -> String:
    var xs = s.index_vals
    var out = String("[")
    var i = 0
    var n = len(xs)
    while i < n:
        out += xs[i]
        if i + 1 < n:
            out += ", "
        i += 1
    out += "]"
    return out
    
fn resample_sum_min(series: List[Float64], freq: Int) -> List[Float64]
    var out = List[Float64]()
    var i = 0
    while i < len(series):
        var s = 0.0
        var k = 0
        while k < freq and i + k < len(series):
            s += series[i + k]
            k += 1
        out.append(s)
        i += freq
    return out