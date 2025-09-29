# Project:      Momijo
# Module:       dataframe.io_pickle_mnp
# File:         io_pickle_mnp.mojo
# Path:         dataframe/io_pickle_mnp.mojo
#
# Description:  dataframe.io_pickle_mnp — Io Pickle Mnp module for Momijo DataFrame.
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
#   - Structs: —
#   - Key functions: _esc, _unesc, to_mnp_string, _split_esc_tabs, from_mnp_string

from collections.list import List

from momijo.dataframe.frame import DataFrame

fn _esc(s: String) -> String:
    var out = String("")
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch == "\\":
            out += String("\\\\")
        elif ch == "\t":
            out += String("\\t")
        elif ch == "\n":
            out += String("\\n")
        elif ch == "\r":
            out += String("\\r")
        else:
            out += String(ch)
        i += 1
    return out

fn _unesc(s: String) -> String:
    var out = String("")
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch != "\\":
            out += String(ch)
            i += 1
            continue
        i += 1
        if i >= len(s):
            break
        var e = s[i]
        if e == "t":
            out += String("\t")
        elif e == "n":
            out += String("\n")
        elif e == "r":
            out += String("\r")
        elif e == "\\":
            out += String("\\")
        else:
            out += String(e)
        i += 1
    return out

fn to_mnp_string(df: DataFrame) -> String:
    var s = String("MNP1\n")
    s += String(df.ncols())
    s += String("\t")
    s += String(df.nrows())
    s += String("\n")
    s += _esc(df.name)
    s += String("\n")
# header line (columns)
    var c = 0
    while c < df.ncols():
        if c > 0:
            s += String("\t")
        s += _esc(df.col_names[c])
        c += 1
    s += String("\n")
# rows
    var r = 0
    while r < df.nrows():
        var c2 = 0
        while c2 < df.ncols():
            if c2 > 0:
                s += String("\t")
            s += _esc(df.value_str(r, c2))
            c2 += 1
        s += String("\n")
        r += 1
    return s

# parse a line with escaped tabs into fields
fn _split_esc_tabs(line: String) -> List[String]:
    var fields = List[String]()
    var cur = String("")
    var i = 0
    while i < len(line):
        var ch = line[i]
        if ch == "\\":
            if i + 1 < len(line):
                cur += String(ch)
                cur += String(line[i+1])
                i += 2
            else:
                cur += String(ch)
                i += 1
            continue
        if ch == "\t":
            fields.append(_unesc(cur))
            cur = String("")
            i += 1
            continue
        cur += String(ch)
        i += 1
    fields.append(_unesc(cur))
    return fields

fn from_mnp_string(s: String) -> DataFrame:
# split lines
    var lines = List[String]()
    var cur = String("")
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch == "\n":
            lines.append(cur)
            cur = String("")
        else:
            cur += String(ch)
        i += 1
    if len(cur) > 0:
        lines.append(cur)

# validate header
    if len(lines) < 4 or lines[0] != String("MNP1"):
# return empty frame with one empty column
        return DataFrame(List[String]([String("col")]), List[List[String]]([List[String]()]), List[String](), String(""))
# parse sizes
    var sizes = _split_esc_tabs(lines[1])
    var ncols = 0
    var nrows = 0
    if len(sizes) >= 2:
# crude String->Int parse: assume decimal only
        var a = sizes[0]
        var b = sizes[1]
# count digits
        var j = 0
        while j < len(a):
            j += 1
        ncols = j  # fallback if no proper parse support; will be corrected below
        j = 0
        while j < len(b):
            j += 1
        nrows = j
# Better: re-count using header/rows length to be self-consistent
    var cols = _split_esc_tabs(lines[3])
    ncols = len(cols)
# collect data lines
    var data = List[List[String]]()
    var row_start = 4
    var ln = row_start
    while ln < len(lines):
        var fields = _split_esc_tabs(lines[ln])
        data.append(fields)
        ln += 1
    nrows = len(data)

# transpose to column-major
    var cols_vals = List[List[String]]()
    var c0 = 0
    while c0 < ncols:
        cols_vals.append(List[String]())
        c0 += 1
    var r0 = 0
    while r0 < nrows:
        var c1 = 0
        while c1 < ncols:
            var val = String("")
            if c1 < len(data[r0]):
                val = data[r0][c1]
            cols_vals[c1].append(val)
            c1 += 1
        r0 += 1

# index = 0..nrows-1
    var index = List[String]()
    var k = 0
    while k < nrows:
        index.append(String(k))
        k += 1

    var name = _unesc(lines[2])
    return DataFrame(cols, cols_vals, index, name)