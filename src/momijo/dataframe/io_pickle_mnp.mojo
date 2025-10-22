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

fn _esc(s: String):
    var out = String("")
    var i = 0
    while i < len(s):
        var ch = s.bytes()[i]
        if ch == UInt8(92):      # '\'
            out = out + String("\\")
        elif ch == UInt8(9):     # '	'
            out = out + String("\t")
        elif ch == UInt8(10):    # '
'
            out = out + String("\n")
        elif ch == UInt8(13):    # '
'
            out = out + String("\r")
        else:
            out = out + String(ch)
        i += 1
    return out
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

fn _unesc(s: String):
    var out = String("")
    var i = 0
    var n = len(s)
    while i < n:
        var ch = s.bytes()[i]
        if ch == UInt8(92) and i + 1 < n:   # '\'
            var nx = s.bytes()[i+1]
            if nx == UInt8(116):       # 't'
                out = out + String("	")
                i += 2
                continue
            elif nx == UInt8(110):     # 'n'
                out = out + String("
")
                i += 2
                continue
            elif nx == UInt8(114):     # 'r'
                out = out + String("
")
                i += 2
                continue
            elif nx == UInt8(92):      # '\'
                out = out + String("\")
                i += 2
                continue
        out = out + String(ch)
        i += 1
    return out
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

fn to_mnp_string(df: DataFrame):
    # MNP layout:
    # line0: MNP
    # line1: nrows	ncols
    # line2: df.name (escaped)
    # line3: colnames separated by '	' (escaped)
    # line[4:]: rows of values, each column escaped, separated by '	'
    var nrows = df.nrows()
    var ncols = df.ncols()
    var out = String("MNP
")
    out = out + String(String(nrows) + String("	") + String(ncols) + String("
"))
    out = out + _esc(df.index_name) + String("
")
    # columns header
    var c = 0
    while c < ncols:
        out = out + _esc(df.col_names[c])
        if c + 1 < ncols:
            out = out + String("	")
        c += 1
    out = out + String("
")
    # rows
    var r = 0
    while r < nrows:
        c = 0
        while c < ncols:
            out = out + _esc(df.cols[c][r])
            if c + 1 < ncols:
                out = out + String("	")
            c += 1
        out = out + String("
")
        r += 1
    return out
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
fn _split_esc_tabs(line: String):
    # Split a line by unescaped tabs, parsing escapes \t, \n, \r, \\.
    var out = List[String]()
    var buf = String("")
    var i = 0
    var n = len(line)
    while i < n:
        var ch = line.bytes()[i]
        if ch == UInt8(92) and i + 1 < n:  # backslash
            var nx = line.bytes()[i+1]
            if nx == UInt8(116):      # t
                buf = buf + String("	")
                i += 2
                continue
            elif nx == UInt8(110):    # n
                buf = buf + String("
")
                i += 2
                continue
            elif nx == UInt8(114):    # r
                buf = buf + String("
")
                i += 2
                continue
            elif nx == UInt8(92):     # backslash
                buf = buf + String("\")
                i += 2
                continue
        if ch == UInt8(9):            # tab delimiter
            out.append(buf)
            buf = String("")
        else:
            buf = buf + String(ch)
        i += 1
    out.append(buf)
    return out
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

fn from_mnp_string(s: String):
    # Parse the MNP format written by to_mnp_string()
    var lines = List[String]()
    var start = 0
    var s_len = len(s)
    # split by raw '
'
    var i = 0
    var cur = String("")
    while i < s_len:
        var ch = s.bytes()[i]
        if ch == UInt8(10):  # 

            lines.append(cur)
            cur = String("")
        else:
            cur = cur + String(ch)
        i += 1
    # include last tail if not empty or if input ended with newline we still push empty
    lines.append(cur)

    if len(lines) < 4:
        return DataFrame(List[String](), List[List[String]](), List[String](), String(""))

    if lines[0] != String("MNP"):
        return DataFrame(List[String](), List[List[String]](), List[String](), String(""))

    # line1: nrows	ncols
    var dims = _split_esc_tabs(lines[1])
    if len(dims) < 2:
        return DataFrame(List[String](), List[List[String]](), List[String](), String(""))
    var nrows = Int(0)
    var ncols = Int(0)
    # parse integers in dims[0], dims[1]
    fn _parse_int(ss: String) -> Int:
        var neg = False
        var i2 = 0
        var n2 = len(ss)
        if n2 > 0 and ss.bytes()[0] == UInt8(45):
            neg = True
            i2 = 1
        var acc: Int = 0
        while i2 < n2:
            var ch2 = ss.bytes()[i2]
            if ch2 < UInt8(48) or ch2 > UInt8(57):
                break
            acc = acc * 10 + (Int(ch2) - Int(48))
            i2 += 1
        if neg: acc = -acc
        return acc

    nrows = _parse_int(dims[0])
    ncols = _parse_int(dims[1])

    # line2: index name (escaped)
    var index_name = _unesc(lines[2])

    # line3: column names
    var cols = _split_esc_tabs(lines[3])

    # next lines: row values
    var cols_vals = List[List[String]]()
    var c = 0
    while c < ncols:
        cols_vals.append(List[String]())
        c += 1

    var r0 = 4
    var rcount = 0
    while r0 < len(lines) and rcount < nrows:
        var parts = _split_esc_tabs(lines[r0])
        # fill missing with empty
        var c1 = 0
        while c1 < ncols:
            var val = String("")
            if c1 < len(parts):
                val = parts[c1]
            cols_vals[c1].append(val)
            c1 += 1
        r0 += 1
        rcount += 1

    # Build default index 0..nrows-1
    var index = List[String]()
    var k = 0
    while k < nrows:
        index.append(String(k))
        k += 1

    return DataFrame(cols, cols_vals, index, index_name)
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