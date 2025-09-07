# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.dataframe
# File: src/momijo/dataframe/io_csv.mojo

from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame
from momijo.dataframe.series_bool import SeriesBool
from momijo.dataframe.series_f64 import SeriesF64
from momijo.dataframe.series_i64 import SeriesI64
from momijo.dataframe.series_str import SeriesStr

fn rstrip_cr(s: String) -> String:
    var n = len(s)
    if n > 0 and s[n - 1] == '\r':
        var out = String("")
        var i = 0
        while i < n - 1:
            out = out + String(s[i])
            i += 1
        return out
    return s

# --------------------------------
# CSV line splitter with quotes
# Supports:
#  - Commas inside quotes: "a,b"
#  - Escaped quotes with "": He said ""Hi""
#  - Trims trailing '\r'
# --------------------------------
fn split_csv_line(line_in: String) -> List[String]:
    var line = rstrip_cr(line_in)
    var out = List[String]()
    var token = String("")
    var in_quotes = False
    var i = 0
    while i < len(line):
        var ch = line[i]
        if in_quotes:
            if ch == '"':
                if (i + 1) < len(line) and line[i + 1] == '"':
                    token = token + String('"')
                    i += 2
                    continue
                else:
                    in_quotes = False
            else:
                token = token + String(ch)
        else:
            if ch == '"':
                in_quotes = True
            elif ch == ',':
                out.append(token)
                token = String("")
            else:
                token = token + String(ch)
        i += 1
    out.append(token)
    return out

# --------------------------------
# Type checks and parsers (ASCII-only)
# --------------------------------
fn is_ascii_digit_code(code: Int) -> Bool:
    # '0'..'9' == 48..57
    return code >= 48 and code <= 57
fn is_bool(s: String) -> Bool:
    # strict: lowercase "true"/"false" (no case-fold to avoid char ops)
    return s == String("true") or s == String("false")
fn is_int(s: String) -> Bool:
    if len(s) == 0:
        return False
    var i = 0
    if s[0] == '-' and len(s) > 1:
        i = 1
    if i >= len(s):
        return False
    while i < len(s):
        var code = ord(s[i])
        if not is_ascii_digit_code(code):
            return False
        i += 1
    return True
fn is_float(s: String) -> Bool:
    if len(s) == 0:
        return False
    var i = 0
    if s[0] == '-' and len(s) > 1:
        i = 1
    if i >= len(s):
        return False
    var dot = 0
    while i < len(s):
        var ch = s[i]
        if ch == '.':
            dot += 1
            if dot > 1:
                return False
        else:
            var code = ord(ch)
            if not is_ascii_digit_code(code):
                return False
        i += 1
    return True
fn parse_i64(s: String) -> Int64:
    var neg = False
    var i = 0
    if s[0] == '-':
        neg = True
        i = 1
    var v: Int64 = 0
    while i < len(s):
        var d: Int64 = Int64(ord(s[i]) - 48)   # '0' => 48
        v = v * 10 + d
        i += 1
    return -v if neg else v
fn parse_f64(s: String) -> Float64:
    var neg = False
    var i = 0
    if s[0] == '-':
        neg = True
        i = 1
    var int_part: Int64 = 0
    while i < len(s):
        var code = ord(s[i])
        if not is_ascii_digit_code(code):
            break
        var d = Int64(code - 48)
        int_part = int_part * 10 + d
        i += 1
    var frac: Float64 = 0.0
    var scale: Float64 = 1.0
    if i < len(s) and s[i] == '.':
        i += 1
        while i < len(s):
            var code2 = ord(s[i])
            if not is_ascii_digit_code(code2):
                break
            var d2 = Float64(code2 - 48)
            scale = scale * 10.0
            frac = frac + d2 / scale
            i += 1
    var v = Float64(int_part) + frac
    if neg:
        v = -v
    return v

# --------------------------------
# Disk IO stubs (kept as no-op)
# --------------------------------
fn read_text_file(path: String) -> String:
    return String("")
fn write_text_file(path: String, content: String) -> Bool:
    return False

# --------------------------------
# Core: parse CSV from a string
# --------------------------------
fn read_csv_from_string(text: String) -> DataFrame:
    if len(text) == 0:
        return DataFrame()

    var lines = List[String]()
    var cur = String("")
    var i = 0
    while i < len(text):
        var ch = text[i]
        if ch == '\n':
            lines.append(cur)
            cur = String("")
        else:
            cur = cur + String(ch)
        i += 1
    if len(cur) > 0:
        lines.append(cur)

    if len(lines) == 0:
        return DataFrame()

    var header = split_csv_line(lines[0])
    var w = len(header)

    var col_strs = List[List[String]]()
    i = 0
    while i < w:
        col_strs.append(List[String]())
        i += 1

    var r = 1
    while r < len(lines):
        if len(lines[r]) == 0:
            r += 1
            continue
        var cells = split_csv_line(lines[r])

        if len(cells) < w:
            var k = len(cells)
            while k < w:
                cells.append(String(""))
                k += 1
        elif len(cells) > w:
            var cut = List[String]()
            var t = 0
            while t < w:
                cut.append(cells[t])
                t += 1
            cells = cut

        i = 0
        while i < w:
            col_strs[i].append(cells[i])
            i += 1
        r += 1

    var names = List[String]()
    var cols = List[Column]()
    i = 0
    while i < w:
        var nm = header[i]
        names.append(nm)

        var all_bool = True
        var all_int = True
        var all_float = True

        var j = 0
        while j < len(col_strs[i]):
            var v = col_strs[i][j]
            if not is_bool(v):   all_bool = False
            if not is_int(v):    all_int = False
            if not is_float(v):  all_float = False
            j += 1

        if all_bool:
            var arr = List[Bool]()
            j = 0
            while j < len(col_strs[i]):
                arr.append(col_strs[i][j] == String("true"))
                j += 1
            var c = Column()
            c.from_bool(SeriesBool(nm, arr))
            cols.append(c)
        elif all_int:
            var arr_i = List[Int64]()
            j = 0
            while j < len(col_strs[i]):
                arr_i.append(parse_i64(col_strs[i][j]))
                j += 1
            var c2 = Column()
            c2.from_i64(SeriesI64(nm, arr_i))
            cols.append(c2)
        elif all_float:
            var arr_f = List[Float64]()
            j = 0
            while j < len(col_strs[i]):
                arr_f.append(parse_f64(col_strs[i][j]))
                j += 1
            var c3 = Column()
            c3.from_f64(SeriesF64(nm, arr_f))
            cols.append(c3)
        else:
            var arr_s = List[String]()
            j = 0
            while j < len(col_strs[i]):
                arr_s.append(col_strs[i][j])
                j += 1
            var c4 = Column()
            c4.from_str(SeriesStr(nm, arr_s))
            cols.append(c4)

        i += 1

    return DataFrame(names, cols)

# Convenience wrapper (kept for API parity)
fn read_csv(path: String) -> DataFrame:
    var text = read_text_file(path)
    return read_csv_from_string(text)

# --------------------------------
# CSV writer (to string + to file)
# --------------------------------
fn escape_cell(s: String) -> String:
    var needs_quote = False
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch == ',' or ch == '"' or ch == '\n' or ch == '\r':
            needs_quote = True
            break
        i += 1
    if not needs_quote:
        return s

    var out = String("\"")
    i = 0
    while i < len(s):
        var ch2 = s[i]
        if ch2 == '"':
            out = out + String("\"\"")
        else:
            out = out + String(ch2)
        i += 1
    out = out + String("\"")
    return out
fn to_csv_string(df: DataFrame) -> String:
    var s = String("")

    # Header
    var i = 0
    while i < len(df.names):
        s = s + escape_cell(df.names[i])
        if i + 1 < len(df.names):
            s = s + String(",")
        i += 1
    s = s + String("\n")

    # Dimensions
    var w = len(df.cols)
    var h = 0
    if w > 0:
        h = df.cols[0].len()

    # Rows
    var r = 0
    while r < h:
        i = 0
        while i < w:
            s = s + escape_cell(df.cols[i].value()_str(r))
            if i + 1 < w:
                s = s + String(",")
            i += 1
        s = s + String("\n")
        r += 1

    return s
fn write_csv(df: DataFrame, path: String) -> Bool:
    var s = to_csv_string(df)
    return write_text_file(path, s)