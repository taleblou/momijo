# Project:      Momijo
# Module:       dataframe.io_json_min
# File:         io_json_min.mojo
# Path:         dataframe/io_json_min.mojo
#
# Description:  dataframe.io_json_min — Minimal JSON I/O for Momijo DataFrame.
#               Provides compact JSON (object with name/columns/index/data) and JSON Lines
#               serializers/deserializers; includes safe string escaping/unescaping,
#               small text file wrappers, and tolerant parsing with width normalization.
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
#   - Key functions: write_text, read_text, _hex, json_escape, json_unescape, to_json_string,
#                    from_json_string, _read_qstr, _read_str_array, _read_2d_str_array,
#                    to_json_lines_string, from_json_lines_string, write_json, read_json




from collections.list import List
from pathlib.path import Path
from momijo.dataframe.frame import DataFrame


fn write_text(path: String, text: String) -> Bool:
    try:
        var f = open(Path(path), "w")
        f.write(text)
        f.close()
        return True
    except:
        return False

fn read_text(path: String) -> String:
    try:
        var f = open(Path(path), "r")
        var s = f.read()
        f.close()
        return s
    except:
        return String("")

fn _hex(n: Int) -> String:
    var s = String("0123456789ABCDEF")
    var out = String("")
    out += s[(n >> 4) & 15]
    out += s[n & 15]
    return out

fn json_escape(s: String) -> String:
    var out = String("")
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch == "\""[0]:
            out += String("\\\"")
        elif ch == "\\"[0]:
            out += String("\\\\")
        elif ch == "\n"[0]:
            out += String("\\n")
        elif ch == "\r"[0]:
            out += String("\\r")
        elif ch == "\t"[0]:
            out += String("\\t")
        else:
            out += String(ch)
        i += 1
    return out

fn json_unescape(s: String) -> String:
    var out = String("")
    var i = 0
    while i < len(s):
        var ch = s[i]
        if ch != "\\"[0]:
            out += String(ch)
            i += 1
            continue
        i += 1
        if i >= len(s):
            break
        var e = s[i]
        if e == "n"[0]:
            out += String("\n")
        elif e == "r"[0]:
            out += String("\r")
        elif e == "t"[0]:
            out += String("\t")
        elif e == "\\"[0]:
            out += String("\\")
        elif e == "\""[0]:
            out += String("\"")
        elif e == "u"[0]:
            var j = 0
            while j < 4 and i + 1 < len(s):
                i += 1
                j += 1
        else:
            out += String(e)
        i += 1
    return out

fn to_json_string(df: DataFrame) -> String:
    var s = String("{\"name\":\"")
    s += json_escape(df.index_name)
    s += String("\",\"columns\":[")
    var c = 0
    while c < df.ncols():
        if c > 0:
            s += String(",")
        s += String("\"")
        s += json_escape(df.col_names[c])
        s += String("\"")
        c += 1
    s += String("],\"index\":[")
    var r = 0
    var nidx = len(df.index_vals)
    while r < df.nrows():
        if r > 0:
            s += String(",")
        s += String("\"")
        var iv = String("")
        if r < nidx:
            iv = df.index_vals[r]
        else:
            iv = String(r)
        s += json_escape(iv)
        s += String("\"")
        r += 1
    s += String("],\"data\":[")
    var rr = 0
    while rr < df.nrows():
        if rr > 0:
            s += String(",")
        s += String("[")
        var cc = 0
        while cc < df.ncols():
            if cc > 0:
                s += String(",")
            s += String("\"")
            s += json_escape(df.cols[cc].value_str(rr))
            s += String("\"")
            cc += 1
        s += String("]")
        rr += 1
    s += String("]}")
    return s

fn from_json_string(s: String) -> DataFrame:
    var name = String("")
    var cols = List[String]()
    var index = List[String]()
    var rows = List[List[String]]()

    fn _read_qstr(s: String, start: Int) -> (String, Int):
        var i = start + 1
        var acc = String("")
        while i < len(s):
            var ch = s[i]
            if ch == "\\"[0]:
                if i + 1 >= len(s):
                    break
                var e = s[i+1]
                acc += String("\\")
                acc += String(e)
                i += 2
                continue
            if ch == "\""[0]:
                return (json_unescape(acc), i + 1)
            acc += String(ch)
            i += 1
        return (json_unescape(acc), i)

    fn _read_str_array(s: String, start: Int) -> (List[String], Int):
        var out = List[String]()
        var i = start
        while i < len(s) and s[i] != "["[0]:
            i += 1
        if i >= len(s):
            return (out, i)
        i += 1
        while i < len(s):
            while i < len(s) and (s[i] == " "[0] or s[i] == ","[0] or s[i] == "\n"[0] or s[i] == "\r"[0] or s[i] == "\t"[0]):
                i += 1
            if i >= len(s):
                break
            if s[i] == "]"[0]:
                i += 1
                break
            if s[i] == "\""[0]:
                var res = _read_qstr(s, i)
                var val = res[0]
                var j = res[1]
                out.append(val)
                i = j
                continue
            i += 1
        return (out, i)

    fn _read_2d_str_array(s: String, start: Int) -> (List[List[String]], Int):
        var out = List[List[String]]()
        var i = start
        while i < len(s) and s[i] != "["[0]:
            i += 1
        if i >= len(s):
            return (out, i)
        i += 1
        while i < len(s):
            while i < len(s) and (s[i] == " "[0] or s[i] == ","[0] or s[i] == "\n"[0] or s[i] == "\r"[0] or s[i] == "\t"[0]):
                i += 1
            if i >= len(s):
                break
            if s[i] == "]"[0]:
                i += 1
                break
            var res = _read_str_array(s, i)
            var row = res[0]
            var j = res[1]
            out.append(row)
            i = j
        return (out, i)

    var i = 0
    while i < len(s):
        while i < len(s) and s[i] != "\""[0]:
            i += 1
        if i >= len(s):
            break
        var kres = _read_qstr(s, i)
        var key = kres[0]
        var j = kres[1]
        i = j
        while i < len(s) and s[i] != ":"[0]:
            i += 1
        if i < len(s):
            i += 1
        if key == String("name"):
            while i < len(s) and s[i] != "\""[0]:
                i += 1
            if i < len(s):
                var nres = _read_qstr(s, i)
                name = nres[0]
                i = nres[1]
        elif key == String("columns"):
            var a1 = _read_str_array(s, i)
            cols = a1[0]
            i = a1[1]
        elif key == String("index"):
            var a2 = _read_str_array(s, i)
            index = a2[0]
            i = a2[1]
        elif key == String("data"):
            var a3 = _read_2d_str_array(s, i)
            rows = a3[0]
            i = a3[1]
        else:
            i += 1

    var w = len(cols)
    var cols_vals = List[List[String]]()
    var c0 = 0
    while c0 < w:
        cols_vals.append(List[String]())
        c0 += 1
    var r0 = 0
    while r0 < len(rows):
        var cc0 = 0
        while cc0 < w and cc0 < len(rows[r0]):
            cols_vals[cc0].append(rows[r0][cc0])
            cc0 += 1
        while cc0 < w:
            cols_vals[cc0].append(String(""))
            cc0 += 1
        r0 += 1

    return DataFrame(cols, cols_vals, index, name)

fn to_json_lines_string(df: DataFrame) -> String:
    var s = String("")
    var r = 0
    while r < df.nrows():
        s += String("{")
        var c = 0
        while c < df.ncols():
            if c > 0:
                s += String(",")
            s += String("\"")
            s += json_escape(df.col_names[c])
            s += String("\":\"")
            s += json_escape(df.cols[c].value_str(r))
            s += String("\"")
            c += 1
        s += String("}\n")
        r += 1
    return s

fn from_json_lines_string(s: String) -> DataFrame:
    var keys = List[String]()
    var rows = List[List[String]]()
    var parts = List[String]()
    var p = 0
    var line = String("")
    while p < len(s):
        var ch = s[p]
        if ch == "\n"[0]:
            parts.append(line)
            line = String("")
        else:
            line += String(ch)
        p += 1
    if len(line) > 0:
        parts.append(line)
    var li = 0
    while li < len(parts):
        var row_k = List[String]()
        var row_v = List[String]()
        var t = parts[li]
        var j = 0
        while j < len(t):
            while j < len(t) and t[j] != "\""[0]:
                j += 1
            if j >= len(t):
                break
            var kres = _read_qstr(t, j)
            var k = kres[0]
            j = kres[1]
            while j < len(t) and t[j] != ":"[0]:
                j += 1
            if j < len(t):
                j += 1
            while j < len(t) and t[j] != "\""[0]:
                j += 1
            if j >= len(t):
                break
            var vres = _read_qstr(t, j)
            var v = vres[0]
            j = vres[1]
            row_k.append(k)
            row_v.append(v)
        var mk = 0
        while mk < len(row_k):
            var seen = False
            var kk = 0
            while kk < len(keys):
                if keys[kk] == row_k[mk]:
                    seen = True
                    break
                kk += 1
            if not seen:
                keys.append(row_k[mk])
            mk += 1
        rows.append(row_v)
        li += 1
    var w = len(keys)
    var cols_vals = List[List[String]]()
    var k2 = 0
    while k2 < w:
        cols_vals.append(List[String]())
        k2 += 1
    var r2 = 0
    while r2 < len(parts):
        var t2 = parts[r2]
        var kv_keys = List[String]()
        var kv_vals = List[String]()
        var j2 = 0
        while j2 < len(t2):
            while j2 < len(t2) and t2[j2] != "\""[0]:
                j2 += 1
            if j2 >= len(t2):
                break
            var k3res = _read_qstr(t2, j2)
            var k3 = k3res[0]
            j2 = k3res[1]
            while j2 < len(t2) and t2[j2] != ":"[0]:
                j2 += 1
            if j2 < len(t2):
                j2 += 1
            while j2 < len(t2) and t2[j2] != "\""[0]:
                j2 += 1
            if j2 >= len(t2):
                break
            var v3res = _read_qstr(t2, j2)
            var v3 = v3res[0]
            j2 = v3res[1]
            kv_keys.append(k3)
            kv_vals.append(v3)
        var kk2 = 0
        while kk2 < w:
            var found = False
            var pos = 0
            while pos < len(kv_keys):
                if kv_keys[pos] == keys[kk2]:
                    cols_vals[kk2].append(kv_vals[pos])
                    found = True
                    break
                pos += 1
            if not found:
                cols_vals[kk2].append(String(""))
            kk2 += 1
        r2 += 1
    var index = List[String]()
    var nrows = 0
    if w > 0:
        nrows = len(cols_vals[0])
    var rr3 = 0
    while rr3 < nrows:
        index.append(String(rr3))
        rr3 += 1
    return DataFrame(keys, cols_vals, index, String(""))

fn write_json(df: DataFrame, path: String) -> Bool:
    var text = to_json_string(df)
    return write_text(path, text)

fn read_json(path: String) -> DataFrame:
    var text = read_text(path)
    if len(text) == 0:
        var empty_cols = List[String]()
        var empty_data = List[List[String]]()
        var empty_idx = List[String]()
        return DataFrame(empty_cols, empty_data, empty_idx, String(""))
    return from_json_string(text)
