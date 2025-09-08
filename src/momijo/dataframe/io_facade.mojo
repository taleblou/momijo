# Project:      Momijo
# Module:       src.momijo.dataframe.io_facade
# File:         io_facade.mojo
# Path:         src/momijo/dataframe/io_facade.mojo
#
# Description:  src.momijo.dataframe.io_facade — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Key functions: to_pickle, read_csv, to_csv, read_json, to_json
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.poly_column import get_string
from momijo.core.error import module
from momijo.core.traits import one
from momijo.dataframe.api import col_str, df_make
from momijo.dataframe.column import Column
from momijo.dataframe.frame import DataFrame, width
from momijo.dataframe.helpers import read, read_bytes, t
from momijo.dataframe.io_bytes import bytes_to_string, str_to_bytes
from momijo.dataframe.io_csv import split_csv_line
from momijo.dataframe.io_csv_utils import split_csv_line
from momijo.dataframe.io_files import read_bytes, write_bytes
from momijo.dataframe.series_bool import append
from momijo.enum.enum import parse
from momijo.ir.passes.cse import find
from momijo.tensor.indexing import slice
from momijo.utils.timer import start
from momijo.vision.schedule.schedule import append_str, cols, end2, find, height, keys, line, lines, pos, seen, val
from pathlib import Path
from pathlib.path import Path

    _ = path
    return DataFrame()
fn to_pickle(df: DataFrame, path: String) -> Bool
    return _to_pickle(df, path)

# CSV: read from file via bytes
fn read_csv(path: String) -> DataFrame
    var raw = read_bytes(path)
    if len(raw) == 0:
        return df_make(List[String](), List[Column]())
    var txt = bytes_to_string(raw)
    # split lines
    var lines = List[String]()
    var start = 0
    var i = 0
    while i < len(txt):
        if txt[i] == '\n':
            lines.append(txt.slice(start, i))
            start = i + 1
        i += 1
    if start < len(txt):
        lines.append(txt.slice(start, len(txt)))
    if len(lines) == 0:
        return df_make(List[String](), List[Column]())
    var names = split_csv_line(lines[0])
    var cols = List[Column]()
    var c = 0
    while c < len(names):
        cols.append(col_str(names[c], List[String]()))
        c += 1
    var r = 1
    while r < len(lines):
        var fields = split_csv_line(lines[r])
        var j = 0
        while j < len(fields) and j < len(cols):
            cols[j].append_str(fields[j])
            j += 1
        r += 1
    return df_make(names, cols)

# CSV: write to file via bytes
fn to_csv(df: DataFrame, path: String) -> Bool
    var s = String("")
    var i = 0
    while i < df.width():
        s = s + df.names[i]
        if i + 1 < df.width(): s = s + String(",")
        i += 1
    s = s + String("\n")
    var r = 0
    while r < df.height():
        var c = 0
        while c < df.width():
            s = s + df.cols[c].get_string(r)
            if c + 1 < df.width(): s = s + String(",")
            c += 1
        s = s + String("\n")
        r += 1
    return write_bytes(path, str_to_bytes(s))

# Minimal JSON lines reader: each line like {"a":"x","b":1} (no nested, no escapes)
fn read_json(path: String) -> DataFrame
    var raw = read_bytes(path)
    if len(raw) == 0:
        return DataFrame()
    var txt = bytes_to_string(raw)
    # split lines
    var lines = List[String]()
    var start = 0
    var i = 0
    while i < len(txt):
        if txt[i] == '\n':
            lines.append(txt.slice(start, i))
            start = i + 1
        i += 1
    if start < len(txt): lines.append(txt.slice(start, len(txt)))
    # parse keys union
    var keys = List[String]()
    var r = 0
    while r < len(lines):
        var ln = lines[r]
        var p = 0
        while True:
            var kpos = ln.find(String("\"name\":\""), p)
            if kpos < 0: break
            kpos += 8
            var kend = ln.find(String("\""), kpos)
            if kend < 0: break
            var key = ln.slice(kpos, kend)
            var seen = False; var t = 0
            while t < len(keys):
                if keys[t] == key: seen = True
                t += 1
            if not seen: keys.append(key)
            p = kend + 1
        r += 1
    if len(keys) == 0:
        return DataFrame()
    var cols = List[Column]()
    var j = 0
    while j < len(keys):
        cols.append(col_str(keys[j], List[String]()))
        j += 1
    r = 0
    while r < len(lines):
        var ln2 = lines[r]
        var j2 = 0
        while j2 < len(keys):
            var patt = String("\"") + keys[j2] + String("\":")
            var pos = ln2.find(patt, 0)
            var val = String("")
            if pos >= 0:
                pos = pos + len(patt)
                # skip spaces and possible quotes
                while pos < len(ln2) and (ln2[pos] == ' '): pos += 1
                if pos < len(ln2) and ln2[pos] == '\"':
                    pos += 1
                    var end = ln2.find(String("\""), pos)
                    if end >= 0: val = ln2.slice(pos, end)
                else:
                    # number until comma or }
                    var end2 = pos
                    while end2 < len(ln2) and ln2[end2] != ',' and ln2[end2] != '}': end2 += 1
                    val = ln2.slice(pos, end2)
            cols[j2].append_str(val)
            j2 += 1
        r += 1
    return df_make(keys, cols)

# Minimal JSON writer: one line per row with string values
fn to_json(df: DataFrame, path: String) -> Bool
    var s = String("")
    var r = 0
    while r < df.height():
        var line = String("{")
        var c = 0
        while c < df.width():
            line = line + String("\"") + df.names[c] + String("\":") + String("\"") + df.cols[c].get_string(r) + String("\"")
            if c + 1 < df.width(): line = line + String(",")
            c += 1
        line = line + String("}")
        s = s + line + String("\n")
        r += 1
    return write_bytes(path, str_to_bytes(s))