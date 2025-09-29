# Project:      Momijo
# Module:       dataframe.io_csv
# File:         io_csv.mojo
#
# Description:  Io Csv module for Momijo DataFrame.
#               Implements CSV read/write helpers for Momijo DataFrame.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# License:      MIT
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand

from collections.list import List
from pathlib.path import Path

from momijo.dataframe.frame import DataFrame

# ---------------- Files (text mode) ----------------
fn write_text(path: String, text: String) -> Bool:
    """
    Write text to `path`. Returns True on success, False on any failure.
    """
    try:
        var f = open(Path(path), "w")
        f.write(text)
        f.close()
        return True
    except:
        return False

fn read_text(path: String) -> String:
    """
    Read whole text file from `path`. On failure returns empty String.
    """
    try:
        var f = open(Path(path), "r")
        var s = f.read()
        f.close()
        return s
    except:
        return String("")

# ---------------- CSV helpers ----------------
fn csv_escape(cell: String) -> String:
    """
    Escape a single CSV cell using RFC4180-ish rules:
    - If comma, quote, or newline present, surround with quotes and double internal quotes.
    """
    var need_quotes: Bool = False
    for ch in cell:
        if ch == ',' or ch == '"' or ch == '\n' or ch == '\r':
            need_quotes = True
            break
    if not need_quotes:
        return cell

    var out = String("")
    out += '"'
    for ch in cell:
        if ch == '"':
            out += "\"\""          # double inner quotes
        else:
            out += String(ch)
    out += '"'
    return out

fn row_to_csv_line(row_vals: List[String]) -> String:
    """
    Convert a list of strings into a single CSV line (no trailing newline).
    """
    var s = String("")
    var first: Bool = True
    for v in row_vals:
        if first:
            first = False
        else:
            s += ","
        s += csv_escape(v)
    return s

fn to_csv_string(df: DataFrame) -> String:
    """
    Convert a DataFrame to CSV string (header + rows).
    """
    var s = String("")
    # Header
    var c = 0
    while c < df.ncols():
        s += csv_escape(df.col_names[c])
        if c + 1 < df.ncols():
            s += ","
        c += 1
    s += "\n"

    # Rows
    var r = 0
    while r < df.nrows():
        var c2 = 0
        while c2 < df.ncols():
            s += csv_escape(df.cols[c2].get_string(r))
            if c2 + 1 < df.ncols():
                s += ","
            c2 += 1
        s += "\n"
        r += 1
    return s

# Parse one CSV line (RFC4180-ish). Returns list of fields.
fn parse_csv_line(line: String) -> List[String]:
    var out = List[String]()
    var cur = String("")
    var in_quotes: Bool = False
    var i: Int = 0
    var n: Int = len(line)
    while i < n:
        var ch = line[i]
        if in_quotes:
            if ch == '"':
                # If next char is also quote => escaped quote
                if i + 1 < n and line[i+1] == '"':
                    cur += '"'
                    i += 1
                else:
                    # end of quoted field (but may continue if char after quote is comma)
                    in_quotes = False
            else:
                cur += String(ch)
        else:
            if ch == '"':
                # start quoted field (or stray quote)
                in_quotes = True
            elif ch == ',':
                out.append(cur)
                cur = String("")
            elif ch == '\r':
                # ignore CR
                pass
            else:
                cur += String(ch)
        i += 1
    # append last field
    out.append(cur)
    return out

# ---------------- Public API ----------------
fn write_csv(df: DataFrame, path: String) -> Bool:
    """
    Write DataFrame to CSV file at `path`. Returns True on success.
    """
    var text = to_csv_string(df)
    return write_text(path, text)

fn read_csv(path: String) -> DataFrame:
    """
    Read CSV file from `path` into DataFrame.
    """
    var text = read_text(path)
    return read_csv_from_string(text)

fn read_csv_from_string(text: String) -> DataFrame:
    """
    Parse CSV content from a string and return a DataFrame.
    Handles header row and normalizes row lengths to header width.
    Empty text => empty DataFrame with zero columns.
    """
    # Split into lines (preserve empty lines between data rows as skipped)
    var lines = List[String]()
    var start: Int = 0
    var i: Int = 0
    var n: Int = len(text)
    while i < n:
        if text[i] == '\n':
            # include substring excluding the trailing newline
            lines.append(text[start:i])
            start = i + 1
        i += 1
    if start < n:
        lines.append(text[start:n])

    # Empty => empty frame
    if len(lines) == 0:
        var empty_cols = List[String]()
        var empty_data = List[List[String]]()
        var empty_idx = List[String]()
        return DataFrame(empty_cols, empty_data, empty_idx, String(""))

    # Header row
    var headers = parse_csv_line(lines[0])
    var w = len(headers)
    var cols = List[List[String]]()
    var j = 0
    while j < w:
        cols.append(List[String]())
        j += 1

    # Parse rows
    var r: Int = 1
    while r < len(lines):
        # skip completely empty lines
        if len(lines[r]) == 0:
            r += 1
            continue
        var fields = parse_csv_line(lines[r])

        # normalize field count: pad or truncate to header width `w`
        if len(fields) < w:
            var k = len(fields)
            while k < w:
                fields.append(String(""))
                k += 1
        elif len(fields) > w:
            var tmp = List[String]()
            var t = 0
            while t < w:
                tmp.append(fields[t])
                t += 1
            fields = tmp

        # append column-major
        var c3 = 0
        while c3 < w:
            cols[c3].append(fields[c3])
            c3 += 1
        r += 1

    # build default index "0..n-1"
    var index = List[String]()
    var nrows = 0
    if w > 0:
        nrows = len(cols[0])
    var rr = 0
    while rr < nrows:
        # convert integer row index to String
        index.append(String(rr))
        rr += 1

    return DataFrame(headers, cols, index, String(""))
