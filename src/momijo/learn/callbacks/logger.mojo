# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.callbacks.logger
# File:         src/momijo/learn/callbacks/logger.mojo
#
# Description:
#   CSV loggers for training/evaluation:
#     - CSVLogger(step, message): simple two-column logger.
#     - CSVTableLogger(schema...): schema-driven multi-column logger.
#   Features:
#     - Safe CSV escaping (commas, quotes, CR/LF) per RFC-4180 style.
#     - Sink-agnostic flush via a user-provided callable `writer(String)`.
#     - Clear/reset APIs and in-memory buffering before flush.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

@always_inline
fn _csv_escape(s: String) -> String:
    # Quote field if it contains comma, double-quote, or newline/CR.
    var needs_quote = False
    var i = 0
    while i < s.__len__():
        var ch = s[i]
        if ch == ',' or ch == '"' or ch == '\n' or ch == '\r':
            needs_quote = True
            break
        i = i + 1

    if not needs_quote:
        return s

    var out = String("")
    out = out + String("\"")
    i = 0
    while i < s.__len__():
        var ch = s[i]
        if ch == '"':
            out = out + String("\"\"")       # escape embedded quotes
        else:
            out = out + String(ch)
        i = i + 1
    out = out + String("\"")
    return out

@always_inline
fn _join_csv(fields: List[String]) -> String:
    var out = String("")
    var i = 0
    var n = Int(fields.size())
    while i < n:
        out = out + _csv_escape(fields[i])
        if i + 1 < n:
            out = out + String(",")
        i = i + 1
    return out

@always_inline
fn _lines_join(lines: List[String]) -> String:
    var out = String("")
    var i = 0
    var n = Int(lines.size())
    while i < n:
        out = out + lines[i]
        if i + 1 < n:
            out = out + String("\n")
        i = i + 1
    return out


# -----------------------------------------------------------------------------
# Simple CSV logger: columns = [step, message]
# -----------------------------------------------------------------------------

struct CSVLogger:
    var path: String
    var include_header: Bool
    var buffer: List[String]

    fn __init__(out self, path: String, include_header: Bool = True):
        self.path = path
        self.include_header = include_header
        self.buffer = List[String]()
        if include_header:
            self.buffer.push_back(String("step,message"))

    fn log(mut self, step: Int, msg: String):
        var fields = List[String]()
        fields.push_back(String(step))   # Int -> String
        fields.push_back(msg)
        var row = _join_csv(fields)
        self.buffer.push_back(row)

    fn to_string(self) -> String:
        return _lines_join(self.buffer)

    fn clear(mut self):
        self.buffer = List[String]()
        if self.include_header:
            self.buffer.push_back(String("step,message"))

    # Writer: a callable that accepts a single String (e.g., a file writer or stdout writer)
    fn flush_with(self, writer):
        writer(self.to_string())


# -----------------------------------------------------------------------------
# Schema-driven CSV logger: arbitrary columns with strict ordering
# -----------------------------------------------------------------------------

struct CSVTableLogger:
    var path: String
    var columns: List[String]          # ordered schema
    var include_header: Bool
    var buffer: List[String]

    fn __init__(out self, path: String, columns: List[String], include_header: Bool = True):
        self.path = path
        self.include_header = include_header
        self.columns = List[String]()
        self.buffer = List[String]()

        var i = 0
        var n = Int(columns.size())
        while i < n:
            self.columns.push_back(columns[i])
            i = i + 1

        if include_header:
            var header = _join_csv(self.columns)
            self.buffer.push_back(header)

    # Log using values aligned with schema order (len(values) must equal len(columns)).
    fn log_row(mut self, values: List[String]):
        var ncols = Int(self.columns.size())
        if Int(values.size()) != ncols:
            # Mismatched row length; ignore safely.
            return
        var row = _join_csv(values)
        self.buffer.push_back(row)

    # Log with name/value pairs. pairs: ["name1","value1","name2","value2", ...]
    fn log_pairs(mut self, pairs: List[String]):
        var ncols = Int(self.columns.size())
        var temp = List[String]()
        temp.reserve(ncols)

        var i = 0
        while i < ncols:
            temp.push_back(String(""))
            i = i + 1

        var m = Int(pairs.size())
        if m % 2 != 0:
            # malformed pairs; ignore
            return

        var j = 0
        while j < m:
            var name = pairs[j]
            var value = pairs[j + 1]
            # find column index by linear scan (schemas are typically small)
            var col_idx = -1
            var k = 0
            while k < ncols:
                if self.columns[k] == name:
                    col_idx = k
                    break
                k = k + 1
            if col_idx >= 0:
                temp[col_idx] = value
            j = j + 2

        self.log_row(temp)

    # Convenience for common schema ["step","name","value"].
    fn log_scalar(mut self, step: Int, name: String, value: String):
        var row = List[String]()
        row.push_back(String(step))
        row.push_back(name)
        row.push_back(value)
        self.log_row(row)

    fn to_string(self) -> String:
        return _lines_join(self.buffer)

    fn clear(mut self):
        self.buffer = List[String]()
        if self.include_header:
            var header = _join_csv(self.columns)
            self.buffer.push_back(header)

    fn flush_with(self, writer):
        writer(self.to_string())
