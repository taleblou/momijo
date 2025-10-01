# Project:      Momijo
# Module:       learn.callbacks.logger
# File:         callbacks/logger.mojo
# Path:         src/momijo/learn/callbacks/logger.mojo
#
# Description:  CSV loggers for training/evaluation. Provides a simple single-
#               column CSVLogger(step,message) and a schema-driven
#               CSVTableLogger with arbitrary columns, safe CSV escaping, and
#               sink-agnostic flushing via callbacks.
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
#   - Types: CSVLogger, CSVTableLogger
#   - Key fns (CSVLogger): log(step,message), to_string(), clear(), flush_with(writer)
#   - Key fns (CSVTableLogger): log_row(values), log(step: Int, fields: Dict-like via pairs),
#                               to_string(), clear(), flush_with(writer)
#   - CSV escaping: RFC-4180-like quoting for comma/quote/newline.

from collections.list import List

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

fn _csv_escape(s: String) -> String:
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
            out = out + String("\"\"")
        else:
            out = out + String(ch)
        i = i + 1
    out = out + String("\"")
    return out

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
        fields.push_back(String(step))
        fields.push_back(msg)
        var row = _join_csv(fields)
        self.buffer.push_back(row)

    fn to_string(self) -> String:
        return _lines_join(self.buffer)

    fn clear(mut self):
        self.buffer = List[String]()
        if self.include_header:
            self.buffer.push_back(String("step,message"))

    # Writer must be a callable that accepts a String
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
        # Defensive copy of schema
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

    # Log using values aligned with schema order (len(values) must equal len(columns))
    fn log_row(mut self, values: List[String]):
        var ncols = Int(self.columns.size())
        if Int(values.size()) != ncols:
            # Mismatched row length; ignore or raise. Here we choose to ignore safely.
            return
        var row = _join_csv(values)
        self.buffer.push_back(row)

    # Convenience: log with (name,value) pairs (order is coerced to schema)
    # `pairs` layout: ["name1","value1","name2","value2", ...] with even length.
    fn log_pairs(mut self, pairs: List[String]):
        var ncols = Int(self.columns.size())
        var temp = List[String]()
        temp.reserve(ncols)
        # init with empty
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
            # find column index (linear scan; schema is small; replace with map if needed)
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

    # Convenience helpers for common patterns
    fn log_scalar(mut self, step: Int, name: String, value: String):
        # expects schema like: ["step","name","value"]
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
