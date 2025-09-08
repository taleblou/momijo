# Project:      Momijo
# Module:       src.momijo.arrow_core.table
# File:         table.mojo
# Path:         src/momijo/arrow_core/table.mojo
#
# Description:  Arrow-inspired columnar primitives (offsets, buffers, list/struct arrays)
#               supporting zero-copy slicing and predictable memory semantics.
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
#   - Structs: Table
#   - Key functions: _join_strings, __init__, from_batches, set_batches, __copyinit__, __len__, n_batches, n_rows ...
#   - Static methods present.


from momijo.arrow_core.record_batch import RecordBatch

fn _join_strings(xs: List[String], sep: String) -> String:
    var out = String("")
    var i = 0
    while i < len(xs):
        out = out + xs[i]
        if i + 1 < len(xs):
            out = out + sep
        i += 1
    return out

struct Table(ExplicitlyCopyable, Movable, Sized):
    var batches: List[RecordBatch]

    # ---------- Constructors ----------
fn __init__(out self) -> None:
        self.batches = List[RecordBatch]()

    @staticmethod
fn from_batches(bs: List[RecordBatch]) -> Self:
        var t = Self()
        t.batches = bs
        return t
fn set_batches(mut self, bs: List[RecordBatch]) -> None:
        self.batches = bs

    # ---------- Traits ----------
fn __copyinit__(out self, other: Self) -> None:
        self.batches = other.batches
fn __len__(self) -> Int:
        return len(self.batches)

    # ---------- Properties ----------
fn n_batches(self) -> Int:
        return len(self.batches)
fn n_rows(self) -> Int:
        var total: Int = 0
        for b in self.batches:
            total += b.n_rows()
        return total
fn n_columns(self) -> Int:
        if len(self.batches) == 0:
            return 0
        return self.batches[0].n_columns()
fn column_names(self) -> List[String]:
        if len(self.batches) == 0:
            return List[String]()
        return self.batches[0].column_names()
fn column_names_str(self) -> String:
        return _join_strings(self.column_names(), String(", "))

    # ---------- Access ----------
    # Precondition: 0 <= i < n_batches()
fn get_batch(self, i: Int) -> RecordBatch:
        return self.batches[i]
fn get_row_as_strings(self, row: Int) -> List[String]:
        var offset: Int = 0
        for b in self.batches:
            var n = b.n_rows()
            if row < offset + n:
                return b.get_row_as_strings(row - offset)
            offset += n
        return List[String]()
fn get_row_as_string(self, row: Int, sep: String) -> String:
        return _join_strings(self.get_row_as_strings(row), sep)
fn get_column(self, name: String) -> List[String]:
        var out = List[String]()
        for b in self.batches:
            var ints = b.int_columns
            for c in ints:
                if c.name == name:
                    var i = 0
                    while i < c.len():
                        out.append(String(c.get_or(i, 0)))
                        i += 1
            var f64s = b.f64_columns
            for c in f64s:
                if c.name == name:
                    var i = 0
                    while i < c.len():
                        out.append(String(c.get_or(i, 0.0)))
                        i += 1
            var strs = b.str_columns
            for c in strs:
                if c.name == name:
                    var i = 0
                    while i < c.len():
                        out.append(c.get_or(i, String("")))
                        i += 1
        return out
fn get_column_str(self, name: String, sep: String) -> String:
        return _join_strings(self.get_column(name), sep)

    # ---------- Mutation ----------
fn add_batch(mut self, b: RecordBatch) -> None:
        self.batches.append(b)
fn clear(mut self) -> None:
        self.batches = List[RecordBatch]()

    # ---------- Conversion ----------
fn to_table_strings(self) -> List[List[String]]:
        var rows = List[List[String]]()
        for b in self.batches:
            var batch_rows = b.to_table_strings()
            for r in batch_rows:
                rows.append(r)
        return rows