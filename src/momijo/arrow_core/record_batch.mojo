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
# Project: momijo.arrow_core
# File: src/momijo/arrow_core/record_batch.mojo

from momijo.arrow_core.array import Array
from momijo.arrow_core.byte_string_array import ByteStringArray
from momijo.arrow_core.column import Column, StringColumn

fn __module_name__() -> String:
    return String("momijo/arrow_core/record_batch.mojo")
fn __self_test__() -> Bool:
    var rb = RecordBatch()
    return rb.n_columns() == 0 and rb.n_rows() == 0 and len(rb) == 0

struct RecordBatch(Copyable, Movable, Sized):
    var columns: List[String]
    var int_columns: List[Column[Int]]
    var f64_columns: List[Column[Float64]]
    var str_columns: List[StringColumn]

    var dummy_int: Column[Int]
    var dummy_f64: Column[Float64]
    var dummy_str: StringColumn
fn __init__(out self) -> None:
        self.columns = List[String]()
        self.int_columns = List[Column[Int]]()
        self.f64_columns = List[Column[Float64]]()
        self.str_columns = List[StringColumn]()

        var di = Column[Int]()
        _ = di.__init__(String("dummy_int"), Array[Int]())
        self.dummy_int = di

        var df = Column[Float64]()
        _ = df.__init__(String("dummy_f64"), Array[Float64]())
        self.dummy_f64 = df

        var ds = StringColumn()
        _ = ds.__init__(String("dummy_str"), ByteStringArray())
        self.dummy_str = ds
fn __len__(self) -> Int:
        return self.n_rows()
fn add_int_column(mut self, col: Column[Int]) -> None:
        self.columns.append(col.name)
        self.int_columns.append(col)
fn add_f64_column(mut self, col: Column[Float64]) -> None:
        self.columns.append(col.name)
        self.f64_columns.append(col)
fn add_str_column(mut self, col: StringColumn) -> None:
        self.columns.append(col.name)
        self.str_columns.append(col)
fn n_columns(self) -> Int:
        return len(self.columns)
fn n_rows(self) -> Int:
        if len(self.int_columns) > 0:
            return self.int_columns[0].len()
        if len(self.f64_columns) > 0:
            return self.f64_columns[0].len()
        if len(self.str_columns) > 0:
            return self.str_columns[0].len()
        return 0
fn column_names(self) -> List[String]:
        return self.columns
fn get_int_column(self, name: String) -> Column[Int]:
        var i = 0
        while i < len(self.int_columns):
            if self.int_columns[i].name == name:
                return self.int_columns[i]
            i += 1
        return self.dummy_int
fn get_f64_column(self, name: String) -> Column[Float64]:
        var i = 0
        while i < len(self.f64_columns):
            if self.f64_columns[i].name == name:
                return self.f64_columns[i]
            i += 1
        return self.dummy_f64
fn get_str_column(self, name: String) -> StringColumn:
        var i = 0
        while i < len(self.str_columns):
            if self.str_columns[i].name == name:
                return self.str_columns[i]
            i += 1
        return self.dummy_str
fn get_row_as_strings(self, row: Int) -> List[String]:
        var out = List[String]()
        var i = 0
        while i < len(self.int_columns):
            out.append(String(self.int_columns[i].get_or(row, 0)))
            i += 1
        i = 0
        while i < len(self.f64_columns):
            out.append(String(self.f64_columns[i].get_or(row, 0.0)))
            i += 1
        i = 0
        while i < len(self.str_columns):
            out.append(self.str_columns[i].get_or(row, String("")))
            i += 1
        return out
fn to_table_strings(self) -> List[List[String]]:
        var rows = List[List[String]]()
        var n = self.n_rows()
        var i = 0
        while i < n:
            rows.append(self.get_row_as_strings(i))
            i += 1
        return rows