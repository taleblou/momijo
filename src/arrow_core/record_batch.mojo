# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.arrow_core
# File: momijo/arrow_core/record_batch.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

from momijo.arrow_core.column import Column, StringColumn

struct RecordBatch(Copyable, Movable, Sized):
    var columns: List[String]          # names
    var int_columns: List[Column[Int]]
    var f64_columns: List[Column[Float64]]
    var str_columns: List[StringColumn]

    # ---------- Constructors ----------

    fn __init__(out self):
        self.columns = List[String]()
        self.int_columns = List[Column[Int]]()
        self.f64_columns = List[Column[Float64]]()
        self.str_columns = List[StringColumn]()

    # Add integer column
    fn add_int_column(mut self, col: Column[Int]):
        self.columns.append(col.name)
        self.int_columns.append(col)

    # Add float column
    fn add_f64_column(mut self, col: Column[Float64]):
        self.columns.append(col.name)
        self.f64_columns.append(col)

    # Add string column
    fn add_str_column(mut self, col: StringColumn):
        self.columns.append(col.name)
        self.str_columns.append(col)

    # ---------- Properties ----------

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

    # ---------- Access ----------

    fn get_int_column(self, name: String) -> Column[Int]:
        var i = 0
        while i < len(self.int_columns):
            if self.int_columns[i].name == name:
                return self.int_columns[i]
            i += 1
        var dummy: Column[Int]
        dummy.__init__("dummy", Array[Int]())
        return dummy

    fn get_f64_column(self, name: String) -> Column[Float64]:
        var i = 0
        while i < len(self.f64_columns):
            if self.f64_columns[i].name == name:
                return self.f64_columns[i]
            i += 1
        var dummy: Column[Float64]
        dummy.__init__("dummy", Array[Float64]())
        return dummy

    fn get_str_column(self, name: String) -> StringColumn:
        var i = 0
        while i < len(self.str_columns):
            if self.str_columns[i].name == name:
                return self.str_columns[i]
            i += 1
        var dummy: StringColumn
        dummy.__init__("dummy", ByteStringArray())
        return dummy

    fn get_row_as_strings(self, row: Int) -> List[String]:
        var out = List[String]()
        for c in self.int_columns:
            out.append(String(c.get_or(row, 0)))
        for c in self.f64_columns:
            out.append(String(c.get_or(row, 0.0)))
        for c in self.str_columns:
            out.append(c.get_or(row, ""))
        return out

    # ---------- Conversion ----------

    fn to_table_strings(self) -> List[List[String]]:
        var rows = List[List[String]]()
        let n = self.n_rows()
        var i = 0
        while i < n:
            rows.append(self.get_row_as_strings(i))
            i += 1
        return rows

