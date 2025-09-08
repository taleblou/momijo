# Project:      Momijo
# Module:       src.momijo.arrow_core.poly_record_batch
# File:         poly_record_batch.mojo
# Path:         src/momijo/arrow_core/poly_record_batch.mojo
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
#   - Structs: PolyRecordBatch
#   - Key functions: __module_name__, __self_test__, __init__, from_columns, __len__, n_columns, n_rows, column_names ...


from momijo.arrow_core.poly_column import (

    PolyColumn,
    POLYTAG_UNKNOWN,
    POLYTAG_INT,
    POLYTAG_F64,
    POLYTAG_STR,
)
fn __module_name__() -> String:
    return String("momijo/arrow_core/poly_record_batch.mojo")
fn __self_test__() -> Bool:
    var rb = PolyRecordBatch()
    return rb.n_columns() == 0 and rb.n_rows() == 0 and len(rb) == 0

struct PolyRecordBatch(Copyable, Movable, Sized):
    var columns: List[PolyColumn]

    # ---------- Constructors ----------
fn __init__(out self) -> None:
        self.columns = List[PolyColumn]()
fn from_columns(mut self, cols: List[PolyColumn]) -> None:
        self.columns = cols

    # ---------- Sized ----------
fn __len__(self) -> Int:
        # define size as number of rows
        return self.n_rows()

    # ---------- Properties ----------
fn n_columns(self) -> Int:
        return len(self.columns)
fn n_rows(self) -> Int:
        if len(self.columns) == 0:
            return 0
        return self.columns[0].len()
fn column_names(self) -> List[String]:
        var names = List[String]()
        var i = 0
        while i < len(self.columns):
            names.append(self.columns[i].name)
            i += 1
        return names

    # ---------- Access by column ----------
fn get_column(self, i: Int) -> PolyColumn:
        if i < 0 or i >= len(self.columns):
            # construct an initialized dummy
            var dummy = PolyColumn(String("invalid"))
            return dummy
        return self.columns[i]
fn get_column_by_name(self, name: String) -> PolyColumn:
        var i = 0
        while i < len(self.columns):
            if self.columns[i].name == name:
                return self.columns[i]
            i += 1
        # construct an initialized dummy
        var dummy = PolyColumn(String("invalid"))
        return dummy

    # ---------- Access by row ----------
fn get_row_as_strings(self, row: Int) -> List[String]:
        var out = List[String]()
        var j = 0
        while j < len(self.columns):
            var col = self.columns[j]
            if col.tag == POLYTAG_INT():
                out.append(String(col.get_int(row)))
            elif col.tag == POLYTAG_F64():
                out.append(String(col.get_f64(row)))
            elif col.tag == POLYTAG_STR():
                out.append(col.get_string(row))
            else:
                out.append(String(""))
            j += 1
        return out

    # ---------- Mutation ----------
fn add_column(mut self, col: PolyColumn) -> None:
        self.columns.append(col)
fn clear(mut self) -> None:
        self.columns = List[PolyColumn]()

    # ---------- Conversion ----------
fn to_table_strings(self) -> List[List[String]]:
        var rows = List[List[String]]()
        var n = self.n_rows()
        var i = 0
        while i < n:
            rows.append(self.get_row_as_strings(i))
            i += 1
        return rows