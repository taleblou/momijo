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
# File: momijo/arrow_core/poly_record_batch.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

from momijo.arrow_core.poly_column import PolyColumn, PolyColumnTag

struct PolyRecordBatch(Copyable, Movable, Sized):
    var columns: List[PolyColumn]

    # ---------- Constructors ----------

    fn __init__(out self):
        self.columns = List[PolyColumn]()

    fn from_columns(out self, cols: List[PolyColumn]):
        self.columns = cols

    # ---------- Properties ----------

    fn n_columns(self) -> Int:
        return len(self.columns)

    fn n_rows(self) -> Int:
        if len(self.columns) == 0:
            return 0
        return self.columns[0].len()

    fn column_names(self) -> List[String]:
        var names = List[String]()
        for c in self.columns:
            names.append(c.name)
        return names

    # ---------- Access by column ----------

    fn get_column(self, i: Int) -> PolyColumn:
        if i < 0 or i >= len(self.columns):
            var dummy: PolyColumn
            dummy.__init__("invalid")
            return dummy
        return self.columns[i]

    fn get_column_by_name(self, name: String) -> PolyColumn:
        for c in self.columns:
            if c.name == name:
                return c
        var dummy: PolyColumn
        dummy.__init__("invalid")
        return dummy

    # ---------- Access by row ----------

    fn get_row_as_strings(self, row: Int) -> List[String]:
        var out = List[String]()
        let ncols = len(self.columns)
        var j = 0
        while j < ncols:
            let col = self.columns[j]
            match col.tag:
                case .INT: out.append(String(col.get_int(row)))
                case .FLOAT64: out.append(String(col.get_f64(row)))
                case .STRING: out.append(col.get_string(row))
                case .UNKNOWN: out.append("")
            j += 1
        return out

    # ---------- Mutation ----------

    fn add_column(mut self, col: PolyColumn):
        self.columns.append(col)

    fn clear(mut self):
        self.columns = List[PolyColumn]()

    # ---------- Conversion ----------

    fn to_table_strings(self) -> List[List[String]]:
        var rows = List[List[String]]()
        let n = self.n_rows()
        var i = 0
        while i < n:
            rows.append(self.get_row_as_strings(i))
            i += 1
        return rows

