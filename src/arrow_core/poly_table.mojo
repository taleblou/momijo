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
# File: momijo/arrow_core/poly_table.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

from momijo.arrow_core.poly_record_batch import PolyRecordBatch
from momijo.arrow_core.poly_column import PolyColumn

struct PolyTable(Copyable, Movable, Sized):
    var batches: List[PolyRecordBatch]

    # ---------- Constructors ----------

    fn __init__(out self):
        self.batches = List[PolyRecordBatch]()

    fn from_batches(out self, bs: List[PolyRecordBatch]):
        self.batches = bs

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

    # ---------- Access ----------

    fn get_batch(self, i: Int) -> PolyRecordBatch:
        if i < 0 or i >= len(self.batches):
            var dummy: PolyRecordBatch
            dummy.__init__()
            return dummy
        return self.batches[i]

    fn get_row_as_strings(self, row: Int) -> List[String]:
        var offset: Int = 0
        for b in self.batches:
            let n = b.n_rows()
            if row < offset + n:
                return b.get_row_as_strings(row - offset)
            offset += n
        return List[String]()

    fn get_column(self, name: String) -> List[String]:
        var out = List[String]()
        for b in self.batches:
            let col = b.get_column_by_name(name)
            let n = b.n_rows()
            var i = 0
            while i < n:
                match col.tag:
                    case .INT: out.append(String(col.get_int(i)))
                    case .FLOAT64: out.append(String(col.get_f64(i)))
                    case .STRING: out.append(col.get_string(i))
                    case .UNKNOWN: out.append("")
                i += 1
        return out

    # ---------- Mutation ----------

    fn add_batch(mut self, b: PolyRecordBatch):
        self.batches.append(b)

    fn clear(mut self):
        self.batches = List[PolyRecordBatch]()

    # ---------- Conversion ----------

    fn to_table_strings(self) -> List[List[String]]:
        var rows = List[List[String]]()
        for b in self.batches:
            let batch_rows = b.to_table_strings()
            for r in batch_rows:
                rows.append(r)
        return rows

