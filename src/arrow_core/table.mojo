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
# File: momijo/arrow_core/table.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

from momijo.arrow_core.record_batch import RecordBatch

struct Table(Copyable, Movable, Sized):
    var batches: List[RecordBatch]

    # ---------- Constructors ----------

    fn __init__(out self):
        self.batches = List[RecordBatch]()

    fn from_batches(out self, bs: List[RecordBatch]):
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

    fn get_batch(self, i: Int) -> RecordBatch:
        if i < 0 or i >= len(self.batches):
            var dummy: RecordBatch
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
            # Search all column types
            let ints = b.int_columns
            for c in ints:
                if c.name == name:
                    var i = 0
                    while i < c.len():
                        out.append(String(c.get_or(i, 0)))
                        i += 1
            let f64s = b.f64_columns
            for c in f64s:
                if c.name == name:
                    var i = 0
                    while i < c.len():
                        out.append(String(c.get_or(i, 0.0)))
                        i += 1
            let strs = b.str_columns
            for c in strs:
                if c.name == name:
                    var i = 0
                    while i < c.len():
                        out.append(c.get_or(i, ""))
                        i += 1
        return out

    # ---------- Mutation ----------

    fn add_batch(mut self, b: RecordBatch):
        self.batches.append(b)

    fn clear(mut self):
        self.batches = List[RecordBatch]()

    # ---------- Conversion ----------

    fn to_table_strings(self) -> List[List[String]]:
        var rows = List[List[String]]()
        for b in self.batches:
            let batch_rows = b.to_table_strings()
            for r in batch_rows:
                rows.append(r)
        return rows

