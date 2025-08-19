
# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.

from momijo.arrow_core.types import Schema, Field
from momijo.arrow_core.poly_column import PolyColumn

struct PolyRecordBatch:
    schema: Schema
    columns: List[PolyColumn]
    nrows: Int

    fn __init__(out self, schema: Schema):
        self.schema = schema
        self.columns = List[PolyColumn]()
        self.nrows = 0

    fn add_column(mut self, c: PolyColumn) raises:
        if len(self.columns) == 0:
            self.nrows = c.len()
        elif c.len() != self.nrows:
            raise Exception("column length mismatch")
        self.columns.append(c)

    fn num_rows(self) -> Int:
        return self.nrows

    fn num_columns(self) -> Int:
        return len(self.columns)

    fn column(self, idx: Int) -> PolyColumn:
        return self.columns[idx]
