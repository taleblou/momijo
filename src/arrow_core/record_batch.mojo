# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.


from momijo.arrow_core.types import Schema
from momijo.arrow_core.column import Column

struct RecordBatch(Copyable, Movable, Sized):
    var schema: Schema
    var columns: List[Column[Int]]
    var nrows: Int

    fn __init__(out self, schema: Schema):
        self.schema = schema
        self.columns = List[Column[Int]]()
        self.nrows = 0

    fn __len__(self) -> Int:
        return self.nrows

    fn add_column(mut self, c: Column[Int]):
        if len(self.columns) == 0:
            self.nrows = c.len()
            self.columns.append(c)
            return
        # If lengths mismatch, clamp to min to avoid exceptions.
        if c.len() != self.nrows:
            self.nrows = c.len() if c.len() < self.nrows else self.nrows
        self.columns.append(c)

    fn num_columns(self) -> Int:
        return len(self.columns)

    fn column(self, idx: Int) -> Column[Int]:
        return self.columns[idx]
