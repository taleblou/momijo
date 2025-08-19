
# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.

from momijo.arrow_core.types import Schema
from momijo.arrow_core.poly_record_batch import PolyRecordBatch

struct PolyTable:
    schema: Schema
    batches: List[PolyRecordBatch]

    fn __init__(out self, schema: Schema):
        self.schema = schema
        self.batches = List[PolyRecordBatch]()

    fn append_batch(mut self, b: PolyRecordBatch):
        self.batches.append(b)

    fn num_rows(self) -> Int:
        var total = 0
        var i = 0
        while i < len(self.batches):
            total += self.batches[i].num_rows()
            i += 1
        return total

    fn ncols(self) -> Int:
        return 0 if len(self.batches) == 0 else self.batches[0].num_columns()
