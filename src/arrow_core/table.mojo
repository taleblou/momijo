# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.


from momijo.arrow_core.types import Schema
from momijo.arrow_core.record_batch import RecordBatch

struct Table(Copyable, Movable, Sized):
    var schema: Schema
    var batches: List[RecordBatch]

    fn __init__(out self, schema: Schema):
        self.schema = schema
        self.batches = List[RecordBatch]()

    fn __len__(self) -> Int:
        return len(self.batches)

    fn append_batch(mut self, b: RecordBatch):
        self.batches.append(b)

    fn ncols(self) -> Int:
        return 0 if len(self.batches) == 0 else self.batches[0].num_columns()

    fn num_rows(self) -> Int:
        var total = 0
        var i = 0
        while i < len(self.batches):
            total += self.batches[i].__len__()
            i += 1
        return total
