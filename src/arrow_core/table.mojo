from arrow_core.record_batch import RecordBatch
from arrow_core.types import Schema, Field, DataType

struct Table:
    schema: Schema
    batches: List[RecordBatch]

    fn __init__(inout self, schema: Schema):
        self.schema = schema
        self.batches = List[RecordBatch]()

    fn append_batch(inout self, b: RecordBatch):
        self.batches.append(b)

    fn num_rows(self) -> Int:
        var total = 0
        for b in self.batches:
            total += b.nrows
        return total
