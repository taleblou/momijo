# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

from momijo.arrow_core.record_batch import RecordBatch
from momijo.arrow_core.types import Schema, Field, DataType

struct Table:
    schema: Schema
    batches: List[RecordBatch]

    fn __init__(out self, schema: Schema):
        self.schema = schema
        self.batches = List[RecordBatch]()

    fn append_batch(mut self, b: RecordBatch):
        self.batches.append(b)

    fn num_rows(self) -> Int:
        var total = 0
        for b in self.batches:
            total += b.nrows
        return total
