# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

from momijo.arrow_core.types import Schema, Field, DataType, UNKNOWN
from momijo.arrow_core.column import Column

struct RecordBatch:
    schema: Schema
    columns: List[Column[Any]]
    nrows: Int

    fn __init__(out self, schema: Schema, columns: List[Column[Any]]):
        self.schema = schema
        self.columns = columns
        self.nrows = 0
        if columns.len() > 0:
            self.nrows = columns[0].len()

    fn select(self, names: List[String]) -> RecordBatch:
        var cols = List[Column[Any]]()
        var fields = List[Field]()
        for c in self.columns:
            if names.contains(c.name):
                cols.append(c)
                fields.append(Field(c.name, c.dtype))
        return RecordBatch(Schema(fields), cols)

    fn slice(self, start: Int, length: Int) -> RecordBatch:
        # Placeholder: return the same batch (wire real slicing later)
        return self
