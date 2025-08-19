
# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.

from momijo.arrow_core.value import Value
from momijo.arrow_core.types import DataType, UNKNOWN

struct PolyColumn:
    name: String
    dtype: DataType
    data: List[Value]

    fn __init__(out self, name: String, dtype: DataType = UNKNOWN()):
        self.name = name
        self.dtype = dtype
        self.data = List[Value]()

    fn len(self) -> Int:
        return len(self.data)

    fn push(mut self, v: Value):
        self.data.append(v)

    fn get(self, i: Int) -> Value:
        return self.data[i]

    fn infer_dtype(self) -> DataType:
        var i = 0
        while i < len(self.data):
            if not self.data[i].is_null:
                return self.data[i].dtype()
            i += 1
        return self.dtype
