# Momijo Arrow Core
# This file is part of the Momijo project. See the LICENSE file at the repository root.


from momijo.arrow_core.types import DataType, UNKNOWN
from momijo.arrow_core.array import Array

struct Column[T: Copyable & Movable](Copyable, Movable, Sized):
    var name: String
    var dtype: DataType
    var data: Array[T]

    fn __init__(out self, name: String, data: Array[T], dtype: DataType = UNKNOWN()):
        self.name = name
        self.data = data
        self.dtype = dtype

    fn __len__(self) -> Int:
        return len(self.data)

    fn len(self) -> Int:
        return len(self.data)

    fn push(mut self, v: T, valid: Bool = True):
        self.data.push(v, valid)

    fn get(self, i: Int) -> T:
        return self.data.get(i)
