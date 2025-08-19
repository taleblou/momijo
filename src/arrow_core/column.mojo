# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

from momijo.arrow_core.array import Array
from momijo.arrow_core.types import DataType, UNKNOWN

# Struct Column: auto-generated docs. Update as needed.
struct Column[T]:
    name: String
    dtype: DataType
    data: Array[T]

# Constructor: __init__(out self, name: String, dtype: DataType = UNKNOWN, data: Array[T] = Array[T](0))
    fn __init__(out self, name: String, dtype: DataType = UNKNOWN, data: Array[T] = Array[T](0)):
        self.name = name
        self.dtype = dtype
        self.data = data

# Function len(self) -> Int
    fn len(self) -> Int:
        return self.data.len()