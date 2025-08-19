# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

from momijo.arrow_core.array import Array
from momijo.arrow_core.types import DataType, UNKNOWN

struct Column[T]:
    name: String
    dtype: DataType
    data: Array[T]

    fn __init__(inout self, name: String, dtype: DataType = UNKNOWN, data: Array[T] = Array[T](0)):
        self.name = name
        self.dtype = dtype
        self.data = data

    fn len(self) -> Int:
        return self.data.len()
