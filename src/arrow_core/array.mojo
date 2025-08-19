# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

from momijo.arrow_core.bitmap import Bitmap
# Struct Array: auto-generated docs. Update as needed.
struct Array[T]:
    values: List[T]
    validity: Bitmap

# Constructor: __init__(out self, n: Int = 0, all_valid: Bool = True)
    fn __init__(out self, n: Int = 0, all_valid: Bool = True):
        self.values = List[T]()
        self.validity = Bitmap(n, all_valid)
        var i = 0
        while i < n:
            # default-init (may be zero)
            self.values.append(T())
            i += 1

# Function len(self) -> Int
    fn len(self) -> Int:
        return self.values.len()

# Function push(mut self, v: T, valid: Bool = True)
    fn push(mut self, v: T, valid: Bool = True):
        self.values.append(v)
        self.validity.ensure_size(self.len(), True)
        self.validity.set_valid(self.len() - 1, valid)

# Function get(self, i: Int) -> T
    fn get(self, i: Int) -> T:
        return self.values.get(i)