# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

from momijo.arrow_core.bitmap import Bitmap
struct Array[T]:
    values: List[T]
    validity: Bitmap

    fn __init__(inout self, n: Int = 0, all_valid: Bool = True):
        self.values = List[T]()
        self.validity = Bitmap(n, all_valid)
        var i = 0
        while i < n:
            # default-init (may be zero)
            self.values.append(T())
            i += 1

    fn len(self) -> Int:
        return self.values.len()

    fn push(inout self, v: T, valid: Bool = True):
        self.values.append(v)
        self.validity.ensure_size(self.len(), True)
        self.validity.set_valid(self.len() - 1, valid)

    fn get(self, i: Int) -> T:
        return self.values.get(i)
