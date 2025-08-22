# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.arrow_core
# File: momijo/arrow_core/column.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

from momijo.arrow_core.array import Array
from momijo.arrow_core.byte_string_array import ByteStringArray
from momijo.arrow_core.bitmap import Bitmap, bitmap_get_valid

struct Column[T: Copyable & Movable](Copyable, Movable, Sized):
    var name: String
    var values: Array[T]

    # ---------- Constructors ----------

    fn __init__(out self, name: String = "", values: Array[T] = Array[T]()) :
        self.name = name
        self.values = values

    fn from_list(out self, name: String, vals: List[T]):
        self.name = name
        var arr: Array[T]
        arr.from_values(vals, True)
        self.values = arr

    # ---------- Properties ----------

    fn __len__(self) -> Int:
        return self.values.len()

    fn len(self) -> Int:
        return self.values.len()

    fn is_valid(self, i: Int) -> Bool:
        return self.values.is_valid(i)

    fn null_count(self) -> Int:
        return self.values.null_count()

    fn get(self, i: Int) -> T:
        return self.values.get(i)

    fn get_or(self, i: Int, default: T) -> T:
        return self.values.get_or(i, default)

    # ---------- Mutation ----------

    fn set(mut self, i: Int, v: T, valid: Bool = True) -> Bool:
        return self.values.set(i, v, valid)

    fn push(mut self, v: T, valid: Bool = True):
        self.values.push(v, valid)

    fn clear(mut self):
        self.values.clear()

    # ---------- Conversion ----------

    fn to_list(self) -> List[T]:
        return self.values.to_list()

    fn compact_values(self) -> List[T]:
        return self.values.compact_values()

# Specialized column for strings

struct StringColumn(Copyable, Movable, Sized):
    var name: String
    var values: ByteStringArray

    fn __init__(out self, name: String = "", values: ByteStringArray = ByteStringArray()):
        self.name = name
        self.values = values

    fn __len__(self) -> Int:
        return self.values.len()

    fn len(self) -> Int:
        return self.values.len()

    fn is_valid(self, i: Int) -> Bool:
        return self.values.is_valid(i)

    fn null_count(self) -> Int:
        return self.values.validity.count_invalid()

    fn get(self, i: Int) -> String:
        return self.values.get(i)

    fn get_or(self, i: Int, default: String) -> String:
        return self.values.get_or(i, default)

    fn push(mut self, s: String, valid: Bool = True):
        self.values.push(s, valid)

    fn push_null(mut self):
        self.values.push_null()

    fn clear(mut self):
        self.values.clear()

    fn to_strings(self) -> List[String]:
        return self.values.to_strings()

    fn to_optional_strings(self) -> List[Optional[String]]:
        return self.values.to_optional_strings()
