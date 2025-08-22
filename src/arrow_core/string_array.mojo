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
# File: momijo/arrow_core/string_array.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 
 

struct StringArray(Copyable, Movable, Sized):
    var inner: ByteStringArray

    # ---------- Constructors ----------

    fn __init__(out self):
        self.inner = ByteStringArray()

    fn from_strings(out self, strings: List[String]):
        var arr: ByteStringArray
        arr.from_strings(strings)
        self.inner = arr

    fn from_optional_strings(out self, strings: List[Optional[String]]):
        var arr: ByteStringArray
        arr.from_optional_strings(strings)
        self.inner = arr

    # ---------- Properties ----------

    fn __len__(self) -> Int:
        return self.inner.len()

    fn len(self) -> Int:
        return self.inner.len()

    # ---------- Access ----------

    fn is_valid(self, i: Int) -> Bool:
        return self.inner.is_valid(i)

    fn is_null(self, i: Int) -> Bool:
        return self.inner.is_null(i)

    fn get(self, i: Int) -> String:
        return self.inner.get(i)

    fn get_or(self, i: Int, default: String) -> String:
        return self.inner.get_or(i, default)

    # ---------- Mutation ----------

    fn push(mut self, s: String, valid: Bool = True):
        self.inner.push(s, valid)

    fn push_null(mut self):
        self.inner.push_null()

    fn clear(mut self):
        self.inner.clear()

    # ---------- Conversion ----------

    fn to_strings(self) -> List[String]:
        return self.inner.to_strings()

    fn to_optional_strings(self) -> List[Optional[String]]:
        return self.inner.to_optional_strings()

    # ---------- Utility ----------

    fn slice(self, start: Int, count: Int) -> StringArray:
        var out: StringArray
        out.inner = self.inner.slice(start, count)
        return out

