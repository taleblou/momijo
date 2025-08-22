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
# File: momijo/arrow_core/byte_string_array.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

from momijo.arrow_core.bitmap import Bitmap, bitmap_set_valid, bitmap_get_valid
from momijo.arrow_core.array import Array
from momijo.arrow_core.offsets import Offsets

struct ByteStringArray(Copyable, Movable, Sized):
    var data: List[UInt8]
    var offsets: Offsets
    var validity: Bitmap

    # ---------- Constructors ----------

    fn __init__(out self):
        self.data = List[UInt8]()
        self.offsets = Offsets()
        self.validity = Bitmap(0, True)

    # From a list of strings (all valid)
    fn from_strings(out self, strings: List[String]):
        self.data = List[UInt8]()
        self.offsets = Offsets()
        self.offsets.push(0)
        for s in strings:
            let bytes = s.bytes()
            for b in bytes:
                self.data.append(b)
            self.offsets.push(len(self.data))
        self.validity = Bitmap(len(strings), True)

    # From list of optional strings (null represented by None)
    fn from_optional_strings(out self, strings: List[Optional[String]]):
        self.data = List[UInt8]()
        self.offsets = Offsets()
        self.offsets.push(0)
        self.validity = Bitmap(len(strings), True)
        var i = 0
        for opt in strings:
            if opt.has_value():
                let bytes = opt.value().bytes()
                for b in bytes:
                    self.data.append(b)
                self.offsets.push(len(self.data))
                bitmap_set_valid(self.validity, i, True)
            else:
                self.offsets.push(len(self.data))
                bitmap_set_valid(self.validity, i, False)
            i += 1

    # ---------- Properties ----------

    fn __len__(self) -> Int:
        return self.offsets.len() - 1

    fn len(self) -> Int:
        return self.offsets.len() - 1

    # ---------- Element access ----------

    fn is_valid(self, i: Int) -> Bool:
        return bitmap_get_valid(self.validity, i)

    fn is_null(self, i: Int) -> Bool:
        return not self.is_valid(i)

    fn get(self, i: Int) -> String:
        if i < 0 or i >= self.len():
            return ""
        if not self.is_valid(i):
            return ""
        let start = self.offsets.get(i)
        let end = self.offsets.get(i+1)
        var chars = List[UInt8]()
        var j = start
        while j < end:
            chars.append(self.data[j])
            j += 1
        return String(chars)

    fn get_or(self, i: Int, default: String) -> String:
        if i < 0 or i >= self.len():
            return default
        if not self.is_valid(i):
            return default
        return self.get(i)

    # ---------- Mutation ----------

    fn push(mut self, s: String, valid: Bool = True):
        let bytes = s.bytes()
        for b in bytes:
            self.data.append(b)
        self.offsets.push(len(self.data))
        let n = self.len()
        self.validity = self.validity.resize(n, True)
        bitmap_set_valid(self.validity, n-1, valid)

    fn push_null(mut self):
        self.offsets.push(len(self.data))
        let n = self.len()
        self.validity = self.validity.resize(n, True)
        bitmap_set_valid(self.validity, n-1, False)

    # ---------- Conversion ----------

    fn to_strings(self) -> List[String]:
        var out = List[String]()
        let n = self.len()
        var i = 0
        while i < n:
            if self.is_valid(i):
                out.append(self.get(i))
            else:
                out.append("")
            i += 1
        return out

    fn to_optional_strings(self) -> List[Optional[String]]:
        var out = List[Optional[String]]()
        let n = self.len()
        var i = 0
        while i < n:
            if self.is_valid(i):
                out.append(Optional[String](self.get(i)))
            else:
                out.append(Optional[String]())
            i += 1
        return out

    # ---------- Utility ----------

    fn clear(mut self):
        self.data = List[UInt8]()
        self.offsets = Offsets()
        self.validity = Bitmap(0, True)

    fn slice(self, start: Int, count: Int) -> ByteStringArray:
        var out: ByteStringArray
        let n = self.len()
        if start < 0 or start >= n or count <= 0:
            out = ByteStringArray()
            return out

        let end = (start + count) if (start + count) <= n else n

        out.data = List[UInt8]()
        out.offsets = Offsets()
        out.offsets.push(0)
        out.validity = Bitmap(end - start, True)

        var i = start
        var out_pos = 0
        while i < end:
            if self.is_valid(i):
                let s = self.get(i)
                let bytes = s.bytes()
                for b in bytes:
                    out.data.append(b)
                out.offsets.push(len(out.data))
                bitmap_set_valid(out.validity, out_pos, True)
            else:
                out.offsets.push(len(out.data))
                bitmap_set_valid(out.validity, out_pos, False)
            i += 1
            out_pos += 1

        return out
