# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

# array of strings using bytes + offsets + validity.
from momijo.arrow_core import Offsets, Bitmap

struct StringArray:
    data: List[UInt8]
    offsets: Offsets
    validity: Bitmap

    fn __init__(inout self):
        self.data = List[UInt8]()
        self.offsets = Offsets()
        self.validity = Bitmap(0, True)

# add one string. Inputs: s (string), valid (valid flag)
fn push(inout self, s: String, valid: Bool = True):
    var i = 0
    while i < s.len():
        self.data.append(UInt8(s.byte_at(i)))
        i += 1
    self.offsets.add_length(s.len().to_int())
    self.validity.ensure_size(self.len(), True)
    self.validity.set_valid(self.len() - 1, valid)

# number of strings. Output: item count
fn len(self) -> Int:
    return self.offsets.len() - 1

# get one string at index. Inputs: i (index). Output: the item at index
fn get(self, i: Int) -> String:
    var start = self.offsets.at(i)
    var end = self.offsets.at(i + 1)
    var n = end - start
    var out = String()
    var k = 0
    while k < n:
        var b = self.data[start + k]
        out = out + String.from_byte(b)
        k += 1
    return out
