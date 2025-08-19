# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

# Offsets buffer for variable-size arrays (e.g., strings).

struct Offsets:
    data: List[Int]

    fn __init__(inout self):
        self.data = List[Int]()
        self.data.append(0)

    fn append_offset(inout self, o: Int):
        self.data.append(o)

    fn len(self) -> Int:
        return self.data.len()


# get the last offset value. Output: last offset value
fn last(self) -> Int:
    return self.data[self.data.len() - 1]


# append a new offset by adding a length. Inputs: length (how many)
fn add_length(inout self, length: Int):
    var new_off = self.last() + length
    if new_off < self.last():
        return
    self.data.append(new_off)


# get the offset at an index. Inputs: i (index). Output: offset value
fn at(self, i: Int) -> Int:
    return self.data[i]


# check offsets never go down and start at 0. Output: true if sizes are OK
fn is_valid(self) -> Bool:
    if self.data.len() == 0:
        return False
    if self.data[0] != 0:
        return False
    var i = 1
    while i < self.data.len():
        if self.data[i] < self.data[i - 1]:
            return False
        i += 1
    return True
