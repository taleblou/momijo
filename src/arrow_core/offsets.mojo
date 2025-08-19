# MIT License
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of arrow_core. See LICENSE at repository root.

struct Offsets:
    var data: List[Int]

# Constructor: __init__(out self)
    fn __init__(out self):
        self.data = List[Int]()
        self.data.append(0)

# Function append_offset(mut self, o: Int)
    fn append_offset(mut self, o: Int):
        self.data.append(o)

# Function add_length(mut self, length: Int)
    fn add_length(mut self, length: Int):
        self.data.append(self.last() + length)

# Function len(self) -> Int
    fn len(self) -> Int:
        return len(self.data)

# Function last(self) -> Int
    fn last(self) -> Int:
        return self.data[len(self.data) - 1]

# Function at(self, i: Int) -> Int
    fn at(self, i: Int) -> Int:
        return self.data[i]

# Function is_valid(self) -> Bool
    fn is_valid(self) -> Bool:
        if len(self.data) == 0:
            return False
        if self.data[0] != 0:
            return False
        var i = 1
        while i < len(self.data):
            if self.data[i] < self.data[i - 1]:
                return False
            i += 1
        return True