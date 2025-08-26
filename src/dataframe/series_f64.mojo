# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Minimal numeric Series (Float64) with a validity bitmap

from collections import BitSet

struct SeriesF64(Copyable, Movable):
    var name: String
    var values: List[Float64]
    var validity: BitSet

    fn __init__(out self, name: String, values: List[Float64]):
        self.name = name
        self.values = values
        self.validity = BitSet(nbits=len(values), all_valid=True)

    fn len(self) -> Int:
        return len(self.values)

    fn is_null(self, i: Int) -> Bool:
        return not self.validity.test(i)

    fn sum(self) -> Float64:
        var s: Float64 = 0.0
        for i in range(0, len(self.values)):
            if self.validity.test(i):
                s += self.values[i]
        return s

    fn mean(self) -> Float64:
        var s: Float64 = 0.0
        var c: Int = 0
        for i in range(0, len(self.values)):
            if self.validity.test(i):
                s += self.values[i]
                c += 1
        return s / Float64(c)

    fn filter(self, mask: BitSet) -> SeriesF64:
        assert(mask.nbits == self.validity.nbits, "Mask length mismatch")
        var out_vals = List[Float64]()
        for i in range(0, len(self.values)):
            if mask.test(i) and self.validity.test(i):
                out_vals.append(self.values[i])
        return SeriesF64(self.name, out_vals)

    fn to_string(self) -> String:
        return "SeriesF64(" + self.name + ", len=" + String(len(self.values)) + ")"


fn get(self, i: Int) -> Float64:
    return self.values[i]

fn gather(self, indices: List[Int]) -> SeriesF64:
    var out = List[Float64]()
    for i in indices:
        if self.validity.is_set(i):
            out.append(self.values[i])
    return SeriesF64(self.name, out)
