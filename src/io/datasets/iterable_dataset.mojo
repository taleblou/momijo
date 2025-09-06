# ============================================================================
# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.io.datasets
# File: src/momijo/io/datasets/iterable_dataset.mojo
# ============================================================================

# -----------------------------------------------------------------------------
# DSIterator: generic iterator wrapper
# -----------------------------------------------------------------------------
struct DSIterator:
    var data: List[Any]
    var pos: Int

    fn __init__(out self, data: List[Any]):
        self.data = data
        self.pos = 0

    fn __iter__(self) -> DSIterator:
        return self

    fn __next__(mut self) -> Any:
        if self.pos >= len(self.data):
            raise StopIteration
        var item = self.data[self.pos]
        self.pos += 1
        return item


# -----------------------------------------------------------------------------
# IterableDataset: base class for datasets that can be iterated
# -----------------------------------------------------------------------------
struct IterableDataset:
    var data: List[Any]

    fn __init__(out self, data: List[Any]):
        self.data = data

    fn __iter__(self) -> DSIterator:
        return DSIterator(self.data)


# -----------------------------------------------------------------------------
# Example subclass of IterableDataset
# -----------------------------------------------------------------------------
struct RangeDataset(IterableDataset):
    var start: Int
    var end: Int

    fn __init__(out self, start: Int, end: Int):
        self.start = start
        self.end = end
        var arr = List[Int]()
        for i in range(start, end):
            arr.append(i)
        super.__init__(arr)


# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------

fn _self_test() -> Bool:
    var ok = True

    # Test RangeDataset
    var ds = RangeDataset(0,5)
    var values = List[Int]()
    for x in ds:
        values.append(x)
    ok = ok and values == [0,1,2,3,4]

    # Test generic IterableDataset
    var ds2 = IterableDataset(["a","b","c"])
    var values2 = List[String]()
    for x in ds2:
        values2.append(x)
    ok = ok and values2 == ["a","b","c"]

    return ok

 