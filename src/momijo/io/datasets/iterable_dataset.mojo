# Project:      Momijo
# Module:       src.momijo.io.datasets.iterable_dataset
# File:         iterable_dataset.mojo
# Path:         src/momijo/io/datasets/iterable_dataset.mojo
#
# Description:  Filesystem/IO helpers with Path-centric APIs and safe resource
#               management (binary/text modes and encoding clarity).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Structs: DSIterator, IterableDataset, RangeDataset
#   - Key functions: __init__, __iter__, __next__, __copyinit__, __moveinit__, __init__, __iter__, __copyinit__ ...


struct DSIterator:
    var data: List[Any]
    var pos: Int
fn __init__(out self, data: List[Any]) -> None:
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
fn __copyinit__(out self, other: Self) -> None:
        self.data = other.data
        self.pos = other.pos
fn __moveinit__(out self, deinit other: Self) -> None:
        self.data = other.data
        self.pos = other.pos
# -----------------------------------------------------------------------------
# IterableDataset: base class for datasets that can be iterated
# -----------------------------------------------------------------------------
struct IterableDataset:
    var data: List[Any]
fn __init__(out self, data: List[Any]) -> None:
        self.data = data
fn __iter__(self) -> DSIterator:
        return DSIterator(self.data)
fn __copyinit__(out self, other: Self) -> None:
        self.data = other.data
fn __moveinit__(out self, deinit other: Self) -> None:
        self.data = other.data
# -----------------------------------------------------------------------------
# Example subclass of IterableDataset
# -----------------------------------------------------------------------------
struct RangeDataset(IterableDataset):
    var start: Int
    var end: Int
fn __init__(out self, start: Int, end: Int) -> None:
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