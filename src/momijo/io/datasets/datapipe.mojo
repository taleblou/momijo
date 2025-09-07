# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.io.datasets
# File: src/momijo/io/datasets/datapipe.mojo

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
# DataPipe: functional-style dataset transformations
# -----------------------------------------------------------------------------
struct DataPipe:
    var source: IterableDataset
fn __init__(out self, source: IterableDataset) -> None:
        self.source = source

    # Apply function to each element
fn map(self, fn_map: fn(Any) -> Any) -> DataPipe:
        var mapped = List[Any]()
        for x in self.source:
            mapped.append(fn_map(x))
        return DataPipe(IterableDataset(mapped))

    # Filter elements by predicate
fn filter(self, fn_filter: fn(Any) -> Bool) -> DataPipe:
        var filtered = List[Any]()
        for x in self.source:
            if fn_filter(x):
                filtered.append(x)
        return DataPipe(IterableDataset(filtered))

    # Group elements into batches
fn batch(self, batch_size: Int) -> DataPipe:
        var batched = List[Any]()
        var buf = List[Any]()
        for x in self.source:
            buf.append(x)
            if len(buf) == batch_size:
                batched.append(buf)
                buf = List[Any]()
        if len(buf) > 0:
            batched.append(buf)
        return DataPipe(IterableDataset(batched))

    # Return iterator
fn __iter__(self) -> DSIterator:
        return self.source.__iter__()
fn __copyinit__(out self, other: Self) -> None:
        self.source = other.source
fn __moveinit__(out self, deinit other: Self) -> None:
        self.source = other.source
# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------
fn _self_test() -> Bool:
    var ok = True
    var dataset = IterableDataset([1,2,3,4,5,6])
    var pipe = DataPipe(dataset)

    # Test map
    var mapped = pipe.map(fn(x: Any) -> Any: return x * 2)
    var values = List[Int]()
    for x in mapped:
        values.append(x)
    ok = ok and values == [2,4,6,8,10,12]

    # Test filter
    var filtered = pipe.filter(fn(x: Any) -> Bool: return x % 2 == 0)
    var values2 = List[Int]()
    for x in filtered:
        values2.append(x)
    ok = ok and values2 == [2,4,6]

    # Test batch
    var batched = pipe.batch(2)
    var values3 = List[List[Int]]()
    for b in batched:
        values3.append(b)
    ok = ok and values3 == [[1,2],[3,4],[5,6]]

    return ok
fn main() -> None:
    if _self_test():
        print("DataPipe module self-test: OK")
    else:
        print("DataPipe module self-test: FAIL")