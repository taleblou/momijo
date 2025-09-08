# Project:      Momijo
# Module:       src.momijo.io.datasets.dataloader
# File:         dataloader.mojo
# Path:         src/momijo/io/datasets/dataloader.mojo
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
#   - Structs: SequentialSampler, RandomSampler, DataLoader, DummyDataset
#   - Key functions: __init__, __iter__, __copyinit__, __moveinit__, __init__, __iter__, __copyinit__, __moveinit__ ...
#   - Uses generic functions/types with explicit trait bounds.


import random

struct SequentialSampler:
    var size: Int
fn __init__(out self, size: Int) -> None:
        self.size = size
fn __iter__(self) -> List[Int]:
        var idxs = List[Int]()
        for i in range(self.size):
            idxs.append(i)
        return idxs
fn __copyinit__(out self, other: Self) -> None:
        self.size = other.size
fn __moveinit__(out self, deinit other: Self) -> None:
        self.size = other.size
# -----------------------------------------------------------------------------
# RandomSampler: yields indices in random order
# -----------------------------------------------------------------------------
struct RandomSampler:
    var size: Int
fn __init__(out self, size: Int) -> None:
        self.size = size
fn __iter__(self) -> List[Int]:
        var idxs = List[Int]()
        for i in range(self.size):
            idxs.append(i)
        random.shuffle(idxs)
        return idxs
fn __copyinit__(out self, other: Self) -> None:
        self.size = other.size
fn __moveinit__(out self, deinit other: Self) -> None:
        self.size = other.size
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
struct DataLoader:
    var dataset: Any
    var batch_size: Int
    var shuffle: Bool
    var drop_last: Bool
    var collate_fn: Optional[fn(List[Any]) -> Any]
    var sampler: Any

    var _indices: List[Int]
    var _pos: Int
fn __init__(out self,
                dataset: Any,
                batch_size: Int = 1,
                shuffle: Bool = False,
                drop_last: Bool = False,
                collate_fn: Optional[fn(List[Any]) -> Any] = None,
                sampler: Optional[Any] = None):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn

        var size = len(dataset)
        if sampler is not None:
            self.sampler = sampler
        else:
            if shuffle:
                self.sampler = RandomSampler(size)
            else:
                self.sampler = SequentialSampler(size)

        self._reset()

    # Reset iteration
fn _reset(mut self) -> None:
        self._indices = self.sampler.__iter__()
        self._pos = 0

    # Iterator
fn __iter__(mut self) -> DataLoader:
        self._reset()
        return self

    # Next batch
fn __next__(mut self) -> Any:
        if self._pos >= len(self._indices):
            raise StopIteration

        var end = self._pos + self.batch_size
        var batch_indices = self._indices[self._pos:end]
        self._pos = end

        if self.drop_last and len(batch_indices) < self.batch_size:
            raise StopIteration

        var batch = List[Any]()
        for idx in batch_indices:
            batch.append(self.dataset[idx])

        if self.collate_fn is not None:
            return self.collate_fn(batch)
        else:
            return batch
fn __copyinit__(out self, other: Self) -> None:
        self.dataset = other.dataset
        self.batch_size = other.batch_size
        self.shuffle = other.shuffle
        self.drop_last = other.drop_last
        self.collate_fn = other.collate_fn
        self.sampler = other.sampler
        self._indices = other._indices
        self._pos = other._pos
fn __moveinit__(out self, deinit other: Self) -> None:
        self.dataset = other.dataset
        self.batch_size = other.batch_size
        self.shuffle = other.shuffle
        self.drop_last = other.drop_last
        self.collate_fn = other.collate_fn
        self.sampler = other.sampler
        self._indices = other._indices
        self._pos = other._pos
# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------

struct DummyDataset:
    var data: List[Int]
fn __init__(out self) -> None:
        self.data = [0,1,2,3,4,5,6,7,8,9]
fn __len__(self) -> Int:
        return len(self.data)
fn __getitem__(self, idx: Int) -> Int:
        return self.data[idx]
fn __copyinit__(out self, other: Self) -> None:
        self.data = other.data
fn __moveinit__(out self, deinit other: Self) -> None:
        self.data = other.data
fn _self_test() -> Bool:
    var ok = True
    var dataset = DummyDataset()
    var loader = DataLoader(dataset, batch_size=3, shuffle=False, drop_last=False)

    var count = 0
    for batch in loader:
        count += 1
        ok = ok and len(batch) <= 3
    ok = ok and count > 0

    var loader2 = DataLoader(dataset, batch_size=4, shuffle=True)
    var count2 = 0
    for batch in loader2:
        count2 += 1
    ok = ok and count2 > 0

    return ok
fn main() -> None:
    if _self_test():
        print("DataLoader module self-test: OK")
    else:
        print("DataLoader module self-test: FAIL")