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
# File: src/momijo/io/datasets/samplers.mojo

import random

struct IndexIter:
    var indices: List[Int]
    var pos: Int
fn __init__(out self, indices: List[Int]) -> None:
        self.indices = indices
        self.pos = 0
fn __iter__(self) -> IndexIter:
        return self
fn __next__(mut self) -> Int:
        if self.pos >= len(self.indices):
            raise StopIteration
        var v = self.indices[self.pos]
        self.pos += 1
        return v
fn __copyinit__(out self, other: Self) -> None:
        self.indices = other.indices
        self.pos = other.pos
fn __moveinit__(out self, deinit other: Self) -> None:
        self.indices = other.indices
        self.pos = other.pos
# -----------------------------------------------------------------------------
# SequentialSampler
# -----------------------------------------------------------------------------
struct SequentialSampler:
    var size: Int
fn __init__(out self, size: Int) -> None:
        self.size = size
fn __iter__(self) -> IndexIter:
        var idxs = List[Int]()
        for i in range(self.size):
            idxs.append(i)
        return IndexIter(idxs)
fn __copyinit__(out self, other: Self) -> None:
        self.size = other.size
fn __moveinit__(out self, deinit other: Self) -> None:
        self.size = other.size
# -----------------------------------------------------------------------------
# RandomSampler
# -----------------------------------------------------------------------------
struct RandomSampler:
    var size: Int
fn __init__(out self, size: Int) -> None:
        self.size = size
fn __iter__(self) -> IndexIter:
        var idxs = List[Int]()
        for i in range(self.size):
            idxs.append(i)
        random.shuffle(idxs)
        return IndexIter(idxs)
fn __copyinit__(out self, other: Self) -> None:
        self.size = other.size
fn __moveinit__(out self, deinit other: Self) -> None:
        self.size = other.size
# -----------------------------------------------------------------------------
# DistributedSampler: split indices among ranks
# -----------------------------------------------------------------------------
struct DistributedSampler:
    var size: Int
    var num_replicas: Int
    var rank: Int
fn __init__(out self, size: Int, num_replicas: Int, rank: Int) -> None:
        self.size = size
        self.num_replicas = num_replicas
        self.rank = rank
fn __iter__(self) -> IndexIter:
        var idxs = List[Int]()
        for i in range(self.size):
            if i % self.num_replicas == self.rank:
                idxs.append(i)
        return IndexIter(idxs)
fn __copyinit__(out self, other: Self) -> None:
        self.size = other.size
        self.num_replicas = other.num_replicas
        self.rank = other.rank
fn __moveinit__(out self, deinit other: Self) -> None:
        self.size = other.size
        self.num_replicas = other.num_replicas
        self.rank = other.rank