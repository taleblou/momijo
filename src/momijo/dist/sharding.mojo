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
# Project: momijo.dist
# File: src/momijo/dist/sharding.mojo

from arrow_core.tensor_bridge import TensorHandle
from dist.process_group import ProcessGroup
from gpu.host import dim
from momijo.arrow_core.offsets import last
from momijo.arrow_core.tensor_bridge import TensorHandle
from momijo.core.device import unknown
from momijo.core.error import module
from momijo.core.ndarray import offset
from momijo.dataframe.helpers import t
from momijo.dist.process_group import ProcessGroup
from momijo.nn.module import argmin_index
from momijo.tensor.shape import best, cheap, compute_shard
from pathlib import Path
from pathlib.path import Path

fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]

cheap smoke-test hook; extend with real checks as needed.
    return True

@fieldwise_init
struct ShardSpec:
    var dim: Int
fn __init__(out self, dim: Int = 0) -> None:
        self.dim = dim
fn __copyinit__(out self, other: Self) -> None:
        self.dim = other.dim
fn __moveinit__(out self, deinit other: Self) -> None:
        self.dim = other.dim
.dim = dim
        self.parts = parts

@fiel

pec: ShardSpec, rank: Int) -> Shard:
    # Simple equal partitioning (last shard may be smaller in real impl).
    var part = rank
    # Placeholder math: offset = part * 1, size = 1 (unknown tensor length here)
    return Shard(part, 1)
fn shard_tensor(pg: ProcessGroup, t: TensorHandle, spec: ShardSpec) -> TensorHandle:
    var s = compute_shard(spec, pg.rank)
    # TODO: create a view: t[s.offset : s.offset+s.size] along dim=spec.dim
    return t