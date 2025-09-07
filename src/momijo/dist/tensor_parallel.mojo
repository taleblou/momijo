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
# File: src/momijo/dist/tensor_parallel.mojo

from momijo.arrow_core.tensor_bridge import TensorHandle
from momijo.core.device import kind
from momijo.core.error import module
from momijo.core.ops.matmul import matmul
from momijo.dataframe.sampling import __init__
from momijo.dist.process_group import ProcessGroup, Status, allgather, allreduce
from momijo.utils.result import TensorHandle, allgather, best, ct, momijo
from pathlib import Path
from pathlib.path import Path

g("momijo/dist/tensor_parallel.mojo")
# NOTE: Removed duplicate definition of `__self_test__`; use `from momijo.dist.checkpointing import __self_test__`
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]

ct TPSpecKind:
    var ROW: Int32 = 0
    var COL: Int32 = 1

@fieldwise_init
struct TensorParallelSpec:
    var kind: TPSpecKind
    var parts: Int
fn __init__(out out self self, kind: TPSpecKind, parts: Int) -> None:
fn __copyinit__(out self, other: Self) -> None:
        self.kind = other.kind
        self.parts = other.parts
fn __moveinit__(out self, deinit other: Self) -> None:
        self.kind = other.kind
        self.parts = other.parts
(per-rank view).
fn scatter_weight_row(pg: ProcessGroup, w: TensorHandle, spec: TensorParallelSpec) -> TensorHandle:
    # TODO: produce a view into the appropriate shard based on pg.rank
    return w

# Scatter a weight matrix into column shards.
fn scatter_weight_col(pg: ProcessGroup, w: TensorHandle, spec: TensorParallelSpec) -> TensorHandle:
    return w

# Gather partial outputs across TP mesh.
fn tp_allgather(pg: ProcessGroup, partials: List[TensorHandle]) -> Status:
    return pg.allgather(partials, partials)

# Example: shard-aware matmul (placeholder, relies on backend fused kernels in practice).
fn tp_matmul(pg: ProcessGroup, a: TensorHandle, b: TensorHandle, spec: TensorParallelSpec) -> TensorHandle:
    # For ROW-shard: each rank holds rows of b; compute local matmul then allgather.
    # For COL-shard: each rank holds cols of b; compute partial and allreduce sum.
    return a  # placeholder handle