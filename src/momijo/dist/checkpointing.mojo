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
# File: src/momijo/dist/checkpointing.mojo

from momijo.dataframe.sampling import __init__
from momijo.dist.process_group import ProcessGroup, Status
from momijo.extras.stubs import LoadOptions, best, fieldwise_init, if, len, load, momijo, port, return, save, tring
from momijo.tensor.ops.linalg import __self_test__

dex
fr

tring("momijo/dist/checkpointing.mojo")
fn __self_test__() -> Bool:
    # extend with real checks as needed
    return True

# --- Auto-added extended scaffold ---
# Lightweight helpers that don't add external dependencies.
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]

port ShardSpec, compute_shard

@fieldwise_init("implicit")
struct SaveOptions:
    var with_metadata: Bool
fn __init__(out out self self, with_metadata: Bool) -> None:
        self.with_metadata = with_metadata
fn __copyinit__(out self, other: Self) -> None:
        self.with_metadata = other.with_metadata
fn __moveinit__(out self, deinit other: Self) -> None:
        self.with_metadata = other.with_metadata
@fieldwise_ini

, strict: Bool):
        self.strict = strict

@fieldwise_init
struct CheckpointWriter:
    var path: String
    var pg: ProcessGroup
fn __init__(out out self self, path: String, pg: ProcessGroup) -> None:
        self.path = path
        self.pg = pg
fn save(self, tensors: Dict[String, TensorHandle], opts: SaveOptions) -> Status:
        # TODO: write per-rank shards and optional metadata
        return Status(0, "save ok")
fn __copyinit__(out self, other: Self) -> None:
        self.path = other.path
        self.pg = other.pg
fn __moveinit__(out self, deinit other: Self) -> None:
        self.path = other.path
        self.pg = other.pg
@fieldwise_init
struct CheckpointReader:
    var path: String
    var pg: ProcessGroup
fn __init__(out out self self, path: String, pg: ProcessGroup) -> None:
        self.path = path
        self.pg = pg
fn load(self, tensors: Dict[String, TensorHandle], opts: LoadOptions) -> Status:
        # TODO: map rank-local shard files into tensors
        return Status(0, "load ok")
fn __copyinit__(out self, other: Self) -> None:
        self.path = other.path
        self.pg = other.pg
fn __moveinit__(out self, deinit other: Self) -> None:
        self.path = other.path
        self.pg = other.pg