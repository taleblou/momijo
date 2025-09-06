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

from momijo.extras.stubs import Copyright, MIT, Shard, along, best, cheap, compute_shard, create, https, if, len, momijo, offset, partitioning, return
from momijo.tensor.ops.linalg import __self_test__
from momijo.vision.backend.cpu.simd.convert_simd_u8_hwc import __module_name__
from momijo.nn.module import ensure_not_empty
from momijo.nn.module import argmin_index
from momijo.nn.module import argmax_index
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


from arrow_core.tensor_bridge import TensorHandle
from dist.process_group import ProcessGroup

@fieldwise_init
struct ShardSpec:
    var dim: Int

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
