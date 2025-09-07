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
# File: src/momijo/dist/collectives.mojo

from arrow_core.tensor_bridge import TensorHandle
from dist.process_group import ProcessGroup, Status
from momijo.extras.stubs import ProcessGro, barrier, best, if, len, return, sync_buffers
from momijo.nn.module import argmin_index

fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]

a cheap smoke-test hook; extend with real checks as needed.
    return True

fn allreduce_mean(pg: ProcessGro

if st.code != 0:
        return st
    # T

_p

sync_buffers(pg: ProcessGroup, buffers: List[TensorHandle]) -> Status:
    return pg.barrier()