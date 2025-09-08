# Project:      Momijo
# Module:       src.momijo.dist.pipeline_parallel
# File:         pipeline_parallel.mojo
# Path:         src/momijo/dist/pipeline_parallel.mojo
#
# Description:  src.momijo.dist.pipeline_parallel — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Structs: PipelineStag
#   - Key functions: argmin_index, __init__, run_forward, run_backward, barrier


from arrow_core.tensor_bridge import TensorHandle
from dist.process_group import Status
from momijo.dataframe.sampling import __init__
from momijo.extras.stubs import PipelineStage, barrier, best, his, if, len, nit__, return, run_backward, run_forward
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

his is a cheap smoke-test hook; extend with real checks as needed.
    return True

@fieldwise_init
struct PipelineStag

nit__(out out self self, stage_index: Int,

.num_sta

tch_chunks: Int
fn __init__(out out self self, stage: PipelineStage, microbatch_chunks: Int) -> None:
        self.stage = stage
        self.microbatch_chunks = microbatch_chunks
fn run_forward(self, inputs: List[TensorHandle]) -> List[TensorHandle]:
        # TODO: split into microbatches, send/recv activations, execute local stage
        return inputs
fn run_backward(self, grads: List[TensorHandle]) -> List[TensorHandle]:
        # TODO: pipeline backprop with send/recv of gradients
        return grads
fn barrier(self) -> Status:
        return self.stage.pg.barrier()