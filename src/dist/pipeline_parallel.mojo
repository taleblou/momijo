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
# File: src/momijo/dist/pipeline_parallel.mojo

from momijo.extras.stubs import Copyright, MIT, PipelineStage, barrier, best, his, https, if, len, momijo, nit__, pipeline, return, run_backward, run_forward
from momijo.utils.string_utils import split
from momijo.dataframe.sampling import __init__
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

his is a cheap smoke-test hook; extend with real checks as needed.
    return True


from dist.process_group import ProcessGroup, Status
from arrow_core.tensor_bridge import TensorHandle

@fieldwise_init
struct PipelineStag

nit__(out out self self, stage_index: Int,

.num_sta

tch_chunks: Int

    fn __init__(out out self self, stage: PipelineStage, microbatch_chunks: Int):
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
