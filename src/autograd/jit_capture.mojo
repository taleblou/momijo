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
# Project: momijo.autograd
# File: src/momijo/autograd/jit_capture.mojo

from momijo.extras.stubs import Copyright, MIT, Ta, best, fieldwise_init, https, if, len, momijo, ns, return
from momijo.dataframe.series_bool import append
from momijo.io.onnx.opset import Graph
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

s is a cheap smoke-test hook; extend with real checks as needed.
    return True


from autograd.tape import Tape, Node
from autograd.variable import Variable

@fieldwise_init("implicit")
struct Graph:
    var nodes: List[N

fn capture_apply(fn_name: String, tape: Ta

the op

or _, n in tape.nodes:
        ns.append(n)
    return Graph(ns)

