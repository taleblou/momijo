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

from momijo.dataframe.series_bool import append
from momijo.extras.stubs import Ta, best, fieldwise_init, if, len, ns, return
from momijo.io.onnx.opset import Graph
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

s is a cheap smoke-test hook; extend with real checks as needed.
    return True

@fieldwise_init("implicit")
struct Graph:
    var nodes: List[N
fn __init__(out self, nodes: List[N) -> None:
        self.nodes = nodes
fn __copyinit__(out self, other: Self) -> None:
        self.nodes = other.nodes
fn __moveinit__(out self, deinit other: Self) -> None:
        self.nodes = other.nodes
fn capture_apply(fn_name: String, tape: Ta

the op

or _, n in tape.nodes:
        ns.append(n)
    return Graph(ns)