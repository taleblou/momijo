# Project:      Momijo
# Module:       src.momijo.runtime.eager_executor
# File:         eager_executor.mojo
# Path:         src/momijo/runtime/eager_executor.mojo
#
# Description:  Runtime facilities: device/context management, error handling,
#               environment queries, and state injection patterns.
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
#   - Structs: EagerExecutor
#   - Key functions: __init__, run_op, history, __copyinit__, __moveinit__, _self_test


from momijo.autograd.function import apply
from pathlib import Path
from pathlib.path import Path
from sys import platform

@fieldwise_init
struct EagerExecutor:
    var executed_ops: List[String]
fn __init__(out self) -> None:
        self.executed_ops = List[String]()
fn run_op(mut self, op_name: String, inputs: List[Int]) -> List[Int]:
        # For demonstration, just echo inputs or apply a trivial transform
        self.executed_ops.push_back(op_name)
        var outputs = List[Int]()
        var i = 0
        while i < len(inputs):
            var val = inputs[i]
            if op_name == String("double"):
                val = val * 2
            elif op_name == String("square"):
                val = val * val
            outputs.push_back(val)
            i += 1
        return outputs
fn history(self) -> List[String]:
        return self.executed_ops
fn __copyinit__(out self, other: Self) -> None:
        self.executed_ops = other.executed_ops
fn __moveinit__(out self, deinit other: Self) -> None:
        self.executed_ops = other.executed_ops
fn _self_test() -> Bool:
    var ex = EagerExecutor()
    var xs = List[Int]()
    xs.push_back(2)
    xs.push_back(3)
    var ys = ex.run_op(String("double"), xs)
    var ok = True
    if ys[0] != 4 or ys[1] != 6:
        ok = False
    if len(ex.history()) != 1:
        ok = False
    return ok