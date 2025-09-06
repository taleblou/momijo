# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.runtime
# File: src/momijo/runtime/eager_executor.mojo

# EagerExecutor: simplified runtime executor for operations in "eager" mode.
# Executes ops immediately rather than building a computation graph.

@fieldwise_init
struct EagerExecutor:
    var executed_ops: List[String]

    fn __init__(out self):
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
