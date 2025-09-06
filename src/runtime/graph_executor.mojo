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
# File: src/momijo/runtime/graph_executor.mojo

# GraphExecutor: runtime executor for operation graphs (vs eager executor).
# Supports simple scheduling and execution order.

@fieldwise_init
struct GraphNode:
    var id: Int
    var op_name: String
    var inputs: List[Int]
    var outputs: List[Int]

    fn __init__(out self, id: Int, op_name: String, inputs: List[Int]):
        self.id = id
        self.op_name = op_name
        self.inputs = inputs
        self.outputs = List[Int]()


@fieldwise_init
struct GraphExecutor:
    var nodes: List[GraphNode]

    fn __init__(out self):
        self.nodes = List[GraphNode]()

    fn add_node(mut self, id: Int, op_name: String, inputs: List[Int]) -> GraphNode:
        var n = GraphNode(id, op_name, inputs)
        self.nodes.push_back(n)
        return n

    fn run(mut self):
        # Simplified: compute outputs deterministically based on op_name
        var i = 0
        while i < len(self.nodes):
            var n = self.nodes[i]
            var outs = List[Int]()
            var j = 0
            while j < len(n.inputs):
                var v = n.inputs[j]
                if n.op_name == String("double"):
                    v = v * 2
                elif n.op_name == String("square"):
                    v = v * v
                outs.push_back(v)
                j += 1
            self.nodes[i].outputs = outs
            i += 1

    fn get_outputs(self, id: Int) -> List[Int]:
        var i = 0
        while i < len(self.nodes):
            if self.nodes[i].id == id:
                return self.nodes[i].outputs
            i += 1
        return List[Int]()


fn _self_test() -> Bool:
    var gx = GraphExecutor()
    var xs = List[Int]()
    xs.push_back(3)
    xs.push_back(4)
    gx.add_node(1, String("double"), xs)
    gx.add_node(2, String("square"), xs)
    gx.run()
    var out1 = gx.get_outputs(1)
    var out2 = gx.get_outputs(2)
    var ok = True
    if out1[0] != 6 or out2[1] != 16:
        ok = False
    return ok
