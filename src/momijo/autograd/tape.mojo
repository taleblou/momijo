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
# File: src/momijo/autograd/tape.mojo

from momijo.arrow_core.tensor_bridge import TensorHandle
from momijo.autograd.hook import GradHook
from momijo.autograd.variable import Variable

struct Node:
    var id: Int
    var op: String
    var input_ids: List[Int]
    var output_ids: List[Int]
    var saved: List[TensorHandle]
    var hooks: List[GradHook]
fn __init__(out self,
                id: Int,
                op: String,
                input_ids: List[Int],
                output_ids: List[Int],
                saved: List[TensorHandle],
                hooks: List[GradHook]) -> None:
        self.id = id
        self.op = op
        self.input_ids = input_ids
        self.output_ids = output_ids
        self.saved = saved
        self.hooks = hooks
fn __copyinit__(out self, other: Self) -> None:
        self.id = other.id
        self.op = other.op
        self.input_ids = other.input_ids
        self.output_ids = other.output_ids
        self.saved = other.saved
        self.hooks = other.hooks
fn __moveinit__(out self, deinit other: Self) -> None:
        self.id = other.id
        self.op = other.op
        self.input_ids = other.input_ids
        self.output_ids = other.output_ids
        self.saved = other.saved
        self.hooks = other.hooks
# -----------------------------
# Tape
# -----------------------------
struct Tape:
    var is_recording: Bool
    var next_var_id: Int
    var next_node_id: Int
    var variables: Dict[Int, Variable]
    var nodes: Dict[Int, Node]
fn __init__(out self, recording: Bool) -> None:
        self.is_recording = recording
        self.next_var_id = 1
        self.next_node_id = 1
        self.variables = Dict[Int, Variable]()
        self.nodes = Dict[Int, Node]()
fn enable_recording(mut self) -> None:
        self.is_recording = True
fn disable_recording(mut self) -> None:
        self.is_recording = False
fn make_leaf(mut self, data: TensorHandle, requires_grad: Bool) -> Variable:
        var vid = self.next_var_id
        self.next_var_id += 1
        var v = Variable(vid, data, requires_grad, True, -1)
        self.variables[vid] = v
        return v
fn record_op(mut self, op: String,
                 inputs: List[Variable],
                 outputs: List[Variable],
                 saved: List[TensorHandle]) -> Node:
        if not self.is_recording:
            # no-op if not recording
            return Node(-1, op, List[Int](), List[Int](), List[TensorHandle](), List[GradHook]())

        var nid = self.next_node_id
        self.next_node_id += 1

        var in_ids = List[Int]()
        var i = 0
        while i < len(inputs):
            in_ids.append(inputs[i].id)
            i += 1

        var out_ids = List[Int]()
        var j = 0
        while j < len(outputs):
            out_ids.append(outputs[j].id)
            j += 1

        var node = Node(nid, op, in_ids, out_ids, saved, List[GradHook]())
        self.nodes[nid] = node

        # Link outputs to producer id
        var k = 0
        while k < len(outputs):
            var v = outputs[k]
            v.producer = nid
            self.variables[v.id] = v
            k += 1

        return node
fn get_variable(self, vid: Int) -> Variable:
        return self.variables[vid]
fn get_producer(self, vid: Int) -> Node:
        var v = self.variables[vid]
        return self.nodes[v.producer]
fn all_nodes(self) -> List[Node]:
        var ns = List[Node]()
        for _, n in self.nodes:
            ns.append(n)
        return ns
fn __copyinit__(out self, other: Self) -> None:
        self.is_recording = other.is_recording
        self.next_var_id = other.next_var_id
        self.next_node_id = other.next_node_id
        self.variables = other.variables
        self.nodes = other.nodes
fn __moveinit__(out self, deinit other: Self) -> None:
        self.is_recording = other.is_recording
        self.next_var_id = other.next_var_id
        self.next_node_id = other.next_node_id
        self.variables = other.variables
        self.nodes = other.nodes
# -----------------------------
# Self-test
# -----------------------------
fn __self_test__() -> Bool:
    var ok = True
    var tape = Tape(True)
    var dummy: TensorHandle
    var v = tape.make_leaf(dummy, True)
    ok = ok and (v.requires_grad)
    return ok