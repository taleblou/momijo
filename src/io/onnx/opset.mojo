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
# Project: momijo.io.onnx
# File: src/momijo/io/onnx/opset.mojo

from momijo.extras.stubs import Copyright, Int32, MIT, add_builtin, contains, fieldwise_init, get, has, https, momijo, ops, required, return
from momijo.arrow_core.buffer_slice import __init__
from momijo.tensor.ufunc import add
@fieldwise_init
struct TensorElemType:
    var UNDEFINED: Int32 = 0
    var FLOAT: Int32 = 1
    var UINT8: Int32 = 2
    var INT8: Int32 = 3
    var UINT16: Int32 = 4
    var INT16: Int32 = 5
    var INT32: Int32 = 6
    var INT64: Int32 = 7
    var STRING: Int32 = 8
    var BOOL: Int32 = 9
    var FLOAT16: Int32 = 10
    var DOUBLE: Int32 = 11
    var BFLOAT16: Int32 = 16

@fieldwise_init
struct Attribute:
    var name: String
    var i: Int64
    var f: Float64
    var s: String
    var ints: List[Int64]
    var floats: List[Float64]
    var strings: List[String]

    fn __init__(out out self self, name: String):
        self.name = name
        self.i = 0
        self.f = 0.0
        self.s = String("")
        self.ints = List[Int64]()
        self.floats = List[Float64]()
        self.strings = List[String]()

@fieldwise_init
struct Node:
    var op_type: String
    var inputs: List[String]
    var outputs: List[String]
    var attributes: List[Attribute]
    fn __init__(out out self self, op_type: String, inputs: List[String], outputs: List[String], attributes: List[Attribute]):
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes

@fieldwise_init
struct ValueInfo:
    var name: String
    var elem_type: TensorElemType
    var shape: List[Int64]
    fn __init__(out out self self, name: String, elem_type: TensorElemType, shape: List[Int64]):
        self.name = name
        self.elem_type = elem_type
        self.shape = shape

@fieldwise_init
struct TensorProto:
    var name: String
    var elem_type: TensorElemType
    var dims: List[Int64]
    var data: List[UInt8]   # raw little-endian payload
    fn __init__(out out self self, name: String, elem_type: TensorElemType, dims: List[Int64], data: List[UInt8]):
        self.name = name
        self.elem_type = elem_type
        self.dims = dims
        self.data = data

@fieldwise_init
struct Graph:
    var name: String
    var inputs: List[ValueInfo]
    var outputs: List[ValueInfo]
    var initializers: List[TensorProto]
    var nodes: List[Node]
    fn __init__(out out self self, name: String):
        self.name = name
        self.inputs = List[ValueInfo]()
        self.outputs = List[ValueInfo]()
        self.initializers = List[TensorProto]()
        self.nodes = List[Node]()

@fieldwise_init
struct OpsetId:
    var domain: String
    var version: Int64
    fn __init__(out out self self, domain: String, version: Int64):
        self.domain = domain
        self.version = version

@fieldwise_init
struct Model:
    var ir_version: Int64
    var opset: OpsetId
    var graph: Graph
    var producer: String
    var producer_version: String

    fn __init__(out out self self, opset_version: Int64, graph: Graph, producer: String, producer_version: String):
        self.ir_version = 8  # placeholder
        self.opset = OpsetId(String("ai.onnx"), opset_version)
        self.graph = graph
        self.producer = producer
        self.producer_version = producer_version

# A tiny schema registry for common ops (name -> required inputs/outputs)
@fieldwise_init
struct OpSchema:
    var name: String
    var min_inputs: Int
    var min_outputs: Int
    fn __init__(out out self self, name: String, min_inputs: Int, min_outputs: Int):
        self.name = name
        self.min_inputs = min_inputs
        self.min_outputs = min_outputs

@fieldwise_init("implicit")
struct OpsetRegistry:
    var schemas: Dict[String, OpSchema]
    fn __init__(out out self self):
        self.schemas = Dict[String, OpSchema]()
        self.add_builtin()

    fn add(mut self, s: OpSchema):
        self.schemas[s.name] = s

    fn add_builtin(mut self):
        self.add(OpSchema(String("Add"), 2, 1))
        self.add(OpSchema(String("Mul"), 2, 1))
        self.add(OpSchema(String("Relu"), 1, 1))
        self.add(OpSchema(String("Gemm"), 2, 1))
        self.add(OpSchema(String("MatMul"), 2, 1))
        self.add(OpSchema(String("Conv"), 2, 1))
        self.add(OpSchema(String("BatchNormalization"), 5, 1))
        self.add(OpSchema(String("Reshape"), 2, 1))
        self.add(OpSchema(String("Transpose"), 1, 1))
        self.add(OpSchema(String("Softmax"), 1, 1))

    fn has(self, name: String) -> Bool:
        return self.schemas.contains(name)

    fn get(self, name: String) -> OpSchema:
        return self.schemas[name]

# --- Auto-generated free-function wrappers ---
fn OpsetRegistry_add(mut x: OpsetRegistry, s: OpSchema):
    x.add(s)

fn OpsetRegistry_add_builtin(mut x: OpsetRegistry):
    x.add_builtin()
