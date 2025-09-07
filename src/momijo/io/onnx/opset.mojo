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

from momijo.arrow_core.buffer_slice import __init__
from momijo.nn.parameter import data
from momijo.tensor.allocator import free
from momijo.tensor.dtype import BFLOAT16
from momijo.tensor.ufunc import add
from momijo.utils.result import contains, fieldwise_init, get, has
from pathlib import Path
from pathlib.path import Path

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
fn __init__(out self, UNDEFINED: Int32 = 0, FLOAT: Int32 = 0, UINT8: Int32 = 0, INT8: Int32 = 0, UINT16: Int32 = 0, INT16: Int32 = 0, INT32: Int32 = 0, INT64: Int32 = 0, STRING: Int32 = 0, BOOL: Int32 = 0, FLOAT16: Int32 = 0, DOUBLE: Int32 = 0, BFLOAT16: Int32 = 0) -> None:
        self.UNDEFINED = UNDEFINED
        self.FLOAT = FLOAT
        self.UINT8 = UINT8
        self.INT8 = INT8
        self.UINT16 = UINT16
        self.INT16 = INT16
        self.INT32 = INT32
        self.INT64 = INT64
        self.STRING = STRING
        self.BOOL = BOOL
        self.FLOAT16 = FLOAT16
        self.DOUBLE = DOUBLE
        self.BFLOAT16 = BFLOAT16
fn __copyinit__(out self, other: Self) -> None:
        self.UNDEFINED = other.UNDEFINED
        self.FLOAT = other.FLOAT
        self.UINT8 = other.UINT8
        self.INT8 = other.INT8
        self.UINT16 = other.UINT16
        self.INT16 = other.INT16
        self.INT32 = other.INT32
        self.INT64 = other.INT64
        self.STRING = other.STRING
        self.BOOL = other.BOOL
        self.FLOAT16 = other.FLOAT16
        self.DOUBLE = other.DOUBLE
        self.BFLOAT16 = other.BFLOAT16
fn __moveinit__(out self, deinit other: Self) -> None:
        self.UNDEFINED = other.UNDEFINED
        self.FLOAT = other.FLOAT
        self.UINT8 = other.UINT8
        self.INT8 = other.INT8
        self.UINT16 = other.UINT16
        self.INT16 = other.INT16
        self.INT32 = other.INT32
        self.INT64 = other.INT64
        self.STRING = other.STRING
        self.BOOL = other.BOOL
        self.FLOAT16 = other.FLOAT16
        self.DOUBLE = other.DOUBLE
        self.BFLOAT16 = other.BFLOAT16
@fieldwise_init
struct Attribute:
    var name: String
    var i: Int64
    var f: Float64
    var s: String
    var ints: List[Int64]
    var floats: List[Float64]
    var strings: List[String]
fn __init__(out out self self, name: String) -> None:
        self.name = name
        self.i = 0
        self.f = 0.0
        self.s = String("")
        self.ints = List[Int64]()
        self.floats = List[Float64]()
        self.strings = List[String]()
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.i = other.i
        self.f = other.f
        self.s = other.s
        self.ints = other.ints
        self.floats = other.floats
        self.strings = other.strings
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.i = other.i
        self.f = other.f
        self.s = other.s
        self.ints = other.ints
        self.floats = other.floats
        self.strings = other.strings
@fieldwise_init
struct Node:
    var op_type: String
    var inputs: List[String]
    var outputs: List[String]
    var attributes: List[Attribute]
fn __init__(out out self self, op_type: String, inputs: List[String], outputs: List[String], attributes: List[Attribute]) -> None:
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.attributes = attributes
fn __copyinit__(out self, other: Self) -> None:
        self.op_type = other.op_type
        self.inputs = other.inputs
        self.outputs = other.outputs
        self.attributes = other.attributes
fn __moveinit__(out self, deinit other: Self) -> None:
        self.op_type = other.op_type
        self.inputs = other.inputs
        self.outputs = other.outputs
        self.attributes = other.attributes
@fieldwise_init
struct ValueInfo:
    var name: String
    var elem_type: TensorElemType
    var shape: List[Int64]
fn __init__(out out self self, name: String, elem_type: TensorElemType, shape: List[Int64]) -> None:
        self.name = name
        self.elem_type = elem_type
        self.shape = shape
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.elem_type = other.elem_type
        self.shape = other.shape
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.elem_type = other.elem_type
        self.shape = other.shape
@fieldwise_init
struct TensorProto:
    var name: String
    var elem_type: TensorElemType
    var dims: List[Int64]
    var data: List[UInt8]   # raw little-endian payload
fn __init__(out out self self, name: String, elem_type: TensorElemType, dims: List[Int64], data: List[UInt8]) -> None:
        self.name = name
        self.elem_type = elem_type
        self.dims = dims
        self.data = data
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.elem_type = other.elem_type
        self.dims = other.dims
        self.data = other.data
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.elem_type = other.elem_type
        self.dims = other.dims
        self.data = other.data
@fieldwise_init
struct Graph:
    var name: String
    var inputs: List[ValueInfo]
    var outputs: List[ValueInfo]
    var initializers: List[TensorProto]
    var nodes: List[Node]
fn __init__(out out self self, name: String) -> None:
        self.name = name
        self.inputs = List[ValueInfo]()
        self.outputs = List[ValueInfo]()
        self.initializers = List[TensorProto]()
        self.nodes = List[Node]()
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.inputs = other.inputs
        self.outputs = other.outputs
        self.initializers = other.initializers
        self.nodes = other.nodes
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.inputs = other.inputs
        self.outputs = other.outputs
        self.initializers = other.initializers
        self.nodes = other.nodes
@fieldwise_init
struct OpsetId:
    var domain: String
    var version: Int64
fn __init__(out out self self, domain: String, version: Int64) -> None:
        self.domain = domain
        self.version = version
fn __copyinit__(out self, other: Self) -> None:
        self.domain = other.domain
        self.version = other.version
fn __moveinit__(out self, deinit other: Self) -> None:
        self.domain = other.domain
        self.version = other.version
@fieldwise_init
struct Model:
    var ir_version: Int64
    var opset: OpsetId
    var graph: Graph
    var producer: String
    var producer_version: String
fn __init__(out out self self, opset_version: Int64, graph: Graph, producer: String, producer_version: String) -> None:
        self.ir_version = 8  # placeholder
        self.opset = OpsetId(String("ai.onnx"), opset_version)
        self.graph = graph
        self.producer = producer
        self.producer_version = producer_version
fn __copyinit__(out self, other: Self) -> None:
        self.ir_version = other.ir_version
        self.opset = other.opset
        self.graph = other.graph
        self.producer = other.producer
        self.producer_version = other.producer_version
fn __moveinit__(out self, deinit other: Self) -> None:
        self.ir_version = other.ir_version
        self.opset = other.opset
        self.graph = other.graph
        self.producer = other.producer
        self.producer_version = other.producer_version
# A tiny schema registry for common ops (name -> required inputs/outputs)
@fieldwise_init
struct OpSchema:
    var name: String
    var min_inputs: Int
    var min_outputs: Int
fn __init__(out out self self, name: String, min_inputs: Int, min_outputs: Int) -> None:
        self.name = name
        self.min_inputs = min_inputs
        self.min_outputs = min_outputs
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.min_inputs = other.min_inputs
        self.min_outputs = other.min_outputs
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.min_inputs = other.min_inputs
        self.min_outputs = other.min_outputs
@fieldwise_init("implicit")
struct OpsetRegistry:
    var schemas: Dict[String, OpSchema]
fn __init__(out out self self) -> None:
        self.schemas = Dict[String, OpSchema]()
        self.add_builtin()
fn add(mut self, s: OpSchema) -> None:
        self.schemas[s.name] = s
fn add_builtin(mut self) -> None:
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
fn __copyinit__(out self, other: Self) -> None:
        self.schemas = other.schemas
fn __moveinit__(out self, deinit other: Self) -> None:
        self.schemas = other.schemas
# --- Auto-generated free-function wrappers ---
fn OpsetRegistry_add(mut x: OpsetRegistry, s: OpSchema) -> None:
    x.add(s)
fn OpsetRegistry_add_builtin(mut x: OpsetRegistry) -> None:
    x.add_builtin()