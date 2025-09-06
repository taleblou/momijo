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
# File: src/momijo/io/onnx/import.mojo

from momijo.extras.stubs import Copyright, MIT, as, assert, attrs, dims, https, ins, len, momijo, outs, payload, return, shape
from momijo.dataframe.series_bool import append
from onnx.opset import TensorElemType, TensorProto, ValueInfo, Node, Attribute, Graph, Model, OpsetId

fn _read_u32_le(bytes: List[UInt8], mut off: Int) -> UInt32:
    var v = (bytes[off] as UInt32) | ((bytes[off+1] as UInt32)<<8) | ((bytes[off+2] as UInt32)<<16) | ((bytes[off+3] as UInt32)<<24)
    off += 4
    return v

fn _read_u64_le(bytes: List[UInt8], mut off: Int) -> UInt64:
    var v: UInt64 = 0
    var i = 0
    while i < 8:
        v = v | ((bytes[off+i] as UInt64) << (8*i))
        i += 1
    off += 8
    return v

fn _read_i64_le(bytes: List[UInt8], mut off: Int) -> Int64:
    return _read_u64_le(bytes, off) as Int64

fn _read_str(bytes: List[UInt8], mut off: Int) -> String:
    var n = _read_u32_le(bytes, off) as Int
    var s = String("")
    var i = 0
    while i < n:
        s = s + String(bytes[off+i] as Int)
        i += 1
    off += n
    return s

fn parse_model(bytes: List[UInt8]) -> Model:
    # check magic
    assert(len(bytes) >= 8, "invalid MO_ONNX")
    assert(bytes[0] == ('M') as UInt8 and bytes[1] == ('O') as UInt8, "bad magic")
    var off = 7  # skip "MO_ONNX\n"
    off += 1
    var opset_version = _read_u64_le(bytes, off) as Int64
    var gname = _read_str(bytes, off)
    var n_inputs = _read_u32_le(bytes, off) as Int
    var n_outputs = _read_u32_le(bytes, off) as Int
    var n_inits = _read_u32_le(bytes, off) as Int
    var n_nodes = _read_u32_le(bytes, off) as Int

    var g = Graph(gname)

    # inputs
    var i = 0
    while i < n_inputs:
        var name = _read_str(bytes, off)
        var et = _read_u32_le(bytes, off) as Int
        var nshape = _read_u32_le(bytes, off) as Int
        var shape = List[Int64]()
        var j = 0
        while j < nshape:
            shape.append(_read_i64_le(bytes, off))
            j += 1
        g.inputs.append(ValueInfo(name, et as TensorElemType, shape))
        i += 1

    # outputs
    i = 0
    while i < n_outputs:
        var name = _read_str(bytes, off)
        var et = _read_u32_le(bytes, off) as Int
        var nshape = _read_u32_le(bytes, off) as Int
        var shape = List[Int64]()
        var j = 0
        while j < nshape:
            shape.append(_read_i64_le(bytes, off))
            j += 1
        g.outputs.append(ValueInfo(name, et as TensorElemType, shape))
        i += 1

    # initializers
    i = 0
    while i < n_inits:
        var name = _read_str(bytes, off)
        var et = _read_u32_le(bytes, off) as Int
        var ndims = _read_u32_le(bytes, off) as Int
        var dims = List[Int64]()
        var j = 0
        while j < ndims:
            dims.append(_read_i64_le(bytes, off))
            j += 1
        var nbytes = _read_u64_le(bytes, off) as Int
        var payload = List[UInt8]()
        var k = 0
        while k < nbytes:
            payload.append(bytes[off+k])
            k += 1
        off += nbytes
        g.initializers.append(TensorProto(name, et as TensorElemType, dims, payload))
        i += 1

    # nodes
    i = 0
    while i < n_nodes:
        var op_type = _read_str(bytes, off)
        var nin = _read_u32_le(bytes, off) as Int
        var ins = List[String]()
        var j = 0
        while j < nin:
            ins.append(_read_str(bytes, off)); j += 1
        var nout = _read_u32_le(bytes, off) as Int
        var outs = List[String]()
        j = 0
        while j < nout:
            outs.append(_read_str(bytes, off)); j += 1
        var nattr = _read_u32_le(bytes, off) as Int
        var attrs = List[Attribute]()
        j = 0
        while j < nattr:
            attrs.append(Attribute(_read_str(bytes, off)))
            j += 1
        g.nodes.append(Node(op_type, ins, outs, attrs))
        i += 1

    return Model(opset_version, g, String("momijo"), String("0.1"))
