# Project:      Momijo
# Module:       src.momijo.runtime.op_signature
# File:         op_signature.mojo
# Path:         src/momijo/runtime/op_signature.mojo
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
#   - Structs: OpSignature
#   - Key functions: __init__, summary, __copyinit__, __moveinit__, make_op_signature, _self_test


@fieldwise_init
struct OpSignature:
    var name: String
    var input_count: Int
    var output_count: Int
fn __init__(out self, name: String, input_count: Int, output_count: Int) -> None:
        self.name = name
        self.input_count = input_count
        self.output_count = output_count
fn summary(self) -> String:
        var s = String("OpSignature(") + self.name
        s = s + String(", in=") + String(self.input_count)
        s = s + String(", out=") + String(self.output_count) + String(")")
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.input_count = other.input_count
        self.output_count = other.output_count
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.input_count = other.input_count
        self.output_count = other.output_count
fn make_op_signature(name: String, inputs: Int, outputs: Int) -> OpSignature:
    return OpSignature(name, inputs, outputs)
fn _self_test() -> Bool:
    var sig = make_op_signature(String("matmul"), 2, 1)
    var ok = True
    if sig.input_count != 2 or sig.output_count != 1:
        ok = False
    if len(sig.summary()) == 0:
        ok = False
    return ok