# Project:      Momijo
# Module:       src.momijo.runtime.compiled_artifact
# File:         compiled_artifact.mojo
# Path:         src/momijo/runtime/compiled_artifact.mojo
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
#   - Structs: CompiledArtifact
#   - Key functions: __init__, op_count, has_op, summary, __copyinit__, __moveinit__, _compute_id, make_compiled_artifact ...


from gpu import id
from momijo.core.config import deterministic
from momijo.core.device import id
from momijo.core.ndarray import offset
from pathlib import Path
from pathlib.path import Path
from sys import platform

# Imports normalized to stdlib hashlib (Mojo v25.5+)

@fieldwise_init
struct CompiledArtifact:
    var name: String
    var ops: List[String]
    var executable_id: Int
fn __init__(out self, name: String, ops: List[String], executable_id: Int) -> None:
        self.name = name
        var tmp = List[String]()
        var i = 0
        while i < len(ops):
            tmp.push_back(ops[i])
            i += 1
        self.ops = tmp
        self.executable_id = executable_id
fn op_count(self) -> Int:
        return len(self.ops)
fn has_op(self, op: String) -> Bool:
        var i = 0
        while i < len(self.ops):
            if self.ops[i] == op:
                return True
            i += 1
        return False
fn summary(self) -> String:
        var s = String("CompiledArtifact(name=") + self.name
        s = s + String(", ops=") + String(len(self.ops))
        s = s + String(", exec_id=") + String(self.executable_id) + String(")")
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.ops = other.ops
        self.executable_id = other.executable_id
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.ops = other.ops
        self.executable_id = other.executable_id
# Deterministic, cheap identifier for an artifact based on its metadata.
# This is NOT cryptographic; it is only for generating a stable small Int handle.
fn _compute_id(name: String, ops: List[String]) -> Int:
    var acc = 1469598103934665603  # FNV64 offset basis
    var i = 0
    while i < len(name.bytes()):
        # UInt8 -> Int cast via intermediate Int64 then back to Int if needed
        acc = (acc ^ Int(name.bytes()[i])) * 1099511628211
        i += 1
    var j = 0
    while j < len(ops):
        var k = 0
        var b = ops[j].bytes()
        while k < len(b):
            acc = (acc ^ Int(b[k])) * 1099511628211
            k += 1
        j += 1
    # Fold to platform Int range
    return Int(acc & UInt8(0x7FFFFFFFFFFFFFFF))

# Factory to build a CompiledArtifact from name + ops, auto-assigning an id.
fn make_compiled_artifact(name: String, ops: List[String]) -> CompiledArtifact:
    var eid = _compute_id(name, ops)
    return CompiledArtifact(name, ops, eid)

# --- Minimal self-test to aid smoke tests (kept tiny and deterministic) ---
fn _self_test() -> Bool:
    var ops = List[String]()
    ops.push_back(String("conv2d"))
    ops.push_back(String("relu"))
    var art = make_compiled_artifact(String("demo"), ops)
    var ok = True
    if not art.has_op(String("conv2d")):
        ok = False
    if art.op_count() != 2:
        ok = False
    if len(art.summary()) == 0:
        ok = False
    return ok