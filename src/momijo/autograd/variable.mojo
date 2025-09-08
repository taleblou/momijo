# Project:      Momijo
# Module:       src.momijo.autograd.variable
# File:         variable.mojo
# Path:         src/momijo/autograd/variable.mojo
#
# Description:  src.momijo.autograd.variable — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Structs: Variable
#   - Key functions: __init__, zero_grad, has_grad, __copyinit__, __moveinit__, Variable_zero_grad, Variable_has_grad, __self_test__
#   - Low-level memory (Pointer/UnsafePointer) used; observe safety invariants.


from memory import Pointer
from momijo.arrow_core.tensor_bridge import TensorHandle
from momijo.autograd.hook import call
from momijo.core.config import deterministic
from momijo.core.device import id
from momijo.core.parameter import state
from momijo.core.traits import zero
from momijo.nn.parameter import data, grad
from momijo.tensor.registry import Backend
from pathlib import Path
from pathlib.path import Path

struct Variable:
    var id: Int
    var data: TensorHandle
    var grad: TensorHandle
    var requires_grad: Bool
    var is_leaf: Bool
    var producer: Int
fn __init__(out self,
                id: Int,
                data: TensorHandle,
                requires_grad: Bool,
                is_leaf: Bool,
                producer: Int) -> None:
        self.id = id
        self.data = data
        # Initialize grad as empty handle
        self.grad = TensorHandle(Pointer[UInt8](0), 0)
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.producer = producer
fn zero_grad(mut self) -> None:
        # Replace with a fresh zero-like grad. Backend op needed; placeholder for now.
        if self.grad.nbytes != 0:
            # TODO: call backend to zero in-place
            pass
fn has_grad(self) -> Bool:
        return self.grad.nbytes != 0
fn __copyinit__(out self, other: Self) -> None:
        self.id = other.id
        self.data = other.data
        self.grad = other.grad
        self.requires_grad = other.requires_grad
        self.is_leaf = other.is_leaf
        self.producer = other.producer
fn __moveinit__(out self, deinit other: Self) -> None:
        self.id = other.id
        self.data = other.data
        self.grad = other.grad
        self.requires_grad = other.requires_grad
        self.is_leaf = other.is_leaf
        self.producer = other.producer
# -----------------------------
# Free-function wrappers
# -----------------------------
fn Variable_zero_grad(mut x: Variable) -> None:
    x.zero_grad()
fn Variable_has_grad(x: Variable) -> Bool:
    return x.has_grad()

# -----------------------------
# Self-test
# -----------------------------
fn __self_test__() -> Bool:
    var ok = True
    var dummy: TensorHandle
    var v = Variable(1, dummy, True, True, -1)
    ok = ok and (v.requires_grad)
    return ok