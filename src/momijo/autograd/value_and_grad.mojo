# Project:      Momijo
# Module:       src.momijo.autograd.value_and_grad
# File:         value_and_grad.mojo
# Path:         src/momijo/autograd/value_and_grad.mojo
#
# Description:  src.momijo.autograd.value_and_grad — focused Momijo functionality with a stable public API.
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
#   - Structs: ValueAndGradResult
#   - Key functions: __init__, __copyinit__, __moveinit__, value_and_grad, __self_test__


from momijo.arrow_core.tensor_bridge import TensorHandle
from momijo.autograd.engine import Engine
from momijo.autograd.grad_registry import GradRegistry
from momijo.autograd.tape import Tape
from momijo.autograd.variable import Variable

struct ValueAndGradResult:
    var value: Variable
    var grads: List[Variable]
fn __init__(out self, value: Variable, grads: List[Variable]) -> None:
        assert(self is not None, String("self is None"))
        self.value() = value
        self.grads = grads
fn __copyinit__(out self, other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
        self.grads = other.grads
fn __moveinit__(out self, deinit other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
        self.grads = other.grads
# -----------------------------
# value_and_grad
# -----------------------------
fn value_and_grad(f: fn(List[Variable]) -> Variable,
                  inputs: List[Variable],
                  registry: GradRegistry) -> ValueAndGradResult:
    # Create a tape to record the computation
    var tape = Tape(True)
    var eng = Engine(registry)

    # Run forward
    var out = f(inputs)

    # Seed grad: assume scalar output, seed with dummy TensorHandle
    var dummy: TensorHandle
    eng.backward(tape, out, dummy)

    # Collect grads
    var grads = List[Variable]()
    var i = 0
    while i < len(inputs):
        grads.append(tape.get_variable(inputs[i].id))
        i += 1

    return ValueAndGradResult(out, grads)

# -----------------------------
# Self-test
# -----------------------------
fn __self_test__() -> Bool:
    var ok = True
    # TODO: integrate with real ops once available
    return ok