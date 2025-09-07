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
# File: src/momijo/autograd/value_and_grad.mojo

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