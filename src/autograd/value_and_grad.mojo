# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.autograd
# File: src/momijo/autograd/value_and_grad.mojo
#
# value_and_grad scaffolding.
# - Provides a helper to run a function and compute its gradient w.r.t inputs.
#
# Conventions (Momijo checklist):
# - Only `var` (no `let`), explicit imports, no `export`.
# - Constructors: `fn __init__(out self, ...)`.
# - Prefer `mut/out` over `inout`. No exceptions unless declared with `raises`.

from momijo.autograd.engine import Engine
from momijo.autograd.tape import Tape
from momijo.autograd.variable import Variable
from momijo.arrow_core.tensor_bridge import TensorHandle
from momijo.autograd.grad_registry import GradRegistry

# -----------------------------
# ValueAndGradResult
# -----------------------------
struct ValueAndGradResult:
    var value: Variable
    var grads: List[Variable]

    fn __init__(out self, value: Variable, grads: List[Variable]):
        self.value = value
        self.grads = grads

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
