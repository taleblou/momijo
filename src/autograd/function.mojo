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
# File: src/momijo/autograd/function.mojo
#
# Autograd function scaffolding.
# - AutogradContext stores saved tensors for backward.
# - Function is a base-like struct that defines forward/backward contracts.
# - This file avoids external dependencies and dictionaries.
#
# Conventions (Momijo checklist):
# - Only `var` (no `let`), 4-space indentation, explicit imports, no `export`.
# - Constructors: `fn __init__(out self, ...)`.
# - Prefer `mut/out` over `inout`. No exceptions unless declared with `raises`.

from momijo.arrow_core.tensor_bridge import TensorHandle

# -----------------------------
# Autograd Context
# -----------------------------
struct AutogradContext:
    var saved: List[TensorHandle]

    fn __init__(out self):
        self.saved = List[TensorHandle]()

    fn save_for_backward(mut self, tensors: List[TensorHandle]):
        # Store input tensors for use in backward
        var i = 0
        while i < len(tensors):
            self.saved.append(tensors[i])
            i += 1

    fn get_saved(self) -> List[TensorHandle]:
        return self.saved

# -----------------------------
# Function
# -----------------------------
struct Function:
    var name: String
    var ctx: AutogradContext

    fn __init__(out self, name: String):
        self.name = name
        self.ctx = AutogradContext()

    # Forward pass: subclasses should override
    fn apply(mut self, inputs: List[TensorHandle]) -> TensorHandle:
        # TODO: implement op forward
        return inputs[0]  # placeholder: identity

    # Backward pass: subclasses should override
    fn _backward(mut self, out_grad: TensorHandle) -> List[TensorHandle]:
        # TODO: implement op backward (using ctx.saved if needed)
        var grads = List[TensorHandle]()
        grads.append(out_grad)  # placeholder: pass-through grad
        return grads

# -----------------------------
# Free-function wrappers
# -----------------------------
fn AutogradContext_save_for_backward(mut x: AutogradContext, tensors: List[TensorHandle]):
    x.save_for_backward(tensors)

fn AutogradContext_get_saved(x: AutogradContext) -> List[TensorHandle]:
    return x.get_saved()

fn Function_apply(mut f: Function, inputs: List[TensorHandle]) -> TensorHandle:
    return f.apply(inputs)

fn Function_backward(mut f: Function, out_grad: TensorHandle) -> List[TensorHandle]:
    return f._backward(out_grad)

# -----------------------------
# Self-test
# -----------------------------
fn __self_test__() -> Bool:
    var ok = True
    # smoke test: construct a dummy function
    var f = Function("identity")
    var dummy: TensorHandle
    var ins = List[TensorHandle]()
    ins.append(dummy)
    var out = f.apply(ins)
    var grads = f._backward(dummy)
    ok = ok and (len(grads) == 1)
    return ok
