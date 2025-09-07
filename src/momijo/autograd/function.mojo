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
# File: src/momijo/autograd/function.mojo

from momijo.arrow_core.tensor_bridge import TensorHandle

struct AutogradContext:
    var saved: List[TensorHandle]
fn __init__(out self) -> None:
        self.saved = List[TensorHandle]()
fn save_for_backward(mut self, tensors: List[TensorHandle]) -> None:
        # Store input tensors for use in backward
        var i = 0
        while i < len(tensors):
            self.saved.append(tensors[i])
            i += 1
fn get_saved(self) -> List[TensorHandle]:
        return self.saved
fn __copyinit__(out self, other: Self) -> None:
        self.saved = other.saved
fn __moveinit__(out self, deinit other: Self) -> None:
        self.saved = other.saved
# -----------------------------
# Function
# -----------------------------
struct Function:
    var name: String
    var ctx: AutogradContext
fn __init__(out self, name: String) -> None:
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
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        self.ctx = other.ctx
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        self.ctx = other.ctx
# -----------------------------
# Free-function wrappers
# -----------------------------
fn AutogradContext_save_for_backward(mut x: AutogradContext, tensors: List[TensorHandle]) -> None:
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