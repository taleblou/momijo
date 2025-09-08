# Project:      Momijo
# Module:       src.momijo.autograd.function
# File:         function.mojo
# Path:         src/momijo/autograd/function.mojo
#
# Description:  src.momijo.autograd.function — focused Momijo functionality with a stable public API.
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
#   - Structs: AutogradContext, Function
#   - Key functions: __init__, save_for_backward, get_saved, __copyinit__, __moveinit__, __init__, apply, _backward ...


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