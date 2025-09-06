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
# File: src/momijo/autograd/variable.mojo
#
# Variable struct for autograd.
# - Wraps a tensor handle with grad metadata.
# - Stores variable id, data, grad, flags, and producer node id.
#
# Conventions (Momijo checklist):
# - Only `var` (no `let`), explicit imports, no `export`.
# - Constructors: `fn __init__(out self, ...)`.
# - Prefer `mut/out` over `inout`. No exceptions unless declared with `raises`.

from momijo.arrow_core.tensor_bridge import TensorHandle

# -----------------------------
# Variable
# -----------------------------
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
                producer: Int):
        self.id = id
        self.data = data
        # Initialize grad as empty handle
        self.grad = TensorHandle(Pointer[UInt8](0), 0)
        self.requires_grad = requires_grad
        self.is_leaf = is_leaf
        self.producer = producer

    fn zero_grad(mut self):
        # Replace with a fresh zero-like grad. Backend op needed; placeholder for now.
        if self.grad.nbytes != 0:
            # TODO: call backend to zero in-place
            pass

    fn has_grad(self) -> Bool:
        return self.grad.nbytes != 0

# -----------------------------
# Free-function wrappers
# -----------------------------
fn Variable_zero_grad(mut x: Variable):
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
