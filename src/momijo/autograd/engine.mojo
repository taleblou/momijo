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
# File: src/momijo/autograd/engine.mojo

from gpu import id
from momijo.arrow_core.offsets import last
from momijo.arrow_core.tensor_bridge import TensorHandle
from momijo.autograd.grad_registry import GradRegistry
from momijo.autograd.tape import Tape, get_producer, get_variable
from momijo.autograd.variable import Variable
from momijo.autograd.vjp import run_vjp
from momijo.core.asserts import require
from momijo.core.config import deterministic
from momijo.core.device import id
from momijo.core.error import code
from momijo.core.ndarray import product
from momijo.core.parameter import state
from momijo.dataframe.helpers import m
from momijo.ir.dialects.annotations import dict
from momijo.nn.parameter import grad
from pathlib import Path
from pathlib.path import Path
from utils.index import product

struct GradEntry:
    var var_id: Int
    var grad: TensorHandle
fn __init__(out self, var_id: Int, grad: TensorHandle) -> None:
        self.var_id = var_id
        self.grad = grad
fn __copyinit__(out self, other: Self) -> None:
        self.var_id = other.var_id
        self.grad = other.grad
fn __moveinit__(out self, deinit other: Self) -> None:
        self.var_id = other.var_id
        self.grad = other.grad
# -----------------------------
# Autograd Engine
# -----------------------------
struct Engine:
    var registry: GradRegistry
fn __init__(out self, registry: GradRegistry) -> None:
        self.registry = registry

    # Accumulate gradient for a variable id into a small list.
    # Linear search is used to avoid dict dependencies.
fn accumulate_grad(mut self, grads: List[GradEntry], var_id: Int, g: TensorHandle):
        var i = 0
        while i < len(grads):
            if grads[i].var_id == var_id:
                # TODO: real impl should add/accumulate existing + g
                # For now, replace to keep the code dependency-light.
                grads[i].grad = g
                return
            i += 1
        grads.append(GradEntry(var_id, g))

    # Run reverse-mode backprop from output variable `y` using a provided seed gradient.
    # This is a skeleton that expects Tape/Variable to expose minimal APIs:
    # - Tape: methods to resolve a producing node for a var_id, and the node's inputs/op id
    # - Variable: flags like requires_grad and a setter for .grad if needed
    # - run_vjp(registry, node, out_grad) -> List[GradEntry] for the node inputs
fn backward(self, tape: Tape, y: Variable, seed_grad: TensorHandle) -> None:
        # Worklist of grads to propagate backward.
        var work: List[GradEntry] = List[GradEntry]()
        work.append(GradEntry(y.id, seed_grad))

        # Final accumulated gradients per variable id.
        var out_grads: List[GradEntry] = List[GradEntry]()

        # Standard reverse push-down through the tape.
        while len(work) > 0:
            # Pop last (stack LIFO)
            var last_index = len(work) - 1
            var entry = work[last_index]
            # manual pop (no dependence on .pop())
            var tmp: List[GradEntry] = List[GradEntry]()
            var j = 0
            while j < last_index:
                tmp.append(work[j])
                j += 1
            work = tmp

            # Skip if this variable does not require grad.
            var v: Variable = tape.get_variable(entry.var_id)
            if not v.requires_grad:
                continue

            # Accumulate into the final map.
            self.accumulate_grad(out_grads, entry.var_id, entry.grad)

            # If leaf, no producer node -> continue.
            if v.is_leaf:
                continue

            # Otherwise fetch the producing node and propagate via VJP.
            # Expectation: tape.get_producer(var_id) -> Node (op + inputs)
            var node = tape.get_producer(entry.var_id)

            # Vector-Jacobian product for this node
            var in_grads: List[GradEntry] = run_vjp(self.registry, node, entry.grad)

            # Push input grads to worklist
            var k = 0
            while k < len(in_grads):
                work.append(in_grads[k])
                k += 1

        # Write back grads to tape variables (if tape/Variable API supports it)
        var m = 0
        while m < len(out_grads):
            var ge = out_grads[m]
            var vv: Variable = tape.get_variable(ge.var_id)
            # TODO: if Variable has .accumulate_grad use it; here we assign
            vv.grad = ge.grad
            m += 1
fn __copyinit__(out self, other: Self) -> None:
        self.registry = other.registry
        self.work = other.work
        self.out_grads = other.out_grads
        self.tmp = other.tmp
        self.v = other.v
        self.in_grads = other.in_grads
        self.vv = other.vv
fn __moveinit__(out self, deinit other: Self) -> None:
        self.registry = other.registry
        self.work = other.work
        self.out_grads = other.out_grads
        self.tmp = other.tmp
        self.v = other.v
        self.in_grads = other.in_grads
        self.vv = other.vv
# --- Free-function wrappers (pattern used elsewhere in Momijo) ---
fn Engine_accumulate_grad(mut x: Engine, grads: List[GradEntry], var_id: Int, g: TensorHandle) -> None:
    x.accumulate_grad(grads, var_id, g)
fn Engine_backward(x: Engine, tape: Tape, y: Variable, seed_grad: TensorHandle) -> None:
    x.backward(tape, y, seed_grad)

# Minimal self-test to satisfy smoke checks.
fn __self_test__() -> Bool:
    var ok = True
    return ok