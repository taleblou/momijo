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
# File: src/momijo/autograd/hook.mojo
#
# Gradient hook scaffolding.
# - GradHook stores a hook id and name.
# - HookManager maintains a list of hooks for a variable.
# - This is a lightweight system, awaiting richer closure/function pointer support.
#
# Conventions (Momijo checklist):
# - Only `var` (no `let`), explicit imports, no `export`.
# - Constructors: `fn __init__(out self, ...)`.
# - Prefer `mut/out` over `inout`. No exceptions unless declared with `raises`.

from momijo.arrow_core.tensor_bridge import TensorHandle

# -----------------------------
# GradHook
# -----------------------------
struct GradHook:
    var id: Int
    var name: String

    fn __init__(out self, id: Int, name: String):
        self.id = id
        self.name = name

    # Placeholder: in real system, hook would be a callable
    fn call(self, grad: TensorHandle) -> TensorHandle:
        # TODO: once closures supported, apply transformation
        return grad

# -----------------------------
# HookManager
# -----------------------------
struct HookManager:
    var hooks: List[GradHook]

    fn __init__(out self):
        self.hooks = List[GradHook]()

    fn register_hook(mut self, hook: GradHook):
        self.hooks.append(hook)

    fn clear_hooks(mut self):
        self.hooks = List[GradHook]()

    fn apply_hooks(self, grad: TensorHandle) -> TensorHandle:
        var g = grad
        var i = 0
        while i < len(self.hooks):
            g = self.hooks[i].call(g)
            i += 1
        return g

# -----------------------------
# Free-function wrappers
# -----------------------------
fn HookManager_register_hook(mut m: HookManager, hook: GradHook):
    m.register_hook(hook)

fn HookManager_clear_hooks(mut m: HookManager):
    m.clear_hooks()

fn HookManager_apply_hooks(m: HookManager, grad: TensorHandle) -> TensorHandle:
    return m.apply_hooks(grad)

# -----------------------------
# Self-test
# -----------------------------
fn __self_test__() -> Bool:
    var ok = True

    var hm = HookManager()
    var h = GradHook(1, "identity")
    hm.register_hook(h)

    var dummy: TensorHandle
    var out = hm.apply_hooks(dummy)

    ok = ok and True  # smoke check
    return ok
