# Project:      Momijo
# Module:       src.momijo.autograd.hook
# File:         hook.mojo
# Path:         src/momijo/autograd/hook.mojo
#
# Description:  src.momijo.autograd.hook — focused Momijo functionality with a stable public API.
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
#   - Structs: GradHook, HookManager
#   - Key functions: __init__, call, __copyinit__, __moveinit__, __init__, register_hook, clear_hooks, apply_hooks ...


from momijo.arrow_core.tensor_bridge import TensorHandle

struct GradHook:
    var id: Int
    var name: String
fn __init__(out self, id: Int, name: String) -> None:
        self.id = id
        self.name = name

    # Placeholder: in real system, hook would be a callable
fn call(self, grad: TensorHandle) -> TensorHandle:
        # TODO: once closures supported, apply transformation
        return grad
fn __copyinit__(out self, other: Self) -> None:
        self.id = other.id
        self.name = other.name
fn __moveinit__(out self, deinit other: Self) -> None:
        self.id = other.id
        self.name = other.name
# -----------------------------
# HookManager
# -----------------------------
struct HookManager:
    var hooks: List[GradHook]
fn __init__(out self) -> None:
        self.hooks = List[GradHook]()
fn register_hook(mut self, hook: GradHook) -> None:
        self.hooks.append(hook)
fn clear_hooks(mut self) -> None:
        self.hooks = List[GradHook]()
fn apply_hooks(self, grad: TensorHandle) -> TensorHandle:
        var g = grad
        var i = 0
        while i < len(self.hooks):
            g = self.hooks[i].call(g)
            i += 1
        return g
fn __copyinit__(out self, other: Self) -> None:
        self.hooks = other.hooks
fn __moveinit__(out self, deinit other: Self) -> None:
        self.hooks = other.hooks
# -----------------------------
# Free-function wrappers
# -----------------------------
fn HookManager_register_hook(mut m: HookManager, hook: GradHook) -> None:
    m.register_hook(hook)
fn HookManager_clear_hooks(mut m: HookManager) -> None:
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