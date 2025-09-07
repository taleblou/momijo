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
# File: src/momijo/autograd/grad_registry.mojo

from momijo.arrow_core.tensor_bridge import TensorHandle


# JVP: (primals, tangents, saved) -> (out_primals, out_tangents)


# -----------------------------
# GradRegistry
# -----------------------------
struct GradRegistry:
    var vjp_map: Dict[String, VJPFn]
    var jvp_map: Dict[String, JVPFn]
fn __init__(out self) -> None:
        self.vjp_map = Dict[String, VJPFn]()
        self.jvp_map = Dict[String, JVPFn]()
fn has_vjp(self, op: String) -> Bool:
        return self.vjp_map.contains(op)
fn has_jvp(self, op: String) -> Bool:
        return self.jvp_map.contains(op)
fn get_vjp(self, op: String) -> VJPFn:
        return self.vjp_map[op]
fn get_jvp(self, op: String) -> JVPFn:
        return self.jvp_map[op]
fn register_vjp(mut self, op: String, fnc: VJPFn) -> None:
        self.vjp_map[op] = fnc
fn register_jvp(mut self, op: String, fnc: JVPFn) -> None:
        self.jvp_map[op] = fnc
fn __copyinit__(out self, other: Self) -> None:
        self.vjp_map = other.vjp_map
        self.jvp_map = other.jvp_map
fn __moveinit__(out self, deinit other: Self) -> None:
        self.vjp_map = other.vjp_map
        self.jvp_map = other.jvp_map
# -----------------------------
# Free-function wrappers
# -----------------------------
fn GradRegistry_has_vjp(x: GradRegistry, op: String) -> Bool:
    return x.has_vjp(op)
fn GradRegistry_has_jvp(x: GradRegistry, op: String) -> Bool:
    return x.has_jvp(op)
fn GradRegistry_get_vjp(x: GradRegistry, op: String) -> VJPFn:
    return x.get_vjp(op)
fn GradRegistry_get_jvp(x: GradRegistry, op: String) -> JVPFn:
    return x.get_jvp(op)
fn GradRegistry_register_vjp(mut x: GradRegistry, op: String, fnc: VJPFn) -> None:
    x.register_vjp(op, fnc)
fn GradRegistry_register_jvp(mut x: GradRegistry, op: String, fnc: JVPFn) -> None:
    x.register_jvp(op, fnc)

# -----------------------------
# Self-test
# -----------------------------
fn __self_test__() -> Bool:
    var ok = True

    # define dummy vjp/jvp
fn dummy_vjp(g: TensorHandle, s: List[TensorHandle], i: List[TensorHandle], o: List[TensorHandle]) -> List[TensorHandle]:
        return i
fn dummy_jvp(p: List[TensorHandle], t: List[TensorHandle], s: List[TensorHandle]) -> (List[TensorHandle], List[TensorHandle]):
        return (p, t)

    var reg = GradRegistry()
    reg.register_vjp("op_add", dummy_vjp)
    reg.register_jvp("op_add", dummy_jvp)

    ok = ok and reg.has_vjp("op_add")
    ok = ok and reg.has_jvp("op_add")
    return ok