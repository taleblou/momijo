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
# File: src/momijo/autograd/jvp.mojo
#
# JVP scaffolding.
# - Forward-mode automatic differentiation primitives.
# - Uses GradRegistry to retrieve registered JVP implementations.
#
# Conventions (Momijo checklist):
# - Only `var` (no `let`), explicit imports, no `export`.
# - Constructors: `fn __init__(out self, ...)`.
# - Prefer `mut/out` over `inout`. No exceptions unless declared with `raises`.

from momijo.autograd.grad_registry import GradRegistry
from momijo.arrow_core.tensor_bridge import TensorHandle

# -----------------------------
# JVPResult
# -----------------------------
struct JVPResult:
    var primals: List[TensorHandle]
    var tangents: List[TensorHandle]

    fn __init__(out self, primals: List[TensorHandle], tangents: List[TensorHandle]):
        self.primals = primals
        self.tangents = tangents

# -----------------------------
# Run JVP
# -----------------------------
fn run_jvp(reg: GradRegistry,
           op: String,
           primals: List[TensorHandle],
           tangents: List[TensorHandle],
           saved: List[TensorHandle]) -> JVPResult:
    if not reg.has_jvp(op):
        # Default: forward tangents unchanged (identity)
        return JVPResult(primals, tangents)
    var fnc = reg.get_jvp(op)
    var out_primals, out_tangents = fnc(primals, tangents, saved)
    return JVPResult(out_primals, out_tangents)

# -----------------------------
# Self-test
# -----------------------------
fn __self_test__() -> Bool:
    var ok = True

    # Dummy registry
    var reg = GradRegistry()

    fn dummy_jvp(p: List[TensorHandle], t: List[TensorHandle], s: List[TensorHandle]) -> (List[TensorHandle], List[TensorHandle]):
        return (p, t)

    reg.register_jvp("op_id", dummy_jvp)

    var primals = List[TensorHandle]()
    var tangents = List[TensorHandle]()
    var saved = List[TensorHandle]()

    var res = run_jvp(reg, "op_id", primals, tangents, saved)
    ok = ok and (len(res.primals) == 0) and (len(res.tangents) == 0)

    return ok
