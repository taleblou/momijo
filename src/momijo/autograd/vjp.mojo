# Project:      Momijo
# Module:       src.momijo.autograd.vjp
# File:         vjp.mojo
# Path:         src/momijo/autograd/vjp.mojo
#
# Description:  src.momijo.autograd.vjp — focused Momijo functionality with a stable public API.
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
#   - Key functions: run_vjp, __self_test__, dummy_vjp
#   - Uses generic functions/types with explicit trait bounds.


from momijo.arrow_core.tensor_bridge import TensorHandle
from momijo.autograd.grad_registry import GradRegistry

fn run_vjp(reg: GradRegistry,
           op: String,
           inputs: List[TensorHandle],
           outputs: List[TensorHandle],
           out_grad: TensorHandle,
           saved: List[TensorHandle]) -> List[(Int, TensorHandle)]:
    # If no registered vjp, propagate same grad to each input
    if not reg.has_vjp(op):
        var in_grads = List[(Int, TensorHandle)]()
        var i = 0
        while i < len(inputs):
            in_grads.append((i, out_grad))
            i += 1
        return in_grads

    var fnc = reg.get_vjp(op)
    var grads = fnc(out_grad, saved, inputs, outputs)

    # Pair grads with input indices
    var paired = List[(Int, TensorHandle)]()
    var j = 0
    while j < len(grads):
        paired.append((j, grads[j]))
        j += 1

    return paired

# -----------------------------
# Self-test
# -----------------------------
fn __self_test__() -> Bool:
    var ok = True

    var reg = GradRegistry()
fn dummy_vjp(g: TensorHandle, s: List[TensorHandle], i: List[TensorHandle], o: List[TensorHandle]) -> List[TensorHandle]:
        return i

    reg.register_vjp("op_id", dummy_vjp)

    var ins = List[TensorHandle]()
    var outs = List[TensorHandle]()
    var dummy: TensorHandle

    var res = run_vjp(reg, "op_id", ins, outs, dummy, List[TensorHandle]())
    ok = ok and (len(res) == len(ins))

    return ok