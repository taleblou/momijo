# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_broadcast_align.mojo
#
# Description:
#   Demo for broadcast alignment using unsqueeze/reshape to add a per-channel
#   bias to a 4D tensor (N, C, H, W).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# Small banner printer
# -----------------------------------------------------------------------------
fn banner(title: String) -> None:
    print("\n=== " + title + " ===")

# -----------------------------------------------------------------------------
# 21) Broadcast align with unsqueeze
# -----------------------------------------------------------------------------
fn demo_broadcast_align() -> None:
    banner("21) BROADCAST ALIGN WITH UNSQUEEZE")

    var n = 2
    var c = 3
    var h = 4
    var w = 4

    # Input with shape (N, C, H, W)
    var x = tensor.randn_int([n, c, h, w])

    # Bias with shape (C,)
    var bias = tensor.randn_int([c])

    # Reshape bias to (1, C, 1, 1) for broadcasting over (N, C, H, W)
    # (Use reshape instead of view to avoid dependency on view semantics.)
    var bias_bc = bias.reshape([1, c, 1, 1])

    # Broadcast add
    var y = x + bias_bc

    # Shapes
    print("x shape: " + x.shape().__str__())
    print("bias shape: " + bias.shape().__str__())
    print("broadcast add bias -> y shape: " + y.shape().__str__())

    # Samples
    print("x:\n" + x.__str__())
    print("bias:\n" + bias.__str__())
    print("broadcast add bias -> y:\n" + y.__str__())

    # Consistency checks:
    # diff = y - x should equal bias (broadcasted) everywhere.
    var diff = y - x

    # chk1: Reduce (mean) over N,H,W → compare to bias
    # Note: assumes mean(axis=[0,2,3]) is supported.
    var chk1 = (diff.mean(axis=[0, 2, 3]) - bias).abs().max()
    print("chk1 (mean over N,H,W vs bias) max abs diff: " + chk1.__str__())

    # chk2: Direct broadcast comparison
    var chk2 = (diff - bias_bc).abs().max()
    print("chk2 (diff vs bias broadcast) max abs diff: " + chk2.__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_broadcast_align()
