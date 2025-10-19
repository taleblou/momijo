# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_autograd_placeholder.mojo
#
# Description:
#   Autograd basics (placeholder) — demonstrates a simple forward computation
#   and the corresponding analytic gradient, without requiring a gradient engine.
#   - English-only comments
#   - Explicit imports (no wildcards)
#   - Var-only, no asserts
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
# 12) Autograd basics (placeholder)
# -----------------------------------------------------------------------------
fn demo_autograd() -> None:
    banner("12) AUTOGRAD BASICS (PLACEHOLDER)")

    # Example: y = sum(x^2)
    var x = tensor.arange_int(0, 6, 1).reshape([2, 3])
    print("x:\n" + x.__str__())

    # Forward pass: y is a scalar (sum over all elements)
    # If your API supports Optional axis, sum(None) reduces all dims.
    var y = x.mul(x).sum(None)
    print("y = sum(x^2): " + y.__str__())

    # Analytic gradient for y = sum(x^2) is dy/dx = 2x (elementwise)
    var grad_expected = x.mul_scalar(2.0)
    print("Expected grad dy/dx = 2x:\n" + grad_expected.__str__())

 

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_autograd()
