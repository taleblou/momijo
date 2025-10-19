# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_autograd_jacobian_placeholder.mojo
#
# Description:
#   Jacobian (placeholder) — compare numerical finite-difference and analytic
#   Jacobians for a simple elementwise function f(x) = x^2 + 2x.
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
# 24) Jacobian (placeholder)
# -----------------------------------------------------------------------------
fn demo_autograd_functional() -> None:
    banner("24) AUTOGRAD JACOBIAN (SMALL)")

    # Input vector x ∈ R^3
    var x = tensor.from_list_float64([-1.0, 0.0, 2.0])
    print("x:\n" + x.__str__())

    # f(x) = x^2 + 2x (elementwise)
    print("f(x) = x^2 + 2x:\n" + tensor.f_vec(x).__str__())

    # Numerical (central-difference) vs. analytic Jacobian
    var j_num = tensor.numeric_jacobian(x, 1e-6)
    var j_ana = tensor.analytic_jacobian(x)

    print("Numeric Jacobian (central diff):\n" + j_num.__str__())
    print("Analytic Jacobian (diag(2x+2)):\n" + j_ana.__str__())

    # Max absolute difference between Jacobians
    var diff = (j_num - j_ana).abs()
    print("max |J_num - J_ana|: " + diff.max().__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_autograd_functional()
