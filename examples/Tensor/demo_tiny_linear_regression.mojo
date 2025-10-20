# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_tiny_linear_regression.mojo
#
# Description:
#   Tiny linear regression with gradient descent (no autograd).
#   - Generates synthetic data y = 2x + 1 + noise
#   - Optimizes parameters (w, b) via closed-form gradients
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
# 17) Tiny linear regression (GD) — clean
# -----------------------------------------------------------------------------
fn demo_tiny_linear_regression(steps: Int = 200, lr: Float64 = 0.05) -> None:
    banner("17) TINY LINEAR REGRESSION")

    # Synthetic data: y = 2x + 1 + noise
    var n = 100
    var x = tensor.linspace_f64(-1.0, 1.0, n).unsqueeze(1)      # (n, 1)
    var true_w = tensor.from_list_float64([[2.0]])               # (1, 1)
    var true_b = tensor.from_list_float64([1.0])                 # (1,)
    var y = (x.matmul(true_w)).view([n, 1]) + true_b.view([1, 1])# (n, 1)
    y = y + tensor.randn_f64([n, 1]).mul_scalar(0.1)             # add noise

    # Parameters to learn
    var w = tensor.randn_f64([1, 1])                             # (1, 1)
    var b = tensor.zeros_f64([1])                                # (1,)

    var t = 0
    while t < steps:
        # Forward
        var y_hat = x.matmul(w) + b.view([1, 1])                 # (n, 1)
        var diff  = y_hat - y                                    # (n, 1)
        var loss  = (diff * diff).mean()                         # scalar

        # Gradients (no autograd):
        # d/dw = 2/N * X^T (Xw + b - y), d/db = 2/N * sum(Xw + b - y)
        var grad_w = x.transpose([1, 0]).matmul(diff).mul_scalar(2.0 / Float64(n))  # (1, 1)
        var grad_b = diff.mean().mul_scalar(2.0)                                          # scalar (rank-0)
        var grad_b_vec = grad_b.view([1])                                                 # (1,)

        # Update
        w = w - grad_w.mul_scalar(lr)
        b = b - grad_b_vec.mul_scalar(lr)

        # Light logging (quarterly)
        var interval = steps // 4
        if interval < 1:
            interval = 1
        if ((t + 1) % interval) == 0:
            print(
                "step " + String(t + 1)
                + ": loss=" + loss.__str__()
                + ", w=" + w.__str__()
                + ", b=" + b.__str__()
            )

        t = t + 1

    print("Final params: w≈" + w.__str__() + " b≈" + b.__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_tiny_linear_regression()
