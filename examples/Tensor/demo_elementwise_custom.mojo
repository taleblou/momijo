# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_elementwise_custom.mojo
#
# Description:
#   Demo for elementwise custom ops (map-style). Implements the swish
#   activation: swish(x) = x * sigmoid(x) over a small Float64 range.
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
# 22) Elementwise custom (map-style)
# -----------------------------------------------------------------------------
fn demo_elementwise_custom() -> None:
    banner("22) ELEMENTWISE CUSTOM (map-style)")

    # Input values from -2 to 2 (5 steps), Float64
    var x = tensor.linspace_f64(-2.0, 2.0, 5)

    # Swish activation: swish(x) = x * sigmoid(x), applied elementwise
    var swish = x * x.sigmoid()

    print("x:\n" + x.__str__())
    print("swish(x):\n" + swish.__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_elementwise_custom()
