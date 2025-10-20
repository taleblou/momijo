# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_numerical_stability.mojo
#
# Description:
#   Demo for numerical stability tricks:
#   - Softmax with the max-subtraction trick
#   - (Optional) log-sum-exp patterns
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
# 16) Numerical stability
# -----------------------------------------------------------------------------
fn demo_numerical_stability() -> None:
    banner("16) NUMERICAL STABILITY")

    var x = tensor.from_list_float64([100.0, 1.0, -100.0])
    print("x:\n" + x.__str__())

    # Softmax with max-subtraction trick:
    # softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    print("softmax with max-sub trick")
    var m = x.max()                         # scalar max
    var e = x.sub_scalar(m).exp()           # exp(x - max(x))
    var denom = e.sum()                     # scalar denominator
    var soft = e.divide(denom)              # elementwise division by scalar

    print("max: " + m.__str__())
    print("exp(x - max(x)): " + e.__str__())
    print("denom: " + denom.__str__())
    print("stable softmax:\n" + soft.__str__())

    # Optional: stable log-sum-exp (pattern)
    # LSE(x) = log( sum_i exp(x_i) ) = m + log( sum_i exp(x_i - m) )
    # Uncomment when tensor.log() / sum over axis are available for 1D:
    #
    # var lse = m.add( e.sum().log() )
    # print("log-sum-exp (stable): " + lse.__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_numerical_stability()
