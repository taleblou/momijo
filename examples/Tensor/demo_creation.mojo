# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_creation.mojo
#
# Description:
#   Demo for Tensor creation utilities and basic factory functions.
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
# 1) Creation & Basics
# -----------------------------------------------------------------------------
fn demo_creation() -> None:
    banner("1) CREATION & BASICS")

    # Basic factories (covering several dtypes and shapes)
    var a = tensor.Tensor[Float64](rows=[[1.0, 2.0], [3.0, 4.0]])  # 2x2 Float64
    var b = tensor.zeros_int([2, 3])                               # 2x3 Int zeros
    var c = tensor.ones_int([3, 1])                                # 3x1 Int ones
    var d = tensor.arange_int(0, 10, 2)                            # Int: 0,2,4,6,8
    var e = tensor.linspace_f64(0.0, 1.0, 5)                       # Float64: 5 steps
    var f = tensor.eye_int(3)                                      # 3x3 Int identity
    var g = tensor.full([2, 2], 7.0)                               # 2x2 Float64 filled with 7.0
    var h = tensor.randn_int([2, 3])                               # 2x3 Int ~ N(0,1) (discrete)
    var i = tensor.rand([2, 3])                                    # 2x3 Float64 uniform [0,1)
    var j = tensor.empty([2, 2])                                   # 2x2 uninitialized (dtype default)
    var k = tensor.zeros_like(a)                                   # same shape/dtype as `a`

    # Prints
    print("a:\n" + a.__str__())
    print("b:\n" + b.__str__())
    print("c:\n" + c.__str__())
    print("d: " + d.__str__())
    print("e: " + e.__str__())
    print("f:\n" + f.__str__())
    print("g:\n" + g.__str__())
    print("h:\n" + h.__str__())
    print("i:\n" + i.__str__())
    print("j (empty):\n" + j.__str__())
    print("k (zeros_like a):\n" + k.__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_creation()
