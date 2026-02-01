# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_arith_logic_broadcast.mojo
#
# Description:
#   Demo for arithmetic, comparisons, logical/bitwise ops, and broadcasting.
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
# 6) Arithmetic, Comparisons, Logical/Bitwise, Broadcasting
# -----------------------------------------------------------------------------
fn demo_arith_logic_broadcast() -> None:
    banner("6) ARITH / LOGICAL / BROADCAST")

    # Random integer tensors for arithmetic tests
    var x = tensor.randn_int([2, 2])
    var y = tensor.randn_int([2, 2])
    print("x:\n" + x.__str__())
    print("y:\n" + y.__str__())

    # Basic arithmetic (elementwise)
    print(
        "add / sub / mul / div / mod:\n"
        + (x + y).__str__() + "\n"
        + (x - y).__str__() + "\n"
        + (x * y).__str__() + "\n"
        + (x / y).__str__() + "\n"
        + (x % y).__str__() + "\n"
    )

    # Bitwise operators (elementwise on integer tensors)
    print(
        "bitwise ^, &, |, ~:\n"
        + (x ^ y).__str__() + "\n"
        + (x & y).__str__() + "\n"
        + (x | y).__str__() + "\n"
        + (~x).__str__() + "\n"
        + (~y).__str__() + "\n"
    )

    # Unary math (use absolute to keep sqrt/log safe for integers)
    print(
        "pow / exp / log / sqrt:\n"
        + x.pow_scalar(2.0).__str__() + "\n"
        + x.exp().__str__() + "\n"
        + x.abs().add_scalar(1e-6).log().__str__() + "\n"
        + x.abs().sqrt().__str__()
    )

    # Comparisons and boolean reductions
    var mask_pos = x.gt_scalar(0.0)
    var any_b = x.le_scalar(0.0).any()
    var all_b = mask_pos.all()

    var any_str: String = "false"
    if any_b:
        any_str = "true"

    var all_str: String = "false"
    if all_b:
        all_str = "true"

    print("compare (>, <=), all/any:\n" + mask_pos.__str__() + "\n" + any_str + " " + all_str)

    # Bitwise ops via explicit methods on integer 1D tensors
    var bi = tensor.from_list_int([1, 2, 3])
    var bj = tensor.from_list_int([3, 1, 1])
    print(
        "bitwise AND/OR/XOR (methods):\n"
        + bi.and_bitwise(bj).__str__() + "\n"
        + bj.or_bitwise(bi).__str__() + "\n"
        + bi.xor_bitwise(bj).__str__()
    )

    # Logical ops on boolean tensors
    print("logical and / or / xor:")
    var mx = x.gt_scalar(0.0)
    var my = y.gt_scalar(0.0)
    print(mx.logical_and(my).__str__())
    print(mx.logical_or(my).__str__())
    print(mx.logical_xor(my).__str__())

    # Broadcasting example with where(): keep positives from x, otherwise 0
    var zeros = tensor.zeros_like(x)
    print("where (x if >0 else 0):\n" + tensor.where(mask_pos, x, zeros).__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_arith_logic_broadcast()
