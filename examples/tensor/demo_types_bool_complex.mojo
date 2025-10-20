# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_types_bool_complex.mojo
#
# Description:
#   Demo for:
#   - Type promotion (Int → Float during mixed ops)
#   - Boolean tensor operations
#   - Complex number utilities (construct, abs/real/imag)
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
# 15) Type promotion / bool / complex
# -----------------------------------------------------------------------------
fn demo_types_bool_complex() -> None:
    banner("15) TYPE PROMOTION / BOOL / COMPLEX")

    # ----- Type promotion -----
    var a = tensor.from_list_int([1, 2, 3])
    var b = tensor.from_list_float64([0.5, 0.25, 0.75])
    var c = a.to_float64() + b
    print("type promotion int + float64 -> " + c.dtype_name() + ": " + c.__str__())

    # ----- Boolean ops -----
    var m = tensor.from_list_bool([True, False, True])
    print("bool logical_not: " + m.logical_not().__str__())

    # ----- Complex numbers -----
    # Construct complex vector z = zr + i*zi
    var zr = tensor.from_list_float64([1.0, 3.0])
    var zi = tensor.from_list_float64([2.0, -4.0])
    var z = tensor.complex(zr, zi)

    print(
        "complex abs / real / imag:\n"
        + tensor.complex_abs(z).__str__()  + "\n"
        + tensor.complex_real(z).__str__() + "\n"
        + tensor.complex_imag(z).__str__()
    )

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_types_bool_complex()
