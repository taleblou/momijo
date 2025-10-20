# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_views_vs_copies.mojo
#
# Description:
#   Demo for views vs. copies:
#   - Copy (clone) is independent: mutations do not affect the original.
#   - View (slice) shares storage: mutations are reflected in the base.
#   - as_strided: construct a shaped view with explicit strides/offset.
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
# 4) Views vs Copies
# -----------------------------------------------------------------------------
fn demo_views_vs_copies() -> None:
    banner("4) VIEWS VS COPIES")

    # Base tensor: shape (3, 4) as Float64 for consistent printing
    var base = tensor.arange_f64(0, 12, 1).reshape([3, 4])
    print("base (3x4):\n" + base.__str__())

    # ---- Copy branch (independent) ----
    # clone() creates a new tensor with independent storage
    var c = base.clone()
    c.set2(0, 0, 1234.0)  # mutate the copy
    print("clone mutated (independent of base):\n" + c.__str__())
    print("base unchanged after clone mutation:\n" + base.__str__())

    # ---- View branch (shares storage) ----
    # slice(dim=1, start=0, len=2) → view of first two columns: shape (3, 2)
    var v = base.slice(1, 0, 2)
    v.set2(0, 0, 999.0)  # write through the view
    print("view (base[:, 0:2]) after set2(0,0)=999:\n" + v.__str__())
    print("base reflects view write:\n" + base.__str__())

    # ---- as_strided: shape (2, 3) from a 1D base 0..5 ----
    # Shapes/strides here:
    #   target shape: [2, 3]
    #   strides:      [3, 1]  (row-major over the same underlying buffer)
    #   offset:       0
    var base1d = tensor.arange_f64(0, 6, 1)
    var two_by_three = base1d.as_strided([2, 3], [3, 1], 0)
    print("as_strided (2x3) from 0..5:\n" + two_by_three.__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_views_vs_copies()
