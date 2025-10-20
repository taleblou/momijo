# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_masks_fill_clamp.mojo
#
# Description:
#   Demo for masked_select, masked_fill, and clamp on small tensors.
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
# 14) Masks / Fill / Clamp
# -----------------------------------------------------------------------------
fn demo_masks_fill_clamp() -> None:
    banner("14) MASKS / FILL / CLAMP")

    var x = tensor.randn_int([2, 3])
    print("x:\n" + x.__str__())

    # Build a boolean mask for positives (x > 0)
    var mask = x.gt_scalar(0.0)
    print("masked_select (>0): " + x.masked_select(mask).__str__())

    # Dtype diagnostics
    print("dtypes -> x: " + x.dtype_name() + " | mask: " + mask.dtype_name())

    # masked_fill: set negatives to 0 by inverting the positive mask
    var neg_mask = mask.not_bitwise()
    var y = x.copy()
    y.masked_fill(neg_mask, 0)
    print("masked_fill negatives to 0:\n" + y.__str__())

    # Clamp to [-0.5, 0.5]
    # Note: Int → Float promotion is expected for float bounds.
    var xc = x.clamped(-0.5, 0.5)
    print("clamp to [-0.5, 0.5]:\n" + xc.__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_masks_fill_clamp()
