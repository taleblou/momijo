# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_masked_reductions.mojo
#
# Description:
#   Demo for masked reductions (per-class means using boolean masks).
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
# 25) Masked reductions
# -----------------------------------------------------------------------------
fn demo_masked_reductions() -> None:
    banner("25) MASKED REDUCTIONS")

    # Values and class ids
    var x   = tensor.from_list_float64([1.0, 2.0, 3.0, 4.0])  # values
    var cls = tensor.from_list_int([0, 1, 0, 1])               # class ids

    # Build boolean masks per class
    var m0 = cls.eq_scalar(0)
    var m1 = cls.eq_scalar(1)

    # Reduce over masked selections → scalar tensors
    var mean0 = x.masked_select(m0).mean()
    var mean1 = x.masked_select(m1).mean()

    # Stack scalars for a compact print: [mean(class 0), mean(class 1)]
    var per_class = tensor.stack([mean0.copy(), mean1.copy()], 0)
    print("per-class means:\n" + per_class.__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_masked_reductions()
