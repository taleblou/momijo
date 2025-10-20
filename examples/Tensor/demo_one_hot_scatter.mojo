# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_one_hot_scatter.mojo
#
# Description:
#   Demo for one-hot encoding and scatter_add reductions:
#   - Convert class indices to one-hot rows
#   - Aggregate counts per class with scatter_add
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
# 11) One-hot / scatter_add
# -----------------------------------------------------------------------------
fn demo_one_hot_scatter() -> None:
    banner("11) ONE-HOT / SCATTER_ADD")

    # Class indices (length 4)
    var idx = tensor.from_list_int([0, 2, 1, 2])

    # One-hot encode to num_classes = 4 → shape (4, 4), then cast to Float64
    var oh = idx.one_hot(4).to_float64()
    print("one_hot:\n" + oh.__str__())

    # Compute per-class sums using scatter_add along dim=0.
    # values: 1 for each sample → shape (4,1)
    var values = tensor.ones_f64([4, 1])
    var out = tensor.zeros_f64([4, 1])

    # Indices for dim=0 must match values rows; unsqueeze to shape (4,1)
    out = out.scatter_add(0, idx.unsqueeze(1), values)

    # Squeeze all singleton dims for a compact vector print
    print("scatter_add sum per class:\n" + out.squeeze_all().__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_one_hot_scatter()
