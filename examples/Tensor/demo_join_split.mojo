# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_join_split.mojo
#
# Description:
#   Demo for joining (cat/stack) and splitting (chunk/split_sizes/unbind).
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
# 9) Joining / Splitting
# -----------------------------------------------------------------------------
fn demo_join_split() -> None:
    banner("9) JOIN / SPLIT")

    var x = tensor.arange(0, 6, 1).reshape([2, 3])
    var y = tensor.arange(100, 106, 1).reshape([2, 3])

    # Concatenation along different dimensions
    print("cat dim=0:\n" + tensor.cat([x.copy(), y.copy()], 0).__str__())
    print("cat dim=1:\n" + tensor.cat([x.copy(), y.copy()], 1).__str__())

    # Stacking adds a new dimension
    var stacked = tensor.stack([x.copy(), y.copy()], 0)
    print("stack dim=0 shape: " + stacked.shape().__str__())

    # Chunk a tensor into equal parts along a dimension
    var cat0 = tensor.cat([x.copy(), y.copy()], 0)
    var parts = cat0.chunk(2, 0)
    print("chunk -> two tensors shapes: " + parts[0].shape().__str__() + " " + parts[1].shape().__str__())

    # Split by explicit sizes along a dimension
    var sp = x.split_sizes([1, 1], 0)
    print("split sizes [1,1] dim=0 shapes: " + sp[0].shape().__str__() + " " + sp[1].shape().__str__())

    # Unbind removes a dimension and returns a list of views/slices
    var ub = x.unbind(0)
    print("unbind dim=0 yields:")
    var i = 0
    while i < len(ub):
        print(ub[i].__str__())
        i += 1

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_join_split()
