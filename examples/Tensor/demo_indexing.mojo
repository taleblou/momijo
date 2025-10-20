# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_indexing.mojo
#
# Description:
#   Demo for indexing: basic, slicing, boolean masks, gather/scatter, take/put.
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
# 3) Indexing: basic, slicing, boolean, gather/scatter/take/put
# -----------------------------------------------------------------------------
fn demo_indexing() -> None:
    banner("3) INDEXING (BASIC/BOOLEAN/ADVANCED)")

    # Base tensor: shape (3, 4), integer values 1..12
    var x = tensor.arange_int(1, 13, 1).reshape([3, 4])
    print("x:\n" + x.__str__())

    # Basic indexing (using [] instead of get2/row)
    print("x[0,0]: " + x[0, 0].__str__() + " | x[1]: " + x[1].__str__())

    # Slicing: x[:, 1:3]
    var x_slice = x[:, 1:3]
    print("x[:, 1:3]:\n" + x_slice.__str__())

    # Boolean mask: even elements (x is Int, so mod works directly)
    var mask = x.mod_scalar(2).eq_scalar(0)
    print("mask even:\n" + mask.__str__())
    # If bracket-boolean indexing is enabled you can do x[mask]; otherwise use boolean_select:
    print("x[mask] (flattened selection): " + x.boolean_select(mask).__str__())

    # Advanced indexing: gather along dim=1 with per-row indices [[0,2,3], ...]
    var idx = tensor.from_2d_list_int([[0, 2, 3], [0, 2, 3], [0, 2, 3]])
    var gathered = x.gather(1, idx)
    print("gather dim=1 with idx [[0,2,3]]:\n" + gathered.__str__())

    # Scatter: write (gathered + 100) back into zeros_like(x) at the same indices
    var target = tensor.zeros_like(x)
    target = target.scatter(1, idx, gathered.add_scalar(100))
    print("scatter -> add 100 at gathered indices:\n" + target.__str__())

    # Take on flattened order (returns a 1D selection)
    var flat = x.take(tensor.from_list_int([0, 5, 9, 11]))
    print("take [0,5,9,11]: " + flat.__str__())

    # Put: write values into a flattened zeros_like(x), then reshape back
    var y = tensor.zeros_like(x).flatten()
    y = y.put(
        tensor.from_list_int([0, 5, 9, 11]),
        tensor.full_like(y, 9)
    )
    print("put at [0,5,9,11] = 9 ->\n" + y.reshape_like(x).__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_indexing()
