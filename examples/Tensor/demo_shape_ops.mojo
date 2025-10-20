# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_shape_ops.mojo
#
# Description:
#   Demo for common shape transforms:
#   - reshape / view (inference with -1)
#   - permute / transpose
#   - squeeze / unsqueeze
#   - flatten (via reshape)
#   - repeat / tile
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
# 5) Reshape, Permute/Transpose, Squeeze/Unsqueeze, Flatten, Repeat/Tile
# -----------------------------------------------------------------------------
fn demo_shape_ops() -> None:
    banner("5) SHAPE OPS")

    var x = tensor.arange(0, 24, 1).reshape([2, 3, 4])
    print("x shape: " + x.shape().__str__())

    # Reshape: change logical shape (row-major)
    print("reshape(3, 8): " + x.reshape([3, 8]).shape().__str__())

    # View: reshape-like, using -1 to infer size (relies on view semantics)
    print("view(-1, 4): " + x.view([-1, 4]).shape().__str__())

    # Permute / transpose (dimension reordering)
    print("permute(0, 2, 1): " + x.permute([0, 2, 1]).shape().__str__())
    print("transpose(1, 2): " + x.transpose(1, 2).shape().__str__())

    # Unsqueeze / squeeze (add/remove singleton dims)
    var x_unsq = x.unsqueeze(0)    # [1, 2, 3, 4]
    print("unsqueeze(0): " + x_unsq.shape().__str__())
    var x_squeezed = x_unsq.squeeze(0)
    print("squeeze(0) back: " + x_squeezed.shape().__str__())

    # Flatten (via reshape) — collapse to 1D
    # (Use reshape([-1]) for maximum portability.)
    print("flatten via reshape(-1): " + x.reshape([-1]).shape().__str__())

    # Repeat / tile:
    # take the first slice along dim=1 (shape [2,1,4]) then tile along that dim 3x → [2,3,4]
    print("repeat(tile) from first slice dim=1 → [2,3,4]: "
          + x.slice(1, 0, 1).repeat([1, 3, 1]).shape().__str__())

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_shape_ops()
