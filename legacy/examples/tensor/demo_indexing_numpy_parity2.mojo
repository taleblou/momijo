# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_indexing_numpy_parity.mojo
#
# Description:
#   Parity demo for indexing/slicing vs. NumPy examples (1D..5D).
#   - English-only comments
#   - Explicit imports (no wildcards)
#   - String-only printing
#   - Helper overloads for typed Tensor printing

from momijo.tensor import tensor
from collections.list import List

# -----------------------------------------------------------------------------
# Small banner printer
# -----------------------------------------------------------------------------
fn banner(title: String) -> None:
    print("\n=== " + title + " ===")

# -----------------------------------------------------------------------------
# Print helpers (only String/scalars allowed)
# -----------------------------------------------------------------------------
fn show_tensor(label: String, x: tensor.Tensor[Int]) -> None:
    print(label + ": " + x.__str__())

fn show_tensor(label: String, x: tensor.Tensor[Float32]) -> None:
    print(label + ": " + x.__str__())

fn show_tensor(label: String, x: tensor.Tensor[Float64]) -> None:
    print(label + ": " + x.__str__())

# -----------------------------------------------------------------------------
# Main demo
# -----------------------------------------------------------------------------
fn demo_indexing_numpy_parity() -> None:
    # =========================
    # 1) 1D examples
    # =========================
    banner("1D")
    var arr1 = tensor.arange(10, 50, 10)  # [10, 20, 30, 40]
    show_tensor("arr1", arr1)

    show_tensor("arr1[1] ->", arr1[1])
    show_tensor("""arr1["1:3"]""", arr1["1:3"])
    show_tensor("""arr1[":"]""", arr1[":"])
    show_tensor("arr1[-1] ->", arr1[-1])
    show_tensor("""arr1["::2"]""", arr1["::2"])
 
    show_tensor("arr1[[0,2]]", arr1[0,2])

    # =========================
    # 2) 2D examples
    # =========================
    banner("2D")
    var arr2 = tensor.arange(1, 10, 1).reshape([3, 3])
    show_tensor("arr2", arr2)

    show_tensor("arr2[1]", arr2[1])
    show_tensor("arr2[1,2] ->", arr2[1,2])
    show_tensor("arr2[1,2]", arr2[1,2])
    show_tensor("""arr2[":,1"]""", arr2[":,1"])
    show_tensor("""arr2["1,:"]""", arr2["1,:"])
    show_tensor("""arr2[":"]""", arr2[":"])
    show_tensor("""arr2[":, :"]""", arr2[":, :"])
    show_tensor("""arr2["::2,1:"]""", arr2["::2,1:"])

    # =========================
    # 3) 3D examples
    # =========================
    banner("3D")
    var arr3 = tensor.arange(0, 24, 1).reshape([2, 3, 4])
    show_tensor("arr3", arr3)
    show_tensor("arr3[0]", arr3[0])
    show_tensor("arr3[1,2]", arr3[1,2])
    show_tensor("""arr3[":,1,:"]""", arr3[":,1,:"])
    show_tensor("""arr3[":,:,0"]""", arr3[":,:,0"])
    show_tensor("""arr3["0,:,1"]""", arr3["0,:,1"])
    show_tensor("""arr3[":"]""", arr3[":"])

    # =========================
    # 4) 4D examples
    # =========================
    banner("4D")
    var arr4 = tensor.arange(0, 48, 1).reshape([2, 2, 3, 4])
    print("arr4.shape -> " + arr4.shape().__str__())
    print("arr4[0].shape -> " + arr4[0].shape().__str__())
    print("arr4[0,1].shape -> " + arr4[0,1].shape().__str__())
    show_tensor("arr4[0,1,2]", arr4[0,1,2])
    show_tensor("arr4[0,1,2,3] ->", arr4[0,1,2,3])
    print("""arr4[":,:,:,0"].shape -> """ + arr4[":,:,:,0"].shape().__str__())
    print("""arr4[":,:,1,:"].shape -> """ + arr4[":,:,1,:"].shape().__str__())
    print("""arr4[":"].shape -> """ + arr4[":"].shape().__str__())

    # =========================
    # 5) 5D examples
    # =========================
    banner("5D")
    var arr5 = tensor.arange(0, 48, 1).reshape([2, 2, 2, 2, 3])
    print("arr5.shape -> " + arr5.shape().__str__())
    print("arr5[0].shape -> " + arr5[0].shape().__str__())
    print("arr5[1,0].shape -> " + arr5[1,0].shape().__str__())
    print("arr5[1,0,1].shape -> " + arr5[1,0,1].shape().__str__())
    print("arr5[1,0,1,1].shape -> " + arr5[1,0,1,1].shape().__str__())

    show_tensor("arr5[1,1,1,1,2] ->", arr5[1, 1, 1, 1, 2])

    print("""arr5[":,:,1,:,:"].shape -> """ + arr5[":,:,1,:,:"].shape().__str__())
    print("""arr5["::2,:,:,:,0"].shape -> """ + arr5["::2,:,:,:,0"].shape().__str__())
    print("""arr5[":,1,:,:,1:3"].shape -> """ + arr5[":,1,:,:,1:3"].shape().__str__())
 
    show_tensor("arr5[[0,1]]", arr5[0,1])

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_indexing_numpy_parity()
