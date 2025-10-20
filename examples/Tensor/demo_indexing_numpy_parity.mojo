# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         tests/tensor/demo_indexing_numpy_parity.mojo
#
# Description:
#   Parity demo for indexing/slicing vs. NumPy examples (1D..5D), showing both
#   native slice syntax and string-spec slice syntax.
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
# Print helpers (only String/scalars allowed)
# -----------------------------------------------------------------------------
fn show_tensor(label: String, x: tensor.Tensor[Int]) -> None:
    print(label + ": " + x.__str__())

fn show_tensor(label: String, x: tensor.Tensor[Float32]) -> None:
    print(label + ": " + x.__str__())

fn show_tensor(label: String, x: tensor.Tensor[Float64]) -> None:
    print(label + ": " + x.__str__())

fn show_tensor(label: String, x: tensor.Tensor[Bool]) -> None:
    print(label + ": " + x.__str__())

# -----------------------------------------------------------------------------
# Main demo
# -----------------------------------------------------------------------------
fn demo_indexing_numpy_parity() -> None:
    # =========================
    # 1) 1D examples
    # =========================
    banner("1D")
    var arr1 = tensor.arange_int(10, 50, 10)  # [10, 20, 30, 40]
    show_tensor("arr1", arr1)

    # ---- Native indexing/slicing ----
    show_tensor("arr1[1]", arr1[1])
    show_tensor("arr1[-1]", arr1[-1])
    show_tensor("arr1[1:3]", arr1[1:3])
    show_tensor("arr1[:]", arr1[:])
    show_tensor("arr1[::2]", arr1[::2])
    show_tensor("arr1[::-1]", arr1[::-1])
    show_tensor("arr1[3:0:-1]", arr1[3:0:-1])

    # Fancy (safe via method)
    var li1 = tensor.from_list_int([0, 2])
    show_tensor("arr1[[0,2]] (gather)", arr1.gather(0, li1))

    # Boolean mask
    var mask1 = arr1.mod_scalar(20).eq_scalar(0)
    show_tensor("mask1 (arr1%20==0)", mask1)
    # If boolean indexing exists: arr1.boolean_select(mask1)

    # ---- String-spec equivalents (require __getitem__(String)) ----
    show_tensor("""arr1["1"]""", arr1["1"])
    show_tensor("""arr1["-1"]""", arr1["-1"])
    show_tensor("""arr1["1:3"]""", arr1["1:3"])
    show_tensor("""arr1[":"]""", arr1[":"])
    show_tensor("""arr1["::2"]""", arr1["::2"])
    show_tensor("""arr1["::-1"]""", arr1["::-1"])
    show_tensor("""arr1["3:0:-1"]""", arr1["3:0:-1"])

    # =========================
    # 2) 2D examples
    # =========================
    banner("2D")
    var arr2 = tensor.arange_int(1, 10, 1).reshape([3, 3])  # 3x3
    print("arr2.shape -> " + arr2.shape().__str__())
    show_tensor("arr2", arr2)

    # ---- Native ----
    show_tensor("arr2[1]", arr2[1])                # row 1
    show_tensor("arr2[1,2]", arr2[1,2])            # scalar
    show_tensor("arr2[-1]", arr2[-1])              # last row
    show_tensor("arr2[:,1]", arr2[:,1])            # column 1
    show_tensor("arr2[1,:]", arr2[1,:])            # row 1
    show_tensor("arr2[:,:]", arr2[:,:])            # full
    show_tensor("arr2[::2,1:]", arr2[::2,1:])      # stride+tail
    show_tensor("arr2[::-1,::-1]", arr2[::-1,::-1])# reverse both axes
    show_tensor("arr2[0:2,0:2]", arr2[0:2,0:2])    # 2x2 window
    show_tensor("arr2[1:,1:]", arr2[1:,1:])        # lower-right window

    # String-spec versions available if your API supports them:
    # show_tensor("""arr2["1"]""", arr2["1"])
    # show_tensor("""arr2["1,2"]""", arr2["1,2"])
    # show_tensor("""arr2[":,1"]""", arr2[":,1"])
    # show_tensor("""arr2["1,:"]""", arr2["1,:"])
    # show_tensor("""arr2[":,:"]""", arr2[":,:"])
    # show_tensor("""arr2["::2,1:"]""", arr2["::2,1:"])
    # show_tensor("""arr2["::-1,::-1"]""", arr2["::-1,::-1"])
    # show_tensor("""arr2["0:2,0:2"]""", arr2["0:2,0:2"])
    # show_tensor("""arr2["1:,1:"]""", arr2["1:,1:"])

    # =========================
    # 3) 3D examples
    # =========================
    banner("3D")
    var arr3 = tensor.arange_int(0, 24, 1).reshape([2, 3, 4])  # (2,3,4)
    print("arr3.shape -> " + arr3.shape().__str__())
    show_tensor("arr3", arr3)

    # ---- Native ----
    show_tensor("arr3[0]", arr3[0])                # first block
    show_tensor("arr3[1,2]", arr3[1,2])            # row in block
    show_tensor("arr3[-1,-1]", arr3[-1,-1])        # last row of last block
    show_tensor("arr3[:,1,:]", arr3[:,1,:])
    show_tensor("arr3[:,:,0]", arr3[:,:,0])
    show_tensor("arr3[0,:,1]", arr3[0,:,1])
    show_tensor("arr3[:]", arr3[:])
    show_tensor("arr3[:, :, ::2]", arr3[:, :, ::2])

    # ---- String-spec ----
    show_tensor("""arr3["0"]""", arr3["0"])
    show_tensor("""arr3["1,2"]""", arr3["1,2"])
    show_tensor("""arr3["-1,-1"]""", arr3["-1,-1"])
    show_tensor("""arr3[":,1,:"]""", arr3[":,1,:"])
    show_tensor("""arr3[":,:,0"]""", arr3[":,:,0"])
    show_tensor("""arr3["0,:,1"]""", arr3["0,:,1"])
    show_tensor("""arr3[":"]""", arr3[":"])
    show_tensor("""arr3["...,1"]""", arr3["...,1"])
    show_tensor("""arr3["1,..."]""", arr3["1,..."])
    show_tensor("""arr3[":,:,::2"]""", arr3[":,:,::2"])

    # =========================
    # 4) 4D examples
    # =========================
    banner("4D")
    var arr4 = tensor.arange_int(0, 48, 1).reshape([2, 2, 3, 4])  # (2,2,3,4)
    print("arr4.shape -> " + arr4.shape().__str__())
    print("arr4[0].shape -> " + arr4[0].shape().__str__())
    print("arr4[0,1].shape -> " + arr4[0,1].shape().__str__())

    # ---- Native ----
    show_tensor("arr4[0,1,2,3]", arr4[0,1,2,3])    # scalar
    print("arr4[:, :, :, 0].shape -> " + arr4[:, :, :, 0].shape().__str__())
    print("arr4[:, :, 1, :].shape -> " + arr4[:, :, 1, :].shape().__str__())
    print("arr4[:].shape -> " + arr4[:].shape().__str__())
    print("arr4[:, 1:, 1:, :2].shape -> " + arr4[:, 1:, 1:, :2].shape().__str__())

    # ---- String-spec ----
    show_tensor("""arr4["0,1,2"]""", arr4["0,1,2"])
    show_tensor("""arr4["0,1,2,3"]""", arr4["0,1,2,3"])
    print("""arr4[":,:,:,0"].shape -> """ + arr4[":,:,:,0"].shape().__str__())
    print("""arr4[":,:,1,:"].shape -> """ + arr4[":,:,1,:"].shape().__str__())
    print("""arr4[":"].shape -> """ + arr4[":"].shape().__str__())
    print("""arr4["...,0"].shape -> """ + arr4["...,0"].shape().__str__())
    print("""arr4["...,::2"].shape -> """ + arr4["...,::2"].shape().__str__())
    print("""arr4["::2,..."].shape -> """ + arr4["::2,..."].shape().__str__())
    print("""arr4[":,1:,1:,:2"].shape -> """ + arr4[":,1:,1:,:2"].shape().__str__())

    # =========================
    # 5) 5D examples
    # =========================
    banner("5D")
    var arr5 = tensor.arange_int(0, 48, 1).reshape([2, 2, 2, 2, 3])  # (2,2,2,2,3)
    print("arr5.shape -> " + arr5.shape().__str__())
    print("arr5[0].shape -> " + arr5[0].shape().__str__())
    print("arr5[1,0].shape -> " + arr5[1,0].shape().__str__())
    print("arr5[1,0,1].shape -> " + arr5[1,0,1].shape().__str__())
    print("arr5[1,0,1,1].shape -> " + arr5[1,0,1,1].shape().__str__())

    # ---- Native ----
    show_tensor("arr5[1,1,1,1,2]", arr5[1,1,1,1,2])
    print("arr5[:, :, 1, :, :].shape -> " + arr5[:, :, 1, :, :].shape().__str__())
    print("arr5[::2, :, :, :, 0].shape -> " + arr5[::2, :, :, :, 0].shape().__str__())
    print("arr5[:, 1, :, :, 1:3].shape -> " + arr5[:, 1, :, :, 1:3].shape().__str__())
    print("arr5[:, :, :, ::-1, :].shape -> " + arr5[:, :, :, ::-1, :].shape().__str__())

    # Fancy (safe via gather)
    var pick0 = tensor.from_list_int([0, 1])
    show_tensor("arr5[[0,1]] (gather dim=0)", arr5.gather(0, pick0))

    # Boolean mask flattened select: multiples of 7
    var mask5 = arr5.mod_scalar(7).eq_scalar(0)
    show_tensor("mask5 (arr5%7==0)", mask5)
    # If boolean indexing exists: arr5.boolean_select(mask5)

    # ---- String-spec ----
    show_tensor("""arr5["1,1,1,1,2"]""", arr5["1,1,1,1,2"])
    print("""arr5[":,:,1,:,:"].shape -> """ + arr5[":,:,1,:,:"].shape().__str__())
    print("""arr5["::2,:,:,:,0"].shape -> """ + arr5["::2,:,:,:,0"].shape().__str__())
    print("""arr5[":,1,:,:,1:3"].shape -> """ + arr5[":,1,:,:,1:3"].shape().__str__())
    print("""arr5["...,1:"].shape -> """ + arr5["...,1:"].shape().__str__())
    print("""arr5[":,:,:,::-1,:"].shape -> """ + arr5[":,:,:,::-1,:"].shape().__str__())

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_indexing_numpy_parity()
