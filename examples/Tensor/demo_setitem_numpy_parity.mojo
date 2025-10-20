# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo.tensor
# File: tests/tensor/demo_setitem_numpy_parity.mojo
# Description: Parity demo for __setitem__ (writing with indices/slices) vs. NumPy-style,
#              covering 1D..5D with native and string-spec selectors.

from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# Small banner printer
# -----------------------------------------------------------------------------
fn banner(title: String) -> None:
    print("\n=== " + title + " ===")

# Print helper to ensure only String/scalars get printed safely
fn show_tensor(label: String, x: tensor.Tensor[Int]) -> None:
    print(label + ":\n" + x.__str__())

# -----------------------------------------------------------------------------
# 1) 1D __setitem__ tests
# -----------------------------------------------------------------------------
fn demo_setitem_1d() -> None:
    banner("1D - __setitem__")

    # Base array: [10, 20, 30, 40]
    var a = tensor.arange_int(10, 50, 10)
    show_tensor("start a", a)

    # Single index
    a[1] = tensor.from_list_int([999])                # set scalar
    show_tensor("a[1] = 999", a)

    a[-1] = tensor.from_list_int([-7])  
    show_tensor("a[-1] = -7", a)

    # Slice: exact-shape assignment
    var seg = tensor.from_list_int([21, 22])
    a[1:3] = seg
    show_tensor("a[1:3] = [21,22]", a)

    # Slice: scalar broadcast
    a[:] = tensor.from_list_int([5])
    show_tensor("a[:] = 5", a)

    # Strided slice: length-matched tensor
    var s2 = tensor.from_list_int([8, 9])
    a[::2] = s2                  # positions 0,2
    show_tensor("a[::2] = [8,9]", a)

    # Reversed slice: scalar broadcast
    a[3:0:-1] = tensor.from_list_int([1])                # indices 3,2,1
    show_tensor("a[3:0:-1] = 1", a)

    # -------- String-spec mirrors --------
    var b = tensor.arange_int(10, 50, 10)
    show_tensor("start b", b)

    b["1"] = tensor.from_list_int([1111])
    show_tensor("""b["1"] = 1111""", b)

    b["-1"] = tensor.from_list_int([-11])
    show_tensor("""b["-1"] = -11""", b)

    b["1:3"] = tensor.from_list_int([31, 32])
    show_tensor("""b["1:3"] = [31,32]""", b)

    b[":"] = tensor.from_list_int([4])
    show_tensor("""b[":"] = 4""", b)

    b["::2"] = tensor.from_list_int([6, 7])
    show_tensor("""b["::2"] = [6,7]""", b)

    b["3:0:-1"] = tensor.from_list_int([2])
    show_tensor("""b["3:0:-1"] = 2""", b)

# -----------------------------------------------------------------------------
# 2) 2D __setitem__ tests
# -----------------------------------------------------------------------------
fn demo_setitem_2d() -> None:
    banner("2D - __setitem__")

    # 3x3 matrix:
    # [[1,2,3],
    #  [4,5,6],
    #  [7,8,9]]
    var m = tensor.arange_int(1, 10, 1).reshape([3, 3])
    show_tensor("start m", m)

    # Row scalar broadcast
    m[1, :] = tensor.from_list_int([100])
    show_tensor("m[1,:] = 100", m)

    # Column vector (exact length)
    m[:, 1] = tensor.from_list_int([10, 20, 30])
    show_tensor("m[:,1] = [10,20,30]", m)

    # Window exact-shape assignment
    var w = tensor.from_list_int([1, 2, 3, 4]).reshape([2, 2])
    m[0:2, 0:2] = w
    show_tensor("m[0:2,0:2] = [[1,2],[3,4]]", m)

    m[1:, 1:] = tensor.from_list_int([9,8,7,6]).reshape([2,2])
    show_tensor("[1:, 1:]=[9,8,7,6]", m)

    # Stride + tail with broadcast
    m[::2, 1:] = tensor.from_list_int([7])
    show_tensor("m[::2,1:] = 7", m)

    # Reverse both axes
    m[1:  , 1:  ] = tensor.from_list_int([9,8,7,6]).reshape([2,2])
    show_tensor("m[1:  , 1:  ] = [[9,8],[7,6]] (bottom-right 2x2)", m)

    # -------- String-spec mirrors --------
    var s = tensor.arange_int(1, 10, 1).reshape([3, 3])
    show_tensor("start s", s)

    s["1,:"] = tensor.from_list_int([500])
    show_tensor("""s["1,:"] = 500""", s)

    s[":,1"] = tensor.from_list_int([11, 22, 33])
    show_tensor("""s[":,1"] = [11,22,33]""", s)

    s["0:2,0:2"] = w
    show_tensor("""s["0:2,0:2"] = [[1,2],[3,4]]""", s)

    s["::2,1:"] = tensor.from_list_int([12])
    show_tensor("""s["::2,1:"] = 12""", s)

    s["::-1,::-1"] = tensor.from_list_int([4,3,2,1]).reshape([2,2])
    show_tensor("""s["::-1,::-1"] = [[4,3],[2,1]]""", s)

# -----------------------------------------------------------------------------
# 3) 3D __setitem__ tests
# -----------------------------------------------------------------------------
fn demo_setitem_3d() -> None:
    banner("3D - __setitem__")

    # Shape: (2,3,4) with values 0..23
    var t = tensor.arange_int(0, 24, 1).reshape([2, 3, 4])
    show_tensor("start t", t)

    # Block write (broadcast scalar over a 2D slice)
    t[0] = tensor.from_list_int([-1])                  # t[0,:,:] = -1
    show_tensor("t[0] = -1", t)

    # Row in block (exact-shape)
    t[1, 2] = tensor.from_list_int([7, 7, 7, 7])
    show_tensor("t[1,2] = [7,7,7,7]", t)

    # Column in all blocks via slice
    t[:, 1, :] = tensor.from_list_int([3])
    show_tensor("t[:,1,:] = 3", t)

    # Single column across depth
    t[:, :, 0] = tensor.from_list_int([1, 2, 3, 4, 5, 6]).reshape([2, 3])
    show_tensor("t[:,:,0] = [[1,2,3],[4,5,6]]", t)

    # Mixed: first block, all rows, col=1 with vector
    t[0, :, 1] = tensor.from_list_int([9, 8, 7])
    show_tensor("t[0,:,1] = [9,8,7]", t)

    # Whole copy
    var u = tensor.arange_int(100, 124, 1).reshape([2, 3, 4])
    t[:] = u
    show_tensor("t[:] = u (copy all)", t)

    # Strided depth (even positions) broadcast
    t[:, :, ::2] = tensor.from_list_int([0])
    show_tensor("t[:,:,::2] = 0", t)

    # -------- String-spec mirrors --------
    var q = tensor.arange_int(0, 24, 1).reshape([2, 3, 4])
    show_tensor("start q", q)

    q["0"] = tensor.from_list_int([-5])
    show_tensor("""q["0"] = -5""", q)

    q["1,2"] = tensor.from_list_int([1, 1, 1, 1])
    show_tensor("""q["1,2"] = [1,1,1,1]""", q)

    q[":,1,:"] = tensor.from_list_int([2])
    show_tensor("""q[":,1,:"] = 2""", q)

    q[":,:,0"] = tensor.from_list_int([2, 2, 2, 3, 3, 3]).reshape([2, 3])
    show_tensor("""q[":,:,0"] = [[2,2,2],[3,3,3]]""", q)

    q["0,:,1"] = tensor.from_list_int([4, 5, 6])
    show_tensor("""q["0,:,1"] = [4,5,6]""", q)

    q[":"] = q.copy()          # copy self to self (no-op but tests path)
    show_tensor("""q[":"] = q.copy()""", q)

    q[":,:," "::2"] = tensor.from_list_int([7 ])       # note: in your sample there was a space, keep "::2"
    show_tensor("""q[":,:,::2"] = 7""", q)

# -----------------------------------------------------------------------------
# 4) 4D __setitem__ tests
# -----------------------------------------------------------------------------
fn demo_setitem_4d() -> None:
    banner("4D - __setitem__")

    # Shape: (2,2,3,4) values 0..47
    var x4 = tensor.arange_int(0, 48, 1).reshape([2, 2, 3, 4])
    show_tensor("start x4", x4)

    x4[:,:,:,0] = tensor.from_list_int([-1])
    show_tensor("x4[:,:,:,0] = -1]", x4)

    # Scalar at full index (sanity)
    x4[0, 1, 2, 3] = tensor.from_list_int([999])
    show_tensor("x4[0,1,2,3] = 999", x4)

    # Last channel for all => broadcast
    x4[:, :, :, 0] = tensor.from_list_int([-1])
    show_tensor("x4[:,:,:,0] = -1", x4)

    # Middle feature plane for all => exact-shape
    var plane = tensor.from_list_int([5, 6, 7, 8]).reshape([1, 1, 1, 4])
    x4[:, :, 1, :] = tensor.from_list_int([2, 2, 1, 4])  # if you have broadcast_to
    show_tensor("x4[:,:,1,:] = broadcasted [5..8]", x4)

    # Full copy
    var y4 = tensor.arange_int(1000, 1048, 1).reshape([2, 2, 3, 4])
    x4[:] = y4
    show_tensor("x4[:] = y4", x4)

    # Mixed window with strides
    x4[:, 1:, 1:, :2] = tensor.from_list_int([42])
    show_tensor("x4[:,1:,1:,:2] = 42", x4)

    # -------- String-spec mirrors --------
    var z4 = tensor.arange_int(0, 48, 1).reshape([2, 2, 3, 4])
    show_tensor("start z4", z4)

    z4["0,1,2,3"] = tensor.from_list_int([-9])
    show_tensor("""z4["0,1,2,3"] = -9""", z4)

    z4[":,:,:,0"] = tensor.from_list_int([1])
    show_tensor("""z4[":,:,:,0"] = 1""", z4)

    z4[":,:,1,:"] = tensor.from_list_int([13])
    show_tensor("""z4[":,:,1,:"] = 13""", z4)

    z4[":"] = z4.copy()
    show_tensor("""z4[":"] = z4.copy()""", z4)

    z4["...,0"] = tensor.from_list_int([0])
    show_tensor("""z4["...,0"] = 0""", z4)

    z4["::2,..."] = tensor.from_list_int([77])      # every other on axis 0
    show_tensor("""z4["::2,..."] = 77""", z4)

# -----------------------------------------------------------------------------
# 5) 5D __setitem__ tests
# -----------------------------------------------------------------------------
fn demo_setitem_5d() -> None:
    banner("5D - __setitem__")

    # Shape: (2,2,2,2,3) values 0..47
    var v5 = tensor.arange_int(0, 48, 1).reshape([2, 2, 2, 2, 3])
    show_tensor("start v5", v5)

    # Full index scalar
    v5[1, 1, 1, 1, 2] = tensor.from_list_int([2025])
    show_tensor("v5[1,1,1,1,2] = 2025", v5)

    # Fix depth=1 across all => broadcast
    v5[:, :, 1, :, :] = tensor.from_list_int([-2])
    show_tensor("v5[:,:,1,:,:] = -2", v5)

    # Axis0 strided, last channel = 0
    v5[::2, :, :, :, 0] = tensor.from_list_int([44])
    show_tensor("v5[::2,:,:,:,0] = 44", v5)

    # Window tail with exact-shape on last axis slice
    v5[:, 1, :, :, 1:3] = tensor.from_list_int([7, 8]).reshape([1, 1, 1, 1, 2])#.broadcast_to([2, 1, 2, 2, 2])
    show_tensor("v5[:,1,:,:,1:3] = broadcasted [7,8]", v5)

    # Reverse one axis with broadcast
    v5[:, :, :, ::-1, :] = tensor.from_list_int([0])
    show_tensor("v5[:,:,:,::-1,:] = 0", v5)

    # -------- String-spec mirrors --------
    var w5 = tensor.arange_int(0, 48, 1).reshape([2, 2, 2, 2, 3])
    show_tensor("start w5", w5)

    w5["1,1,1,1,2"] = tensor.from_list_int([3030])
    show_tensor("""w5["1,1,1,1,2"] = 3030""", w5)

    v5[:,1,:,:,1:3] = tensor.from_list_int([7,8]).reshape([1,1,1,1,2])   
    show_tensor("""w5[":,1,:,:,1:3"] = [7,8]""", w5)

    w5[":,:,1,:,:"] = tensor.from_list_int([5])
    show_tensor("""w5[":,:,1,:,:"] = 5""", w5)

    w5["::2,:,:,:,0"] = tensor.from_list_int([-3])
    show_tensor("""w5["::2,:,:,:,0"] = -3""", w5)

    w5[":,1,:,:,1:3"] = tensor.from_list_int([1, 2]).reshape([1, 1, 1, 1, 2])#.broadcast_to([2, 1, 2, 2, 2])
    show_tensor("""w5[":,1,:,:,1:3"] = broadcasted [1,2]""", w5)

    w5[":,:,:,::-1,:"] = tensor.from_list_int([6])
    show_tensor("""w5[":,:,:,::-1,:"] = 6""", w5)

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
fn main() -> None:
    demo_setitem_1d()
    #demo_setitem_2d()
    #demo_setitem_3d()
    #demo_setitem_4d()
    #demo_setitem_5d()
