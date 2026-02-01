# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.test_tensor_axis_windows
# File:         examples/Tensor/demo_tensor_axis_windows.mojo
#
# Description:
#   Axis/window utilities test suite covering:
#   - moveaxis (single & multi)
#   - roll (flat, single-axis, multi-axis)
#   - swapaxes
#   - mean with axis/keepdims
#   - pad across 2D/3D/4D
#   - sliding_window (1D direct and ND via flatten)
#   - 2D flips (fliplr/flipud) 

from momijo.tensor import tensor
from collections.list import List

# -----------------------------------------------------------------------------
# utilities
# -----------------------------------------------------------------------------
fn check_shape(name: String, got: List[Int], exp: List[Int]) -> None:
    var ok = True
    if len(got) != len(exp):
        ok = False
    else:
        var i = 0
        while i < len(got):
            if got[i] != exp[i]:
                ok = False
                break
            i += 1

    if ok:
        print(name + " PASS: " + got.__str__())
    else:
        print(name + " FAIL: got=" + got.__str__() + " expected=" + exp.__str__())

# -----------------------------------------------------------------------------
# moveaxis suite
# -----------------------------------------------------------------------------
fn test_moveaxis_single_all_dims() -> None:
    print("\n=== moveaxis (single-axis) on 2D/3D/4D ===")

    # 2D
    var a2 = tensor.arange(0, 12).reshape([3, 4])
    var m20 = tensor.moveaxis(a2, 0, 1)   # [4,3]
    var m21 = tensor.moveaxis(a2, 1, 0)   # [4,3]
    check_shape("moveaxis 2D (0->1)", m20.shape(), [4, 3])
    check_shape("moveaxis 2D (1->0)", m21.shape(), [4, 3])

    # 3D
    var a3 = tensor.arange(0, 24).reshape([2, 3, 4])
    var m30 = tensor.moveaxis(a3, 0, -1)  # [3,4,2]
    var m31 = tensor.moveaxis(a3, 1, 0)   # [3,2,4]
    var m32 = tensor.moveaxis(a3, 2, 0)   # [4,2,3]
    check_shape("moveaxis 3D (0->-1)", m30.shape(), [3, 4, 2])
    check_shape("moveaxis 3D (1->0)",  m31.shape(), [3, 2, 4])
    check_shape("moveaxis 3D (2->0)",  m32.shape(), [4, 2, 3])

    # 4D
    var a4 = tensor.arange(0, 120).reshape([2, 3, 4, 5])
    var m40 = tensor.moveaxis(a4, 0, 3)   # [3,4,5,2]
    var m41 = tensor.moveaxis(a4, 1, 0)   # [3,2,4,5]
    var m42 = tensor.moveaxis(a4, 2, 0)   # [4,2,3,5]
    var m43 = tensor.moveaxis(a4, 3, 0)   # [5,2,3,4]
    check_shape("moveaxis 4D (0->3)", m40.shape(), [3, 4, 5, 2])
    check_shape("moveaxis 4D (1->0)", m41.shape(), [3, 2, 4, 5])
    check_shape("moveaxis 4D (2->0)", m42.shape(), [4, 2, 3, 5])
    check_shape("moveaxis 4D (3->0)", m43.shape(), [5, 2, 3, 4])

fn test_moveaxis_multi_all_dims() -> None:
    print("\n=== moveaxis (multi-axis) on 3D/4D ===")

    # 3D: (0,2)->(2,0) => [4,3,2]
    var a3 = tensor.arange(0, 24).reshape([2, 3, 4])
    var mm3 = tensor.moveaxis(a3, [0, 2], [2, 0])
    check_shape("moveaxis 3D multi (0,2)->(2,0)", mm3.shape(), [4, 3, 2])

    # 4D: (0,3)->(3,0) => [5,3,4,2]
    var a4 = tensor.arange(0, 120).reshape([2, 3, 4, 5])
    var mm4 = tensor.moveaxis(a4, [0, 3], [3, 0])
    check_shape("moveaxis 4D multi (0,3)->(3,0)", mm4.shape(), [5, 3, 4, 2])

# -----------------------------------------------------------------------------
# roll suite
# -----------------------------------------------------------------------------
fn test_roll_flat_all_dims() -> None:
    print("\n=== roll flat (axis=None) on 1D/2D/3D/4D ===")

    var v1 = tensor.arange(0, 6)                    # [6]
    var r1 = tensor.roll(v1, -2)
    check_shape("roll flat 1D", r1.shape(), [6])

    var a2 = tensor.arange(0, 12).reshape([3, 4])   # [3,4]
    var r2 = tensor.roll(a2, 5)
    check_shape("roll flat 2D", r2.shape(), [3, 4])

    var a3 = tensor.arange(0, 24).reshape([2, 3, 4])  # [2,3,4]
    var r3 = tensor.roll(a3, 9)
    check_shape("roll flat 3D", r3.shape(), [2, 3, 4])

    var a4 = tensor.arange(0, 120).reshape([2, 3, 4, 5])  # [2,3,4,5]
    var r4 = tensor.roll(a4, -7)
    check_shape("roll flat 4D", r4.shape(), [2, 3, 4, 5])

fn test_roll_axis_single_all_dims() -> None:
    print("\n=== roll (single axis) on 1D/2D/3D/4D ===")

    # 1D
    var v1 = tensor.arange(0, 6)
    var r10 = tensor.roll(v1, 1, 0)
    check_shape("roll 1D axis=0", r10.shape(), [6])

    # 2D
    var a2 = tensor.arange(0, 12).reshape([3, 4])
    var r2y = tensor.roll(a2, 2, 0)   # along rows
    var r2x = tensor.roll(a2, -1, 1)  # along cols
    check_shape("roll 2D axis=0", r2y.shape(), [3, 4])
    check_shape("roll 2D axis=1", r2x.shape(), [3, 4])

    # 3D
    var a3 = tensor.arange(0, 24).reshape([2, 3, 4])
    var r3a = tensor.roll(a3, 1, 0)
    var r3b = tensor.roll(a3, -2, 1)
    var r3c = tensor.roll(a3, 3, 2)
    check_shape("roll 3D axis=0", r3a.shape(), [2, 3, 4])
    check_shape("roll 3D axis=1", r3b.shape(), [2, 3, 4])
    check_shape("roll 3D axis=2", r3c.shape(), [2, 3, 4])

    # 4D
    var a4 = tensor.arange(0, 120).reshape([2, 3, 4, 5])
    var r4a = tensor.roll(a4, 1, 0)
    var r4b = tensor.roll(a4, -2, 1)
    var r4c = tensor.roll(a4, 3, 2)
    var r4d = tensor.roll(a4, -4, 3)
    check_shape("roll 4D axis=0", r4a.shape(), [2, 3, 4, 5])
    check_shape("roll 4D axis=1", r4b.shape(), [2, 3, 4, 5])
    check_shape("roll 4D axis=2", r4c.shape(), [2, 3, 4, 5])
    check_shape("roll 4D axis=3", r4d.shape(), [2, 3, 4, 5])

fn test_roll_multi_all_dims() -> None:
    print("\n=== roll (multi-axis) on 3D/4D ===")

    # 3D
    var a3 = tensor.arange(0, 24).reshape([2, 3, 4])
    var rm3 = tensor.roll(a3, [1, -2], [0, 2])
    check_shape("roll multi 3D [0,2]", rm3.shape(), [2, 3, 4])

    # 4D
    var a4 = tensor.arange(0, 120).reshape([2, 3, 4, 5])
    var rm4 = tensor.roll(a4, [1, -2, 3], [0, 2, 3])
    check_shape("roll multi 4D [0,2,3]", rm4.shape(), [2, 3, 4, 5])

# -----------------------------------------------------------------------------
# swapaxes suite
# -----------------------------------------------------------------------------
fn test_swapaxes_all_dims() -> None:
    print("\n=== swapaxes on 2D/3D/4D ===")

    # 2D
    var a2 = tensor.arange(0, 12).reshape([3, 4])
    var s2 = tensor.swapaxes(a2, 0, 1)
    check_shape("swapaxes 2D (0,1)", s2.shape(), [4, 3])

    # 3D
    var a3 = tensor.arange(0, 24).reshape([2, 3, 4])
    var s3a = tensor.swapaxes(a3, 0, 1)  # [3,2,4]
    var s3b = tensor.swapaxes(a3, 1, 2)  # [2,4,3]
    check_shape("swapaxes 3D (0,1)", s3a.shape(), [3, 2, 4])
    check_shape("swapaxes 3D (1,2)", s3b.shape(), [2, 4, 3])

    # 4D
    var a4 = tensor.arange(0, 120).reshape([2, 3, 4, 5])
    var s4a = tensor.swapaxes(a4, 0, 3)  # [5,3,4,2]
    var s4b = tensor.swapaxes(a4, 1, 3)  # [2,5,4,3]
    check_shape("swapaxes 4D (0,3)", s4a.shape(), [5, 3, 4, 2])
    check_shape("swapaxes 4D (1,3)", s4b.shape(), [2, 5, 4, 3])

# -----------------------------------------------------------------------------
# mean suite
# -----------------------------------------------------------------------------
fn test_mean_all_dims() -> None:
    print("\n=== mean (axis & keepdims) on 2D/3D/4D ===")

    # 2D
    var a2 = tensor.arange(0, 12).reshape([3, 4])
    var m2a = tensor.mean(a2, axis = 0)                   # [4]
    var m2b = tensor.mean(a2, axis = 1)                   # [3]
    var m2k = tensor.mean(a2, axis = 1, keepdims = True)  # [3,1]
    check_shape("mean 2D axis=0", m2a.shape(), [4])
    check_shape("mean 2D axis=1", m2b.shape(), [3])
    check_shape("mean 2D keepdims=1", m2k.shape(), [3, 1])

    # 3D
    var a3 = tensor.arange(0, 24).reshape([2, 3, 4])
    var m3a = tensor.mean(a3, axis = 0)                    # [3,4]
    var m3b = tensor.mean(a3, axis = 1)                    # [2,4]
    var m3c = tensor.mean(a3, axis = 2)                    # [2,3]
    var m3k = tensor.mean(a3, axis = 2, keepdims = True)   # [2,3,1]
    check_shape("mean 3D axis=0", m3a.shape(), [3, 4])
    check_shape("mean 3D axis=1", m3b.shape(), [2, 4])
    check_shape("mean 3D axis=2", m3c.shape(), [2, 3])
    check_shape("mean 3D keepdims=1", m3k.shape(), [2, 3, 1])

    # 4D
    var a4 = tensor.arange(0, 120).reshape([2, 3, 4, 5])
    var m4a = tensor.mean(a4, axis = 0)                    # [3,4,5]
    var m4b = tensor.mean(a4, axis = 1)                    # [2,4,5]
    var m4c = tensor.mean(a4, axis = 2)                    # [2,3,5]
    var m4d = tensor.mean(a4, axis = 3)                    # [2,3,4]
    var m4k = tensor.mean(a4, axis = 2, keepdims = True)   # [2,3,1,5]
    check_shape("mean 4D axis=0", m4a.shape(), [3, 4, 5])
    check_shape("mean 4D axis=1", m4b.shape(), [2, 4, 5])
    check_shape("mean 4D axis=2", m4c.shape(), [2, 3, 5])
    check_shape("mean 4D axis=3", m4d.shape(), [2, 3, 4])
    check_shape("mean 4D keepdims=1", m4k.shape(), [2, 3, 1, 5])

# -----------------------------------------------------------------------------
# pad suite
# -----------------------------------------------------------------------------
fn test_pad_all_dims() -> None:
    print("\n=== pad on 2D/3D/4D ===")

    # 2D: ((1,1),(2,2)) => [5,8]
    var a2 = tensor.arange(0, 12).reshape([3, 4])
    var p2 = tensor.pad(a2, [(1, 1), (2, 2)], Float64(-1.0))
    check_shape("pad 2D", p2.shape(), [5, 8])

    # 3D: ((1,1),(0,0),(2,2)) => [4,3,8]
    var a3 = tensor.arange(0, 24).reshape([2, 3, 4])
    var p3 = tensor.pad(a3, [(1, 1), (0, 0), (2, 2)], Float64(-1.0))
    check_shape("pad 3D", p3.shape(), [4, 3, 8])

    # 4D: ((1,1),(0,0),(2,2),(3,3)) => [4,3,8,11]
    var a4 = tensor.arange(0, 120).reshape([2, 3, 4, 5])
    var p4 = tensor.pad(a4, [(1, 1), (0, 0), (2, 2), (3, 3)], Float64(-1.0))
    check_shape("pad 4D", p4.shape(), [4, 3, 8, 11])

# -----------------------------------------------------------------------------
# sliding window suite
# -----------------------------------------------------------------------------
fn test_sliding_window_all_dims() -> None:
    print("\n=== sliding_window on 1D + flattened ND ===")

    # 1D direct
    var v1 = tensor.arange(0, 10)
    var w13 = tensor.sliding_window(v1, 3)     # [8,3]
    var w15 = tensor.sliding_window(v1, 5)     # [6,5]
    check_shape("window 1D k=3", w13.shape(), [8, 3])
    check_shape("window 1D k=5", w15.shape(), [6, 5])

    # ND via flatten (consistent with implementation that targets 1D)
    var a3 = tensor.arange(0, 24).reshape([2, 3, 4])
    var f3 = a3.reshape([24])
    var w3 = tensor.sliding_window(f3, 4)      # [21,4]
    check_shape("window from 3D->1D k=4", w3.shape(), [21, 4])

    var a4 = tensor.arange(0, 120).reshape([2, 3, 4, 5])
    var f4 = a4.reshape([120])
    var w4 = tensor.sliding_window(f4, 7)      # [114,7]
    check_shape("window from 4D->1D k=7", w4.shape(), [114, 7])

# -----------------------------------------------------------------------------
# flips (2D)
# -----------------------------------------------------------------------------
fn test_flips_2d() -> None:
    print("\n=== fliplr / flipud on 2D ===")

    var a2 = tensor.arange(0, 12).reshape([3, 4])
    var lr = tensor.fliplr(a2)
    var ud = tensor.flipud(a2)
    check_shape("fliplr 2D", lr.shape(), [3, 4])
    check_shape("flipud 2D", ud.shape(), [3, 4])

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
fn main() -> None:
    test_moveaxis_single_all_dims()
    test_moveaxis_multi_all_dims()

    test_roll_flat_all_dims()
    test_roll_axis_single_all_dims()
    test_roll_multi_all_dims()

    test_swapaxes_all_dims()
    test_mean_all_dims()
    test_pad_all_dims()
    test_sliding_window_all_dims()
    test_flips_2d()
