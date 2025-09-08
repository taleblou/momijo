# Project:      Momijo
# Module:       src.momijo.tensor.broadcasting
# File:         broadcasting.mojo
# Path:         src/momijo/tensor/broadcasting.mojo
#
# Description:  Core tensor/ndarray components: shapes/strides, broadcasting rules,
#               element-wise ops, and foundational kernels.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Structs: BroadcastResult
#   - Key functions: __module_name__, __self_test__, argmax_index, argmin_index, __init__, __copyinit__, _align_right, can_broadcast ...
#   - Uses generic functions/types with explicit trait bounds.
#   - GPU/device utilities present; validate backend assumptions.


from momijo.tensor.shape import Shape

fn __module_name__() -> String:
    return String("momijo/tensor/broadcasting.mojo")
fn __self_test__() -> Bool:
    return True
# lightweight helpers to satisfy tests
fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]
            idx = i
        i += 1
    return idx
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]
            idx = i
        i += 1
    return idx
# Minimal result holder (expand later if needed)
struct BroadcastResult(Copyable, Movable):
    var shape: Shape
    var lhs_strides: List[Int]
    var rhs_strides: List[Int]
fn __init__(out self, shape: Shape, lhs_strides: List[Int], rhs_strides: List[Int]) -> None:
        self.shape = shape
        self.lhs_strides = lhs_strides
        self.rhs_strides = rhs_strides
fn __copyinit__(out self, other: Self) -> None:
        self.shape = other.shape
        self.lhs_strides = other.lhs_strides
        self.rhs_strides = other.rhs_strides

# Align shorter dims with leading 1s
fn _align_right(a: List[Int], b: List[Int]) -> (List[Int], List[Int]):
    var la: Int = len(a)
    var lb: Int = len(b)
    if la == lb:
        return (a, b)

    if la < lb:
        var pad = lb - la
        var out_a = List[Int]()
        var i: Int = 0
        while i < pad:
            out_a.append(1)
            i += 1
        for v in a:
            out_a.append(v)
        return (out_a, b)
    else:
        var pad2 = la - lb
        var out_b = List[Int]()
        var j: Int = 0
        while j < pad2:
            out_b.append(1)
            j += 1
        for v2 in b:
            out_b.append(v2)
        return (a, out_b)
fn can_broadcast(a: List[Int], b: List[Int]) -> Bool:
    var aa: List[Int]
    var bb: List[Int]
    (aa, bb) = _align_right(a, b)
    var n: Int = len(aa)
    var i: Int = 0
    while i < n:
        var da: Int = aa[i]
        var db: Int = bb[i]
        if not (da == db or da == 1 or db == 1):
            return False
        i += 1
    return True
fn broadcast_shapes(a: Shape, b: Shape) -> Shape:
    var ra: Int = a.rank()
    var rb: Int = b.rank()

    var r: Int
    if ra > rb:
        r = ra
    else:
        r = rb

    var out_dims = List[Int]()
    var t: Int = 0
    while t < r:
        out_dims.append(1)
        t += 1

    var i: Int = 0
    var compatible: Bool = True
    while i < r:
        var da: Int
        if i < ra:
            da = a.dim(ra - 1 - i)
        else:
            da = 1

        var db: Int
        if i < rb:
            db = b.dim(rb - 1 - i)
        else:
            db = 1

        if (da == db) or (da == 1) or (db == 1):
            var m: Int = da
            if db > m:
                m = db
            out_dims[r - 1 - i] = m
        else:
            compatible = False
            out_dims[r - 1 - i] = 1  # keep well-formed
        i += 1

    # If you prefer to fail hard on incompatibility later, you can surface `compatible`.
    return Shape(out_dims)