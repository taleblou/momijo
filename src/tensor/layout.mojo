# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.tensor
# File: momijo/tensor/layout.mojo

from momijo.tensor.shape import Shape

fn __module_name__() -> String:
    return String("momijo/tensor/layout.mojo")

fn __self_test__() -> Bool:
    var sh = Shape([2, 3, 4])
    var st = Strides.for_c_contiguous(sh)
    var ly = Layout(sh, st)
    if not ly.is_contiguous():
        return False
    var lv = LayoutView(ly, 0)
    var idx = List[Int]()
    idx.append(1); idx.append(2); idx.append(3)
    var lin = lv.linear_index(idx)
    if lin != (1 * st.vals[0] + 2 * st.vals[1] + 3 * st.vals[2]):
        return False
    return True

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

struct Strides(Copyable, Movable):
    var vals: List[Int]

    fn __init__(out self):
        self.vals = List[Int]()

    fn __init__(out self, vals: List[Int]):
        self.vals = vals

    fn __copyinit__(out self, other: Self):
        var tmp = List[Int]()
        var i = 0
        while i < len(other.vals):
            tmp.append(other.vals[i])
            i += 1
        self.vals = tmp

    @staticmethod
    fn for_c_contiguous(shape: Shape) -> Strides:
        var nd = shape.rank()
        var s = List[Int]()
        var i = 0
        while i < nd:
            s.append(0)
            i += 1
        var stride: Int = 1
        var j: Int = nd - 1
        while j >= 0:
            s[j] = stride
            stride = stride * shape.dim(j)
            j = j - 1
        return Strides(s)

struct Layout(Copyable, Movable):
    var shape: Shape
    var strides: Strides

    fn __init__(out self, shape: Shape, strides: Strides):
        self.shape = shape
        self.strides = strides

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape
        self.strides = other.strides

    fn is_contiguous(self) -> Bool:
        return self.strides.vals == Strides.for_c_contiguous(self.shape).vals

    fn numel(self) -> Int:
        return self.shape.numel()

struct LayoutView(Copyable, Movable):
    var layout: Layout
    var offset: Int

    fn __init__(out self, layout: Layout, offset: Int = 0):
        self.layout = layout
        self.offset = offset

    fn __copyinit__(out self, other: Self):
        self.layout = other.layout
        self.offset = other.offset

    fn linear_index(self, idx: List[Int]) -> Int:
        var s: Int = self.offset
        var i: Int = 0
        while i < len(idx):
            s = s + idx[i] * self.layout.strides.vals[i]
            i = i + 1
        return s
