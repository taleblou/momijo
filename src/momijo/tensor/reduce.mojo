# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.tensor
# File: src/momijo/tensor/reduce.mojo

 from momijo.tensor.tensor_base import Tensor
from momijo.tensor.errors import require
from momijo.tensor.strides import compute_strides_rowmajor
 
from momijo.core.ndarray import offset
from momijo.arrow_core.array_stats import range
 

from momijo.tensor.tensor import Tensor

fn sum1d(a: Tensor[Float64]) -> Float64:
    assert(len(a.shape) == 1, "sum1d expects rank-1 tensor")
    var s: Float64 = 0.0
    var n = a.size()
    for i in range(0, n):
        s += a.data[a.offset + i]
    return s

fn mean1d(a: Tensor[Float64]) -> Float64:
    return sum1d(a) / Float64(a.size())


fn _normalize_axis(axis: Int, ndim: Int) -> Int:
    var ax = axis
    if ax < 0: ax = ndim + ax
    require(ax >= 0 and ax < ndim, "axis out of range")
    return ax

fn _out_shape_for_reduce(shape: List[Int], axis: Int, keepdims: Bool) -> List[Int]:
    var out_shape = List[Int]()
    var i = 0
    while i < len(shape):
        if i == axis:
            if keepdims: out_shape.append(1)
        else:
            out_shape.append(shape[i])
        i += 1
    if len(out_shape) == 0:
        out_shape.append(1)
    return out_shape

fn reduce_sum[T: Copyable & Movable](t: Tensor[T], axis: Int = -999, keepdims: Bool = False) -> Tensor[T]:
    if axis == -999:
        # full reduction -> scalar-shaped [1]
        var acc = t.get_flat(0)
        var i = 1
        while i < t.size():
            acc = acc + t.get_flat(i)
            i += 1
        var out = Tensor[T]([1], acc, t._dtype)
        return out
    var ax = _normalize_axis(axis, t.ndim())
    var out_shape = _out_shape_for_reduce(t.shape(), ax, keepdims)
    var out = Tensor[T](out_shape, t.get_flat(0), t._dtype)
    # naive loops for up to 3D; fallback to flatten by blocks
    if t.ndim() == 1:
        # sum over axis 0 -> [1] or [] with keepdims
        var s = t.get([0])
        var i = 1
        while i < t.shape()[0]:
            s = s + t.get([i])
            i += 1
        if keepdims:
            out.set([0], s)
        else:
            out = Tensor[T]([1], s, t._dtype)
        return out
    if t.ndim() == 2:
        var R = t.shape()[0]
        var C = t.shape()[1]
        if ax == 0:
            # sum rows -> [1,C] or [C]
            var c = 0
            while c < C:
                var s = t.get([0, c])
                var r = 1
                while r < R:
                    s = s + t.get([r, c])
                    r += 1
                if keepdims: out.set([0, c], s)
                else: out.set([c], s)
                c += 1
            return out
        else:
            # ax == 1: sum cols -> [R,1] or [R]
            var r = 0
            while r < R:
                var s = t.get([r, 0])
                var c = 1
                while c < C:
                    s = s + t.get([r, c])
                    c += 1
                if keepdims: out.set([r, 0], s)
                else: out.set([r], s)
                r += 1
            return out
    # fallback: flatten blocks along axis
    var n_before = 1
    var i = 0
    while i < ax:
        n_before *= t.shape()[i]
        i += 1
    var n_axis = t.shape()[ax]
    var n_after = 1
    i = ax + 1
    while i < t.ndim():
        n_after *= t.shape()[i]
        i += 1
    var out_elems = n_before * n_after
    # initialize to zeros via first block
    var base = 0
    while base < out_elems:
        var acc = t.get_flat(base * n_axis)
        var k = 1
        while k < n_axis:
            acc = acc + t.get_flat(base * n_axis + k)
            k += 1
        out.set_flat(base, acc)
        base += 1
    return out

fn reduce_mean[T: Copyable & Movable](t: Tensor[T], axis: Int = -999, keepdims: Bool = False) -> Tensor[T]:
    var s = reduce_sum(t, axis, keepdims)
    var count = t.size() if axis == -999 else t.shape()[_normalize_axis(axis, t.ndim())]
    # divide elementwise by count
    var i = 0
    while i < s.size():
        s.set_flat(i, s.get_flat(i) / count)
        i += 1
    return s

fn reduce_min[T: Copyable & Movable](t: Tensor[T], axis: Int = -999, keepdims: Bool = False) -> Tensor[T]:
    if axis == -999:
        var m = t.get_flat(0)
        var i = 1
        while i < t.size():
            var v = t.get_flat(i)
            if v < m: m = v
            i += 1
        return Tensor[T]([1], m, t._dtype)
    var ax = _normalize_axis(axis, t.ndim())
    var out_shape = _out_shape_for_reduce(t.shape(), ax, keepdims)
    var out = Tensor[T](out_shape, t.get_flat(0), t._dtype)
    if t.ndim() == 2:
        var R = t.shape()[0]
        var C = t.shape()[1]
        if ax == 0:
            var c = 0
            while c < C:
                var m = t.get([0, c])
                var r = 1
                while r < R:
                    var v = t.get([r, c])
                    if v < m: m = v
                    r += 1
                if keepdims: out.set([0, c], m) else: out.set([c], m)
                c += 1
            return out
        else:
            var r = 0
            while r < R:
                var m = t.get([r, 0])
                var c = 1
                while c < C:
                    var v = t.get([r, c])
                    if v < m: m = v
                    c += 1
                if keepdims: out.set([r, 0], m) else: out.set([r], m)
                r += 1
            return out
    # generic fallback similar to reduce_sum
    var n_before = 1
    var i2 = 0
    while i2 < ax:
        n_before *= t.shape()[i2]
        i2 += 1
    var n_axis = t.shape()[ax]
    var n_after = 1
    i2 = ax + 1
    while i2 < t.ndim():
        n_after *= t.shape()[i2]
        i2 += 1
    var out_elems = n_before * n_after
    var base = 0
    while base < out_elems:
        var m = t.get_flat(base * n_axis)
        var k = 1
        while k < n_axis:
            var v = t.get_flat(base * n_axis + k)
            if v < m: m = v
            k += 1
        out.set_flat(base, m)
        base += 1
    return out

fn reduce_max[T: Copyable & Movable](t: Tensor[T], axis: Int = -999, keepdims: Bool = False) -> Tensor[T]:
    if axis == -999:
        var m = t.get_flat(0)
        var i = 1
        while i < t.size():
            var v = t.get_flat(i)
            if v > m: m = v
            i += 1
        return Tensor[T]([1], m, t._dtype)
    var ax = _normalize_axis(axis, t.ndim())
    var out_shape = _out_shape_for_reduce(t.shape(), ax, keepdims)
    var out = Tensor[T](out_shape, t.get_flat(0), t._dtype)
    if t.ndim() == 2:
        var R = t.shape()[0]
        var C = t.shape()[1]
        if ax == 0:
            var c = 0
            while c < C:
                var m = t.get([0, c])
                var r = 1
                while r < R:
                    var v = t.get([r, c])
                    if v > m: m = v
                    r += 1
                if keepdims: out.set([0, c], m) else: out.set([c], m)
                c += 1
            return out
        else:
            var r = 0
            while r < R:
                var m = t.get([r, 0])
                var c = 1
                while c < C:
                    var v = t.get([r, c])
                    if v > m: m = v
                    c += 1
                if keepdims: out.set([r, 0], m) else: out.set([r], m)
                r += 1
            return out
    # generic fallback
    var n_before = 1
    var i2 = 0
    while i2 < ax:
        n_before *= t.shape()[i2]
        i2 += 1
    var n_axis = t.shape()[ax]
    var n_after = 1
    i2 = ax + 1
    while i2 < t.ndim():
        n_after *= t.shape()[i2]
        i2 += 1
    var out_elems = n_before * n_after
    var base = 0
    while base < out_elems:
        var m = t.get_flat(base * n_axis)
        var k = 1
        while k < n_axis:
            var v = t.get_flat(base * n_axis + k)
            if v > m: m = v
            k += 1
        out.set_flat(base, m)
        base += 1
    return out
