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
# File: src/momijo/tensor/broadcast.mojo

 
 
from momijo.tensor.tensor_base import strides  # chosen by proximity
from momijo.tensor.tensor import index
from momijo.tensor.tensor import Tensor   # keep generic if your Tensor is generic
from momijo.tensor.layout import Shape    # optional; kept for readability

# -------------------- small utils --------------------

fn _prod(xs: List[Int]) -> Int:
    var p: Int = 1
    for v in xs:
        p = p * v
    return p

fn _row_major_strides(shape: List[Int]) -> List[Int]:
    # e.g., shape [s0, s1, ..., sn-1] -> strides [s1*s2*...*sn-1, ..., 1]
    var n: Int = len(shape)
    var strides = List[Int]()
    if n == 0:
        return strides
    # build from right to left
    var acc: Int = 1
    var i: Int = n - 1
    while i >= 0:
        strides.append(acc)
        acc = acc * shape[i]
        i = i - 1
    # reverse to align with row-major (left-to-right)
    var r = List[Int]()
    var j: Int = len(strides) - 1
    while j >= 0:
        r.append(strides[j])
        j = j - 1
    return r

fn _align_right(a: List[Int], b: List[Int]) -> (List[Int], List[Int]):
    # Pad the shorter shape with leading 1s so len(a)==len(b)
    var la: Int = len(a)
    var lb: Int = len(b)
    if la == lb:
        return (a, b)
    if la < lb:
        var out_a = List[Int]()
        var pad: Int = lb - la
        var i: Int = 0
        while i < pad:
            out_a.append(1)
            i += 1
        for v in a:
            out_a.append(v)
        return (out_a, b)
    else:
        var out_b = List[Int]()
        var pad2: Int = la - lb
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

fn broadcast_shape(a: List[Int], b: List[Int]) -> List[Int]:
    var aa: List[Int]
    var bb: List[Int]
    (aa, bb) = _align_right(a, b)
    var n: Int = len(aa)
    var out = List[Int]()
    var i: Int = 0
    while i < n:
        var da: Int = aa[i]
        var db: Int = bb[i]
        if da == db:
            out.append(da)
        elif da == 1:
            out.append(db)
        else:
            # db must be 1 (otherwise can_broadcast should have failed)
            out.append(da)
        i += 1
    return out

# ravel / unravel for row-major layout
fn _ravel_index(indices: List[Int], shape: List[Int]) -> Int:
    # idx = sum(indices[i] * strides[i])
    var strides = _row_major_strides(shape)
    var idx: Int = 0
    var n: Int = len(shape)
    var i: Int = 0
    while i < n:
        idx = idx + indices[i] * strides[i]
        i += 1
    return idx

fn _unravel_index(flat: Int, shape: List[Int]) -> List[Int]:
    # inverse of ravel; row-major
    var strides = _row_major_strides(shape)
    var out = List[Int]()
    var rem: Int = flat
    var n: Int = len(shape)
    var i: Int = 0
    while i < n:
        var s: Int = strides[i]
        # ensure Int division
        var v: Int = Int(rem / s)
        out.append(v)
        rem = rem - v * s
        i += 1
    return out

# Map an index of the broadcasted output back to (index in A, index in B)
fn _map_indices(out_idx: List[Int], shape_a: List[Int], shape_b: List[Int]) -> (List[Int], List[Int]):
    var aa: List[Int]
    var bb: List[Int]
    (aa, bb) = _align_right(shape_a, shape_b)

    var n: Int = len(out_idx)
    var mapped_a = List[Int]()
    var mapped_b = List[Int]()
    var i: Int = 0
    while i < n:
        var da: Int = aa[i]
        var db: Int = bb[i]
        var oi: Int = out_idx[i]
        var ia: Int = oi
        var ib: Int = oi
        if da == 1:
            ia = 0
        if db == 1:
            ib = 0
        mapped_a.append(ia)
        mapped_b.append(ib)
        i += 1
    return (mapped_a, mapped_b)

# -------------------- BroadcastMapper --------------------

struct BroadcastMapper(Copyable, Movable):
    var shape_a: List[Int]
    var shape_b: List[Int]
    var out_shape: List[Int]
    var total: Int
    var index: Int

    fn __init__(out self, shape_a: List[Int], shape_b: List[Int]):
        self.shape_a = shape_a
        self.shape_b = shape_b
        self.out_shape = broadcast_shape(shape_a, shape_b)
        self.total = _prod(self.out_shape)
        self.index = 0

    fn __copyinit__(out self, other: Self):
        self.shape_a = other.shape_a
        self.shape_b = other.shape_b
        self.out_shape = other.out_shape
        self.total = other.total
        self.index = other.index

    fn __len__(self) -> Int:
        return self.total

    fn reset(mut self):
        self.index = 0

    fn valid(self) -> Bool:
        return self.index < self.total

    fn next(mut self) -> (Int, Int, Int):
        # returns (out_flat, a_flat, b_flat)
        var out_flat: Int = self.index
        var out_idx = _unravel_index(out_flat, self.out_shape)
        var (ai, bi) = _map_indices(out_idx, self.shape_a, self.shape_b)
        var a_flat: Int = _ravel_index(ai, _align_right(self.shape_a, self.shape_b)[0])
        var b_flat: Int = _ravel_index(bi, _align_right(self.shape_a, self.shape_b)[1])
        self.index = self.index + 1
        return (out_flat, a_flat, b_flat)

# -------------------- Apply (stub) --------------------------------------------

fn apply_broadcast_binary_float32(
    a: Tensor[Float32],
    b: Tensor[Float32],
    out_tensor: Tensor[Float32],
    op: Int = 0  # 0: add, 1: sub, 2: mul, 3: div (placeholder)
):
    # NOTE: This is a stub. Implement once Tensor exposes safe element access
    # (e.g., get_flat/set_flat or __getitem__/__setitem__).
    # For now: do nothing, just keep signature for callers.
    pass