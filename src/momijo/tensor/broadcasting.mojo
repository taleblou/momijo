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
# File: src/momijo/tensor/broadcasting.mojo

from momijo.tensor.tensor_base import Tensor
from momijo.tensor.errors import check_broadcastable_like, require
from momijo.tensor.strides import shape_product

from momijo.tensor.shape import rank  # chosen by proximity
 
from momijo.tensor.shape import dim
from momijo.tensor.shape import Shape



# Minimal result holder (expand later if needed)
struct BroadcastResult(Copyable, Movable):
    var shape: Shape
    var lhs_strides: List[Int]
    var rhs_strides: List[Int]

    fn __init__(out self, shape: Shape, lhs_strides: List[Int], rhs_strides: List[Int]):
        self.shape = shape
        self.lhs_strides = lhs_strides
        self.rhs_strides = rhs_strides

    fn __copyinit__(out self, other: Self):
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

fn can_broadcast(a: List[Int], b: List[Int]) -> Bool:
    var ia = len(a) - 1
    var ib = len(b) - 1
    while ia >= 0 or ib >= 0:
        var da = 1
        var db = 1
        if ia >= 0: da = a[ia]
        if ib >= 0: db = b[ib]
        if not (da == db or da == 1 or db == 1):
            return False
        ia -= 1
        ib -= 1
    return True

fn broadcast_shape(a: List[Int], b: List[Int]) -> List[Int]:
    var out = List[Int]()
    var ia = len(a) - 1
    var ib = len(b) - 1
    while ia >= 0 or ib >= 0:
        var da = 1
        var db = 1
        if ia >= 0: da = a[ia]
        if ib >= 0: db = b[ib]
        if da == db or da == 1 or db == 1:
            var d = da if da != 1 else db
            if db != 1: d = db
            if da != 1 and db == 1: d = da
            if da == db: d = da
            out.insert(0, d)
        else:
            # incompatible; return [] to indicate failure
            return List[Int]()
        ia -= 1
        ib -= 1
    return out

fn broadcast_to[T: Copyable & Movable](t: Tensor[T], new_shape: List[Int]) -> Tensor[T]:
    # Validate compatibility
    var ok = can_broadcast(t.shape(), new_shape)
    require(ok, "broadcast_to: incompatible shapes")
    var out = Tensor[T](new_shape, t.get_flat(0), t._dtype)
    # Materialize by repeating along axes where original dim==1
    if len(new_shape) == 1:
        var N = new_shape[0]
        var i = 0
        if t.ndim() == 1 and t.shape()[0] == 1:
            while i < N:
                out.set([i], t.get([0]))
                i += 1
        elif t.ndim() == 1 and t.shape()[0] == N:
            i = 0
            while i < N:
                out.set([i], t.get([i]))
                i += 1
        else:
            # right-align general case
            var base = t
            var bN = base.shape()[0] if base.ndim() > 0 else 1
            i = 0
            while i < N:
                out.set([i], base.get([i % bN]))
                i += 1
        return out
    # 2D common case
    if len(new_shape) == 2:
        var R = new_shape[0]
        var C = new_shape[1]
        var r_base = 1
        var c_base = 1
        if t.ndim() == 2:
            r_base = t.shape()[0]
            c_base = t.shape()[1]
        elif t.ndim() == 1:
            r_base = t.shape()[0]
            c_base = 1
        var r = 0
        while r < R:
            var c = 0
            while c < C:
                var rr = r % r_base
                var cc = c % c_base
                if t.ndim() == 2:
                    out.set([r, c], t.get([rr, cc]))
                elif t.ndim() == 1:
                    out.set([r, c], t.get([rr]))
                else:
                    out.set([r, c], t.get_flat(0))
                c += 1
            r += 1
        return out
    # Fallback materialization for higher ranks (simple nested repeat)
    var total = shape_product(new_shape)
    var i = 0
    while i < total:
        out.set_flat(i, t.get_flat(i % t.size()))
        i += 1
    return out
