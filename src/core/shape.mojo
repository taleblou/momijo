# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Taleblou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.core
# File: momijo/core/shape.mojo
#
# This file is part of the Momijo project.
# See the LICENSE file at the repository root for license information. 

from momijo.core.error import MomijoError
from momijo.core.result import Result

# -------------------------
# Shape
# -------------------------

struct Shape(Copyable, Movable, EqualityComparable):
    var dims: List[Int]

    fn __init__(out self, dims: List[Int] = List[Int]()):
        # store as-is; callers can validate with validate()
        self.dims = dims

    fn rank(self) -> Int:
        return len(self.dims)

    fn count(self) -> Int:
        if len(self.dims) == 0: return 0
        var n = 1
        var i = 0
        while i < len(self.dims):
            n = n * self.dims[i]
            i += 1
        return n

    fn is_scalar(self) -> Bool: return self.rank() == 0
    fn is_vector(self) -> Bool: return self.rank() == 1
    fn is_matrix(self) -> Bool: return self.rank() == 2

    fn to_string(self) -> String:
        var s = "["
        var i = 0
        while i < len(self.dims):
            if i > 0: s = s + ", "
            s = s + String(self.dims[i])
            i += 1
        s = s + "]"
        return s

    fn validate(self) -> Result[Shape]:
        var i = 0
        while i < len(self.dims):
            if self.dims[i] < 0:
                return Result[Shape].fail(MomijoError.range_error("negative dimension in shape", "momijo.core.shape"), self)
            i += 1
        return Result[Shape].ok(self)

    fn clone(self) -> Shape:
        var out = List[Int]()
        out.reserve(len(self.dims))
        var i = 0
        while i < len(self.dims):
            out.append(self.dims[i])
            i += 1
        return Shape(dims=out)

    # ---- modifiers ----
    fn append(self, d: Int) -> Shape:
        var out = self.clone()
        out.dims.append(d)
        return out

    fn prepend(self, d: Int) -> Shape:
        var out = List[Int]()
        out.append(d)
        var i = 0
        while i < len(self.dims):
            out.append(self.dims[i])
            i += 1
        return Shape(dims=out)

    fn insert_axis(self, axis: Int, d: Int) -> Shape:
        var r = self.rank()
        var ax = axis
        if ax < 0: ax = 0
        if ax > r: ax = r
        var out = List[Int]()
        var i = 0
        while i < r:
            if i == ax: out.append(d)
            out.append(self.dims[i])
            i += 1
        if ax == r: out.append(d)
        return Shape(dims=out)

    fn remove_axis(self, axis: Int) -> Shape:
        var r = self.rank()
        if r == 0: return self
        var ax = axis
        if ax < 0: ax = 0
        if ax >= r: ax = r - 1
        var out = List[Int]()
        var i = 0
        while i < r:
            if i != ax: out.append(self.dims[i])
            i += 1
        return Shape(dims=out)

    fn squeeze(self) -> Shape:
        var out = List[Int]()
        var i = 0
        while i < len(self.dims):
            if self.dims[i] != 1:
                out.append(self.dims[i])
            i += 1
        return Shape(dims=out)

    fn unsqueeze(self, axis: Int) -> Shape:
        return self.insert_axis(axis, 1)

    fn expand_dims(self, axes: List[Int]) -> Shape:
        var out = self
        var i = 0
        while i < len(axes):
            out = out.unsqueeze(axes[i])
            i += 1
        return out

    fn reshape(self, new_dims: List[Int]) -> Result[Shape]:
        var new_shape = Shape(dims=new_dims)
        if self.count() != new_shape.count():
            return Result[Shape].fail(MomijoError.invalid_argument("reshape changes element count", "momijo.core.shape"), self)
        return Result[Shape].ok(new_shape)

    fn transpose(self, perm: List[Int]) -> Result[Shape]:
        var r = self.rank()
        if len(perm) != r:
            return Result[Shape].fail(MomijoError.invalid_argument("perm length mismatch", "momijo.core.shape"), self)
        var seen = List[Int]()
        seen.reserve(r)
        var i = 0
        while i < r:
            seen.append(0)
            i += 1
        i = 0
        while i < r:
            var p = perm[i]
            if p < 0 or p >= r:
                return Result[Shape].fail(MomijoError.range_error("perm element out of range", "momijo.core.shape"), self)
            seen[p] = seen[p] + 1
            i += 1
        i = 0
        while i < r:
            if seen[i] != 1:
                return Result[Shape].fail(MomijoError.invalid_argument("perm is not a permutation", "momijo.core.shape"), self)
            i += 1
        var out_dims = List[Int]()
        i = 0
        while i < r:
            out_dims.append(self.dims[perm[i]])
            i += 1
        return Result[Shape].ok(Shape(dims=out_dims))

    # ---- broadcasting ----
    fn align_right(self, ndim: Int) -> Shape:
        var r = self.rank()
        if r >= ndim: return self
        var pad = ndim - r
        var out = List[Int]()
        var i = 0
        while i < pad:
            out.append(1)
            i += 1
        i = 0
        while i < r:
            out.append(self.dims[i])
            i += 1
        return Shape(dims=out)

    fn is_broadcastable_to(self, target: Shape) -> Bool:
        var a = self.align_right(target.rank())
        var t = target
        var i = target.rank() - 1
        while i >= 0:
            var da = a.dims[i]
            var dt = t.dims[i]
            if not (da == dt or da == 1):
                return False
            i -= 1
        return True

    fn broadcast_to(self, target: Shape) -> Result[Shape]:
        if self.is_broadcastable_to(target):
            return Result[Shape].ok(target)
        return Result[Shape].fail(MomijoError.invalid_argument("not broadcastable to target " + target.to_string(), "momijo.core.shape"), self)

# -------------------------
# Strides
# -------------------------

struct Strides(Copyable, Movable, EqualityComparable):
    var steps: List[Int]

    fn __init__(out self, steps: List[Int] = List[Int]()):
        self.steps = steps

    @staticmethod
    fn row_major_for(shape: Shape) -> Strides:
        var r = shape.rank()
        var st = List[Int]()
        st.reserve(r)
        var i = 0
        while i < r:
            st.append(0)
            i += 1
        var acc = 1
        var k = r - 1
        while k >= 0:
            st[k] = acc
            acc = acc * shape.dims[k]
            k -= 1
        return Strides(steps=st)

    fn to_string(self) -> String:
        var s = "["
        var i = 0
        while i < len(self.steps):
            if i > 0: s = s + ", "
            s = s + String(self.steps[i])
            i += 1
        s = s + "]"
        return s

    fn is_contiguous(self, shape: Shape) -> Bool:
        var want = Strides.row_major_for(shape)
        if len(want.steps) != len(self.steps): return False
        var i = 0
        while i < len(self.steps):
            if self.steps[i] != want.steps[i]: return False
            i += 1
        return True

# For compatibility with older code where Strides(shape) computed row-major:
# Provide an alternate constructor wrapper.
@staticmethod
fn Strides_from_shape(shape: Shape) -> Strides:
    return Strides.row_major_for(shape)

# -------------------------
# Slicing
# -------------------------

struct Slice(Copyable, Movable, EqualityComparable):
    var start: Int
    var stop: Int
    var step: Int

    fn __init__(out self, start: Int = 0, stop: Int = -1, step: Int = 1):
        self.start = start
        self.stop = stop
        self.step = (step <= 0) ? 1 : step

@staticmethod
fn normalize_slice(s: Slice, axis_len: Int) -> (Int, Int, Int, Int):
    var start = s.start
    var stop = s.stop
    var step = (s.step <= 0) ? 1 : s.step
    if start < 0: start = 0
    if stop < 0 or stop > axis_len: stop = axis_len
    if start > axis_len: start = axis_len
    if stop < start: stop = start
    var new_len = (stop - start + step - 1) // step
    return (start, stop, step, new_len)

@staticmethod
fn apply_slices(shape: Shape, strides: Strides, offset: Int, specs: List[Slice]) -> (Shape, Strides, Int):
    var r = shape.rank()
    if len(specs) != r:
        return (shape, strides, offset)
    var new_dims = List[Int]()
    var new_steps = List[Int]()
    var new_off = offset
    var i = 0
    while i < r:
        var (start, stop, step, new_len) = normalize_slice(specs[i], shape.dims[i])
        new_off = new_off + start * strides.steps[i]
        new_dims.append(new_len)
        new_steps.append(strides.steps[i] * step)
        i += 1
    return (Shape(dims=new_dims), Strides(steps=new_steps), new_off)

# -------------------------
# Index conversions
# -------------------------

@staticmethod
fn flatten_index(shape: Shape, strides: Strides, index: List[Int], offset: Int = 0) -> Int:
    var off = offset
    var r = shape.rank()
    var i = 0
    while i < r:
        off = off + index[i] * strides.steps[i]
        i += 1
    return off

@staticmethod
fn unflatten_index(shape: Shape, linear: Int) -> List[Int]:
    var r = shape.rank()
    var idx = List[Int]()
    idx.reserve(r)
    var i = 0
    while i < r:
        idx.append(0)
        i += 1
    if r == 0: return idx
    var st = Strides.row_major_for(shape)
    var rem = linear
    i = 0
    while i < r:
        var s = st.steps[i]
        idx[i] = rem // s
        rem = rem % s
        i += 1
    return idx

# -------------------------
# Broadcasting helpers
# -------------------------

@staticmethod
fn broadcast_shapes(a: Shape, b: Shape) -> Result[Shape]:
    var ra = a.rank()
    var rb = b.rank()
    var r = ra
    if rb > r: r = rb

    var aa = a.align_right(r)
    var bb = b.align_right(r)

    var out = List[Int]()
    var i = r - 1
    while i >= 0:
        var da = aa.dims[i]
        var db = bb.dims[i]
        if da == db:
            out.insert(0, da)
        elif da == 1:
            out.insert(0, db)
        elif db == 1:
            out.insert(0, da)
        else:
            return Result[Shape].fail(MomijoError.invalid_argument("shapes not broadcastable: " + a.to_string() + " vs " + b.to_string(), "momijo.core.shape"), a)
        i -= 1
    return Result[Shape].ok(Shape(dims=out))

@staticmethod
fn broadcast_strides(in_shape: Shape, in_strides: Strides, out_shape: Shape) -> Result[Strides]:
    # Requires in_shape broadcastable to out_shape. For broadcasted axes (size=1),
    # the stride becomes 0 (so repeated reads of the same element occur).
    if not in_shape.is_broadcastable_to(out_shape):
        return Result[Strides].fail(MomijoError.invalid_argument("shape not broadcastable to output", "momijo.core.shape"), Strides.row_major_for(out_shape))
    var r = out_shape.rank()
    var a = in_shape.align_right(r)
    var s = List[Int]()
    s.reserve(r)
    var i = 0
    while i < r:
        s.append(0)
        i += 1
    i = r - 1
    var j = in_shape.rank() - 1
    while i >= 0:
        var dim_out = out_shape.dims[i]
        var dim_in = a.dims[i]
        if dim_in == dim_out:
            # carry stride (need to map from original in_strides; if in_shape had fewer dims, align)
            var idx_in = j
            if idx_in < 0:
                s[i] = 0
            else:
                s[i] = in_strides.steps[idx_in]
                j -= 1
        elif dim_in == 1:
            s[i] = 0
            j -= 1
        else:
            # should not happen due to earlier check
            s[i] = 0
            j -= 1
        i -= 1
    return Result[Strides].ok(Strides(steps=s))
