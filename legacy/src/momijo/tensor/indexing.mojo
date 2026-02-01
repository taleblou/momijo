# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.tensor.indexing
# File:         src/momijo/tensor/indexing.mojo
#
# Description:
#   High-performance, generic indexing & slicing utilities for Tensor[T].
#   - Shape/Layout views, NDIter/NDLinearIter
#   - Materialized slice and view slice
#   - 1D/2D/3D/4D slicing, head/tail
#   - take/take_along_axis/scatter/gather_nd/scatter_nd
#   - rows/cols, diag/tril/triu/band, where/masked ops, nonzero/argwhere
#   - im2col/col2im (2D core)
#   - Blocked & unrolled GEMM (sgemm/dgemm) with contiguous/strided paths
#
# Notes:
#   - No 'let' and no 'assert'.
#   - English-only comments.

from collections.list import List

from momijo.tensor.tensor import (
    Tensor,
    TensorView,
)

from momijo.tensor.anytensor import (
    AnyTensor,
    AnyBuffer,
)
from momijo.tensor.broadcast import clamp,broadcast_shapes,clamp_axis
from momijo.tensor.creation import scalar_zero_tensor,zeros_with_shape

# Helpers (no wildcard imports)
from momijo.tensor.helpers import (
    copy_list_int,
    compute_row_major_strides,
    numel,
    normalize_axis,
    min_i,
    max_i,
    zero_scalar_of,
    unravel_index,
    is_row_major_contiguous,wrap_index,
)
from momijo.tensor.cast import *

from momijo.tensor.gpu.runtime import (
    gpu_available,
    device_ptr,
    device_ptr_mut,
    memset_f32,
    atomic_add_f32,
    # 1D launchers:
    _launch_1d_where_same_f32,_kernel_where_same_f32,Kernel1D_WhereSameF32

)

fn _numel(shape: List[Int]) -> Int:
    var p = 1
    var i = 0
    while i < len(shape):
        p = p * shape[i]
        i = i + 1
    return p

# row-major contiguous check for arbitrary rank
fn _is_row_major_contig(shape: List[Int], strides: List[Int]) -> Bool:
    var r = len(shape)
    if r != len(strides): return False
    if r == 0: return True
    var expected = 1
    var i = r - 1
    while i >= 0:
        if strides[i] != expected:
            return False
        expected = expected * shape[i]
        i = i - 1
    return True

# -----------------------------------------------------------------------------
# Shape / Layout
# -----------------------------------------------------------------------------

struct Shape(Copyable, Movable):
    var dims: List[Int]

    fn __init__(out self, dims: List[Int]):
        self.dims = copy_list_int(dims)

    @always_inline
    fn ndim(self) -> Int:
        return len(self.dims)

    fn numel(self) -> Int:
        var n = 1
        var i = 0
        while i < len(self.dims):
            n = n * self.dims[i]
            i += 1
        return n

    fn __copyinit__(out self, other: Self):
        self.dims = copy_list_int(other.dims)

struct LayoutView(Copyable, Movable):
    var offset: Int
    var strides: List[Int]

    fn __init__(out self, offset: Int, strides: List[Int]):
        self.offset = offset
        self.strides = copy_list_int(strides)

    fn __copyinit__(out self, other: Self):
        self.offset = other.offset
        self.strides = copy_list_int(other.strides)

# -----------------------------------------------------------------------------
# NDIter (multi-index) + NDLinearIter (linear offset with strides)
# -----------------------------------------------------------------------------

struct NDIter(Copyable, Movable):
    var shape: Shape
    var idx: List[Int]
    var started: Bool
    var finished: Bool

    fn __init__(out self, shape: Shape):
        self.shape = Shape(shape.dims)
        self.idx = List[Int]()
        var r = shape.ndim()
        self.idx.reserve(r)
        var k = 0
        while k < r:
            self.idx.append(0)
            k += 1
        self.started = False
        self.finished = (shape.numel() == 0)

    fn __copyinit__(out self, other: Self):
        self.shape = Shape(copy_list_int(other.shape.dims))
        self.idx = copy_list_int(other.idx)
        self.started = other.started
        self.finished = other.finished

    @always_inline
    fn reset(mut self):
        var r = self.shape.ndim()
        var i = 0
        while i < r:
            self.idx[i] = 0
            i += 1
        self.started = False
        self.finished = (self.shape.numel() == 0)

    @always_inline
    fn next_inplace(mut self, out out_idx: List[Int]) -> Bool:
        if self.finished:
            return False
        if not self.started:
            out_idx.clear()
            var n = len(self.idx)
            out_idx.reserve(n)
            var i = 0
            while i < n:
                out_idx.append(self.idx[i])
                i += 1
            self.started = True
            if self.shape.ndim() == 0:
                self.finished = True
            return True

        var r = self.shape.ndim()
        var ax = r - 1
        while ax >= 0:
            self.idx[ax] += 1
            if self.idx[ax] < self.shape.dims[ax]:
                out_idx.clear()
                var j = 0
                while j < r:
                    out_idx.append(self.idx[j])
                    j += 1
                return True
            self.idx[ax] = 0
            ax -= 1
        self.finished = True
        return False

    fn next(mut self) -> (Bool, List[Int]):
        var out_idx = List[Int]()
        var ok = self.next_inplace(out_idx)
        return (ok, out_idx)

struct NDLinearIter(Copyable, Movable):
    var shape: Shape
    var view: LayoutView
    var idx: List[Int]
    var lin: Int
    var started: Bool
    var finished: Bool

    fn __init__(out self, shape: Shape, view: LayoutView):
        self.shape = Shape(shape.dims)
        self.view = LayoutView(view.offset, view.strides)
        self.idx = List[Int]()
        var r = shape.ndim()
        self.idx.reserve(r)
        var i = 0
        while i < r:
            self.idx.append(0)
            i += 1
        self.lin = view.offset
        self.started = False
        self.finished = (shape.numel() == 0)

    fn __copyinit__(out self, other: Self):
        self.shape = Shape(copy_list_int(other.shape.dims))
        self.view = LayoutView(other.view.offset, copy_list_int(other.view.strides))
        self.idx = copy_list_int(other.idx)
        self.lin = other.lin
        self.started = other.started
        self.finished = other.finished

    @always_inline
    fn reset(mut self):
        var r = self.shape.ndim()
        var i = 0
        while i < r:
            self.idx[i] = 0
            i += 1
        self.lin = self.view.offset
        self.started = False
        self.finished = (self.shape.numel() == 0)

    @always_inline
    fn next_inplace(mut self, out out_idx: List[Int]) -> (Bool, Int):
        if self.finished:
            return (False, 0)
        if not self.started:
            out_idx.clear()
            var n = len(self.idx)
            out_idx.reserve(n)
            var i = 0
            while i < n:
                out_idx.append(self.idx[i])
                i += 1
            self.started = True
            if self.shape.ndim() == 0:
                self.finished = True
            return (True, self.lin)

        var r = self.shape.ndim()
        var ax = r - 1
        while ax >= 0:
            self.idx[ax] += 1
            if self.idx[ax] < self.shape.dims[ax]:
                self.lin = self.lin + self.view.strides[ax]
                out_idx.clear()
                var j = 0
                while j < r:
                    out_idx.append(self.idx[j])
                    j += 1
                return (True, self.lin)
            self.lin = self.lin - (self.shape.dims[ax] - 1) * self.view.strides[ax]
            self.idx[ax] = 0
            ax -= 1
        self.finished = True
        return (False, 0)

    fn next(mut self) -> (Bool, Int, List[Int]):
        var out_idx = List[Int]()
        var (ok, lin) = self.next_inplace(out_idx)
        return (ok, lin, out_idx)

# -----------------------------------------------------------------------------
# Slice (materialize + view)
# -----------------------------------------------------------------------------
fn tensor_slice[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], begin: List[Int], end: List[Int], step: List[Int]
) -> Tensor[T]:
    var sh = x._shape.copy()
    var st = x._strides.copy()
    var rank = len(sh)

    var b = List[Int]()
    var e = List[Int]()
    var s = List[Int]()
    b.reserve(rank)
    e.reserve(rank)
    s.reserve(rank)

    var full = True
    var i = 0
    while i < rank:
        var bi = 0
        if i < len(begin):
            bi = begin[i]
        var ei = sh[i]
        if i < len(end):
            ei = end[i]
        var si = 1
        if i < len(step):
            si = step[i]
        if si == 0:
            si = 1
        if bi < 0:
            bi = bi + sh[i]
        if ei < 0:
            ei = ei + sh[i]
        bi = clamp(bi, 0, sh[i])
        ei = clamp(ei, 0, sh[i])
        b.append(bi)
        e.append(ei)
        s.append(si)
        if not (bi == 0 and ei == sh[i] and si == 1):
            full = False
        i += 1

    if full:
        return x.copy()

    var out_shape = List[Int]()
    out_shape.reserve(rank)
    i = 0
    while i < rank:
        var len_i = 0
        var bi2 = b[i]
        var ei2 = e[i]
        var si2 = s[i]
        if si2 > 0:
            if ei2 > bi2:
                var d = ei2 - bi2
                len_i = d // si2
                if d % si2 != 0:
                    len_i += 1
        else:
            var stp = -si2
            if ei2 < bi2:
                var d2 = bi2 - ei2
                len_i = d2 // stp
                if d2 % stp != 0:
                    len_i += 1
        if len_i < 0:
            len_i = 0
        out_shape.append(len_i)
        i += 1

    var out_n = 1
    i = 0
    while i < rank:
        out_n = out_n * out_shape[i]
        i += 1

    var out_data = List[T]()
    out_data.reserve(out_n)
    var rm = compute_row_major_strides(out_shape)
    var oi = 0
    while oi < out_n:
        var ai = 0
        var d = 0
        while d < rank:
            var stride = rm[d]
            var sidx = 0
            if stride != 0 and out_shape[d] != 0:
                sidx = (oi // stride) % out_shape[d]
            var src = b[d] + sidx * s[d]
            ai = ai + src * st[d]
            d += 1
        out_data.append(x._data[ai])
        oi += 1

    return Tensor[T](out_shape, out_data)


fn slice_view[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], begin: List[Int], end: List[Int], step: List[Int]
) -> TensorView[T]:
    var rank = len(x._shape)
    var b = List[Int]()
    var e = List[Int]()
    var s = List[Int]()
    var i = 0
    while i < rank:
        var bi = 0
        if i < len(begin):
            bi = begin[i]
        var ei = x._shape[i]
        if i < len(end):
            ei = end[i]
        var si = 1
        if i < len(step):
            si = step[i]
        if si == 0:
            si = 1
        if bi < 0:
            bi = bi + x._shape[i]
        if ei < 0:
            ei = ei + x._shape[i]
        bi = clamp(bi, 0, x._shape[i])
        ei = clamp(ei, 0, x._shape[i])
        b.append(bi)
        e.append(ei)
        s.append(si)
        i += 1

    var out = TensorView[T](x)
    out._shape = List[Int]()
    out._strides = List[Int]()
    out._offset = 0

    i = 0
    while i < rank:
        var len_i = 0
        var bi2 = b[i]
        var ei2 = e[i]
        var si2 = s[i]
        if si2 > 0:
            if ei2 > bi2:
                var d = ei2 - bi2
                len_i = d // si2
                if d % si2 != 0:
                    len_i += 1
        else:
            var stp = -si2
            if ei2 < bi2:
                var d2 = bi2 - ei2
                len_i = d2 // stp
                if d2 % stp != 0:
                    len_i += 1
        if len_i < 0:
            len_i = 0
        out._shape.append(len_i)
        var stride_mul = s[i]
        if stride_mul == 0:
            stride_mul = 1
        out._strides.append(x._strides[i] * stride_mul)
        out._offset = out._offset + bi2 * x._strides[i]
        i += 1
    return out.copy()

# -----------------------------------------------------------------------------
# Public slicing helpers (1D/2D/3D/4D wrappers)
# -----------------------------------------------------------------------------

fn slice1d[T: ImplicitlyCopyable & Copyable & Movable](t: Tensor[T], start: Int, stop: Int, step: Int = 1) -> Tensor[T]:
    var shp = t._shape
    if len(shp) == 0:
        return Tensor[T](shape=[0], fill=zero_scalar_of[T](cast_from_f64[T]))
    var n = shp[0]
    var nr = norm_range(start, stop, step, n)  # [a,b,s]
    var a = nr[0]
    var b = nr[1]
    var s = nr[2]
    if s <= 0 or a >= b:
        return Tensor[T](shape=[0], fill=zero_scalar_of[T](cast_from_f64[T]))
    return tensor_slice[T](t, [a], [b], [s])

fn slice2d[T: ImplicitlyCopyable & Copyable & Movable](
    t: Tensor[T], r0: Int, r1: Int, c0: Int, c1: Int, rs: Int = 1, cs: Int = 1
) -> Tensor[T]:
    var shp = t._shape
    if len(shp) < 2:
        return Tensor[T](shape=[0, 0], fill=zero_scalar_of[T](cast_from_f64[T]))
    var R = shp[0]
    var C = shp[1]
    var ra = norm_range(r0, (r1 if r1 != 0 else R), rs, R)
    var ca = norm_range(c0, (c1 if c1 != 0 else C), cs, C)
    if ra[2] <= 0 or ca[2] <= 0 or ra[0] >= ra[1] or ca[0] >= ca[1]:
        return Tensor[T](shape=[0, 0], fill=zero_scalar_of[T](cast_from_f64[T]))
    return tensor_slice[T](t, [ra[0], ca[0]], [ra[1], ca[1]], [ra[2], ca[2]])

fn slice3d[T: ImplicitlyCopyable & Copyable & Movable](
    t: Tensor[T], d: Int, r0: Int = 0, r1: Int = 0, c0: Int = 0, c1: Int = 0, rs: Int = 1, cs: Int = 1
) -> Tensor[T]:
    var shp = t._shape
    if len(shp) < 3:
        return Tensor[T](shape=[0, 0], fill=zero_scalar_of[T](cast_from_f64[T]))
    var D = shp[0]
    var R = shp[1]
    var C = shp[2]
    var di = clamp(d, 0, max_i(0, D - 1))
    var ra = norm_range(r0, (r1 if r1 != 0 else R), rs, R)
    var ca = norm_range(c0, (c1 if c1 != 0 else C), cs, C)
    var y = tensor_slice[T](t, [di, ra[0], ca[0]], [di + 1, ra[1], ca[1]], [1, ra[2], ca[2]])
    return y.reshape([ra[1] - ra[0], ca[1] - ca[0]])

fn slice4d[T: ImplicitlyCopyable & Copyable & Movable](
    t: Tensor[T],
    a0: Int, a1: Int, b0: Int, b1: Int, c0: Int, c1: Int, d0: Int, d1: Int,
    as_: Int = 1, bs_: Int = 1, cs_: Int = 1, ds_: Int = 1
) -> Tensor[T]:
    var shp = t._shape
    if len(shp) < 4:
        return Tensor[T](shape=[0, 0, 0, 0], fill=zero_scalar_of[T](cast_from_f64[T]))
    var A = shp[0]
    var B = shp[1]
    var C = shp[2]
    var D = shp[3]
    var aa = norm_range(a0, (a1 if a1 != 0 else A), as_, A)
    var bb = norm_range(b0, (b1 if b1 != 0 else B), bs_, B)
    var cc = norm_range(c0, (c1 if c1 != 0 else C), cs_, C)
    var dd = norm_range(d0, (d1 if d1 != 0 else D), ds_, D)
    return tensor_slice[T](
        t,
        [aa[0], bb[0], cc[0], dd[0]],
        [aa[1], bb[1], cc[1], dd[1]],
        [aa[2], bb[2], cc[2], dd[2]]
    )

# -----------------------------------------------------------------------------
# Head / Tail
# -----------------------------------------------------------------------------

fn head[T: ImplicitlyCopyable & Copyable & Movable](t: Tensor[T], n: Int) -> Tensor[T]:
    return slice1d[T](t, 0, n, 1)

fn tail[T: ImplicitlyCopyable & Copyable & Movable](t: Tensor[T], n: Int) -> Tensor[T]:
    var shp = t._shape
    if len(shp) == 0:
        return Tensor[T](shape=[0], fill=zero_scalar_of[T](cast_from_f64[T]))
    var N = shp[0]
    var s = N - n
    if s < 0:
        s = 0
    return slice1d[T](t, s, N, 1)

# -----------------------------------------------------------------------------
# Take / Gather
# -----------------------------------------------------------------------------



# Gather along a given axis, preserving other dimensions.
# Output shape = pre + index.shape + post
# Gather along 'axis' preserving other dims.
# out_shape = x.shape[:axis] + index.shape + x.shape[axis+1:]
@always_inline
fn gather[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], axis: Int, index: Tensor[Int]
) -> Tensor[T]:
    var r = len(x._shape)
    var ax = normalize_axis(axis, r)

    var out_shape = index._shape.copy()
    if numel(out_shape) == 0 or numel(x._shape) == 0:
        var empty = List[T]()
        var shp0 = List[Int](); shp0.append(0)
        var str0 = compute_row_major_strides(shp0)
        return Tensor[T](empty, shp0, str0, 0)

    var out_data = List[T]()
    var n = numel(out_shape)
    out_data.reserve(n)

    var rm_idx = _row_major_multipliers(index._shape)

    var pre = ax
    var post = r - ax - 1

    var lin = 0
    while lin < n:
        # coords of index at linear 'lin'
        var icoord = List[Int]()
        icoord.reserve(len(index._shape))
        var d = 0
        while d < len(index._shape):
            var s = 0
            if index._shape[d] != 0 and rm_idx[d] != 0:
                s = (lin // rm_idx[d]) % index._shape[d]
            icoord.append(s)
            d += 1

        # read index value (respect strides/offset)
        var idx_off = _offset_from_linear(index._shape, index._strides, index._offset, rm_idx, lin)
        var k = index._data[idx_off]
        if k < 0: k = k + x._shape[ax]
        if k < 0: k = 0
        var mx = x._shape[ax] - 1
        if k > mx: k = mx

        # build x coords and compute offset inline
        var xoff = x._offset
        var dd = 0
        while dd < pre:
            xoff = xoff + icoord[dd] * x._strides[dd]
            dd += 1
        xoff = xoff + k * x._strides[ax]
        var ee = 0
        while ee < post:
            var dim = pre + 1 + ee
            xoff = xoff + icoord[dim] * x._strides[dim]
            ee += 1

        out_data.append(x._data[xoff])
        lin += 1
    var out_strides = compute_row_major_strides(out_shape)
    return Tensor[T](out_data, out_shape, out_strides, 0)



# -----------------------------------------------------------------------------
# Dim-0 slice & last-dim planes (3D/4D)
# -----------------------------------------------------------------------------

fn slice_dim0[T: ImplicitlyCopyable & Copyable & Movable](a: Tensor[T], i: Int) -> Tensor[T]:
    var shp = a._shape.copy()
    if len(shp) != 3:
        return Tensor[T]([0, 0], List[T]())
    var B = shp[0]
    var M = shp[1]
    var N = shp[2]
    if B <= 0 or M <= 0 or N <= 0:
        return Tensor[T]([0, 0], List[T]())

    var ii = i
    if ii < 0:
        ii = 0
    if ii > B - 1:
        ii = B - 1

    var starts = List[Int](); starts.append(ii); starts.append(0);  starts.append(0)
    var stops  = List[Int](); stops.append(ii + 1); stops.append(M); stops.append(N)
    var steps  = List[Int](); steps.append(1); steps.append(1); steps.append(1)

    return tensor_slice[T](a, starts, stops, steps).reshape([M, N])


fn get_last_dim_plane3[T: ImplicitlyCopyable & Copyable & Movable](a: Tensor[T], dim: Int, index: Int) -> Tensor[T]:
    var shp = a._shape.copy()
    if len(shp) != 3 or dim != 2:
        return Tensor[T](shape=[0, 0], fill=zero_scalar_of[T](cast_from_f64[T]))
    var B = shp[0]
    var M = shp[1]
    var N = shp[2]
    if B <= 0 or M <= 0 or N <= 0:
        return Tensor[T](shape=[0, 0], fill=zero_scalar_of[T](cast_from_f64[T]))
    var k = clamp(index, 0, N - 1)
    return tensor_slice[T](a, [0, 0, k], [B, M, k + 1], [1, 1, 1]).reshape([B, M])

fn assign_last_dim_plane3[T: ImplicitlyCopyable & Copyable & Movable](mut a: Tensor[T], dim: Int, index: Int, rhs: Tensor[T]) -> Tensor[T]:
    var shp = a._shape.copy()
    if len(shp) != 3 or dim != 2:
        return a
    var B = shp[0]
    var M = shp[1]
    var N = shp[2]
    if B <= 0 or M <= 0 or N <= 0:
        return a
    var k = clamp(index, 0, N - 1)
    var r2 = rhs
    if len(rhs._shape) != 2 or rhs._shape[0] != B or rhs._shape[1] != M:
        r2 = rhs.broadcast_to([B, M])
    var i = 0
    while i < B:
        var j = 0
        while j < M:
            var dst = i * a._strides[0] + j * a._strides[1] + k * a._strides[2]
            var src = i * M + j
            a._data[dst] = r2._data[src]
            j += 1
        i += 1
    return a

fn get_last_dim_plane4[T: ImplicitlyCopyable & Copyable & Movable](a: Tensor[T], dim: Int, index: Int) -> Tensor[T]:
    var shp = a._shape.copy()
    if len(shp) != 4 or dim != 3:
        return Tensor[T](shape=[0, 0, 0], fill=zero_scalar_of[T](cast_from_f64[T]))
    var A = shp[0]
    var B = shp[1]
    var C = shp[2]
    var D = shp[3]
    if A <= 0 or B <= 0 or C <= 0 or D <= 0:
        return Tensor[T](shape=[0, 0, 0], fill=zero_scalar_of[T](cast_from_f64[T]))
    var k = clamp(index, 0, D - 1)
    return tensor_slice[T](a, [0, 0, 0, k], [A, B, C, k + 1], [1, 1, 1, 1]).reshape([A, B, C])

fn assign_last_dim_plane4[T: ImplicitlyCopyable & Copyable & Movable](mut a: Tensor[T], dim: Int, index: Int, rhs: Tensor[T]) -> Tensor[T]:
    var shp = a._shape.copy()
    if len(shp) != 4 or dim != 3:
        return a
    var A = shp[0]
    var B = shp[1]
    var C = shp[2]
    var D = shp[3]
    if A <= 0 or B <= 0 or C <= 0 or D <= 0:
        return a
    var k = clamp(index, 0, D - 1)
    var r2 = rhs
    if len(rhs._shape) != 3 or rhs._shape[0] != A or rhs._shape[1] != B or rhs._shape[2] != C:
        r2 = rhs.broadcast_to([A, B, C])
    var i = 0
    while i < A:
        var j = 0
        while j < B:
            var h = 0
            while h < C:
                var dst = i * a._strides[0] + j * a._strides[1] + h * a._strides[2] + k * a._strides[3]
                var src = (i * B + j) * C + h
                a._data[dst] = r2._data[src]
                h += 1
            j += 1
        i += 1
    return a

# -----------------------------------------------------------------------------
# Basic item access
# -----------------------------------------------------------------------------

fn getitem[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], idx: Int) -> T:
    var sh = x._shape.copy()
    var st = x._strides.copy()
    var r = len(sh)
    if r == 0:
        return x._data[0]
    var n = numel(sh)
    var i = clamp(idx, 0, n - 1)
    var rm = compute_row_major_strides(sh)
    var off = 0
    var d = 0
    while d < r:
        var stride = rm[d]
        var s = 0
        if stride != 0 and sh[d] != 0:
            s = (i // stride) % sh[d]
        off = off + s * st[d]
        d += 1
    return x._data[off]

fn setitem[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], idx: Int, v: T):
    var sh = x._shape.copy()
    var st = x._strides.copy()
    var r = len(sh)
    if r == 0:
        x._data[0] = v
        return
    var n = numel(sh)
    var i = clamp(idx, 0, n - 1)
    var rm = compute_row_major_strides(sh)
    var off = 0
    var d = 0
    while d < r:
        var stride = rm[d]
        var s = 0
        if stride != 0 and sh[d] != 0:
            s = (i // stride) % sh[d]
        off = off + s * st[d]
        d += 1
    x._data[off] = v


@always_inline
fn item(x: Tensor[Float64]) -> Float64:
    var n = numel(x._shape)
    if n == 0:
        return 0.0
    return x._data[0]

@always_inline
fn item(x: Tensor[Float32]) -> Float32:
    var n = numel(x._shape)
    if n == 0:
        return Float32(0.0)
    return x._data[0]

@always_inline
fn item(x: Tensor[Int]) -> Int:
    var n = numel(x._shape)
    if n == 0:
        return 0
    return x._data[0]

@always_inline
fn item(x: Tensor[Int32]) -> Int32:
    var n = numel(x._shape)
    if n == 0:
        return Int32(0)
    return x._data[0]

@always_inline
fn item(x: Tensor[Int16]) -> Int16:
    var n = numel(x._shape)
    if n == 0:
        return Int16(0)
    return x._data[0]

@always_inline
fn item(x: Tensor[Bool]) -> Bool:
    var n = numel(x._shape)
    if n == 0:
        return False
    return x._data[0]

fn at[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], i: Int, j: Int) -> T:
    var sh = x._shape.copy()
    if len(sh) < 2:
        return x._data[0]
    var r = sh[0]
    var c = sh[1]
    var ii = clamp(i, 0, r - 1)
    var jj = clamp(j, 0, c - 1)
    var st = x._strides.copy()
    return x._data[ii * st[0] + jj * st[1]]

fn set_at[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], i: Int, j: Int, v: T):
    var sh = x._shape.copy()
    if len(sh) < 2:
        x._data[0] = v
        return
    var r = sh[0]
    var c = sh[1]
    var ii = clamp(i, 0, r - 1)
    var jj = clamp(j, 0, c - 1)
    var st = x._strides.copy()
    x._data[ii * st[0] + jj * st[1]] = v

@always_inline
fn _clamp_row_index(i: Int, r: Int) -> Int:
    var ii = i
    if ii < 0:  ii = 0
    if ii >= r: ii = r - 1
    return ii

fn row[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], i: Int) -> Tensor[T]:
    var sh = x._shape.copy()
    if len(sh) < 2:
        return x.copy()

    var r  = sh[0]
    var c  = sh[1]
    var ii = _clamp_row_index(i, r)

    var st  = x._strides.copy()
    var off = x._offset + ii * st[0]

    var buf = List[T]()
    buf.reserve(c)

    var j = 0
    while j < c:
        buf.append(x._data[off + j * st[1]])
        j += 1

    # سازنده‌ی (data, shape) -> row-major, offset=0
    return Tensor[T](data=buf, shape=[c])


fn col[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], j: Int) -> Tensor[T]:
    var sh = x._shape.copy()
    if len(sh) < 2:
        return x.copy()

    var r = sh[0]
    var c = sh[1]
    if r <= 0 or c <= 0:
        return Tensor[T](List[T]())   # empty 1D

    var jj = clamp(j, 0, c - 1)
    var st = x._strides.copy()

    var out_data = List[T]()
    out_data.reserve(r)

    var step = st[0]
    var idx  = jj * st[1]

    var i = 0
    var lim = (r // 8) * 8
    while i < lim:
        out_data.append(x._data[idx]); idx += step
        out_data.append(x._data[idx]); idx += step
        out_data.append(x._data[idx]); idx += step
        out_data.append(x._data[idx]); idx += step
        out_data.append(x._data[idx]); idx += step
        out_data.append(x._data[idx]); idx += step
        out_data.append(x._data[idx]); idx += step
        out_data.append(x._data[idx]); idx += step
        i += 8
    while i < r:
        out_data.append(x._data[idx]); idx += step
        i += 1

    return Tensor[T](out_data)  # 1D shape=[r]

# -----------------------------------------------------------------------------
# Diagonals & bands
# -----------------------------------------------------------------------------

fn diag[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], k: Int = 0) -> Tensor[T]:
    var sh = x._shape.copy()
    if len(sh) < 2:
        return Tensor[T](shape=[0], fill=zero_scalar_of[T](cast_from_f64[T]))
    var r = sh[0]
    var c = sh[1]
    var st = x._strides.copy()
    var i0 = 0
    var j0 = 0
    if k > 0:
        j0 = k
    if k < 0:
        i0 = -k
    var n = r - i0
    var m = c - j0
    if n <= 0 or m <= 0:
        return Tensor[T](shape=[0], fill=zero_scalar_of[T](cast_from_f64[T]))
    var d = n if n < m else m
    var out = Tensor[T](shape=[d], fill=zero_scalar_of[T](cast_from_f64[T]))
    var t = 0
    while t < d:
        out._data[t] = x._data[(i0 + t) * st[0] + (j0 + t) * st[1]]
        t += 1
    return out.copy()

fn diagflat[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[T]:
    var n = numel(x._shape)
    var out = Tensor[T](shape=[n, n], fill=zero_scalar_of[T](cast_from_f64[T]))
    var st = out._strides
    var i = 0
    while i < n:
        out._data[i * st[0] + i * st[1]] = x._data[i]
        i += 1
    return out.copy()

fn fill_diagonal[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], v: T, k: Int = 0):
    var sh = x._shape.copy()
    if len(sh) < 2:
        return
    var r = sh[0]
    var c = sh[1]
    var st = x._strides.copy()
    var i0 = 0
    var j0 = 0
    if k > 0:
        j0 = k
    if k < 0:
        i0 = -k
    var n = r - i0
    var m = c - j0
    if n <= 0 or m <= 0:
        return
    var d = n if n < m else m
    var t = 0
    while t < d:
        x._data[(i0 + t) * st[0] + (j0 + t) * st[1]] = v
        t += 1

fn tril[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], k: Int = 0) -> Tensor[T]:
    var sh = x._shape.copy()
    if len(sh) < 2:
        return x.copy()
    var r = sh[0]
    var c = sh[1]
    var out = Tensor[T](shape=[r, c], fill=zero_scalar_of[T](cast_from_f64[T]))
    var stx = x._strides.copy()
    var sto = out._strides
    var i = 0
    while i < r:
        var j = 0
        while j < c:
            if j - i <= k:
                out._data[i * sto[0] + j * sto[1]] = x._data[i * stx[0] + j * stx[1]]
            j += 1
        i += 1
    return out.copy()

fn triu[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], k: Int = 0) -> Tensor[T]:
    var sh = x._shape.copy()
    if len(sh) < 2:
        return x.copy()
    var r = sh[0]
    var c = sh[1]
    var out = Tensor[T](shape=[r, c], fill=zero_scalar_of[T](cast_from_f64[T]))
    var stx = x._strides.copy()
    var sto = out._strides
    var i = 0
    while i < r:
        var j = 0
        while j < c:
            if j - i >= k:
                out._data[i * sto[0] + j * sto[1]] = x._data[i * stx[0] + j * stx[1]]
            j += 1
        i += 1
    return out.copy()

fn band[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], k1: Int, k2: Int) -> Tensor[T]:
    var sh = x._shape.copy()
    if len(sh) < 2:
        return x.copy()
    var r = sh[0]
    var c = sh[1]
    var out = Tensor[T](shape=[r, c], fill=zero_scalar_of[T](cast_from_f64[T]))
    var stx = x._strides.copy()
    var sto = out._strides
    var i = 0
    while i < r:
        var j = 0
        while j < c:
            var d = j - i
            if d >= k1 and d <= k2:
                out._data[i * sto[0] + j * sto[1]] = x._data[i * stx[0] + j * stx[1]]
            j += 1
        i += 1
    return out.copy()

# -----------------------------------------------------------------------------
# Nonzero / masked / where
# -----------------------------------------------------------------------------

fn nonzero[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[Int32]:
    var sh = x._shape.copy()
    var r = len(sh)
    var n = numel(sh)
    var rows = List[Int32]()
    rows.reserve(n * r)
    var rm = compute_row_major_strides(sh)
    var i = 0
    while i < n:
        var off = 0
        var d = 0
        while d < r:
            var stride = rm[d]
            var s = 0
            if stride != 0 and sh[d] != 0:
                s = (i // stride) % sh[d]
            off = off + s * x._strides[d]
            d += 1
        if x._data[off] != zero_scalar_of[T](cast_from_f64[T]):
            var dd = 0
            while dd < r:
                var stride2 = rm[dd]
                var s2 = 0
                if stride2 != 0 and sh[dd] != 0:
                    s2 = (i // stride2) % sh[dd]
                rows.append(Int32(s2))
                dd += 1
        i += 1
    var out_rows = len(rows) // r
    var out = Tensor[Int32](shape=[out_rows, r], fill=0)
    var yi = 0
    while yi < out_rows:
        var dj = 0
        while dj < r:
            out._data[yi * out._strides[0] + dj * out._strides[1]] = rows[yi * r + dj]
            dj += 1
        yi += 1
    return out.copy()

@always_inline
fn masked_select[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    mask: Tensor[Int]
) -> Tensor[T]:
    # Broadcast logical shape
    var br = broadcast_shapes(x._shape, mask._shape)
    var sh = br.shape.copy()                # NOTE: use the correct field name from your BroadcastResult

    var n = numel(sh)
    # Empty fast-path: return a []-length 1D tensor
    if n == 0:
        var shp0 = List[Int](); shp0.append(0)
        var str0 = compute_row_major_strides(shp0)
        var dat0 = List[T]()        # empty payload
        return Tensor[T](dat0, shp0, str0, 0)

    var rm = compute_row_major_strides(sh)

    # We don't know the kept count upfront; append then build exact-sized tensor.
    var out_list = List[T]()

    var st_x   = x._strides.copy()
    var st_m   = mask._strides.copy()
    var base_x = x._offset
    var base_m = mask._offset

    var i = 0
    while i < n:
        var offx = base_x
        var offm = base_m

        # Map the i-th logical index to x/mask linear offsets via strides.
        var d = 0
        while d < len(sh):
            var stride = rm[d]
            var s = 0
            if stride != 0 and sh[d] != 0:
                s = (i // stride) % sh[d]

            # Map to x-dimension (right-aligned broadcast)
            var jx = d + len(x._shape) - len(sh)
            var ix = s
            if jx < 0 or x._shape[jx] == 1:
                ix = 0
            if jx >= 0:
                offx = offx + ix * st_x[jx]

            # Map to mask-dimension
            var jm = d + len(mask._shape) - len(sh)
            var im = s
            if jm < 0 or mask._shape[jm] == 1:
                im = 0
            if jm >= 0:
                offm = offm + im * st_m[jm]

            d += 1

        # Keep if mask != 0
        if mask._data[offm] != 0:
            out_list.append(x._data[offx])

        i += 1

    # Build the output tensor [k], k = len(out_list)
    var out_shape = List[Int](); out_shape.append(len(out_list))
    var out_strides = compute_row_major_strides(out_shape)
    return Tensor[T](out_list.copy(), out_shape, out_strides, 0)

@always_inline
fn masked_fill[T: ImplicitlyCopyable & Copyable & Movable](
    mut x: Tensor[T], mask: Tensor[Int], value: T
) -> None:
    var br = broadcast_shapes(x._shape, mask._shape)
    var sh = br.shape.copy()
    var n  = numel(sh)
    if n == 0: return

    # Fast path: same shape, both contiguous, same layout
    var expect = compute_row_major_strides(sh)
    var contig = (x._offset == 0 and mask._offset == 0)
    contig = contig and (x._strides == expect) and (mask._strides == expect)
    if contig:
        var i = 0
        while i < n:
            if mask._data[i] != 0:
                x._data[i] = value
            i += 1
        return

    # General ND path (stride-aware)
    var rm    = compute_row_major_strides(sh)
    var st_x  = x._strides.copy()
    var st_m  = mask._strides.copy()
    var base_x = x._offset
    var base_m = mask._offset

    var idx = 0
    var L = len(sh)
    while idx < n:
        var offx = base_x
        var offm = base_m

        var d = 0
        while d < L:
            var stride = rm[d]
            var s = 0
            if stride != 0 and sh[d] != 0:
                s = (idx // stride) % sh[d]

            var jx = d + len(x._shape) - L
            var ix = s
            if jx < 0 or x._shape[jx] == 1:
                  ix = 0
            if jx >= 0: offx = offx + ix * st_x[jx]

            var jm = d + len(mask._shape) - L
            var im = s
            if jm < 0 or mask._shape[jm] == 1:
                  im = 0
            if jm >= 0: offm = offm + im * st_m[jm]

            d += 1

        if mask._data[offm] != 0:
            x._data[offx] = value

        idx += 1



# ==========================
# WHERE with broadcasting
# ==========================
@always_inline
fn where[T: ImplicitlyCopyable & Copyable & Movable](
    cond: Tensor[Int], x: Tensor[T], y: Tensor[T],
    from_f64: fn (Float64) -> T              # F64 -> T converter (required)
) -> Tensor[T]:
    # 1) Broadcast planning between x and y

    var br= broadcast_shapes(x._shape, y._shape)
    var ok_xy     = br.ok
    var xy_shape  = br.shape.copy()
    #var x_pad     = br.lhs_padded.copy()
    #var y_pad     = br.rhs_padded.copy()
    if not ok_xy:
        # Not broadcastable -> return scalar-zero tensor (shape=[])
        return scalar_zero_tensor[T](from_f64)

    # 2) Broadcast planning with cond

    var br2 = broadcast_shapes(xy_shape, cond._shape)
    var ok_all  = br2.ok
    var out_shape = br2.shape.copy()
    #var tmp1  = br2.lhs_padded.copy()
    #var tmp2  = br2.rhs_padded.copy()
    if not ok_all:
        # Not broadcastable -> scalar-zero tensor
        return scalar_zero_tensor[T](from_f64)

    var r = len(out_shape)
    var n = numel(out_shape)

    # 3) Allocate output as zeros (uses provided converter)
    var out = zeros_with_shape[T](out_shape, from_f64)
    if n == 0:
        # Zero-size tensor (some dim == 0)
        return out.copy()

    # 4) Fast path: equal shapes + row-major contiguous
    if x._shape == out_shape and
       y._shape == out_shape and
       cond._shape == out_shape and
       is_row_major_contiguous(x._shape, x._strides) and
       is_row_major_contiguous(y._shape, y._strides) and
       is_row_major_contiguous(cond._shape, cond._strides) and
       is_row_major_contiguous(out._shape, out._strides):

        var i = 0
        var un = 8
        var lim = (n // un) * un
        while i < lim:
            if cond._data[i] != 0:         out._data[i]     = x._data[i]
            else:                          out._data[i]     = y._data[i]
            if cond._data[i + 1] != 0:     out._data[i + 1] = x._data[i + 1]
            else:                          out._data[i + 1] = y._data[i + 1]
            if cond._data[i + 2] != 0:     out._data[i + 2] = x._data[i + 2]
            else:                          out._data[i + 2] = y._data[i + 2]
            if cond._data[i + 3] != 0:     out._data[i + 3] = x._data[i + 3]
            else:                          out._data[i + 3] = y._data[i + 3]
            if cond._data[i + 4] != 0:     out._data[i + 4] = x._data[i + 4]
            else:                          out._data[i + 4] = y._data[i + 4]
            if cond._data[i + 5] != 0:     out._data[i + 5] = x._data[i + 5]
            else:                          out._data[i + 5] = y._data[i + 5]
            if cond._data[i + 6] != 0:     out._data[i + 6] = x._data[i + 6]
            else:                          out._data[i + 6] = y._data[i + 6]
            if cond._data[i + 7] != 0:     out._data[i + 7] = x._data[i + 7]
            else:                          out._data[i + 7] = y._data[i + 7]
            i += un
        while i < n:
            if cond._data[i] != 0:
                out._data[i] = x._data[i]
            else:
                out._data[i] = y._data[i]
            i += 1
        return out.copy()

    # 5) General broadcast path: build aligned strides against out_shape
    var xr = len(x._shape)
    var yr = len(y._shape)
    var cr = len(cond._shape)

    var sx = List[Int]()  # stride per out-dim for x
    var sy = List[Int]()  # stride per out-dim for y
    var sc = List[Int]()  # stride per out-dim for cond
    var so = List[Int]()  # stride per out-dim for out
    sx.reserve(r); sy.reserve(r); sc.reserve(r); so.reserve(r)

    var i = 0
    while i < r:
        var ox = 0
        var oy = 0
        var oc = 0

        var jx = i + xr - r
        var jy = i + yr - r
        var jc = i + cr - r

        if jx >= 0:
            if x._shape[jx] == 1: ox = 0
            else:                 ox = x._strides[jx]
        if jy >= 0:
            if y._shape[jy] == 1: oy = 0
            else:                 oy = y._strides[jy]
        if jc >= 0:
            if cond._shape[jc] == 1: oc = 0
            else:                    oc = cond._strides[jc]

        sx.append(ox)
        sy.append(oy)
        sc.append(oc)
        so.append(out._strides[i])

        i += 1

    # 6) N-D index walk with stride offsets
    var offx = 0
    var offy = 0
    var offc = 0
    var offo = 0

    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    while k < n:
        if cond._data[offc] != 0:
            out._data[offo] = x._data[offx]
        else:
            out._data[offo] = y._data[offy]

        var d = r - 1
        while d >= 0:
            idx[d] += 1
            offx += sx[d]
            offy += sy[d]
            offc += sc[d]
            offo += so[d]

            if idx[d] < out_shape[d]:
                break

            var span = out_shape[d]
            idx[d] = 0
            offx -= sx[d] * span
            offy -= sy[d] * span
            offc -= sc[d] * span
            offo -= so[d] * span
            d -= 1

        k += 1

    return out.copy()
# ---------------------------------------------------
# where_f32: CPU fast-path + general broadcast
# ---------------------------------------------------

# ---------- where() CPU ----------
fn where_f32_cpu(
    cond: tensor.Tensor[Int],
    x: tensor.Tensor[Float32],
    y: tensor.Tensor[Float32]
) -> tensor.Tensor[Float32]:
    var br = broadcast_shapes(x._shape, y._shape)
    if not br.ok:
        return tensor.zeros([0])
    var xy_shape = br.shape.copy()

    var br2 = broadcast_shapes(xy_shape, cond._shape)
    if not br2.ok:
        return tensor.zeros([0])

    var out_shape = br2.shape.copy()
    var n = _numel(out_shape)
    var out = tensor.zeros(out_shape.copy())
    if n == 0:
        return out.copy()

    # fast-path: same shape + contiguous
    var fast = (x._shape == out_shape and y._shape == out_shape and cond._shape == out_shape)
    if fast: fast = fast and _is_row_major_contig(x._shape, x._strides)
    if fast: fast = fast and _is_row_major_contig(y._shape, y._strides)
    if fast: fast = fast and _is_row_major_contig(cond._shape, cond._strides)
    if fast: fast = fast and _is_row_major_contig(out._shape, out._strides)

    if fast:
        var i = 0
        var un = 8
        var lim = (n // un) * un
        while i < lim:
            if cond._data[i + 0] != 0: out._data[i + 0] = x._data[i + 0] else: out._data[i + 0] = y._data[i + 0]
            if cond._data[i + 1] != 0: out._data[i + 1] = x._data[i + 1] else: out._data[i + 1] = y._data[i + 1]
            if cond._data[i + 2] != 0: out._data[i + 2] = x._data[i + 2] else: out._data[i + 2] = y._data[i + 2]
            if cond._data[i + 3] != 0: out._data[i + 3] = x._data[i + 3] else: out._data[i + 3] = y._data[i + 3]
            if cond._data[i + 4] != 0: out._data[i + 4] = x._data[i + 4] else: out._data[i + 4] = y._data[i + 4]
            if cond._data[i + 5] != 0: out._data[i + 5] = x._data[i + 5] else: out._data[i + 5] = y._data[i + 5]
            if cond._data[i + 6] != 0: out._data[i + 6] = x._data[i + 6] else: out._data[i + 6] = y._data[i + 6]
            if cond._data[i + 7] != 0: out._data[i + 7] = x._data[i + 7] else: out._data[i + 7] = y._data[i + 7]
            i = i + un
        while i < n:
            if cond._data[i] != 0: out._data[i] = x._data[i] else: out._data[i] = y._data[i]
            i = i + 1
        return out.copy()

    # generic strided path (broadcast-aware)
    var r = len(out_shape)
    var xr = len(x._shape)
    var yr = len(y._shape)
    var cr = len(cond._shape)

    var sx = List[Int](); var sy = List[Int](); var sc = List[Int](); var so = List[Int]()
    sx.reserve(r); sy.reserve(r); sc.reserve(r); so.reserve(r)

    var i2 = 0
    while i2 < r:
        var ox = 0; var oy = 0; var oc = 0
        var jx = i2 + xr - r
        var jy = i2 + yr - r
        var jc = i2 + cr - r
        if jx >= 0:
            ox =x._strides[jx]
            if x._shape[jx] == 1: ox =0
        if jy >= 0:
            oy =y._strides[jy]
            if y._shape[jy] == 1: oy =0
        if jc >= 0:
            oc =cond._strides[jc]
            if cond._shape[jc] == 1: oc =0
        sx.append(ox); sy.append(oy); sc.append(oc); so.append(out._strides[i2])
        i2 = i2 + 1

    var offx = 0; var offy = 0; var offc = 0; var offo = 0
    var idx = List[Int](); idx.reserve(r)
    var j = 0
    while j < r: idx.append(0); j = j + 1

    var k = 0
    while k < n:
        if cond._data[offc] != 0:
            out._data[offo] = x._data[offx]
        else:
            out._data[offo] = y._data[offy]

        var d = r - 1
        while d >= 0:
            idx[d] = idx[d] + 1
            offx = offx + sx[d]; offy = offy + sy[d]; offc = offc + sc[d]; offo = offo + so[d]
            if idx[d] < out_shape[d]: break
            var span = out_shape[d]
            idx[d] = 0
            offx = offx - sx[d] * span
            offy = offy - sy[d] * span
            offc = offc - sc[d] * span
            offo = offo - so[d] * span
            d = d - 1
        k = k + 1

    return out.copy()


# ---------- where() GPU (fallback-safe) ----------
fn where_f32_gpu(
    cond: tensor.Tensor[Int],
    x: tensor.Tensor[Float32],
    y: tensor.Tensor[Float32]
) -> tensor.Tensor[Float32]:
    if not (x._shape == y._shape and y._shape == cond._shape):
        return where_f32_cpu(cond, x, y)

    if not (_is_row_major_contig(x._shape, x._strides) and
            _is_row_major_contig(y._shape, y._strides) and
            _is_row_major_contig(cond._shape, cond._strides)):
        return where_f32_cpu(cond, x, y)

    var n = _numel(x._shape)
    var out = tensor.zeros(x._shape.copy())
    var obuf = out._data.copy()

    var total = n
    var block =256
    if n < 256: block =n

    _launch_1d_where_same_f32(
        total, block, _kernel_where_same_f32,
        cond._data, x._data, y._data, obuf, n
    )

    out._data = obuf.copy()
    return out.copy()


# ---------- AUTO & wrapper ----------
fn where_f32_auto(
    cond: tensor.Tensor[Int],
    x: tensor.Tensor[Float32],
    y: tensor.Tensor[Float32]
) -> tensor.Tensor[Float32]:
    if gpu_available():
        return where_f32_gpu(cond, x, y)
    return where_f32_cpu(cond, x, y)




@always_inline
fn where_f64(cond: Tensor[Int], x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Float64]:
    # Routed to generic where with Float64 converter
    return where[Float64](cond, x, y, f64_to)

@always_inline
fn where_f32(cond: Tensor[Int], x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Float32]:
    # Routed to generic where with Float32 converter
    return where_f32_auto(cond, x, y)

@always_inline
fn where_int(cond: Tensor[Int], x: Tensor[Int], y: Tensor[Int]) -> Tensor[Int]:
    # Routed to generic where with Int converter
    return where[Int](cond, x, y, f64_to_int)


@always_inline
fn where(cond: Tensor[Int], x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Float64]:
    # Dispatch to Float64 implementation
    return where_f64(cond, x, y)

@always_inline
fn where(cond: Tensor[Int], x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Float32]:
    # Dispatch to Float32 implementation
    return where_f32(cond, x, y)

@always_inline
fn where(cond: Tensor[Int], x: Tensor[Int], y: Tensor[Int]) -> Tensor[Int]:
    # Dispatch to Int implementation
    return where_int(cond, x, y)
# -----------------------------------------------------------------------------
# Take / Take-along / Scatter / gather_nd / scatter_nd
# -----------------------------------------------------------------------------
# take1d: gather along axis 0 for a rank>=1 tensor, returning 1D tensor
# NOTE: use positional ctor (shape, fill). Respect x._off and out._off.
# ---------------------------------------------------------------------
fn take1d[T: ImplicitlyCopyable & Copyable & Movable](t: Tensor[T], indices: List[Int]) -> Tensor[T]:
    var shp = t._shape
    var n0 = 0
    if len(shp) > 0:
        n0 = shp[0]

    # shape = [len(indices)]
    var out_shape = List[Int]()
    out_shape.append(len(indices))

    var out = Tensor[T](out_shape.copy(), zero_scalar_of[T](cast_from_f64[T]))
    if n0 <= 0:
        return out.copy()

    var s0 = t._strides[0]
    var i = 0
    var lim = (len(indices) // 8) * 8
    while i < lim:
        var i0 = clamp(indices[i + 0], 0, n0 - 1)
        var i1 = clamp(indices[i + 1], 0, n0 - 1)
        var i2 = clamp(indices[i + 2], 0, n0 - 1)
        var i3 = clamp(indices[i + 3], 0, n0 - 1)
        var i4 = clamp(indices[i + 4], 0, n0 - 1)
        var i5 = clamp(indices[i + 5], 0, n0 - 1)
        var i6 = clamp(indices[i + 6], 0, n0 - 1)
        var i7 = clamp(indices[i + 7], 0, n0 - 1)

        out._data[out._off + i + 0] = t._data[t._off + i0 * s0]
        out._data[out._off + i + 1] = t._data[t._off + i1 * s0]
        out._data[out._off + i + 2] = t._data[t._off + i2 * s0]
        out._data[out._off + i + 3] = t._data[t._off + i3 * s0]
        out._data[out._off + i + 4] = t._data[t._off + i4 * s0]
        out._data[out._off + i + 5] = t._data[t._off + i5 * s0]
        out._data[out._off + i + 6] = t._data[t._off + i6 * s0]
        out._data[out._off + i + 7] = t._data[t._off + i7 * s0]
        i += 8
    while i < len(indices):
        var idx = clamp(indices[i], 0, n0 - 1)
        out._data[out._off + i] = t._data[t._off + idx * s0]
        i += 1
    return out.copy()

# ---------------------------------------------------------------------
# take: gather with Tensor[Int] indices along an arbitrary axis (copy)
# Uses your existing helpers: _row_major_multipliers / _coords_from_linear / _offset_from_coords
# Renamed normalize_axis -> normalize_axis, and fixed _off usage and constructors.
# ---------------------------------------------------------------------
@always_inline
fn take[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], indices: Tensor[Int], axis: Int = 0
) -> Tensor[T]:
    var r = len(x._shape)
    var ax = normalize_axis(axis, r)

    var idx_len = len(indices._data)
    var out_shape = x._shape.copy()
    out_shape[ax] = idx_len

    # num elements
    var n = numel(out_shape)
    if n == 0:

        var empty = List[T]()
        return Tensor[T](empty, out_shape.copy()).copy()

    # flat buffer with dummy init copied from x
    var flat = List[T]()
    flat.reserve(n)
    var init_val = x._data[x._offset]
    var ii = 0
    while ii < n:
        flat.append(init_val)
        ii += 1

    # out tensor from (data, shape)
    var out = Tensor[T](flat.copy(), out_shape.copy())

    var rm_out = _row_major_multipliers(out_shape)

    var pre = ax
    var post = r - ax - 1

    var lin = 0
    while lin < n:
        var ocoord = _coords_from_linear(out_shape, rm_out, lin)

        var k = indices._data[indices._offset + ocoord[ax]]
        if k < 0: k = k + x._shape[ax]
        if k < 0: k = 0
        var mx = x._shape[ax] - 1
        if k > mx: k = mx

        var xcoord = List[Int]()
        xcoord.reserve(r)

        var d = 0
        while d < pre:
            xcoord.append(ocoord[d])
            d += 1
        xcoord.append(k)
        var e = 0
        while e < post:
            xcoord.append(ocoord[pre + 1 + e])
            e += 1

        var xoff = _offset_from_coords(x._strides, x._offset, xcoord)
        out._data[out._offset + lin] = x._data[xoff]
        lin += 1
    return out.copy()


@always_inline
fn take[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], indices: List[Int], axis: Int = 0
) -> Tensor[T]:
    # shape of the index tensor: [len(indices)]
    var ish = List[Int](); ish.append(len(indices))

    # flat data for indices tensor
    var id_flat = List[Int]()
    id_flat.reserve(len(indices))
    var i = 0
    while i < len(indices):
        id_flat.append(indices[i])
        i += 1

    # 1D contiguous strides and zero offset → disambiguates ctor
    var strides = List[Int](); strides.append(1)
    var idx = Tensor[Int](id_flat.copy(), ish.copy(), strides.copy(), 0)

    return take(x, idx, axis)


# ---------------------------------------------------------------------
# take_along_axis: shape of indices matches output; gather per element along axis
# Fixed: constructors (no keyword), offsets (_off), stride math intact.
# ---------------------------------------------------------------------
fn take_along_axis[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], indices: Tensor[Int], axis: Int) -> Tensor[T]:
    var ax = normalize_axis(axis, len(x._shape))
    var sh = x._shape.copy()
    var sh_idx = indices._shape

    var out_shape = sh.copy()
    out_shape[ax] = sh_idx[ax]

    var out = Tensor[T](out_shape.copy(), zero_scalar_of[T](cast_from_f64[T]))

    var total = numel(out_shape)
    var rm = compute_row_major_strides(out_shape)

    var i = 0
    while i < total:
        # decode linear i to coords idxs[]
        var idxs = List[Int]()
        idxs.reserve(len(out_shape))
        var d = 0
        while d < len(out_shape):
            var stride = rm[d]
            var s = 0
            if stride != 0 and out_shape[d] != 0:
                s = (i // stride) % out_shape[d]
            idxs.append(s)
            d += 1

        # offset inside indices
        var off_idx = indices._off
        var dd = 0
        while dd < len(sh):
            off_idx = off_idx + idxs[dd] * indices._strides[dd]
            dd += 1

        # clamp selected position along axis
        var gather_i = clamp(indices._data[off_idx], 0, sh[ax] - 1)

        # build src index (same as idxs but axis replaced by gather_i)
        var src_idxs = idxs
        src_idxs[ax] = gather_i

        # compute linear offsets for x and out
        var offx = x._off
        var offo = out._off
        var d2 = 0
        while d2 < len(sh):
            offx = offx + src_idxs[d2] * x._strides[d2]
            offo = offo + idxs[d2] * out._strides[d2]
            d2 += 1

        out._data[offo] = x._data[offx]
        i += 1
    return out.copy()



@always_inline
fn scatter[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], axis: Int, index: Tensor[Int], updates: Tensor[T]
) -> Tensor[T]:
    var r = len(x._shape)
    var ax = normalize_axis(axis, r)

    var out = x.copy()
    var n = numel(index._shape)
    if n == 0:
        return out.copy()

    var rm_idx = _row_major_multipliers(index._shape)

    var pre = ax
    var post = r - ax - 1

    var lin = 0
    while lin < n:
        var icoord = _coords_from_linear(index._shape, rm_idx, lin)

        # dest coords in out
        var dcoord = List[Int]()
        dcoord.reserve(r)

        var d = 0
        while d < pre:
            dcoord.append(icoord[d])
            d += 1

        var idx_off = _offset_from_linear(index._shape, index._strides, index._offset, rm_idx, lin)
        var k = index._data[idx_off]
        if k < 0:
            k = k + out._shape[ax]
        if k < 0:
            k = 0
        var mx = out._shape[ax] - 1
        if k > mx:
            k = mx
        dcoord.append(k)

        var e = 0
        while e < post:
            dcoord.append(icoord[pre + 1 + e])
            e += 1

        var doff = _offset_from_coords(out._strides, out._offset, dcoord)

        var uoff = 0
        if len(updates._shape) == 0 or numel(updates._shape) == 1 and numel(updates._shape) == 1 and updates._shape[0] == 1:
            uoff = updates._offset
        else:
            uoff = _offset_from_linear(updates._shape, updates._strides, updates._offset, rm_idx, lin)

        out._data[doff] = updates._data[uoff]
        lin += 1
    return out.copy()


fn scatter_nd[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], indices: Tensor[Int], updates: Tensor[T]):
    var sh = x._shape.copy()
    var r = len(sh)
    if len(indices._shape) == 0:
        return
    var K = indices._shape[len(indices._shape) - 1]
    if K != r:
        return
    var up_shape = List[Int]()
    var i = 0
    while i < len(indices._shape) - 1:
        up_shape.append(indices._shape[i])
        i += 1
    var total = numel(up_shape)
    var rm = compute_row_major_strides(up_shape)
    var p = 0
    while p < total:
        var idxs = List[Int]()
        idxs.reserve(len(up_shape))
        var d = 0
        while d < len(up_shape):
            var stride = rm[d]
            var s = 0
            if stride != 0 and up_shape[d] != 0:
                s = (p // stride) % up_shape[d]
            idxs.append(s)
            d += 1
        var off_ind = 0
        var dd = 0
        while dd < len(up_shape):
            off_ind = off_ind + idxs[dd] * indices._strides[dd]
            dd += 1
        var base = off_ind
        var j = 0
        var coord = List[Int]()
        coord.reserve(r)
        while j < r:
            var offj = base + j * indices._strides[len(indices._shape) - 1]
            var v = clamp(indices._data[offj], 0, sh[j] - 1)
            coord.append(v)
            j += 1
        var offx = 0
        var d2 = 0
        while d2 < r:
            offx = offx + coord[d2] * x._strides[d2]
            d2 += 1
        var offu = 0
        var d3 = 0
        while d3 < len(up_shape):
            offu = offu + idxs[d3] * updates._strides[d3]
            d3 += 1
        x._data[offx] = updates._data[offu]
        p += 1


# -----------------------------------------------------------------------------
# Utilities (compact, English-only comments; no asserts)
# -----------------------------------------------------------------------------
@always_inline
fn _same_shape(a: List[Int], b: List[Int]) -> Bool:
    var ra = len(a)
    if ra != len(b): return False
    var i = 0
    while i < ra:
        if a[i] != b[i]: return False
        i += 1
    return True

@always_inline
fn _row_major_multipliers(shape: List[Int]) -> List[Int]:
    var r = len(shape)
    var rm = List[Int]()
    rm.reserve(r)
    var i = 0
    while i < r:
        rm.append(0)
        i += 1
    var acc = 1
    var k = r - 1
    while k >= 0:
        rm[k] = acc
        acc = acc * shape[k]
        k -= 1
    return rm.copy()

@always_inline
fn _offset_from_linear(
    shape: List[Int], strides: List[Int], off: Int, rm: List[Int], lin: Int
) -> Int:
    var r = len(shape)
    var o = off
    var d = 0
    while d < r:
        var s = 0
        if shape[d] != 0 and rm[d] != 0:
            s = (lin // rm[d]) % shape[d]
        o = o + s * strides[d]
        d += 1
    return o


@always_inline
fn _offset_from_coords(coords: List[Int], strides: List[Int], base: Int) -> Int:
    var off = base
    var d = 0
    var r = len(coords)
    while d < r:
        off = off + coords[d] * strides[d]
        d += 1
    return off


@always_inline
fn _offset_from_coords(strides: List[Int], off: Int, coords: List[Int]) -> Int:
    var o = off
    var d = 0
    while d < len(coords):
        o = o + coords[d] * strides[d]
        d += 1
    return o

# -----------------------------------------------------------------------------
# argwhere (Tensor) and argwhere_any (AnyTensor)
# -----------------------------------------------------------------------------

fn argwhere[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[Int32]:
    var r = len(x._shape)
    var n = len(x._data)
    var rows = List[Int32]()
    rows.reserve(max_i(1, n) * r)
    var rm = compute_row_major_strides(x._shape)
    var i = 0
    while i < n:
        var off = 0
        var d = 0
        while d < r:
            var stride = rm[d]
            var s = 0
            if stride != 0 and x._shape[d] != 0:
                s = (i // stride) % x._shape[d]
            off = off + s * x._strides[d]
            d += 1
        if x._data[off] != zero_scalar_of[T](cast_from_f64[T]):
            var dd = 0
            while dd < r:
                var stride2 = rm[dd]
                var s2 = 0
                if stride2 != 0 and x._shape[dd] != 0:
                    s2 = (i // stride2) % x._shape[dd]
                rows.append(Int32(s2))
                dd += 1
        i += 1
    var out_shape = [len(rows) // r, r]
    return Tensor[Int32](rows, out_shape)

fn argwhere_any(x: AnyTensor) -> Tensor[Int32]:
    var rank = len(x._shape)
    var n = x._buf.length()
    var rows = List[Int32]()
    rows.reserve(max_i(1, n) * rank)
    var rm = compute_row_major_strides(x._shape)
    var i = 0
    while i < n:
        var off = 0
        var d = 0
        while d < rank:
            var stride = rm[d]
            var s = 0
            if stride != 0 and x._shape[d] != 0:
                s = (i // stride) % x._shape[d]
            off = off + s * x._strides[d]
            d += 1
        if x._buf.get_f64(off) != 0.0:
            var dd = 0
            while dd < rank:
                var stride2 = rm[dd]
                var s2 = 0
                if stride2 != 0 and x._shape[dd] != 0:
                    s2 = (i // stride2) % x._shape[dd]
                rows.append(Int32(s2))
                dd += 1
        i += 1
    var out_shape = [len(rows) // rank, rank]
    return Tensor[Int32](rows, out_shape)

# -----------------------------------------------------------------------------
# topk (Tensor[T]) and topk_any (AnyTensor)
# -----------------------------------------------------------------------------

@always_inline
fn _topk_whole_Int(x: Tensor[Int], k: Int, largest: Bool)
    -> (Tensor[Int], Tensor[Int32]):
    var n = len(x._data)
    var idx = List[Int](); idx.reserve(n)
    var a = x._data.copy()
    var i = 0
    while i < n: idx.append(i); i += 1

    var j = 0
    while j < n:
        var best = j
        var t = j + 1
        while t < n:
            var cond = a[t] > a[best] if largest else a[t] < a[best]
            if cond: best = t
            t += 1
        var tmp = a[j]; a[j] = a[best]; a[best] = tmp
        var ti = idx[j]; idx[j] = idx[best]; idx[best] = ti
        j += 1
    var k_eff = min_i(k, n)

    var kv = List[Int](); var ki = List[Int32]()
    kv.reserve(k_eff); ki.reserve(k_eff)
    i = 0
    while i < k_eff:
        kv.append(a[i]); ki.append(Int32(idx[i])); i += 1

    var shp = List[Int](); shp.append(k_eff)
    var st = compute_row_major_strides(shp)
    return (Tensor[Int](kv, shp, st, 0), Tensor[Int32](ki, shp, st, 0))


@always_inline
fn _topk_axis_Int(x: Tensor[Int], k: Int, largest: Bool, ax: Int)
    -> (Tensor[Int], Tensor[Int32]):
    var rank = len(x._shape)
    var out_shape = List[Int]()
    var d = 0
    while d < rank:
        if d != ax: out_shape.append(x._shape[d])
        d += 1
    if len(out_shape) == 0: out_shape.append(1)

    var L = x._shape[ax]
    var k_eff = min_i(k, L)
    var outer_n = numel(out_shape)

    var values = List[Int]()
    var indices = List[Int32]()
    values.reserve(outer_n * k_eff)
    indices.reserve(outer_n * k_eff)

    var rm = compute_row_major_strides(out_shape)
    var base_idx = List[Int](); base_idx.reserve(len(out_shape))
    var full = List[Int](); full.reserve(rank)

    var oi = 0
    while oi < outer_n:
        base_idx.clear()
        var dd = 0
        while dd < len(out_shape):
            var stride = rm[dd]
            var s = 0
            if stride != 0 and out_shape[dd] != 0:
                s = (oi // stride) % out_shape[dd]
            base_idx.append(s)
            dd += 1

        full.clear()
        var p = 0; var j2 = 0
        while j2 < rank:
            if j2 == ax: full.append(0)
            else: full.append(base_idx[p]); p += 1
            j2 += 1

        var line = List[Int](); line.reserve(L)
        var idline = List[Int](); idline.reserve(L)

        var t2 = 0
        while t2 < L:
            full[ax] = t2
            var li = 0
            var q = 0
            while q < rank:
                li = li + full[q] * x._strides[q]
                q += 1
            line.append(x._data[li])
            idline.append(t2)
            t2 += 1

        var i2 = 0
        while i2 < L:
            var best = i2
            var u = i2 + 1
            while u < L:
                var cond2 = line[u] > line[best] if largest else line[u] < line[best]
                if cond2: best = u
                u += 1
            var tmp2 = line[i2]; line[i2] = line[best]; line[best] = tmp2
            var ti2 = idline[i2]; idline[i2] = idline[best]; idline[best] = ti2
            i2 += 1
        var r2 = 0
        while r2 < k_eff:
            values.append(line[r2])
            indices.append(Int32(idline[r2]))
            r2 += 1

        oi += 1
    var final_shape = out_shape.copy()
    final_shape.append(k_eff)
    var st = compute_row_major_strides(final_shape)
    return (Tensor[Int](values, final_shape, st, 0),
            Tensor[Int32](indices, final_shape, st, 0))


@always_inline
fn topk(x: Tensor[Int], k: Int, largest: Bool = True, axis: Optional[Int] = None)
    -> (Tensor[Int], Tensor[Int32]):
    if axis is None:
        return _topk_whole_Int(x, k, largest)
    var ax = normalize_axis(axis.value(), len(x._shape))
    return _topk_axis_Int(x, k, largest, ax)

@always_inline
fn topk(x: Tensor[Float32], k: Int, largest: Bool = True, axis: Optional[Int] = None)
    -> (Tensor[Float32], Tensor[Int32]):
    if axis is None:
        var n = len(x._data)
        var idx = List[Int](); idx.reserve(n)
        var a = x._data.copy()
        var i = 0
        while i < n: idx.append(i); i += 1
        var j = 0
        while j < n:
            var best = j
            var t = j + 1
            while t < n:
                var cond = a[t] > a[best] if largest else a[t] < a[best]
                if cond: best = t
                t += 1
            var tmp = a[j]; a[j] = a[best]; a[best] = tmp
            var ti = idx[j]; idx[j] = idx[best]; idx[best] = ti
            j += 1
        var k_eff = min_i(k, n)
        var kv = List[Float32](); var ki = List[Int32]()
        kv.reserve(k_eff); ki.reserve(k_eff)
        i = 0
        while i < k_eff:
            kv.append(a[i]); ki.append(Int32(idx[i])); i += 1
        var shp = List[Int](); shp.append(k_eff)
        var st = compute_row_major_strides(shp)
        return (Tensor[Float32](kv, shp, st, 0), Tensor[Int32](ki, shp, st, 0))

    var ax = normalize_axis(axis.value(), len(x._shape))
    var rank = len(x._shape)

    var out_shape = List[Int](); var d = 0
    while d < rank:
        if d != ax: out_shape.append(x._shape[d])
        d += 1
    if len(out_shape) == 0: out_shape.append(1)

    var L = x._shape[ax]
    var k_eff = min_i(k, L)
    var outer_n = numel(out_shape)

    var values = List[Float32](); var indices = List[Int32]()
    values.reserve(outer_n * k_eff); indices.reserve(outer_n * k_eff)

    var rm = compute_row_major_strides(out_shape)
    var base_idx = List[Int](); base_idx.reserve(len(out_shape))
    var full = List[Int](); full.reserve(rank)

    var oi = 0
    while oi < outer_n:
        base_idx.clear()
        var dd = 0
        while dd < len(out_shape):
            var stride = rm[dd]
            var s = 0
            if stride != 0 and out_shape[dd] != 0:
                s = (oi // stride) % out_shape[dd]
            base_idx.append(s)
            dd += 1

        full.clear()
        var p = 0; var j2 = 0
        while j2 < rank:
            if j2 == ax: full.append(0)
            else: full.append(base_idx[p]); p += 1
            j2 += 1

        var line = List[Float32](); line.reserve(L)
        var idline = List[Int](); idline.reserve(L)

        var t2 = 0
        while t2 < L:
            full[ax] = t2
            var li = 0
            var q = 0
            while q < rank:
                li = li + full[q] * x._strides[q]
                q += 1
            line.append(x._data[li]); idline.append(t2)
            t2 += 1

        var i2 = 0
        while i2 < L:
            var best = i2
            var u = i2 + 1
            while u < L:
                var cond2 = line[u] > line[best] if largest else line[u] < line[best]
                if cond2: best = u
                u += 1
            var tmp2 = line[i2]; line[i2] = line[best]; line[best] = tmp2
            var ti2 = idline[i2]; idline[i2] = idline[best]; idline[best] = ti2
            i2 += 1

        var r2 = 0
        while r2 < k_eff:
            values.append(line[r2]); indices.append(Int32(idline[r2])); r2 += 1
        oi += 1

    var final_shape = out_shape.copy(); final_shape.append(k_eff)
    var st = compute_row_major_strides(final_shape)
    return (Tensor[Float32](values, final_shape, st, 0),
            Tensor[Int32](indices, final_shape, st, 0))


@always_inline
fn topk(x: Tensor[Float64], k: Int, largest: Bool = True, axis: Optional[Int] = None)
    -> (Tensor[Float64], Tensor[Int32]):
    if axis is None:
        var n = len(x._data)
        var idx = List[Int](); idx.reserve(n)
        var a = x._data.copy()
        var i = 0
        while i < n: idx.append(i); i += 1
        var j = 0
        while j < n:
            var best = j
            var t = j + 1
            while t < n:
                var cond = a[t] > a[best] if largest else a[t] < a[best]
                if cond: best = t
                t += 1
            var tmp = a[j]; a[j] = a[best]; a[best] = tmp
            var ti = idx[j]; idx[j] = idx[best]; idx[best] = ti
            j += 1
        var k_eff = min_i(k, n)
        var kv = List[Float64](); var ki = List[Int32]()
        kv.reserve(k_eff); ki.reserve(k_eff)
        i = 0
        while i < k_eff:
            kv.append(a[i]); ki.append(Int32(idx[i])); i += 1
        var shp = List[Int](); shp.append(k_eff)
        var st = compute_row_major_strides(shp)
        return (Tensor[Float64](kv, shp, st, 0), Tensor[Int32](ki, shp, st, 0))

    var ax = normalize_axis(axis.value(), len(x._shape))
    var rank = len(x._shape)

    var out_shape = List[Int](); var d = 0
    while d < rank:
        if d != ax: out_shape.append(x._shape[d])
        d += 1
    if len(out_shape) == 0: out_shape.append(1)

    var L = x._shape[ax]
    var k_eff = min_i(k, L)
    var outer_n = numel(out_shape)

    var values = List[Float64](); var indices = List[Int32]()
    values.reserve(outer_n * k_eff); indices.reserve(outer_n * k_eff)

    var rm = compute_row_major_strides(out_shape)
    var base_idx = List[Int](); base_idx.reserve(len(out_shape))
    var full = List[Int](); full.reserve(rank)

    var oi = 0
    while oi < outer_n:
        base_idx.clear()
        var dd = 0
        while dd < len(out_shape):
            var stride = rm[dd]
            var s = 0
            if stride != 0 and out_shape[dd] != 0:
                s = (oi // stride) % out_shape[dd]
            base_idx.append(s)
            dd += 1

        full.clear()
        var p = 0; var j2 = 0
        while j2 < rank:
            if j2 == ax: full.append(0)
            else: full.append(base_idx[p]); p += 1
            j2 += 1

        var line = List[Float64](); line.reserve(L)
        var idline = List[Int](); idline.reserve(L)

        var t2 = 0
        while t2 < L:
            full[ax] = t2
            var li = 0
            var q = 0
            while q < rank:
                li = li + full[q] * x._strides[q]
                q += 1
            line.append(x._data[li]); idline.append(t2)
            t2 += 1

        var i2 = 0
        while i2 < L:
            var best = i2
            var u = i2 + 1
            while u < L:
                var cond2 = line[u] > line[best] if largest else line[u] < line[best]
                if cond2: best = u
                u += 1
            var tmp2 = line[i2]; line[i2] = line[best]; line[best] = tmp2
            var ti2 = idline[i2]; idline[i2] = idline[best]; idline[best] = ti2
            i2 += 1

        var r2 = 0
        while r2 < k_eff:
            values.append(line[r2]); indices.append(Int32(idline[r2])); r2 += 1
        oi += 1

    var final_shape = out_shape.copy(); final_shape.append(k_eff)
    var st = compute_row_major_strides(final_shape)
    return (Tensor[Float64](values, final_shape, st, 0),
            Tensor[Int32](indices, final_shape, st, 0))


# -----------------------------------------------------------------------------
# im2col / col2im (2D core)
# -----------------------------------------------------------------------------

@always_inline
fn tensor_from_flat[T: Copyable & Movable](data: List[T], shape: List[Int]) -> Tensor[T]:
    return Tensor[T](data, shape)

fn im2col_core[T: Copyable & Movable](x: Tensor[T], kH: Int, kW: Int, stride: Int = 1, padding: Int = 0) -> Tensor[T]:
    var shp = x.shape()
    var H = shp[0] if len(shp) > 0 else 0
    var W = shp[1] if len(shp) > 1 else 0
    var s = stride if stride > 0 else 1
    var p = padding if padding >= 0 else 0
    var Hout = 0
    var Wout = 0
    if H + 2 * p >= kH and W + 2 * p >= kW:
        Hout = ((H + 2 * p - kH) // s) + 1
        Wout = ((W + 2 * p - kW) // s) + 1
    var cols = kH * kW
    var out = List[T]()
    out.reserve(cols * Hout * Wout)
    var rm = compute_row_major_strides([H, W])
    var co = 0
    while co < Hout * Wout:
        var oy = co // Wout
        var ox = co % Wout
        var base_y = oy * s - p
        var base_x = ox * s - p
        var ky = 0
        while ky < kH:
            var y = base_y + ky
            var kx = 0
            while kx < kW:
                var x2 = base_x + kx
                var v: T = zero_scalar_of[T](cast_from_f64[T])
                if y >= 0 and y < H and x2 >= 0 and x2 < W:
                    var pin = y * rm[0] + x2 * rm[1]
                    v = x._data[pin]
                out.append(v)
                kx += 1
            ky += 1
        co += 1
    return tensor_from_flat[T](out, [cols, Hout * Wout])

fn col2im_core[T: Copyable & Movable](cols: Tensor[T], out_shape: List[Int], kH: Int, kW: Int, stride: Int = 1, padding: Int = 0) -> Tensor[T]:
    var H = out_shape[0] if len(out_shape) > 0 else 0
    var W = out_shape[1] if len(out_shape) > 1 else 0
    var s = stride if stride > 0 else 1
    var p = padding if padding >= 0 else 0
    var Hout = 0
    var Wout = 0
    if H + 2 * p >= kH and W + 2 * p >= kW:
        Hout = ((H + 2 * p - kH) // s) + 1
        Wout = ((W + 2 * p - kW) // s) + 1
    var out = List[T]()
    out.reserve(H * W)
    var i = 0
    while i < H * W:
        out.append(zero_scalar_of[T](cast_from_f64[T]))
        i += 1
    var rm = compute_row_major_strides([H, W])
    var co = 0
    while co < Hout * Wout:
        var oy = co // Wout
        var ox = co % Wout
        var base_y = oy * s - p
        var base_x = ox * s - p
        var ky = 0
        while ky < kH:
            var y = base_y + ky
            var kx = 0
            while kx < kW:
                var x2 = base_x + kx
                var v: T = cols._data[ky * kW + kx + co * (kH * kW)]
                if y >= 0 and y < H and x2 >= 0 and x2 < W:
                    var pin = y * rm[0] + x2 * rm[1]
                    out[pin] = out[pin] + v
                kx += 1
            ky += 1
        co += 1
    return tensor_from_flat[T](out, out_shape)

# -----------------------------------------------------------------------------
# GEMM (sgemm/dgemm) — blocked + unrolled
# -----------------------------------------------------------------------------



@always_inline
fn is_2d_f32(x: Tensor[Float32]) -> Bool:
    return len(x._shape) == 2

@always_inline
fn is_2d_f64(x: Tensor[Float64]) -> Bool:
    return len(x._shape) == 2

@always_inline
fn rows_f32(x: Tensor[Float32]) -> Int:
    return x._shape[0]

@always_inline
fn cols_f32(x: Tensor[Float32]) -> Int:
    return x._shape[1]

@always_inline
fn rows_f64(x: Tensor[Float64]) -> Int:
    return x._shape[0]

@always_inline
fn cols_f64(x: Tensor[Float64]) -> Int:
    return x._shape[1]

fn check_sgemm_contract(a: Tensor[Float32], b: Tensor[Float32], c: Tensor[Float32]) -> Bool:
    if not is_2d_f32(a) or not is_2d_f32(b) or not is_2d_f32(c):
        return False
    var m = rows_f32(a)
    var k = cols_f32(a)
    if rows_f32(b) != k:
        return False
    var n = cols_f32(b)
    if rows_f32(c) != m or cols_f32(c) != n:
        return False
    return True

fn check_dgemm_contract(a: Tensor[Float64], b: Tensor[Float64], c: Tensor[Float64]) -> Bool:
    if not is_2d_f64(a) or not is_2d_f64(b) or not is_2d_f64(c):
        return False
    var m = rows_f64(a)
    var k = cols_f64(a)
    if rows_f64(b) != k:
        return False
    var n = cols_f64(b)
    if rows_f64(c) != m or cols_f64(c) != n:
        return False
    return True

@always_inline
fn zero_out_f32(mut c: Tensor[Float32]):
    var i = 0
    var n = len(c._data)
    while i < n:
        c._data[i] = 0.0f32
        i += 1

@always_inline
fn zero_out_f64(mut c: Tensor[Float64]):
    var i = 0
    var n = len(c._data)
    while i < n:
        c._data[i] = 0.0
        i += 1

@always_inline
fn scale_inplace_f32(mut c: Tensor[Float32], beta: Float32):
    if beta == 1.0f32:
        return
    if beta == 0.0f32:
        zero_out_f32(c)
        return
    var i = 0
    var n = len(c._data)
    while i < n:
        c._data[i] = c._data[i] * beta
        i += 1

@always_inline
fn scale_inplace_f64(mut c: Tensor[Float64], beta: Float64):
    if beta == 1.0:
        return
    if beta == 0.0:
        zero_out_f64(c)
        return
    var i = 0
    var n = len(c._data)
    while i < n:
        c._data[i] = c._data[i] * beta
        i += 1

fn select_tiles(M: Int, N: Int, K: Int, is_f64: Bool) -> (Int, Int, Int, Int):
    var tune = _TUNE_F64 if is_f64 else _TUNE_F32
    var bm = tune.bm
    var bn = tune.bn
    var bk = tune.bk
    var un = tune.unroll
    if M < 64 or N < 64 or K < 64:
        bm = min_i(bm, 32)
        bn = min_i(bn, 32)
        bk = min_i(bk, 32)
    if M >= 256 and N >= 256 and K >= 256:
        bm = min_i(min_i(bm, 128), M)
        bn = min_i(min_i(bn, 128), N)
    if un != 8 and un != 4:
        un = 8
    return (bm, bn, bk, un)

fn sgemm_contig_kernel_base(
    A: Tensor[Float32], B: Tensor[Float32], mut C: Tensor[Float32],
    alpha: Float32, beta: Float32, baseA: Int, baseB: Int, baseC: Int
) -> Bool:
    var M = rows_f32(A)
    var K = cols_f32(A)
    var N = cols_f32(B)
    scale_inplace_f32(C, beta)
    var (BM, BN, BK, UN) = select_tiles(M, N, K, False)
    var i0 = 0
    while i0 < M:
        var iMax = min_i(i0 + BM, M)
        var j0 = 0
        while j0 < N:
            var jMax = min_i(j0 + BN, N)
            var k0 = 0
            while k0 < K:
                var kMax = min_i(k0 + BK, K)
                var i = i0
                while i < iMax:
                    var a_base = baseA + i * K
                    var c_base = baseC + i * N
                    var j = j0
                    while j < jMax:
                        var acc: Float32 = 0.0f32
                        var k = k0
                        if UN == 8:
                            var kup = k0 + ((kMax - k0) // 8) * 8
                            while k < kup:
                                acc = acc
                                    + A._data[a_base + k    ] * B._data[baseB + (k    ) * N + j]
                                    + A._data[a_base + k + 1] * B._data[baseB + (k + 1) * N + j]
                                    + A._data[a_base + k + 2] * B._data[baseB + (k + 2) * N + j]
                                    + A._data[a_base + k + 3] * B._data[baseB + (k + 3) * N + j]
                                    + A._data[a_base + k + 4] * B._data[baseB + (k + 4) * N + j]
                                    + A._data[a_base + k + 5] * B._data[baseB + (k + 5) * N + j]
                                    + A._data[a_base + k + 6] * B._data[baseB + (k + 6) * N + j]
                                    + A._data[a_base + k + 7] * B._data[baseB + (k + 7) * N + j]
                                k += 8
                        else:
                            var kup4 = k0 + ((kMax - k0) // 4) * 4
                            while k < kup4:
                                acc = acc
                                    + A._data[a_base + k    ] * B._data[baseB + (k    ) * N + j]
                                    + A._data[a_base + k + 1] * B._data[baseB + (k + 1) * N + j]
                                    + A._data[a_base + k + 2] * B._data[baseB + (k + 2) * N + j]
                                    + A._data[a_base + k + 3] * B._data[baseB + (k + 3) * N + j]
                                k += 4
                        while k < kMax:
                            acc = acc + A._data[a_base + k] * B._data[baseB + k * N + j]
                            k += 1
                        C._data[c_base + j] = C._data[c_base + j] + alpha * acc
                        j += 1
                    i += 1
                k0 += BK
            j0 += BN
        i0 += BM
    return True

fn dgemm_contig_kernel_base(
    A: Tensor[Float64], B: Tensor[Float64], mut C: Tensor[Float64],
    alpha: Float64, beta: Float64, baseA: Int, baseB: Int, baseC: Int
) -> Bool:
    var M = rows_f64(A)
    var K = cols_f64(A)
    var N = cols_f64(B)
    scale_inplace_f64(C, beta)
    var (BM, BN, BK, UN) = select_tiles(M, N, K, True)
    var i0 = 0
    while i0 < M:
        var iMax = min_i(i0 + BM, M)
        var j0 = 0
        while j0 < N:
            var jMax = min_i(j0 + BN, N)
            var k0 = 0
            while k0 < K:
                var kMax = min_i(k0 + BK, K)
                var i = i0
                while i < iMax:
                    var a_base = baseA + i * K
                    var c_base = baseC + i * N
                    var j = j0
                    while j < jMax:
                        var acc: Float64 = 0.0
                        var k = k0
                        if UN == 8:
                            var kup = k0 + ((kMax - k0) // 8) * 8
                            while k < kup:
                                acc = acc
                                    + A._data[a_base + k    ] * B._data[baseB + (k    ) * N + j]
                                    + A._data[a_base + k + 1] * B._data[baseB + (k + 1) * N + j]
                                    + A._data[a_base + k + 2] * B._data[baseB + (k + 2) * N + j]
                                    + A._data[a_base + k + 3] * B._data[baseB + (k + 3) * N + j]
                                    + A._data[a_base + k + 4] * B._data[baseB + (k + 4) * N + j]
                                    + A._data[a_base + k + 5] * B._data[baseB + (k + 5) * N + j]
                                    + A._data[a_base + k + 6] * B._data[baseB + (k + 6) * N + j]
                                    + A._data[a_base + k + 7] * B._data[baseB + (k + 7) * N + j]
                                k += 8
                        else:
                            var kup4 = k0 + ((kMax - k0) // 4) * 4
                            while k < kup4:
                                acc = acc
                                    + A._data[a_base + k    ] * B._data[baseB + (k    ) * N + j]
                                    + A._data[a_base + k + 1] * B._data[baseB + (k + 1) * N + j]
                                    + A._data[a_base + k + 2] * B._data[baseB + (k + 2) * N + j]
                                    + A._data[a_base + k + 3] * B._data[baseB + (k + 3) * N + j]
                                k += 4
                        while k < kMax:
                            acc = acc + A._data[a_base + k] * B._data[baseB + k * N + j]
                            k += 1
                        C._data[c_base + j] = C._data[c_base + j] + alpha * acc
                        j += 1
                    i += 1
                k0 += BK
            j0 += BN
        i0 += BM
    return True

fn sgemm_strided_kernel_base(
    A: Tensor[Float32], B: Tensor[Float32], mut C: Tensor[Float32],
    alpha: Float32, beta: Float32, baseA: Int, baseB: Int, baseC: Int
) -> Bool:
    var M = rows_f32(A)
    var K = cols_f32(A)
    var N = cols_f32(B)
    scale_inplace_f32(C, beta)
    var as0 = A._strides[0]
    var as1 = A._strides[1]
    var bs0 = B._strides[0]
    var bs1 = B._strides[1]
    var cs0 = C._strides[0]
    var cs1 = C._strides[1]
    var (BM, BN, BK, _) = select_tiles(M, N, K, False)
    var i0 = 0
    while i0 < M:
        var iMax = min_i(i0 + BM, M)
        var j0 = 0
        while j0 < N:
            var jMax = min_i(j0 + BN, N)
            var k0 = 0
            while k0 < K:
                var kMax = min_i(k0 + BK, K)
                var i = i0
                while i < iMax:
                    var j = j0
                    while j < jMax:
                        var acc: Float32 = 0.0f32
                        var k = k0
                        while k < kMax:
                            var ai = baseA + i * as0 + k * as1
                            var bi = baseB + k * bs0 + j * bs1
                            acc = acc + A._data[ai] * B._data[bi]
                            k += 1
                        var ci = baseC + i * cs0 + j * cs1
                        C._data[ci] = C._data[ci] + alpha * acc
                        j += 1
                    i += 1
                k0 += BK
            j0 += BN
        i0 += BM
    return True

fn dgemm_strided_kernel_base(
    A: Tensor[Float64], B: Tensor[Float64], mut C: Tensor[Float64],
    alpha: Float64, beta: Float64, baseA: Int, baseB: Int, baseC: Int
) -> Bool:
    var M = rows_f64(A)
    var K = cols_f64(A)
    var N = cols_f64(B)
    scale_inplace_f64(C, beta)
    var as0 = A._strides[0]
    var as1 = A._strides[1]
    var bs0 = B._strides[0]
    var bs1 = B._strides[1]
    var cs0 = C._strides[0]
    var cs1 = C._strides[1]
    var (BM, BN, BK, _) = select_tiles(M, N, K, True)
    var i0 = 0
    while i0 < M:
        var iMax = min_i(i0 + BM, M)
        var j0 = 0
        while j0 < N:
            var jMax = min_i(j0 + BN, N)
            var k0 = 0
            while k0 < K:
                var kMax = min_i(k0 + BK, K)
                var i = i0
                while i < iMax:
                    var j = j0
                    while j < jMax:
                        var acc: Float64 = 0.0
                        var k = k0
                        while k < kMax:
                            var ai = baseA + i * as0 + k * as1
                            var bi = baseB + k * bs0 + j * bs1
                            acc = acc + A._data[ai] * B._data[bi]
                            k += 1
                        var ci = baseC + i * cs0 + j * cs1
                        C._data[ci] = C._data[ci] + alpha * acc
                        j += 1
                    i += 1
                k0 += BK
            j0 += BN
        i0 += BM
    return True

fn sgemm(
    a: Tensor[Float32], b: Tensor[Float32], out_tensor: Tensor[Float32],
    alpha: Float32 = 1.0f32, beta: Float32 = 0.0f32
) -> Bool:
    if not check_sgemm_contract(a, b, out_tensor):
        return False
    var a_contig = a.is_contiguous() and a._strides[1] == 1
    var b_contig = b.is_contiguous() and b._strides[1] == 1
    var c_contig = out_tensor.is_contiguous() and out_tensor._strides[1] == 1
    if a_contig and b_contig and c_contig:
        return sgemm_contig_kernel_base(a, b, out_tensor, alpha, beta, 0, 0, 0)
    return sgemm_strided_kernel_base(a, b, out_tensor, alpha, beta, 0, 0, 0)

fn dgemm(
    a: Tensor[Float64], b: Tensor[Float64], out_tensor: Tensor[Float64],
    alpha: Float64 = 1.0, beta: Float64 = 0.0
) -> Bool:
    if not check_dgemm_contract(a, b, out_tensor):
        return False
    var a_contig = a.is_contiguous() and a._strides[1] == 1
    var b_contig = b.is_contiguous() and b._strides[1] == 1
    var c_contig = out_tensor.is_contiguous() and out_tensor._strides[1] == 1
    if a_contig and b_contig and c_contig:
        return dgemm_contig_kernel_base(a, b, out_tensor, alpha, beta, 0, 0, 0)
    return dgemm_strided_kernel_base(a, b, out_tensor, alpha, beta, 0, 0, 0)

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

@always_inline
fn lin_index(idx: List[Int], strides: List[Int]) -> Int:
    var li = 0
    var r = len(idx)
    var i = 0
    while i < r:
        li = li + idx[i] * strides[i]
        i += 1
    return li

@always_inline
fn ravel_index_with_offset(idx: List[Int], strides: List[Int], offset: Int) -> Int:
    var r = len(idx)
    var acc = offset
    var i = 0
    while i < r:
        acc = acc + idx[i] * strides[i]
        i += 1
    return acc

# Return normalized range as [start, stop, step] list (not a tuple).
@always_inline
fn norm_range(start: Int, stop: Int, step: Int, dim: Int) -> List[Int]:
    var s = step
    if s == 0:
        s = 1
    var a = start
    var b = stop
    if a < 0:
        a = dim + a
    if b <= 0:
        b = dim + b
    if a < 0:
        a = 0
    if b < 0:
        b = 0
    if a > dim:
        a = dim
    if b > dim:
        b = dim
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(s)
    return out.copy()

@always_inline
fn list_select_one[T: ImplicitlyCopyable & Copyable & Movable](xs: List[T], index: Int) -> List[T]:
    var n = len(xs)
    if n == 0:
        return List[T]()
    var idx = index
    if idx < 0:
        idx = 0
    if idx >= n:
        idx = n - 1
    var out = List[T]()
    out.append(xs[idx])
    return out.copy()

@always_inline
fn list_slice_generic[T: ImplicitlyCopyable & Copyable & Movable](xs: List[T], start: Int, stop: Int, step: Int = 1) -> List[T]:
    var n = len(xs)
    var nr = norm_range(start, stop, step, n)
    var a = nr[0]
    var b = nr[1]
    var s = nr[2]
    var out = List[T]()
    if s <= 0 or a >= b:
        return out.copy()
    var m = b - a
    var cnt = m // s
    if m % s != 0:
        cnt += 1
    if cnt < 0:
        cnt = 0
    out.reserve(cnt)
    var i = 0
    while i < cnt:
        out.append(xs[a + i * s])
        i += 1
    return out.copy()

# -----------------------------------------------------------------------------
# Histograms
# -----------------------------------------------------------------------------

fn histogram_core(x: Tensor[Float64], bins: Int, rmin: Optional[Float64], rmax: Optional[Float64]) -> Tensor[Float64]:
    var n = len(x._data)
    var B = bins if bins > 1 else 1
    var lo = 0.0
    var hi = 1.0
    if rmin is None or rmax is None:
        if n > 0:
            lo = x._data[0]
            hi = x._data[0]
            var i0 = 1
            while i0 < n:
                var v = x._data[i0]
                if v < lo:
                    lo = v
                if v > hi:
                    hi = v
                i0 += 1
        else:
            lo = 0.0
            hi = 0.0
    else:
        lo = rmin.value()
        hi = rmax.value()
    var counts = List[Float64]()
    counts.reserve(B)
    var i = 0
    while i < B:
        counts.append(0.0)
        i += 1
    var width = hi - lo
    if width <= 0.0:
        var j = 0
        while j < n:
            counts[B - 1] = counts[B - 1] + 1.0
            j += 1
    else:
        var k = 0
        while k < n:
            var v2 = x._data[k]
            var t = (v2 - lo) / width
            if t < 0.0:
                t = 0.0
            if t >= 1.0:
                t = 0.999999999999
            var b = Int(t * Float64(B))
            if b < 0:
                b = 0
            if b >= B:
                b = B - 1
            counts[b] = counts[b] + 1.0
            k += 1
    return Tensor[Float64](counts, [B])

fn histogram2d_core(
    x: Tensor[Float64], y: Tensor[Float64], bins_h: Int, bins_w: Int,
    r_h_min: Optional[Float64], r_h_max: Optional[Float64],
    r_w_min: Optional[Float64], r_w_max: Optional[Float64]
) -> Tensor[Float64]:
    var n = len(x._data)
    var Bh = bins_h if bins_h > 1 else 1
    var Bw = bins_w if bins_w > 1 else 1
    var loh = 0.0
    var hih = 1.0
    var low = 0.0
    var hiw = 1.0

    if r_h_min is None or r_h_max is None:
        if n > 0:
            loh = x._data[0]
            hih = x._data[0]
            var i0 = 1
            while i0 < n:
                var v = x._data[i0]
                if v < loh:
                    loh = v
                if v > hih:
                    hih = v
                i0 += 1
        else:
            loh = 0.0
            hih = 0.0
    else:
        loh = r_h_min.value()
        hih = r_h_max.value()

    if r_w_min is None or r_w_max is None:
        if n > 0:
            low = y._data[0]
            hiw = y._data[0]
            var j0 = 1
            while j0 < n:
                var u = y._data[j0]
                if u < low:
                    low = u
                if u > hiw:
                    hiw = u
                j0 += 1
        else:
            low = 0.0
            hiw = 0.0
    else:
        low = r_w_min.value()
        hiw = r_w_max.value()

    var out = List[Float64]()
    out.reserve(Bh * Bw)
    var i = 0
    while i < Bh * Bw:
        out.append(0.0)
        i += 1

    var wh = hih - loh
    var ww = hiw - low
    if wh <= 0.0:
        wh = 1.0
    if ww <= 0.0:
        ww = 1.0

    var k = 0
    while k < n:
        var vx = x._data[k]
        var vy = y._data[k]
        var tx = (vx - loh) / wh
        var ty = (vy - low) / ww
        if tx < 0.0:
            tx = 0.0
        if tx >= 1.0:
            tx = 0.999999999999
        if ty < 0.0:
            ty = 0.0
        if ty >= 1.0:
            ty = 0.999999999999
        var bx = Int(tx * Float64(Bh))
        var by = Int(ty * Float64(Bw))
        if bx < 0:
            bx = 0
        if bx >= Bh:
            bx = Bh - 1
        if by < 0:
            by = 0
        if by >= Bw:
            by = Bw - 1
        out[bx * Bw + by] = out[bx * Bw + by] + 1.0
        k += 1

    return Tensor[Float64](out, [Bh, Bw])



fn plane[T: ImplicitlyCopyable & Copyable & Movable](a: Tensor[T], dim: Int, index: Int) -> Tensor[T]:
    var shp = a.shape()
    if len(shp) != 3:
        return Tensor[T]([0, 0], List[T]())

    var B = shp[0]
    var M = shp[1]
    var N = shp[2]

    if dim != 2:
        return Tensor[T]([0, 0], List[T]())
    if not (index >= 0 and index < N):
        return Tensor[T]([0, 0], List[T]())

    var s0 = a._strides[0]
    var s1 = a._strides[1]
    var s2 = a._strides[2]

    var flat = List[T]()
    flat.reserve(B * M)

    var i = 0
    while i < B:
        var j = 0
        while j < M:
            var lin = i * s0 + j * s1 + index * s2
            flat.append(a._data[lin])
            j = j + 1
        i = i + 1

    return Tensor[T]([B, M], flat)







# ============================================================================
# Shape inference & flatten for nested lists (1D..5D), rectangular checks
# ============================================================================

# ---------- 1D ----------
@always_inline
fn infer_shape_1d[T: ImplicitlyCopyable & Copyable & Movable](rows: List[T]) -> List[Int]:
    var shp = List[Int]()
    shp.append(len(rows))
    return shp.copy()

@always_inline
fn _flatten_1d[T: ImplicitlyCopyable & Copyable & Movable](rows: List[T]) -> List[T]:
    return rows.copy()

# ---------- 2D ----------
@always_inline
fn infer_shape_2d[T: ImplicitlyCopyable & Copyable & Movable](rows: List[List[T]]) -> (Int, Int):
    var n0 = len(rows)
    var n1 = 0
    if n0 > 0:
        n1 = len(rows[0])
        var i = 1
        while i < n0:
            var li = len(rows[i])
            if li < n1: n1 = li
            i += 1
    return (n0, n1)

@always_inline
fn flatten_2d[T: ImplicitlyCopyable & Copyable & Movable](rows: List[List[T]], n0: Int, n1: Int) -> List[T]:
    var out = List[T]()
    out.reserve(n0 * n1)
    var i = 0
    while i < n0:
        var row_i = rows[i].copy()
        var j = 0
        while j < n1:
            out.append(row_i[j])
            j += 1
        i += 1
    return out.copy()

# ---------- 3D ----------
@always_inline
fn infer_shape_3d[T: ImplicitlyCopyable & Copyable & Movable](rows: List[List[List[T]]]) -> (Int, Int, Int):
    var d0 = len(rows)
    var d1 = 0
    var d2 = 0
    if d0 > 0:
        d1 = len(rows[0])
        var i = 1
        while i < d0:
            var l1 = len(rows[i])
            if l1 < d1: d1 = l1
            i += 1
        if d1 > 0:
            d2 = len(rows[0][0])
            var ii = 0
            while ii < d0:
                var j = 0
                while j < d1:
                    var l2 = len(rows[ii][j])
                    if l2 < d2: d2 = l2
                    j += 1
                ii += 1
    return (d0, d1, d2)

@always_inline
fn flatten_3d[T: ImplicitlyCopyable & Copyable & Movable](rows: List[List[List[T]]], d0: Int, d1: Int, d2: Int) -> List[T]:
    var out = List[T]()
    out.reserve(d0 * d1 * d2)
    var i = 0
    while i < d0:
        var r_i = rows[i].copy()
        var j = 0
        while j < d1:
            var r_ij = r_i[j].copy()
            var k = 0
            while k < d2:
                out.append(r_ij[k])
                k += 1
            j += 1
        i += 1
    return out.copy()

# ---------- 4D ----------
@always_inline
fn infer_shape_4d[T: ImplicitlyCopyable & Copyable & Movable](rows: List[List[List[List[T]]]]) -> (Int, Int, Int, Int):
    var a = len(rows)
    var b = 0
    var c = 0
    var d = 0
    if a > 0:
        b = len(rows[0])
        var i = 1
        while i < a:
            var lb = len(rows[i])
            if lb < b: b = lb
            i += 1
        if b > 0:
            c = len(rows[0][0])
            var ii = 0
            while ii < a:
                var j = 0
                while j < b:
                    var lc = len(rows[ii][j])
                    if lc < c: c = lc
                    j += 1
                ii += 1
            if c > 0:
                d = len(rows[0][0][0])
                var iii = 0
                while iii < a:
                    var jj = 0
                    while jj < b:
                        var kk = 0
                        while kk < c:
                            var ld = len(rows[iii][jj][kk])
                            if ld < d: d = ld
                            kk += 1
                        jj += 1
                    iii += 1
    return (a, b, c, d)

@always_inline
fn flatten_4d[T: ImplicitlyCopyable & Copyable & Movable](
    rows: List[List[List[List[T]]]], a: Int, b: Int, c: Int, d: Int
) -> List[T]:
    var out = List[T]()
    out.reserve(a * b * c * d)
    var i = 0
    while i < a:
        var r_i = rows[i].copy()
        var j = 0
        while j < b:
            var r_ij = r_i[j].copy()
            var k = 0
            while k < c:
                var r_ijk = r_ij[k].copy()
                var t = 0
                while t < d:
                    out.append(r_ijk[t])
                    t += 1
                k += 1
            j += 1
        i += 1
    return out.copy()

# ---------- 5D ----------
@always_inline
fn infer_shape_5d[T: ImplicitlyCopyable & Copyable & Movable](
    rows: List[List[List[List[List[T]]]]]
) -> (Int, Int, Int, Int, Int):
    var a = len(rows)
    var b = 0
    var c = 0
    var d = 0
    var e = 0
    if a > 0:
        b = len(rows[0])
        var i = 1
        while i < a:
            var lb = len(rows[i])
            if lb < b: b = lb
            i += 1
        if b > 0:
            c = len(rows[0][0])
            var ii = 0
            while ii < a:
                var j = 0
                while j < b:
                    var lc = len(rows[ii][j])
                    if lc < c: c = lc
                    j += 1
                ii += 1
            if c > 0:
                d = len(rows[0][0][0])
                var iii = 0
                while iii < a:
                    var jj = 0
                    while jj < b:
                        var kk = 0
                        while kk < c:
                            var ld = len(rows[iii][jj][kk])
                            if ld < d: d = ld
                            kk += 1
                        jj += 1
                    iii += 1
                if d > 0:
                    e = len(rows[0][0][0][0])
                    var i4 = 0
                    while i4 < a:
                        var j4 = 0
                        while j4 < b:
                            var k4 = 0
                            while k4 < c:
                                var t4 = 0
                                while t4 < d:
                                    var le = len(rows[i4][j4][k4][t4])
                                    if le < e: e = le
                                    t4 += 1
                                k4 += 1
                            j4 += 1
                        i4 += 1
    return (a, b, c, d, e)

@always_inline
fn flatten_5d[T: ImplicitlyCopyable & Copyable & Movable](
    rows: List[List[List[List[List[T]]]]],
    a: Int, b: Int, c: Int, d: Int, e: Int
) -> List[T]:
    var out = List[T]()
    out.reserve(a * b * c * d * e)
    var i = 0
    while i < a:
        var r_i = rows[i].copy()
        var j = 0
        while j < b:
            var r_ij = r_i[j].copy()
            var k = 0
            while k < c:
                var r_ijk = r_ij[k].copy()
                var t = 0
                while t < d:
                    var r_ijkt = r_ijk[t].copy()
                    var u = 0
                    while u < e:
                        out.append(r_ijkt[u])
                        u += 1
                    t += 1
                k += 1
            j += 1
        i += 1
    return out.copy()

@always_inline
fn fill[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], v: T) -> None:
    var r = len(x._shape)

    # Rank-0 (scalar)
    if r == 0:
        x._data[x._offset] = v
        return

    # Fast path: contiguous logical layout
    var expect = compute_row_major_strides(x._shape)
    var is_contig = True
    var i = 0
    while i < r:
        if x._strides[i] != expect[i]:
            is_contig = False
            break
        i += 1

    if is_contig:
        # Only the logical block starting at offset is written
        var total = 1
        i = 0
        while i < r:
            total = total * x._shape[i]
            i += 1
        var base = x._offset
        var k = 0
        while k < total:
            x._data[base + k] = v
            k += 1
        return

    # General ND path: odometer over shape using strides
    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var done = False
    while True:
        var lin = x._offset
        i = 0
        while i < r:
            lin = lin + idx[i] * x._strides[i]
            i += 1
        x._data[lin] = v

        # increment multi-index
        i = r - 1
        while True:
            if i < 0:
                done = True
                break
            idx[i] = idx[i] + 1
            if idx[i] < x._shape[i]:
                break
            idx[i] = 0
            i = i - 1
        if done:
            break



@always_inline
fn _coords_from_linear(shape: List[Int], rm: List[Int], lin: Int) -> List[Int]:
    var r = len(shape)
    var coord = List[Int]()
    coord.reserve(r)
    var d = 0
    while d < r:
        var s = 0
        if shape[d] != 0 and rm[d] != 0:
            s = (lin // rm[d]) % shape[d]
        coord.append(s)
        d += 1
    return coord.copy()


# -----------------------------------------------------------------------------
# put: write flat indices from `index` with values from `src` into a COPY of xx
# - Works with non-contiguous/view tensors (uses strides mapping)
# - Negative indices allowed (normalized by n); out-of-range are clamped
# - Broadcasting rules for `src`: scalar → all; len>=m → ith; len<m → repeat last
# - Last-write-wins on duplicate indices
# -----------------------------------------------------------------------------

@always_inline
fn put[T: ImplicitlyCopyable & Copyable & Movable](
    xx: Tensor[T], index: Tensor[Int], src: Tensor[T]
) -> Tensor[T]:
    var r = len(xx._shape)
    var n = 1
    var di = 0
    while di < r:
        n = n * xx._shape[di]
        di += 1
    if n == 0:
        return xx.copy()

    var m = 1
    var i = 0
    while i < len(index._shape):
        m = m * index._shape[i]
        i += 1
    if m == 0:
        return xx.copy()

    var sN = 1
    var j = 0
    while j < len(src._shape):
        sN = sN * src._shape[j]
        j += 1

    var out = xx.copy()

    var rm_idx = _row_major_multipliers(index._shape)
    var rm_dst = _row_major_multipliers(xx._shape)
    var rm_src = _row_major_multipliers(src._shape)

    var k = 0
    while k < m:
        var ioff = _offset_from_linear(index._shape, index._strides, index._offset, rm_idx, k)
        var pos = index._data[ioff]

        if pos < 0:
            pos = pos + n
        if pos < 0:
            pos = 0
        var maxp = n - 1
        if pos > maxp:
            pos = maxp

        var val_off = src._offset
        if not (len(src._shape) == 0 or sN == 1):
            var s_lin = k
            if s_lin >= sN:
                s_lin = sN - 1
            val_off = _offset_from_linear(src._shape, src._strides, src._offset, rm_src, s_lin)

        var doff = _offset_from_linear(xx._shape, xx._strides, xx._offset, rm_dst, pos)
        out._data[doff] = src._data[val_off]

        k += 1
    return out.copy()
