# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.tensor.helpers
# File:         src/momijo/tensor/helpers.mojo
#
# Description:
#   High-performance list/shape/stride helpers and generic tensor utilities.
#   - Fast zero/reserve/copy for List[Int]/List[T] with loop unrolling
#   - Small integer utilities (min/max/clamp/normalize_axis)
#   - Shape/stride core: numel, row-major strides, contiguity checks
#   - Index transforms: unravel_index / lin_index (allocation-aware)
#   - Tensor helpers: size, dim0_length, ensure_{2d,4d}, maybe_copy, gather
#   - Generic casting via astype_with (converter-function)
#   - Flattened conversions via *_with variants (converter-function)
#
# Notes:
#   - No 'let' and no 'assert'.
#   - Generic functions require T: ImplicitlyCopyable & Copyable & Movable.
#   - Predictable branches; avoid hidden allocations in hot paths.

from collections.list import List
from momijo.tensor.tensor import Tensor
from momijo.tensor.cast import *
from momijo.tensor.math import *
from momijo.tensor.transform import clamp_int

# ---------------------------------------------------------------------------
# Public alias for a normalized 1-D slice triplet: (start, stop, step)
# ---------------------------------------------------------------------------
alias SliceSpec = (Int, Int, Int)

# ---------------------------------------------------------------------------
# IndexSel (index | slice | fancy)
# --------------------------------------------------------------------------- 
# Value-type selector (no shared refs/aliases)
struct IndexSel(ImplicitlyCopyable, Copyable, Movable):
    var tag:   Int8        # 0=index, 1=slice, 2=fancy
    var i:     Int         # for tag=0
    var start: Int         # for tag=1
    var stop:  Int         # for tag=1
    var step:  Int         # for tag=1
    var idxs:  List[Int]   # for tag=2

     
    fn __init__(out self, t: Int8, ii: Int, a: Int, b: Int, c: Int, js: List[Int]):
        self.tag   = t
        self.i     = ii
        self.start = a
        self.stop  = b
        self.step  = c
        self.idxs  = js.copy()        # avoid aliasing

    # Manual copy constructor (because List[Int] is not implicitly copyable)
    fn __copyinit__(out self, other: Self):
        self.tag   = other.tag
        self.i     = other.i
        self.start = other.start
        self.stop  = other.stop
        self.step  = other.step
        self.idxs  = other.idxs.copy()  # deep copy

    @staticmethod
    @always_inline
    fn index(i: Int) -> IndexSel:
        var js = List[Int]()
        return IndexSel(0, i, 0, 0, 1, js)

    @staticmethod
    @always_inline
    fn slice(a: Int, b: Int, s: Int = 1) -> IndexSel:
        var js = List[Int]()
        return IndexSel(1, 0, a, b, s, js)

    @staticmethod
    @always_inline
    fn fancy(js_in: List[Int]) -> IndexSel:
        return IndexSel(2, 0, 0, 0, 1, js_in.copy())


# ============================== Core list utils ==============================

@always_inline
fn reserve_len(mut xs: List[Int], n: Int) -> None:
    # Extends list by n zeros (not just capacity).
    xs.reserve(len(xs) + n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        xs.append(0); xs.append(0); xs.append(0); xs.append(0)
        xs.append(0); xs.append(0); xs.append(0); xs.append(0)
        i += 8
    while i < n:
        xs.append(0)
        i += 1

@always_inline
fn zeros(n: Int) -> List[Int]:
    var xs = List[Int]()
    xs.reserve(n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        xs.append(0); xs.append(0); xs.append(0); xs.append(0)
        xs.append(0); xs.append(0); xs.append(0); xs.append(0)
        i += 8
    while i < n:
        xs.append(0)
        i += 1
    return xs

# Generic fast copy (unified for all T).
@always_inline
fn copy_list[T: ImplicitlyCopyable & Copyable & Movable](src: List[T]) -> List[T]:
    var out = List[T]()
    var n = len(src)
    out.reserve(n)
    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out.append(src[i +  0]); out.append(src[i +  1])
        out.append(src[i +  2]); out.append(src[i +  3])
        out.append(src[i +  4]); out.append(src[i +  5])
        out.append(src[i +  6]); out.append(src[i +  7])
        out.append(src[i +  8]); out.append(src[i +  9])
        out.append(src[i + 10]); out.append(src[i + 11])
        out.append(src[i + 12]); out.append(src[i + 13])
        out.append(src[i + 14]); out.append(src[i + 15])
        i += 16
    while i < n:
        out.append(src[i])
        i += 1
    return out.copy()

# Thin wrappers to preserve legacy names.
@always_inline
fn copy_list_int(src: List[Int]) -> List[Int]:
    return copy_list[Int](src)

@always_inline
fn copy_list_T[T: ImplicitlyCopyable & Copyable & Movable](src: List[T]) -> List[T]:
    return copy_list[T](src)

# ============================== Small int helpers ============================

@always_inline
fn min_i(a: Int, b: Int) -> Int:
    if a < b: return a
    return b

@always_inline
fn max_i(a: Int, b: Int) -> Int:
    if a > b: return a
    return b

@always_inline
fn clamp_nonneg(x: Int) -> Int:
    if x < 0: return 0
    return x

@always_inline
fn normalize_axis(ax_in: Int, rank: Int) -> Int:
    if rank <= 0: return 0
    var ax = ax_in
    if ax < 0:
        # Bring negative into range without negative modulo
        while ax < 0:
            ax = ax + rank
    if ax >= rank:
        ax = ax % rank
    if ax < 0: return 0
    if ax >= rank: return rank - 1
    return ax

@always_inline
fn identity_perm(r: Int) -> List[Int]:
    var p = List[Int]()
    p.reserve(r)
    var i = 0
    while i < r:
        p.append(i)
        i += 1
    return p.copy()

@always_inline
fn same_shape(a: List[Int], b: List[Int]) -> Bool:
    var ra = len(a)
    if ra != len(b):
        return False
    var i = 0
    while i < ra:
        if a[i] != b[i]:
            return False
        i += 1
    return True

# ============================== Shape/stride core ============================

@always_inline
fn numel(shape: List[Int]) -> Int:
    var r = len(shape)
    if r == 0: return 1
    var n = 1
    var i = 0
    while i < r:
        var d = shape[i]
        if d <= 0: return 0
        n = n * d
        i += 1
    return n

@always_inline
fn numel_shape(shape: List[Int]) -> Int:
    var n = 1
    var r = len(shape)
    var i = 0
    while i < r:
        n = n * shape[i]
        i += 1
    return n

@always_inline
fn row_major_strides(shp: List[Int]) -> List[Int]:
    var r = len(shp)
    var out = List[Int]()
    out.reserve(r)
    if r == 0:
        return out.copy()
    out.append(1)              # placeholder; will reverse-fill
    var i = 1
    while i < r:
        out.append(1)
        i = i + 1
    # fill from the right
    var acc = 1
    var k = r - 1
    while k >= 0:
        out[k] = acc
        acc = acc * shp[k]
        k = k - 1
    return out.copy()

@always_inline
fn is_row_major_contiguous(shape: List[Int], strides: List[Int]) -> Bool:
    var r = len(shape)
    if len(strides) != r:
        return False
    var acc = 1
    var k = r - 1
    while k >= 0:
        if strides[k] != acc:
            return False
        acc = acc * shape[k]
        k -= 1
    return True

@always_inline
fn is_contig_2d(strides: List[Int], width: Int) -> Bool:
    if len(strides) != 2:
        return False
    return (strides[1] == 1) and (strides[0] == width)

@always_inline
fn ensure_strides(strides: List[Int], shape: List[Int]) -> List[Int]:
    # If given strides look valid, keep them; otherwise compute row-major.
    if len(strides) == len(shape):
        var i = 0
        var ok = True
        while i < len(strides):
            # accept non-negative strides; zero is fine (e.g., broadcasted dim)
            if strides[i] < 0:
                ok = False
                break
            i += 1
        if ok:
            return strides.copy()
    return row_major_strides(shape)
# ---- unravel_index / lin_index (allocation-aware) ----

@always_inline
fn unravel_index(lin: Int, shape: List[Int], mut out_idx: List[Int]) -> None:
    var r = len(shape)
    out_idx.clear()
    out_idx.reserve(r)

    var i = 0
    while i < r:
        out_idx.append(0)
        i = i + 1

    if r == 0:
        return

    var x = lin
    i = r - 1
    while i >= 0:
        var d = shape[i]
        var v = 0
        if d > 0:
            v = x % d
            x = x // d
        out_idx[i] = v
        i = i - 1

@always_inline
fn unravel_index(lin: Int, shape: List[Int]) -> List[Int]:
    var r = len(shape)
    var out_idx = List[Int]()
    out_idx.reserve(r)

    var i = 0
    while i < r:
        out_idx.append(0)
        i = i + 1

    if r == 0:
        return out_idx.copy()

    var x = lin
    i = r - 1
    while i >= 0:
        var d = shape[i]
        var v = 0
        if d > 0:
            v = x % d
            x = x // d
        out_idx[i] = v
        i = i - 1
    return out_idx.copy()

@always_inline
fn lin_index(idx: List[Int], strides: List[Int]) -> Int:
    var n = len(idx)
    var li = 0
    var i = 0
    while i < n:
        li = li + idx[i] * strides[i]
        i += 1
    return li

# ============================== Tensor accessors =============================

@always_inline
fn size[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Int:
    return len(x._data)

@always_inline
fn dim0_length[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Int:
    if len(x._shape) == 0: return 0
    return x._shape[0]

@always_inline
fn ensure_2d[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[T]:
    var shp = x._shape
    var r = len(shp)
    if r == 2: return x
    if r == 1: return x.reshape([shp[0], 1])
    if r == 0: return x.reshape([1, 1])
    var size0 = 1
    var i = 0
    while i < r - 1:
        size0 = size0 * shp[i]
        i += 1
    return x.reshape([size0, shp[r - 1]])

@always_inline
fn ensure_4d[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[T]:
    var shp = x._shape
    var r = len(shp)
    if r == 4: return x
    if r == 3: return x.reshape([1, shp[0], shp[1], shp[2]])
    if r == 2: return x.reshape([1, 1, shp[0], shp[1]])
    if r == 1: return x.reshape([1, 1, 1, shp[0]])
    return x.reshape([1, 1, 1, 1])

# Materialize a row-major contiguous copy of x.
# - Fast path: already row-major & offset==0 -> unrolled memcpy-like copy
# - General path: stride/offset-aware traversal with odometer counters
fn maybe_copy[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[T]:
    var shp = x._shape.copy()
    var n_all = numel(shp)

    # Early out for truly row-major contiguous with zero offset
    if is_row_major_contiguous(shp, x._strides) and x._offset == 0:
        var out = List[T]()
        out.reserve(n_all)

        var i = 0
        var lim = (n_all // 16) * 16
        while i < lim:
            out.append(x._data[i    ]); out.append(x._data[i +  1])
            out.append(x._data[i +  2]); out.append(x._data[i +  3])
            out.append(x._data[i +  4]); out.append(x._data[i +  5])
            out.append(x._data[i +  6]); out.append(x._data[i +  7])
            out.append(x._data[i +  8]); out.append(x._data[i +  9])
            out.append(x._data[i + 10]); out.append(x._data[i + 11])
            out.append(x._data[i + 12]); out.append(x._data[i + 13])
            out.append(x._data[i + 14]); out.append(x._data[i + 15])
            i += 16
        while i < n_all:
            out.append(x._data[i])
            i += 1

        var st = compute_row_major_strides(shp)
        return Tensor[T](out, shp, st, 0)

    # General path: flatten via strides/offset (works for any non-contiguous slice/view)
    var out2 = List[T]()
    out2.reserve(n_all)

    var ndim = len(shp)
    var idxs = List[Int]()
    idxs.reserve(ndim)
    var ax = 0
    while ax < ndim:
        idxs.append(0)
        ax += 1

    var base = x._offset
    var done: Bool = False
    while not done:
        # linear position from multi-index using strides
        var pos = base
        var k = 0
        while k < ndim:
            pos = pos + idxs[k] * x._strides[k]
            k += 1
        out2.append(x._data[pos])

        # odometer increment
        var d = ndim - 1
        while True:
            if d < 0:
                done = True
                break
            idxs[d] = idxs[d] + 1
            if idxs[d] < shp[d]:
                break
            idxs[d] = 0
            d = d - 1

    var st2 = compute_row_major_strides(shp)
    return Tensor[T](out2, shp, st2, 0)




 
# ============================== Flattened conversions ========================

# Converter-based flattened extractors. Provide f: (T) -> {Float64,Int,Bool}.
fn tolist[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], f: fn (T) -> Float64) -> List[Float64]:
    var shp = x._shape
    var n = numel(shp)
    var out = List[Float64]()
    out.reserve(n)
    var idx = List[Int]()
    var i = 0
    while i < n:
        unravel_index(i, shp, idx)
        var li = lin_index(idx, x._strides)
        out.append(f(x._data[li]))
        i += 1
    return out

fn to_int_list[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], f: fn (T) -> Int) -> List[Int]:
    var shp = x._shape
    var n = numel(shp)
    var out = List[Int]()
    out.reserve(n)
    var idx = List[Int]()
    var i = 0
    while i < n:
        unravel_index(i, shp, idx)
        var li = lin_index(idx, x._strides)
        out.append(f(x._data[li]))
        i += 1
    return out

fn to_bool_list[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], f: fn (T) -> Bool) -> List[Bool]:
    var shp = x._shape
    var n = numel(shp)
    var out = List[Bool]()
    out.reserve(n)
    var idx = List[Int]()
    var i = 0
    while i < n:
        unravel_index(i, shp, idx)
        var li = lin_index(idx, x._strides)
        out.append(f(x._data[li]))
        i += 1
    return out

@always_inline
fn f64_of_f64(x: Float64) -> Float64:
    return x

@always_inline
fn f64_of_f32(x: Float32) -> Float64:
    return x as Float64

@always_inline
fn f64_of_int(x: Int) -> Float64:
    return x as Float64


@always_inline
fn int_of_f64(x: Float64) -> Int:
    return x as Int

@always_inline
fn int_of_f32(x: Float32) -> Int:
    return x as Int

@always_inline
fn int_of_int(x: Int) -> Int:
    return x


@always_inline
fn bool_of_f64(x: Float64) -> Bool:
    return x != 0.0

@always_inline
fn bool_of_f32(x: Float32) -> Bool:
    return x != 0.0

@always_inline
fn bool_of_int(x: Int) -> Bool:
    return x != 0


# ============================== Low-level blocks =============================

@always_inline 
fn append_block_unrolled16[T: ImplicitlyCopyable & Copyable & Movable](
    mut dst: List[T], src: List[T], src_lo: Int, src_hi: Int
) -> None:
    var n = src_hi - src_lo
    if n <= 0: return
    var i = 0
    var base = src_lo
    var lim = (n // 16) * 16
    while i < lim:
        dst.append(src[base + i + 0])
        dst.append(src[base + i + 1])
        dst.append(src[base + i + 2])
        dst.append(src[base + i + 3])
        dst.append(src[base + i + 4])
        dst.append(src[base + i + 5])
        dst.append(src[base + i + 6])
        dst.append(src[base + i + 7])
        dst.append(src[base + i + 8])
        dst.append(src[base + i + 9])
        dst.append(src[base + i + 10])
        dst.append(src[base + i + 11])
        dst.append(src[base + i + 12])
        dst.append(src[base + i + 13])
        dst.append(src[base + i + 14])
        dst.append(src[base + i + 15])
        i += 16
    while i < n:
        dst.append(src[base + i])
        i += 1


@always_inline
fn append_repeat_unrolled8[T: ImplicitlyCopyable & Copyable & Movable](
    mut dst: List[T], v: T, count: Int
) -> None:
    var i = 0
    var lim = (count // 8) * 8
    while i < lim:
        dst.append(v); dst.append(v); dst.append(v); dst.append(v)
        dst.append(v); dst.append(v); dst.append(v); dst.append(v)
        i += 8
    while i < count:
        dst.append(v)
        i += 1

@always_inline
fn lists_equal_int(a: List[Int], b: List[Int]) -> Bool:
    return same_shape(a, b)

# ============================== Views / shapes ===============================

fn clone_header_share_data[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    shape: List[Int],
    strides: List[Int]
) -> Tensor[T]:
    # sanitize shape (no negatives)
    var rank = len(shape)
    var clean_shape = List[Int]()
    clean_shape.reserve(rank)
    var i = 0
    while i < rank:
        var d = shape[i]
        if d < 0:
            d = 0
        clean_shape.append(d)
        i = i + 1

    # if caller gave strides of the right length, keep them; else use row-major
    var have_strides = (len(strides) == rank)

    # fast path: row-major contiguous → can alias same flat buffer
    if have_strides:
        var want = row_major_strides(clean_shape)
        var is_row_major = True
        i = 0
        while i < rank:
            if strides[i] != want[i]:
                is_row_major = False
                break
            i = i + 1

        if is_row_major:
            # constructor signature is (shape, flat)
            # this shares the same List buffer with x (assuming ctor doesn't copy)
            return Tensor[T](clean_shape, x._data) 

    # generic path: materialize a contiguous row-major copy following given strides
    var out_n = 1
    i = 0
    while i < rank:
        out_n = out_n * clean_shape[i]
        i = i + 1

    var flat = List[T]()
    flat.reserve(out_n)

    if rank == 0:
        # scalar-like
        if len(x._data) > 0:
            flat.append(x._data[0])
        return Tensor[T](clean_shape, flat)

    # iterate logical indices and gather using provided strides (or row-major if none)
    var use_strides: List[Int]
    if have_strides:
        use_strides = strides.copy()
    else:
        use_strides = row_major_strides(clean_shape)

    var counter = List[Int]()
    counter.reserve(rank)
    i = 0
    while i < rank:
        counter.append(0)
        i = i + 1

    var done = False
    while not done:
        var lin = 0
        var ax = 0
        while ax < rank:
            lin = lin + counter[ax] * use_strides[ax]
            ax = ax + 1
        flat.append(x._data[lin])

        # increment mixed-radix counter
        ax = rank - 1
        while ax >= 0:
            counter[ax] = counter[ax] + 1
            if counter[ax] < clean_shape[ax]:
                break
            counter[ax] = 0
            ax = ax - 1
        if ax < 0:
            done = True

    return Tensor[T](clean_shape, flat)


@always_inline
fn shape_drop_axis(base: List[Int], axis: Int) -> List[Int]:
    var out = List[Int]()
    var r = len(base)
    var i = 0
    while i < r:
        if i != axis:
            out.append(base[i])
        i += 1
    if len(out) == 0:
        out.append(1)
    return out.copy()

fn compute_new_shape(old_shape: List[Int], pad_pairs: List[(Int, Int)]) -> List[Int]:
    var rank = len(old_shape)
    var out = List[Int]()
    out.reserve(rank)
    if len(pad_pairs) != rank:
        return out.copy()
    var d = 0
    while d < rank:
        var before = clamp_nonneg(pad_pairs[d][0])
        var after  = clamp_nonneg(pad_pairs[d][1])
        var nd = old_shape[d] + before + after
        if nd < 0:
            nd = 0
        out.append(nd)
        d += 1
    return out.copy()

@always_inline
fn zero_scalar_of[T: ImplicitlyCopyable & Copyable & Movable](f: fn (Float64) -> T) -> T:
    return f(0.0)
# @always_inline
# fn zero_scalar_of[T: ImplicitlyCopyable & Copyable & Movable]() -> T:
#     # Return the default "zero" value for T
#     var z = T()
#     return z


fn one_scalar_of[T: ImplicitlyCopyable & Copyable & Movable]() -> T:
    var d = List[Int64](); d.append(1)
    var o = astype[Int64, T](Tensor[Int64](d))
    return o._data[0]


# 1) same-type overload (T == T)
fn write_planeT: ImplicitlyCopyable & Copyable & Movable](
    mut a: Tensor[T], dim: Int, index: Int, rhs: Tensor[T]
) -> None:
    var shp = a.shape()
    if len(shp) != 3: return
    var B = shp[0]; var M = shp[1]; var N = shp[2]
    if dim != 2: return
    if not (index >= 0 and index < N): return

    var rsh = rhs.shape()
    if len(rsh) != 2: return
    if rsh[0] != B: return
    if not (rsh[1] == 1 or rsh[1] == M): return

    var s0 = a._strides[0]; var s1 = a._strides[1]; var s2 = a._strides[2]

    var i = 0
    while i < B:
        var row = rhs.index_axis(0, i)
        var j = 0
        while j < M:
            var jj = 0
            if rsh[1] == 1:
                jj = 0
            else:
                jj = j
            var v = row[jj]
            var lin = i * s0 + j * s1 + index * s2
            a._data[lin] = v
            j = j + 1
        i = i + 1



@always_inline
fn write_plane(
    mut a: Tensor[Float64], dim: Int, index: Int, rhs: Tensor[Int]
) -> None:
    # a: (B, M, N) ، فقط وقتی dim == 2 (بعد آخر) می‌نویسیم
    var shp = a.shape()
    if len(shp) != 3: return
    var B = shp[0]; var M = shp[1]; var N = shp[2]
    if dim != 2: return
    if not (index >= 0 and index < N): return

    # rhs: (B, 1) یا (B, M)
    var rsh = rhs.shape()
    if len(rsh) != 2: return
    if rsh[0] != B: return
    if not (rsh[1] == 1 or rsh[1] == M): return

    # strides و offset در a
    var s0 = a._strides[0]
    var s1 = a._strides[1]
    var s2 = a._strides[2]
    var a_off = a._offset

    var i = 0
    while i < B:
        # row شکلش (M,) یا (1,) است
        var row = rhs.index_axis(0, i)

        # دسترسی امن به اسکالر از row (با درنظرگرفتن آفست و استراید)
        var row_off = row._offset
        var row_st0 = row._strides[0]

        var j = 0
        while j < M:
            var jj: Int
            if rsh[1] == 1:
                jj = 0
            else:
                jj = j

            # مقدار Int اسکالر از row
            var rij = row._data[row_off + jj * row_st0]
            var v = Float64(rij)

            # نوشتن در صفحه‌ی dim==2 در a (با درنظرگرفتن offset)
            var lin = a_off + i * s0 + j * s1 + index * s2
            a._data[lin] = v

            j = j + 1
        i = i + 1




# ============================== Contiguity (unified) =========================

@always_inline
fn is_contiguous(shape: List[Int], strides: List[Int], row_major: Bool = True) -> Bool:
    if row_major:
        return is_row_major_contiguous(shape, strides)
    else:
        # Only row-major supported in this helper set; add col-major if needed.
        return False

fn is_contiguous_tensor[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Bool:
    return is_row_major_contiguous(x._shape, x._strides)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
@always_inline
fn _copy_shape(src: Tensor[any]) -> List[Int]:
    var shp = List[Int]()
    var d = len(src._shape)
    var i = 0
    while i < d:
        shp.append(src._shape[i])
        i += 1
    return shp.copy()

@always_inline
fn _is_row_major_contiguous(x: Tensor[any]) -> Bool:
    # Contiguous if offset==0 and strides match row-major
    if x._offset != 0:
        return False
    var want = compute_row_major_strides(x._shape)
    var r = len(want)
    if r != len(x._strides):
        return False
    var i = 0
    while i < r:
        if x._strides[i] != want[i]:
            return False
        i += 1
    return True

# ------------------------------------------------------------
# Core generic: use a user-supplied converter function
# ------------------------------------------------------------
@always_inline
fn astype_with[T: ImplicitlyCopyable & Copyable & Movable,
               U: ImplicitlyCopyable & Copyable & Movable](
    src: Tensor[T],
    f: fn (T) -> U
) -> Tensor[U]:
    # Materialize shape and element count
    var shp = _copy_shape(src)
    var n = numel(shp)

    # Fast path: 1D or contiguous row-major — read raw buffer linearly
    if _is_row_major_contiguous(src):
        var out = List[U]()
        out.reserve(n)

        var i = 0
        var lim = (n // 16) * 16
        while i < lim:
            out.append(f(src._data[i     ]))
            out.append(f(src._data[i +  1]))
            out.append(f(src._data[i +  2]))
            out.append(f(src._data[i +  3]))
            out.append(f(src._data[i +  4]))
            out.append(f(src._data[i +  5]))
            out.append(f(src._data[i +  6]))
            out.append(f(src._data[i +  7]))
            out.append(f(src._data[i +  8]))
            out.append(f(src._data[i +  9]))
            out.append(f(src._data[i + 10]))
            out.append(f(src._data[i + 11]))
            out.append(f(src._data[i + 12]))
            out.append(f(src._data[i + 13]))
            out.append(f(src._data[i + 14]))
            out.append(f(src._data[i + 15]))
            i += 16
        while i < n:
            out.append(f(src._data[i]))
            i += 1

        var st = compute_row_major_strides(shp)
        return Tensor[U](out, shp, st, 0)

    # Strided path: respect offset/strides (works for any slicing)
    var out2 = List[U]()
    out2.reserve(n)

    var ndim = len(src._shape)
    if ndim == 0:
        # Scalar-shaped tensor
        var st0 = compute_row_major_strides(shp)
        return Tensor[U](out2, shp, st0, 0)

    # Multi-dimensional odometer
    var idxs = List[Int]()
    idxs.reserve(ndim)
    var ax = 0
    while ax < ndim:
        idxs.append(0)
        ax += 1

    var base = src._offset
    var done: Bool = False
    while not done:
        var pos = base
        var k = 0
        while k < ndim:
            pos = pos + idxs[k] * src._strides[k]
            k += 1
        out2.append(f(src._data[pos]))

        # increment odometer
        var d = ndim - 1
        while True:
            if d < 0:
                done = True
                break
            idxs[d] = idxs[d] + 1
            if idxs[d] < src._shape[d]:
                break
            idxs[d] = 0
            d = d - 1

    var st2 = compute_row_major_strides(shp)
    return Tensor[U](out2, shp, st2, 0)

# ------------------------------------------------------------
# Via Float64 bridge (T -> Float64 -> U)
# ------------------------------------------------------------
@always_inline
fn astype_via_f64[T: ImplicitlyCopyable & Copyable & Movable,
                  U: ImplicitlyCopyable & Copyable & Movable](
    src: Tensor[T],
    to_f64: fn (T) -> Float64,
    from_f64: fn (Float64) -> U
) -> Tensor[U]:
    var shp = _copy_shape(src)
    var n = numel(shp)

    # Fast path for contiguous buffers
    if _is_row_major_contiguous(src):
        var out = List[U]()
        out.reserve(n)

        var k = 0
        var lim = (n // 16) * 16
        while k < lim:
            out.append(from_f64(to_f64(src._data[k     ])))
            out.append(from_f64(to_f64(src._data[k +  1])))
            out.append(from_f64(to_f64(src._data[k +  2])))
            out.append(from_f64(to_f64(src._data[k +  3])))
            out.append(from_f64(to_f64(src._data[k +  4])))
            out.append(from_f64(to_f64(src._data[k +  5])))
            out.append(from_f64(to_f64(src._data[k +  6])))
            out.append(from_f64(to_f64(src._data[k +  7])))
            out.append(from_f64(to_f64(src._data[k +  8])))
            out.append(from_f64(to_f64(src._data[k +  9])))
            out.append(from_f64(to_f64(src._data[k + 10])))
            out.append(from_f64(to_f64(src._data[k + 11])))
            out.append(from_f64(to_f64(src._data[k + 12])))
            out.append(from_f64(to_f64(src._data[k + 13])))
            out.append(from_f64(to_f64(src._data[k + 14])))
            out.append(from_f64(to_f64(src._data[k + 15])))
            k += 16
        while k < n:
            out.append(from_f64(to_f64(src._data[k])))
            k += 1

        var st = compute_row_major_strides(shp)
        return Tensor[U](out, shp, st, 0)

    # Strided path
    var out2 = List[U]()
    out2.reserve(n)

    var ndim = len(src._shape)
    if ndim == 0:
        var st0 = compute_row_major_strides(shp)
        return Tensor[U](out2, shp, st0, 0)

    var idxs = List[Int]()
    idxs.reserve(ndim)
    var ax = 0
    while ax < ndim:
        idxs.append(0)
        ax += 1

    var base = src._offset
    var done: Bool = False
    while not done:
        var pos = base
        var j = 0
        while j < ndim:
            pos = pos + idxs[j] * src._strides[j]
            j += 1
        out2.append(from_f64(to_f64(src._data[pos])))

        var d = ndim - 1
        while True:
            if d < 0:
                done = True
                break
            idxs[d] = idxs[d] + 1
            if idxs[d] < src._shape[d]:
                break
            idxs[d] = 0
            d = d - 1

    var st2 = compute_row_major_strides(shp)
    return Tensor[U](out2, shp, st2, 0)


@always_inline
fn astype[T: ImplicitlyCopyable & Copyable & Movable](src: Tensor[T], _dst: Float64) -> Tensor[Float64]:
    return astype_via_f64[T, Float64](src, to_float64_of, f64_to)

@always_inline
fn astype[T: ImplicitlyCopyable & Copyable & Movable](src: Tensor[T], _dst: Float32) -> Tensor[Float32]:
    return astype_via_f64[T, Float32](src, to_float64_of, f64_to_float32)

@always_inline
fn astype[T: ImplicitlyCopyable & Copyable & Movable](src: Tensor[T], _dst: Int8) -> Tensor[Int8]:
    return astype_via_f64[T, Int8](src, to_float64_of, f64_to_int8)

@always_inline
fn astype[T: ImplicitlyCopyable & Copyable & Movable](src: Tensor[T], _dst: Int16) -> Tensor[Int16]:
    return astype_via_f64[T, Int16](src, to_float64_of, f64_to_int16)

@always_inline
fn astype[T: ImplicitlyCopyable & Copyable & Movable](src: Tensor[T], _dst: Int32) -> Tensor[Int32]:
    return astype_via_f64[T, Int32](src, to_float64_of, f64_to_int32)

@always_inline
fn astype[T: ImplicitlyCopyable & Copyable & Movable](src: Tensor[T], _dst: Int64) -> Tensor[Int64]:
    return astype_via_f64[T, Int64](src, to_float64_of, f64_to_int64)

@always_inline
fn astype[T: ImplicitlyCopyable & Copyable & Movable](src: Tensor[T], _dst: Int) -> Tensor[Int]:
    return astype_via_f64[T, Int](src, to_float64_of, f64_to_int)

@always_inline
fn astype[T: ImplicitlyCopyable & Copyable & Movable](src: Tensor[T], _dst: UInt8) -> Tensor[UInt8]:
    return astype_via_f64[T, UInt8](src, to_float64_of, f64_to_uint8)

@always_inline
fn astype[T: ImplicitlyCopyable & Copyable & Movable](src: Tensor[T], _dst: UInt16) -> Tensor[UInt16]:
    return astype_via_f64[T, UInt16](src, to_float64_of, f64_to_uint16)

@always_inline
fn astype[T: ImplicitlyCopyable & Copyable & Movable](src: Tensor[T], _dst: UInt32) -> Tensor[UInt32]:
    return astype_via_f64[T, UInt32](src, to_float64_of, f64_to_uint32)

@always_inline
fn astype[T: ImplicitlyCopyable & Copyable & Movable](src: Tensor[T], _dst: UInt64) -> Tensor[UInt64]:
    return astype_via_f64[T, UInt64](src, to_float64_of, f64_to_uint64)

@always_inline
fn _is_matrix(shape: List[Int]) -> Bool:
    return len(shape) == 2

@always_inline
fn _is_vector(shape: List[Int]) -> Bool:
    return len(shape) == 1

# Row-major strides for a given shape. For 0-D returns [].
@always_inline
fn compute_row_major_strides(shape: List[Int]) -> List[Int]:
    var r = len(shape)
    # Pre-size with zeros
    var st = List[Int]()
    st.reserve(r)
    var i = 0
    while i < r:
        st.append(0)
        i += 1
    if r == 0:
        return st.copy()   # empty
    # Last axis stride = 1
    st[r - 1] = 1
    # Backward pass to fill strides
    var k = r - 2
    while k >= 0:
        st[k] = st[k + 1] * shape[k + 1]
        k -= 1
    return st.copy() 

               

# helpers
fn copy_flat_list[T: ImplicitlyCopyable & Copyable & Movable](t: Tensor[T]) -> List[T]:
    var out = List[T]()
    out.reserve(len(t._data))
    var i = 0
    while i < len(t._data):
        out.append(t._data[i])
        i = i + 1
    return out.copy() 

fn insertion_sort[T: ImplicitlyCopyable & Copyable & Movable](xs: mut List[T]) -> None:
    var n = len(xs)
    var i = 1
    while i < n:
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j = j - 1
        xs[j + 1] = key
        i = i + 1

fn unique_from_sorted[T: ImplicitlyCopyable & Copyable & Movable](xs: List[T]) -> List[T]:
    var out = List[T]()
    var n = len(xs)
    if n == 0:
        return out
    out.append(xs[0])
    var i = 1
    while i < n:
        if xs[i] != out[len(out) - 1]:
            out.append(xs[i])
        i = i + 1
    return out.copy() 

fn sorted_unique[T: ImplicitlyCopyable & Copyable & Movable](t: Tensor[T]) -> List[T]:
    var xs = copy_flat_list(t)
    insertion_sort(xs)
    return unique_from_sorted(xs)



# comparators for common types
@always_inline
fn eq_int(a: Int, b: Int) -> Bool:
    return a == b

@always_inline
fn lt_int(a: Int, b: Int) -> Bool:
    return a < b

@always_inline
fn eq_f64(a: Float64, b: Float64) -> Bool:
    return a == b

@always_inline
fn lt_f64(a: Float64, b: Float64) -> Bool:
    return a < b

# ---- helpers used by Float64 pipeline ----
fn setops_copy_flat_list[T: ImplicitlyCopyable & Copyable & Movable](t: Tensor[T]) -> List[T]:
    var out = List[T]()
    out.reserve(len(t._data))
    var i = 0
    while i < len(t._data):
        out.append(t._data[i])
        i = i + 1
    return out.copy()

# ---------- Int pipeline (no function-typed args) ----------
# Set-ops insertion sort (Int)
fn setops_insertion_sort_int(mut xs: List[Int]) -> None:
    var n = len(xs)
    var i = 1
    while i < n:
        var key = xs[i]
        var j = i - 1
        while j >= 0 and key < xs[j]:
            xs[j + 1] = xs[j]
            j = j - 1
        xs[j + 1] = key
        i = i + 1


@always_inline
fn setops_unique_from_sorted_int(xs: List[Int]) -> List[Int]:
    var n = len(xs)
    var out = List[Int]()
    if n == 0:
        return out.copy()

    out.reserve(n)
    out.append(xs[0])
    var last = xs[0]

    var i = 1
    while i < n:
        var v = xs[i]
        if v != last:
            out.append(v)
            last = v
        i = i + 1

    return out.copy()


fn setops_sorted_unique_int(t: Tensor[Int]) -> List[Int]:
    var xs = setops_copy_flat_list[Int](t)
    setops_insertion_sort_int(xs)
    return setops_unique_from_sorted_int(xs)


# Casting Int tensor -> Float64 tensor (stride/offset aware)
fn astype_float64(x: Tensor[Int]) -> Tensor[Float64]:
    var shp = x._shape.copy()
    var ints = setops_copy_flat_list[Int](x)
    var n = len(ints)

    var flat = List[Float64]()
    flat.reserve(n)

    var i = 0
    while i < n:
        flat.append(Float64(ints[i]))
        i = i + 1

    # Constructor expects (data, shape)
    return Tensor[Float64](flat, shp)


# Set-ops insertion sort (Float64)
fn setops_insertion_sort_f64(mut xs: List[Float64]) -> None:
    var n = len(xs)
    var i = 1
    while i < n:
        var key = xs[i]
        var j = i - 1
        while j >= 0 and key < xs[j]:
            xs[j + 1] = xs[j]
            j = j - 1
        xs[j + 1] = key
        i = i + 1


@always_inline
fn setops_unique_from_sorted_f64(xs: List[Float64]) -> List[Float64]:
    var n = len(xs)
    var out = List[Float64]()
    if n == 0:
        return out.copy()

    out.reserve(n)
    out.append(xs[0])
    var last = xs[0]

    var i = 1
    while i < n:
        var v = xs[i]
        if v != last:
            out.append(v)
            last = v
        i = i + 1

    return out.copy()


fn setops_sorted_unique_f64(t: Tensor[Float64]) -> List[Float64]:
    var xs = setops_copy_flat_list[Float64](t)
    setops_insertion_sort_f64(xs)
    return setops_unique_from_sorted_f64(xs)


@always_inline
fn insertion_sort_inplace(mut xs: List[Int]) -> None:
    var n = len(xs)
    var k = 1
    while k < n:
        var key = xs[k]
        var j = k - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j = j - 1
        xs[j + 1] = key
        k = k + 1



# ---------------- public free functions (Int only) ------------- 
# Ascending sort for Tensor[Int]
# - Fast 1D path (stride-aware)
# - General N-D path via flatten with strides/offset
fn sort_int(x: Tensor[Int]) -> Tensor[Int]:
    # 1D fast path
    if len(x._shape) == 1:
        var n = x._shape[0]

        var data = List[Int]()
        data.reserve(n)

        var base = x._offset
        var step = x._strides[0]

        var i = 0
        while i < n:
            data.append(x._data[base + i * step])
            i += 1

        insertion_sort_inplace(data)

        var shape = List[Int]()
        shape.append(n)

        var strides = compute_row_major_strides(shape)
        return Tensor[Int](data, shape, strides, 0)

    # General N-D: flatten respecting strides/offset, then sort
    var total = numel(x._shape)

    var flat = List[Int]()
    flat.reserve(total)

    var ndim = len(x._shape)
    var idxs = List[Int]()
    idxs.reserve(ndim)

    var ax = 0
    while ax < ndim:
        idxs.append(0)
        ax += 1

    var base = x._offset
    var done: Bool = False

    while not done:
        var pos = base
        var k = 0
        while k < ndim:
            pos = pos + idxs[k] * x._strides[k]
            k += 1
        flat.append(x._data[pos])

        # odometer-style increment
        var d = ndim - 1
        while True:
            if d < 0:
                done = True
                break
            idxs[d] = idxs[d] + 1
            if idxs[d] < x._shape[d]:
                break
            idxs[d] = 0
            d = d - 1

    insertion_sort_inplace(flat)

    var shape_flat = List[Int]()
    shape_flat.append(total)
    var strides_flat = compute_row_major_strides(shape_flat)
    return Tensor[Int](flat, shape_flat, strides_flat, 0)

# ---------------- helpers (Float64 / Float32) ----------------
@always_inline
fn _insertion_sort_inplace_f64(mut a: List[Float64]) -> None:
    var n = len(a)
    var i = 1
    while i < n:
        var key = a[i]
        var j = i - 1 
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j = j - 1
        a[j + 1] = key
        i = i + 1

@always_inline
fn _insertion_sort_inplace_f32(mut a: List[Float32]) -> None:
    var n = len(a)
    var i = 1
    while i < n:
        var key = a[i]
        var j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j = j - 1
        a[j + 1] = key
        i = i + 1


# ---------------- public free functions (Float64) -------------
# Ascending sort for Tensor[Float64]
# - Fast 1D path (stride-aware)
# - General N-D path via flatten with strides/offset
@always_inline
fn sort_f64(x: Tensor[Float64]) -> Tensor[Float64]:
    # 1D fast path
    if len(x._shape) == 1:
        var n = x._shape[0]

        var data = List[Float64]()
        data.reserve(n)

        var base = x._offset
        var step = x._strides[0]

        var i = 0
        while i < n:
            data.append(x._data[base + i * step])
            i = i + 1

        _insertion_sort_inplace_f64(data)

        var shape = List[Int]()
        shape.append(n)

        var strides = compute_row_major_strides(shape)
        return Tensor[Float64](data, shape, strides, 0)

    # General N-D: flatten respecting strides/offset, then sort
    var total = numel(x._shape)

    var flat = List[Float64]()
    flat.reserve(total)

    var ndim = len(x._shape)
    var idxs = List[Int]()
    idxs.reserve(ndim)

    var ax = 0
    while ax < ndim:
        idxs.append(0)
        ax = ax + 1

    var base = x._offset
    var done: Bool = False

    while not done:
        var pos = base
        var k = 0
        while k < ndim:
            pos = pos + idxs[k] * x._strides[k]
            k = k + 1
        flat.append(x._data[pos])

        # odometer-style increment
        var d = ndim - 1
        while True:
            if d < 0:
                done = True
                break
            idxs[d] = idxs[d] + 1
            if idxs[d] < x._shape[d]:
                break
            idxs[d] = 0
            d = d - 1

    _insertion_sort_inplace_f64(flat)

    var shape_flat = List[Int]()
    shape_flat.append(total)
    var strides_flat = compute_row_major_strides(shape_flat)
    return Tensor[Float64](flat, shape_flat, strides_flat, 0)


# ---------------- public free functions (Float32) -------------
# Ascending sort for Tensor[Float32]
# - Fast 1D path (stride-aware)
# - General N-D path via flatten with strides/offset
@always_inline
fn sort_f32(x: Tensor[Float32]) -> Tensor[Float32]:
    # 1D fast path
    if len(x._shape) == 1:
        var n = x._shape[0]

        var data = List[Float32]()
        data.reserve(n)

        var base = x._offset
        var step = x._strides[0]

        var i = 0
        while i < n:
            data.append(x._data[base + i * step])
            i = i + 1

        _insertion_sort_inplace_f32(data)

        var shape = List[Int]()
        shape.append(n)

        var strides = compute_row_major_strides(shape)
        return Tensor[Float32](data, shape, strides, 0)

    # General N-D: flatten respecting strides/offset, then sort
    var total = numel(x._shape)

    var flat = List[Float32]()
    flat.reserve(total)

    var ndim = len(x._shape)
    var idxs = List[Int]()
    idxs.reserve(ndim)

    var ax = 0
    while ax < ndim:
        idxs.append(0)
        ax = ax + 1

    var base = x._offset
    var done: Bool = False

    while not done:
        var pos = base
        var k = 0
        while k < ndim:
            pos = pos + idxs[k] * x._strides[k]
            k = k + 1
        flat.append(x._data[pos])

        # odometer-style increment
        var d = ndim - 1
        while True:
            if d < 0:
                done = True
                break
            idxs[d] = idxs[d] + 1
            if idxs[d] < x._shape[d]:
                break
            idxs[d] = 0
            d = d - 1

    _insertion_sort_inplace_f32(flat)

    var shape_flat = List[Int]()
    shape_flat.append(total)
    var strides_flat = compute_row_major_strides(shape_flat)
    return Tensor[Float32](flat, shape_flat, strides_flat, 0)

# ======================= ARGSORT HELPERS =======================
# Stable insertion-argsort for parallel (vals, idxs).
@always_inline
fn _insertion_argsort_inplace_i64(mut vals: List[Int], mut idxs: List[Int]) -> None:
    var n = len(vals)
    var i = 1
    while i < n:
        var kv = vals[i]
        var ki = idxs[i]
        var j = i - 1
        while j >= 0 and vals[j] > kv:
            vals[j + 1] = vals[j]
            idxs[j + 1] = idxs[j]
            j = j - 1
        vals[j + 1] = kv
        idxs[j + 1] = ki
        i = i + 1

@always_inline
fn _insertion_argsort_inplace_f64(mut vals: List[Float64],mut  idxs: List[Int]) -> None:
    var n = len(vals)
    var i = 1
    while i < n:
        var kv = vals[i]
        var ki = idxs[i]
        var j = i - 1
        # Note: NaN comparisons are False; NaNs keep relative order (simple policy).
        while j >= 0 and vals[j] > kv:
            vals[j + 1] = vals[j]
            idxs[j + 1] = idxs[j]
            j = j - 1
        vals[j + 1] = kv
        idxs[j + 1] = ki
        i = i + 1

@always_inline
fn _insertion_argsort_inplace_f32(mut vals: List[Float32],mut  idxs: List[Int]) -> None:
    var n = len(vals)
    var i = 1
    while i < n:
        var kv = vals[i]
        var ki = idxs[i]
        var j = i - 1
        while j >= 0 and vals[j] > kv:
            vals[j + 1] = vals[j]
            idxs[j + 1] = idxs[j]
            j = j - 1
        vals[j + 1] = kv
        idxs[j + 1] = ki
        i = i + 1


# ======================= ARGSORT: INT ==========================
# Ascending argsort for Tensor[Int]
# - 1D fast path: respects strides/offset without materializing full tensor
# - N-D path: flattens by walking with strides/offset, then sorts indices
@always_inline
fn argsort_int(x: Tensor[Int]) -> Tensor[Int]:
    # 1D fast path
    if len(x._shape) == 1:
        var n = x._shape[0]
        var vals = List[Int]();    vals.reserve(n)
        var idxs = List[Int]();    idxs.reserve(n)
        var base = x._offset
        var step = x._strides[0]

        var i = 0
        while i < n:
            vals.append(x._data[base + i * step])
            idxs.append(i)                 # index in the flattened 1D view
            i = i + 1

        _insertion_argsort_inplace_i64(vals, idxs)

        var shape = List[Int](); shape.append(n)
        var strides = compute_row_major_strides(shape)
        return Tensor[Int](idxs, shape, strides, 0)

    # General N-D
    var total = numel(x._shape)
    var vals = List[Int]();  vals.reserve(total)
    var idxs = List[Int]();  idxs.reserve(total)

    # Odometer over N-D respecting strides/offset
    var ndim = len(x._shape)
    var coord = List[Int](); coord.reserve(ndim)
    var ax = 0
    while ax < ndim:
        coord.append(0)
        ax = ax + 1

    var base = x._offset
    var done: Bool = False
    var flat_i = 0
    while not done:
        var pos = base
        var k = 0
        while k < ndim:
            pos = pos + coord[k] * x._strides[k]
            k = k + 1
        vals.append(x._data[pos])
        idxs.append(flat_i)

        # increment
        var d = ndim - 1
        while True:
            if d < 0:
                done = True
                break
            coord[d] = coord[d] + 1
            if coord[d] < x._shape[d]:
                break
            coord[d] = 0
            d = d - 1
        flat_i = flat_i + 1

    _insertion_argsort_inplace_i64(vals, idxs)

    var shape_flat = List[Int](); shape_flat.append(total)
    var strides_flat = compute_row_major_strides(shape_flat)
    return Tensor[Int](idxs, shape_flat, strides_flat, 0)


# ======================= ARGSORT: FLOAT64 ======================
@always_inline
fn argsort_f64(x: Tensor[Float64]) -> Tensor[Int]:
    if len(x._shape) == 1:
        var n = x._shape[0]
        var vals = List[Float64](); vals.reserve(n)
        var idxs = List[Int]();     idxs.reserve(n)

        var base = x._offset
        var step = x._strides[0]
        var i = 0
        while i < n:
            vals.append(x._data[base + i * step])
            idxs.append(i)
            i = i + 1

        _insertion_argsort_inplace_f64(vals, idxs)

        var shape = List[Int](); shape.append(n)
        var strides = compute_row_major_strides(shape)
        return Tensor[Int](idxs, shape, strides, 0)

    var total = numel(x._shape)
    var vals = List[Float64](); vals.reserve(total)
    var idxs = List[Int]();     idxs.reserve(total)

    var ndim = len(x._shape)
    var coord = List[Int](); coord.reserve(ndim)
    var ax = 0
    while ax < ndim:
        coord.append(0)
        ax = ax + 1

    var base = x._offset
    var done: Bool = False
    var flat_i = 0
    while not done:
        var pos = base
        var k = 0
        while k < ndim:
            pos = pos + coord[k] * x._strides[k]
            k = k + 1
        vals.append(x._data[pos])
        idxs.append(flat_i)

        var d = ndim - 1
        while True:
            if d < 0:
                done = True
                break
            coord[d] = coord[d] + 1
            if coord[d] < x._shape[d]:
                break
            coord[d] = 0
            d = d - 1
        flat_i = flat_i + 1

    _insertion_argsort_inplace_f64(vals, idxs)

    var shape_flat = List[Int](); shape_flat.append(total)
    var strides_flat = compute_row_major_strides(shape_flat)
    return Tensor[Int](idxs, shape_flat, strides_flat, 0)


# ======================= ARGSORT: FLOAT32 ======================
@always_inline
fn argsort_f32(x: Tensor[Float32]) -> Tensor[Int]:
    if len(x._shape) == 1:
        var n = x._shape[0]
        var vals = List[Float32](); vals.reserve(n)
        var idxs = List[Int]();     idxs.reserve(n)

        var base = x._offset
        var step = x._strides[0]
        var i = 0
        while i < n:
            vals.append(x._data[base + i * step])
            idxs.append(i)
            i = i + 1

        _insertion_argsort_inplace_f32(vals, idxs)

        var shape = List[Int](); shape.append(n)
        var strides = compute_row_major_strides(shape)
        return Tensor[Int](idxs, shape, strides, 0)

    var total = numel(x._shape)
    var vals = List[Float32](); vals.reserve(total)
    var idxs = List[Int]();     idxs.reserve(total)

    var ndim = len(x._shape)
    var coord = List[Int](); coord.reserve(ndim)
    var ax = 0
    while ax < ndim:
        coord.append(0)
        ax = ax + 1

    var base = x._offset
    var done: Bool = False
    var flat_i = 0
    while not done:
        var pos = base
        var k = 0
        while k < ndim:
            pos = pos + coord[k] * x._strides[k]
            k = k + 1
        vals.append(x._data[pos])
        idxs.append(flat_i)

        var d = ndim - 1
        while True:
            if d < 0:
                done = True
                break
            coord[d] = 0 if coord[d] + 1 == x._shape[d] else coord[d] + 1
            if coord[d] != 0:
                break
            d = d - 1
        flat_i = flat_i + 1

    _insertion_argsort_inplace_f32(vals, idxs)

    var shape_flat = List[Int](); shape_flat.append(total)
    var strides_flat = compute_row_major_strides(shape_flat)
    return Tensor[Int](idxs, shape_flat, strides_flat, 0)



@always_inline
fn unique_int(x: Tensor[Int], return_counts: Bool) -> UniqueResult[Int]:
    # Flatten (stride/offset aware) and sort
    var ys = setops_copy_flat_list[Int](x)
    var n = len(ys)

    if n == 0:
        # unique: [], counts: [] (shape=[0])
        var shp0 = List[Int](); shp0.append(0)
        var st0  = compute_row_major_strides(shp0)

        var u0 = Tensor[Int](List[Int](), shp0, st0, 0)
        var c0 = Tensor[Int](List[Int](), shp0, st0, 0)

        if return_counts:
            return UniqueResult[Int](u0, c0)
        else:
            return UniqueResult[Int](u0, c0)   # counts empty (shape=[0])

    insertion_sort_inplace(ys)

    # Run-length encode
    var vals = List[Int](); vals.reserve(n)
    var cnts = List[Int](); cnts.reserve(n)

    var cur = ys[0]
    var cnt = 1
    var i = 1
    while i < n:
        var v = ys[i]
        if v == cur:
            cnt = cnt + 1
        else:
            vals.append(cur)
            cnts.append(cnt)
            cur = v
            cnt = 1
        i = i + 1
    vals.append(cur)
    cnts.append(cnt)

    # Build 1D tensors with explicit strides/offset to avoid ctor ambiguity
    var shp_u = List[Int](); shp_u.append(len(vals))
    var st_u  = compute_row_major_strides(shp_u)
    var u     = Tensor[Int](vals, shp_u, st_u, 0)

    var shp_c = List[Int](); shp_c.append(len(cnts))
    var st_c  = compute_row_major_strides(shp_c)
    var c     = Tensor[Int](cnts, shp_c, st_c, 0)

    if return_counts:
        return UniqueResult[Int](u, c)
    else:
        # Return empty counts (shape=[0])
        var shp0 = List[Int](); shp0.append(0)
        var st0  = compute_row_major_strides(shp0)
        var empty_counts = Tensor[Int](List[Int](), shp0, st0, 0)
        return UniqueResult[Int](u, empty_counts)


@always_inline
fn unique(x: Tensor[Int]) -> UniqueResult[Int]:
    var ys = setops_copy_flat_list[Int](x)
    var n = len(ys)
    if n == 0:
        var shp1 = List[Int](); shp1.append(0)
        var shp2 = List[Int](); shp2.append(0)
        var u0 = Tensor[Int](List[Int](), shp1)
        var c0 = Tensor[Int](List[Int](), shp2)
        return UniqueResult[Int](u0, c0)

    insertion_sort_inplace(ys)

    # Run-length encode
    var vals = List[Int](); vals.reserve(n)
    var cnts = List[Int](); cnts.reserve(n)

    var cur = ys[0]
    var cnt = 1
    var i = 1
    while i < n:
        var v = ys[i]
        if v == cur:
            cnt = cnt + 1
        else:
            vals.append(cur); cnts.append(cnt)
            cur = v; cnt = 1
        i = i + 1
    vals.append(cur); cnts.append(cnt)

    var shp_u = List[Int](); shp_u.append(len(vals))
    var shp_c = List[Int](); shp_c.append(len(cnts))
    var u = Tensor[Int](vals, shp_u)
    var c = Tensor[Int](cnts, shp_c)
    return UniqueResult[Int](u, c)


# ========= helpers for Float64 / Float32 =========
@always_inline
fn _isnan64(x: Float64) -> Bool:
    return x != x

@always_inline
fn _isnan32(x: Float32) -> Bool:
    return x != x

# a < b with NaN sorted last
@always_inline
fn _lt_f64(a: Float64, b: Float64) -> Bool:
    if _isnan64(a):
        return False               # a is NaN -> not less than b (NaN goes last)
    if _isnan64(b):
        return True                # b is NaN, a is number -> a < b
    return a < b

@always_inline
fn _lt_f32(a: Float32, b: Float32) -> Bool:
    if _isnan32(a):
        return False
    if _isnan32(b):
        return True
    return a < b

# equality that treats NaNs as equal
@always_inline
fn _eq_f64(a: Float64, b: Float64) -> Bool:
    return (a == b) or (_isnan64(a) and _isnan64(b))

@always_inline
fn _eq_f32(a: Float32, b: Float32) -> Bool:
    return (a == b) or (_isnan32(a) and _isnan32(b))

 
 

# ================= Float64 =================

@always_inline
fn unique_float64(x: Tensor[Float64], return_counts: Bool) -> UniqueResult[Float64]:
    var ys = setops_copy_flat_list[Float64](x)
    var n = len(ys)

    if n == 0:
        var shp0 = List[Int](); shp0.append(0)
        var st0  = compute_row_major_strides(shp0)
        var u0 = Tensor[Float64](List[Float64](), shp0, st0, 0)
        var c0 = Tensor[Int](List[Int](), shp0, st0, 0)
        return UniqueResult[Float64](u0, c0)

    _insertion_sort_inplace_f64(ys)

    # Run-length encode (treat NaNs equal)
    var vals = List[Float64](); vals.reserve(n)
    var cnts = List[Int]();     cnts.reserve(n)

    var cur = ys[0]
    var cnt = 1
    var i = 1
    while i < n:
        var v = ys[i]
        if _eq_f64(v, cur):
            cnt = cnt + 1
        else:
            vals.append(cur); cnts.append(cnt)
            cur = v; cnt = 1
        i = i + 1
    vals.append(cur); cnts.append(cnt)

    var shp_u = List[Int](); shp_u.append(len(vals))
    var st_u  = compute_row_major_strides(shp_u)
    var u     = Tensor[Float64](vals, shp_u, st_u, 0)

    var shp_c = List[Int](); shp_c.append(len(cnts))
    var st_c  = compute_row_major_strides(shp_c)
    var c     = Tensor[Int](cnts, shp_c, st_c, 0)

    if return_counts:
        return UniqueResult[Float64](u, c)
    else:
        var shp0 = List[Int](); shp0.append(0)
        var st0  = compute_row_major_strides(shp0)
        var empty_counts = Tensor[Int](List[Int](), shp0, st0, 0)
        return UniqueResult[Float64](u, empty_counts)

@always_inline
fn unique(x: Tensor[Float64]) -> UniqueResult[Float64]:
    var ys = setops_copy_flat_list[Float64](x)
    var n = len(ys)
    if n == 0:
        var shp0 = List[Int](); shp0.append(0)
        var u0 = Tensor[Float64](List[Float64](), shp0)
        var c0 = Tensor[Int](List[Int](), shp0)
        return UniqueResult[Float64](u0, c0)

    _insertion_sort_inplace_f64(ys)

    var vals = List[Float64](); vals.reserve(n)
    var cnts = List[Int]();     cnts.reserve(n)

    var cur = ys[0]
    var cnt = 1
    var i = 1
    while i < n:
        var v = ys[i]
        if _eq_f64(v, cur):
            cnt = cnt + 1
        else:
            vals.append(cur); cnts.append(cnt)
            cur = v; cnt = 1
        i = i + 1
    vals.append(cur); cnts.append(cnt)

    var shp_u = List[Int](); shp_u.append(len(vals))
    var shp_c = List[Int](); shp_c.append(len(cnts))
    var u = Tensor[Float64](vals, shp_u)
    var c = Tensor[Int](cnts, shp_c)
    return UniqueResult[Float64](u, c)


# ================= Float32 =================

@always_inline
fn unique_float32(x: Tensor[Float32], return_counts: Bool) -> UniqueResult[Float32]:
    var ys = setops_copy_flat_list[Float32](x)
    var n = len(ys)

    if n == 0:
        var shp0 = List[Int](); shp0.append(0)
        var st0  = compute_row_major_strides(shp0)
        var u0 = Tensor[Float32](List[Float32](), shp0, st0, 0)
        var c0 = Tensor[Int](List[Int](), shp0, st0, 0)
        return UniqueResult[Float32](u0, c0)

    _insertion_sort_inplace_f32(ys)

    # Run-length encode (treat NaNs equal)
    var vals = List[Float32](); vals.reserve(n)
    var cnts = List[Int]();     cnts.reserve(n)

    var cur = ys[0]
    var cnt = 1
    var i = 1
    while i < n:
        var v = ys[i]
        if _eq_f32(v, cur):
            cnt = cnt + 1
        else:
            vals.append(cur); cnts.append(cnt)
            cur = v; cnt = 1
        i = i + 1
    vals.append(cur); cnts.append(cnt)

    var shp_u = List[Int](); shp_u.append(len(vals))
    var st_u  = compute_row_major_strides(shp_u)
    var u     = Tensor[Float32](vals, shp_u, st_u, 0)

    var shp_c = List[Int](); shp_c.append(len(cnts))
    var st_c  = compute_row_major_strides(shp_c)
    var c     = Tensor[Int](cnts, shp_c, st_c, 0)

    if return_counts:
        return UniqueResult[Float32](u, c)
    else:
        var shp0 = List[Int](); shp0.append(0)
        var st0  = compute_row_major_strides(shp0)
        var empty_counts = Tensor[Int](List[Int](), shp0, st0, 0)
        return UniqueResult[Float32](u, empty_counts)

@always_inline
fn unique(x: Tensor[Float32]) -> UniqueResult[Float32]:
    var ys = setops_copy_flat_list[Float32](x)
    var n = len(ys)
    if n == 0:
        var shp0 = List[Int](); shp0.append(0)
        var u0 = Tensor[Float32](List[Float32](), shp0)
        var c0 = Tensor[Int](List[Int](), shp0)
        return UniqueResult[Float32](u0, c0)

    _insertion_sort_inplace_f32(ys)

    var vals = List[Float32](); vals.reserve(n)
    var cnts = List[Int]();     cnts.reserve(n)

    var cur = ys[0]
    var cnt = 1
    var i = 1
    while i < n:
        var v = ys[i]
        if _eq_f32(v, cur):
            cnt = cnt + 1
        else:
            vals.append(cur); cnts.append(cnt)
            cur = v; cnt = 1
        i = i + 1
    vals.append(cur); cnts.append(cnt)

    var shp_u = List[Int](); shp_u.append(len(vals))
    var shp_c = List[Int](); shp_c.append(len(cnts))
    var u = Tensor[Float32](vals, shp_u)
    var c = Tensor[Int](cnts, shp_c)
    return UniqueResult[Float32](u, c)

# Count non-negative integers in x; output shape is [max(x)+1].
fn bincount_int(x: Tensor[Int]) -> Tensor[Int]:
    # Flatten (stride/offset aware)
    var ys = setops_copy_flat_list[Int](x)
    var n = len(ys)

    # Empty input -> empty histogram with shape [0]
    if n == 0:
        var data0 = List[Int]()                 # []
        var shp0  = List[Int]()                 # [0]
        shp0.append(0)
        var str0  = compute_row_major_strides(shp0)
        return Tensor[Int](data0, shp0, str0, 0)

    # Find max (if all negative -> mx becomes 0 below)
    var mx = ys[0]
    var i = 1
    while i < n:
        var v = ys[i]
        if v > mx:
            mx = v
        i += 1
    if mx < 0:
        mx = 0

    # Allocate histogram of length m = mx + 1
    var m = mx + 1
    var h = List[Int]()
    h.reserve(m)
    i = 0
    while i < m:
        h.append(0)
        i += 1

    # Accumulate counts for non-negative values within range
    i = 0
    while i < n:
        var vv = ys[i]
        if vv >= 0 and vv < m:
            h[vv] = h[vv] + 1
        i += 1

    # Build Tensor with explicit strides and offset
    var shp = List[Int]()
    shp.append(m)
    var strd = compute_row_major_strides(shp)
    return Tensor[Int](h, shp, strd, 0)

 

# Histogram with explicit bin edges for Int tensors.
# bins = [e0, e1, ..., eN], semi-open intervals: [e_k, e_{k+1})
fn histogram_int(x: Tensor[Int], bins: List[Int]) -> UniqueResult[Int]:
    var nb = len(bins)
    if nb < 2:
        # Return two distinct empty tensors (shape [0]) with explicit strides/offset
        var shp_e0 = List[Int](); shp_e0.append(0)
        var str_e0 = compute_row_major_strides(shp_e0)
        var edges0 = Tensor[Int](List[Int](), shp_e0, str_e0, 0)

        var shp_h0 = List[Int](); shp_h0.append(0)
        var str_h0 = compute_row_major_strides(shp_h0)
        var hist0  = Tensor[Int](List[Int](), shp_h0, str_h0, 0)

        return UniqueResult[Int](edges0, hist0)

    # Build counts with nb-1 zeros
    var counts = List[Int]()
    counts.reserve(nb - 1)
    var i = 0
    while i < nb - 1:
        counts.append(0)
        i += 1

    # Flatten x (stride/offset aware) into ys
    var ys = setops_copy_flat_list[Int](x)
    var n = len(ys)

    # Accumulate counts: for each v, find k s.t. bins[k] <= v < bins[k+1]
    var j = 0
    while j < n:
        var v = ys[j]
        var k = 0
        while k < nb - 1:
            if v >= bins[k] and v < bins[k + 1]:
                counts[k] = counts[k] + 1
                break
            k += 1
        j += 1

    # Pack histogram tensor
    var shp_c = List[Int](); shp_c.append(nb - 1)
    var str_c = compute_row_major_strides(shp_c)
    var hist = Tensor[Int](counts, shp_c, str_c, 0)

    # Pack edges tensor
    var shp_e = List[Int](); shp_e.append(nb)
    var str_e = compute_row_major_strides(shp_e)
    var edges = Tensor[Int](bins, shp_e, str_e, 0)

    # Return both (values = edges, counts = hist)
    return UniqueResult[Int](edges, hist)


# Digitize with right-closed bins:
# (-inf, e0], (e0, e1], ..., (e_{ne-2}, e_{ne-1}], (e_{ne-1}, +inf)
fn digitize_int(x: Tensor[Int], edges: List[Int]) -> Tensor[Int]:
    # Flatten x stride/offset-aware
    var ys = setops_copy_flat_list[Int](x)
    var n = len(ys)
    var ne = len(edges)

    var out = List[Int]()
    out.reserve(n)

    # Fast path: no edges -> everything maps to bin 0 (only interval is (-inf, +inf))
    if ne == 0:
        var i = 0
        while i < n:
            out.append(0)
            i += 1

        var shp0 = List[Int](); shp0.append(n)
        var st0  = compute_row_major_strides(shp0)
        return Tensor[Int](out, shp0, st0, 0)

    # Check if edges are non-decreasing to allow binary search
    var is_sorted = True
    var t = 1
    while t < ne:
        if edges[t] < edges[t - 1]:
            is_sorted = False
            break
        t += 1

    if is_sorted:
        # lower_bound: first index k in [0..ne] with edges[k] >= v
        var i = 0
        while i < n:
            var v = ys[i]
            var lo = 0
            var hi = ne
            # classic lower_bound loop
            while lo < hi:
                var mid = (lo + hi) // 2
                if edges[mid] < v:
                    lo = mid + 1
                else:
                    hi = mid
            # lo is k in [0..ne]; handles v > edges[ne-1] -> lo == ne
            out.append(lo)
            i += 1
    else:
        # Fallback: linear scan to find smallest k with v <= edges[k]
        var i = 0
        while i < n:
            var v = ys[i]
            var k = 0
            while k < ne and v > edges[k]:
                k += 1
            out.append(k)  # in [0..ne]
            i += 1

    # Build output tensor explicitly (no ctor ambiguity)
    var shp = List[Int](); shp.append(n)
    var st  = compute_row_major_strides(shp)
    return Tensor[Int](out, shp, st, 0)



# ---------------- thin object-style wrappers ----------------

# x.sort()
fn tensor_sort_f64(x: Tensor[Float64]) -> Tensor[Float64]:
    return sort_f64(x)

fn tensor_sort_f32(x: Tensor[Float32]) -> Tensor[Float32]:
    return sort_f32(x)

fn tensor_sort_int(x: Tensor[Int]) -> Tensor[Int]:
    return sort_int(x)

# x.unique(return_counts=True/False)
fn tensor_unique_int(x: Tensor[Int], return_counts: Bool = False) -> UniqueResult[Int]:
    return unique_int(x, return_counts)

fn tensor_unique_f64(x: Tensor[Float64], return_counts: Bool = False) -> UniqueResult[Float64]:
    return unique_float64(x, return_counts)

fn tensor_unique_f32(x: Tensor[Float32], return_counts: Bool = False) -> UniqueResult[Float32]:
    return unique_float32(x, return_counts)

# x.bincount()
fn tensor_bincount_int(x: Tensor[Int]) -> Tensor[Int]:
    return bincount_int(x)

# x.histogram(bins=[...])
fn tensor_histogram_int(x: Tensor[Int], bins: List[Int]) -> UniqueResult[Int]:
    return histogram_int(x, bins)

# x.digitize([edges...])
fn tensor_digitize_int(x: Tensor[Int], edges: List[Int]) -> Tensor[Int]:
    return digitize_int(x, edges)

# x.len()  -> number of elements
fn tensor_len[T](x: Tensor[T]) -> Int:
    return numel(x._shape)

# counts.sum() for Int tensors -> Int
 
fn tensor_sum_float(x: Tensor[Int]) -> Int:
    var n = len(x._data)
    var i = 0
    var s: Int = 0
    var lim = (n // 16) * 16
    while i < lim:
        s = s + x._data[i    ] + x._data[i + 1 ] + x._data[i + 2 ] + x._data[i + 3 ]
        s = s + x._data[i + 4] + x._data[i + 5 ] + x._data[i + 6 ] + x._data[i + 7 ]
        s = s + x._data[i + 8] + x._data[i + 9 ] + x._data[i + 10] + x._data[i + 11]
        s = s + x._data[i + 12] + x._data[i + 13] + x._data[i + 14] + x._data[i + 15]
        i = i + 16
    while i < n:
        s = s + x._data[i]
        i = i + 1
    return s 

fn tensor_sum_float(x: Tensor[Float64]) -> Float64:
    var n = len(x._data)
    var i = 0
    var s: Float64 = 0.0
    var lim = (n // 16) * 16
    while i < lim:
        s = s + x._data[i    ] + x._data[i + 1 ] + x._data[i + 2 ] + x._data[i + 3 ]
        s = s + x._data[i + 4] + x._data[i + 5 ] + x._data[i + 6 ] + x._data[i + 7 ]
        s = s + x._data[i + 8] + x._data[i + 9 ] + x._data[i + 10] + x._data[i + 11]
        s = s + x._data[i + 12] + x._data[i + 13] + x._data[i + 14] + x._data[i + 15]
        i = i + 16
    while i < n:
        s = s + x._data[i]
        i = i + 1
    return s

fn tensor_sum_int(x: Tensor[Int]) -> Int:
    var n = len(x._data)
    var i = 0
    var s = 0
    var lim = (n // 16) * 16
    while i < lim:
        s = s + x._data[i    ] + x._data[i + 1 ] + x._data[i + 2 ] + x._data[i + 3 ]
        s = s + x._data[i + 4] + x._data[i + 5 ] + x._data[i + 6 ] + x._data[i + 7 ]
        s = s + x._data[i + 8] + x._data[i + 9 ] + x._data[i + 10] + x._data[i + 11]
        s = s + x._data[i + 12] + x._data[i + 13] + x._data[i + 14] + x._data[i + 15]
        i += 16
    while i < n:
        s = s + x._data[i]
        i += 1
    return s
                
struct UniqueResult[T: ImplicitlyCopyable & Copyable & Movable]:
    var values: Tensor[T]
    var counts: Tensor[Int]

    fn __init__(out self, values: Tensor[T], counts: Tensor[Int]):
        self.values = values.copy()
        self.counts = counts.copy()

@always_inline
fn wrap_axis_index(i0: Int, dim: Int) -> Int:
    var i = i0
    if i < 0: i = i + dim
    return clamp_int(i, 0, dim - 1)

@always_inline
fn ceil_div_pos(a: Int, b: Int) -> Int:
    # a>=0, b>0
    return (a + b - 1) // b

@always_inline
fn axis_len_from_slice(start: Int, stop: Int, step: Int) -> Int:
    var s = step
    if s == 0: s = 1
    var out: Int
    if s > 0:
        if stop <= start: out = 0
        else:
            var diff = stop - start
            out = (diff + s - 1) // s
    else:
        if stop >= start: out = 0
        else:
            var diff = start - stop
            var m = -s
            out = (diff + m - 1) // m 
    return out



@always_inline
fn copy_ints(xs: List[Int]) -> List[Int]:
    var out = List[Int]()
    var n = len(xs)
    out.reserve(n)
    var i = 0
    while i < n:
        out.append(xs[i])
        i += 1
    return out.copy()
 


@always_inline
fn _index_sel_str(sel: IndexSel) -> String:
    var s = String("[tag=") + String(sel.tag) + "] "
    if sel.tag == 0:
        s += "index i=" + String(sel.i)
    elif sel.tag == 1:
        s += "slice " + String(sel.start) + ":" + String(sel.stop) + ":" + String(sel.step)
    else:
        s += "fancy idxs=" + _list_str(sel.idxs)
    return s
@always_inline
fn _list_str(xs: List[Int]) -> String:
    var s = String("[")
    var i = 0
    var n = len(xs)
    while i < n:
        s = s + String(xs[i])
        if i + 1 < n: s = s + ", "
        i += 1
    s = s + "]"
    return s
# Shorthands used across indexing paths
# ---------------------------------------------------------------------------
@always_inline
fn make_index_sel(i: Int) -> IndexSel:
    var s = IndexSel.index(i) 
    return s.copy()

@always_inline
fn make_slice_sel(t: Tuple[Int, Int, Int]) -> IndexSel: 
    var s = IndexSel.slice(t[0], t[1], t[2]) 
    return s.copy()
@always_inline
fn make_slice_sel(start: Int, stop: Int, step: Int) -> IndexSel:
    var s = IndexSel.slice(start, stop, step) 
    return s.copy()

@always_inline
fn make_fancy_sel(js: List[Int]) -> IndexSel:
    return IndexSel.fancy(js)

@always_inline
fn make_slice(start: Int, stop: Int, step: Int) -> SliceSpec:
    return SliceSpec(start, stop, step)

@always_inline
fn full_axis_slice(dim: Int) -> SliceSpec:
    return SliceSpec(0, dim, 1)

# Type checks
@always_inline 
fn is_index(s: IndexSel) -> Bool: 
    return s.tag == 0
@always_inline 
fn is_slice(s: IndexSel) -> Bool: 
    return s.tag == 1
@always_inline 
fn is_fancy(s: IndexSel) -> Bool: 
    return s.tag == 2

# Accessors
@always_inline 
fn get_index(s: IndexSel) -> Int: 
    return s.i
@always_inline 
fn get_slice(s: IndexSel) -> Tuple[Int, Int, Int]: 
    return (s.start, s.stop, s.step)
@always_inline 
fn get_fancy_list(s: IndexSel) -> List[Int]: 
    return s.idxs.copy()

# Strides helper (assumes you already have compute_row_major_strides)
@always_inline
fn mk_strides(shape: List[Int]) -> List[Int]:
    return compute_row_major_strides(shape)

  
  


 



@always_inline
fn apply_bin_Int64(a: Tensor[Int64], b: Tensor[Int64], code: Int) -> Tensor[Int64]:
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok: return a.copy()
    var ast = strides_for_shape(ash)
    var bst = strides_for_shape(bsh)

    var r = len(out_shape)
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    var out = List[Int64]()
    out.reserve(n)

    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    while k < n:
        var ai = flat_index_bcast(idx, ash, ast)
        var bi = flat_index_bcast(idx, bsh, bst)
        var x = a._data[ai]
        var y = b._data[bi]

        if      code == 0: out.append(x + y)
        elif    code == 1: out.append(x - y)
        elif    code == 2: out.append(x * y)
        elif    code == 3: out.append(0 if y == 0 else x // y)
        elif    code == 4: out.append(0 if y == 0 else x %  y)
        elif    code == 5: out.append(x & y)
        elif    code == 6: out.append(x | y)
        else:               out.append(x ^ y)

        index_advance(idx, out_shape)
        k += 1

    var st = compute_row_major_strides(out_shape)
    return Tensor[Int64](out, out_shape, st, 0)

@always_inline
fn apply_cmp_Int64(a: Tensor[Int64], b: Tensor[Int64], code: Int) -> Tensor[Int]:
    # Int mask 0/1
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok:
        var shp0 = List[Int](); shp0.append(0)
        var st0  = compute_row_major_strides(shp0)
        return Tensor[Int](List[Int](), shp0, st0, 0)

    var ast = strides_for_shape(ash)
    var bst = strides_for_shape(bsh)

    var r = len(out_shape)
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    var out = List[Int]()
    out.reserve(n)

    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    while k < n:
        var ai = flat_index_bcast(idx, ash, ast)
        var bi = flat_index_bcast(idx, bsh, bst)
        var x = a._data[ai]
        var y = b._data[bi]

        if     code == 0: out.append(1 if x == y else 0)
        elif   code == 1: out.append(1 if x != y else 0)
        elif   code == 2: out.append(1 if x <  y else 0)
        elif   code == 3: out.append(1 if x <= y else 0)
        elif   code == 4: out.append(1 if x >  y else 0)
        else:             out.append(1 if x >= y else 0)

        index_advance(idx, out_shape)
        k += 1

    var st = compute_row_major_strides(out_shape)
    return Tensor[Int](out, out_shape, st, 0)

@always_inline 
fn scalar_int64(s: Int64) -> Tensor[Int64]:
    var shp = List[Int](); shp.append(1)
    var st  = compute_row_major_strides(shp)
    return Tensor[Int64](List[Int64]([s]), shp, st, 0)

@always_inline
fn _apply_bin_bool(a: Tensor[Bool], b: Tensor[Bool], code: Int) -> Tensor[Bool]:
    # code: 5:and 6:or 7:xor ; for +,-,*,/,%, return False (defensive)
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok: return a.copy()

    var ast = strides_for_shape(ash)
    var bst = strides_for_shape(bsh)

    var r = len(out_shape)
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    var out = List[Bool]()
    out.reserve(n)

    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    while k < n:
        var ai = flat_index_bcast(idx, ash, ast)
        var bi = flat_index_bcast(idx, bsh, bst)
        var x = a._data[ai]
        var y = b._data[bi]

        if      code == 5: out.append(x and y)
        elif    code == 6: out.append(x or y)
        else:              out.append((x and not y) or (not x and y))  # xor

        index_advance(idx, out_shape)
        k += 1

    var st = compute_row_major_strides(out_shape)
    return Tensor[Bool](out, out_shape, st, 0)

@always_inline 
fn _scalar_bool(s: Bool) -> Tensor[Bool]:
    var shp = List[Int](); shp.append(1)
    var st  = compute_row_major_strides(shp)
    return Tensor[Bool](List[Bool]([s]), shp, st, 0)

@always_inline
fn _apply_bin_UInt8(a: Tensor[UInt8], b: Tensor[UInt8], code: Int) -> Tensor[UInt8]:
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok: return a.copy()

    var ast = strides_for_shape(ash)
    var bst = strides_for_shape(bsh)

    var r = len(out_shape)
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    var out = List[UInt8]()
    out.reserve(n)

    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    while k < n:
        var ai = flat_index_bcast(idx, ash, ast)
        var bi = flat_index_bcast(idx, bsh, bst)
        var x = a._data[ai]
        var y = b._data[bi]

        if      code == 0: out.append(x + y)
        elif    code == 1: out.append(x - y)
        elif    code == 2: out.append(x * y)
        elif    code == 3: out.append(0 if y == 0 else x // y)
        elif    code == 4: out.append(0 if y == 0 else x %  y)
        elif    code == 5: out.append(x & y)
        elif    code == 6: out.append(x | y)
        else:               out.append(x ^ y)

        index_advance(idx, out_shape)
        k += 1

    var st = compute_row_major_strides(out_shape)
    return Tensor[UInt8](out, out_shape, st, 0)

@always_inline
fn apply_cmp_UInt8(a: Tensor[UInt8], b: Tensor[UInt8], code: Int) -> Tensor[Int]:
    # Int mask 0/1
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok:
        var shp0 = List[Int](); shp0.append(0)
        var st0  = compute_row_major_strides(shp0)
        return Tensor[Int](List[Int](), shp0, st0, 0)

    var ast = strides_for_shape(ash)
    var bst = strides_for_shape(bsh)

    var r = len(out_shape)
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    var out = List[Int]()
    out.reserve(n)

    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    while k < n:
        var ai = flat_index_bcast(idx, ash, ast)
        var bi = flat_index_bcast(idx, bsh, bst)
        var x = a._data[ai]
        var y = b._data[bi]

        if     code == 0: out.append(1 if x == y else 0)
        elif   code == 1: out.append(1 if x != y else 0)
        elif   code == 2: out.append(1 if x <  y else 0)
        elif   code == 3: out.append(1 if x <= y else 0)
        elif   code == 4: out.append(1 if x >  y else 0)
        else:             out.append(1 if x >= y else 0)

        index_advance(idx, out_shape)
        k += 1

    var st = compute_row_major_strides(out_shape)
    return Tensor[Int](out, out_shape, st, 0)

@always_inline 
fn scalar_UInt8(s: UInt8) -> Tensor[UInt8]:
    var shp = List[Int](); shp.append(1)
    var st  = compute_row_major_strides(shp)
    return Tensor[UInt8](List[UInt8]([s]), shp, st, 0)

@always_inline
fn apply_bin_Int32(a: Tensor[Int32], b: Tensor[Int32], code: Int) -> Tensor[Int32]:
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok: return a.copy()

    var ast = strides_for_shape(ash)
    var bst = strides_for_shape(bsh)

    var r = len(out_shape)
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    var out = List[Int32]()
    out.reserve(n)

    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    while k < n:
        var ai = flat_index_bcast(idx, ash, ast)
        var bi = flat_index_bcast(idx, bsh, bst)
        var x = a._data[ai]
        var y = b._data[bi]

        if      code == 0: out.append(x + y)
        elif    code == 1: out.append(x - y)
        elif    code == 2: out.append(x * y)
        elif    code == 3: out.append(0 if y == 0 else x // y)
        elif    code == 4: out.append(0 if y == 0 else x %  y)
        elif    code == 5: out.append(x & y)
        elif    code == 6: out.append(x | y)
        else:               out.append(x ^ y)

        index_advance(idx, out_shape)
        k += 1

    var st = compute_row_major_strides(out_shape)
    return Tensor[Int32](out, out_shape, st, 0)

@always_inline
fn apply_cmp_Int32(a: Tensor[Int32], b: Tensor[Int32], code: Int) -> Tensor[Int]:
    # Int mask 0/1
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok:
        var shp0 = List[Int](); shp0.append(0)
        var st0  = compute_row_major_strides(shp0)
        return Tensor[Int](List[Int](), shp0, st0, 0)

    var ast = strides_for_shape(ash)
    var bst = strides_for_shape(bsh)

    var r = len(out_shape)
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    var out = List[Int]()
    out.reserve(n)

    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    while k < n:
        var ai = flat_index_bcast(idx, ash, ast)
        var bi = flat_index_bcast(idx, bsh, bst)
        var x = a._data[ai]
        var y = b._data[bi]

        if     code == 0: out.append(1 if x == y else 0)
        elif   code == 1: out.append(1 if x != y else 0)
        elif   code == 2: out.append(1 if x <  y else 0)
        elif   code == 3: out.append(1 if x <= y else 0)
        elif   code == 4: out.append(1 if x >  y else 0)
        else:             out.append(1 if x >= y else 0)

        index_advance(idx, out_shape)
        k += 1

    var st = compute_row_major_strides(out_shape)
    return Tensor[Int](out, out_shape, st, 0)

@always_inline 
fn scalar_int32(s: Int32) -> Tensor[Int32]:
    var shp = List[Int](); shp.append(1)
    var st  = compute_row_major_strides(shp)
    return Tensor[Int32](List[Int32]([s]), shp, st, 0)

@always_inline
fn _apply_bin_Int(a: Tensor[Int], b: Tensor[Int], code: Int) -> Tensor[Int]:
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok: return a.copy()

    var ast = strides_for_shape(ash)
    var bst = strides_for_shape(bsh)

    var r = len(out_shape)
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    var out = List[Int]()
    out.reserve(n)

    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    while k < n:
        var ai = flat_index_bcast(idx, ash, ast)
        var bi = flat_index_bcast(idx, bsh, bst)
        var x = a._data[ai]
        var y = b._data[bi]

        if      code == 0: out.append(x + y)
        elif    code == 1: out.append(x - y)
        elif    code == 2: out.append(x * y)
        elif    code == 3: out.append(0 if y == 0 else x // y)
        elif    code == 4: out.append(0 if y == 0 else x %  y)
        elif    code == 5: out.append(x & y)
        elif    code == 6: out.append(x | y)
        else:               out.append(x ^ y)

        index_advance(idx, out_shape)
        k += 1

    var st = compute_row_major_strides(out_shape)
    return Tensor[Int](out, out_shape, st, 0)

@always_inline
fn apply_cmp_Int(a: Tensor[Int], b: Tensor[Int], code: Int) -> Tensor[Int]:
    # Int mask 0/1
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok:
        var shp0 = List[Int](); shp0.append(0)
        var st0  = compute_row_major_strides(shp0)
        return Tensor[Int](List[Int](), shp0, st0, 0)

    var ast = strides_for_shape(ash)
    var bst = strides_for_shape(bsh)

    var r = len(out_shape)
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    var out = List[Int]()
    out.reserve(n)

    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    while k < n:
        var ai = flat_index_bcast(idx, ash, ast)
        var bi = flat_index_bcast(idx, bsh, bst)
        var x = a._data[ai]
        var y = b._data[bi]

        if     code == 0: out.append(1 if x == y else 0)
        elif   code == 1: out.append(1 if x != y else 0)
        elif   code == 2: out.append(1 if x <  y else 0)
        elif   code == 3: out.append(1 if x <= y else 0)
        elif   code == 4: out.append(1 if x >  y else 0)
        else:             out.append(1 if x >= y else 0)

        index_advance(idx, out_shape)
        k += 1

    var st = compute_row_major_strides(out_shape)
    return Tensor[Int](out, out_shape, st, 0)




@always_inline
fn apply_bin_f32(a: Tensor[Float32], b: Tensor[Float32], code: Int) -> Tensor[Float32]:
    # Broadcast planning
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok:
        return a.copy()

    var ast = strides_for_shape(ash)
    var bst = strides_for_shape(bsh)

    # total elements
    var r = len(out_shape)
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    # Fast path: exact same shape as output and contiguous row-major
    var out_rm = compute_row_major_strides(out_shape)
    var a_is_fast = (ash == out_shape) and (a._strides == out_rm) and (a._offset == 0)
    var b_is_fast = (bsh == out_shape) and (b._strides == out_rm) and (b._offset == 0)

    var out = List[Float32]()
    out.reserve(n)

    if a_is_fast and b_is_fast:
        # No broadcasting, contiguous: single linear loop
        var k = 0
        if code == 0:
            while k < n: out.append(a._data[k] + b._data[k]); k += 1
        elif code == 1:
            while k < n: out.append(a._data[k] - b._data[k]); k += 1
        elif code == 2:
            while k < n: out.append(a._data[k] * b._data[k]); k += 1
        elif code == 3:
            while k < n:
                var y = b._data[k]
                out.append(0.0 if y == 0.0 else a._data[k] / y)
                k += 1
        elif code == 4:
            # x % y with div-by-zero -> 0.0
            while k < n:
                var x = a._data[k]
                var y = b._data[k]
                if y == 0.0:
                    out.append(0.0)
                else:
                    var q = Int32(x / y)
                    out.append(x - Float32(q) * y)
                k += 1
        elif code == 5:
            # logical AND (nonzero→1.0, zero→0.0)
            while k < n:
                out.append(1.0 if (a._data[k] != 0.0 and b._data[k] != 0.0) else 0.0)
                k += 1
        elif code == 6:
            # logical OR
            while k < n:
                out.append(1.0 if (a._data[k] != 0.0 or b._data[k] != 0.0) else 0.0)
                k += 1
        else:
            # logical XOR
            while k < n:
                var ax = a._data[k] != 0.0
                var by = b._data[k] != 0.0
                out.append(1.0 if (ax != by) else 0.0)
                k += 1

        var st = compute_row_major_strides(out_shape)
        return Tensor[Float32](out, out_shape, st, 0)

    # General path: broadcasting via index vector
    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    if code == 0:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            out.append(a._data[ai] + b._data[bi])
            index_advance(idx, out_shape); k += 1
    elif code == 1:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            out.append(a._data[ai] - b._data[bi])
            index_advance(idx, out_shape); k += 1
    elif code == 2:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            out.append(a._data[ai] * b._data[bi])
            index_advance(idx, out_shape); k += 1
    elif code == 3:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            var x = a._data[ai]; var y = b._data[bi]
            out.append(0.0 if y == 0.0 else x / y)
            index_advance(idx, out_shape); k += 1
    elif code == 4:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            var x = a._data[ai]; var y = b._data[bi]
            if y == 0.0:
                out.append(0.0)
            else:
                var q = Int32(x / y)
                out.append(x - Float32(q) * y)
            index_advance(idx, out_shape); k += 1
    elif code == 5:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            out.append(1.0 if (a._data[ai] != 0.0 and b._data[bi] != 0.0) else 0.0)
            index_advance(idx, out_shape); k += 1
    elif code == 6:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            out.append(1.0 if (a._data[ai] != 0.0 or b._data[bi] != 0.0) else 0.0)
            index_advance(idx, out_shape); k += 1
    else:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            var ax = a._data[ai] != 0.0
            var by = b._data[bi] != 0.0
            out.append(1.0 if (ax != by) else 0.0)
            index_advance(idx, out_shape); k += 1

    var stg = compute_row_major_strides(out_shape)
    return Tensor[Float32](out, out_shape, stg, 0)


@always_inline
fn apply_cmp_f32(a: Tensor[Float32], b: Tensor[Float32], code: Int) -> Tensor[Float32]:
    # Broadcast planning
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok:
        return a.copy()

    var ast = strides_for_shape(ash)
    var bst = strides_for_shape(bsh)

    # total elements
    var r = len(out_shape)
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    # Fast path: exact match + contiguous
    var out_rm = compute_row_major_strides(out_shape)
    var a_is_fast = (ash == out_shape) and (a._strides == out_rm) and (a._offset == 0)
    var b_is_fast = (bsh == out_shape) and (b._strides == out_rm) and (b._offset == 0)

    var out = List[Float32]()
    out.reserve(n)

    if a_is_fast and b_is_fast:
        var k = 0
        if code == 0:
            while k < n: out.append(1.0 if a._data[k] == b._data[k] else 0.0); k += 1
        elif code == 1:
            while k < n: out.append(1.0 if a._data[k] != b._data[k] else 0.0); k += 1
        elif code == 2:
            while k < n: out.append(1.0 if a._data[k] <  b._data[k] else 0.0); k += 1
        elif code == 3:
            while k < n: out.append(1.0 if a._data[k] <= b._data[k] else 0.0); k += 1
        elif code == 4:
            while k < n: out.append(1.0 if a._data[k] >  b._data[k] else 0.0); k += 1
        else:
            while k < n: out.append(1.0 if a._data[k] >= b._data[k] else 0.0); k += 1

        var st = compute_row_major_strides(out_shape)
        return Tensor[Float32](out, out_shape, st, 0)

    # General path
    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    if code == 0:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            out.append(1.0 if a._data[ai] == b._data[bi] else 0.0)
            index_advance(idx, out_shape); k += 1
    elif code == 1:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            out.append(1.0 if a._data[ai] != b._data[bi] else 0.0)
            index_advance(idx, out_shape); k += 1
    elif code == 2:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            out.append(1.0 if a._data[ai] <  b._data[bi] else 0.0)
            index_advance(idx, out_shape); k += 1
    elif code == 3:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            out.append(1.0 if a._data[ai] <= b._data[bi] else 0.0)
            index_advance(idx, out_shape); k += 1
    elif code == 4:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            out.append(1.0 if a._data[ai] >  b._data[bi] else 0.0)
            index_advance(idx, out_shape); k += 1
    else:
        while k < n:
            var ai = a._offset + flat_index_bcast(idx, ash, ast)
            var bi = b._offset + flat_index_bcast(idx, bsh, bst)
            out.append(1.0 if a._data[ai] >= b._data[bi] else 0.0)
            index_advance(idx, out_shape); k += 1

    var stg = compute_row_major_strides(out_shape)
    return Tensor[Float32](out, out_shape, stg, 0)


 
@always_inline
fn _rank(sh: List[Int]) -> Int:
    return len(sh)

@always_inline
fn make_strides_row_major(sh: List[Int]) -> List[Int]:
    var n = len(sh)
    var st = List[Int]()
    st.reserve(n)
    if n == 0:
        return st
    var i = 0
    while i < n:
        st.append(0)  # placeholder
        i = i + 1
    # last stride = 1
    st[n - 1] = 1
    var k = n - 2
    while k >= 0:
        st[k] = st[k + 1] * (sh[k + 1] if sh[k + 1] > 0 else 1)
        k = k - 1
    return st





@always_inline
fn index_advance(idx: List[Int], sh: List[Int]) -> None:
    # idx is same rank as sh
    var r = len(sh)
    var k = r - 1
    while k >= 0:
        idx[k] = idx[k] + 1
        if idx[k] < sh[k]:
            return
        idx[k] = 0
        k = k - 1

@always_inline
fn flat_index(idx: List[Int], sh: List[Int]) -> Int:
    # row-major flat index
    var r = len(sh)
    if r == 0:
        return 0
    var st = make_strides_row_major(sh)
    var s = 0
    var i = 0
    while i < r:
        s = s + idx[i] * st[i]
        i = i + 1
    return s

@always_inline
fn flat_index_bcast(idx: List[Int], sh: List[Int], st: List[Int]) -> Int:
    # like flat_index, but for broadcasting dims (dim size 1 => idx contribution 0)
    var r = len(sh)
    var s = 0
    var i = 0
    while i < r:
        var ii = idx[i] if sh[i] != 1 else 0
        s = s + ii * st[i]
        i = i + 1
    return s

# Build strides for possibly-padded shapes
@always_inline
fn strides_for_shape(sh: List[Int]) -> List[Int]:
    return make_strides_row_major(sh)

# ============ Typed kernels (Float64 / Float32 / Int / Int32 / Int64 / UInt8 / Bool) ============

# Macro-like generator via copy-paste patterns (Mojo doesn't have macros yet).

# ---- Float64 ----
@always_inline
fn apply_bin_f64(a: Tensor[Float64], b: Tensor[Float64], code: Int) -> Tensor[Float64]:
    # code: 0:+ 1:- 2:* 3:/ 4:% 5:AND(logical) 6:OR(logical) 7:XOR(logical)
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok:
        return a.copy()

    var r = len(out_shape)
    var ast = compute_row_major_strides(ash)
    var bst = compute_row_major_strides(bsh)

    # total elements
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    var out = List[Float64]()
    out.reserve(n)

    # multi-index for broadcasting
    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    while k < n:
        var ai = flat_index_bcast(idx, ash, ast)
        var bi = flat_index_bcast(idx, bsh, bst)
        var x = a._data[ai]
        var y = b._data[bi]

        if code == 0:
            out.append(x + y)
        elif code == 1:
            out.append(x - y)
        elif code == 2:
            out.append(x * y)
        elif code == 3:
            # safe divide
            out.append(0.0 if y == 0.0 else x / y)
        elif code == 4:
            # float "mod" via truncating quotient (fast)
            if y == 0.0:
                out.append(0.0)
            else:
                var q = Int64(x / y)
                out.append(x - Float64(q) * y)
        elif code == 5:
            # logical AND -> 1.0 if both non-zero else 0.0
            out.append(1.0 if (x != 0.0 and y != 0.0) else 0.0)
        elif code == 6:
            # logical OR
            out.append(1.0 if (x != 0.0 or  y != 0.0) else 0.0)
        else:
            # code == 7, logical XOR
            var ax = (x != 0.0)
            var by = (y != 0.0)
            out.append(1.0 if (ax != by) else 0.0)

        index_advance(idx, out_shape)
        k += 1

    # explicit (data, shape, strides, offset) -> no ctor ambiguity
    var st = compute_row_major_strides(out_shape)
    return Tensor[Float64](out, out_shape, st, 0)


@always_inline
fn apply_cmp_f64(a: Tensor[Float64], b: Tensor[Float64], code: Int) -> Tensor[Float64]:
    # code: 0:== 1:!= 2:< 3:<= 4:> 5:>= ; returns float mask (0.0/1.0)
    var ok, out_shape, ash, bsh = broadcast_shapes(a._shape, b._shape)
    if not ok:
        return a.copy()

    var r = len(out_shape)
    var ast = compute_row_major_strides(ash)
    var bst = compute_row_major_strides(bsh)

    # total elements
    var n = 1
    var i = 0
    while i < r:
        n = n * out_shape[i]
        i += 1

    var out = List[Float64]()
    out.reserve(n)

    var idx = List[Int]()
    idx.reserve(r)
    i = 0
    while i < r:
        idx.append(0)
        i += 1

    var k = 0
    while k < n:
        var ai = flat_index_bcast(idx, ash, ast)
        var bi = flat_index_bcast(idx, bsh, bst)
        var x = a._data[ai]
        var y = b._data[bi]

        if code == 0:
            out.append(1.0 if x == y else 0.0)
        elif code == 1:
            out.append(1.0 if x != y else 0.0)
        elif code == 2:
            out.append(1.0 if x <  y else 0.0)
        elif code == 3:
            out.append(1.0 if x <= y else 0.0)
        elif code == 4:
            out.append(1.0 if x >  y else 0.0)
        else:
            out.append(1.0 if x >= y else 0.0)

        index_advance(idx, out_shape)
        k += 1

    var st = compute_row_major_strides(out_shape)
    return Tensor[Float64](out, out_shape, st, 0)


 

# ---------------- small helpers ----------------
 
@staticmethod
@always_inline
fn _exp_scalar_(x: Float64) -> Float64:
    # Polynomial (Horner) approximation of exp(x) up to x^6/6!
    if x < -20.0:
        return 0.0
    if x > 20.0:
        return 485165195.4097903  # ~exp(20)

    var y = 0.001388888888888889    # 1/720
    y = 0.008333333333333333 + x * y  # 1/120 + x*(1/720)
    y = 0.041666666666666664 + x * y  # 1/24
    y = 0.16666666666666666  + x * y  # 1/6
    y = 0.5                   + x * y # 1/2
    y = 1.0                   + x * y
    y = 1.0                   + x * y
    return y


@always_inline
fn _exp_scalar_f64(x: Float64) -> Float64:
    # Horner polynomial approximation of exp(x) up to x^6/6!
    # Simple clamps to keep values reasonable.
    if x < -20.0:
        return 0.0
    if x > 20.0:
        return 485165195.4097903  # ~exp(20)

    var y = 0.001388888888888889    # 1/720
    y = 0.008333333333333333 + x * y  # 1/120 + x*(1/720)
    y = 0.041666666666666664 + x * y  # 1/24  + ...
    y = 0.16666666666666666  + x * y  # 1/6
    y = 0.5                   + x * y # 1/2
    y = 1.0                   + x * y
    y = 1.0                   + x * y
    return y

 
# Core: generic loops; element formatting is passed in as `fmt`
@always_inline
fn tensor_to_string_core[U: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[U],
    fmt: fn (U) -> String
) -> String:
    var s = String("[Tensor shape=")
    s = s + x._shape.__str__()
    s = s + ", n=" + numel_shape(x._shape).__str__()
    s = s + ", data="

    var ndim = len(x._shape)

    if ndim == 0:
        s = s + "[]]"
        return s

    if ndim == 1:
        var d0 = x._shape[0]
        s = s + "["
        var i = 0
        while i < d0:
            if i > 0: s = s + ", "
            s = s + fmt(x._data[i])
            i += 1
        s = s + "]]"
        return s

    if ndim == 2:
        var d0 = x._shape[0]
        var d1 = x._shape[1]
        s = s + "["
        var i = 0
        var base = 0
        while i < d0:
            if i > 0: s = s + ", "
            s = s + "["
            var j = 0
            while j < d1:
                if j > 0: s = s + ", "
                s = s + fmt(x._data[base + j])
                j += 1
            s = s + "]"
            base = base + d1
            i += 1
        s = s + "]]"
        return s

    if ndim == 3:
        var d0 = x._shape[0]
        var d1 = x._shape[1]
        var d2 = x._shape[2]
        s = s + "["
        var i = 0
        var base0 = 0
        while i < d0:
            if i > 0: s = s + ", "
            s = s + "["
            var j = 0
            var base1 = base0
            while j < d1:
                if j > 0: s = s + ", "
                s = s + "["
                var k = 0
                while k < d2:
                    if k > 0: s = s + ", "
                    s = s + fmt(x._data[base1 + k])
                    k += 1
                s = s + "]"
                base1 = base1 + d2
                j += 1
            s = s + "]"
            base0 = base0 + d1 * d2
            i += 1
        s = s + "]]"
        return s

    # ndim >= 4 → pretty-print first 4 dims (row-major)
    var d0 = x._shape[0]
    var d1 = x._shape[1]
    var d2 = x._shape[2]
    var d3 = x._shape[3]
    s = s + "["
    var i0 = 0
    var base0 = 0
    while i0 < d0:
        if i0 > 0: s = s + ", "
        s = s + "["
        var i1 = 0
        var base1 = base0
        while i1 < d1:
            if i1 > 0: s = s + ", "
            s = s + "["
            var i2 = 0
            var base2 = base1
            while i2 < d2:
                if i2 > 0: s = s + ", "
                s = s + "["
                var i3 = 0
                while i3 < d3:
                    if i3 > 0: s = s + ", "
                    s = s + fmt(x._data[base2 + i3])
                    i3 += 1
                s = s + "]"
                base2 = base2 + d3
                i2 += 1
            s = s + "]"
            base1 = base1 + d2 * d3
            i1 += 1
        s = s + "]"
        base0 = base0 + d1 * d2 * d3
        i0 += 1
    s = s + "]]"
    return s

# -------- element formatters for concrete primitive dtypes --------
 
 
 
 # -------- Helpers (internal) --------
@always_inline
fn full_axis(dim: Int) -> IndexSel:
    # ":" over the full axis: start=0, stop=dim, step=1
    return make_slice_sel((0, dim, 1))
 
 

 # ---------- Selector types & factories (keep once) ----------
# ---------- IndexSel tuple utilities ----------
# Assume: type IndexSel = Tuple[Bool, Int, (Int, Int, Int)]

 

 

@always_inline
fn sel_is_index(sel: IndexSel) -> Bool:
    var (is_idx, _, _) = sel
    return is_idx

@always_inline
fn sel_index(sel: IndexSel) -> Int:
    var (_, idx, _) = sel
    return idx

@always_inline
fn sel_slice(sel: IndexSel) -> (Int, Int, Int):
    var (_, _, sl) = sel
    return sl  # (start, stop, step)

# ---------- small helpers ----------
 

@always_inline
fn wrap_index(i: Int, dim: Int) -> Int:
    if dim <= 0:
        return 0
    var idx = i
    if idx < 0:
        idx = dim + idx
    return clamp_int(idx, 0, dim - 1)

@always_inline
fn range_len(start: Int, stop: Int, step_in: Int) -> Int:
    var step = step_in
    if step == 0:
        step = 1
    if step > 0:
        if start >= stop:
            return 0
        return (stop - start + step - 1) // step
    else:
        if start <= stop:
            return 0
        var m = -step
        return (start - stop + m - 1) // m

  
# Trim spaces (simple, ASCII)
fn trim_ascii(s: String) -> String:
    var n = len(s)
    if n == 0:
        return s
    var i = 0
    var j = n - 1
    # left
    while i < n:
        var c = s[i]
        if c == ' ' or c == '\t' or c == '\n' or c == '\r':
            i = i + 1
        else:
            break
    # right
    while j >= i:
        var c2 = s[j]
        if c2 == ' ' or c2 == '\t' or c2 == '\n' or c2 == '\r':
            j = j - 1
        else:
            break
    # slice [i..j]
    var out = String("")
    var k = i
    while k <= j:
        out = out + s[k]
        k = k + 1
    return out

# Convert a one-char String "0".."9" to (ok, value) without raising.
@always_inline
fn digit_from_one_char(ch: String) -> (Bool, Int):
    if ch == "0": return (True, 0)
    if ch == "1": return (True, 1)
    if ch == "2": return (True, 2)
    if ch == "3": return (True, 3)
    if ch == "4": return (True, 4)
    if ch == "5": return (True, 5)
    if ch == "6": return (True, 6)
    if ch == "7": return (True, 7)
    if ch == "8": return (True, 8)
    if ch == "9": return (True, 9)
    return (False, 0)

# Convert StringSlice to owned String (همون که قبلاً داشتی)
@always_inline
fn to_string_owned(x: StringSlice) -> String:
    return String(x)

# Safe, non-raising integer parser for decimal signed ints.
fn parse_int_safe(s: String) -> (Bool, Int):
    var n = len(s)
    if n == 0:
        return (False, 0)

    var i = 0
    var sign = 1
    var acc: Int = 0

    # optional sign
    var c0 = to_string_owned(s[0])
    if c0 == "-" or c0 == "+":
        if c0 == "-":
            sign = -1
        i = 1
        if i >= n:
            return (False, 0)

    var has_digit = False
    while i < n:
        # s[i] → StringSlice → to owned String (one char)
        var ch = to_string_owned(s[i])
        var (ok_d, dv) = digit_from_one_char(ch)
        if not ok_d:
            return (False, 0)
        has_digit = True
        acc = acc * 10 + dv
        i = i + 1

    if not has_digit:
        return (False, 0)

    return (True, sign * acc)


@always_inline
fn tensor1d_from_list[T: ImplicitlyCopyable & Copyable & Movable](data: List[T]) -> Tensor[T]:
    var shp = List[Int]()
    shp.append(len(data))
    var strides = compute_row_major_strides(shp)
    return Tensor[T](data, shp, strides, 0)



# ------------------------------
# Small local RNG (XorShift64*)
# ------------------------------
struct XorShift64:
    var state: UInt64

    @always_inline
    fn __init__(out self, seed: UInt64):
        # Zero-seed guard to avoid degenerate sequence
        var s = seed
        if s == UInt64(0):
            s = UInt64(0x9E3779B97F4A7C15)   # non-zero default
        self.state = s 

    @always_inline
    fn next_u64(mut self) -> UInt64:
        var x = self.state
        x ^= x >> UInt64(12)
        x ^= x << UInt64(25)
        x ^= x >> UInt64(27)
        self.state = x
        # xorshift* final mix (no '&*' in Mojo): wrap to 64 bits
        var MIX: UInt64 = 0x2545F4914F6CDD1D
        x = x * MIX
        x = x & UInt64(0xFFFFFFFFFFFFFFFF)
        return x

    @always_inline
    fn next_unit_f64(mut self) -> Float64:
        # Map top 53 bits to [0,1)
        var v = self.next_u64() >> UInt64(11)  # keep 53 bits
        return Float64(v) / 9007199254740992.0 # 2^53

# -----------------------------------------------
# Uniform helpers (out-of-place & in-place, f64)
# -----------------------------------------------

@always_inline
fn uniform(low: Float64, high: Float64, shape: List[Int], seed: Optional[Int] = None) -> Tensor[Float64]:
    var out = empty(shape)  # Float64 by default
    var s = UInt64(seed.value()) 
    if seed is None: s =UInt64(0xD1B54A32D192ED03)  
    var rng = XorShift64(s)

    var n = len(out._data)
    var i = 0
    var span = high - low
    while i < n:
        var u = rng.next_unit_f64()           # in [0,1)
        out._data[i] = low + span * u
        i += 1
    return out


@always_inline
fn uniform_f32(low: Float32, high: Float32, shape: List[Int], seed: Optional[Int] = None) -> Tensor[Float32]:
    var tmp = uniform(Float64(low), Float64(high), shape, seed)  # sample in f64
    return tmp.to_float32()
