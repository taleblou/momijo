# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.tensor.teansform
# File:         src/momijo/tensor/teansform.mojo
#
 
 

from collections.list import List

from momijo.tensor.tensor import Tensor

from momijo.tensor.math import floor
from momijo.tensor.indexing import ravel_index_with_offset

from momijo.tensor.helpers import (
    is_row_major_contiguous,
    clamp_nonneg,
    compute_new_shape,
    row_major_strides,
    numel,
    pairs_from_flat_pads,
    copy_list_int,
    copy_list_T,
    normalize_axis,
    is_contig_2d,
    clone_header_share_data, 
    copy_list,
    compute_row_major_strides,
)

from momijo.tensor.creation import empty_tensor_with,zero_scalar_of
from momijo.tensor.cast import *
# ============================================================================
#                          Boundary index maps for padding
# ============================================================================

@always_inline
fn reflect_index(i: Int, n: Int) -> Int:
    if n <= 1:
        return 0
    var period = 2 * n - 2
    var ii = i % period
    if ii < 0:
        ii = ii + period
    if ii < n:
        return ii
    return period - ii

@always_inline
fn symmetric_index(i: Int, n: Int) -> Int:
    if n <= 1:
        return 0
    var ii = i
    if ii < 0:
        ii = -ii - 1
    var period = 2 * n
    ii = ii % period
    if ii < 0:
        ii = ii + period
    if ii < n:
        return ii
    return 2 * n - 1 - ii

@always_inline
fn replicate_index(i: Int, n: Int) -> Int:
    if n <= 0:
        return 0
    if i < 0:
        return 0
    if i >= n:
        return n - 1
    return i

@always_inline
fn circular_index(i: Int, n: Int) -> Int:
    if n <= 0:
        return 0
    var ii = i % n
    if ii < 0:
        ii = ii + n
    return ii

@always_inline
fn pad_map_index(i: Int, n: Int, mode: Int) -> Int:
    # mode: 0=replicate, 1=reflect, 2=symmetric, 3=circular
    if mode == 0:
        return replicate_index(i, n)
    if mode == 1:
        return reflect_index(i, n)
    if mode == 2:
        return symmetric_index(i, n)
    return circular_index(i, n)

# ----------------------------------------------------------------------------
# Small helpers for fast paths
# ----------------------------------------------------------------------------

@always_inline
fn is_row_major_full(shape: List[Int], strides: List[Int]) -> Bool:
    return is_row_major_contiguous(shape, strides)

@always_inline
fn is_contig2d_rowmajor(strides: List[Int], w: Int) -> Bool:
    if len(strides) < 2:
        return False
    return (strides[len(strides) - 1] == 1) and (strides[len(strides) - 2] == w)

@always_inline
fn is_contig3d_lastdim(shape: List[Int], strides: List[Int]) -> Bool:
    if len(shape) < 3 or len(strides) < 3:
        return False
    var d2 = shape[len(shape) - 1]     # W
    var s2 = strides[len(strides) - 1] # stride(W)
    var s1 = strides[len(strides) - 2] # stride(H)
    return (s2 == 1) and (s1 == d2)

@always_inline
fn is_contig4d_spatial_last(shape: List[Int], strides: List[Int]) -> Bool:
    # For ... H, W tail: require stride(W)=1 and stride(H)=W
    if len(shape) < 4 or len(strides) < 4:
        return False
    var W  = shape[len(shape) - 1]
    var sW = strides[len(strides) - 1]
    var sH = strides[len(strides) - 2]
    return (sW == 1) and (sH == W)

@always_inline
fn _prod(xs: List[Int]) -> Int:
    var p = 1
    var i = 0
    var n = len(xs)
    while i < n:
        p = p * xs[i]
        i += 1
    return p

@always_inline
fn _spatial_axes(shape: List[Int]) -> (Int, Int):
    var r = len(shape)
    var h_ax = r - 2 if r >= 2 else 0
    var w_ax = r - 1 if r >= 1 else 0
    return (h_ax, w_ax)

# ============================================================================
#                       Generic nD constant pad
# ============================================================================

fn pad_nd[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    pad_pairs: List[(Int, Int)],
    constant: T
) -> Tensor[T]:
    var rank = len(x._shape)
    if len(pad_pairs) != rank:
        return x.copy()

    var no_op = True
    var dchk = 0
    while dchk < rank:
        if clamp_nonneg(pad_pairs[dchk][0]) != 0 or clamp_nonneg(pad_pairs[dchk][1]) != 0:
            no_op = False
            break
        dchk += 1
    if no_op:
        return x.copy()

    var out_shape = compute_new_shape(x._shape, pad_pairs)
    var out = Tensor[T](out_shape, constant)
    if rank == 0:
        return out.copy()

    var out_strides = row_major_strides(out_shape)
    var in_strides  = x._strides.copy()
    var idx = List[Int]()
    idx.reserve(rank)
    var r0 = 0
    while r0 < rank:
        idx.append(0)
        r0 += 1

    var done = False
    var lin_in = 0
    var last = rank - 1

    while not done:
        var out_lin = 0
        var d = 0
        while d < rank:
            var out_coord = idx[d] + clamp_nonneg(pad_pairs[d][0])
            out_lin = out_lin + out_coord * out_strides[d]
            d += 1
        out._data[out_lin] = x._data[lin_in]

        var ax = last
        while True:
            if idx[ax] + 1 < x._shape[ax]:
                idx[ax] = idx[ax] + 1
                lin_in  = lin_in + in_strides[ax]
                break
            var span = x._shape[ax] - 1
            lin_in = lin_in - span * in_strides[ax]
            idx[ax] = 0
            ax = ax - 1
            if ax < 0:
                done = True
                break

    return out.copy()

# ============================================================================
#         pad (constant) with 1D/2D fast paths and 3D/4D spatial fast paths
# ============================================================================

fn pad[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    pad_width: List[(Int, Int)],
    constant: T
) -> Tensor[T]:
    var rank = len(x._shape)

    # -------------------- 1D fast path --------------------
    if rank == 1 and len(pad_width) == 1:
        var n = x._shape[0]
        var l = clamp_nonneg(pad_width[0][0])
        var r = clamp_nonneg(pad_width[0][1])
        var on = l + n + r
        if on <= 0:
            return Tensor[T]([0], constant).copy()
        var out = Tensor[T]([on], constant)
        var i = 0
        var lim = (n // 8) * 8
        while i < lim:
            out._data[l + i + 0] = x._data[i + 0]
            out._data[l + i + 1] = x._data[i + 1]
            out._data[l + i + 2] = x._data[i + 2]
            out._data[l + i + 3] = x._data[i + 3]
            out._data[l + i + 4] = x._data[i + 4]
            out._data[l + i + 5] = x._data[i + 5]
            out._data[l + i + 6] = x._data[i + 6]
            out._data[l + i + 7] = x._data[i + 7]
            i += 8
        while i < n:
            out._data[l + i] = x._data[i]
            i += 1
        return out.copy()

    # -------------------- 2D fast path --------------------
    if rank == 2 and len(pad_width) == 2:
        var h = x._shape[0]
        var w = x._shape[1]
        var pt = clamp_nonneg(pad_width[0][0])
        var pb = clamp_nonneg(pad_width[0][1])
        var pl = clamp_nonneg(pad_width[1][0])
        var pr = clamp_nonneg(pad_width[1][1])
        var oh = h + pt + pb
        var ow = w + pl + pr
        if oh <= 0 or ow <= 0:
            return Tensor[T]([0, 0], constant).copy()

        var out = Tensor[T]([oh, ow], constant)
        if is_contig2d_rowmajor(x._strides, w):
            var r2 = 0
            while r2 < h:
                var src_row = r2 * w
                var dst_row = (pt + r2) * ow + pl
                var c = 0
                var lim2 = (w // 8) * 8
                while c < lim2:
                    out._data[dst_row + c + 0] = x._data[src_row + c + 0]
                    out._data[dst_row + c + 1] = x._data[src_row + c + 1]
                    out._data[dst_row + c + 2] = x._data[src_row + c + 2]
                    out._data[dst_row + c + 3] = x._data[src_row + c + 3]
                    out._data[dst_row + c + 4] = x._data[src_row + c + 4]
                    out._data[dst_row + c + 5] = x._data[src_row + c + 5]
                    out._data[dst_row + c + 6] = x._data[src_row + c + 6]
                    out._data[dst_row + c + 7] = x._data[src_row + c + 7]
                    c += 8
                while c < w:
                    out._data[dst_row + c] = x._data[src_row + c]
                    c += 1
                r2 += 1
            return out.copy()

        # general 2D strides
        var rs0 = 0
        var rs1 = 0
        if len(x._strides) >= 2:
            rs0 = x._strides[0]
            rs1 = x._strides[1]
        var r3 = 0
        while r3 < h:
            var c3 = 0
            while c3 < w:
                var src = r3 * rs0 + c3 * rs1
                var dst = (pt + r3) * ow + (pl + c3)
                out._data[dst] = x._data[src]
                c3 += 1
            r3 += 1
        return out.copy()

    # -------------------- 3D fast path (last dim contiguous) --------------------
    if rank == 3 and len(pad_width) == 3 and is_contig3d_lastdim(x._shape, x._strides):
        # Typical shape: [C, H, W] or [D, H, W] with W contiguous.
        var D0 = x._shape[0]; var H = x._shape[1]; var W = x._shape[2]
        var p0l = clamp_nonneg(pad_width[0][0]); var p0r = clamp_nonneg(pad_width[0][1])
        var p1t = clamp_nonneg(pad_width[1][0]); var p1b = clamp_nonneg(pad_width[1][1])
        var p2l = clamp_nonneg(pad_width[2][0]); var p2r = clamp_nonneg(pad_width[2][1])
        var OD0 = D0 + p0l + p0r; var OH = H + p1t + p1b; var OW = W + p2l + p2r
        if OD0 <= 0 or OH <= 0 or OW <= 0:
            return Tensor[T]([0, 0, 0], constant).copy()

        var out = Tensor[T]([OD0, OH, OW], constant)
        var oc_stride = OH * OW
        var d = 0
        while d < D0:
            var oh_base_c = (p0l + d) * oc_stride
            var r4 = 0
            while r4 < H:
                var src_row = ((d * H) + r4) * W
                var dst_row = oh_base_c + (p1t + r4) * OW + p2l
                var c4 = 0
                var lim3 = (W // 8) * 8
                while c4 < lim3:
                    out._data[dst_row + c4 + 0] = x._data[src_row + c4 + 0]
                    out._data[dst_row + c4 + 1] = x._data[src_row + c4 + 1]
                    out._data[dst_row + c4 + 2] = x._data[src_row + c4 + 2]
                    out._data[dst_row + c4 + 3] = x._data[src_row + c4 + 3]
                    out._data[dst_row + c4 + 4] = x._data[src_row + c4 + 4]
                    out._data[dst_row + c4 + 5] = x._data[src_row + c4 + 5]
                    out._data[dst_row + c4 + 6] = x._data[src_row + c4 + 6]
                    out._data[dst_row + c4 + 7] = x._data[src_row + c4 + 7]
                    c4 += 8
                while c4 < W:
                    out._data[dst_row + c4] = x._data[src_row + c4]
                    c4 += 1
                r4 += 1
            d += 1
        return out.copy()

    # -------------------- 4D fast path (N,C,H,W; pad only H,W) --------------------
    if rank == 4 and len(pad_width) == 4 and is_contig4d_spatial_last(x._shape, x._strides):
        var N = x._shape[0]; var C = x._shape[1]; var H = x._shape[2]; var W = x._shape[3]
        var pNl = clamp_nonneg(pad_width[0][0]); var pNr = clamp_nonneg(pad_width[0][1])
        var pCl = clamp_nonneg(pad_width[1][0]); var pCr = clamp_nonneg(pad_width[1][1])
        var pHt = clamp_nonneg(pad_width[2][0]); var pHb = clamp_nonneg(pad_width[2][1])
        var pWl = clamp_nonneg(pad_width[3][0]); var pWr = clamp_nonneg(pad_width[3][1])

        var ON = N + pNl + pNr
        var OC = C + pCl + pCr
        var OH = H + pHt + pHb
        var OW = W + pWl + pWr
        if ON <= 0 or OC <= 0 or OH <= 0 or OW <= 0:
            return Tensor[T]([0, 0, 0, 0], constant).copy()

        var out = Tensor[T]([ON, OC, OH, OW], constant)
        var plane_in  = H * W
        var plane_out = OH * OW

        var n = 0
        while n < N:
            var on = pNl + n
            var c = 0
            while c < C:
                var oc = pCl + c
                var base_src_plane = ((n * C) + c) * plane_in
                var base_dst_plane = ((on * OC) + oc) * plane_out
                var r5 = 0
                while r5 < H:
                    var src_row = base_src_plane + r5 * W
                    var dst_row = base_dst_plane + (pHt + r5) * OW + pWl
                    var j = 0
                    var lim4 = (W // 8) * 8
                    while j < lim4:
                        out._data[dst_row + j + 0] = x._data[src_row + j + 0]
                        out._data[dst_row + j + 1] = x._data[src_row + j + 1]
                        out._data[dst_row + j + 2] = x._data[src_row + j + 2]
                        out._data[dst_row + j + 3] = x._data[src_row + j + 3]
                        out._data[dst_row + j + 4] = x._data[src_row + j + 4]
                        out._data[dst_row + j + 5] = x._data[src_row + j + 5]
                        out._data[dst_row + j + 6] = x._data[src_row + j + 6]
                        out._data[dst_row + j + 7] = x._data[src_row + j + 7]
                        j += 8
                    while j < W:
                        out._data[dst_row + j] = x._data[src_row + j]
                        j += 1
                    r5 += 1
                c += 1
            n += 1
        return out.copy()

    # -------------------- general fallback --------------------
    return pad_nd[T](x, pad_width, constant)

# ---------------- Convenience wrappers ----------------

fn pad1d[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], left: Int, right: Int, constant: T) -> Tensor[T]:
    return pad[T](x, [(clamp_nonneg(left), clamp_nonneg(right))], constant)

fn pad2d[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], top: Int, bottom: Int, left: Int, right: Int, constant: T
) -> Tensor[T]:
    return pad[T](
        x,
        [(clamp_nonneg(top), clamp_nonneg(bottom)),
         (clamp_nonneg(left), clamp_nonneg(right))],
        constant
    )

fn pad_nd_flat[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], pads: List[Int], constant: T) -> Tensor[T]:
    var rank = len(x._shape)
    if rank == 0:
        return x.copy()
    var pairs = pairs_from_flat_pads(pads, rank)
    if len(pairs) != rank:
        return x.copy()
    return pad_nd[T](x, pairs, constant)

# ============================================================================
#                     Pad by mode (replicate/reflect/symmetric/circular)
# ============================================================================

fn pad_constant[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], pads: List[Int], constant: T) -> Tensor[T]:
    var rank = len(x._shape)
    if len(pads) != 2 * rank:
        return x.copy()
    var pad_pairs = pairs_from_flat_pads(pads, rank)
    var out_shape = compute_new_shape(x._shape, pad_pairs)
    var out = Tensor[T](out_shape, constant)

    var n_out = numel(out_shape)
    var out_strides = row_major_strides(out_shape)

    var i = 0
    while i < n_out:
        # unravel i w.r.t. out_shape
        var rem = i
        var coord = List[Int]()
        coord.reserve(rank)
        var d0 = 0
        while d0 < rank:
            var st = out_strides[d0]
            var dim = out_shape[d0]
            var v = 0
            if st != 0 and dim != 0:
                v = (rem // st) % dim
            coord.append(v)
            d0 += 1

        var inside = True
        var off_in = 0
        var d = 0
        while d < rank:
            var before = clamp_nonneg(pad_pairs[d][0])
            var pos = coord[d] - before
            if pos < 0 or pos >= x._shape[d]:
                inside = False
                break
            off_in = off_in + pos * x._strides[d]
            d += 1

        if inside:
            var off_out = 0
            var k = 0
            while k < rank:
                off_out = off_out + coord[k] * out._strides[k]
                k += 1
            out._data[off_out] = x._data[off_in]
        i += 1
    return out.copy()

fn pad_nd_map[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], pads: List[Int], mode: Int, constant: T) -> Tensor[T]:
    # mode: 0=replicate, 1=reflect, 2=symmetric, 3=circular, 4=constant
    if mode == 4:
        return pad_constant(x, pads, constant)

    var rank = len(x._shape)
    if len(pads) != 2 * rank:
        return x.copy()

    var pad_pairs = pairs_from_flat_pads(pads, rank)
    var out_shape = compute_new_shape(x._shape, pad_pairs)
    var out = Tensor[T](out_shape, constant)

    var n_out = numel(out_shape)
    var out_strides = row_major_strides(out_shape)

    var i = 0
    while i < n_out:
        var rem = i
        var coord = List[Int]()
        coord.reserve(rank)
        var d0 = 0
        while d0 < rank:
            var st = out_strides[d0]
            var dim = out_shape[d0]
            var v = 0
            if st != 0 and dim != 0:
                v = (rem // st) % dim
            coord.append(v)
            d0 += 1

        var off_in = 0
        var off_out = 0
        var d = 0
        while d < rank:
            var before = clamp_nonneg(pad_pairs[d][0])
            var pos = coord[d] - before
            var idx_m = pad_map_index(pos, x._shape[d], mode)
            off_in  = off_in  + idx_m    * x._strides[d]
            off_out = off_out + coord[d] * out._strides[d]
            d += 1
        out._data[off_out] = x._data[off_in]
        i += 1
    return out.copy()

fn pad_dispatch[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], pads: List[Int], mode: String = "constant", value: T = zero_scalar_of[T](f)
) -> Tensor[T]:
    if mode == "constant":
        return pad_constant(x, pads, value)
    if mode == "replicate":
        return pad_nd_map(x, pads, 0, value)
    if mode == "reflect":
        return pad_nd_map(x, pads, 1, value)
    if mode == "symmetric":
        return pad_nd_map(x, pads, 2, value)
    if mode == "circular":
        return pad_nd_map(x, pads, 3, value)
    return pad_constant(x, pads, value)

# ============================================================================
#                                   Flip
# ============================================================================

fn flip[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Int) -> Tensor[T]:
    var r = len(x._shape)
    if r == 0:
        return x.copy()

    var ax = normalize_axis(axis, r)
    var fill = zero_scalar_of[T](f)
    if len(x._data) > 0:
        fill = x._data[0]
    var out = Tensor[T](x._shape, fill)

    if r == 1:
        var n = x._shape[0]
        var i = 0
        while i < n:
            out._data[i] = x._data[n - 1 - i]
            i += 1
        return out.copy()

    var in_strides = x._strides.copy()
    var out_strides = row_major_strides(x._shape)
    var idx = List[Int]()
    idx.reserve(r)
    var d0 = 0
    while d0 < r:
        idx.append(0)
        d0 += 1

    var done = False
    while not done:
        var src_lin = 0
        var d1 = 0
        while d1 < r:
            var coord = idx[d1]
            if d1 == ax:
                coord = x._shape[d1] - 1 - coord
            src_lin = src_lin + coord * in_strides[d1]
            d1 += 1
        var dst_lin = 0
        var d2 = 0
        while d2 < r:
            dst_lin = dst_lin + idx[d2] * out_strides[d2]
            d2 += 1
        out._data[dst_lin] = x._data[src_lin]

        var dim = r - 1
        while True:
            if idx[dim] + 1 < x._shape[dim]:
                idx[dim] = idx[dim] + 1
                break
            idx[dim] = 0
            dim = dim - 1
            if dim < 0:
                done = True
                break
    return out.copy()

# 2D optimized flip along axis (0: flip rows / vertical, else: flip columns / horizontal)
fn flip2d_axis[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Int) -> Tensor[T]:
    var r = len(x._shape)
    if r != 2:
        return x.copy()

    var h = x._shape[0]
    var w = x._shape[1]
    if h == 0 or w == 0:
        return x.copy()

    # prepare output buffer filled with a valid T (reuse first element)
    var n = h * w
    var fill = x._data[0]
    var out_data = List[T]()
    out_data.reserve(n)
    var i_init = 0
    var lim_init = (n // 8) * 8
    while i_init < lim_init:
        out_data.append(fill); out_data.append(fill); out_data.append(fill); out_data.append(fill)
        out_data.append(fill); out_data.append(fill); out_data.append(fill); out_data.append(fill)
        i_init += 8
    while i_init < n:
        out_data.append(fill)
        i_init += 1

    var shp = List[Int]()
    shp.append(h)
    shp.append(w)
    var out = Tensor[T](out_data, shp)

    # fast path: row-major contiguous 2D
    if is_contig_2d(x._strides, w):
        var i = 0
        if axis == 0:
            # flip vertically (reverse rows)
            while i < h:
                var src_base = (h - 1 - i) * w
                var dst_base = i * w
                var j = 0
                var lim = (w // 8) * 8
                while j < lim:
                    out._data[dst_base + j + 0] = x._data[src_base + j + 0]
                    out._data[dst_base + j + 1] = x._data[src_base + j + 1]
                    out._data[dst_base + j + 2] = x._data[src_base + j + 2]
                    out._data[dst_base + j + 3] = x._data[src_base + j + 3]
                    out._data[dst_base + j + 4] = x._data[src_base + j + 4]
                    out._data[dst_base + j + 5] = x._data[src_base + j + 5]
                    out._data[dst_base + j + 6] = x._data[src_base + j + 6]
                    out._data[dst_base + j + 7] = x._data[src_base + j + 7]
                    j += 8
                while j < w:
                    out._data[dst_base + j] = x._data[src_base + j]
                    j += 1
                i += 1
        else:
            # flip horizontally (reverse columns)
            while i < h:
                var src_base2 = i * w
                var dst_base2 = i * w
                var j2 = 0
                var lim2 = (w // 4) * 4
                while j2 < lim2:
                    out._data[dst_base2 + j2 + 0] = x._data[src_base2 + (w - 1 - (j2 + 0))]
                    out._data[dst_base2 + j2 + 1] = x._data[src_base2 + (w - 1 - (j2 + 1))]
                    out._data[dst_base2 + j2 + 2] = x._data[src_base2 + (w - 1 - (j2 + 2))]
                    out._data[dst_base2 + j2 + 3] = x._data[src_base2 + (w - 1 - (j2 + 3))]
                    j2 += 4
                while j2 < w:
                    out._data[dst_base2 + j2] = x._data[src_base2 + (w - 1 - j2)]
                    j2 += 1
                i += 1
        return out.copy()

    # general 2D strides
    var rs0 = 0
    var rs1 = 0
    if len(x._strides) >= 2:
        rs0 = x._strides[0]
        rs1 = x._strides[1]

    var i2 = 0
    if axis == 0:
        # flip vertically using strides
        while i2 < h:
            var j3 = 0
            while j3 < w:
                var si = h - 1 - i2
                var sj = j3
                var src = si * rs0 + sj * rs1
                out._data[i2 * w + j3] = x._data[src]
                j3 += 1
            i2 += 1
    else:
        # flip horizontally using strides
        while i2 < h:
            var j4 = 0
            while j4 < w:
                var si2 = i2
                var sj2 = w - 1 - j4
                var src2 = si2 * rs0 + sj2 * rs1
                out._data[i2 * w + j4] = x._data[src2]
                j4 += 1
            i2 += 1

    return out.copy()

@always_inline
fn fliplr[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[T]:
    return flip2d_axis[T](x, 1)

@always_inline
fn flipud[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[T]:
    return flip2d_axis[T](x, 0)

# ============================================================================
#                                Upsampling
# ============================================================================

fn upsample_nearest[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], scale: Int) -> Tensor[T]:
    var s = scale
    if s < 1:
        s = 1
    var in_shape = copy_list_int(x._shape)
    var r = len(in_shape)
    if r < 2:
        return x.copy()

    var (h_ax, w_ax) = _spatial_axes(in_shape)
    var H = in_shape[h_ax]; var W = in_shape[w_ax]
    var out_shape = copy_list_int(in_shape)
    out_shape[h_ax] = H * s
    out_shape[w_ax] = W * s

    var in_strides = row_major_strides(in_shape)
    var out_strides = row_major_strides(out_shape)
    var out_n = _prod(out_shape)
    var out = Tensor[T](out_shape, zero_scalar_of[T](f))

    var idx = List[Int]()
    idx.reserve(r)
    var z = 0
    while z < r:
        idx.append(0)
        z += 1

    var out_idx = 0
    while out_idx < out_n:
        # unravel out_idx
        var rem = out_idx
        var ax = 0
        while ax < r:
            var stride = out_strides[ax]
            var dim = out_shape[ax]
            var v = 0
            if stride != 0 and dim != 0:
                v = (rem // stride) % dim
            idx[ax] = v
            ax += 1

        var oy = idx[h_ax]; var ox = idx[w_ax]
        var iy = oy // s;   var ix = ox // s

        var in_lin = 0
        var a = 0
        while a < r:
            if a == h_ax:
                in_lin = in_lin + iy * in_strides[a]
            elif a == w_ax:
                in_lin = in_lin + ix * in_strides[a]
            else:
                in_lin = in_lin + idx[a] * in_strides[a]
            a += 1

        out._data[out_idx] = x._data[in_lin]
        out_idx += 1
    return out.copy()

@always_inline
fn clamp_int(x0: Int, lo: Int, hi: Int) -> Int:
    var x = x0
    if x < lo: x = lo
    if x > hi: x = hi
    return x
@always_inline
fn clamp_f64(x: Float64, lo: Float64, hi: Float64) -> Float64:
    var v = x
    if v < lo: v = lo
    if v > hi: v = hi
    return v

fn _to_f64_buffer[T: ImplicitlyCopyable & Copyable & Movable](src: List[T]) -> List[Float64]:
    var n = len(src)
    var out = List[Float64]()
    out.reserve(n)
    var i = 0
    while i < n:
        out.append(Float64(src[i]))
        i += 1
    return out

fn _from_f64_buffer[T: ImplicitlyCopyable & Copyable & Movable](src: List[Float64]) -> List[T]:
    var n = len(src)
    var out = List[T]()
    out.reserve(n)
    var i = 0
    while i < n:
        out.append(T(src[i]))
        i += 1
    return out

fn upsample_bilinear[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], scale: Int) -> Tensor[T]:
    var s = scale
    if s < 1:
        s = 1
    var in_shape = copy_list_int(x._shape)
    var r = len(in_shape)
    if r < 2:
        return x.copy()

    var (h_ax, w_ax) = _spatial_axes(in_shape)
    var H = in_shape[h_ax]; var W = in_shape[w_ax]
    var H2 = H * s; var W2 = W * s
    var out_shape = copy_list_int(in_shape)
    out_shape[h_ax] = H2; out_shape[w_ax] = W2

    var in_strides = row_major_strides(in_shape)
    var out_strides = row_major_strides(out_shape)
    var out_n = _prod(out_shape)

    var xin = _to_f64_buffer[T](x._data)
    var out_f = List[Float64]()
    out_f.reserve(out_n)
    var i0 = 0
    while i0 < out_n:
        out_f.append(0.0)
        i0 += 1

    var idx = List[Int]()
    idx.reserve(r)
    var t = 0
    while t < r:
        idx.append(0)
        t += 1

    var out_idx = 0
    var inv_s = 1.0 / Float64(s)

    while out_idx < out_n:
        var rem = out_idx
        var ax = 0
        while ax < r:
            var stride = out_strides[ax]
            var dim = out_shape[ax]
            var v = 0
            if stride != 0 and dim != 0:
                v = (rem // stride) % dim
            idx[ax] = v
            ax += 1

        var oy = idx[h_ax]; var ox = idx[w_ax]
        var fy = inv_s * (Float64(oy) + 0.5) - 0.5
        var fx = inv_s * (Float64(ox) + 0.5) - 0.5

        var y0 = clamp_int(Int(floor(fy)), 0, H - 1)
        var x0 = clamp_int(Int(floor(fx)), 0, W - 1)
        var y1 = clamp_int(y0 + 1, 0, H - 1)
        var x1 = clamp_int(x0 + 1, 0, W - 1)

        var wy = fy - Float64(y0)
        if wy < 0.0:
            wy = 0.0
        if wy > 1.0:
            wy = 1.0
        var wx = fx - Float64(x0)
        if wx < 0.0:
            wx = 0.0
        if wx > 1.0:
            wx = 1.0

        var base0 = 0
        var a = 0
        while a < r:
            if a == h_ax:
                base0 = base0 + y0 * in_strides[a]
            elif a == w_ax:
                base0 = base0 + x0 * in_strides[a]
            else:
                base0 = base0 + idx[a] * in_strides[a]
            a += 1

        var base1 = base0 + (x1 - x0) * in_strides[w_ax]
        var base2 = base0 + (y1 - y0) * in_strides[h_ax]
        var base3 = base2 + (x1 - x0) * in_strides[w_ax]

        var v00 = xin[base0]
        var v01 = xin[base1]
        var v10 = xin[base2]
        var v11 = xin[base3]

        var w00 = (1.0 - wy) * (1.0 - wx)
        var w01 = (1.0 - wy) * wx
        var w10 = wy * (1.0 - wx)
        var w11 = wy * wx

        out_f[out_idx] = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11
        out_idx += 1

    var out_t = _from_f64_buffer[T](out_f)
    return Tensor[T](out_t, out_shape)

# ============================================================================
#                           Downsample (mean s×s)
# ============================================================================

fn downsample_mean[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], scale: Int) -> Tensor[T]:
    var s = scale
    if s < 1:
        s = 1
    var in_shape = copy_list_int(x._shape)
    var r = len(in_shape)
    if r < 2:
        return x.copy()

    var (h_ax, w_ax) = _spatial_axes(in_shape)
    var H = in_shape[h_ax]; var W = in_shape[w_ax]
    var H2 = H // s; var W2 = W // s

    var out_shape = copy_list_int(in_shape)
    out_shape[h_ax] = H2; out_shape[w_ax] = W2

    var in_strides = row_major_strides(in_shape)
    var out_strides = row_major_strides(out_shape)
    var out_n = _prod(out_shape)
    var out_f = List[Float64]()
    out_f.reserve(out_n)
    var i = 0
    while i < out_n:
        out_f.append(0.0)
        i += 1

    var xin = _to_f64_buffer[T](x._data)
    var idx = List[Int]()
    idx.reserve(r)
    var t = 0
    while t < r:
        idx.append(0)
        t += 1

    var out_idx = 0
    var win_area = Float64(s * s)

    while out_idx < out_n:
        var rem = out_idx
        var ax = 0
        while ax < r:
            var stride = out_strides[ax]
            var dim = out_shape[ax]
            var v = 0
            if stride != 0 and dim != 0:
                v = (rem // stride) % dim
            idx[ax] = v
            ax += 1

        var oy = idx[h_ax]; var ox = idx[w_ax]
        var y0 = oy * s; var x0 = ox * s

        var sum = 0.0
        var dy = 0
        while dy < s:
            var yy = y0 + dy
            var dx = 0
            while dx < s:
                var xx = x0 + dx
                var in_lin = 0
                var a = 0
                while a < r:
                    if a == h_ax:
                        in_lin = in_lin + yy * in_strides[a]
                    elif a == w_ax:
                        in_lin = in_lin + xx * in_strides[a]
                    else:
                        in_lin = in_lin + idx[a] * in_strides[a]
                    a += 1
                sum = sum + xin[in_lin]
                dx += 1
            dy += 1

        out_f[out_idx] = sum / win_area
        out_idx += 1

    var out_t = _from_f64_buffer[T](out_f)
    return Tensor[T](out_t, out_shape)

# ============================================================================
#                                   Dilate
# ============================================================================

fn dilate[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], k: Int, axis: Int = -1) -> Tensor[T]:
    var factor = k
    if factor < 1:
        factor = 1
    var in_shape = copy_list_int(x._shape)
    var r = len(in_shape)
    var ax = axis
    while ax < 0:
        ax = ax + r
    if ax >= r:
        ax = r - 1

    var out_shape = copy_list_int(in_shape)
    var n_ax = in_shape[ax]
    var out_len_ax = 0
    if n_ax > 0:
        out_len_ax = n_ax + (n_ax - 1) * (factor - 1)
    out_shape[ax] = out_len_ax

    var in_strides = row_major_strides(in_shape)
    var out_strides = row_major_strides(out_shape)
    var out_n = _prod(out_shape)
    var out = Tensor[T](out_shape, zero_scalar_of[T](f))

    var in_n = _prod(in_shape)
    var idx = List[Int]()
    idx.reserve(r)
    var t = 0
    while t < r:
        idx.append(0)
        t += 1

    var in_idx = 0
    while in_idx < in_n:
        var rem = in_idx
        var a = 0
        while a < r:
            var stride = in_strides[a]
            var dim = in_shape[a]
            var v = 0
            if stride != 0 and dim != 0:
                v = (rem // stride) % dim
            idx[a] = v
            a += 1

        var out_lin = 0
        var d = 0
        while d < r:
            var pos = idx[d]
            if d == ax:
                pos = pos * factor
            out_lin = out_lin + pos * out_strides[d]
            d += 1

        out._data[out_lin] = x._data[in_idx]
        in_idx += 1
    return out.copy()

# ============================================================================
#                           Sliding window (1D → 2D)
# ============================================================================

@always_inline
fn _empty_2d[T: ImplicitlyCopyable & Copyable & Movable]() -> Tensor[T]:
    var empty = List[T]()
    return Tensor[T](empty, [0, 0]).copy()

fn sliding_window_core[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], win: Int, step: Int) -> Tensor[T]:
    var rank = len(x._shape)
    if rank != 1:
        return x.copy()

    var n = x._shape[0]
    if n <= 0:
        return _empty_2d[T]()

    if win <= 0 or win > n:
        return _empty_2d[T]()

    var st = step
    if st <= 0:
        st = 1

    var rows = 1 + ((n - win) // st)
    var cols = win
    if rows <= 0 or cols <= 0:
        return _empty_2d[T]()

    var shp = List[Int]()
    shp.append(rows)
    shp.append(cols)

    var total = rows * cols
    var out_data = List[T]()
    out_data.reserve(total)
    var fill = x._data[0]
    var k = 0
    var lim8 = (total // 8) * 8
    while k < lim8:
        out_data.append(fill); out_data.append(fill); out_data.append(fill); out_data.append(fill)
        out_data.append(fill); out_data.append(fill); out_data.append(fill); out_data.append(fill)
        k += 8
    while k < total:
        out_data.append(fill)
        k += 1

    var out = Tensor[T](out_data, shp)

    var s0 = 1
    if len(x._strides) > 0:
        s0 = x._strides[0]

    var i = 0
    while i < rows:
        var base_dst = i * cols
        var cur = (i * st) * s0

        var j = 0
        var lim16 = (cols // 16) * 16
        while j < lim16:
            out._data[base_dst + j +  0] = x._data[cur]; cur += s0
            out._data[base_dst + j +  1] = x._data[cur]; cur += s0
            out._data[base_dst + j +  2] = x._data[cur]; cur += s0
            out._data[base_dst + j +  3] = x._data[cur]; cur += s0
            out._data[base_dst + j +  4] = x._data[cur]; cur += s0
            out._data[base_dst + j +  5] = x._data[cur]; cur += s0
            out._data[base_dst + j +  6] = x._data[cur]; cur += s0
            out._data[base_dst + j +  7] = x._data[cur]; cur += s0
            out._data[base_dst + j +  8] = x._data[cur]; cur += s0
            out._data[base_dst + j +  9] = x._data[cur]; cur += s0
            out._data[base_dst + j + 10] = x._data[cur]; cur += s0
            out._data[base_dst + j + 11] = x._data[cur]; cur += s0
            out._data[base_dst + j + 12] = x._data[cur]; cur += s0
            out._data[base_dst + j + 13] = x._data[cur]; cur += s0
            out._data[base_dst + j + 14] = x._data[cur]; cur += s0
            out._data[base_dst + j + 15] = x._data[cur]; cur += s0
            j += 16

        var lim8b = (cols // 8) * 8
        while j < lim8b:
            out._data[base_dst + j + 0] = x._data[cur]; cur += s0
            out._data[base_dst + j + 1] = x._data[cur]; cur += s0
            out._data[base_dst + j + 2] = x._data[cur]; cur += s0
            out._data[base_dst + j + 3] = x._data[cur]; cur += s0
            out._data[base_dst + j + 4] = x._data[cur]; cur += s0
            out._data[base_dst + j + 5] = x._data[cur]; cur += s0
            out._data[base_dst + j + 6] = x._data[cur]; cur += s0
            out._data[base_dst + j + 7] = x._data[cur]; cur += s0
            j += 8

        while j < cols:
            out._data[base_dst + j] = x._data[cur]; cur += s0
            j += 1

        i += 1

    return out.copy()

@always_inline
fn sliding_window[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], win: Int) -> Tensor[T]:
    return sliding_window_core[T](x, win, 1)

@always_inline
fn sliding_window_step[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], win: Int, step: Int) -> Tensor[T]:
    return sliding_window_core[T](x, win, step)

# ============================================================================
#                               Generic view ops
# ============================================================================

# as_strided: unsafe view; caller must ensure validity.
fn as_strided[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], size: List[Int], stride: List[Int], storage_offset: Int = 0
) -> Tensor[T]:
    var _ = storage_offset  # silence unused param
    var out = clone_header_share_data[T](x, x._shape, x._strides)
    out._shape = copy_list_int(size)
    out._strides = copy_list_int(stride)
    return out.copy()

# expand: broadcast via stride=0 on expanded dims
fn expand[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], new_shape: List[Int]) -> Tensor[T]:
    var xr = len(x._shape)
    var yr = len(new_shape)

    var pad = 0
    if yr > xr:
        pad = yr - xr

    var shapeA = List[Int]()
    shapeA.reserve(yr)
    var strideA = List[Int]()
    strideA.reserve(yr)

    var i = 0
    while i < pad:
        shapeA.append(1)
        strideA.append(0)
        i += 1
    var j = 0
    while j < xr:
        shapeA.append(x._shape[j])
        strideA.append(x._strides[j])
        j += 1

    var out_strides = List[Int]()
    out_strides.reserve(yr)
    var d = 0
    while d < yr:
        var sd = shapeA[d]
        var nd = new_shape[d]
        if sd == nd:
            out_strides.append(strideA[d])
        else:
            if sd == 1 and nd > 1:
                out_strides.append(0)  # broadcast view
            else:
                out_strides.append(strideA[d])  # best-effort
        d += 1

    var out = clone_header_share_data[T](x, x._shape, x._strides)
    out._shape = copy_list_int(new_shape)
    out._strides = out_strides
    return out.copy()

@always_inline
fn expand_as[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], other: Tensor[T]) -> Tensor[T]:
    return expand(x, other._shape)

# -----------------------------------------------------------------------------
# permute: header-only view (no data copy)
# - axes length must equal rank
# - each axis must be in-range and appear exactly once
# -----------------------------------------------------------------------------
@always_inline
fn _clamp_axis(ax: Int, rank: Int) -> Int:
    var i = ax
    if i < 0: i = i + rank
    #assert(0 <= i and i < rank and "permute: axis out of range")
    return i

# -----------------------------------------------------------------------------
# Safe axis normalize without assert: returns Optional[Int]
# -----------------------------------------------------------------------------
@always_inline
fn _norm_axis_safe(ax: Int, rank: Int) -> Optional[Int]:
    var i = ax
    if i < 0:
        i = i + rank
    if 0 <= i and i < rank:
        return Optional[Int](i)
    return None

# -----------------------------------------------------------------------------
# permute: header-only view (no data copy), no asserts.
# If axes invalid → returns x.copy() as a safe fallback.
# -----------------------------------------------------------------------------
@always_inline
fn permute[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axes: List[Int]) -> Tensor[T]:
    var r = len(x._shape)
    if r <= 1:
        return x.copy()

    if len(axes) != r:
        return x.copy()

    # seen mask without BitSet
    var seen = List[Bool]()
    seen.reserve(r)
    var t = 0
    while t < r:
        seen.append(False)
        t += 1

    var a = List[Int]()
    a.reserve(r)

    var i = 0
    while i < r:
        var v_opt = _norm_axis_safe(axes[i], r)
        if v_opt is None:
            return x.copy()
        var v = v_opt.value()
        if seen[v]:
            return x.copy()
        seen[v] = True
        a.append(v)
        i += 1

    var new_shape = List[Int]()
    new_shape.reserve(r)
    var new_strides = List[Int]()
    new_strides.reserve(r)

    var k = 0
    while k < r:
        var idx = a[k]
        new_shape.append(x._shape[idx])
        new_strides.append(x._strides[idx])
        k += 1

    var out = clone_header_share_data[T](x, x._shape, x._strides)
    out._shape = new_shape.copy()
    out._strides = new_strides.copy()
    return out.copy()


# size
@always_inline
fn size[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Int:
    return numel(x._shape)

fn contiguous[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[T]:
    if is_row_major_contiguous(x._shape, x._strides):
        # already contiguous: return a header-clone that aliases data (or use x.copy() if you want a deep copy)
        return clone_header_share_data[T](x, x._shape.copy(), x._strides.copy())

    var n = numel(x._shape)

    # allocate a flat buffer of size n (initialize with first element to avoid default-inits)
    var out_data = List[T]()
    out_data.reserve(n)
    if n > 0:
        var fill = x._data[0]
        var k = 0
        var lim = (n // 16) * 16
        while k < lim:
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            out_data.append(fill)
            k = k + 16
        while k < n:
            out_data.append(fill)
            k = k + 1

    # ctor order is (shape, flat)
    var out = Tensor[T](x._shape.copy(), out_data)
    out._strides = row_major_strides(x._shape)

    # iterate logical indices and gather from x into out (row-major)
    var rank = len(x._shape)
    var idx = List[Int]()
    idx.reserve(rank)
    var i0 = 0
    while i0 < rank:
        idx.append(0)
        i0 = i0 + 1

    var lin = 0
    while lin < n:
        var off = ravel_index_with_offset(idx, x._strides, 0)
        out._data[lin] = x._data[off]

        # increment mixed-radix counter
        var d = rank - 1
        while d >= 0:
            idx[d] = idx[d] + 1
            if idx[d] < x._shape[d]:
                break
            idx[d] = 0
            d = d - 1
        lin = lin + 1

    # ensure caller gets a tensor with its own header/buffer (optional; remove .copy() if aliasing is okay)
    return out.copy()
# -----------------------------------------------------------------------------
# Internal: compute the target strides for a reshape *without copy*.
# We only allow view-reshape when the source is fully row-major contiguous;
# this matches common fast-paths and keeps logic simple & safe.
# -----------------------------------------------------------------------------
@always_inline
fn view_reshape_strides_if_contiguous(
    old_shape: List[Int],
    new_shape: List[Int]
) -> (Bool, List[Int]):
    var ok = is_row_major_contiguous(old_shape, compute_row_major_strides(old_shape))
    if not ok:
        return (False, List[Int]())

    # Row-major view: standard row-major strides for the new shape
    var ns = compute_row_major_strides(new_shape)
    return (True, ns)
 
# -----------------------------------------------------------------------------
# Strict reshape: size must match. Returns a view when C-contiguous; otherwise
# materializes a contiguous buffer and updates header.
# -----------------------------------------------------------------------------
fn reshape[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], new_shape: List[Int]
) -> Tensor[T]:
    # Size must be preserved
    var n_old = numel(x._shape)
    var n_new = numel(new_shape)
    if n_old != n_new:
        return x.copy()

    # View path: if source is row-major contiguous, build a header-only view
    if is_row_major_contiguous(x._shape, x._strides):
        var st = compute_row_major_strides(new_shape)
        return Tensor[T](x._data, new_shape, st, x._offset)

    # Fallback: materialize contiguous, then update header
    var c = contiguous(x)
    var st2 = compute_row_major_strides(new_shape)
    return Tensor[T](c._data, new_shape, st2, 0)

# -----------------------------------------------------------------------------
# Reshape with a single -1 (NumPy/PyTorch semantics).
# Additionally supports the zero-size corner case:
# - If total == 0 and there is exactly one -1:
#     * If known == 0 (any zero dimension in the explicit part), set the inferred
#       dimension to 0.
#     * Else (known > 0), inferred = 0 as well to keep total size 0.
# Multiple -1s or negative dims (other than -1) are rejected.
# -----------------------------------------------------------------------------
fn reshape_infer[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], new_shape_in: List[Int]
) -> Tensor[T]:
    var new_shape = List[Int]()
    var known = 1
    var infer_idx = -1

    var i = 0
    var n = len(new_shape_in)
    new_shape.reserve(n)
    while i < n:
        var d = new_shape_in[i]
        if d == -1:
            if infer_idx != -1:
                return x.copy()  # multiple -1 not allowed
            infer_idx = i
            new_shape.append(1)  # placeholder
        else:
            if d < 0:
                return x.copy()
            known = known * d
            new_shape.append(d)
        i += 1

    var total = numel(x._shape)

    if infer_idx != -1:
        # Zero-size corner case policy:
        # If total == 0, infer the missing dimension as 0 regardless of 'known'.
        if total == 0:
            new_shape[infer_idx] = 0
            return reshape(x, new_shape)

        # Regular inference
        if known == 0:
            # total > 0 but known == 0 is inconsistent
            return x.copy()
        if total % known != 0:
            return x.copy()
        new_shape[infer_idx] = total // known
    else:
        if numel(new_shape) != total:
            return x.copy()

    return reshape(x, new_shape)

# -----------------------------------------------------------------------------
# Convenience: reshape_like
# -----------------------------------------------------------------------------
@always_inline
fn reshape_like[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], other: Tensor[T]
) -> Tensor[T]:
    return reshape(x, other._shape)

# -----------------------------------------------------------------------------
# Resize-with-pad/truncate to match 'target_like' element count, then reshape
# to target_like.shape. If target has more elements, pad with the last copied
# value (or zero when source is empty), otherwise truncate. Produces row-major.
# -----------------------------------------------------------------------------
@always_inline
fn _resize_like_with_pad_core[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    target_like: Tensor[T],
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    var target = target_like._shape.copy()
    var n_src = numel(x._shape)
    var n_tgt = numel(target)

    # Equal-size: strict reshape
    if n_src == n_tgt:
        return reshape[T](x, target)

    # Materialize flat contiguous source
    var c = contiguous[T](x)

    # Destination buffer
    var data = List[T]()
    data.reserve(n_tgt)

    # Copy the overlapping prefix (unrolled by 16)
    var nmin = n_src if n_src < n_tgt else n_tgt
    var i = 0
    var lim = (nmin // 16) * 16
    while i < lim:
        data.append(c._data[i    ]); data.append(c._data[i + 1 ])
        data.append(c._data[i + 2]); data.append(c._data[i + 3 ])
        data.append(c._data[i + 4]); data.append(c._data[i + 5 ])
        data.append(c._data[i + 6]); data.append(c._data[i + 7 ])
        data.append(c._data[i + 8]); data.append(c._data[i + 9 ])
        data.append(c._data[i +10]); data.append(c._data[i +11])
        data.append(c._data[i +12]); data.append(c._data[i +13])
        data.append(c._data[i +14]); data.append(c._data[i +15])
        i += 16
    while i < nmin:
        data.append(c._data[i])
        i += 1

    # Pad if needed (repeat last or zero if source empty)
    if n_tgt > nmin:
        var fill = data[nmin - 1] if nmin > 0 else from_f64(0.0)
        var r = n_tgt - nmin
        var k = 0
        var lim2 = (r // 8) * 8
        while k < lim2:
            data.append(fill); data.append(fill); data.append(fill); data.append(fill)
            data.append(fill); data.append(fill); data.append(fill); data.append(fill)
            k += 8
        while k < r:
            data.append(fill)
            k += 1

    # Row-major destination with target shape
    return Tensor[T](data, target)
# -------------------------------- Overloads ----------------------------------
@always_inline
fn resize_like_with_pad(x: Tensor[Int], target_like: Tensor[Int]) -> Tensor[Int]:
    return _resize_like_with_pad_core[Int](x, target_like, f64_to_int)

@always_inline
fn resize_like_with_pad(x: Tensor[Float32], target_like: Tensor[Float32]) -> Tensor[Float32]:
    return _resize_like_with_pad_core[Float32](x, target_like, f64_to_float32)

@always_inline
fn resize_like_with_pad(x: Tensor[Float64], target_like: Tensor[Float64]) -> Tensor[Float64]:
    return _resize_like_with_pad_core[Float64](x, target_like, f64_to)


        
 

@always_inline
fn ravel[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[T]:
    return flatten(x)

# squeeze / unsqueeze
fn squeeze_all[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[T]:
    var r = len(x._shape)
    var keep = List[Int]()
    keep.reserve(r)
    var ks = List[Int]()
    ks.reserve(r)
    var i = 0
    while i < r:
        if x._shape[i] != 1:
            keep.append(x._shape[i])
            ks.append(x._strides[i])
        i += 1
    if len(keep) == 0:
        keep.append(1)
        ks.append(1)
    var out = clone_header_share_data[T](x, x._shape, x._strides)
    out._shape = keep.copy()
    out._strides = ks.copy()
    return out.copy()

fn squeeze_axis[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Int) -> Tensor[T]:
    var r = len(x._shape)
    var a = normalize_axis(axis, r)
    var keep = List[Int]()
    keep.reserve(r)
    var ks = List[Int]()
    ks.reserve(r)
    var i = 0
    while i < r:
        if i != a or x._shape[i] != 1:
            keep.append(x._shape[i])
            ks.append(x._strides[i])
        i += 1
    if len(keep) == 0:
        keep.append(1)
        ks.append(1)
    var out = clone_header_share_data[T](x, x._shape, x._strides)
    out._shape = keep.copy()
    out._strides = ks.copy()
    return out.copy()

fn unsqueeze[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis_in: Int) -> Tensor[T]:
    var r = len(x._shape)
    var a = axis_in
    while a < 0:
        a = a + (r + 1)
    if a < 0:
        a = 0
    if a > r:
        a = r
    var ns = List[Int]()
    ns.reserve(r + 1)
    var st = List[Int]()
    st.reserve(r + 1)
    var i = 0
    while i < a:
        ns.append(x._shape[i])
        st.append(x._strides[i])
        i += 1
    ns.append(1)
    st.append(1)
    var j = a
    while j < r:
        ns.append(x._shape[j])
        st.append(x._strides[j])
        j += 1
    var out = clone_header_share_data[T](x, x._shape, x._strides)
    out._shape = ns.copy()
    out._strides = st.copy()
    return out.copy()

# ============================================================================
#                         Tensor-ops variants (utility)
# ============================================================================



fn tensor_squeeze[T: Copyable & Movable](t: Tensor[T]) -> Tensor[T]:
    var sh = t._shape
    var r = len(sh)
    var out = List[Int]()
    out.reserve(r)
    var changed = False
    var i = 0
    while i < r:
        var d = sh[i]
        if d != 1:
            out.append(d)
        else:
            changed = True
        i += 1
    if len(out) == 0:
        out.append(1)
    if not changed:
        return t.copy()
    return t.reshape(out)

fn tensor_expand_dims[T: Copyable & Movable](t: Tensor[T], axis_in: Int) -> Tensor[T]:
    var r = len(t._shape)
    var axis = axis_in
    if axis < 0:
        axis = axis + r + 1
    if axis < 0:
        axis = 0
    if axis > r:
        axis = r
    var ns = List[Int]()
    ns.reserve(r + 1)
    var i = 0
    while i < axis:
        ns.append(t._shape[i])
        i += 1
    ns.append(1)
    while i < r:
        ns.append(t._shape[i])
        i += 1
    return t.reshape(ns)

fn tensor_flatten_view[T: Copyable & Movable](x: Tensor[T]) -> Tensor[T]:
    var n = numel(x._shape)
    if is_row_major_full(x._shape, x._strides):
        return Tensor[T](x._data, [n])
    var c = contiguous(x)
    return Tensor[T](c._data, [n])

fn tensor_as_strided[T: Copyable & Movable](x: Tensor[T], size: List[Int], stride: List[Int], storage_offset: Int = 0) -> Tensor[T]:
    var _ = storage_offset
    var out = Tensor[T](x._data, size)
    out._strides = stride
    return out.copy()

fn tensor_expand[T: Copyable & Movable](x: Tensor[T], new_shape: List[Int]) -> Tensor[T]:
    var xr = len(x._shape)
    var yr = len(new_shape)
    var pad = 0
    if yr > xr:
        pad = yr - xr

    var shapeA = List[Int]()
    shapeA.reserve(yr)
    var strideA = List[Int]()
    strideA.reserve(yr)

    var i = 0
    while i < pad:
        shapeA.append(1)
        strideA.append(0)
        i += 1
    var j = 0
    while j < xr:
        shapeA.append(x._shape[j])
        strideA.append(x._strides[j])
        j += 1

    var out_strides = List[Int]()
    out_strides.reserve(yr)
    var d = 0
    while d < yr:
        var sd = shapeA[d]
        if sd == new_shape[d]:
            out_strides.append(strideA[d])
        else:
            if sd == 1 and new_shape[d] > 1:
                out_strides.append(0)
            else:
                out_strides.append(strideA[d])
        d += 1

    var out = Tensor[T](x._data, new_shape)
    out._strides = out_strides
    return out.copy()

fn tensor_permute[T: Copyable & Movable](x: Tensor[T], axes: List[Int]) -> Tensor[T]:
    var r = len(x._shape)
    var a = List[Int]()
    a.reserve(len(axes))
    var i = 0
    while i < len(axes):
        var ax = axes[i]
        if ax < 0:
            ax = ax + r
        if ax < 0:
            ax = 0
        if ax >= r:
            ax = r - 1 if r > 0 else 0
        a.append(ax)
        i += 1

    var new_shape = List[Int]()
    new_shape.reserve(r)
    var new_strides = List[Int]()
    new_strides.reserve(r)
    var k = 0
    while k < r:
        var idx = a[k]
        new_shape.append(x._shape[idx])
        new_strides.append(x._strides[idx])
        k += 1

    var out = Tensor[T](x._data, new_shape)
    out._strides = new_strides.copy()
    return out.copy()



 

# ----------------------------- utility function -----------------------------


@always_inline
fn clamp_axis(ax: Int, rank: Int) -> Int:
    var v = ax
    if v < 0: v = v + rank
    if v < 0: v = 0
    if v >= rank: v = rank - 1
    return v

@always_inline
fn is_identity_perm(clean: List[Int]) -> Bool:
    var i = 0
    var n = len(clean)
    while i < n:
        if clean[i] != i: return False
        i = i + 1
    return True

 

@always_inline
fn is_row_major(shape: List[Int], strides: List[Int]) -> Bool:
    # Avoid allocating a temp stride list; check on-the-fly.
    var r = len(shape)
    if r != len(strides): return False
    var expected = 1
    var k = r - 1
    while k >= 0:
        if strides[k] != expected: return False
        expected = expected * shape[k]
        k = k - 1
    return True
 
 

# -----------------------------------------------------------------------------
# view (reshape):
# - Single pass to copy shape & gather (-1) info
# - Product checks with early exits
# - Contiguity check without temp allocations
# - Falls back to flatten(x) only if not row-major
# -----------------------------------------------------------------------------
@always_inline
fn view[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], shape: List[Int]
) -> Tensor[T]:
    # ----------------------------
    # 1) Validate & prepare shape
    # ----------------------------
    var new_shape = List[Int]()
    var infer_pos = -1
    var neg_count = 0
    var known = 1

    var nshape = len(shape)
    new_shape.reserve(nshape)

    var i = 0
    while i < nshape:
        var d = shape[i]
        new_shape.append(d)

        if d == -1:
            neg_count = neg_count + 1
            infer_pos = i
        else:
            # Disallow non-positive dims (0 and negatives other than -1)
            if d <= 0:
                return x.copy()
            known = known * d
        i = i + 1

    if neg_count > 1:
        return x.copy()

    # ----------------------------
    # 2) Compute total elements
    # ----------------------------
    var total = 1
    var r = len(x._shape)
    i = 0
    while i < r:
        total = total * x._shape[i]
        i = i + 1

    # ----------------------------
    # 3) Infer the -1 dimension
    # ----------------------------
    if infer_pos >= 0:
        # known cannot be 0 here because we rejected d <= 0 above
        var q = total // known
        if q * known != total:
            return x.copy()
        new_shape[infer_pos] = q

    # ----------------------------
    # 4) Final product must match
    # ----------------------------
    var prod = 1
    i = 0
    var new_r = len(new_shape)
    while i < new_r:
        # new_shape[i] is guaranteed > 0 now
        prod = prod * new_shape[i]
        i = i + 1

    if prod != total:
        return x.copy()

    # -------------------------------------------------
    # 5) Ensure row-major contiguity (zero-copy if possible)
    # -------------------------------------------------
    # If x is already row-major contiguous, avoid copying.
    # Otherwise, materialize a contiguous base using flatten(x).
    var base: Tensor[T]
    if is_row_major(x._shape, x._strides):
        base = x.copy()
    else:
        base = flatten(x)  # project fast path: produces row-major contiguous storage

    # ----------------------------
    # 6) Build row-major strides
    # ----------------------------
    var want_strides = row_major_strides(new_shape)

    # ----------------------------
    # 7) Zero-copy rewrap
    # ----------------------------
    return Tensor[T](base._data, new_shape, want_strides, base._offset)


@always_inline
fn _normalize_reps(rank: Int, reps: List[Int]) -> List[Int]:
    # Normalize reps to match tensor rank without throwing.
    # - Scalars (rank==0): shape becomes reps (negatives -> 0).
    # - Non-scalars:
    #   * If len(reps) < rank: left-pad with 1s.
    #   * If len(reps) > rank: keep the rightmost `rank` values (tail-align).
    #   * Clamp negatives to 0 in all cases.

    var rlen = len(reps)
    var out  = List[Int]()

    if rank == 0:
        var i = 0
        while i < rlen:
            var v = reps[i]
            if v < 0: v = 0
            out.append(v)
            i += 1
        return out.copy()

    # Non-scalar
    if rlen < rank:
        # left-pad with 1s
        var pad = rank - rlen
        var i = 0
        while i < pad:
            out.append(1)
            i += 1
        # then append reps
        i = 0
        while i < rlen:
            var v = reps[i]
            if v < 0: v = 0
            out.append(v)
            i += 1
        return out.copy()

    # rlen >= rank → tail-align: take the last `rank` entries
    var start = rlen - rank
    var i = 0
    while i < rank:
        var v = reps[start + i]
        if v < 0: v = 0
        out.append(v)
        i += 1
    return out.copy()

# -----------------------------------------------------------------------------
# Repeat (tile) values along each dimension.
# - reps[d] == k replicates size along dim d by factor k
# - reps shorter than rank is left-padded with 1s
# - For rank==0 (scalar), result rank becomes len(reps) with shape==reps
# -----------------------------------------------------------------------------
@always_inline
fn repeat[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], reps: List[Int]
) -> Tensor[T]:
    var rank = len(x._shape)

    # Fast path: nothing to do
    if rank == 0 and len(reps) == 0:
        return x.copy()
    if rank > 0 and len(reps) == 0:
        return x.copy()

    var nreps = _normalize_reps(rank, reps)

    # Build output shape
    var out_shape = List[Int]()
    if rank == 0:
        # Scalar: shape is the reps themselves
        var i = 0
        while i < len(nreps):
            out_shape.append(nreps[i])
            i += 1
    else:
        var d = 0
        while d < rank:
            var r = nreps[d]
            var dim = x._shape[d]
            # If r==0 the output size becomes 0 along this dim
            out_shape.append(dim * r)
            d += 1

    # Compute counts
    var out_numel: Int = 1
    var i = 0
    while i < len(out_shape):
        out_numel = out_numel * out_shape[i]
        i += 1

    # Early exit for any zero-sized dimension
    if out_numel == 0:
        # Build an empty tensor with the right shape
        var strides = compute_row_major_strides(out_shape)
        var data = List[T]()            # empty data
        return Tensor[T](data, out_shape, strides, 0)

    # Prepare strides (row-major) for output to decode multi-index
    var out_strides = compute_row_major_strides(out_shape)

    # Allocate output; we'll append in a single pass
    var out_data = List[T]()
    out_data.reserve(out_numel)

    if rank == 0:
        # Tile the scalar value to the whole output
        var val = x._data[x._offset]
        var k = 0
        # unrolled append in chunks of 8 for speed
        var lim = (out_numel // 8) * 8
        while k < lim:
            out_data.append(val); out_data.append(val); out_data.append(val); out_data.append(val)
            out_data.append(val); out_data.append(val); out_data.append(val); out_data.append(val)
            k += 8
        while k < out_numel:
            out_data.append(val)
            k += 1
        return Tensor[T](out_data, out_shape, out_strides, 0)

    # General case (rank >= 1):
    # Map each output index to input index via modulo on each dimension.
    var in_shape = x._shape.copy()
    var in_strides = x._strides.copy()
    var base_off = x._offset

    var k = 0
    while k < out_numel:
        # Decode k into multi-index using out_strides
        var rem = k
        var in_lin = base_off

        var d = 0
        while d < rank:
            var step = out_strides[d]
            # idx along dim d in output
            var idx_out = rem // step
            rem = rem - idx_out * step
            # fold back into input range
            var idx_in =(idx_out % in_shape[d])
            if in_shape[d] == 0: dx_in =0
            in_lin = in_lin + idx_in * in_strides[d]
            d += 1
        # Append the corresponding input element
        out_data.append(x._data[in_lin])

        k += 1

    return Tensor[T](out_data, out_shape, out_strides, 0)

# Optional: NumPy-style alias
@always_inline
fn tile[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], reps: List[Int]
) -> Tensor[T]:
    return x.repeat(reps)


@always_inline
fn _swap_perm(rank: Int, a: Int, b: Int) -> List[Int]:
    var r = rank
    var i = a; var j = b
    # clamp_axis(i, r) و clamp_axis(j, r) را قبلاً داری
    i = clamp_axis(i, r)
    j = clamp_axis(j, r)
    var perm = List[Int]()
    var k = 0
    while k < r:
        perm.append(k)
        k += 1
    if i != j and r > 1:
        var tmp = perm[i]
        perm[i] = perm[j]
        perm[j] = tmp
    return perm.copy()
 
@always_inline
fn transpose[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], i: Int, j: Int) -> Tensor[T]:
    var r = len(x._shape)
    if r <= 1 or i == j:
        return x.copy()

    var ii = _clamp_axis(i, r)
    var jj = _clamp_axis(j, r)

    var perm = List[Int]()
    var k = 0
    while k < r:
        perm.append(k)
        k += 1

    var tmp = perm[ii]
    perm[ii] = perm[jj]
    perm[jj] = tmp

    return permute(x, perm)
 
@always_inline
fn transpose[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], perm: List[Int]) -> Tensor[T]: 
    return x.permute(perm)

@always_inline
fn _norm_axis_safe_inclusive(ax: Int, rank: Int) -> Int:
    # Normalize negatives; clamp into [0, rank-1]
    var i = ax
    if i < 0:
        i = i + rank
    if i < 0:
        i = 0
    if i >= rank:
        i = rank - 1
    return i

@always_inline
fn flatten[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    start_dim: Int = 0,
    end_dim_opt: Optional[Int] = None
) -> Tensor[T]:
    var r = len(x._shape)
    if r == 0:
        # Scalar: nothing to flatten
        return x.copy()

    # Safe normalize start/end (no assert)
    var sd = _norm_axis_safe_inclusive(start_dim, r)
    var ed: Int
    if end_dim_opt is None:
        ed = r - 1
    else:
        ed = _norm_axis_safe_inclusive(end_dim_opt.value(), r)

    # If range inverted, no-op
    if sd > ed:
        return x.copy()

    # Build new shape: keep dims [0..sd-1], merge [sd..ed], keep [ed+1..]
    var new_shape = List[Int]()
    var k = 0
    while k < sd:
        new_shape.append(x._shape[k])
        k += 1

    var merged = 1
    k = sd
    while k <= ed:
        merged = merged * x._shape[k]
        k += 1
    new_shape.append(merged)

    k = ed + 1
    while k < r:
        new_shape.append(x._shape[k])
        k += 1

    return x.reshape(new_shape)
