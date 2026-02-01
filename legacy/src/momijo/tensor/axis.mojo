# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       tensor.axis
# File:         src/momijo/tensor/axis.mojo
#
# Description:
#   Axis ops as free functions: moveaxis (single/multi), swapaxes, roll (single/multi).
#   Includes fast row-major paths and minimal allocations. Also provides a small,
#   self-contained reduction/arg-reduction toolkit used by some ops.
#
# Notes:
#   - No 'let' and no 'assert'.
#   - English-only comments.
#   - Integer division uses '//' per project rules.

from collections.list import List
from momijo.tensor.tensor import Tensor

# Helpers (explicit imports; no wildcards)
from momijo.tensor.helpers import identity_perm,normalize_axis
from momijo.tensor.helpers import row_major_strides
from momijo.tensor.helpers import is_row_major_contiguous
from momijo.tensor.helpers import unravel_index
from momijo.tensor.helpers import lin_index
from momijo.tensor.helpers import numel
from momijo.tensor.helpers import shape_drop_axis as _shape_drop_axis  # may be missing in some trees

# -------------------------------- local light helpers --------------------------------

@always_inline
fn isnan_f64(x: Float64) -> Bool:
    return x != x

# Robust Float64 sqrt via Newton-Raphson (few iterations are enough for normalization)
@always_inline
fn sqrt64(x: Float64) -> Float64:
    var v = x
    if v <= 0.0:
        return 0.0
    var g = v
    if g < 1.0:
        g = 1.0
    var i = 0
    while i < 6:
        g = 0.5 * (g + v / g)
        i += 1
    return g
    
 

@always_inline
fn normalize_axes(axs: List[Int], rank: Int) -> List[Int]:
    var out = List[Int]()
    out.reserve(len(axs))
    var i = 0
    while i < len(axs):
        out.append(normalize_axis(axs[i], rank))
        i = i + 1
    return out.copy()


# Multi-axis drop (stable ascending handling)
fn shape_drop_axes(shape: List[Int], axes: List[Int]) -> List[Int]:
    if len(shape) == 0:
        return List[Int]()
    if len(axes) == 0:
        var cp = List[Int]()
        cp.reserve(len(shape))
        var i0 = 0
        while i0 < len(shape):
            cp.append(shape[i0])
            i0 += 1
        return cp.copy()

    # normalize + uniqueness
    var r = len(shape)
    var norm = List[Int]()
    norm.reserve(len(axes))
    var i = 0
    while i < len(axes):
        var a = normalize_axis(axes[i], r)
        var seen = False
        var j = 0
        while j < len(norm):
            if norm[j] == a:
                seen = True
                break
            j += 1
        if not seen:
            norm.append(a)
        i += 1

    # insertion sort (ascending)
    var k = 1
    while k < len(norm):
        var key = norm[k]
        var p = k - 1
        while p >= 0 and norm[p] > key:
            norm[p + 1] = norm[p]
            p -= 1
        norm[p + 1] = key
        k += 1

    # build dropped shape
    var out = List[Int]()
    out.reserve(r - len(norm) if r > len(norm) else 1)
    var d = 0
    var cur = 0
    while d < r:
        if cur < len(norm) and norm[cur] == d:
            cur += 1
        else:
            out.append(shape[d])
        d += 1
    if len(out) == 0:
        out.append(1)
    return out

@always_inline
fn drop_axes(shape: List[Int], axes: List[Int]) -> List[Int]:
    # Prefer helper if wired; fallback to local
    return _shape_drop_axis(shape, axes[0]) if len(axes) == 1 else shape_drop_axes(shape, axes)

# -------------------------------- moveaxis / moveaxes / swapaxes ------------------------------
 
# Single-axis move: works for any rank >= 1
fn moveaxis[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], src: Int, dst: Int) -> Tensor[T]:
    var r = len(x._shape)
    if r <= 1:
        return x.copy()

    var s = normalize_axis(src, r)
    var d = normalize_axis(dst, r)
    if s == d:
        return x.copy()

    # Build order = [0,1,...,r-1] without s
    var order_wo = List[Int]()
    order_wo.reserve(r - 1)
    var i = 0
    while i < r:
        if i != s:
            order_wo.append(i)
        i += 1

    # Insert s at index d (after removal). d can be == len(order_wo) to mean "at end".
    var perm = List[Int]()
    perm.reserve(r)
    var j = 0
    while j < len(order_wo):
        if j == d:
            perm.append(s)
        perm.append(order_wo[j])
        j += 1
    if len(perm) < r:
        # If d == len(order_wo), append s at the end
        perm.append(s)

    return x.transpose(perm)

# ---------- Multi-axis variant (NumPy-like): moveaxis(x, [s1,...], [d1,...]) ----------

@always_inline
fn _contains(xs: List[Int], v: Int) -> Bool:
    var i = 0
    while i < len(xs):
        if xs[i] == v:
            return True
        i += 1
    return False

fn moveaxis[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], srcs: List[Int], dsts: List[Int]) -> Tensor[T]:
    var r = len(x._shape)
    if r <= 1:
        return x.copy()

    # Validate lengths
    if len(srcs) != len(dsts):
        # Fallback: identity transpose (no-op copy) to avoid throwing
        var id = List[Int]()
        id.reserve(r)
        var k = 0
        while k < r:
            id.append(k)
            k += 1
        return x.transpose(id)

    # Normalize axes
    var S = List[Int]()
    var D = List[Int]()
    S.reserve(len(srcs))
    D.reserve(len(dsts))

    var i = 0
    while i < len(srcs):
        S.append(normalize_axis(srcs[i], r))
        D.append(normalize_axis(dsts[i], r))
        i += 1

    # Build placeholder permutation of length r, -1 means empty
    var perm = List[Int]()
    perm.reserve(r)
    var t = 0
    while t < r:
        perm.append(-1)
        t += 1

    # Mark which axes are being moved
    var moving = List[Int]()
    moving.reserve(len(S))
    i = 0
    while i < len(S):
        moving.append(S[i])
        i += 1

    # Remaining axes in natural order (those not in S)
    var others = List[Int]()
    others.reserve(r - len(S))
    var a = 0
    while a < r:
        if not _contains(moving, a):
            others.append(a)
        a += 1

    # Place moved axes at their destination positions (final positions)
    i = 0
    while i < len(S):
        perm[D[i]] = S[i]
        i += 1

    # Fill remaining holes with 'others' in order
    var oi = 0
    var p = 0
    while p < r:
        if perm[p] == -1:
            perm[p] = others[oi]
            oi += 1
        p += 1

    return x.transpose(perm)


fn swapaxes[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], a: Int, b: Int) -> Tensor[T]:
    var r = len(x._shape)
    if r <= 1:
        return x.copy()
    var ax = normalize_axis(a, r)
    var bx = normalize_axis(b, r)
    if ax == bx:
        return x.copy()
    var perm = identity_perm(r)
    var tmp = perm[ax]
    perm[ax] = perm[bx]
    perm[bx] = tmp
    return x.transpose(perm)

# -------------------------------- roll (single/multi/flat) --------------------------------

@always_inline
fn _copy_list_int(xs: List[Int]) -> List[Int]:
    var out = List[Int]()
    out.reserve(len(xs))
    var i = 0
    while i < len(xs):
        out.append(xs[i])
        i += 1
    return out.copy()

 
# Fill a freshly reserved List[T] with copies of a value (T is copyable)
@always_inline
fn _fill_list[T: ImplicitlyCopyable & Copyable & Movable](mut dst: List[T], n: Int, value: T) -> None:
    var i = 0
    while i < n:
        dst.append(value)
        i += 1

# ---------------------- core roll ----------------------

# Roll along a single axis (keeps shape, respects true strides).
# Fast path when tensor is row-major contiguous and rolling the last dimension.
fn roll_one[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], ax: Int, sft: Int) -> Tensor[T]:
    var r = len(x._shape)
    if r == 0:
        return x.copy()

    var shape = _copy_list_int(x._shape)
    var s = normalize_axis(ax, r)
    var dim_ax = shape[s]
    if dim_ax == 0:
        return x.copy()

    var shift = sft % dim_ax
    if shift < 0:
        shift = shift + dim_ax
    if shift == 0:
        return x.copy()

    # Fast path: row-major and rolling the last dimension
    if is_row_major_contiguous(shape, x._strides) and s == r - 1:
        var total = numel(shape)
        var n = dim_ax

        # outer = product of dims except last
        var outer = 1
        var i0 = 0
        while i0 < r - 1:
            outer = outer * shape[i0]
            i0 += 1

        var out_data = List[T]()
        out_data.reserve(total)
        if total > 0:
            _fill_list(out_data, total, x._data[0])

        var block = n
        var base = 0
        var u = 0
        while u < outer:
            var cut = shift
            var tail = cut
            var head = n - cut

            # copy tail
            var j = 0
            var src_base_tail = base + head
            while j < tail:
                out_data[base + j] = x._data[src_base_tail + j]
                j += 1

            # copy head (unrolled 8x)
            var k = 0
            var dst_head = base + tail
            var lim = (head // 8) * 8
            while k < lim:
                out_data[dst_head + k    ] = x._data[base + k    ]
                out_data[dst_head + k + 1] = x._data[base + k + 1]
                out_data[dst_head + k + 2] = x._data[base + k + 2]
                out_data[dst_head + k + 3] = x._data[base + k + 3]
                out_data[dst_head + k + 4] = x._data[base + k + 4]
                out_data[dst_head + k + 5] = x._data[base + k + 5]
                out_data[dst_head + k + 6] = x._data[base + k + 6]
                out_data[dst_head + k + 7] = x._data[base + k + 7]
                k += 8
            while k < head:
                out_data[dst_head + k] = x._data[base + k]
                k += 1

            base = base + block
            u += 1
        return Tensor[T](out_data, shape)

    # General path: respect true (possibly non-contiguous) strides.
    var in_strides = _copy_list_int(x._strides)
    var out_strides = row_major_strides(shape)  # we materialize as row-major

    var outer_rank = r - 1
    var outer_shape = List[Int]()
    var outer_in_strides = List[Int]()
    var outer_out_strides = List[Int]()
    outer_shape.reserve(outer_rank)
    outer_in_strides.reserve(outer_rank)
    outer_out_strides.reserve(outer_rank)

    var d = 0
    while d < r:
        if d != s:
            outer_shape.append(shape[d])
            outer_in_strides.append(in_strides[d])
            outer_out_strides.append(out_strides[d])
        d += 1

    var outer_rm = row_major_strides(outer_shape)
    var outer_total = 1
    var oi = 0
    while oi < len(outer_shape):
        outer_total = outer_total * outer_shape[oi]
        oi += 1

    var total = numel(shape)
    var out_data = List[T]()
    out_data.reserve(total)
    if total > 0:
        _fill_list(out_data, total, x._data[0])

    # Iterate all coordinates except along axis s, using row-major on outer_shape
    var u2 = 0
    while u2 < outer_total:
        # unravel u2 in outer_shape with outer_rm
        var rem = u2
        var base_in = 0
        var base_out = 0

        var pos_outer = 0
        var q = 0
        while q < r:
            if q == s:
                q += 1
                continue
            var st = outer_rm[pos_outer]
            var dim = outer_shape[pos_outer]
            var coord = 0
            if st != 0 and dim != 0:
                coord = (rem // st) % dim
            base_in = base_in + coord * in_strides[q]
            base_out = base_out + coord * out_strides[q]
            pos_outer += 1
            q += 1

        # walk along axis s
        var j2 = 0
        while j2 < dim_ax:
            var src_ax = j2 - shift
            if src_ax < 0:
                src_ax = src_ax + dim_ax
            var src_off = base_in + src_ax * in_strides[s]
            var dst_off = base_out + j2 * out_strides[s]
            out_data[dst_off] = x._data[src_off]
            j2 += 1

        u2 += 1

    return Tensor[T](out_data, shape)

# Roll with single axis
fn roll[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], shift: Int, axis: Int) -> Tensor[T]:
    var r = len(x._shape)
    if r == 0:
        return x.copy()

    var ax = normalize_axis(axis, r)
    if r == 1:
        var n = x._shape[0]
        if n == 0:
            return x.copy()
        var sft = shift % n
        if sft < 0:
            sft = sft + n
        if sft == 0:
            return x.copy()
        var out_data = List[T]()
        out_data.reserve(n)
        var i = 0
        while i < n:
            var src = i - sft
            if src < 0:
                src = src + n
            out_data.append(x._data[src])
            i += 1
        return Tensor[T](out_data, [n])

    return roll_one[T](x, ax, shift)

# Flat roll (axis=None) — rolls raveled order but keeps original shape
fn roll_flat[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], shift: Int) -> Tensor[T]:
    var n = numel(x._shape)
    if n == 0:
        return x.copy()

    var sft = shift % n
    if sft < 0:
        sft = sft + n
    if sft == 0:
        return x.copy()

    # Logical row-major ravel mapping -> back to physical offsets
    var r = len(x._shape)
    var strides_rm = row_major_strides(x._shape)
    var out = List[T]()
    out.reserve(n)
    _fill_list(out, n, x._data[0])

    var pos = 0
    while pos < n:
        var src_lin = pos - sft
        if src_lin < 0:
            src_lin = src_lin + n

        # unravel src_lin using row-major logical strides
        var rem2 = src_lin
        var src_off = 0
        var i = 0
        while i < r:
            var st2 = strides_rm[i]
            var dim2 = x._shape[i]
            var v2 = 0
            if st2 != 0 and dim2 != 0:
                v2 = (rem2 // st2) % dim2
            src_off = src_off + v2 * x._strides[i]
            i += 1

        out[pos] = x._data[src_off]
        pos += 1
    return Tensor[T](out, x._shape)

# Roll with multiple axes (NumPy-like)
# Applies each (axis_i, shift_i) sequentially; order is the list order.
fn roll_multi[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], shifts: List[Int], axes: List[Int]) -> Tensor[T]:
    var r = len(x._shape)
    if r == 0:
        return x.copy()

    if len(shifts) == 0 or len(shifts) != len(axes):
        return x.copy()

    var axn = normalize_axes(axes, r)
    var out = x.copy()
    var i = 0
    while i < len(shifts):
        out = roll_one[T](out.copy(), axn[i], shifts[i])
        i += 1
    return out.copy()

# Convenience overloads (optional to re-export as tensor.roll)
@always_inline
fn roll[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], shift: Int) -> Tensor[T]:
    # flat roll (axis=None)
    return roll_flat[T](x, shift)

@always_inline
fn roll[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], shifts: List[Int], axes: List[Int]) -> Tensor[T]:
    return roll_multi[T](x, shifts, axes)
# -------------------------------- minimal reductions used here --------------------------------

@always_inline
fn sum1d_contig[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Float64:
    var n = len(x._data)
    var s = 0.0
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        s = s + Float64(x._data[i    ])
        s = s + Float64(x._data[i + 1])
        s = s + Float64(x._data[i + 2])
        s = s + Float64(x._data[i + 3])
        s = s + Float64(x._data[i + 4])
        s = s + Float64(x._data[i + 5])
        s = s + Float64(x._data[i + 6])
        s = s + Float64(x._data[i + 7])
        i += 8
    while i < n:
        s = s + Float64(x._data[i])
        i += 1
    return s

# ------------------------------- generic reduce engine (float64 accumulator) -------------------------------

struct ReduceSpec:
    var op_id: Int      # 0=sum, 1=min, 2=max, 3=mean, 4=prod
    var nan_mode: Int   # 0=normal, 1=skip-NaN
    var has_mask: Bool  # reserved (mask handled by caller)

    fn __init__(out self, op_id: Int, nan_mode: Int, has_mask: Bool):
        self.op_id = op_id
        self.nan_mode = nan_mode
        self.has_mask = has_mask

@always_inline
fn neutral(spec: ReduceSpec) -> Float64:
    # Avoid reliance on ±Infinity; 'inited' flag handles min/max first-sample adoption
    if spec.op_id == 4:
        return 1.0   # prod
    return 0.0       # sum/mean/min/max (min/max ignore this until first sample)

@always_inline
fn apply_acc(acc: Float64, v: Float64, spec: ReduceSpec, inited: Bool) -> (Float64, Bool):
    var out = acc
    var got = inited
    if spec.nan_mode == 1 and isnan_f64(v):
        return (out, got)
    if spec.op_id == 0:
        out = acc + v
        got = True
    elif spec.op_id == 1:
        if not got:
            out = v
            got = True
        elif v < out:
            out = v
    elif spec.op_id == 2:
        if not got:
            out = v
            got = True
        elif v > out:
            out = v
    elif spec.op_id == 3:
        out = acc + v
        got = True
    else:
        if not got:
            out = v
            got = True
        else:
            out = acc * v
    return (out, got)

# Reduce along a single axis; returns (accumulated, out_shape, out_strides_rowmajor)
fn reduce_along_axis[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    ax: Int,
    spec: ReduceSpec,
    mask: Optional[Tensor[UInt8]] = None,
) -> (List[Float64], List[Int], List[Int]):
    var r = len(x._shape)
    var axn = normalize_axis(ax, r)

    # out shape with keepdims=True for stride stability
    var out_shape = List[Int]()
    out_shape.reserve(r)
    var d0 = 0
    while d0 < r:
        out_shape.append(1 if d0 == axn else x._shape[d0])
        d0 += 1
    var out_rm = row_major_strides(out_shape)
    var out_n = numel(out_shape)

    var out = List[Float64]()
    out.reserve(out_n)
    var rm_x = x._strides.copy()

    var use_mask = not (mask is None)
    var rm_m = List[Int]()
    if use_mask:
        rm_m = mask.value()._strides

    var base_idx = List[Int]()
    var oi = 0
    while oi < out_n:
        unravel_index(oi, out_shape, base_idx)
        # build full index with axis position=0 then scan axis
        var full = List[Int]()
        full.reserve(r)
        var d = 0
        var bi = 0
        while d < r:
            if d == axn:
                full.append(0)
            else:
                full.append(base_idx[bi])
                bi += 1
            d += 1

        var acc = neutral(spec)
        var inited = False
        var count = 0
        var k = 0
        var K = x._shape[axn]
        while k < K:
            full[axn] = k
            var li = lin_index(full, rm_x)
            var use_it = True
            if use_mask:
                var lm = lin_index(full, rm_m)
                use_it = (mask.value()._data[lm] != 0)
            if use_it:
                var vv = Float64(x._data[li])
                var res = apply_acc(acc, vv, spec, inited)
                acc = res[0]
                inited = res[1]
                count += 1
            k += 1
        if spec.op_id == 3:
            var denom = Float64(count if count > 0 else 1)
            acc = acc / denom
        out.append(acc)
        oi += 1

    return (out, out_shape, out_rm)

fn reduce_multi_axis[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    axes: List[Int],
    keepdims: Bool,
    spec: ReduceSpec,
    mask: Optional[Tensor[UInt8]] = None,
) -> Tensor[T]:
    var r = len(x._shape)
    if len(axes) == 0:
        var all = List[Int]()
        all.reserve(r)
        var i0 = 0
        while i0 < r:
            all.append(i0)
            i0 += 1
        return reduce_multi_axis[T](x, all, keepdims, spec, mask)

    # normalize + unique + sort asc; then reduce from last to first
    var axn = normalize_axes(axes, r)
    var uniq = List[Int]()
    var i = 0
    while i < len(axn):
        var a = axn[i]
        var seen = False
        var j = 0
        while j < len(uniq):
            if uniq[j] == a:
                seen = True
                break
            j += 1
        if not seen:
            uniq.append(a)
        i += 1
    var k = 1
    while k < len(uniq):
        var key = uniq[k]
        var p = k - 1
        while p >= 0 and uniq[p] > key:
            uniq[p + 1] = uniq[p]
            p -= 1
        uniq[p + 1] = key
        k += 1
    var rev = List[Int]()
    rev.reserve(len(uniq))
    var t = len(uniq) - 1
    while t >= 0:
        rev.append(uniq[t])
        t -= 1

    var cur = x
    var q = 0
    while q < len(rev):
        var a2 = rev[q]
        var (accs, out_shape_k, _) = reduce_along_axis[T](cur, a2, spec, mask)
        var next_shape = out_shape_k
        if not keepdims:
            next_shape = shape_drop_axes(out_shape_k, [a2])
        var casted = List[T]()
        casted.reserve(len(accs))
        var i2 = 0
        while i2 < len(accs):
            casted.append(T(accs[i2]))
            i2 += 1
        cur = Tensor[T](casted, next_shape)
        q += 1
    return cur

# Public reduction API
fn axis_to_list(axis: Optional[Int], axes_list: Optional[List[Int]], rank: Int) -> List[Int]:
    var axes = List[Int]()
    if not (axes_list is None):
        var xs = axes_list.value()
        var i = 0
        while i < len(xs):
            axes.append(xs[i])
            i += 1
        return normalize_axes(axes, rank)
    if axis is None:
        var i2 = 0
        while i2 < rank:
            axes.append(i2)
            i2 += 1
        return axes
    axes.append(normalize_axis(axis.value(), rank))
    return axes

# sum / min / max / mean / prod

fn reduce_sum[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None
) -> Tensor[T]:
    var r = len(x._shape)
    if r == 0:
        return Tensor[T]([zero_scalar_of[T]()], [1])

    if (axis is None) and (axes is None):
        if r == 1 and is_row_major_contiguous(x._shape, x._strides):
            var s = sum1d_contig(x)
            return Tensor[T]([T(s)], [1])
        var s2 = 0.0
        var i = 0
        var n = len(x._data)
        while i < n:
            s2 = s2 + Float64(x._data[i])
            i += 1
        return Tensor[T]([T(s2)], [1])

    var axlist = axis_to_list(axis, axes, r)
    var spec = ReduceSpec(0, 0, False)
    return reduce_multi_axis[T](x, axlist, keepdims, spec, None)

fn reduce_min[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None) -> Tensor[T]:
    var spec = ReduceSpec(1, 0, False)
    return reduce_multi_axis[T](x, axis_to_list(axis, axes, len(x._shape)), keepdims, spec, None)

fn reduce_max[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None) -> Tensor[T]:
    var spec = ReduceSpec(2, 0, False)
    return reduce_multi_axis[T](x, axis_to_list(axis, axes, len(x._shape)), keepdims, spec, None)

fn reduce_mean[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None) -> Tensor[T]:
    var spec = ReduceSpec(3, 0, False)
    return reduce_multi_axis[T](x, axis_to_list(axis, axes, len(x._shape)), keepdims, spec, None)

fn reduce_prod[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None) -> Tensor[T]:
    var spec = ReduceSpec(4, 0, False)
    return reduce_multi_axis[T](x, axis_to_list(axis, axes, len(x._shape)), keepdims, spec, None)

# NaN-aware variants

fn nan_sum[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None) -> Tensor[T]:
    var spec = ReduceSpec(0, 1, False)
    return reduce_multi_axis[T](x, axis_to_list(axis, axes, len(x._shape)), keepdims, spec, None)

fn nan_min[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None) -> Tensor[T]:
    var spec = ReduceSpec(1, 1, False)
    return reduce_multi_axis[T](x, axis_to_list(axis, axes, len(x._shape)), keepdims, spec, None)

fn nan_max[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None) -> Tensor[T]:
    var spec = ReduceSpec(2, 1, False)
    return reduce_multi_axis[T](x, axis_to_list(axis, axes, len(x._shape)), keepdims, spec, None)

fn nan_mean[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None) -> Tensor[T]:
    var spec = ReduceSpec(3, 1, False)
    return reduce_multi_axis[T](x, axis_to_list(axis, axes, len(x._shape)), keepdims, spec, None)

fn nan_prod[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None) -> Tensor[T]:
    var spec = ReduceSpec(4, 1, False)
    return reduce_multi_axis[T](x, axis_to_list(axis, axes, len(x._shape)), keepdims, spec, None)

# ------------------------------- arg reduce / minmax -------------------------------

# mode: 0=argmin, 1=argmax
fn arg_reduce[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, axes: Optional[List[Int]] = None, mode: Int = 1) -> Tensor[Int]:
    var r = len(x._shape)
    if (axis is None) and (axes is None):
        var n = len(x._data)
        var best = 0
        var bestv = Float64(0.0)
        var inited = False
        var i = 0
        while i < n:
            var v = Float64(x._data[i])
            if not inited:
                best = i
                bestv = v
                inited = True
            else:
                if mode == 0:
                    if v < bestv:
                        best = i
                        bestv = v
                else:
                    if v > bestv:
                        best = i
                        bestv = v
            i += 1
        return Tensor[Int]([best], [1])

    var axlist = axis_to_list(axis, axes, r)
    var ax = axlist[0]
    var shp = x._shape.copy()
    var out_shape = shape_drop_axes(shp, [ax])
    var out_n = numel(out_shape)
    var rm = row_major_strides(shp)
    var base_idx = List[Int]()
    var idxs = List[Int]()
    idxs.reserve(out_n)

    var k = 0
    while k < out_n:
        unravel_index(k, out_shape, base_idx)
        var full = List[Int]()
        full.reserve(len(shp))
        var p = 0
        var d = 0
        while d < len(shp):
            if d == ax:
                full.append(0)
            else:
                full.append(base_idx[p])
                p += 1
            d += 1
        var besti = 0
        var bestv2 = Float64(0.0)
        var inited2 = False
        var s = 0
        var L = shp[ax]
        while s < L:
            full[ax] = s
            var v2 = Float64(x._data[lin_index(full, rm)])
            if not inited2:
                besti = s
                bestv2 = v2
                inited2 = True
            else:
                if mode == 0:
                    if v2 < bestv2:
                        besti = s
                        bestv2 = v2
                else:
                    if v2 > bestv2:
                        besti = s
                        bestv2 = v2
            s += 1
        idxs.append(besti)
        k += 1
    return Tensor[Int](idxs, out_shape)

fn nanargmin[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, axes: Optional[List[Int]] = None) -> Tensor[Int]:
    var r = len(x._shape)
    if (axis is None) and (axes is None):
        var n = len(x._data)
        var best = 0
        var bestv = Float64(0.0)
        var inited = False
        var i = 0
        while i < n:
            var v = Float64(x._data[i])
            if not isnan_f64(v):
                if not inited:
                    best = i
                    bestv = v
                    inited = True
                elif v < bestv:
                    best = i
                    bestv = v
            i += 1
        return Tensor[Int]([best], [1])
    return arg_reduce[T](x, axis, axes, 0)

fn nanargmax[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, axes: Optional[List[Int]] = None) -> Tensor[Int]:
    var r = len(x._shape)
    if (axis is None) and (axes is None):
        var n = len(x._data)
        var best = 0
        var bestv = Float64(0.0)
        var inited = False
        var i = 0
        while i < n:
            var v = Float64(x._data[i])
            if not isnan_f64(v):
                if not inited:
                    best = i
                    bestv = v
                    inited = True
                elif v > bestv:
                    best = i
                    bestv = v
            i += 1
        return Tensor[Int]([best], [1])
    return arg_reduce[T](x, axis, axes, 1)

fn minmax[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[Float64]:
    var n = len(x._data)
    var lo = 0.0
    var hi = 0.0
    if n > 0:
        lo = Float64(x._data[0])
        hi = Float64(x._data[0])
        var i = 1
        while i < n:
            var v = Float64(x._data[i])
            if v < lo:
                lo = v
            if v > hi:
                hi = v
            i += 1
    var out = List[Float64]()
    out.reserve(2)
    out.append(lo)
    out.append(hi)
    return Tensor[Float64](out, [2])

# ------------------------------- cumulative ops -------------------------------

fn cumsum[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Int) -> Tensor[T]:
    var shp = x._shape.copy()
    var r = len(shp)
    var ax = normalize_axis(axis, r)
    var L = shp[ax]
    var inner = 1
    var i_in = ax + 1
    while i_in < r:
        inner = inner * shp[i_in]
        i_in += 1
    var out = List[T]()
    out.reserve(len(x._data))
    var base_stride = inner
    var block = L * inner
    var outer = numel(shape_drop_axes(shp, [ax]))
    var o = 0
    while o < outer:
        var base = (o // inner) * block + (o % inner)
        var acc = 0.0
        var k = 0
        while k < L:
            acc = acc + Float64(x._data[base + k * base_stride])
            out.append(T(acc))
            k += 1
        o += 1
    return Tensor[T](out, shp)

fn cumprod[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Int) -> Tensor[T]:
    var shp = x._shape.copy()
    var r = len(shp)
    var ax = normalize_axis(axis, r)
    var L = shp[ax]
    var inner = 1
    var i_in = ax + 1
    while i_in < r:
        inner = inner * shp[i_in]
        i_in += 1
    var out = List[T]()
    out.reserve(len(x._data))
    var base_stride = inner
    var block = L * inner
    var outer = numel(shape_drop_axes(shp, [ax]))
    var o = 0
    while o < outer:
        var base = (o // inner) * block + (o % inner)
        var acc = 1.0
        var k = 0
        while k < L:
            acc = acc * Float64(x._data[base + k * base_stride])
            out.append(T(acc))
            k += 1
        o += 1
    return Tensor[T](out, shp)

# ------------------------------- quantile / percentile / median -------------------------------

@always_inline
fn partition(mut a: List[Float64], l: Int, r: Int, pivot: Float64) -> (Int, Int):
    var i = l
    var lt = l
    var gt = r
    while i <= gt:
        var v = a[i]
        if v < pivot:
            var t0 = a[i]
            a[i] = a[lt]
            a[lt] = t0
            lt += 1
            i += 1
        elif v > pivot:
            var t1 = a[i]
            a[i] = a[gt]
            a[gt] = t1
            gt -= 1
        else:
            i += 1
    return (lt, gt)

fn select_kth(mut a: List[Float64], k: Int) -> Float64:
    var l = 0
    var r = len(a) - 1
    while l <= r:
        var mid = (l + r) // 2
        var pivot = a[mid]
        var (lt, gt) = partition(a, l, r, pivot)
        if k < lt:
            r = lt - 1
        elif k > gt:
            l = gt + 1
        else:
            return pivot
    return a[k]

fn quantile_1d_partial(xs: List[Float64], q: Float64, interpolation: String) -> Float64:
    var n = len(xs)
    if n == 0:
        return 0.0
    var a = List[Float64]()
    a.reserve(n)
    var i = 0
    while i < n:
        a.append(xs[i])
        i += 1
    var qq = q
    if qq < 0.0:
        qq = 0.0
    if qq > 1.0:
        qq = 1.0
    var pos = qq * Float64(n - 1)
    var lo = Int(pos)
    var hi = lo + 1
    if hi >= n:
        hi = n - 1
    if interpolation == "lower":
        return select_kth(a, lo)
    if interpolation == "higher":
        return select_kth(a, hi)
    if interpolation == "midpoint":
        var alo = select_kth(a, lo)
        var ahi = select_kth(a, hi)
        return 0.5 * (alo + ahi)
    if interpolation == "nearest":
        var frac = pos - Float64(lo)
        var idx = hi
        if frac < 0.5:
            idx = lo
        return select_kth(a, idx)
    var alo2 = select_kth(a, lo)
    var ahi2 = select_kth(a, hi)
    var frac2 = pos - Float64(lo)
    return alo2 + (ahi2 - alo2) * frac2

fn quantile[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], q: Float64, axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None, interpolation: String = "linear"
) -> Tensor[T]:
    var xc = x.astype[Float64]()  # contiguous() optional
    var shp = xc._shape
    var r = len(shp)
    var axlist = axis_to_list(axis, axes, r)
    if len(axlist) == 0:
        var v = quantile_1d_partial(xc._data, q, interpolation)
        return Tensor[T]([T(v)], [1])
    var ax = normalize_axis(axlist[0], r)
    var L = shp[ax]
    var inner = 1
    var i_in = ax + 1
    while i_in < r:
        inner = inner * shp[i_in]
        i_in += 1
    var out_shape = shape_drop_axes(shp, [ax])
    if keepdims:
        var kd = List[Int]()
        kd.reserve(r)
        var i0 = 0
        while i0 < r:
            kd.append(1 if i0 == ax else shp[i0])
            i0 += 1
        out_shape = kd
    var outer = numel(shape_drop_axes(shp, [ax]))
    var out = List[T]()
    out.reserve(outer)
    var block = L * inner
    var base_stride = inner
    var buf = List[Float64]()
    var o = 0
    while o < outer:
        buf.clear()
        buf.reserve(L)
        var base = (o // inner) * block + (o % inner)
        var k = 0
        while k < L:
            buf.append(xc._data[base + k * base_stride])
            k += 1
        out.append(T(quantile_1d_partial(buf, q, interpolation)))
        o += 1
    return Tensor[T](out, out_shape)

fn percentile[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], p: Float64, axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None, interpolation: String = "linear") -> Tensor[T]:
    return quantile[T](x, p / 100.0, axis, keepdims, axes, interpolation)

fn median[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False, axes: Optional[List[Int]] = None) -> Tensor[T]:
    return quantile[T](x, 0.5, axis, keepdims, axes, "midpoint")

# ------------------------------- unique / bincount -------------------------------

fn unique[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None) -> Tensor[T]:
    # Axis-aware unique not implemented here; we use flattened semantics.
    var xc = x.astype[Float64]()
    var seen = List[Float64]()
    var out = List[Float64]()
    var n = len(xc._data)
    var i = 0
    while i < n:
        var v = xc._data[i]
        var found = False
        var j = 0
        var m = len(seen)
        while j < m:
            if seen[j] == v:
                found = True
                break
            j += 1
        if not found:
            seen.append(v)
            out.append(v)
        i += 1
    return Tensor[Float64](out, [len(out)]).astype[T]()

fn bincount(x: Tensor[Int], minlength: Int = 0) -> Tensor[Int]:
    var n = numel(x._shape)
    var data = x._data
    var maxv = 0
    var i = 0
    while i < n:
        if data[i] > maxv:
            maxv = data[i]
        i += 1
    var size = maxv + 1
    if size < minlength:
        size = minlength
    var cnt = List[Int]()
    cnt.reserve(size)
    var k = 0
    while k < size:
        cnt.append(0)
        k += 1
    var t = 0
    while t < n:
        var u = data[t]
        if u >= 0 and u < size:
            cnt[u] = cnt[u] + 1
        t += 1
    return Tensor[Int](cnt, [size])

# ------------------------------- normalize (L2) -------------------------------

fn normalize[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, eps: Float64 = 1e-12, axes: Optional[List[Int]] = None) -> Tensor[T]:
    var shp = x._shape.copy()
    var r = len(shp)
    var axlist = axis_to_list(axis, axes, r)
    if len(axlist) == 0:
        var n = len(x._data)
        var s2 = 0.0
        var i = 0
        var lim = (n // 8) * 8
        while i < lim:
            var v0 = Float64(x._data[i    ])
            s2 = s2 + v0 * v0
            var v1 = Float64(x._data[i + 1])
            s2 = s2 + v1 * v1
            var v2 = Float64(x._data[i + 2])
            s2 = s2 + v2 * v2
            var v3 = Float64(x._data[i + 3])
            s2 = s2 + v3 * v3
            var v4 = Float64(x._data[i + 4])
            s2 = s2 + v4 * v4
            var v5 = Float64(x._data[i + 5])
            s2 = s2 + v5 * v5
            var v6 = Float64(x._data[i + 6])
            s2 = s2 + v6 * v6
            var v7 = Float64(x._data[i + 7])
            s2 = s2 + v7 * v7
            i += 8
        while i < n:
            var v = Float64(x._data[i])
            s2 = s2 + v * v
            i += 1
        var inv = 1.0 / sqrt64(s2 + eps)
        var out = List[T]()
        out.reserve(n)
        var j = 0
        var lim2 = (n // 8) * 8
        while j < lim2:
            out.append(T(Float64(x._data[j    ]) * inv))
            out.append(T(Float64(x._data[j + 1]) * inv))
            out.append(T(Float64(x._data[j + 2]) * inv))
            out.append(T(Float64(x._data[j + 3]) * inv))
            out.append(T(Float64(x._data[j + 4]) * inv))
            out.append(T(Float64(x._data[j + 5]) * inv))
            out.append(T(Float64(x._data[j + 6]) * inv))
            out.append(T(Float64(x._data[j + 7]) * inv))
            j += 8
        while j < n:
            out.append(T(Float64(x._data[j]) * inv))
            j += 1
        return Tensor[T](out, shp)

    var cur = x
    var idx = 0
    while idx < len(axlist):
        var ax = normalize_axis(axlist[idx], len(cur._shape))
        var left = 1
        var i0 = 0
        while i0 < ax:
            left = left * cur._shape[i0]
            i0 += 1
        var mid = cur._shape[ax]
        var right = 1
        var i1 = ax + 1
        while i1 < len(cur._shape):
            right = right * cur._shape[i1]
            i1 += 1
        var n_total = left * mid * right
        var out = List[T]()
        out.reserve(n_total)
        var l = 0
        while l < left:
            var r2 = 0
            while r2 < right:
                var s2 = 0.0
                var m = 0
                while m < mid:
                    var idx2 = (l * mid + m) * right + r2
                    var vv = Float64(cur._data[idx2])
                    s2 = s2 + vv * vv
                    m += 1
                var inv = 1.0 / sqrt64(s2 + eps)
                m = 0
                while m < mid:
                    var idx3 = (l * mid + m) * right + r2
                    out.append(T(Float64(cur._data[idx3]) * inv))
                    m += 1
                r2 += 1
            l += 1
        cur = Tensor[T](out, cur._shape)
        idx += 1
    return cur

fn normalize_[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], axis: Optional[Int] = None, eps: Float64 = 1e-12, axes: Optional[List[Int]] = None) -> None:
    var t = normalize[T](x, axis, eps, axes)
    var n = len(t._data)
    var i = 0
    while i < n:
        x._data[i] = t._data[i]
        i += 1


