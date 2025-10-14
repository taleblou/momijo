# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.tensor.nanops
# File:         src/momijo/tensor/nanops.mojo
#
# Description:
#   Numerically-safe helpers and NaN/Inf-aware ops for Momijo Tensor.
#   - Scalar-safe helpers: safe_div, safe_log, safe_sqrt
#   - Masks: isnan / isinf / isfinite  -> Tensor[Int] (0/1)
#   - Elementwise transform: nan_to_num (generic; preserves dtype T)
#   - Reductions with axis/keepdims: nansum, nanmean, nanmin
#     * Generic wrappers accept Tensor[T], compute in Float64
#     * Return Float64 results; keepdims supported
#   - Contiguous fast path; otherwise uses contiguous() once
#
# Notes:
#   - No 'let' and no 'assert'.
#   - Tight unrolled loops on hot paths.
#   - English-only comments for portability.

from collections.list import List
from momijo.tensor.tensor import Tensor
from momijo.tensor.helpers import (
    is_row_major_contiguous,
    normalize_axis,
)
from momijo.tensor.math import (
    clone_header_share_data,
    contiguous,
)

# ============================== Utilities ===============================

@always_inline
fn _max_f64() -> Float64:
    return 1.7976931348623157e308

@always_inline
fn _min_f64() -> Float64:
    return -1.7976931348623157e308

@always_inline
fn _is_nan64(x: Float64) -> Int:
    # NaN is the only float that is not equal to itself
    return 1 if x != x else 0

@always_inline
fn _is_inf64(x: Float64) -> Int:
    # Infinity compares beyond any finite max/min
    if x == x:
        if x > _max_f64(): return 1
        if x < _min_f64(): return 1
    return 0

@always_inline
fn _is_finite64(x: Float64) -> Int:
    return 1 if (_is_nan64(x) == 0 and _is_inf64(x) == 0) else 0

@always_inline
fn _shape_without_axis(shape: List[Int], ax: Int, keepdims: Bool) -> List[Int]:
    var out = List[Int]()
    var r = len(shape)
    if keepdims:
        var i = 0
        while i < r:
            out.append(1 if i == ax else shape[i])
            i += 1
        return out
    var i2 = 0
    while i2 < r:
        if i2 != ax: out.append(shape[i2])
        i2 += 1
    if len(out) == 0:
        out.append(1)
    return out

@always_inline
fn _ensure_contig[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[T]:
    if is_row_major_contiguous(x._shape, x._strides):
        return clone_header_share_data[T](x, x._shape, x._strides)
    return contiguous(x)

@always_inline
fn _to_f64_tensor[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[Float64]:
    var n = len(x._data)
    var out = List[Float64](); out.reserve(n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(Float64(x._data[i + 0]))
        out.append(Float64(x._data[i + 1]))
        out.append(Float64(x._data[i + 2]))
        out.append(Float64(x._data[i + 3]))
        out.append(Float64(x._data[i + 4]))
        out.append(Float64(x._data[i + 5]))
        out.append(Float64(x._data[i + 6]))
        out.append(Float64(x._data[i + 7]))
        i += 8
    while i < n:
        out.append(Float64(x._data[i]))
        i += 1
    return Tensor[Float64](out, x._shape)

@always_inline
fn _from_f64_tensor[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[Float64]) -> Tensor[T]:
    var n = len(x._data)
    var out = List[T](); out.reserve(n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(T(x._data[i + 0]))
        out.append(T(x._data[i + 1]))
        out.append(T(x._data[i + 2]))
        out.append(T(x._data[i + 3]))
        out.append(T(x._data[i + 4]))
        out.append(T(x._data[i + 5]))
        out.append(T(x._data[i + 6]))
        out.append(T(x._data[i + 7]))
        i += 8
    while i < n:
        out.append(T(x._data[i]))
        i += 1
    return Tensor[T](out, x._shape)

# ============================ Safe helpers ==============================

@always_inline
fn safe_div_scalar(a: Float64, b: Float64, eps: Float64 = 1e-12) -> Float64:
    var den = b
    if den == 0.0:
        den = eps
    return a / den

@always_inline
fn safe_log_scalar(x: Float64, eps: Float64 = 1e-12) -> Float64:
    var v = x
    if v <= 0.0: v = eps
    return Float64.log(v)

@always_inline
fn safe_sqrt_scalar(x: Float64, eps: Float64 = 0.0) -> Float64:
    var v = x
    if v < eps: v = eps
    return Float64.sqrt(v)

fn safe_div(x: Tensor[Float64], y: Tensor[Float64], eps: Float64 = 1e-12) -> Tensor[Float64]:
    # Same-shape elementwise division with epsilon safeguard
    var n = len(x._data)
    var out = List[Float64](); out.reserve(n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(safe_div_scalar(x._data[i + 0], y._data[i + 0], eps))
        out.append(safe_div_scalar(x._data[i + 1], y._data[i + 1], eps))
        out.append(safe_div_scalar(x._data[i + 2], y._data[i + 2], eps))
        out.append(safe_div_scalar(x._data[i + 3], y._data[i + 3], eps))
        out.append(safe_div_scalar(x._data[i + 4], y._data[i + 4], eps))
        out.append(safe_div_scalar(x._data[i + 5], y._data[i + 5], eps))
        out.append(safe_div_scalar(x._data[i + 6], y._data[i + 6], eps))
        out.append(safe_div_scalar(x._data[i + 7], y._data[i + 7], eps))
        i += 8
    while i < n:
        out.append(safe_div_scalar(x._data[i], y._data[i], eps))
        i += 1
    return Tensor[Float64](out, x._shape)

fn safe_log(x: Tensor[Float64], eps: Float64 = 1e-12) -> Tensor[Float64]:
    var n = len(x._data)
    var out = List[Float64](); out.reserve(n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(safe_log_scalar(x._data[i + 0], eps))
        out.append(safe_log_scalar(x._data[i + 1], eps))
        out.append(safe_log_scalar(x._data[i + 2], eps))
        out.append(safe_log_scalar(x._data[i + 3], eps))
        out.append(safe_log_scalar(x._data[i + 4], eps))
        out.append(safe_log_scalar(x._data[i + 5], eps))
        out.append(safe_log_scalar(x._data[i + 6], eps))
        out.append(safe_log_scalar(x._data[i + 7], eps))
        i += 8
    while i < n:
        out.append(safe_log_scalar(x._data[i], eps))
        i += 1
    return Tensor[Float64](out, x._shape)

fn safe_sqrt(x: Tensor[Float64], eps: Float64 = 0.0) -> Tensor[Float64]:
    var n = len(x._data)
    var out = List[Float64](); out.reserve(n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(safe_sqrt_scalar(x._data[i + 0], eps))
        out.append(safe_sqrt_scalar(x._data[i + 1], eps))
        out.append(safe_sqrt_scalar(x._data[i + 2], eps))
        out.append(safe_sqrt_scalar(x._data[i + 3], eps))
        out.append(safe_sqrt_scalar(x._data[i + 4], eps))
        out.append(safe_sqrt_scalar(x._data[i + 5], eps))
        out.append(safe_sqrt_scalar(x._data[i + 6], eps))
        out.append(safe_sqrt_scalar(x._data[i + 7], eps))
        i += 8
    while i < n:
        out.append(safe_sqrt_scalar(x._data[i], eps))
        i += 1
    return Tensor[Float64](out, x._shape)

# Generic wrappers for safe helpers
fn safe_div[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], y: Tensor[T], eps: Float64 = 1e-12) -> Tensor[Float64]:
    return safe_div(_to_f64_tensor[T](x), _to_f64_tensor[T](y), eps)

fn safe_log[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], eps: Float64 = 1e-12) -> Tensor[Float64]:
    return safe_log(_to_f64_tensor[T](x), eps)

fn safe_sqrt[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], eps: Float64 = 0.0) -> Tensor[Float64]:
    return safe_sqrt(_to_f64_tensor[T](x), eps)

# ================================ Masks =================================

# Float64 backends -> Tensor[Int] (0/1)
fn isnan_f64(x: Tensor[Float64]) -> Tensor[Int]:
    var n = len(x._data)
    var out = List[Int](); out.reserve(n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(_is_nan64(x._data[i + 0]))
        out.append(_is_nan64(x._data[i + 1]))
        out.append(_is_nan64(x._data[i + 2]))
        out.append(_is_nan64(x._data[i + 3]))
        out.append(_is_nan64(x._data[i + 4]))
        out.append(_is_nan64(x._data[i + 5]))
        out.append(_is_nan64(x._data[i + 6]))
        out.append(_is_nan64(x._data[i + 7]))
        i += 8
    while i < n:
        out.append(_is_nan64(x._data[i]))
        i += 1
    return Tensor[Int](out, x._shape)

fn isinf_f64(x: Tensor[Float64]) -> Tensor[Int]:
    var n = len(x._data)
    var out = List[Int](); out.reserve(n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(_is_inf64(x._data[i + 0]))
        out.append(_is_inf64(x._data[i + 1]))
        out.append(_is_inf64(x._data[i + 2]))
        out.append(_is_inf64(x._data[i + 3]))
        out.append(_is_inf64(x._data[i + 4]))
        out.append(_is_inf64(x._data[i + 5]))
        out.append(_is_inf64(x._data[i + 6]))
        out.append(_is_inf64(x._data[i + 7]))
        i += 8
    while i < n:
        out.append(_is_inf64(x._data[i]))
        i += 1
    return Tensor[Int](out, x._shape)

fn isfinite_f64(x: Tensor[Float64]) -> Tensor[Int]:
    var n = len(x._data)
    var out = List[Int](); out.reserve(n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(_is_finite64(x._data[i + 0]))
        out.append(_is_finite64(x._data[i + 1]))
        out.append(_is_finite64(x._data[i + 2]))
        out.append(_is_finite64(x._data[i + 3]))
        out.append(_is_finite64(x._data[i + 4]))
        out.append(_is_finite64(x._data[i + 5]))
        out.append(_is_finite64(x._data[i + 6]))
        out.append(_is_finite64(x._data[i + 7]))
        i += 8
    while i < n:
        out.append(_is_finite64(x._data[i]))
        i += 1
    return Tensor[Int](out, x._shape)

# Generic wrappers (accept Tensor[T] -> convert to Float64)
fn isnan[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[Int]:
    return isnan_f64(_to_f64_tensor[T](x))

fn isinf[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[Int]:
    return isinf_f64(_to_f64_tensor[T](x))

fn isfinite[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T]) -> Tensor[Int]:
    return isfinite_f64(_to_f64_tensor[T](x))

# ============================= nan_to_num ===============================

fn nan_to_num_f64(
    x: Tensor[Float64],
    nan: Float64,
    posinf: Optional[Float64],
    neginf: Optional[Float64]
) -> Tensor[Float64]:
    var n = len(x._data)
    var out = List[Float64](); out.reserve(n)

    var posv = _max_f64()
    var negv = _min_f64()
    if not (posinf is None): posv = posinf.value()
    if not (neginf is None): negv = neginf.value()

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        var v0 = x._data[i + 0]; var v1 = x._data[i + 1]
        var v2 = x._data[i + 2]; var v3 = x._data[i + 3]
        var v4 = x._data[i + 4]; var v5 = x._data[i + 5]
        var v6 = x._data[i + 6]; var v7 = x._data[i + 7]

        out.append( nan if _is_nan64(v0) == 1 else (posv if (_is_inf64(v0) == 1 and v0 > 0.0) else (negv if _is_inf64(v0) == 1 else v0)) )
        out.append( nan if _is_nan64(v1) == 1 else (posv if (_is_inf64(v1) == 1 and v1 > 0.0) else (negv if _is_inf64(v1) == 1 else v1)) )
        out.append( nan if _is_nan64(v2) == 1 else (posv if (_is_inf64(v2) == 1 and v2 > 0.0) else (negv if _is_inf64(v2) == 1 else v2)) )
        out.append( nan if _is_nan64(v3) == 1 else (posv if (_is_inf64(v3) == 1 and v3 > 0.0) else (negv if _is_inf64(v3) == 1 else v3)) )
        out.append( nan if _is_nan64(v4) == 1 else (posv if (_is_inf64(v4) == 1 and v4 > 0.0) else (negv if _is_inf64(v4) == 1 else v4)) )
        out.append( nan if _is_nan64(v5) == 1 else (posv if (_is_inf64(v5) == 1 and v5 > 0.0) else (negv if _is_inf64(v5) == 1 else v5)) )
        out.append( nan if _is_nan64(v6) == 1 else (posv if (_is_inf64(v6) == 1 and v6 > 0.0) else (negv if _is_inf64(v6) == 1 else v6)) )
        out.append( nan if _is_nan64(v7) == 1 else (posv if (_is_inf64(v7) == 1 and v7 > 0.0) else (negv if _is_inf64(v7) == 1 else v7)) )
        i += 8
    while i < n:
        var v = x._data[i]
        if _is_nan64(v) == 1:
            out.append(nan)
        else:
            if _is_inf64(v) == 1:
                out.append(posv if v > 0.0 else negv)
            else:
                out.append(v)
        i += 1

    return Tensor[Float64](out, x._shape)

fn nan_to_num[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    nan: Float64 = 0.0,
    posinf: Optional[Float64] = None,
    neginf: Optional[Float64] = None
) -> Tensor[T]:
    var xf = _to_f64_tensor[T](x)
    var yf = nan_to_num_f64(xf, nan, posinf, neginf)
    return _from_f64_tensor[T](yf)

# ========================== NaN-aware reductions =========================
# public: nansum / nanmean / nanmin

# ---- All-elements (axis=None), computed in Float64 ----

fn nansum_all_f64(x: Tensor[Float64]) -> Tensor[Float64]:
    var a = _ensure_contig(x)
    var n = len(a._data)
    var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0
    var i = 0
    var lim = (n // 4) * 4
    while i < lim:
        var v0 = a._data[i + 0]; if v0 == v0: s0 = s0 + v0
        var v1 = a._data[i + 1]; if v1 == v1: s1 = s1 + v1
        var v2 = a._data[i + 2]; if v2 == v2: s2 = s2 + v2
        var v3 = a._data[i + 3]; if v3 == v3: s3 = s3 + v3
        i += 4
    while i < n:
        var v = a._data[i]; if v == v: s0 = s0 + v
        i += 1
    var total = ((s0 + s1) + (s2 + s3))
    return Tensor[Float64]([total], [1])

fn nanmean_all_f64(x: Tensor[Float64]) -> Tensor[Float64]:
    var a = _ensure_contig(x)
    var n = len(a._data)
    var sum0 = 0.0; var cnt0 = 0.0
    var i = 0
    var lim = (n // 4) * 4
    while i < lim:
        var v0 = a._data[i + 0]; if v0 == v0: sum0 = sum0 + v0; cnt0 = cnt0 + 1.0
        var v1 = a._data[i + 1]; if v1 == v1: sum0 = sum0 + v1; cnt0 = cnt0 + 1.0
        var v2 = a._data[i + 2]; if v2 == v2: sum0 = sum0 + v2; cnt0 = cnt0 + 1.0
        var v3 = a._data[i + 3]; if v3 == v3: sum0 = sum0 + v3; cnt0 = cnt0 + 1.0
        i += 4
    while i < n:
        var v = a._data[i]; if v == v: sum0 = sum0 + v; cnt0 = cnt0 + 1.0
        i += 1
    if cnt0 == 0.0:
        var nanv = 0.0 / 0.0
        return Tensor[Float64]([nanv], [1])
    var meanv = sum0 / cnt0
    return Tensor[Float64]([meanv], [1])

fn nanmin_all_f64(x: Tensor[Float64]) -> Tensor[Float64]:
    var a = _ensure_contig(x)
    var n = len(a._data)
    var found = 0
    var cur = 0.0
    var i = 0
    while i < n:
        var v = a._data[i]
        if v == v:
            if found == 0:
                cur = v; found = 1
            else:
                if v < cur: cur = v
        i += 1
    if found == 0:
        var nanv = 0.0 / 0.0
        return Tensor[Float64]([nanv], [1])
    return Tensor[Float64]([cur], [1])

# ---- Axis-aware (contiguous fast path), computed in Float64 ----

fn nansum_axis_contig_f64(data: List[Float64], shape: List[Int], axis: Int, keepdims: Bool) -> Tensor[Float64]:
    var r = len(shape)
    var ax = axis

    var outer = 1
    var i = 0
    while i < ax:
        outer = outer * shape[i]
        i += 1
    var axis_len = shape[ax]
    var inner = 1
    var j = ax + 1
    while j < r:
        inner = inner * shape[j]
        j += 1

    var out_elems = outer * inner
    var out = List[Float64](); out.reserve(out_elems)
    var z = 0
    while z < out_elems:
        out.append(0.0)
        z += 1

    var base = 0
    var o = 0
    while o < outer:
        var off = o * inner
        var k = 0
        while k < axis_len:
            var b = base + k * inner
            var t = 0
            var lim = (inner // 4) * 4
            while t < lim:
                var v0 = data[b + t + 0]; if v0 == v0: out[off + t + 0] = out[off + t + 0] + v0
                var v1 = data[b + t + 1]; if v1 == v1: out[off + t + 1] = out[off + t + 1] + v1
                var v2 = data[b + t + 2]; if v2 == v2: out[off + t + 2] = out[off + t + 2] + v2
                var v3 = data[b + t + 3]; if v3 == v3: out[off + t + 3] = out[off + t + 3] + v3
                t += 4
            while t < inner:
                var v = data[b + t]
                if v == v: out[off + t] = out[off + t] + v
                t += 1
            k += 1
        base = base + axis_len * inner
        o += 1

    var out_shape = _shape_without_axis(shape, ax, keepdims)
    return Tensor[Float64](out, out_shape)

fn nanmean_axis_contig_f64(data: List[Float64], shape: List[Int], axis: Int, keepdims: Bool) -> Tensor[Float64]:
    var r = len(shape)
    var ax = axis

    var outer = 1
    var i = 0
    while i < ax:
        outer = outer * shape[i]
        i += 1
    var axis_len = shape[ax]
    var inner = 1
    var j = ax + 1
    while j < r:
        inner = inner * shape[j]
        j += 1

    var out_elems = outer * inner
    var sumv = List[Float64](); sumv.reserve(out_elems)
    var cntv = List[Float64](); cntv.reserve(out_elems)
    var z = 0
    while z < out_elems:
        sumv.append(0.0); cntv.append(0.0)
        z += 1

    var base = 0
    var o = 0
    while o < outer:
        var off = o * inner
        var k = 0
        while k < axis_len:
            var b = base + k * inner
            var t = 0
            while t < inner:
                var v = data[b + t]
                if v == v:
                    sumv[off + t] = sumv[off + t] + v
                    cntv[off + t] = cntv[off + t] + 1.0
                t += 1
            k += 1
        base = base + axis_len * inner
        o += 1

    var out = List[Float64](); out.reserve(out_elems)
    var q = 0
    while q < out_elems:
        if cntv[q] == 0.0:
            out.append(0.0 / 0.0)
        else:
            out.append(sumv[q] / cntv[q])
        q += 1

    var out_shape = _shape_without_axis(shape, ax, keepdims)
    return Tensor[Float64](out, out_shape)

fn nanmin_axis_contig_f64(data: List[Float64], shape: List[Int], axis: Int, keepdims: Bool) -> Tensor[Float64]:
    var r = len(shape)
    var ax = axis

    var outer = 1
    var i = 0
    while i < ax:
        outer = outer * shape[i]
        i += 1
    var axis_len = shape[ax]
    var inner = 1
    var j = ax + 1
    while j < r:
        inner = inner * shape[j]
        j += 1

    var out_elems = outer * inner
    var out = List[Float64](); out.reserve(out_elems)
    var seen = List[Int](); seen.reserve(out_elems)
    var z = 0
    while z < out_elems:
        out.append(0.0); seen.append(0)
        z += 1

    var base = 0
    var o = 0
    while o < outer:
        var off = o * inner
        var k = 0
        while k < axis_len:
            var b = base + k * inner
            var t = 0
            while t < inner:
                var v = data[b + t]
                if v == v:
                    if seen[off + t] == 0:
                        out[off + t] = v; seen[off + t] = 1
                    else:
                        if v < out[off + t]: out[off + t] = v
                t += 1
            k += 1
        base = base + axis_len * inner
        o += 1

    var q = 0
    while q < out_elems:
        if seen[q] == 0:
            out[q] = 0.0 / 0.0
        q += 1

    var out_shape = _shape_without_axis(shape, ax, keepdims)
    return Tensor[Float64](out, out_shape)

# ---- Public API (T -> Float64 compute) ----

fn nansum[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
    var xf = _to_f64_tensor[T](x)
    if axis is None:
        return nansum_all_f64(xf)
    var a = _ensure_contig(xf)
    var ax = normalize_axis(axis.value(), len(a._shape))
    return nansum_axis_contig_f64(a._data, a._shape, ax, keepdims)

fn nanmean[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
    var xf = _to_f64_tensor[T](x)
    if axis is None:
        return nanmean_all_f64(xf)
    var a = _ensure_contig(xf)
    var ax = normalize_axis(axis.value(), len(a._shape))
    return nanmean_axis_contig_f64(a._data, a._shape, ax, keepdims)

fn nanmin[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
    var xf = _to_f64_tensor[T](x)
    if axis is None:
        return nanmin_all_f64(xf)
    var a = _ensure_contig(xf)
    var ax = normalize_axis(axis.value(), len(a._shape))
    return nanmin_axis_contig_f64(a._data, a._shape, ax, keepdims)
