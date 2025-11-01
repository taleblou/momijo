# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.nn.activations
# File:         src/momijo/learn/nn/activations.mojo
#
# Description:
#   Activation functions for Momijo Learn.
#   - Scalar & List APIs (backend-free) with stable approximations.
#   - Tensor APIs (elementwise + stable softmax/log_softmax along axis).
#   - GELU (tanh-approx), SiLU/Swish, hard variants, shrink/softplus/softsign.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor  # <- facade import per project policy

# --------------------------------------------
# Small numeric helpers (backend-free)
# --------------------------------------------

@always_inline
fn _clamp(x: Float64, lo: Float64, hi: Float64) -> Float64:
    var v = x
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v

# exp approximation on a bounded range using (1 + x/n)^n with n=64 and range clamp
fn _exp_approx(x: Float64) -> Float64:
    var xv = _clamp(x, -20.0, 20.0)
    var n = 64.0
    var base = 1.0 + (xv / n)
    var y = base
    y = y * y   # ^2
    y = y * y   # ^4
    y = y * y   # ^8
    y = y * y   # ^16
    y = y * y   # ^32
    y = y * y   # ^64
    return y

# tanh via exp approximation
fn _tanh_approx(x: Float64) -> Float64:
    var xv = _clamp(x, -10.0, 10.0)
    var e2x = _exp_approx(2.0 * xv)
    return (e2x - 1.0) / (e2x + 1.0)

# log approximation with simple range reduction and atanh-series
# ln(x) = ln(m * 2^k) = ln(m) + k * ln2, reduce m to ~[0.75, 1.5], then
# ln(m) ≈ 2 * [t + t^3/3 + t^5/5 + ...], t = (m-1)/(m+1)
fn _log_approx(x: Float64) -> Float64:
    if x <= 0.0:
        # represent -inf; keep in a large negative bound
        return -1.7976931348623157e308
    var ln2 = 0.6931471805599453
    var k = 0
    var m = x
    # coarse reduction by factors of 2 into [0.75, 1.5]
    while m > 1.5:
        m = m * 0.5
        k = k + 1
    while m < 0.75:
        m = m * 2.0
        k = k - 1
    var t_num = m - 1.0
    var t_den = m + 1.0
    var t = t_num / t_den
    var t2 = t * t
    var term = t
    var sum = term
    var i = 1
    # use 6 odd terms -> accurate enough for activations
    while i <= 6:
        term = term * t2
        var denom = Float64(2 * i + 1)
        sum = sum + (term / denom)
        i = i + 1
    return 2.0 * sum + Float64(k) * ln2

# --------------------------------------------
# List utility
# --------------------------------------------

fn _map(xs: List[Float64], f) -> List[Float64]:
    var out = List[Float64]()
    out.reserve(len(xs))
    var i = 0
    var n = len(xs)
    while i < n:
        out.push_back(f(xs[i]))
        i = i + 1
    return out

# --------------------------------------------
# Scalar & List activations (backend-free)
# --------------------------------------------

# ReLU family
fn relu(x: Float64) -> Float64:
    var v = x
    if v < 0.0:
        v = 0.0
    return v

fn relu(xs: List[Float64]) -> List[Float64]:
    return _map(xs, relu)

fn leaky_relu(x: Float64, negative_slope: Float64 = 0.01) -> Float64:
    if x >= 0.0:
        return x
    return negative_slope * x

fn leaky_relu(xs: List[Float64], negative_slope: Float64 = 0.01) -> List[Float64]:
    var f = fn (v: Float64) -> Float64:
        if v >= 0.0:
            return v
        return negative_slope * v
    return _map(xs, f)

fn relu6(x: Float64) -> Float64:
    var v = x
    if v < 0.0:
        v = 0.0
    if v > 6.0:
        v = 6.0
    return v

fn relu6(xs: List[Float64]) -> List[Float64]:
    return _map(xs, relu6)

# Sigmoid / Tanh / SiLU (Swish)
fn sigmoid(x: Float64) -> Float64:
    # stable: sigmoid(x) = 1/(1+exp(-x)) with approximated exp
    var e = _exp_approx(-x)
    return 1.0 / (1.0 + e)

fn sigmoid(xs: List[Float64]) -> List[Float64]:
    return _map(xs, sigmoid)

fn tanh(x: Float64) -> Float64:
    return _tanh_approx(x)

fn tanh(xs: List[Float64]) -> List[Float64]:
    return _map(xs, tanh)

fn silu(x: Float64) -> Float64:
    return x * sigmoid(x)

fn silu(xs: List[Float64]) -> List[Float64]:
    return _map(xs, silu)

# GELU (tanh approximation)
fn gelu(x: Float64) -> Float64:
    var s = 0.7978845608028654    # sqrt(2/pi)
    var x3 = x * x * x
    var inner = s * (x + 0.044715 * x3)
    var t = _tanh_approx(inner)
    return 0.5 * x * (1.0 + t)

fn gelu(xs: List[Float64]) -> List[Float64]:
    return _map(xs, gelu)

# ELU / SELU
fn elu(x: Float64, alpha: Float64 = 1.0) -> Float64:
    if x >= 0.0:
        return x
    return alpha * (_exp_approx(x) - 1.0)

fn elu(xs: List[Float64], alpha: Float64 = 1.0) -> List[Float64]:
    var f = fn (v: Float64) -> Float64:
        if v >= 0.0:
            return v
        return alpha * (_exp_approx(v) - 1.0)
    return _map(xs, f)

fn selu(x: Float64) -> Float64:
    var lmbd = 1.0507009873554805
    var alpha = 1.6732632423543772
    if x >= 0.0:
        return lmbd * x
    return lmbd * (alpha * (_exp_approx(x) - 1.0))

fn selu(xs: List[Float64]) -> List[Float64]:
    return _map(xs, selu)

# Hard variants
fn hard_sigmoid(x: Float64) -> Float64:
    var y = (x / 6.0) + 0.5
    if y < 0.0:
        y = 0.0
    if y > 1.0:
        y = 1.0
    return y

fn hard_sigmoid(xs: List[Float64]) -> List[Float64]:
    return _map(xs, hard_sigmoid)

fn hard_swish(x: Float64) -> Float64:
    return x * hard_sigmoid(x)

fn hard_swish(xs: List[Float64]) -> List[Float64]:
    return _map(xs, hard_swish)

# Softplus / Softsign / Shrinks
fn softplus(x: Float64) -> Float64:
    # max(0,x) + log1p(exp(-|x|))  (approximate log1p via ln)
    var a = x
    var ab = a
    if ab < 0.0:
        ab = -ab
    var max0a = a
    if max0a < 0.0:
        max0a = 0.0
    var e = _exp_approx(-ab)
    var log1p_e = _log_approx(1.0 + e)
    return max0a + log1p_e

fn softplus(xs: List[Float64]) -> List[Float64]:
    return _map(xs, softplus)

fn softsign(x: Float64) -> Float64:
    var ab = x
    if ab < 0.0:
        ab = -ab
    return x / (1.0 + ab)

fn softsign(xs: List[Float64]) -> List[Float64]:
    return _map(xs, softsign)

fn softshrink(x: Float64, lambd: Float64 = 0.5) -> Float64:
    var ab = x
    if ab < 0.0:
        ab = -ab
    var m = ab - lambd
    if m < 0.0:
        m = 0.0
    var s = 1.0
    if x < 0.0:
        s = -1.0
    return s * m

fn softshrink(xs: List[Float64], lambd: Float64 = 0.5) -> List[Float64]:
    var f = fn (v: Float64) -> Float64:
        var ab = v
        if ab < 0.0:
            ab = -ab
        var m = ab - lambd
        if m < 0.0:
            m = 0.0
        var s = 1.0
        if v < 0.0:
            s = -1.0
        return s * m
    return _map(xs, f)

fn hardshrink(x: Float64, lambd: Float64 = 0.5) -> Float64:
    var ab = x
    if ab < 0.0:
        ab = -ab
    if ab > lambd:
        return x
    return 0.0

fn hardshrink(xs: List[Float64], lambd: Float64 = 0.5) -> List[Float64]:
    var f = fn (v: Float64) -> Float64:
        var ab = v
        if ab < 0.0:
            ab = -ab
        if ab > lambd:
            return v
        return 0.0
    return _map(xs, f)

# --------------------------------------------
# Softmax (List)
# --------------------------------------------

fn _softmax_1d(xs: List[Float64]) -> List[Float64]:
    var n = len(xs)
    if n == 0:
        return List[Float64]()
    var m = xs[0]
    var i = 1
    while i < n:
        var v = xs[i]
        if v > m:
            m = v
        i = i + 1
    var exps = List[Float64]()
    exps.reserve(n)
    var sum_e = 0.0
    i = 0
    while i < n:
        var e = _exp_approx(xs[i] - m)
        exps.push_back(e)
        sum_e = sum_e + e
        i = i + 1
    var out = List[Float64]()
    out.reserve(n)
    i = 0
    while i < n:
        out.push_back(exps[i] / sum_e)
        i = i + 1
    return out

fn softmax(xs: List[Float64]) -> List[Float64]:
    return _softmax_1d(xs)

# 2D softmax with dimension control (List[List[Float64]])
fn softmax(x2d: List[List[Float64]], dim: Int = -1) -> List[List[Float64]]:
    var rows = len(x2d)
    if rows == 0:
        return List[List[Float64]]()
    if dim == -1 or dim == 1:
        var out = List[List[Float64]]()
        out.reserve(rows)
        var r = 0
        while r < rows:
            out.push_back(_softmax_1d(x2d[r]))
            r = r + 1
        return out
    if dim == 0:
        var cols = 0
        var r2 = 0
        while r2 < rows:
            var l = len(x2d[r2])
            if l > cols:
                cols = l
            r2 = r2 + 1
        var out2 = List[List[Float64]]()
        out2.reserve(rows)
        r2 = 0
        while r2 < rows:
            var row = List[Float64]()
            var c2 = 0
            var rl = len(x2d[r2])
            while c2 < rl:
                row.push_back(0.0)
                c2 = c2 + 1
            out2.push_back(row)
            r2 = r2 + 1
        var c = 0
        while c < cols:
            var col_vals = List[Float64]()
            var row_ids = List[Int]()
            row_ids.reserve(rows)
            var rr = 0
            while rr < rows:
                if c < len(x2d[rr]):
                    col_vals.push_back(x2d[rr][c])
                    row_ids.push_back(rr)
                rr = rr + 1
            var col_sm = _softmax_1d(col_vals)
            var k = 0
            var clen = len(col_sm)
            while k < clen:
                var rid = row_ids[k]
                out2[rid][c] = col_sm[k]
                k = k + 1
            c = c + 1
        return out2
    # default: row-wise
    return softmax(x2d, -1)

# --------------------------------------------
# Tensor helpers
# --------------------------------------------

@always_inline
fn _numel(shape: List[Int]) -> Int:
    var p = 1
    var i = 0
    var n = len(shape)
    while i < n:
        p = p * shape[i]
        i = i + 1
    return p

@always_inline
fn _normalize_axis(axis: Int, rank: Int) -> Int:
    var ax = axis
    if ax < 0:
        ax = ax + rank
    if ax < 0:
        ax = 0
    if ax >= rank:
        ax = rank - 1
    return ax

fn _row_major_strides(shape: List[Int]) -> List[Int]:
    var r = len(shape)
    var st = List[Int]()
    st.reserve(r)
    var i = 0
    while i < r:
        st.append(0)
        i = i + 1
    var acc = 1
    var j = r - 1
    while j >= 0:
        st[j] = acc
        acc = acc * shape[j]
        j = j - 1
    return st

@always_inline
fn _zeros_like_shape[T: ImplicitlyCopyable & Copyable & Movable](shape: List[Int]) -> tensor.Tensor[T]:
    return tensor.Tensor[T](shape, T(0))

fn _apply_eltwise1[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T], f: fn (Float64) -> Float64) -> tensor.Tensor[T]:
    var shp = x.shape()
    var n = _numel(shp)
    var out = tensor.Tensor[T](shp, T(0))
    var xo = x._data   # assuming public-accessible per Momijo Tensor facade
    var yo = out._data
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        yo[i    ] = T(f(Float64(xo[i    ])))
        yo[i + 1] = T(f(Float64(xo[i + 1])))
        yo[i + 2] = T(f(Float64(xo[i + 2])))
        yo[i + 3] = T(f(Float64(xo[i + 3])))
        yo[i + 4] = T(f(Float64(xo[i + 4])))
        yo[i + 5] = T(f(Float64(xo[i + 5])))
        yo[i + 6] = T(f(Float64(xo[i + 6])))
        yo[i + 7] = T(f(Float64(xo[i + 7])))
        i = i + 8
    while i < n:
        yo[i] = T(f(Float64(xo[i])))
        i = i + 1
    return out

# --------------------------------------------
# Tensor activations (elementwise)
# --------------------------------------------

fn relu[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T]) -> tensor.Tensor[T]:
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        var z = v
        if z < 0.0:
            z = 0.0
        return z
    )

fn leaky_relu[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T], negative_slope: Float64 = 0.01) -> tensor.Tensor[T]:
    var a = negative_slope
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        if v >= 0.0:
            return v
        return a * v
    )

fn elu[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T], alpha: Float64 = 1.0) -> tensor.Tensor[T]:
    var a = alpha
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        if v > 0.0:
            return v
        return a * (_exp_approx(v) - 1.0)
    )

fn selu[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T]) -> tensor.Tensor[T]:
    var scale = 1.0507009873554805
    var alpha = 1.6732632423543772
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        if v > 0.0:
            return scale * v
        return scale * (alpha * (_exp_approx(v) - 1.0))
    )

fn sigmoid[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T]) -> tensor.Tensor[T]:
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        var e = _exp_approx(-v)
        return 1.0 / (1.0 + e)
    )

fn tanh[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T]) -> tensor.Tensor[T]:
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        return _tanh_approx(v)
    )

fn softplus[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T]) -> tensor.Tensor[T]:
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        var ab = v
        if ab < 0.0:
            ab = -ab
        var max0 = v
        if max0 < 0.0:
            max0 = 0.0
        var e = _exp_approx(-ab)
        var log1p_e = _log_approx(1.0 + e)
        return max0 + log1p_e
    )

fn softsign[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T]) -> tensor.Tensor[T]:
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        var ab = v
        if ab < 0.0:
            ab = -ab
        return v / (1.0 + ab)
    )

fn softshrink[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T], lambd: Float64 = 0.5) -> tensor.Tensor[T]:
    var l = lambd
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        var ab = v
        if ab < 0.0:
            ab = -ab
        var m = ab - l
        if m < 0.0:
            m = 0.0
        var s = 1.0
        if v < 0.0:
            s = -1.0
        return s * m
    )

fn hardshrink[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T], lambd: Float64 = 0.5) -> tensor.Tensor[T]:
    var l = lambd
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        var ab = v
        if ab < 0.0:
            ab = -ab
        if ab > l:
            return v
        return 0.0
    )

fn hardsigmoid[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T]) -> tensor.Tensor[T]:
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        var y = (v * 0.2) + 0.5
        if y < 0.0:
            y = 0.0
        if y > 1.0:
            y = 1.0
        return y
    )

fn hardtanh[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T], min_val: Float64 = -1.0, max_val: Float64 = 1.0) -> tensor.Tensor[T]:
    var lo = min_val
    var hi = max_val
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        var y = v
        if y < lo:
            y = lo
        if y > hi:
            y = hi
        return y
    )

fn swish[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T]) -> tensor.Tensor[T]:
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        var e = _exp_approx(-v)
        var s = 1.0 / (1.0 + e)
        return v * s
    )

fn mish[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T]) -> tensor.Tensor[T]:
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        var ab = v
        if ab < 0.0:
            ab = -ab
        var e = _exp_approx(-ab)
        var sp = v
        if sp < 0.0:
            sp = 0.0
        var sp2 = _log_approx(1.0 + e)
        var softp = sp + sp2
        var th = _tanh_approx(softp)
        return v * th
    )

fn gelu[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T]) -> tensor.Tensor[T]:
    var c0 = 0.7978845608028654
    var c1 = 0.044715
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        var v3 = v * v * v
        var u = c0 * (v + c1 * v3)
        var th = _tanh_approx(u)
        return 0.5 * v * (1.0 + th)
    )

fn threshold[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T], th: Float64, value: Float64) -> tensor.Tensor[T]:
    var t = th
    var val = value
    return _apply_eltwise1[T](x, fn (v: Float64) -> Float64:
        if v > t:
            return v
        return val
    )

# --------------------------------------------
# Tensor softmax / log_softmax (stable, axis)
# --------------------------------------------

fn softmax_stable[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T], axis: Int = -1) -> tensor.Tensor[T]:
    var shp = x.shape()
    var rank = len(shp)
    var ax = _normalize_axis(axis, rank)
    var dim = shp[ax]
    var n = _numel(shp)
    var out = tensor.Tensor[T](shp, T(0))
    var xd = x._data
    var yd = out._data
    var strides = _row_major_strides(shp)
    var stride_ax = strides[ax]
    var rows = n // dim

    var r = 0
    while r < rows:
        # Compute base offset for this logical row across axis ax
        var offset = 0
        var rem = r
        var k = 0
        while k < rank:
            if k == ax:
                k = k + 1
                continue
            var block = 1
            var t = k + 1
            while t < rank:
                if t != ax:
                    block = block * shp[t]
                t = t + 1
            var coord = 0
            if block != 0:
                coord = rem // block
                rem = rem % block
            offset = offset + coord * strides[k]
            k = k + 1

        # 1) row max
        var maxv = -1.7976931348623157e308
        var j = 0
        while j < dim:
            var idx = offset + j * stride_ax
            var v = Float64(xd[idx])
            if v > maxv:
                maxv = v
            j = j + 1
        # 2) exp and sum
        var sumv = 0.0
        j = 0
        while j < dim:
            var idx = offset + j * stride_ax
            var z = _exp_approx(Float64(xd[idx]) - maxv)
            yd[idx] = T(z)
            sumv = sumv + z
            j = j + 1
        # 3) normalize
        var inv = 1.0
        if sumv != 0.0:
            inv = 1.0 / sumv
        j = 0
        while j < dim:
            var idx = offset + j * stride_ax
            yd[idx] = T(Float64(yd[idx]) * inv)
            j = j + 1

        r = r + 1
    return out

fn log_softmax[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T], axis: Int = -1) -> tensor.Tensor[T]:
    var shp = x.shape()
    var rank = len(shp)
    var ax = _normalize_axis(axis, rank)
    var dim = shp[ax]
    var n = _numel(shp)
    var out = tensor.Tensor[T](shp, T(0))
    var xd = x._data
    var yd = out._data
    var strides = _row_major_strides(shp)
    var stride_ax = strides[ax]
    var rows = n // dim

    var r = 0
    while r < rows:
        var offset = 0
        var rem = r
        var k = 0
        while k < rank:
            if k == ax:
                k = k + 1
                continue
            var block = 1
            var t = k + 1
            while t < rank:
                if t != ax:
                    block = block * shp[t]
                t = t + 1
            var coord = 0
            if block != 0:
                coord = rem // block
                rem = rem % block
            offset = offset + coord * strides[k]
            k = k + 1

        var maxv = -1.7976931348623157e308
        var j = 0
        while j < dim:
            var idx = offset + j * stride_ax
            var v = Float64(xd[idx])
            if v > maxv:
                maxv = v
            j = j + 1

        var sum_exp = 0.0
        j = 0
        while j < dim:
            var idx = offset + j * stride_ax
            var z = _exp_approx(Float64(xd[idx]) - maxv)
            sum_exp = sum_exp + z
            j = j + 1
        var lse = maxv + _log_approx(sum_exp)

        j = 0
        while j < dim:
            var idx = offset + j * stride_ax
            yd[idx] = T(Float64(xd[idx]) - lse)
            j = j + 1

        r = r + 1
    return out
