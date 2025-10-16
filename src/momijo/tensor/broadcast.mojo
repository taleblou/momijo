# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       momijo.tensor.ops
# File:         src/momijo/tensor/ops.mojo
#
# Description:
#   Broadcasting utilities, materialization (expand/broadcast_to),
#   elementwise binary/unary ops with fast Float32/64 paths,
#   comparisons (Int mask), and batched matmul with broadcast.
#   Includes safer generic pow for integer exponents and keeps
#   align_right/keepdims_shape helpers for compatibility.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor.tensor import Tensor
from momijo.tensor.helpers import (
    numel,
    unravel_index,
    pad_to,
    pad_strides,
    _is_vector,
    _is_matrix,
    is_row_major_contiguous,
    zero_scalar_of
)
from momijo.tensor.math import exp
from momijo.tensor.math import log
from momijo.tensor.math import sqrt
from momijo.tensor.creation import empty_f64 
from momijo.tensor.cast import *

# ============================================================
# Broadcast utilities
# ============================================================

 
# Pack result to avoid juggling multiple returns.
struct BroadcastResult:
    var ok: Bool
    var shape: List[Int]
    var lhs_padded: List[Int]
    var rhs_padded: List[Int]

    fn __init__(out self, ok_b: Bool, shape_b: List[Int],
                lhs_padded_b: List[Int], rhs_padded_b: List[Int]):
        self.ok = ok_b
        self.shape = shape_b.copy()
        self.lhs_padded = lhs_padded_b.copy()   
        self.rhs_padded = rhs_padded_b.copy()


@always_inline
fn same_shape(a: List[Int], b: List[Int]) -> Bool:
    var ra = len(a)
    if ra != len(b): return False
    var i = 0
    while i < ra:
        if a[i] != b[i]: return False
        i += 1
    return True

# Right-aligned broadcast; returns out-shape + effective strides (0 => expanded)
fn prepare_broadcast(
    lhs_shape: List[Int], lhs_strides: List[Int],
    rhs_shape: List[Int], rhs_strides: List[Int]
) -> (Bool, BroadcastResult):
    var la = len(lhs_shape)
    var lb = len(rhs_shape)
    var r = la if la > lb else lb

    var a_shape = pad_to(lhs_shape, r)
    var b_shape = pad_to(rhs_shape, r)
    var a_str   = pad_strides(lhs_strides, r - la)
    var b_str   = pad_strides(rhs_strides, r - lb)

    var out_shape = List[Int](); out_shape.reserve(r)
    var out_as    = List[Int](); out_as.reserve(r)
    var out_bs    = List[Int](); out_bs.reserve(r)

    var i = 0
    while i < r:
        var da = a_shape[i]
        var db = b_shape[i]
        if da == db:
            out_shape.append(da); out_as.append(a_str[i]); out_bs.append(b_str[i])
        elif da == 1:
            out_shape.append(db); out_as.append(0);        out_bs.append(b_str[i])
        elif db == 1:
            out_shape.append(da); out_as.append(a_str[i]); out_bs.append(0)
        else:
            return (False, BroadcastResult())
        i += 1

    var br = BroadcastResult()
    br.shape = out_shape
    br.lhs_strides = out_as
    br.rhs_strides = out_bs
    return (True, br)

@always_inline
fn pad_left_ones(x: List[Int], r: Int) -> List[Int]:
    var rx = len(x)
    var out = List[Int]()
    out.reserve(r)
    var i = 0
    while i < (r - rx):
        out.append(1)
        i += 1
    var j = 0
    while j < rx:
        out.append(x[j])
        j += 1
    return out.copy()




@always_inline
fn broadcast_shapes(a: List[Int], b: List[Int]) -> BroadcastResult:
    var ra = len(a)
    var rb = len(b)
    var r  = ra if ra >= rb else rb

    var ash = pad_left_ones(a, r)
    var bsh = pad_left_ones(b, r)

    var out = List[Int]()
    out.reserve(r)

    var i = 0
    while i < r:
        var da = ash[i]
        var db = bsh[i]
        if da == db:
            out.append(da)
        elif da == 1:
            out.append(db)
        elif db == 1:
            out.append(da)
        else:
            return BroadcastResult(False, List[Int](), ash, bsh)
        i += 1

    return BroadcastResult(True, out, ash, bsh)

# --------------------- variadic (left-fold) ----------------------

fn broadcast_shapes_many(shapes: List[List[Int]], out out_shape: List[Int]) -> Bool:
    out_shape.clear()
    var n = len(shapes)
    if n == 0:
        out_shape.append(1)
        return True

    var cur = shapes[0].copy()
    var k = 1
    while k < n:
        var br = broadcast_shapes(cur, shapes[k])
        if not br.ok:
            out_shape.clear()
            return False
        cur = br.out_shape   # consume the out shape from result
        k += 1

    # write back to out parameter
    var i = 0
    var cn = len(cur)
    while i < cn:
        out_shape.append(cur[i])
        i += 1
    return True

# ============================================================
# Materialize helpers: expand / broadcast_to / broadcast_like
# ============================================================

fn expand[T: ImplicitlyCopyable & Copyable & Movable](t: Tensor[T], new_shape: List[Int]) -> Tensor[T]:
    if same_shape(t._shape, new_shape):
        return t.copy()

    var ar = len(t._shape)
    var tr = len(new_shape)
    var r = tr

    # Right-align source dims/strides
    var aligned_dim = List[Int](); aligned_dim.reserve(r)
    var aligned_str = List[Int](); aligned_str.reserve(r)
    var i = 0
    while i < r:
        var iax = i - r + ar
        var dim = 1
        var st = 0
        if iax >= 0:
            dim = t._shape[iax]
            st = t._strides[iax]
        aligned_dim.append(dim)
        aligned_str.append(st)
        i += 1

    # Check broadcastability
    i = 0
    while i < r:
        var sd = aligned_dim[i]
        var td = new_shape[i]
        if not (sd == 1 or sd == td):
            return t.copy()
        i += 1

    var out_n = numel(new_shape)
    var out_data = List[T]()
    out_data.reserve(out_n)

    var idx = List[Int]()
    var li = 0
    while li < out_n:
        unravel_index(li, new_shape, idx)
        var ai = 0
        var k2 = 0
        while k2 < r:
            var pick = 0
            if aligned_dim[k2] != 1:
                pick = idx[k2]
            ai = ai + pick * aligned_str[k2]
            k2 += 1
        out_data.append(t._data[ai])
        li += 1
    return Tensor[T](out_data, new_shape)

@always_inline
fn broadcast_to[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], new_shape: List[Int]) -> Tensor[T]:
    return expand(x, new_shape)

@always_inline
fn broadcast_like[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], other: Tensor[T]) -> Tensor[T]:
    return expand(x, other._shape)

# ============================================================
# Binary ops: add/sub/mul/div/pow/min/max
# ============================================================

# op ids: 0=add, 1=sub, 2=mul, 3=div, 4=pow, 5=min, 6=max
@always_inline
fn bin32(x: Float32, y: Float32, op: Int) -> Float32:
    if op == 0: return x + y
    if op == 1: return x - y
    if op == 2: return x * y
    if op == 3: return x / y
    if op == 4:
        # Float pow via exp/log; for x<=0 return 0 for stability.
        if x <= 0.0: return 0.0
        return Float32(exp(Float64(y) * log(Float64(x))))
    if op == 5: return (x if x <= y else y)
    return (x if x >= y else y)

@always_inline
fn bin64(x: Float64, y: Float64, op: Int) -> Float64:
    if op == 0: return x + y
    if op == 1: return x - y
    if op == 2: return x * y
    if op == 3: return x / y
    if op == 4:
        if x <= 0.0: return 0.0
        return exp(y * log(x))
    if op == 5: return (x if x <= y else y)
    return (x if x >= y else y)

# -------- safer generic pow (for non-floats) --------
@always_inline
fn ipow_generic[T](base: T, exp_i: Int) -> T:
    var e = exp_i
    var b = base
    var res = T(1)
    while e > 0:
        if (e & 1) != 0:
            res = res * b
        e = e >> 1
        if e > 0:
            b = b * b
    return res

@always_inline
fn pow_generic_safe[T](x: T, y_val: T) -> T:
    # Interpret exponent as integer
    var iy = Int(y_val)
    if iy == 0:
        # 0^0 => 1
        return T(1)
    if iy > 0:
        return ipow_generic[T](x, iy)
    # Negative exponent: for non-floats, return stable integer behavior
    # |x| != 1 -> 0;  1 -> 1;  -1 -> +/-1 depending on parity.
    if x == T(1): return T(1)
    if x == T(-1):
        return (T(1) if ((-iy) & 1) == 0 else T(-1))
    return zero_scalar_of[T]()

# ------------------------------------------

fn apply_broadcast_binary_float32(a: Tensor[Float32], b: Tensor[Float32], out: Tensor[Float32], op: Int):
    if same_shape(a._shape, b._shape) and same_shape(a._shape, out._shape):
        var n = len(out._data)
        var i = 0
        var lim = (n // 16) * 16
        while i < lim:
            out._data[i    ] = bin32(a._data[i    ], b._data[i    ], op)
            out._data[i + 1] = bin32(a._data[i + 1], b._data[i + 1], op)
            out._data[i + 2] = bin32(a._data[i + 2], b._data[i + 2], op)
            out._data[i + 3] = bin32(a._data[i + 3], b._data[i + 3], op)
            out._data[i + 4] = bin32(a._data[i + 4], b._data[i + 4], op)
            out._data[i + 5] = bin32(a._data[i + 5], b._data[i + 5], op)
            out._data[i + 6] = bin32(a._data[i + 6], b._data[i + 6], op)
            out._data[i + 7] = bin32(a._data[i + 7], b._data[i + 7], op)
            out._data[i + 8] = bin32(a._data[i + 8], b._data[i + 8], op)
            out._data[i + 9] = bin32(a._data[i + 9], b._data[i + 9], op)
            out._data[i +10] = bin32(a._data[i +10], b._data[i +10], op)
            out._data[i +11] = bin32(a._data[i +11], b._data[i +11], op)
            out._data[i +12] = bin32(a._data[i +12], b._data[i +12], op)
            out._data[i +13] = bin32(a._data[i +13], b._data[i +13], op)
            out._data[i +14] = bin32(a._data[i +14], b._data[i +14], op)
            out._data[i +15] = bin32(a._data[i +15], b._data[i +15], op)
            i += 16
        while i < n:
            out._data[i] = bin32(a._data[i], b._data[i], op)
            i += 1
        return

    var (ok, br) = prepare_broadcast(a._shape, a._strides, b._shape, b._strides)
    if not ok: return

    var out_n = numel(br.shape)
    var R = len(br.shape)
    var i3 = 0
    while i3 < out_n:
        var ai = 0
        var bi = 0
        var tmp = i3
        var d = R - 1
        while d >= 0:
            var dim = br.shape[d]
            var od = tmp % dim
            tmp = tmp // dim
            ai = ai + od * br.lhs_strides[d]
            bi = bi + od * br.rhs_strides[d]
            d -= 1
        out._data[i3] = bin32(a._data[ai], b._data[bi], op)
        i3 += 1

fn apply_broadcast_binary_float64(a: Tensor[Float64], b: Tensor[Float64], out: Tensor[Float64], op: Int):
    if same_shape(a._shape, b._shape) and same_shape(a._shape, out._shape):
        var n = len(out._data)
        var i = 0
        var lim = (n // 16) * 16
        while i < lim:
            out._data[i    ] = bin64(a._data[i    ], b._data[i    ], op)
            out._data[i + 1] = bin64(a._data[i + 1], b._data[i + 1], op)
            out._data[i + 2] = bin64(a._data[i + 2], b._data[i + 2], op)
            out._data[i + 3] = bin64(a._data[i + 3], b._data[i + 3], op)
            out._data[i + 4] = bin64(a._data[i + 4], b._data[i + 4], op)
            out._data[i + 5] = bin64(a._data[i + 5], b._data[i + 5], op)
            out._data[i + 6] = bin64(a._data[i + 6], b._data[i + 6], op)
            out._data[i + 7] = bin64(a._data[i + 7], b._data[i + 7], op)
            out._data[i + 8] = bin64(a._data[i + 8], b._data[i + 8], op)
            out._data[i + 9] = bin64(a._data[i + 9], b._data[i + 9], op)
            out._data[i +10] = bin64(a._data[i +10], b._data[i +10], op)
            out._data[i +11] = bin64(a._data[i +11], b._data[i +11], op)
            out._data[i +12] = bin64(a._data[i +12], b._data[i +12], op)
            out._data[i +13] = bin64(a._data[i +13], b._data[i +13], op)
            out._data[i +14] = bin64(a._data[i +14], b._data[i +14], op)
            out._data[i +15] = bin64(a._data[i +15], b._data[i +15], op)
            i += 16
        while i < n:
            out._data[i] = bin64(a._data[i], b._data[i], op)
            i += 1
        return

    var (ok, br) = prepare_broadcast(a._shape, a._strides, b._shape, b._strides)
    if not ok: return

    var out_n = numel(br.shape)
    var R = len(br.shape)
    var i3 = 0
    while i3 < out_n:
        var ai = 0
        var bi = 0
        var tmp = i3
        var d = R - 1
        while d >= 0:
            var dim = br.shape[d]
            var od = tmp % dim
            tmp = tmp // dim
            ai = ai + od * br.lhs_strides[d]
            bi = bi + od * br.rhs_strides[d]
            d -= 1
        out._data[i3] = bin64(a._data[ai], b._data[bi], op)
        i3 += 1

fn apply_broadcast_binary[T: Copyable & Movable](a: Tensor[T], b: Tensor[T], out: Tensor[T], op: Int):
    if T is Float32:
        apply_broadcast_binary_float32(
            unsafe_bitcast[Tensor[Float32]](a),
            unsafe_bitcast[Tensor[Float32]](b),
            unsafe_bitcast[Tensor[Float32]](out),
            op
        )
        return
    if T is Float64:
        apply_broadcast_binary_float64(
            unsafe_bitcast[Tensor[Float64]](a),
            unsafe_bitcast[Tensor[Float64]](b),
            unsafe_bitcast[Tensor[Float64]](out),
            op
        )
        return

    # generic (safer pow)
    if same_shape(a._shape, b._shape) and same_shape(a._shape, out._shape):
        var n = len(out._data)
        var i = 0
        while i < n:
            var ax = a._data[i]
            var bx = b._data[i]
            if op == 0: out._data[i] = ax + bx
            elif op == 1: out._data[i] = ax - bx
            elif op == 2: out._data[i] = ax * bx
            elif op == 3: out._data[i] = ax / bx
            elif op == 5: out._data[i] = (ax if ax <= bx else bx)
            elif op == 6: out._data[i] = (ax if ax >= bx else bx)
            else:
                out._data[i] = pow_generic_safe[T](ax, bx)
            i += 1
        return

    var (ok, br) = prepare_broadcast(a._shape, a._strides, b._shape, b._strides)
    if not ok: return

    var out_n = numel(br.shape)
    var R = len(br.shape)
    var i3 = 0
    while i3 < out_n:
        var ai = 0
        var bi = 0
        var tmp = i3
        var d = R - 1
        while d >= 0:
            var dim = br.shape[d]
            var od = tmp % dim
            tmp = tmp // dim
            ai = ai + od * br.lhs_strides[d]
            bi = bi + od * br.rhs_strides[d]
            d -= 1
        var ax = a._data[ai]
        var bx = b._data[bi]
        if op == 0: out._data[i3] = ax + bx
        elif op == 1: out._data[i3] = ax - bx
        elif op == 2: out._data[i3] = ax * bx
        elif op == 3: out._data[i3] = ax / bx
        elif op == 5: out._data[i3] = (ax if ax <= bx else bx)
        elif op == 6: out._data[i3] = (ax if ax >= bx else bx)
        else:
            out._data[i3] = pow_generic_safe[T](ax, bx)
        i3 += 1

# Friendly wrappers
@always_inline
fn add_[T: Copyable & Movable](a: Tensor[T], b: Tensor[T], mut out: Tensor[T]) -> None:
    apply_broadcast_binary(a, b, out, 0)  # add

@always_inline
fn sub_[T: Copyable & Movable](a: Tensor[T], b: Tensor[T], mut out: Tensor[T]) -> None:
    apply_broadcast_binary(a, b, out, 1)  # sub

@always_inline
fn mul_[T: Copyable & Movable](a: Tensor[T], b: Tensor[T], mut out: Tensor[T]) -> None:
    apply_broadcast_binary(a, b, out, 2)  # mul

@always_inline
fn div_[T: Copyable & Movable](a: Tensor[T], b: Tensor[T], mut out: Tensor[T]) -> None:
    apply_broadcast_binary(a, b, out, 3)  # div

@always_inline
fn pow_[T: Copyable & Movable](a: Tensor[T], b: Tensor[T], mut out: Tensor[T]) -> None:
    apply_broadcast_binary(a, b, out, 4)  # pow

@always_inline
fn minimum_[T: Copyable & Movable](a: Tensor[T], b: Tensor[T], mut out: Tensor[T]) -> None:
    apply_broadcast_binary(a, b, out, 5)  # min

@always_inline
fn maximum_[T: Copyable & Movable](a: Tensor[T], b: Tensor[T], mut out: Tensor[T]) -> None:
    apply_broadcast_binary(a, b, out, 6)  # max


# ============================================================
# Unary ops: neg/abs/exp/log/sqrt/tanh/relu/expm1 (+clip/sign)
# ============================================================

@always_inline
fn tanh64(x: Float64) -> Float64:
    var e2x = exp(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)

# uops: 0=neg,1=abs,2=exp,3=log,4=sqrt,5=tanh,6=relu,7=expm1
@always_inline
fn u32(x: Float32, op: Int) -> Float32:
    if op == 0: return -x
    if op == 1: return (x if x >= 0.0 else -x)
    if op == 2: return Float32(exp(Float64(x)))
    if op == 3: return Float32(log(Float64(x)))
    if op == 4: return Float32(sqrt(Float64(x)))
    if op == 5: return Float32(tanh64(Float64(x)))
    if op == 6: return (x if x > 0.0 else 0.0)
    return Float32(exp(Float64(x)) - 1.0)

@always_inline
fn u64(x: Float64, op: Int) -> Float64:
    if op == 0: return -x
    if op == 1: return (x if x >= 0.0 else -x)
    if op == 2: return exp(x)
    if op == 3: return log(x)
    if op == 4: return sqrt(x)
    if op == 5: return tanh64(x)
    if op == 6: return (x if x > 0.0 else 0.0)
    return exp(x) - 1.0

fn apply_unary_float32(x: Tensor[Float32], out: Tensor[Float32], op: Int):
    var n = len(x._data)
    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out._data[i    ] = u32(x._data[i    ], op)
        out._data[i + 1] = u32(x._data[i + 1], op)
        out._data[i + 2] = u32(x._data[i + 2], op)
        out._data[i + 3] = u32(x._data[i + 3], op)
        out._data[i + 4] = u32(x._data[i + 4], op)
        out._data[i + 5] = u32(x._data[i + 5], op)
        out._data[i + 6] = u32(x._data[i + 6], op)
        out._data[i + 7] = u32(x._data[i + 7], op)
        out._data[i + 8] = u32(x._data[i + 8], op)
        out._data[i + 9] = u32(x._data[i + 9], op)
        out._data[i +10] = u32(x._data[i +10], op)
        out._data[i +11] = u32(x._data[i +11], op)
        out._data[i +12] = u32(x._data[i +12], op)
        out._data[i +13] = u32(x._data[i +13], op)
        out._data[i +14] = u32(x._data[i +14], op)
        out._data[i +15] = u32(x._data[i +15], op)
        i += 16
    while i < n:
        out._data[i] = u32(x._data[i], op)
        i += 1

fn apply_unary_float64(x: Tensor[Float64], out: Tensor[Float64], op: Int):
    var n = len(x._data)
    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out._data[i    ] = u64(x._data[i    ], op)
        out._data[i + 1] = u64(x._data[i + 1], op)
        out._data[i + 2] = u64(x._data[i + 2], op)
        out._data[i + 3] = u64(x._data[i + 3], op)
        out._data[i + 4] = u64(x._data[i + 4], op)
        out._data[i + 5] = u64(x._data[i + 5], op)
        out._data[i + 6] = u64(x._data[i + 6], op)
        out._data[i + 7] = u64(x._data[i + 7], op)
        out._data[i + 8] = u64(x._data[i + 8], op)
        out._data[i + 9] = u64(x._data[i + 9], op)
        out._data[i +10] = u64(x._data[i +10], op)
        out._data[i +11] = u64(x._data[i +11], op)
        out._data[i +12] = u64(x._data[i +12], op)
        out._data[i +13] = u64(x._data[i +13], op)
        out._data[i +14] = u64(x._data[i +14], op)
        out._data[i +15] = u64(x._data[i +15], op)
        i += 16
    while i < n:
        out._data[i] = u64(x._data[i], op)
        i += 1

fn apply_unary[T: Copyable & Movable](x: Tensor[T], out: Tensor[T], op: Int):
    if T is Float32:
        apply_unary_float32(
            unsafe_bitcast[Tensor[Float32]](x),
            unsafe_bitcast[Tensor[Float32]](out),
            op
        )
        return
    if T is Float64:
        apply_unary_float64(
            unsafe_bitcast[Tensor[Float64]](x),
            unsafe_bitcast[Tensor[Float64]](out),
            op
        )
        return

    # generic (neg/abs; others = identity)
    var n = len(x._data)
    var i = 0
    if op == 0:
        while i < n:
            out._data[i] = -x._data[i]
            i += 1
        return
    if op == 1:
        while i < n:
            var v = x._data[i]
            out._data[i] = (v if v >= zero_scalar_of[T]() else -v)
            i += 1
        return
    while i < n:
        out._data[i] = x._data[i]
        i += 1

# Specialized clip/sign
@always_inline
fn _clip_value[T: ImplicitlyCopyable & Copyable & Movable](x: T, lo: T, hi: T) -> T:
    var v = x
    if v < lo: v = lo
    if v > hi: v = hi
    return v

# ----------------------- Specialized fast paths -----------------------

@always_inline
fn _clip_f32(x: Tensor[Float32], out: Tensor[Float32], lo: Float32, hi: Float32):
    var n = len(x._data)
    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        var v0 = x._data[i    ]; out._data[i    ] = (hi if v0 > hi else (lo if v0 < lo else v0))
        var v1 = x._data[i + 1]; out._data[i + 1] = (hi if v1 > hi else (lo if v1 < lo else v1))
        var v2 = x._data[i + 2]; out._data[i + 2] = (hi if v2 > hi else (lo if v2 < lo else v2))
        var v3 = x._data[i + 3]; out._data[i + 3] = (hi if v3 > hi else (lo if v3 < lo else v3))
        var v4 = x._data[i + 4]; out._data[i + 4] = (hi if v4 > hi else (lo if v4 < lo else v4))
        var v5 = x._data[i + 5]; out._data[i + 5] = (hi if v5 > hi else (lo if v5 < lo else v5))
        var v6 = x._data[i + 6]; out._data[i + 6] = (hi if v6 > hi else (lo if v6 < lo else v6))
        var v7 = x._data[i + 7]; out._data[i + 7] = (hi if v7 > hi else (lo if v7 < lo else v7))
        var v8 = x._data[i + 8]; out._data[i + 8] = (hi if v8 > hi else (lo if v8 < lo else v8))
        var v9 = x._data[i + 9]; out._data[i + 9] = (hi if v9 > hi else (lo if v9 < lo else v9))
        var va = x._data[i +10]; out._data[i +10] = (hi if va > hi else (lo if va < lo else va))
        var vb = x._data[i +11]; out._data[i +11] = (hi if vb > hi else (lo if vb < lo else vb))
        var vc = x._data[i +12]; out._data[i +12] = (hi if vc > hi else (lo if vc < lo else vc))
        var vd = x._data[i +13]; out._data[i +13] = (hi if vd > hi else (lo if vd < lo else vd))
        var ve = x._data[i +14]; out._data[i +14] = (hi if ve > hi else (lo if ve < lo else ve))
        var vf = x._data[i +15]; out._data[i +15] = (hi if vf > hi else (lo if vf < lo else vf))
        i += 16
    while i < n:
        var v = x._data[i]
        out._data[i] = (hi if v > hi else (lo if v < lo else v))
        i += 1

@always_inline
fn _clip_f64(x: Tensor[Float64], out: Tensor[Float64], lo: Float64, hi: Float64):
    var n = len(x._data)
    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        var v0 = x._data[i    ]; out._data[i    ] = (hi if v0 > hi else (lo if v0 < lo else v0))
        var v1 = x._data[i + 1]; out._data[i + 1] = (hi if v1 > hi else (lo if v1 < lo else v1))
        var v2 = x._data[i + 2]; out._data[i + 2] = (hi if v2 > hi else (lo if v2 < lo else v2))
        var v3 = x._data[i + 3]; out._data[i + 3] = (hi if v3 > hi else (lo if v3 < lo else v3))
        var v4 = x._data[i + 4]; out._data[i + 4] = (hi if v4 > hi else (lo if v4 < lo else v4))
        var v5 = x._data[i + 5]; out._data[i + 5] = (hi if v5 > hi else (lo if v5 < lo else v5))
        var v6 = x._data[i + 6]; out._data[i + 6] = (hi if v6 > hi else (lo if v6 < lo else v6))
        var v7 = x._data[i + 7]; out._data[i + 7] = (hi if v7 > hi else (lo if v7 < lo else v7))
        var v8 = x._data[i + 8]; out._data[i + 8] = (hi if v8 > hi else (lo if v8 < lo else v8))
        var v9 = x._data[i + 9]; out._data[i + 9] = (hi if v9 > hi else (lo if v9 < lo else v9))
        var va = x._data[i +10]; out._data[i +10] = (hi if va > hi else (lo if va < lo else va))
        var vb = x._data[i +11]; out._data[i +11] = (hi if vb > hi else (lo if vb < lo else vb))
        var vc = x._data[i +12]; out._data[i +12] = (hi if vc > hi else (lo if vc < lo else vc))
        var vd = x._data[i +13]; out._data[i +13] = (hi if vd > hi else (lo if vd < lo else vd))
        var ve = x._data[i +14]; out._data[i +14] = (hi if ve > hi else (lo if ve < lo else ve))
        var vf = x._data[i +15]; out._data[i +15] = (hi if vf > hi else (lo if vf < lo else vf))
        i += 16
    while i < n:
        var v = x._data[i]
        out._data[i] = (hi if v > hi else (lo if v < lo else v))
        i += 1

# ----------------------- Generic (any numeric T) -----------------------

@always_inline
fn _clip_generic[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], out: Tensor[T], lo: T, hi: T):
    var n = len(x._data)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out._data[i    ] = _clip_value[T](x._data[i    ], lo, hi)
        out._data[i + 1] = _clip_value[T](x._data[i + 1], lo, hi)
        out._data[i + 2] = _clip_value[T](x._data[i + 2], lo, hi)
        out._data[i + 3] = _clip_value[T](x._data[i + 3], lo, hi)
        out._data[i + 4] = _clip_value[T](x._data[i + 4], lo, hi)
        out._data[i + 5] = _clip_value[T](x._data[i + 5], lo, hi)
        out._data[i + 6] = _clip_value[T](x._data[i + 6], lo, hi)
        out._data[i + 7] = _clip_value[T](x._data[i + 7], lo, hi)
        i += 8
    while i < n:
        out._data[i] = _clip_value[T](x._data[i], lo, hi)
        i += 1

# ---------------------------- Public API ----------------------------

# Out-of-place: returns a new tensor with same shape as x.
fn clip[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], lo: T, hi: T) -> Tensor[T]:
    var n = len(x._data)
    var buf = List[T](); buf.reserve(n)
    var out = Tensor[T](buf, x._shape)   # assumes Tensor(List[T], shape) ctor exists in your tree
    if T is Float32:
        _clip_f32(
            unsafe_bitcast[Tensor[Float32]](x),
            unsafe_bitcast[Tensor[Float32]](out),
            unsafe_bitcast[Float32](lo),
            unsafe_bitcast[Float32](hi)
        )
        return out
    if T is Float64:
        _clip_f64(
            unsafe_bitcast[Tensor[Float64]](x),
            unsafe_bitcast[Tensor[Float64]](out),
            unsafe_bitcast[Float64](lo),
            unsafe_bitcast[Float64](hi)
        )
        return out
    _clip_generic[T](x, out, lo, hi)
    return out

# In-place: modifies x and returns it (useful to avoid allocation). 
fn clip(mut x: Tensor[Float32], lo: Float32, hi: Float32) -> Tensor[Float32]:
    _clip_f32(x, x, lo, hi)
    return x

fn clip(mut x: Tensor[Float64], lo: Float64, hi: Float64) -> Tensor[Float64]:
    _clip_f64(x, x, lo, hi)
    return x

fn clip[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], lo: T, hi: T) -> Tensor[T]:
    _clip_generic[T](x, x, lo, hi)
    return x


@always_inline
fn clamp(x: Int, lo: Int, hi: Int) -> Int:
    var a = lo
    var b = hi
    if b < a:
        var t = a; a = b; b = t
    var v = x
    if v < a: v = a
    if v > b: v = b
    return v


@always_inline
fn clamp(x: Float32, lo: Float32, hi: Float32) -> Float32:
    var a = lo
    var b = hi
    if b < a:
        var t = a; a = b; b = t
    var v = x
    if v < a: v = a
    if v > b: v = b
    return v

@always_inline
fn clamp(x: Float64, lo: Float64, hi: Float64) -> Float64:
    var a = lo
    var b = hi
    if b < a:
        var t = a; a = b; b = t
    var v = x
    if v < a: v = a
    if v > b: v = b
    return v

fn sign_float32(x: Tensor[Float32], out: Tensor[Float32]):
    var n = len(x._data)
    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out._data[i    ] = (1.0 if x._data[i    ] > 0.0 else (-1.0 if x._data[i    ] < 0.0 else 0.0))
        out._data[i + 1] = (1.0 if x._data[i + 1] > 0.0 else (-1.0 if x._data[i + 1] < 0.0 else 0.0))
        out._data[i + 2] = (1.0 if x._data[i + 2] > 0.0 else (-1.0 if x._data[i + 2] < 0.0 else 0.0))
        out._data[i + 3] = (1.0 if x._data[i + 3] > 0.0 else (-1.0 if x._data[i + 3] < 0.0 else 0.0))
        out._data[i + 4] = (1.0 if x._data[i + 4] > 0.0 else (-1.0 if x._data[i + 4] < 0.0 else 0.0))
        out._data[i + 5] = (1.0 if x._data[i + 5] > 0.0 else (-1.0 if x._data[i + 5] < 0.0 else 0.0))
        out._data[i + 6] = (1.0 if x._data[i + 6] > 0.0 else (-1.0 if x._data[i + 6] < 0.0 else 0.0))
        out._data[i + 7] = (1.0 if x._data[i + 7] > 0.0 else (-1.0 if x._data[i + 7] < 0.0 else 0.0))
        out._data[i + 8] = (1.0 if x._data[i + 8] > 0.0 else (-1.0 if x._data[i + 8] < 0.0 else 0.0))
        out._data[i + 9] = (1.0 if x._data[i + 9] > 0.0 else (-1.0 if x._data[i + 9] < 0.0 else 0.0))
        out._data[i +10] = (1.0 if x._data[i +10] > 0.0 else (-1.0 if x._data[i +10] < 0.0 else 0.0))
        out._data[i +11] = (1.0 if x._data[i +11] > 0.0 else (-1.0 if x._data[i +11] < 0.0 else 0.0))
        out._data[i +12] = (1.0 if x._data[i +12] > 0.0 else (-1.0 if x._data[i +12] < 0.0 else 0.0))
        out._data[i +13] = (1.0 if x._data[i +13] > 0.0 else (-1.0 if x._data[i +13] < 0.0 else 0.0))
        out._data[i +14] = (1.0 if x._data[i +14] > 0.0 else (-1.0 if x._data[i +14] < 0.0 else 0.0))
        out._data[i +15] = (1.0 if x._data[i +15] > 0.0 else (-1.0 if x._data[i +15] < 0.0 else 0.0))
        i += 16
    while i < n:
        var v = x._data[i]
        out._data[i] = (1.0 if v > 0.0 else (-1.0 if v < 0.0 else 0.0))
        i += 1

fn sign_float64(x: Tensor[Float64], out: Tensor[Float64]):
    var n = len(x._data)
    var i = 0
    var lim = (n // 16) * 16
    while i < lim:
        out._data[i    ] = (1.0 if x._data[i    ] > 0.0 else (-1.0 if x._data[i    ] < 0.0 else 0.0))
        out._data[i + 1] = (1.0 if x._data[i + 1] > 0.0 else (-1.0 if x._data[i + 1] < 0.0 else 0.0))
        out._data[i + 2] = (1.0 if x._data[i + 2] > 0.0 else (-1.0 if x._data[i + 2] < 0.0 else 0.0))
        out._data[i + 3] = (1.0 if x._data[i + 3] > 0.0 else (-1.0 if x._data[i + 3] < 0.0 else 0.0))
        out._data[i + 4] = (1.0 if x._data[i + 4] > 0.0 else (-1.0 if x._data[i + 4] < 0.0 else 0.0))
        out._data[i + 5] = (1.0 if x._data[i + 5] > 0.0 else (-1.0 if x._data[i + 5] < 0.0 else 0.0))
        out._data[i + 6] = (1.0 if x._data[i + 6] > 0.0 else (-1.0 if x._data[i + 6] < 0.0 else 0.0))
        out._data[i + 7] = (1.0 if x._data[i + 7] > 0.0 else (-1.0 if x._data[i + 7] < 0.0 else 0.0))
        out._data[i + 8] = (1.0 if x._data[i + 8] > 0.0 else (-1.0 if x._data[i + 8] < 0.0 else 0.0))
        out._data[i + 9] = (1.0 if x._data[i + 9] > 0.0 else (-1.0 if x._data[i + 9] < 0.0 else 0.0))
        out._data[i +10] = (1.0 if x._data[i +10] > 0.0 else (-1.0 if x._data[i +10] < 0.0 else 0.0))
        out._data[i +11] = (1.0 if x._data[i +11] > 0.0 else (-1.0 if x._data[i +11] < 0.0 else 0.0))
        out._data[i +12] = (1.0 if x._data[i +12] > 0.0 else (-1.0 if x._data[i +12] < 0.0 else 0.0))
        out._data[i +13] = (1.0 if x._data[i +13] > 0.0 else (-1.0 if x._data[i +13] < 0.0 else 0.0))
        out._data[i +14] = (1.0 if x._data[i +14] > 0.0 else (-1.0 if x._data[i +14] < 0.0 else 0.0))
        out._data[i +15] = (1.0 if x._data[i +15] > 0.0 else (-1.0 if x._data[i +15] < 0.0 else 0.0))
        i += 16
    while i < n:
        var v = x._data[i]
        out._data[i] = (1.0 if v > 0.0 else (-1.0 if v < 0.0 else 0.0))
        i += 1

fn sign_[T: Copyable & Movable](x: Tensor[T], out: Tensor[T]):
    if T is Float32:
        sign_float32(unsafe_bitcast[Tensor[Float32]](x), unsafe_bitcast[Tensor[Float32]](out))
        return
    if T is Float64:
        sign_float64(unsafe_bitcast[Tensor[Float64]](x), unsafe_bitcast[Tensor[Float64]](out))
        return
    var n = len(x._data)
    var i = 0
    while i < n:
        var v = x._data[i]
        out._data[i] = (T(1) if v > zero_scalar_of[T]() else (T(-1) if v < zero_scalar_of[T]() else zero_scalar_of[T]()))
        i += 1

# ============================================================
# Comparisons: eq/ne/lt/le/gt/ge  (mask Int 0/1)
# ============================================================

@always_inline
fn cmp32(a: Float32, b: Float32, m: Int) -> Int:
    if m == 0: return 1 if a == b else 0
    if m == 1: return 1 if a != b else 0
    if m == 2: return 1 if a <  b else 0
    if m == 3: return 1 if a <= b else 0
    if m == 4: return 1 if a >  b else 0
    return 1 if a >= b else 0

@always_inline
fn cmp64(a: Float64, b: Float64, m: Int) -> Int:
    if m == 0: return 1 if a == b else 0
    if m == 1: return 1 if a != b else 0
    if m == 2: return 1 if a <  b else 0
    if m == 3: return 1 if a <= b else 0
    if m == 4: return 1 if a >  b else 0
    return 1 if a >= b else 0

fn compare_float32(a: Tensor[Float32], b: Tensor[Float32], out_mask: Tensor[Int], mode: Int):
    if same_shape(a._shape, b._shape) and len(out_mask._data) == len(a._data):
        var n = len(out_mask._data)
        var i = 0
        var lim = (n // 16) * 16
        while i < lim:
            out_mask._data[i    ] = cmp32(a._data[i    ], b._data[i    ], mode)
            out_mask._data[i + 1] = cmp32(a._data[i + 1], b._data[i + 1], mode)
            out_mask._data[i + 2] = cmp32(a._data[i + 2], b._data[i + 2], mode)
            out_mask._data[i + 3] = cmp32(a._data[i + 3], b._data[i + 3], mode)
            out_mask._data[i + 4] = cmp32(a._data[i + 4], b._data[i + 4], mode)
            out_mask._data[i + 5] = cmp32(a._data[i + 5], b._data[i + 5], mode)
            out_mask._data[i + 6] = cmp32(a._data[i + 6], b._data[i + 6], mode)
            out_mask._data[i + 7] = cmp32(a._data[i + 7], b._data[i + 7], mode)
            out_mask._data[i + 8] = cmp32(a._data[i + 8], b._data[i + 8], mode)
            out_mask._data[i + 9] = cmp32(a._data[i + 9], b._data[i + 9], mode)
            out_mask._data[i +10] = cmp32(a._data[i +10], b._data[i +10], mode)
            out_mask._data[i +11] = cmp32(a._data[i +11], b._data[i +11], mode)
            out_mask._data[i +12] = cmp32(a._data[i +12], b._data[i +12], mode)
            out_mask._data[i +13] = cmp32(a._data[i +13], b._data[i +13], mode)
            out_mask._data[i +14] = cmp32(a._data[i +14], b._data[i +14], mode)
            out_mask._data[i +15] = cmp32(a._data[i +15], b._data[i +15], mode)
            i += 16
        while i < n:
            out_mask._data[i] = cmp32(a._data[i], b._data[i], mode)
            i += 1
        return

    var (ok, br) = prepare_broadcast(a._shape, a._strides, b._shape, b._strides)
    if not ok: return

    var out_n = numel(br.shape)
    var R = len(br.shape)
    var i3 = 0
    while i3 < out_n:
        var ai = 0
        var bi = 0
        var tmp = i3
        var d = R - 1
        while d >= 0:
            var dim = br.shape[d]
            var od = tmp % dim
            tmp = tmp // dim
            ai = ai + od * br.lhs_strides[d]
            bi = bi + od * br.rhs_strides[d]
            d -= 1
        out_mask._data[i3] = cmp32(a._data[ai], b._data[bi], mode)
        i3 += 1

fn compare_float64(a: Tensor[Float64], b: Tensor[Float64], out_mask: Tensor[Int], mode: Int):
    if same_shape(a._shape, b._shape) and len(out_mask._data) == len(a._data):
        var n = len(out_mask._data)
        var i = 0
        var lim = (n // 16) * 16
        while i < lim:
            out_mask._data[i    ] = cmp64(a._data[i    ], b._data[i    ], mode)
            out_mask._data[i + 1] = cmp64(a._data[i + 1], b._data[i + 1], mode)
            out_mask._data[i + 2] = cmp64(a._data[i + 2], b._data[i + 2], mode)
            out_mask._data[i + 3] = cmp64(a._data[i + 3], b._data[i + 3], mode)
            out_mask._data[i + 4] = cmp64(a._data[i + 4], b._data[i + 4], mode)
            out_mask._data[i + 5] = cmp64(a._data[i + 5], b._data[i + 5], mode)
            out_mask._data[i + 6] = cmp64(a._data[i + 6], b._data[i + 6], mode)
            out_mask._data[i + 7] = cmp64(a._data[i + 7], b._data[i + 7], mode)
            out_mask._data[i + 8] = cmp64(a._data[i + 8], b._data[i + 8], mode)
            out_mask._data[i + 9] = cmp64(a._data[i + 9], b._data[i + 9], mode)
            out_mask._data[i +10] = cmp64(a._data[i +10], b._data[i +10], mode)
            out_mask._data[i +11] = cmp64(a._data[i +11], b._data[i +11], mode)
            out_mask._data[i +12] = cmp64(a._data[i +12], b._data[i +12], mode)
            out_mask._data[i +13] = cmp64(a._data[i +13], b._data[i +13], mode)
            out_mask._data[i +14] = cmp64(a._data[i +14], b._data[i +14], mode)
            out_mask._data[i +15] = cmp64(a._data[i +15], b._data[i +15], mode)
            i += 16
        while i < n:
            out_mask._data[i] = cmp64(a._data[i], b._data[i], mode)
            i += 1
        return

    var (ok, br) = prepare_broadcast(a._shape, a._strides, b._shape, b._strides)
    if not ok: return

    var out_n = numel(br.shape)
    var R = len(br.shape)
    var i3 = 0
    while i3 < out_n:
        var ai = 0
        var bi = 0
        var tmp = i3
        var d = R - 1
        while d >= 0:
            var dim = br.shape[d]
            var od = tmp % dim
            tmp = tmp // dim
            ai = ai + od * br.lhs_strides[d]
            bi = bi + od * br.rhs_strides[d]
            d -= 1
        out_mask._data[i3] = cmp64(a._data[ai], b._data[bi], mode)
        i3 += 1

fn compare[T: Copyable & Movable](a: Tensor[T], b: Tensor[T], out_mask: Tensor[Int], mode: Int):
    if T is Float32:
        compare_float32(unsafe_bitcast[Tensor[Float32]](a), unsafe_bitcast[Tensor[Float32]](b), out_mask, mode)
        return
    if T is Float64:
        compare_float64(unsafe_bitcast[Tensor[Float64]](a), unsafe_bitcast[Tensor[Float64]](b), out_mask, mode)
        return

    var (ok, br) = prepare_broadcast(a._shape, a._strides, b._shape, b._strides)
    if not ok: return

    var out_n = numel(br.shape)
    var R = len(br.shape)
    var i = 0
    while i < out_n:
        var ai = 0
        var bi = 0
        var tmp = i
        var d = R - 1
        while d >= 0:
            var dim = br.shape[d]
            var od = tmp % dim
            tmp = tmp // dim
            ai = ai + od * br.lhs_strides[d]
            bi = bi + od * br.rhs_strides[d]
            d -= 1

        var A = a._data[ai]
        var B = b._data[bi]
        var res: Int = 0
        if mode == 0: res = 1 if A == B else 0
        elif mode == 1: res = 1 if A != B else 0
        elif mode == 2: res = 1 if A <  B else 0
        elif mode == 3: res = 1 if A <= B else 0
        elif mode == 4: res = 1 if A >  B else 0
        else:           res = 1 if A >= B else 0
        out_mask._data[i] = res
        i += 1

# Broadcast two batch shapes (align-right). Unlimited rank.
fn _broadcast_batch(a: List[Int], b: List[Int]) -> (Bool, List[Int]):
    var la = len(a)
    var lb = len(b)
    var L = la
    if lb > L:
        L = lb

    # Build in reverse order, then flip once (since List has no prepend)
    var rev = List[Int]()
    rev.reserve(L)

    var ia = la - 1
    var ib = lb - 1
    var pos = L - 1
    while pos >= 0:
        var da = 1
        var db = 1
        if ia >= 0:
            da = a[ia]
        if ib >= 0:
            db = b[ib]

        if da == db:
            rev.append(da)
        elif da == 1:
            rev.append(db)
        elif db == 1:
            rev.append(da)
        else:
            return (False, List[Int]())

        if ia >= 0:
            ia = ia - 1
        if ib >= 0:
            ib = ib - 1
        if pos == 0:
            break
        pos = pos - 1

    # reverse `rev` into `out`
    var out = List[Int]()
    out.reserve(len(rev))
    var i = len(rev) - 1
    while i >= 0:
        out.append(rev[i])
        if i == 0:
            break
        i = i - 1

    return (True, out.copy())




# Effective strides for broadcast: 0 stride when expanded.
fn broadcast_batch_strides(in_shape: List[Int], in_strides: List[Int], out_shape: List[Int]) -> List[Int]:
    var la = len(in_shape)
    var L = len(out_shape)

    # Build in reverse (since List has no prepend), then reverse once.
    var rev = List[Int]()
    rev.reserve(L)

    var ia = la - 1
    var pos = L - 1
    while pos >= 0:
        var dim_out = out_shape[pos]
        var dim_in = 1
        var stride_in = 0
        if ia >= 0:
            dim_in = in_shape[ia]
            stride_in = in_strides[ia]

        if dim_in == dim_out:
            rev.append(stride_in)
        elif dim_in == 1:
            rev.append(0)
        else:
            # unreachable if caller validated broadcast-compatibility
            rev.append(0)

        if ia >= 0:
            ia = ia - 1
        if pos == 0:
            break
        pos = pos - 1

    # reverse rev -> eff
    var eff = List[Int]()
    eff.reserve(len(rev))
    var i = len(rev) - 1
    while i >= 0:
        eff.append(rev[i])
        if i == 0:
            break
        i = i - 1
    
    return eff.copy()

@always_inline
fn _zero_of[T: ImplicitlyCopyable & Copyable & Movable]() -> T:
    # Handle common numeric & Bool types explicitly.
    if T is Float64: return unsafe_bitcast[T](Float64(0.0))
    if T is Float32: return unsafe_bitcast[T](Float32(0.0))

    if T is Int:     return unsafe_bitcast[T](Int(0))
    if T is Int64:   return unsafe_bitcast[T](Int64(0))
    if T is Int32:   return unsafe_bitcast[T](Int32(0))
    if T is Int16:   return unsafe_bitcast[T](Int16(0))
    if T is Int8:    return unsafe_bitcast[T](Int8(0))

    if T is UInt64:  return unsafe_bitcast[T](UInt64(0))
    if T is UInt32:  return unsafe_bitcast[T](UInt32(0))
    if T is UInt16:  return unsafe_bitcast[T](UInt16(0))
    if T is UInt8:   return unsafe_bitcast[T](UInt8(0))

    if T is Bool:    return unsafe_bitcast[T](Bool(False))

    # Fallback: if you have other scalar types, add cases above.
    # As a last resort, prefer first-term initialization in your kernels
    # (e.g., acc = a*b; loop from j=1) so you never need a "typed zero".
    return unsafe_bitcast[T](Int(0))
# ================================================================
# Core kernel in Float64: A[..., M, N] @ x[..., N] -> [..., M]
# ================================================================
# matrix-vector (batched) matmul: A(..., M, N) @ x(..., N) -> y(..., M) 
@always_inline
fn matmul_core_vec(A: Tensor[Float64], x: Tensor[Float64]) -> Tensor[Float64]:
    var rA = len(A._shape)
    var rx = len(x._shape)
    if rA != 2 or rx != 1: return empty_f64()
    var M = A._shape[0]
    var N = A._shape[1]
    if x._shape[0] != N: return empty_f64()

    # ---- fast path: contiguous row-major ----
    if is_row_major_contiguous(A._shape, A._strides) \
       and is_row_major_contiguous(x._shape, x._strides):
        var y = List[Float64](); y.reserve(M)
        var i = 0
        var base = 0
        var unroll = 8
        while i < M:
            var acc = 0.0
            var j = 0
            var lim = (N // unroll) * unroll
            while j < lim:
                acc = acc + A._data[base + (j    )] * x._data[(j    )]
                acc = acc + A._data[base + (j + 1)] * x._data[(j + 1)]
                acc = acc + A._data[base + (j + 2)] * x._data[(j + 2)]
                acc = acc + A._data[base + (j + 3)] * x._data[(j + 3)]
                acc = acc + A._data[base + (j + 4)] * x._data[(j + 4)]
                acc = acc + A._data[base + (j + 5)] * x._data[(j + 5)]
                acc = acc + A._data[base + (j + 6)] * x._data[(j + 6)]
                acc = acc + A._data[base + (j + 7)] * x._data[(j + 7)]
                j = j + 8
            while j < N:
                acc = acc + A._data[base + j] * x._data[j]
                j = j + 1
            y.append(acc)
            base = base + N
            i = i + 1
        return Tensor[Float64](y, [M])

    # ---- generic 2D@1D بدون broadcast (خیلی مهم: قبل از هر broadcast) ----
    var sAm = A._strides[0]
    var sAn = A._strides[1]
    var sXn = x._strides[0]
    var out = List[Float64](); out.reserve(M)

    var i2 = 0
    var un = 8
    while i2 < M:
        var acc2 = 0.0
        var k = 0
        var baseA = i2 * sAm
        var lim = (N // un) * un
        while k < lim:
            acc2 = acc2 + A._data[baseA + (k    ) * sAn] * x._data[(k    ) * sXn]
            acc2 = acc2 + A._data[baseA + (k + 1) * sAn] * x._data[(k + 1) * sXn]
            acc2 = acc2 + A._data[baseA + (k + 2) * sAn] * x._data[(k + 2) * sXn]
            acc2 = acc2 + A._data[baseA + (k + 3) * sAn] * x._data[(k + 3) * sXn]
            acc2 = acc2 + A._data[baseA + (k + 4) * sAn] * x._data[(k + 4) * sXn]
            acc2 = acc2 + A._data[baseA + (k + 5) * sAn] * x._data[(k + 5) * sXn]
            acc2 = acc2 + A._data[baseA + (k + 6) * sAn] * x._data[(k + 6) * sXn]
            acc2 = acc2 + A._data[baseA + (k + 7) * sAn] * x._data[(k + 7) * sXn]
            k = k + 8
        while k < N:
            acc2 = acc2 + A._data[baseA + k * sAn] * x._data[k * sXn]
            k = k + 1
        out.append(acc2)
        i2 = i2 + 1

    return Tensor[Float64](out, [M])


@always_inline
fn matmul_core_mm(A: Tensor[Float64], B: Tensor[Float64]) -> Tensor[Float64]:
    var rA = len(A._shape); var rB = len(B._shape)
    if rA != 2 or rB != 2: return empty_f64()
    var M = A._shape[0]; var K = A._shape[1]
    if B._shape[0] != K: return empty_f64()
    var N = B._shape[1]

    # contiguous fast path
    if is_row_major_contiguous(A._shape, A._strides) \
       and is_row_major_contiguous(B._shape, B._strides):
        var out = List[Float64](); out.reserve(M*N)
        var i = 0; var un = 8
        while i < M:
            var j = 0; var baseA = i * K
            while j < N:
                var acc = 0.0; var k = 0; var baseB = j
                var lim = (K // un) * un
                while k < lim:
                    acc = acc + A._data[baseA + (k    )] * B._data[(k    ) * N + baseB]
                    acc = acc + A._data[baseA + (k + 1)] * B._data[(k + 1) * N + baseB]
                    acc = acc + A._data[baseA + (k + 2)] * B._data[(k + 2) * N + baseB]
                    acc = acc + A._data[baseA + (k + 3)] * B._data[(k + 3) * N + baseB]
                    acc = acc + A._data[baseA + (k + 4)] * B._data[(k + 4) * N + baseB]
                    acc = acc + A._data[baseA + (k + 5)] * B._data[(k + 5) * N + baseB]
                    acc = acc + A._data[baseA + (k + 6)] * B._data[(k + 6) * N + baseB]
                    acc = acc + A._data[baseA + (k + 7)] * B._data[(k + 7) * N + baseB]
                    k = k + 8
                while k < K:
                    acc = acc + A._data[baseA + k] * B._data[k * N + baseB]
                    k = k + 1
                out.append(acc)
                j = j + 1
            i = i + 1
        return Tensor[Float64](out, [M, N])

    # generic strides
    var sAm = A._strides[0]; var sAk = A._strides[1]
    var sBk = B._strides[0]; var sBn = B._strides[1]
    var out2 = List[Float64](); out2.reserve(M*N)
    var ii = 0; var un2 = 8
    while ii < M:
        var jj = 0; var rowA = ii * sAm
        while jj < N:
            var acc2 = 0.0; var kk = 0; var colB = jj * sBn
            var lim2 = (K // un2) * un2
            while kk < lim2:
                acc2 = acc2 + A._data[rowA + (kk    ) * sAk] * B._data[(kk    ) * sBk + colB]
                acc2 = acc2 + A._data[rowA + (kk + 1) * sAk] * B._data[(kk + 1) * sBk + colB]
                acc2 = acc2 + A._data[rowA + (kk + 2) * sAk] * B._data[(kk + 2) * sBk + colB]
                acc2 = acc2 + A._data[rowA + (kk + 3) * sAk] * B._data[(kk + 3) * sBk + colB]
                acc2 = acc2 + A._data[rowA + (kk + 4) * sAk] * B._data[(kk + 4) * sBk + colB]
                acc2 = acc2 + A._data[rowA + (kk + 5) * sAk] * B._data[(kk + 5) * sBk + colB]
                acc2 = acc2 + A._data[rowA + (kk + 6) * sAk] * B._data[(kk + 6) * sBk + colB]
                acc2 = acc2 + A._data[rowA + (kk + 7) * sAk] * B._data[(kk + 7) * sBk + colB]
                kk = kk + 8
            while kk < K:
                acc2 = acc2 + A._data[rowA + kk * sAk] * B._data[kk * sBk + colB]
                kk = kk + 1
            out2.append(acc2)
            jj = jj + 1
        ii = ii + 1
    return Tensor[Float64](out2, [M, N])

 
@always_inline
fn matmul(self: Tensor[Float64], other: Tensor[Float64]) -> Tensor[Float64]:
    var r_self = len(self._shape)
    var r_other = len(other._shape)
    if r_self == 2 and r_other == 1:
        return matmul_core_vec(self, other)
    elif r_self == 2 and r_other == 2:
        return matmul_core_mm(self, other) 
    return empty_f64()




 
    
# ---------------- free function ----------------
# Contract last dim of A with first dim of B:
# A: (m x n), B: (n x p)  ->  C: (m x p)
# Float64 × Float64 → Float64
# A(..., M, K) ⨂ B(..., K, N)  with axes=1  ->  out(..., M, N)
@always_inline
fn tensordot(A: Tensor[Float64], B: Tensor[Float64], axes: Int = 1) -> Tensor[Float64]:
    # ----- validate axes and ranks -----
    if axes != 1:
        return empty_f64()
    var rA = len(A._shape)
    var rB = len(B._shape)
    if rA < 2 or rB < 2:
        return empty_f64()

    # dims
    var M = A._shape[rA - 2]
    var K = A._shape[rA - 1]
    var Kb = B._shape[rB - 2]  # contract with B's second-to-last axis
    var N = B._shape[rB - 1]
    if K != Kb:
        return empty_f64()

    # ================= fast paths =================

    # 1) pure 2D @ 2D contiguous  -> use flat, cache-friendly inner loops
    if rA == 2 and rB == 2 \
       and is_row_major_contiguous(A._shape, A._strides) \
       and is_row_major_contiguous(B._shape, B._strides):
        var out = List[Float64]()
        out.reserve(M * N)

        var i = 0
        var unroll = 8
        while i < M:
            var j = 0
            var baseA = i * K                  # row i of A
            while j < N:
                var acc = 0.0
                var k = 0
                var baseB = j                  # column j of B (row-major)
                var lim = (K // unroll) * unroll
                while k < lim:
                    acc = acc + A._data[baseA + (k    )] * B._data[(k    ) * N + baseB]
                    acc = acc + A._data[baseA + (k + 1)] * B._data[(k + 1) * N + baseB]
                    acc = acc + A._data[baseA + (k + 2)] * B._data[(k + 2) * N + baseB]
                    acc = acc + A._data[baseA + (k + 3)] * B._data[(k + 3) * N + baseB]
                    acc = acc + A._data[baseA + (k + 4)] * B._data[(k + 4) * N + baseB]
                    acc = acc + A._data[baseA + (k + 5)] * B._data[(k + 5) * N + baseB]
                    acc = acc + A._data[baseA + (k + 6)] * B._data[(k + 6) * N + baseB]
                    acc = acc + A._data[baseA + (k + 7)] * B._data[(k + 7) * N + baseB]
                    k = k + 8
                while k < K:
                    acc = acc + A._data[baseA + k] * B._data[k * N + baseB]
                    k = k + 1
                out.append(acc)
                j = j + 1
            i = i + 1

        return Tensor[Float64](out, [M, N])

    # 2) batched contiguous with identical batch shape  -> linear scans per batch
    var same_batch = True
    var ba = rA - 2
    var bb = rB - 2
    if ba != bb:
        same_batch = False
    var d = 0
    while d < ba and d < bb:
        if A._shape[d] != B._shape[d]:
            same_batch = False
        d = d + 1

    if is_row_major_contiguous(A._shape, A._strides) \
       and is_row_major_contiguous(B._shape, B._strides) \
       and same_batch \
       and ba >= 1:
        # batch_elems = product over batch dims
        var batch_elems = 1
        var t = 0
        while t < ba:
            batch_elems = batch_elems * A._shape[t]
            t = t + 1

        var out2 = List[Float64]()
        out2.reserve(batch_elems * M * N)

        var bidx = 0
        var unroll_b = 8
        while bidx < batch_elems:
            var offA = bidx * (M * K)       # each A-batch is M*K
            var offB = bidx * (K * N)       # each B-batch is K*N

            var i2 = 0
            while i2 < M:
                var j2 = 0
                var baseA2 = offA + i2 * K
                while j2 < N:
                    var acc2 = 0.0
                    var k2 = 0
                    var baseB2 = offB + j2
                    var lim2 = (K // unroll_b) * unroll_b
                    while k2 < lim2:
                        acc2 = acc2 + A._data[baseA2 + (k2    )] * B._data[offB + (k2    ) * N + j2]
                        acc2 = acc2 + A._data[baseA2 + (k2 + 1)] * B._data[offB + (k2 + 1) * N + j2]
                        acc2 = acc2 + A._data[baseA2 + (k2 + 2)] * B._data[offB + (k2 + 2) * N + j2]
                        acc2 = acc2 + A._data[baseA2 + (k2 + 3)] * B._data[offB + (k2 + 3) * N + j2]
                        acc2 = acc2 + A._data[baseA2 + (k2 + 4)] * B._data[offB + (k2 + 4) * N + j2]
                        acc2 = acc2 + A._data[baseA2 + (k2 + 5)] * B._data[offB + (k2 + 5) * N + j2]
                        acc2 = acc2 + A._data[baseA2 + (k2 + 6)] * B._data[offB + (k2 + 6) * N + j2]
                        acc2 = acc2 + A._data[baseA2 + (k2 + 7)] * B._data[offB + (k2 + 7) * N + j2]
                        k2 = k2 + 8
                    while k2 < K:
                        acc2 = acc2 + A._data[baseA2 + k2] * B._data[offB + k2 * N + j2]
                        k2 = k2 + 1
                    out2.append(acc2)
                    j2 = j2 + 1
                i2 = i2 + 1

            bidx = bidx + 1

        # out shape = batch_shape + [M, N]
        var out_shape2 = List[Int]()
        var q = 0
        while q < ba:
            out_shape2.append(A._shape[q])
            q = q + 1
        out_shape2.append(M)
        out_shape2.append(N)
        return Tensor[Float64](out2, out_shape2)

    # ================= generic path (arbitrary strides; up to 5D batches) =================

    # 1) collect batch shapes (exclude last 2 dims of A and last 2 dims of B)
    var batchA = List[Int]()
    var iA = 0
    while iA < rA - 2:
        batchA.append(A._shape[iA])
        iA = iA + 1

    var batchB = List[Int]()
    var iB = 0
    while iB < rB - 2:
        batchB.append(B._shape[iB])
        iB = iB + 1

    # 2) broadcast only if any batch dim exists
    var Bshape = List[Int]()
    var ok = True
    if (rA - 2) > 0 or (rB - 2) > 0:
        var res = _broadcast_batch(batchA, batchB)   # -> (Bool, List[Int])
        ok = res[0]
        Bshape = res[1].copy()
        if not ok:
            return empty_f64()

    # 3) broadcasted batch strides
    var A_batch_strides_src = List[Int]()
    var a_s = 0
    while a_s < rA - 2:
        A_batch_strides_src.append(A._strides[a_s])
        a_s = a_s + 1

    var B_batch_strides_src = List[Int]()
    var b_s = 0
    while b_s < rB - 2:
        B_batch_strides_src.append(B._strides[b_s])
        b_s = b_s + 1

    var A_bstrides = broadcast_batch_strides(batchA, A_batch_strides_src, Bshape)
    var B_bstrides = broadcast_batch_strides(batchB, B_batch_strides_src, Bshape)

    # 4) inner (contracted/output) strides
    var strideA_M = A._strides[rA - 2]
    var strideA_K = A._strides[rA - 1]
    var strideB_K = B._strides[rB - 2]
    var strideB_N = B._strides[rB - 1]

    # 5) output shape = Bshape + [M, N]
    var out_shape = List[Int]()
    var b = 0
    while b < len(Bshape):
        out_shape.append(Bshape[b])
        b = b + 1
    out_shape.append(M)
    out_shape.append(N)

    var out = List[Float64]()
    out.reserve(numel(out_shape))

    # 6) up to 5D nested loops over broadcasted batch dims (Bn <= 5)
    var Bn = len(Bshape)
    var B0 = 1; var B1 = 1; var B2 = 1; var B3 = 1; var B4 = 1
    if Bn > 0: B0 = Bshape[0]
    if Bn > 1: B1 = Bshape[1]
    if Bn > 2: B2 = Bshape[2]
    if Bn > 3: B3 = Bshape[3]
    if Bn > 4: B4 = Bshape[4]

    var sA0 = 0; var sA1 = 0; var sA2 = 0; var sA3 = 0; var sA4 = 0
    var sB0 = 0; var sB1 = 0; var sB2 = 0; var sB3 = 0; var sB4 = 0
    if Bn > 0: sA0 = A_bstrides[0]; sB0 = B_bstrides[0]
    if Bn > 1: sA1 = A_bstrides[1]; sB1 = B_bstrides[1]
    if Bn > 2: sA2 = A_bstrides[2]; sB2 = B_bstrides[2]
    if Bn > 3: sA3 = A_bstrides[3]; sB3 = B_bstrides[3]
    if Bn > 4: sA4 = A_bstrides[4]; sB4 = B_bstrides[4]

    var un = 8

    var b0 = 0
    while b0 < B0:
        var b1 = 0
        while b1 < B1:
            var b2 = 0
            while b2 < B2:
                var b3 = 0
                while b3 < B3:
                    var b4 = 0
                    while b4 < B4:
                        var offA = b0 * sA0 + b1 * sA1 + b2 * sA2 + b3 * sA3 + b4 * sA4
                        var offB = b0 * sB0 + b1 * sB1 + b2 * sB2 + b3 * sB3 + b4 * sB4

                        var i = 0
                        while i < M:
                            var j = 0
                            var baseA = offA + i * strideA_M
                            while j < N:
                                var acc = 0.0
                                var k = 0
                                var baseB = offB + j * strideB_N
                                var lim = (K // un) * un
                                while k < lim:
                                    acc = acc + A._data[baseA + (k    ) * strideA_K] * B._data[offB + (k    ) * strideB_K + j * strideB_N]
                                    acc = acc + A._data[baseA + (k + 1) * strideA_K] * B._data[offB + (k + 1) * strideB_K + j * strideB_N]
                                    acc = acc + A._data[baseA + (k + 2) * strideA_K] * B._data[offB + (k + 2) * strideB_K + j * strideB_N]
                                    acc = acc + A._data[baseA + (k + 3) * strideA_K] * B._data[offB + (k + 3) * strideB_K + j * strideB_N]
                                    acc = acc + A._data[baseA + (k + 4) * strideA_K] * B._data[offB + (k + 4) * strideB_K + j * strideB_N]
                                    acc = acc + A._data[baseA + (k + 5) * strideA_K] * B._data[offB + (k + 5) * strideB_K + j * strideB_N]
                                    acc = acc + A._data[baseA + (k + 6) * strideA_K] * B._data[offB + (k + 6) * strideB_K + j * strideB_N]
                                    acc = acc + A._data[baseA + (k + 7) * strideA_K] * B._data[offB + (k + 7) * strideB_K + j * strideB_N]
                                    k = k + 8
                                while k < K:
                                    acc = acc + A._data[baseA + k * strideA_K] * B._data[offB + k * strideB_K + j * strideB_N]
                                    k = k + 1
                                out.append(acc)
                                j = j + 1
                            i = i + 1

                        b4 = b4 + 1
                    b3 = b3 + 1
                b2 = b2 + 1
            b1 = b1 + 1
        b0 = b0 + 1

    return Tensor[Float64](out, out_shape)


 

# ============================================================
# Compatibility helpers kept (used elsewhere in codebase)
# ============================================================

@always_inline
fn align_right(a: List[Int], b: List[Int]) -> (List[Int], List[Int]):
    var la = len(a)
    var lb = len(b)
    if la == lb:
        return (a, b)
    if la < lb:
        var pad = lb - la
        var aa = List[Int](); aa.reserve(lb)
        var i = 0
        while i < pad:
            aa.append(1)
            i += 1
        var j = 0
        while j < la:
            aa.append(a[j])
            j += 1
        return (aa, b)
    var pad2 = la - lb
    var bb = List[Int](); bb.reserve(la)
    var p = 0
    while p < pad2:
        bb.append(1)
        p += 1
    var q = 0
    while q < lb:
        bb.append(b[q])
        q += 1
    return (a, bb)

@always_inline
fn keepdims_shape(shape: List[Int], ax: Int) -> List[Int]:
    var r = len(shape)
    var out = List[Int](); out.reserve(r)
    var i = 0
    while i < r:
        if i == ax:
            out.append(1)
        else:
            out.append(shape[i])
        i += 1
    return out.copy()

# Broadcasting shape-compatibility checks
# - NumPy-style: trailing alignment, missing leading dims treated as 1
# - Strict: ranks must match, and each aligned pair must be equal or 1
# Notes: English-only comments (Momijo rule), var-only, @always_inline for speed.

@always_inline
fn _dims_compatible(ad: Int, bd: Int) -> Bool:
    # Both must be positive; zero/negatives are invalid here.
    if ad <= 0 or bd <= 0: 
        return False
    # Equal or one of them is 1
    return ad == bd or ad == 1 or bd == 1


# -------------------------------
# NumPy-style broadcasting check
# -------------------------------
@always_inline
fn can_broadcast_shapes_numpy(a: List[Int], b: List[Int]) -> Bool:
    var ra = len(a)
    var rb = len(b)

    var ia = ra - 1
    var ib = rb - 1

    # Max steps equals the larger rank
    var steps = ra
    if rb > steps:
        steps = rb

    var k = 0
    while k < steps:
        # Treat missing leading dims as 1
        var ad = 1
        var bd = 1
        if ia >= 0:
            ad = a[ia]
        if ib >= 0:
            bd = b[ib]

        if not _dims_compatible(ad, bd):
            return False

        ia = ia - 1
        ib = ib - 1
        k = k + 1

    return True


# -------------------------------
# Strict broadcasting check
# (Ranks must match)
# -------------------------------
@always_inline
fn can_broadcast_shapes_strict(a: List[Int], b: List[Int]) -> Bool:
    var ra = len(a)
    var rb = len(b)
    if ra != rb:
        return False

    var i = 0
    while i < ra:
        # No padding; dimensions align 1:1
        var ad = a[i]
        var bd = b[i]
        if not _dims_compatible(ad, bd):
            return False
        i = i + 1

    return True


# ----------------------------------------------
# Wrapper with compile-time behavior selection
# Set STRICT=true to enforce strict broadcasting
# ----------------------------------------------
 

@always_inline
fn can_broadcast_shapes(a: List[Int], b: List[Int] ,strict: Bool = False) -> Bool:
    if strict:
        return can_broadcast_shapes_strict(a, b)
    else:
        return can_broadcast_shapes_numpy(a, b)
 
#@always_inline
#fn can_broadcast_shapes(a: List[Int], b: List[Int]) -> Bool:
#    var ra = len(a)
#    var rb = len(b)
#
#    var ia = ra - 1
#    var ib = rb - 1
#
#    var r :Int
#    if ra >= rb:
#        r = ra
#    else:
#        r = rb
#
#    var steps = 0
#    while steps < r:
#        var ad = 1
#        var bd = 1
#
#        if ia >= 0:
#            ad = a[ia]
#        if ib >= 0:
#            bd = b[ib]
#
#        var ok = False
#        if ad == bd:
#            ok = True
#        else:
#            if ad == 1:
#                ok = True
#            else:
#                if bd == 1:
#                    ok = True
#
#        if not ok:
#            return False
#
#        ia = ia - 1
#        ib = ib - 1
#        steps = steps + 1
#
#    return True
#
#
# 