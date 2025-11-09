# Project:      Momijo
# Module:       tensor.math
# File:         math.mojo
# Path:         src/momijo/tensor/math.mojo
#
# Description:
#   High-performance math ops for Tensor[T]: unary, binary (+broadcast),
#   scalar pow/clip/lerp/normalize, comparisons, dot, and reductions.
#   - Explicit imports (no wildcards)
#   - Tight loops with simple unrolling
#   - Defensive code paths (no assertions)
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT

from collections.list import List
from momijo.tensor.tensor import Tensor

from momijo.tensor.cast import *
from momijo.tensor.creation import empty_tensor
# Helper imports (explicit)
from momijo.tensor.broadcast import broadcast_shapes
from momijo.tensor.helpers import same_shape
from momijo.tensor.helpers import is_contiguous
from momijo.tensor.helpers import ensure_strides
from momijo.tensor.helpers import row_major_strides
from momijo.tensor.helpers import numel,is_row_major_contiguous
from momijo.tensor.broadcast import keepdims_shape,clip
from momijo.tensor.helpers import normalize_axis,unravel_index ,lin_index
from momijo.tensor.helpers import shape_drop_axis ,astype_with ,zero_scalar_of ,compute_row_major_strides


from math import *
# ======================= Math helpers (Float64) =======================

@always_inline
fn abs64(x: Float64) -> Float64:
    # Returns |x|
    if x >= 0.0:
        return x
    return -x

@always_inline
fn floor64(x: Float64) -> Float64:
    # Largest integer <= x
    var i = Int(x)
    var fi = Float64(i)
    if x >= 0.0:
        return fi
    if fi == x:
        return x
    return Float64(i - 1)

@always_inline
fn ceil64(x: Float64) -> Float64:
    # Smallest integer >= x
    var i = Int(x)
    var fi = Float64(i)
    if x <= 0.0:
        if fi == x:
            return x
        return Float64(i)
    if fi == x:
        return x
    return Float64(i + 1)

@always_inline
fn round64(x: Float64) -> Float64:
    # Round half-up
    if x >= 0.0:
        return floor64(x + 0.5)
    return ceil64(x - 0.5)

@always_inline
fn sign64(x: Float64) -> Float64:
    # Returns -1.0, 0.0, or 1.0
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0

@always_inline
fn log1p64(x: Float64) -> Float64:
    # Numerically stable log(1+x); series for small |x|
    var ax = abs64(x)
    if ax < 1e-6:
        var x2 = x * x
        var x3 = x2 * x
        var x4 = x2 * x2
        # log1p(x) ≈ x - x^2/2 + x^3/3 - x^4/4
        return x - 0.5 * x2 + (x3 / 3.0) - 0.25 * x4
    var y = 1.0 + x
    return log64(y)   # <-- fixed: call log64, not log1p64

@always_inline
fn expm1_64(x: Float64) -> Float64:
    # Numerically stable exp(x)-1; series for small |x|
    var ax = abs64(x)
    if ax < 1e-6:
        var x2 = x * x
        var x3 = x2 * x
        var x4 = x2 * x2
        # expm1(x) ≈ x + x^2/2 + x^3/6 + x^4/24
        return x + 0.5 * x2 + (x3 / 6.0) + (x4 / 24.0)
    return exp64(x) - 1.0

@always_inline
fn safe_div(a: Float64, b: Float64, eps: Float64 = 1e-12) -> Float64:
    # Divide with tiny bias to avoid 0-division; preserves sign of denominator
    var d = b
    if d >= 0.0:
        d = d + eps
    else:
        d = d - eps
    return a / d

@always_inline
fn safe_sqrt(x: Float64, eps: Float64 = 1e-12) -> Float64:
    # Clamp negatives to 0 and add epsilon before sqrt
    var v = x
    if v < 0.0:
        v = 0.0
    return sqrt64(v + eps)   # <-- fixed: use sqrt64 wrapper
# ====================== Extra Float64 helpers ======================
@always_inline
fn sigmoid64(x: Float64) -> Float64:
    # 1 / (1 + exp(-x))
    return 1.0 / (1.0 + exp64(-x))

@always_inline
fn tanh64(x: Float64) -> Float64:
    # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    var e2x = exp64(2.0 * x)
    return (e2x - 1.0) / (e2x + 1.0)

@always_inline
fn silu64(x: Float64) -> Float64:
    # SiLU(x) = x * sigmoid(x)
    return x * sigmoid64(x)

@always_inline
fn gelu64(x: Float64) -> Float64:
    # GELU approx (tanh-based):
    # 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
    var k = 0.7978845608028654    # sqrt(2/pi)
    var c = 0.044715
    var x3 = x * x * x
    var t = k * (x + c * x3)
    return 0.5 * x * (1.0 + tanh64(t))

@always_inline
fn elu64(x: Float64) -> Float64:
    # ELU with alpha = 1.0 (common default)
    if x >= 0.0:
        return x
    return exp64(x) - 1.0

@always_inline
fn selu64(x: Float64) -> Float64:
    # SELU with standard constants:
    # lambda = 1.0507009873554804, alpha = 1.6732632423543772
    var lambda_v = 1.0507009873554804
    var alpha_v  = 1.6732632423543772
    var y:Float64
    if x >= 0.0:
        y = x
    else:
        y = alpha_v * (exp64(x) - 1.0)
    return lambda_v * y


#############################

@always_inline
fn abs32(x: Float32) -> Float32:
    # Returns |x|
    if x >= Float32(0.0):
        return x
    return -x

@always_inline
fn floor32(x: Float32) -> Float32:
    # Largest integer <= x
    var i = Int(x)
    var fi = Float32(i)
    if x >= Float32(0.0):
        return fi
    if fi == x:
        return x
    return Float32(i - 1)

@always_inline
fn ceil32(x: Float32) -> Float32:
    # Smallest integer >= x
    var i = Int(x)
    var fi = Float32(i)
    if x <= Float32(0.0):
        if fi == x:
            return x
        return Float32(i)
    if fi == x:
        return x
    return Float32(i + 1)

@always_inline
fn round32(x: Float32) -> Float32:
    # Round half-up
    if x >= Float32(0.0):
        return floor32(x + Float32(0.5))
    return ceil32(x - Float32(0.5))

@always_inline
fn sign32(x: Float32) -> Float32:
    # Returns -1.0, 0.0, or 1.0
    if x > Float32(0.0):
        return Float32(1.0)
    if x < Float32(0.0):
        return Float32(-1.0)
    return Float32(0.0)

@always_inline
fn log1p32(x: Float32) -> Float32:
    # Numerically stable log(1+x); series for small |x|
    var ax = abs32(x)
    if ax < Float32(1e-6):
        var x2 = x * x
        var x3 = x2 * x
        var x4 = x2 * x2
        # log1p(x) ≈ x - x^2/2 + x^3/3 - x^4/4
        return x - Float32(0.5) * x2 + (x3 / Float32(3.0)) - Float32(0.25) * x4
    var y = Float32(1.0) + x
    return log32(y)

@always_inline
fn expm1_32(x: Float32) -> Float32:
    # Numerically stable exp(x)-1; series for small |x|
    var ax = abs32(x)
    if ax < Float32(1e-6):
        var x2 = x * x
        var x3 = x2 * x
        var x4 = x2 * x2
        # expm1(x) ≈ x + x^2/2 + x^3/6 + x^4/24
        return x + Float32(0.5) * x2 + (x3 / Float32(6.0)) + (x4 / Float32(24.0))
    return exp32(x) - Float32(1.0)

@always_inline
fn safe_div32(a: Float32, b: Float32, eps: Float32 = Float32(1e-7)) -> Float32:
    # Divide with tiny bias to avoid 0-division; preserves sign of denominator
    var d = b
    if d >= Float32(0.0):
        d = d + eps
    else:
        d = d - eps
    return a / d

@always_inline
fn safe_sqrt32(x: Float32, eps: Float32 = Float32(1e-7)) -> Float32:
    # Clamp negatives to 0 and add epsilon before sqrt
    var v = x
    if v < Float32(0.0):
        v = Float32(0.0)
    return sqrt32(v + eps)

# ====================== Extra Float32 helpers ======================

@always_inline
fn sigmoid32(x: Float32) -> Float32:
    # 1 / (1 + exp(-x))
    return Float32(1.0) / (Float32(1.0) + exp32(-x))

@always_inline
fn tanh32(x: Float32) -> Float32:
    # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    var e2x = exp32(Float32(2.0) * x)
    return (e2x - Float32(1.0)) / (e2x + Float32(1.0))

@always_inline
fn silu32(x: Float32) -> Float32:
    # SiLU(x) = x * sigmoid(x)
    return x * sigmoid32(x)

@always_inline
fn gelu32(x: Float32) -> Float32:
    # GELU approx (tanh-based):
    # 0.5 * x * (1 + tanh( sqrt(2/pi)*(x + 0.044715*x^3) ))
    var k = Float32(0.7978845608028654)   # sqrt(2/pi)
    var c = Float32(0.044715)
    var x3 = x * x * x
    var t = k * (x + c * x3)
    return Float32(0.5) * x * (Float32(1.0) + tanh32(t))

@always_inline
fn elu32(x: Float32) -> Float32:
    # ELU with alpha = 1.0
    if x >= Float32(0.0):
        return x
    return exp32(x) - Float32(1.0)

@always_inline
fn selu32(x: Float32) -> Float32:
    # SELU with standard constants (Float32)
    var lambda_v = Float32(1.0507009873554804)
    var alpha_v  = Float32(1.6732632423543772)
    var y: Float32
    if x >= Float32(0.0):
        y = x
    else:
        y = alpha_v * (exp32(x) - Float32(1.0))
    return lambda_v * y
# ====================== Softmax (stable, last-dim) ======================
@always_inline
fn _softmax_lastdim_f64(x: Tensor[Float64]) -> Tensor[Float64]:
    # Assumes row-major contiguous storage and applies softmax along the last axis.
    var shape = x._shape.copy()
    var rank  = len(shape)
    #assert(rank >= 1)

    var last = shape[rank - 1]
    #assert(last > 0)

    # total elements and number of vectors (outer) of length 'last'
    var n_total = 1
    var d = 0
    while d < rank:
        n_total = n_total * shape[d]
        d += 1
    var outer = n_total // last

    var src = x._data.copy()
    var out = List[Float64]()
    out.reserve(n_total)

    # We'll fill out with placeholder then mutate (since List has only append)
    var i = 0
    while i < n_total:
        out.append(0.0)
        i += 1

    var o = 0
    while o < outer:
        var base = o * last

        # 1) find max
        var m = src[base]
        var j = 1
        while j < last:
            var v = src[base + j]
            if v > m:
                m = v
            j += 1

        # 2) exp(x - m) and sum
        var sum_exp = 0.0
        j = 0
        while j < last:
            var e = exp64(src[base + j] - m)
            out[base + j] = e
            sum_exp = sum_exp + e
            j += 1

        # 3) normalize
        var inv = 1.0 / sum_exp
        j = 0
        while j < last:
            out[base + j] = out[base + j] * inv
            j += 1

        o += 1

    return Tensor[Float64](out, shape)


@always_inline
fn _softmax_lastdim_f32(x: Tensor[Float32]) -> Tensor[Float32]:
    # Assumes row-major contiguous storage and applies softmax along the last axis.
    var shape = x._shape.copy()
    var rank  = len(shape)
    #assert(rank >= 1)

    var last = shape[rank - 1]
    #assert(last > 0)

    # total elements and number of vectors (outer) of length 'last'
    var n_total = 1
    var d = 0
    while d < rank:
        n_total = n_total * shape[d]
        d += 1
    var outer = n_total // last

    var src = x._data.copy()
    var out = List[Float32]()
    out.reserve(n_total)

    # We'll fill out with placeholder then mutate (since List has only append)
    var i = 0
    while i < n_total:
        out.append(0.0)
        i += 1

    var o = 0
    while o < outer:
        var base = o * last

        # 1) find max
        var m = src[base]
        var j = 1
        while j < last:
            var v = src[base + j]
            if v > m:
                m = v
            j += 1

        # 2) exp(x - m) and sum
        var sum_exp:Float32 = 0.0
        j = 0
        while j < last:
            var e = exp32(src[base + j] - m)
            out[base + j] = e
            sum_exp = sum_exp + e
            j += 1

        # 3) normalize
        var inv = 1.0 / sum_exp
        j = 0
        while j < last:
            out[base + j] = out[base + j] * inv
            j += 1

        o += 1

    return Tensor[Float32](out, shape)

# Optional: 1D fast-path (calls the same impl but keeps intent clear)
@always_inline
fn _softmax1d_f64(x: Tensor[Float64]) -> Tensor[Float64]:
    #assert(len(x._shape) == 1)
    return _softmax_lastdim_f64(x)

# ====================== Public API (axis = -1 only) ======================
@always_inline
fn softmax(x: Tensor[Float64], axis: Int = -1) -> Tensor[Float64]:
    # Only last-dimension supported here; extend as needed.
    #assert(axis == -1)   # extend for other axes if/when needed
    return _softmax_lastdim_f64(x)

@always_inline
fn softmax(x: Tensor[Float32], axis: Int = -1) -> Tensor[Float32]:
    return _softmax_lastdim_f32(x)

@always_inline
fn softmax(x: Tensor[Int], axis: Int = -1) -> Tensor[Float32]:
    var xf32 = to_float32(x)
    return softmax(xf32, axis)


# Soft row-wise max via softmax-weighted expectation (smooth approx)
fn _row_max_approx(x: tensor.Tensor[Float64], k: Float64 = 64.0) -> tensor.Tensor[Float64]:
    var kx = x * k
    var w = kx.exp()
    var Z = _row_sum(w) + 1e-12     # avoid divide-by-zero
    var p = w / Z
    return _row_sum(x * p)
# Stable log_softmax per row
fn log_softmax(logits: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
    var m = _row_max_approx(logits)
    var z = logits - m
    var e = z.exp()
    var s = _row_sum(e) + 1e-12
    var lse = tensor.log(s) + m
    return logits - lse
# ======================= Unary core =======================
# ===== Float64 math helpers (assumes these exist in your math module) =====
# If your names differ (e.g., sqrt), replace the calls below accordingly.
fn sqrt64(x: Float64) -> Float64: return sqrt(x)
fn exp64 (x: Float64) -> Float64: return exp(x)
fn log64 (x: Float64) -> Float64: return log(x)
fn sin64 (x: Float64) -> Float64: return sin(x)
fn cos64 (x: Float64) -> Float64: return cos(x)
fn tan64 (x: Float64) -> Float64: return tan(x)
fn pow64 (a: Float64, b: Float64) -> Float64: return a.(b)


fn sqrt32(x: Float32) -> Float32: return sqrt(x)
fn exp32 (x: Float32) -> Float32: return exp(x)
fn log32 (x: Float32) -> Float32: return log(x)
fn sin32 (x: Float32) -> Float32: return sin(x)
fn cos32 (x: Float32) -> Float32: return cos(x)
fn tan32 (x: Float32) -> Float32: return tan(x)
fn pow32 (a: Float32, b: Float32) -> Float32: return a.(b)

# ====================== Unary core (via converters) ======================
# uop_id mapping (extended):
# 0: neg, 1: abs, 2: sqrt, 3: exp, 4: log, 5: sin, 6: cos, 7: tan,
# 8: relu, 9: expm1, 10: log1p, 11: floor, 12: ceil, 13: round, 14: sign,
# 15: sigmoid, 16: tanh, 17: silu, 18: gelu, 19: elu(alpha=1), 20: selu
@always_inline
fn unary_eval_impl[T: ImplicitlyCopyable & Copyable & Movable](
    x: T,
    uop_id: Int,
    to_f64: fn (T) -> Float64,
    from_f64: fn (Float64) -> T
) -> T:
    var xf = to_f64(x)

    if uop_id == 0:  return from_f64(-xf)
    if uop_id == 1:  return from_f64(xf if xf >= 0.0 else -xf)
    if uop_id == 2:  return from_f64(sqrt64(xf))
    if uop_id == 3:  return from_f64(exp64(xf))
    if uop_id == 4:  return from_f64(log64(xf))
    if uop_id == 5:  return from_f64(sin64(xf))
    if uop_id == 6:  return from_f64(cos64(xf))
    if uop_id == 7:  return from_f64(tan64(xf))
    if uop_id == 8:
        var yf = xf
        if yf < 0.0: yf = 0.0
        return from_f64(yf)
    if uop_id == 9:  return from_f64(expm1_64(xf))
    if uop_id == 10: return from_f64(log1p64(xf))
    if uop_id == 11: return from_f64(floor64(xf))
    if uop_id == 12: return from_f64(ceil64(xf))
    if uop_id == 13: return from_f64(round64(xf))
    if uop_id == 15: return from_f64(sigmoid64(xf))
    if uop_id == 16: return from_f64(tanh64(xf))
    if uop_id == 17: return from_f64(silu64(xf))
    if uop_id == 18: return from_f64(gelu64(xf))
    if uop_id == 19: return from_f64(elu64(xf))
    if uop_id == 20: return from_f64(selu64(xf))

    # default: sign (14)
    return from_f64(sign64(xf))


fn apply_unary_impl[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    uop_id: Int,
    to_f64: fn (T) -> Float64,
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    var n = len(x._data)
    var out = List[T]()
    out.reserve(n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(unary_eval_impl[T](x._data[i    ], uop_id, to_f64, from_f64))
        out.append(unary_eval_impl[T](x._data[i + 1], uop_id, to_f64, from_f64))
        out.append(unary_eval_impl[T](x._data[i + 2], uop_id, to_f64, from_f64))
        out.append(unary_eval_impl[T](x._data[i + 3], uop_id, to_f64, from_f64))
        out.append(unary_eval_impl[T](x._data[i + 4], uop_id, to_f64, from_f64))
        out.append(unary_eval_impl[T](x._data[i + 5], uop_id, to_f64, from_f64))
        out.append(unary_eval_impl[T](x._data[i + 6], uop_id, to_f64, from_f64))
        out.append(unary_eval_impl[T](x._data[i + 7], uop_id, to_f64, from_f64))
        i += 8
    while i < n:
        out.append(unary_eval_impl[T](x._data[i], uop_id, to_f64, from_f64))
        i += 1
    return Tensor[T](out, x._shape)

@always_inline
fn unary_eval_impl32[T: ImplicitlyCopyable & Copyable & Movable](
    x: T,
    uop_id: Int,
    to_f32: fn (T) -> Float32,
    from_f32: fn (Float32) -> T
) -> T:
    var xf = to_f32(x)

    if uop_id == 0:  return from_f32(-xf)
    if uop_id == 1:  return from_f32(xf if xf >= 0.0 else -xf)
    if uop_id == 2:  return from_f32(sqrt32(xf))
    if uop_id == 3:  return from_f32(exp32(xf))
    if uop_id == 4:  return from_f32(log32(xf))
    if uop_id == 5:  return from_f32(sin32(xf))
    if uop_id == 6:  return from_f32(cos32(xf))
    if uop_id == 7:  return from_f32(tan32(xf))
    if uop_id == 8:
        var yf = xf
        if yf < 0.0: yf = 0.0
        return from_f32(yf)
    if uop_id == 9:  return from_f32(expm1_32(xf))
    if uop_id == 10: return from_f32(log1p32(xf))
    if uop_id == 11: return from_f32(floor32(xf))
    if uop_id == 12: return from_f32(ceil32(xf))
    if uop_id == 13: return from_f32(round32(xf))
    if uop_id == 15: return from_f32(sigmoid32(xf))
    if uop_id == 16: return from_f32(tanh32(xf))
    if uop_id == 17: return from_f32(silu32(xf))
    if uop_id == 18: return from_f32(gelu32(xf))
    if uop_id == 19: return from_f32(elu32(xf))
    if uop_id == 20: return from_f32(selu32(xf))

    # default: sign (14)
    return from_f32(sign32(xf))


fn apply_unary_impl32[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    uop_id: Int,
    to_f32: fn (T) -> Float32,
    from_f32: fn (Float32) -> T
) -> Tensor[T]:
    var n = len(x._data)
    var out = List[T]()
    out.reserve(n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(unary_eval_impl32[T](x._data[i    ], uop_id, to_f32, from_f32))
        out.append(unary_eval_impl32[T](x._data[i + 1], uop_id, to_f32, from_f32))
        out.append(unary_eval_impl32[T](x._data[i + 2], uop_id, to_f32, from_f32))
        out.append(unary_eval_impl32[T](x._data[i + 3], uop_id, to_f32, from_f32))
        out.append(unary_eval_impl32[T](x._data[i + 4], uop_id, to_f32, from_f32))
        out.append(unary_eval_impl32[T](x._data[i + 5], uop_id, to_f32, from_f32))
        out.append(unary_eval_impl32[T](x._data[i + 6], uop_id, to_f32, from_f32))
        out.append(unary_eval_impl32[T](x._data[i + 7], uop_id, to_f32, from_f32))
        i += 8
    while i < n:
        out.append(unary_eval_impl32[T](x._data[i], uop_id, to_f32, from_f32))
        i += 1
    return Tensor[T](out, x._shape)


# ---- public unary (per-dtype) ----
# Import the tensor-level converters from your cast module
@always_inline
fn apply_unary(x: Tensor[Float64], uop_id: Int) -> Tensor[Float64]:
    # Already Float64; apply directly in Float64 space.
    return apply_unary_impl[Float64](x, uop_id, to_float64_of, f64_to)

@always_inline
fn apply_unary(x: Tensor[Float32], uop_id: Int) -> Tensor[Float32]:
    return apply_unary_impl32[Float32](x, uop_id, to_float32_of, f32_to)

@always_inline
fn apply_unary(x: Tensor[Int8], uop_id: Int) -> Tensor[Float64]:
    var xf64 = to_float64(x)
    return apply_unary_impl[Float64](xf64, uop_id, to_float64_of, f64_to)

@always_inline
fn apply_unary(x: Tensor[Int16], uop_id: Int) -> Tensor[Float64]:
    var xf64 = to_float64(x)
    return apply_unary_impl[Float64](xf64, uop_id, to_float64_of, f64_to)

@always_inline
fn apply_unary(x: Tensor[Int32], uop_id: Int) -> Tensor[Float64]:
    var xf64 = to_float64(x)
    return apply_unary_impl[Float64](xf64, uop_id, to_float64_of, f64_to)

@always_inline
fn apply_unary(x: Tensor[Int64], uop_id: Int) -> Tensor[Float64]:
    var xf64 = to_float64(x)
    return apply_unary_impl[Float64](xf64, uop_id, to_float64_of, f64_to)

@always_inline
fn apply_unary(x: Tensor[Int], uop_id: Int) -> Tensor[Float64]:
    var xf64 = to_float64(x)
    return apply_unary_impl[Float64](xf64, uop_id, to_float64_of, f64_to)

@always_inline
fn apply_unary(x: Tensor[UInt8], uop_id: Int) -> Tensor[Float64]:
    var xf64 = to_float64(x)
    return apply_unary_impl[Float64](xf64, uop_id, to_float64_of, f64_to)

@always_inline
fn apply_unary(x: Tensor[UInt16], uop_id: Int) -> Tensor[Float64]:
    var xf64 = to_float64(x)
    return apply_unary_impl[Float64](xf64, uop_id, to_float64_of, f64_to)

@always_inline
fn apply_unary(x: Tensor[UInt32], uop_id: Int) -> Tensor[Float64]:
    var xf64 = to_float64(x)
    return apply_unary_impl[Float64](xf64, uop_id, to_float64_of, f64_to)

@always_inline
fn apply_unary(x: Tensor[UInt64], uop_id: Int) -> Tensor[Float64]:
    var xf64 = to_float64(x)
    return apply_unary_impl[Float64](xf64, uop_id, to_float64_of, f64_to)


# Small wrappers matching your names
# -------- NEG (op 0) --------
@always_inline
fn neg_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 0)
@always_inline
fn neg_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 0)
@always_inline
fn neg_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 0)


# -------- ABS (op 1) --------
@always_inline
fn abs_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 1)
@always_inline
fn abs_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 1)
@always_inline
fn abs_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 1)

# -------- SQRT (op 2) --------
@always_inline
fn sqrt_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 2)
@always_inline
fn sqrt_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 2)
@always_inline
fn sqrt_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 2)


# -------- EXP (op 3) --------
@always_inline
fn exp_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 3)
@always_inline
fn exp_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 3)
@always_inline
fn exp_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 3)


# -------- LOG (op 4) --------
@always_inline
fn log_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 4)
@always_inline
fn log_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 4)
@always_inline
fn log_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 4)


# -------- SIN (op 5) --------
@always_inline
fn sin_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 5)
@always_inline
fn sin_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 5)
@always_inline
fn sin_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 5)

# -------- COS (op 6) --------
@always_inline
fn cos_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 6)
@always_inline
fn cos_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 6)
@always_inline
fn cos_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 6)


# -------- TAN (op 7) --------
@always_inline
fn tan_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 7)
@always_inline
fn tan_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 7)
@always_inline
fn tan_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 7)


# -------- RELU (op 8) --------
@always_inline
fn relu_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 8)
@always_inline
fn relu_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 8)
@always_inline
fn relu_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 8)

# -------- EXPM1 (op 9) --------
@always_inline
fn expm1_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 9)
@always_inline
fn expm1_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 9)
@always_inline
fn expm1_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 9)


# -------- LOG1P (op 10) --------
@always_inline
fn log1p_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 10)
@always_inline
fn log1p_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 10)
@always_inline
fn log1p_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 10)


# -------- FLOOR (op 11) --------
@always_inline
fn floor_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 11)
@always_inline
fn floor_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 11)
@always_inline
fn floor_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 11)


# -------- CEIL (op 12) --------
@always_inline
fn ceil_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 12)
@always_inline
fn ceil_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 12)
@always_inline
fn ceil_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 12)


# -------- ROUND (op 13) --------
@always_inline
fn round_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 13)
@always_inline
fn round_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 13)
@always_inline
fn round_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 13)

# -------- SIGN (op 14) --------
@always_inline
fn sign_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 14)
@always_inline
fn sign_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 14)
@always_inline
fn sign_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 14)

# -------- SIGMOID (op 15) --------
@always_inline
fn sigmoid_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 15)
@always_inline
fn sigmoid_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 15)
@always_inline
fn sigmoid_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 15)

# -------- TANH (op 16) --------
@always_inline
fn tanh_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 16)
@always_inline
fn tanh_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 16)
@always_inline
fn tanh_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 16)

# -------- SiLU (op 17) --------
@always_inline
fn silu_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 17)
@always_inline
fn silu_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 17)
@always_inline
fn silu_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 17)

# -------- GELU (op 18) --------
@always_inline
fn gelu_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 18)
@always_inline
fn gelu_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 18)
@always_inline
fn gelu_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 18)

# -------- ELU (alpha=1) (op 19) --------
@always_inline
fn elu_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 19)
@always_inline
fn elu_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 19)
@always_inline
fn elu_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 19)

# -------- SELU (op 20) --------
@always_inline
fn selu_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 20)
@always_inline
fn selu_t(x: Tensor[Float32]) -> Tensor[Float32]:
    return apply_unary(x, 20)
@always_inline
fn selu_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 20)


# =========================
# Fast helpers (row-major)
# =========================

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

@always_inline
fn _offset_from_coords(strides: List[Int], off: Int, coords: List[Int]) -> Int:
    var o = off
    var d = 0
    while d < len(coords):
        o = o + coords[d] * strides[d]
        d += 1
    return o

# ====================== Binary: comparisons (mask Float64) ======================

@always_inline
fn _cmp_eval_impl[T: ImplicitlyCopyable & Copyable & Movable](
    a: T, b: T, cmp_id: Int, to_f64: fn (T) -> Float64
) -> Float64:
    var af = to_f64(a)
    var bf = to_f64(b)
    if cmp_id == 0:   return 1.0 if af == bf else 0.0
    if cmp_id == 1:   return 1.0 if af != bf else 0.0
    if cmp_id == 2:   return 1.0 if af <  bf else 0.0
    if cmp_id == 3:   return 1.0 if af <= bf else 0.0
    if cmp_id == 4:   return 1.0 if af >  bf else 0.0
    return 1.0 if af >= bf else 0.0

@always_inline
fn _apply_compare_impl[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    y: Tensor[T],
    cmp_id: Int,
    to_f64: fn (T) -> Float64
) -> Tensor[Float64]:
    # Scalar RHS fast path
    if len(y._shape) == 0:
        var n = 1
        var d = 0
        while d < len(x._shape):
            n = n * x._shape[d]
            d += 1

        var out = List[Float64]()
        out.reserve(n)
        if n == 0:
            var shp0 = List[Int](); shp0.append(0)
            var str0 = compute_row_major_strides(shp0)
            return Tensor[Float64](out, shp0, str0, 0)

        var rm_x = _row_major_multipliers(x._shape)
        var sval = y._data[y._offset]
        var lin = 0
        while lin < n:
            var offx = _offset_from_linear(x._shape, x._strides, x._offset, rm_x, lin)
            out.append(_cmp_eval_impl[T](x._data[offx], sval, cmp_id, to_f64))
            lin += 1
        var strx = compute_row_major_strides(x._shape)
        return Tensor[Float64](out, x._shape, strx, 0)

    # General path: shape-equal, stride/offset-aware
    var nlog = 1
    var i = 0
    while i < len(x._shape):
        nlog = nlog * x._shape[i]
        i += 1

    var out = List[Float64]()
    out.reserve(nlog)
    if nlog == 0:
        var shp0 = List[Int](); shp0.append(0)
        var str0 = compute_row_major_strides(shp0)
        return Tensor[Float64](out, shp0, str0, 0)

    var rm_x = _row_major_multipliers(x._shape)
    var rm_y = _row_major_multipliers(y._shape)

    var lin2 = 0
    while lin2 < nlog:
        var offx = _offset_from_linear(x._shape, x._strides, x._offset, rm_x, lin2)
        var offy = _offset_from_linear(y._shape, y._strides, y._offset, rm_y, lin2)
        out.append(_cmp_eval_impl[T](x._data[offx], y._data[offy], cmp_id, to_f64))
        lin2 += 1

    var str = compute_row_major_strides(x._shape)
    return Tensor[Float64](out, x._shape, str, 0)



# Per-dtype dispatch (mirrors your unary style)
@always_inline
fn _apply_compare(x: Tensor[Float64], y: Tensor[Float64], cmp_id: Int) -> Tensor[Int]:
    var mf64 = _apply_compare_impl[Float64](x, y, cmp_id, to_float64_of)
    return to_int(mf64)


@always_inline
fn _apply_compare(x: Tensor[Float32], y: Tensor[Float32], cmp_id: Int) -> Tensor[Int]:
    var mf64 = _apply_compare_impl[Float32](x, y, cmp_id, to_float64_of)
    return to_int(mf64)


@always_inline
fn _apply_compare(x: Tensor[Int], y: Tensor[Int], cmp_id: Int) -> Tensor[Int]:
    var mf64 = _apply_compare_impl[Int](x, y, cmp_id, to_float64_of)
    return to_int(mf64)


# -------- EQ / NE / LT / LE / GT / GE --------
@always_inline
fn eq_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]: return _apply_compare(x, y, 0)
@always_inline
fn eq_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]: return _apply_compare(x, y, 0)
@always_inline
fn eq_t(x: Tensor[Int],     y: Tensor[Int])     -> Tensor[Int]: return _apply_compare(x, y, 0)


@always_inline
fn ne_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]: return _apply_compare(x, y, 1)
@always_inline
fn ne_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]: return _apply_compare(x, y, 1)
@always_inline
fn ne_t(x: Tensor[Int],     y: Tensor[Int])     -> Tensor[Int]: return _apply_compare(x, y, 1)


@always_inline
fn lt_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]: return _apply_compare(x, y, 2)
@always_inline
fn lt_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]: return _apply_compare(x, y, 2)
@always_inline
fn lt_t(x: Tensor[Int],     y: Tensor[Int])     -> Tensor[Int]: return _apply_compare(x, y, 2)


@always_inline
fn le_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]: return _apply_compare(x, y, 3)
@always_inline
fn le_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]: return _apply_compare(x, y, 3)
@always_inline
fn le_t(x: Tensor[Int],     y: Tensor[Int])     -> Tensor[Int]: return _apply_compare(x, y, 3)


@always_inline
fn gt_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]: return _apply_compare(x, y, 4)
@always_inline
fn gt_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]: return _apply_compare(x, y, 4)
@always_inline
fn gt_t(x: Tensor[Int],     y: Tensor[Int])     -> Tensor[Int]: return _apply_compare(x, y, 4)


@always_inline
fn ge_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]: return _apply_compare(x, y, 5)
@always_inline
fn ge_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]: return _apply_compare(x, y, 5)
@always_inline
fn ge_t(x: Tensor[Int],     y: Tensor[Int])     -> Tensor[Int]: return _apply_compare(x, y, 5)




# -------- Public API: AND / OR / XOR / NAND / NOR / XNOR / ANDNOT / NOT --------


# Repeat the tiny wrappers for other dtypes as needed (or call apply_unary directly).

# ==================== Binary core (+ broadcast) via converters ====================
@always_inline
fn bin_combine_impl[T: ImplicitlyCopyable & Copyable & Movable](
    a: T, b: T, op_id: Int,
    to_f64: fn (T) -> Float64,
    from_f64: fn (Float64) -> T
) -> T:
    var af = to_f64(a)
    var bf = to_f64(b)
    var ia = 0
    if to_f64(a) != 0.0: ia = 1
    var ib = 0
    if to_f64(b) != 0.0: ib = 1

    var rf: Float64 = 0.0
    if op_id == 0:            # add
        rf = af + bf
    elif op_id == 1:          # sub
        rf = af - bf
    elif op_id == 2:          # mul
        rf = af * bf
    elif op_id == 3:          # div (avoid integer '/')
        rf = af / bf
    elif op_id == 4:          # pow
        rf = pow(af, bf)
    elif op_id == 5:          # max
        rf = af if af >= bf else bf
    elif op_id == 6:                    # min
        rf = af if af <= bf else bf
    elif op_id == 7:
        rf =af % bf

    elif op_id == 10 or op_id == 20 :            # AND
        rf = ia & ib
    elif op_id == 11 or op_id == 21 :          # OR
        rf = ia | ib
    elif op_id == 12 or op_id == 22 :          # XOR
        rf = ia ^ ib
    elif op_id == 13 or op_id == 23 :          # NAND

        rf = 1 - (ia & ib)
    elif op_id == 14 or op_id == 24 :          # NOR
        rf = 1 - (ia | ib)
    elif op_id == 15 or op_id == 25 :          # XNOR
        rf = 1 - (ia ^ ib)
    elif op_id == 16 or op_id == 26 :          # ANDNOT (a & ~b)
        rf = ia & (1 - ib)


    return from_f64(rf)
@always_inline
fn elemwise2_same_contig_impl[T: ImplicitlyCopyable & Copyable & Movable](
    a: Tensor[T], b: Tensor[T], op_id: Int,
    to_f64: fn (T) -> Float64,
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    var n = len(a._data)
    var out_tensor = Tensor[T](shape=a._shape, fill=zero_scalar_of[T](from_f64))

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out_tensor._data[i    ] = bin_combine_impl[T](a._data[i    ], b._data[i    ], op_id, to_f64, from_f64)
        out_tensor._data[i + 1] = bin_combine_impl[T](a._data[i + 1], b._data[i + 1], op_id, to_f64, from_f64)
        out_tensor._data[i + 2] = bin_combine_impl[T](a._data[i + 2], b._data[i + 2], op_id, to_f64, from_f64)
        out_tensor._data[i + 3] = bin_combine_impl[T](a._data[i + 3], b._data[i + 3], op_id, to_f64, from_f64)
        out_tensor._data[i + 4] = bin_combine_impl[T](a._data[i + 4], b._data[i + 4], op_id, to_f64, from_f64)
        out_tensor._data[i + 5] = bin_combine_impl[T](a._data[i + 5], b._data[i + 5], op_id, to_f64, from_f64)
        out_tensor._data[i + 6] = bin_combine_impl[T](a._data[i + 6], b._data[i + 6], op_id, to_f64, from_f64)
        out_tensor._data[i + 7] = bin_combine_impl[T](a._data[i + 7], b._data[i + 7], op_id, to_f64, from_f64)
        i += 8
    while i < n:
        out_tensor._data[i] = bin_combine_impl[T](a._data[i], b._data[i], op_id, to_f64, from_f64)
        i += 1

    return out_tensor.copy()

@always_inline
fn elemwise_scalar_right_impl[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], s: T, op_id: Int,
    to_f64: fn (T) -> Float64,
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    var n = len(x._data)
    var out_tensor = Tensor[T](shape=x._shape, fill=zero_scalar_of[T](from_f64))

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out_tensor._data[i    ] = bin_combine_impl[T](x._data[i    ], s, op_id, to_f64, from_f64)
        out_tensor._data[i + 1] = bin_combine_impl[T](x._data[i + 1], s, op_id, to_f64, from_f64)
        out_tensor._data[i + 2] = bin_combine_impl[T](x._data[i + 2], s, op_id, to_f64, from_f64)
        out_tensor._data[i + 3] = bin_combine_impl[T](x._data[i + 3], s, op_id, to_f64, from_f64)
        out_tensor._data[i + 4] = bin_combine_impl[T](x._data[i + 4], s, op_id, to_f64, from_f64)
        out_tensor._data[i + 5] = bin_combine_impl[T](x._data[i + 5], s, op_id, to_f64, from_f64)
        out_tensor._data[i + 6] = bin_combine_impl[T](x._data[i + 6], s, op_id, to_f64, from_f64)
        out_tensor._data[i + 7] = bin_combine_impl[T](x._data[i + 7], s, op_id, to_f64, from_f64)
        i += 8
    while i < n:
        out_tensor._data[i] = bin_combine_impl[T](x._data[i], s, op_id, to_f64, from_f64)
        i += 1

    return out_tensor.copy()

@always_inline
fn elemwise_scalar_left_impl[T: ImplicitlyCopyable & Copyable & Movable](
    s: T, x: Tensor[T], op_id: Int,
    to_f64: fn (T) -> Float64,
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    var n = len(x._data)
    var out_tensor = Tensor[T](shape=x._shape, fill=zero_scalar_of[T](from_f64))

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out_tensor._data[i    ] = bin_combine_impl[T](s, x._data[i    ], op_id, to_f64, from_f64)
        out_tensor._data[i + 1] = bin_combine_impl[T](s, x._data[i + 1], op_id, to_f64, from_f64)
        out_tensor._data[i + 2] = bin_combine_impl[T](s, x._data[i + 2], op_id, to_f64, from_f64)
        out_tensor._data[i + 3] = bin_combine_impl[T](s, x._data[i + 3], op_id, to_f64, from_f64)
        out_tensor._data[i + 4] = bin_combine_impl[T](s, x._data[i + 4], op_id, to_f64, from_f64)
        out_tensor._data[i + 5] = bin_combine_impl[T](s, x._data[i + 5], op_id, to_f64, from_f64)
        out_tensor._data[i + 6] = bin_combine_impl[T](s, x._data[i + 6], op_id, to_f64, from_f64)
        out_tensor._data[i + 7] = bin_combine_impl[T](s, x._data[i + 7], op_id, to_f64, from_f64)
        i += 8
    while i < n:
        out_tensor._data[i] = bin_combine_impl[T](s, x._data[i], op_id, to_f64, from_f64)
        i += 1

    return out_tensor.copy()

@always_inline
fn new_tensor_from_list[T: ImplicitlyCopyable & Copyable & Movable](
    data: List[T], shape: List[Int],
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    var init = zero_scalar_of[T](from_f64)
    var t = Tensor[T](shape=shape, fill=init)

    var n_expected = numel(shape)
    var n_src = len(data)
    var k = n_src if n_src < n_expected else n_expected

    var i = 0
    while i < k:
        t._data[i] = data[i]
        i += 1

    return t.copy()

@always_inline
fn apply_broadcast2_impl[T: ImplicitlyCopyable & Copyable & Movable](
    a: Tensor[T], b: Tensor[T], op_id: Int,
    to_f64: fn (T) -> Float64,
    from_f64: fn (Float64) -> T
) -> Tensor[T]:
    var ashp = a._shape.copy()
    var bshp = b._shape.copy()

    var rest = broadcast_shapes(ashp.copy(), bshp.copy())
    if not rest.ok:
        return empty_tensor[T]()

    var oshp = rest.shape.copy()
    var apad = rest.lhs_padded.copy()
    var bpad = rest.rhs_padded.copy()

    # Legacy equal-length 1D path (kept for compatibility)
    if len(oshp) == 0:
        var n = len(a._data)
        var m = len(b._data)
        var k = n if n < m else m
        var out_list = List[T]()
        out_list.reserve(k)
        var i0 = 0
        while i0 < k:
            out_list.append(bin_combine_impl[T](a._data[i0], b._data[i0], op_id, to_f64, from_f64))
            i0 += 1
        var shp1 = List[Int](); shp1.append(k)
        return new_tensor_from_list[T](out_list, shp1, from_f64)

    var an = len(a._data)
    var bn = len(b._data)

    if an == bn and same_shape(ashp, bshp)
       and is_row_major_contiguous(ashp, a._strides)
       and is_row_major_contiguous(bshp, b._strides):
        return elemwise2_same_contig_impl[T](a, b, op_id, to_f64, from_f64)

    if bn == 1:
        return elemwise_scalar_right_impl[T](a, b._data[0], op_id, to_f64, from_f64)
    if an == 1:
        return elemwise_scalar_left_impl[T](a._data[0], b, op_id, to_f64, from_f64)

    # General N-D broadcast
    var a_str = ensure_strides(a._strides, apad)   # fixed: use padded shape
    var b_str = ensure_strides(b._strides, bpad)   # fixed: use padded shape
    var o_str = row_major_strides(oshp)

    var rank_o = len(oshp)
    var rank_a = len(apad)
    var rank_b = len(bpad)

    var n_out = numel(oshp)
    var out_tensor = Tensor[T](shape=oshp, fill=zero_scalar_of[T](from_f64))

    var idx = 0
    while idx < n_out:
        var rem = idx
        var oa = 0
        var ob = 0
        var d = 0
        while d < rank_o:
            var step = o_str[d]
            var cur = 0
            if step != 0:
                cur = (rem // step) % oshp[d]
            var ai = rank_a - rank_o + d
            var bi = rank_b - rank_o + d
            if ai >= 0 and apad[ai] != 1:
                oa = oa + cur * a_str[ai]
            if bi >= 0 and bpad[bi] != 1:
                ob = ob + cur * b_str[bi]
            if step != 0:
                rem = rem % step
            d += 1

        out_tensor._data[idx] = bin_combine_impl[T](a._data[oa], b._data[ob], op_id, to_f64, from_f64)
        idx += 1

    return out_tensor.copy()


@always_inline
fn apply_broadcast2_inplace_impl[T: ImplicitlyCopyable & Copyable & Movable](
    mut x: Tensor[T], y: Tensor[T], op_id: Int,
    to_f64: fn (T) -> Float64,
    from_f64: fn (Float64) -> T
) -> None:
    var xshp = x._shape.copy()
    var yshp = y._shape.copy()

    var rest = broadcast_shapes(xshp.copy(), yshp.copy())
    if not rest.ok:
        return

    var oshp = rest.shape.copy()
    var xpad = rest.lhs_padded.copy()
    var ypad = rest.rhs_padded.copy()

    # Fast path: identical shapes and row-major contiguous on both sides
    if same_shape(xshp, yshp)
       and is_row_major_contiguous(xshp, x._strides)
       and is_row_major_contiguous(yshp, y._strides):
        var n = len(x._data)
        var i = 0
        var lim = (n // 8) * 8
        while i < lim:
            x._data[i    ] = bin_combine_impl[T](x._data[i    ], y._data[i    ], op_id, to_f64, from_f64)
            x._data[i + 1] = bin_combine_impl[T](x._data[i + 1], y._data[i + 1], op_id, to_f64, from_f64)
            x._data[i + 2] = bin_combine_impl[T](x._data[i + 2], y._data[i + 2], op_id, to_f64, from_f64)
            x._data[i + 3] = bin_combine_impl[T](x._data[i + 3], y._data[i + 3], op_id, to_f64, from_f64)
            x._data[i + 4] = bin_combine_impl[T](x._data[i + 4], y._data[i + 4], op_id, to_f64, from_f64)
            x._data[i + 5] = bin_combine_impl[T](x._data[i + 5], y._data[i + 5], op_id, to_f64, from_f64)
            x._data[i + 6] = bin_combine_impl[T](x._data[i + 6], y._data[i + 6], op_id, to_f64, from_f64)
            x._data[i + 7] = bin_combine_impl[T](x._data[i + 7], y._data[i + 7], op_id, to_f64, from_f64)
            i += 8
        while i < n:
            x._data[i] = bin_combine_impl[T](x._data[i], y._data[i], op_id, to_f64, from_f64)
            i += 1
        return

    # Legacy equal-length 1D fallback (when oshp is empty)
    if len(oshp) == 0:
        var xn = len(x._data)
        var yn = len(y._data)
        var k = xn if xn < yn else yn
        var i0 = 0
        while i0 < k:
            x._data[i0] = bin_combine_impl[T](x._data[i0], y._data[i0], op_id, to_f64, from_f64)
            i0 += 1
        return

    # General N-D broadcast (use padded shapes for correct indexing)
    var x_str = ensure_strides(x._strides, xpad)
    var y_str = ensure_strides(y._strides, ypad)
    var o_str = row_major_strides(oshp)

    var rank_o = len(oshp)
    var rank_x = len(xpad)
    var rank_y = len(ypad)
    var n_out = numel(oshp)

    var idx = 0
    while idx < n_out:
        var rem = idx
        var ox = 0
        var oy = 0
        var d = 0
        while d < rank_o:
            var step = o_str[d]
            var cur = 0
            if step != 0:
                cur = (rem // step) % oshp[d]
            var xi = rank_x - rank_o + d
            var yi = rank_y - rank_o + d
            if xi >= 0 and xpad[xi] != 1:
                ox = ox + cur * x_str[xi]
            if yi >= 0 and ypad[yi] != 1:
                oy = oy + cur * y_str[yi]
            if step != 0:
                rem = rem % step
            d += 1
        x._data[ox] = bin_combine_impl[T](x._data[ox], y._data[oy], op_id, to_f64, from_f64)
        idx += 1


# ---- public broadcasted binary (per-dtype) ----
# Typed conversions to Float64
# --- typed converters: T -> Float64 ---
# ---- Typed conversions to Float64 (one per dtype) ----



@always_inline
fn apply_broadcast5(a: Tensor[Float64], b: Tensor[Float64], op_id: Int) -> Tensor[Bool]:
    return to_bool(apply_broadcast2_impl[Float64](a, b, op_id, to_f64_f64, f64_to))

@always_inline
fn apply_broadcast5(a: Tensor[Float32], b: Tensor[Float32], op_id: Int) -> Tensor[Bool]:
    return to_bool(apply_broadcast2_impl[Float32](a, b, op_id, to_f64_f32, f64_to_float32))

@always_inline
fn apply_broadcast5(a: Tensor[Int8], b: Tensor[Int8], op_id: Int) -> Tensor[Bool]:
    return to_bool(apply_broadcast2_impl[Int8](a, b, op_id, to_f64_i8, f64_to_int8))

@always_inline
fn apply_broadcast5(a: Tensor[Int16], b: Tensor[Int16], op_id: Int) -> Tensor[Bool]:
    return to_bool(apply_broadcast2_impl[Int16](a, b, op_id, to_f64_i16, f64_to_int16))

@always_inline
fn apply_broadcast5(a: Tensor[Int32], b: Tensor[Int32], op_id: Int) -> Tensor[Bool]:
    return to_bool(apply_broadcast2_impl[Int32](a, b, op_id, to_f64_i32, f64_to_int32))

@always_inline
fn apply_broadcast5(a: Tensor[Int64], b: Tensor[Int64], op_id: Int) -> Tensor[Bool]:
    return to_bool(apply_broadcast2_impl[Int64](a, b, op_id, to_f64_i64, f64_to_int64))

@always_inline
fn apply_broadcast5(a: Tensor[Int], b: Tensor[Int], op_id: Int) -> Tensor[Bool]:
    return to_bool(apply_broadcast2_impl[Int](a, b, op_id, to_f64_int, f64_to_int))

@always_inline
fn apply_broadcast5(a: Tensor[UInt8], b: Tensor[UInt8], op_id: Int) -> Tensor[Bool]:
    return to_bool(apply_broadcast2_impl[UInt8](a, b, op_id, to_f64_u8, f64_to_uint8))

@always_inline
fn apply_broadcast5(a: Tensor[UInt16], b: Tensor[UInt16], op_id: Int) -> Tensor[Bool]:
    return to_bool(apply_broadcast2_impl[UInt16](a, b, op_id, to_f64_u16, f64_to_uint16))

@always_inline
fn apply_broadcast5(a: Tensor[UInt32], b: Tensor[UInt32], op_id: Int) -> Tensor[Bool]:
    return to_bool(apply_broadcast2_impl[UInt32](a, b, op_id, to_f64_u32, f64_to_uint32))

@always_inline
fn apply_broadcast5(a: Tensor[UInt64], b: Tensor[UInt64], op_id: Int) -> Tensor[Bool]:
    return to_bool(apply_broadcast2_impl[UInt64](a, b, op_id, to_f64_u64, f64_to_uint64))

@always_inline
fn apply_broadcast5(a: Tensor[Bool], b: Tensor[Bool], op_id: Int) -> Tensor[Bool]:
    return to_bool(apply_broadcast2_impl[Bool](a, b, op_id, to_f64_bool, f64_to_bool))


@always_inline
fn apply_broadcast4(a: Tensor[Float64], b: Tensor[Float64], op_id: Int) -> Tensor[Int]:
    # Inputs read as Float64; results written as Int.
    return to_int(apply_broadcast2_impl[Float64](a, b, op_id, to_f64_f64, f64_to))

@always_inline
fn apply_broadcast4(a: Tensor[Float32], b: Tensor[Float32], op_id: Int) -> Tensor[Int]:
    return to_int(apply_broadcast2_impl[Float32](a, b, op_id, to_f64_f32, f64_to_float32))

@always_inline
fn apply_broadcast4(a: Tensor[Int8], b: Tensor[Int8], op_id: Int) -> Tensor[Int]:
    return to_int(apply_broadcast2_impl[Int8](a, b, op_id, to_f64_i8, f64_to_int8))

@always_inline
fn apply_broadcast4(a: Tensor[Int16], b: Tensor[Int16], op_id: Int) -> Tensor[Int]:
    return to_int(apply_broadcast2_impl[Int16](a, b, op_id, to_f64_i16, f64_to_int16))

@always_inline
fn apply_broadcast4(a: Tensor[Int32], b: Tensor[Int32], op_id: Int) -> Tensor[Int]:
    return to_int(apply_broadcast2_impl[Int32](a, b, op_id, to_f64_i32, f64_to_int32))

@always_inline
fn apply_broadcast4(a: Tensor[Int64], b: Tensor[Int64], op_id: Int) -> Tensor[Int]:
    return to_int(apply_broadcast2_impl[Int64](a, b, op_id, to_f64_i64, f64_to_int64))

@always_inline
fn apply_broadcast4(a: Tensor[Int], b: Tensor[Int], op_id: Int) -> Tensor[Int]:
    return to_int(apply_broadcast2_impl[Int](a, b, op_id, to_f64_int, f64_to_int))

@always_inline
fn apply_broadcast4(a: Tensor[UInt8], b: Tensor[UInt8], op_id: Int) -> Tensor[Int]:
    return to_int(apply_broadcast2_impl[UInt8](a, b, op_id, to_f64_u8, f64_to_uint8))

@always_inline
fn apply_broadcast4(a: Tensor[UInt16], b: Tensor[UInt16], op_id: Int) -> Tensor[Int]:
    return to_int(apply_broadcast2_impl[UInt16](a, b, op_id, to_f64_u16, f64_to_uint16))

@always_inline
fn apply_broadcast4(a: Tensor[UInt32], b: Tensor[UInt32], op_id: Int) -> Tensor[Int]:
    return to_int(apply_broadcast2_impl[UInt32](a, b, op_id, to_f64_u32, f64_to_uint32))

@always_inline
fn apply_broadcast4(a: Tensor[UInt64], b: Tensor[UInt64], op_id: Int) -> Tensor[Int]:
    return to_int(apply_broadcast2_impl[UInt64](a, b, op_id, to_f64_u64, f64_to_uint64))




@always_inline
fn apply_broadcast3(a: Tensor[Float64], b: Tensor[Float64], op_id: Int) -> Tensor[Float64]:
    return apply_broadcast2_impl[Float64](a, b, op_id, to_f64_f64, f64_to)

@always_inline
fn apply_broadcast3(a: Tensor[Float32], b: Tensor[Float32], op_id: Int) -> Tensor[Float64]:
    var af64 = to_float64(a)
    var bf64 = to_float64(b)
    return apply_broadcast2_impl[Float64](af64, bf64, op_id, to_f64_f64, f64_to)

@always_inline
fn apply_broadcast3(a: Tensor[Int8], b: Tensor[Int8], op_id: Int) -> Tensor[Float64]:
    var af64 = to_float64(a)
    var bf64 = to_float64(b)
    return apply_broadcast2_impl[Float64](af64, bf64, op_id, to_f64_f64, f64_to)

@always_inline
fn apply_broadcast3(a: Tensor[Int16], b: Tensor[Int16], op_id: Int) -> Tensor[Float64]:
    var af64 = to_float64(a)
    var bf64 = to_float64(b)
    return apply_broadcast2_impl[Float64](af64, bf64, op_id, to_f64_f64, f64_to)

@always_inline
fn apply_broadcast3(a: Tensor[Int32], b: Tensor[Int32], op_id: Int) -> Tensor[Float64]:
    var af64 = to_float64(a)
    var bf64 = to_float64(b)
    return apply_broadcast2_impl[Float64](af64, bf64, op_id, to_f64_f64, f64_to)

@always_inline
fn apply_broadcast3(a: Tensor[Int64], b: Tensor[Int64], op_id: Int) -> Tensor[Float64]:
    var af64 = to_float64(a)
    var bf64 = to_float64(b)
    return apply_broadcast2_impl[Float64](af64, bf64, op_id, to_f64_f64, f64_to)

@always_inline
fn apply_broadcast3(a: Tensor[Int], b: Tensor[Int], op_id: Int) -> Tensor[Float64]:
    var af64 = to_float64(a)
    var bf64 = to_float64(b)
    return apply_broadcast2_impl[Float64](af64, bf64, op_id, to_f64_f64, f64_to)

@always_inline
fn apply_broadcast3(a: Tensor[UInt8], b: Tensor[UInt8], op_id: Int) -> Tensor[Float64]:
    var af64 = to_float64(a)
    var bf64 = to_float64(b)
    return apply_broadcast2_impl[Float64](af64, bf64, op_id, to_f64_f64, f64_to)

@always_inline
fn apply_broadcast3(a: Tensor[UInt16], b: Tensor[UInt16], op_id: Int) -> Tensor[Float64]:
    var af64 = to_float64(a)
    var bf64 = to_float64(b)
    return apply_broadcast2_impl[Float64](af64, bf64, op_id, to_f64_f64, f64_to)

@always_inline
fn apply_broadcast3(a: Tensor[UInt32], b: Tensor[UInt32], op_id: Int) -> Tensor[Float64]:
    var af64 = to_float64(a)
    var bf64 = to_float64(b)
    return apply_broadcast2_impl[Float64](af64, bf64, op_id, to_f64_f64, f64_to)

@always_inline
fn apply_broadcast3(a: Tensor[UInt64], b: Tensor[UInt64], op_id: Int) -> Tensor[Float64]:
    var af64 = to_float64(a)
    var bf64 = to_float64(b)
    return apply_broadcast2_impl[Float64](af64, bf64, op_id, to_f64_f64, f64_to)





@always_inline
fn apply_broadcast2(a: Tensor[Float64], b: Tensor[Float64], op_id: Int) -> Tensor[Float64]:
    return apply_broadcast2_impl[Float64](a, b, op_id, to_f64_f64, f64_to)

@always_inline
fn apply_broadcast2(a: Tensor[Float32], b: Tensor[Float32], op_id: Int) -> Tensor[Float32]:
    return apply_broadcast2_impl[Float32](a, b, op_id, to_f64_f32, f64_to_float32)

@always_inline
fn apply_broadcast2(a: Tensor[Int8], b: Tensor[Int8], op_id: Int) -> Tensor[Int8]:
    return apply_broadcast2_impl[Int8](a, b, op_id, to_f64_i8, f64_to_int8)

@always_inline
fn apply_broadcast2(a: Tensor[Int16], b: Tensor[Int16], op_id: Int) -> Tensor[Int16]:
    return apply_broadcast2_impl[Int16](a, b, op_id, to_f64_i16, f64_to_int16)

@always_inline
fn apply_broadcast2(a: Tensor[Int32], b: Tensor[Int32], op_id: Int) -> Tensor[Int32]:
    return apply_broadcast2_impl[Int32](a, b, op_id, to_f64_i32, f64_to_int32)

@always_inline
fn apply_broadcast2(a: Tensor[Int64], b: Tensor[Int64], op_id: Int) -> Tensor[Int64]:
    return apply_broadcast2_impl[Int64](a, b, op_id, to_f64_i64, f64_to_int64)

@always_inline
fn apply_broadcast2(a: Tensor[Int], b: Tensor[Int], op_id: Int) -> Tensor[Int]:
    return apply_broadcast2_impl[Int](a, b, op_id, to_f64_int, f64_to_int)

@always_inline
fn apply_broadcast2(a: Tensor[UInt8], b: Tensor[UInt8], op_id: Int) -> Tensor[UInt8]:
    return apply_broadcast2_impl[UInt8](a, b, op_id, to_f64_u8, f64_to_uint8)

@always_inline
fn apply_broadcast2(a: Tensor[UInt16], b: Tensor[UInt16], op_id: Int) -> Tensor[UInt16]:
    return apply_broadcast2_impl[UInt16](a, b, op_id, to_f64_u16, f64_to_uint16)

@always_inline
fn apply_broadcast2(a: Tensor[UInt32], b: Tensor[UInt32], op_id: Int) -> Tensor[UInt32]:
    return apply_broadcast2_impl[UInt32](a, b, op_id, to_f64_u32, f64_to_uint32)

@always_inline
fn apply_broadcast2(a: Tensor[UInt64], b: Tensor[UInt64], op_id: Int) -> Tensor[UInt64]:
    return apply_broadcast2_impl[UInt64](a, b, op_id, to_f64_u64, f64_to_uint64)



@always_inline
fn apply_broadcast2_inplace(mut x: Tensor[Float64], y: Tensor[Float64], op_id: Int) -> None:
    apply_broadcast2_inplace_impl[Float64](x, y, op_id, to_f64_f64, f64_to)

@always_inline
fn apply_broadcast2_inplace(mut x: Tensor[Float32], y: Tensor[Float32], op_id: Int) -> None:
    apply_broadcast2_inplace_impl[Float32](x, y, op_id, to_f64_f32, f64_to_float32)

@always_inline
fn apply_broadcast2_inplace(mut x: Tensor[Int8], y: Tensor[Int8], op_id: Int) -> None:
    apply_broadcast2_inplace_impl[Int8](x, y, op_id, to_f64_i8, f64_to_int8)

@always_inline
fn apply_broadcast2_inplace(mut x: Tensor[Int16], y: Tensor[Int16], op_id: Int) -> None:
    apply_broadcast2_inplace_impl[Int16](x, y, op_id, to_f64_i16, f64_to_int16)

@always_inline
fn apply_broadcast2_inplace(mut x: Tensor[Int32], y: Tensor[Int32], op_id: Int) -> None:
    apply_broadcast2_inplace_impl[Int32](x, y, op_id, to_f64_i32, f64_to_int32)

@always_inline
fn apply_broadcast2_inplace(mut x: Tensor[Int64], y: Tensor[Int64], op_id: Int) -> None:
    apply_broadcast2_inplace_impl[Int64](x, y, op_id, to_f64_i64, f64_to_int64)

@always_inline
fn apply_broadcast2_inplace(mut x: Tensor[Int], y: Tensor[Int], op_id: Int) -> None:
    apply_broadcast2_inplace_impl[Int](x, y, op_id, to_f64_int, f64_to_int)

@always_inline
fn apply_broadcast2_inplace(mut x: Tensor[UInt8], y: Tensor[UInt8], op_id: Int) -> None:
    apply_broadcast2_inplace_impl[UInt8](x, y, op_id, to_f64_u8, f64_to_uint8)

@always_inline
fn apply_broadcast2_inplace(mut x: Tensor[UInt16], y: Tensor[UInt16], op_id: Int) -> None:
    apply_broadcast2_inplace_impl[UInt16](x, y, op_id, to_f64_u16, f64_to_uint16)

@always_inline
fn apply_broadcast2_inplace(mut x: Tensor[UInt32], y: Tensor[UInt32], op_id: Int) -> None:
    apply_broadcast2_inplace_impl[UInt32](x, y, op_id, to_f64_u32, f64_to_uint32)

@always_inline
fn apply_broadcast2_inplace(mut x: Tensor[UInt64], y: Tensor[UInt64], op_id: Int) -> None:
    apply_broadcast2_inplace_impl[UInt64](x, y, op_id, to_f64_u64, f64_to_uint64)



# ===== Int =====
@always_inline
fn add_t(a: Tensor[Int], b: Tensor[Int]) -> Tensor[Int]:
    return apply_broadcast2(a, b, 0)

@always_inline
fn sub_t(a: Tensor[Int], b: Tensor[Int]) -> Tensor[Int]:
    return apply_broadcast2(a, b, 1)

@always_inline
fn mul_t(a: Tensor[Int], b: Tensor[Int]) -> Tensor[Int]:
    return apply_broadcast2(a, b, 2)

@always_inline
fn div_t(a: Tensor[Int], b: Tensor[Int]) -> Tensor[Float64]:
    return apply_broadcast3(a, b, 3)

@always_inline
fn pow_t(a: Tensor[Int], b: Tensor[Int]) -> Tensor[Int]:
    return apply_broadcast2(a, b, 4)

@always_inline
fn maximum_t(a: Tensor[Int], b: Tensor[Int]) -> Tensor[Int]:
    return apply_broadcast2(a, b, 5)

@always_inline
fn minimum_t(a: Tensor[Int], b: Tensor[Int]) -> Tensor[Int]:
    return apply_broadcast2(a, b, 6)

@always_inline
fn mod_t(a: Tensor[Int], b: Tensor[Int]) -> Tensor[Float64]:
    return apply_broadcast3(a, b, 7)


@always_inline
fn and_t(x: Tensor[Int], y: Tensor[Int]) -> Tensor[Int]:
     return apply_broadcast4(x, y, 10)
@always_inline
fn or_t(x: Tensor[Int], y: Tensor[Int]) -> Tensor[Int]:
    return apply_broadcast4(x, y, 11)
@always_inline
fn xor_t(x: Tensor[Int], y: Tensor[Int]) -> Tensor[Int]:
     return apply_broadcast4(x, y, 12)
@always_inline
fn nand_t(x: Tensor[Int], y: Tensor[Int]) -> Tensor[Int]:
    return apply_broadcast4(x, y, 13)
@always_inline
fn nor_t(x: Tensor[Int], y: Tensor[Int]) -> Tensor[Int]:
    return apply_broadcast4(x, y, 14)
@always_inline
fn xnor_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]:
     return apply_broadcast4(x, y, 15)
@always_inline
fn andnot_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]:
     return apply_broadcast4(x, y, 16)



@always_inline
fn land_t(x: Tensor[Int], y: Tensor[Int]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 20)
@always_inline
fn lor_t(x: Tensor[Int], y: Tensor[Int]) -> Tensor[Bool]:
    return apply_broadcast5(x, y, 21)
@always_inline
fn lxor_t(x: Tensor[Int], y: Tensor[Int]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 22)
@always_inline
fn lnand_t(x: Tensor[Int], y: Tensor[Int]) -> Tensor[Bool]:
    return apply_broadcast5(x, y, 23)
@always_inline
fn lnor_t(x: Tensor[Int], y: Tensor[Int]) -> Tensor[Bool]:
    return apply_broadcast5(x, y, 24)
@always_inline
fn lxnor_t(x: Tensor[Int], y: Tensor[Int]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 25)
@always_inline
fn landnot_t(x: Tensor[Int], y: Tensor[Int]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 26)



# ===== Float64 =====
@always_inline
fn add_t(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    return apply_broadcast2(a, b, 0)

@always_inline
fn sub_t(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    return apply_broadcast2(a, b, 1)

@always_inline
fn mul_t(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    return apply_broadcast2(a, b, 2)

@always_inline
fn div_t(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    return apply_broadcast3(a, b, 3)

@always_inline
fn pow_t(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    return apply_broadcast2(a, b, 4)

@always_inline
fn maximum_t(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    return apply_broadcast2(a, b, 5)

@always_inline
fn minimum_t(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    return apply_broadcast2(a, b, 6)

@always_inline
fn mod_t(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    return apply_broadcast3(a, b, 7)



@always_inline
fn and_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]:
     return apply_broadcast4(x, y, 10)
@always_inline
fn or_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]:
    return apply_broadcast4(x, y, 11)
@always_inline
fn xor_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]:
     return apply_broadcast4(x, y, 12)
@always_inline
fn nand_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]:
    return apply_broadcast4(x, y, 13)
@always_inline
fn nor_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]:
    return apply_broadcast4(x, y, 14)
@always_inline
fn xnor_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]:
     return apply_broadcast4(x, y,1 5)
@always_inline
fn andnot_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]:
     return apply_broadcast4(x, y, 16)



@always_inline
fn land_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 20)
@always_inline
fn lor_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Bool]:
    return apply_broadcast5(x, y, 21)
@always_inline
fn lxor_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 22)
@always_inline
fn lnand_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Bool]:
    return apply_broadcast5(x, y, 23)
@always_inline
fn lnor_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Bool]:
    return apply_broadcast5(x, y, 24)
@always_inline
fn lxnor_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 25)
@always_inline
fn landnot_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 26)


# ===== Float32 =====
@always_inline
fn add_t(a: Tensor[Float32], b: Tensor[Float32]) -> Tensor[Float32]:
    return apply_broadcast2(a, b, 0)

@always_inline
fn sub_t(a: Tensor[Float32], b: Tensor[Float32]) -> Tensor[Float32]:
    return apply_broadcast2(a, b, 1)

@always_inline
fn mul_t(a: Tensor[Float32], b: Tensor[Float32]) -> Tensor[Float32]:
    return apply_broadcast2(a, b, 2)

@always_inline
fn div_t(a: Tensor[Float32], b: Tensor[Float32]) -> Tensor[Float32]:
    return apply_broadcast2(a, b, 3)

@always_inline
fn pow_t(a: Tensor[Float32], b: Tensor[Float32]) -> Tensor[Float32]:
    return apply_broadcast2(a, b, 4)

@always_inline
fn maximum_t(a: Tensor[Float32], b: Tensor[Float32]) -> Tensor[Float32]:
    return apply_broadcast2(a, b, 5)

@always_inline
fn minimum_t(a: Tensor[Float32], b: Tensor[Float32]) -> Tensor[Float32]:
    return apply_broadcast2(a, b, 6)

@always_inline
fn mod_t(a: Tensor[Float32], b: Tensor[Float32]) -> Tensor[Float32]:
    return apply_broadcast3(a, b, 7)
@always_inline
fn xnor_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Float32]:
     return _apply_logic(x, y, 5)



@always_inline
fn and_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]:
     return apply_broadcast4(x, y, 10)
@always_inline
fn or_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]:
    return apply_broadcast4(x, y, 11)
@always_inline
fn xor_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]:
     return apply_broadcast4(x, y, 12)
@always_inline
fn nand_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]:
    return apply_broadcast4(x, y, 13)
@always_inline
fn nor_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]:
    return apply_broadcast4(x, y, 14)
@always_inline
fn xnor_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Int]:
     return apply_broadcast4(x, y, 15)
@always_inline
fn andnot_t(x: Tensor[Float64], y: Tensor[Float64]) -> Tensor[Int]:
     return apply_broadcast4(x, y, 6)




@always_inline
fn land_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 20)
@always_inline
fn lor_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Bool]:
    return apply_broadcast5(x, y, 21)
@always_inline
fn lxor_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 22)
@always_inline
fn lnand_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Bool]:
    return apply_broadcast5(x, y, 23)
@always_inline
fn lnor_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Bool]:
    return apply_broadcast5(x, y, 24)
@always_inline
fn lxnor_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 25)
@always_inline
fn landnot_t(x: Tensor[Float32], y: Tensor[Float32]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 26)

@always_inline
fn land_t(x: Tensor[Bool], y: Tensor[Bool]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 20)
@always_inline
fn lor_t(x: Tensor[Bool], y: Tensor[Bool]) -> Tensor[Bool]:
    return apply_broadcast5(x, y, 21)
@always_inline
fn lxor_t(x: Tensor[Bool], y: Tensor[Bool]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 22)
@always_inline
fn lnand_t(x: Tensor[Bool], y: Tensor[Bool]) -> Tensor[Bool]:
    return apply_broadcast5(x, y, 23)
@always_inline
fn lnor_t(x: Tensor[Bool], y: Tensor[Bool]) -> Tensor[Bool]:
    return apply_broadcast5(x, y, 24)
@always_inline
fn lxnor_t(x: Tensor[Bool], y: Tensor[Bool]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 25)
@always_inline
fn landnot_t(x: Tensor[Bool], y: Tensor[Bool]) -> Tensor[Bool]:
     return apply_broadcast5(x, y, 26)



@always_inline
fn _not_to_int_impl[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], to_f64: fn (T) -> Float64
) -> Tensor[Int]:
    # Build Int mask: out[i] = 1 if x[i] == 0 else 0
    var n = len(x._data)

    var out = List[Int]()
    out.reserve(n)

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        var a0 = 1
        if to_f64(x._data[i    ]) != 0.0: a0 = 0
        var a1 = 1
        if to_f64(x._data[i + 1]) != 0.0: a1 = 0
        var a2 = 1
        if to_f64(x._data[i + 2]) != 0.0: a2 = 0
        var a3 = 1
        if to_f64(x._data[i + 3]) != 0.0: a3 = 0
        var a4 = 1
        if to_f64(x._data[i + 4]) != 0.0: a4 = 0
        var a5 = 1
        if to_f64(x._data[i + 5]) != 0.0: a5 = 0
        var a6 = 1
        if to_f64(x._data[i + 6]) != 0.0: a6 = 0
        var a7 = 1
        if to_f64(x._data[i + 7]) != 0.0: a7 = 0

        out.append(a0); out.append(a1); out.append(a2); out.append(a3)
        out.append(a4); out.append(a5); out.append(a6); out.append(a7)
        i += 8

    while i < n:
        var m = 1
        if to_f64(x._data[i]) != 0.0: m = 0
        out.append(m)
        i += 1

    # Fresh shape copy + row-major strides; offset = 0
    var shp = x._shape.copy()
    var strides = compute_row_major_strides(shp)
    return Tensor[Int](out, shp, strides, 0)



# Build a Bool mask: out[i] = True iff x[i] == 0
@always_inline
fn _not_to_bool_impl[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], to_f64: fn (T) -> Float64
) -> Tensor[Bool]:
    # Row-major dense build
    var n = len(x._data)
    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        var b0 = True
        if to_f64(x._data[i    ]) != 0.0:
            b0 = False
        var b1 = True
        if to_f64(x._data[i + 1]) != 0.0:
            b1 = False
        var b2 = True
        if to_f64(x._data[i + 2]) != 0.0:
            b2 = False
        var b3 = True
        if to_f64(x._data[i + 3]) != 0.0:
            b3 = False
        var b4 = True
        if to_f64(x._data[i + 4]) != 0.0:
            b4 = False
        var b5 = True
        if to_f64(x._data[i + 5]) != 0.0:
            b5 = False
        var b6 = True
        if to_f64(x._data[i + 6]) != 0.0:
            b6 = False
        var b7 = True
        if to_f64(x._data[i + 7]) != 0.0:
            b7 = False

        out.append(b0)
        out.append(b1)
        out.append(b2)
        out.append(b3)
        out.append(b4)
        out.append(b5)
        out.append(b6)
        out.append(b7)
        i += 8

    while i < n:
        var b = True
        if to_f64(x._data[i]) != 0.0:
            b = False
        out.append(b)
        i += 1

    # Use the 2-arg ctor (data, shape) → row-major strides, offset=0
    return Tensor[Bool](out, x._shape)

    # If your Tensor only has the 4-arg ctor, comment the line above and use:
    # var strides = compute_row_major_strides(x._shape)
    # return Tensor[Bool](out, x._shape, strides, 0)


# -----------------------------
# NOT → Int masks
# -----------------------------
@always_inline
fn not_t(x: Tensor[Float64]) -> Tensor[Int]:
    return _not_to_int_impl[Float64](x, to_float64_of)

@always_inline
fn not_t(x: Tensor[Float32]) -> Tensor[Int]:
    return _not_to_int_impl[Float32](x, to_float64_of)

@always_inline
fn not_t(x: Tensor[Int]) -> Tensor[Int]:
    return _not_to_int_impl[Int](x, to_float64_of)

# -----------------------------
# Logical NOT → Bool masks
# -----------------------------
@always_inline
fn lnot_t(x: Tensor[Float64]) -> Tensor[Bool]:
    return _not_to_bool_impl[Float64](x, to_float64_of)

@always_inline
fn lnot_t(x: Tensor[Float32]) -> Tensor[Bool]:
    return _not_to_bool_impl[Float32](x, to_float64_of)
@always_inline
fn lnot_t(x: Tensor[Int]) -> Tensor[Bool]:
    return _not_to_bool_impl[Int](x, to_float64_of)

@always_inline
fn lnot_t(x: Tensor[Bool]) -> Tensor[Bool]:
    # Fast path for Bool: just negate.
    var n = len(x._data)
    var out = List[Bool]()
    out.reserve(n)

    var i = 0
    while i < n:
        out.append(not x._data[i])
        i += 1

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](out, x._shape, strides, 0)







@always_inline
fn iadd_t[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], y: Tensor[T]) -> None:
    apply_broadcast2_inplace(x, y, 0)
@always_inline
fn isub_t[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], y: Tensor[T]) -> None:
    apply_broadcast2_inplace(x, y, 1)
@always_inline
fn imul_t[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], y: Tensor[T]) -> None:
    apply_broadcast2_inplace(x, y, 2)
@always_inline
fn idiv_t[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], y: Tensor[T]) -> None:
    apply_broadcast2_inplace(x, y, 3)
@always_inline
fn imaximum_t[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], y: Tensor[T]) -> None:
    apply_broadcast2_inplace(x, y, 5)
@always_inline
fn iminimum_t[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], y: Tensor[T]) -> None:
    apply_broadcast2_inplace(x, y, 6)

# ======================= Scalar pow/clip/lerp/normalize =======================

# ---------- Pow (scalar) ----------
fn pow_scalar_right[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], p: Float64) -> Tensor[T]:
    var n = len(x._data)
    var out = List[T]()
    out.reserve(n)

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(T(pow(Float64(x._data[i    ]), p)))
        out.append(T(pow(Float64(x._data[i + 1]), p)))
        out.append(T(pow(Float64(x._data[i + 2]), p)))
        out.append(T(pow(Float64(x._data[i + 3]), p)))
        out.append(T(pow(Float64(x._data[i + 4]), p)))
        out.append(T(pow(Float64(x._data[i + 5]), p)))
        out.append(T(pow(Float64(x._data[i + 6]), p)))
        out.append(T(pow(Float64(x._data[i + 7]), p)))
        i = i + 8
    while i < n:
        out.append(T(pow(Float64(x._data[i]), p)))
        i = i + 1

    return Tensor[T](x._shape, out)

fn pow_scalar_left[T: ImplicitlyCopyable & Copyable & Movable](p: Float64, x: Tensor[T]) -> Tensor[T]:
    var n = len(x._data)
    var out = List[T]()
    out.reserve(n)

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(T(pow(p, Float64(x._data[i    ]))))
        out.append(T(pow(p, Float64(x._data[i + 1]))))
        out.append(T(pow(p, Float64(x._data[i + 2]))))
        out.append(T(pow(p, Float64(x._data[i + 3]))))
        out.append(T(pow(p, Float64(x._data[i + 4]))))
        out.append(T(pow(p, Float64(x._data[i + 5]))))
        out.append(T(pow(p, Float64(x._data[i + 6]))))
        out.append(T(pow(p, Float64(x._data[i + 7]))))
        i = i + 8
    while i < n:
        out.append(T(pow(p, Float64(x._data[i]))))
        i = i + 1

    return Tensor[T](x._shape, out)

fn ipow_scalar[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], p: Float64) -> None:
    var n = len(x._data)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        x._data[i    ] = T(pow(Float64(x._data[i    ]), p))
        x._data[i + 1] = T(pow(Float64(x._data[i + 1]), p))
        x._data[i + 2] = T(pow(Float64(x._data[i + 2]), p))
        x._data[i + 3] = T(pow(Float64(x._data[i + 3]), p))
        x._data[i + 4] = T(pow(Float64(x._data[i + 4]), p))
        x._data[i + 5] = T(pow(Float64(x._data[i + 5]), p))
        x._data[i + 6] = T(pow(Float64(x._data[i + 6]), p))
        x._data[i + 7] = T(pow(Float64(x._data[i + 7]), p))
        i = i + 8
    while i < n:
        x._data[i] = T(pow(Float64(x._data[i]), p))
        i = i + 1

# ---------- Lerp (broadcast-aware) ----------
fn lerp[T: ImplicitlyCopyable & Copyable & Movable](a: Tensor[T], b: Tensor[T], weight: Float64) -> Tensor[T]:
    var ashp = a._shape.copy()
    var bshp = b._shape.copy()
    var an = len(a._data)
    var bn = len(b._data)

    if an == bn and same_shape(ashp, bshp) and is_contiguous(ashp, a._strides) and is_contiguous(bshp, b._strides):
        var n = an
        var out = List[T]()
        out.reserve(n)
        var w = weight
        var inv = 1.0 - w

        var i = 0
        var lim = (n // 8) * 8
        while i < lim:
            out.append(T(Float64(a._data[i    ]) * inv + Float64(b._data[i    ]) * w))
            out.append(T(Float64(a._data[i + 1]) * inv + Float64(b._data[i + 1]) * w))
            out.append(T(Float64(a._data[i + 2]) * inv + Float64(b._data[i + 2]) * w))
            out.append(T(Float64(a._data[i + 3]) * inv + Float64(b._data[i + 3]) * w))
            out.append(T(Float64(a._data[i + 4]) * inv + Float64(b._data[i + 4]) * w))
            out.append(T(Float64(a._data[i + 5]) * inv + Float64(b._data[i + 5]) * w))
            out.append(T(Float64(a._data[i + 6]) * inv + Float64(b._data[i + 6]) * w))
            out.append(T(Float64(a._data[i + 7]) * inv + Float64(b._data[i + 7]) * w))
            i = i + 8
        while i < n:
            out.append(T(Float64(a._data[i]) * inv + Float64(b._data[i]) * w))
            i = i + 1

        return Tensor[T](ashp, out)


    var oshp = broadcast_shapes(ashp, bshp)
    if len(oshp) == 0:
        var k = an
        if bn < k:
            k = bn
        var out0 = List[T]()
        out0.reserve(k)
        var i0 = 0
        var w0 = weight
        var inv0 = 1.0 - w0
        while i0 < k:
            out0.append(T(Float64(a._data[i0]) * inv0 + Float64(b._data[i0]) * w0))
            i0 = i0 + 1
        var shp1 = List[Int]()
        shp1.append(k)
        return Tensor[T](shp1, out0)


    var a_str = ensure_strides(a._strides, apad)
    var b_str = ensure_strides(b._strides, bpad)
    var o_str = row_major_strides(oshp)

    var rank_o = len(oshp)
    var rank_a = len(ashp)
    var rank_b = len(bshp)
    var n_out = numel(oshp)

    var w2 = weight
    var inv2 = 1.0 - w2
    var out2 = List[T]()
    out2.reserve(n_out)

    var idx = 0
    while idx < n_out:
        var rem = idx
        var oa = 0
        var ob = 0
        var d = 0
        while d < rank_o:
            var step = o_str[d]
            var cur = 0
            if step != 0:
                cur = (rem // step) % oshp[d]

            var ai = rank_a - rank_o + d
            var bi = rank_b - rank_o + d

            if ai >= 0 and ashp[ai] != 1:
                oa = oa + cur * a_str[ai]
            if bi >= 0 and bshp[bi] != 1:
                ob = ob + cur * b_str[bi]

            if step != 0:
                rem = rem % step
            d = d + 1


        out2.append(T(Float64(a._data[oa]) * inv2 + Float64(b._data[ob]) * w2))
        idx = idx + 1


    return Tensor[T](oshp, out2)
# ======================= Dot (1D) =======================
fn dot[T: ImplicitlyCopyable & Copyable & Movable](a: Tensor[T], b: Tensor[T]) -> T:
    var n = len(a._data)
    var m = len(b._data)

    var k = n
    if m < k:
        k = m

    var s = zero_scalar_of[T]()

    var i = 0
    var lim = (k // 8) * 8
    while i < lim:
        s = s + a._data[i    ] * b._data[i    ]
        s = s + a._data[i + 1] * b._data[i + 1]
        s = s + a._data[i + 2] * b._data[i + 2]
        s = s + a._data[i + 3] * b._data[i + 3]
        s = s + a._data[i + 4] * b._data[i + 4]
        s = s + a._data[i + 5] * b._data[i + 5]
        s = s + a._data[i + 6] * b._data[i + 6]
        s = s + a._data[i + 7] * b._data[i + 7]
        i = i + 8


    while i < k:
        s = s + a._data[i] * b._data[i]
        i = i + 1


    return s

# ---------- Normalize (L2) ----------

fn normalize[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, eps: Float64 = 1e-12) -> Tensor[T]:
    var shp = x._shape.copy()
    var lenr = len(shp)

    if axis is None:
        var n = len(x._data)
        var s2 = 0.0
        var i = 0
        var lim = (n // 8) * 8
        while i < lim:
            var v0 = Float64(x._data[i    ]); s2 = s2 + v0 * v0
            var v1 = Float64(x._data[i + 1]); s2 = s2 + v1 * v1
            var v2 = Float64(x._data[i + 2]); s2 = s2 + v2 * v2
            var v3 = Float64(x._data[i + 3]); s2 = s2 + v3 * v3
            var v4 = Float64(x._data[i + 4]); s2 = s2 + v4 * v4
            var v5 = Float64(x._data[i + 5]); s2 = s2 + v5 * v5
            var v6 = Float64(x._data[i + 6]); s2 = s2 + v6 * v6
            var v7 = Float64(x._data[i + 7]); s2 = s2 + v7 * v7
            i += 8
        while i < n:
            var v = Float64(x._data[i]); s2 = s2 + v * v
            i += 1
        var inv = 1.0 / (sqrt(s2) + eps)
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

    var ax = normalize_axis(axis.value(), lenr)
    var left = 1
    var right = 1
    var mid = shp[ax]

    var i0 = 0
    while i0 < ax:
        left = left * shp[i0]
        i0 += 1
    var i1 = ax + 1
    while i1 < lenr:
        right = right * shp[i1]
        i1 += 1

    var n_total = left * mid * right
    var out2 = List[T]()
    out2.reserve(n_total)

    var l = 0
    while l < left:
        var r = 0
        while r < right:
            var s2b = 0.0
            var m = 0
            while m < mid:
                var idx = (l * mid + m) * right + r
                var v = Float64(x._data[idx]); s2b = s2b + v * v
                m += 1
            var inv2 = 1.0 / (sqrt(s2b) + eps)
            m = 0
            while m < mid:
                var idx2 = (l * mid + m) * right + r
                out2.append(T(Float64(x._data[idx2]) * inv2))
                m += 1
            r += 1
        l += 1
    return Tensor[T](out2, shp)

fn normalize_[T: ImplicitlyCopyable & Copyable & Movable](mut x: Tensor[T], axis: Optional[Int] = None, eps: Float64 = 1e-12) -> None:
    var shp = x._shape.copy()
    var lenr = len(shp)

    if axis is None:
        var n = len(x._data)
        var s2 = 0.0
        var i = 0
        var lim = (n // 8) * 8
        while i < lim:
            var v0 = Float64(x._data[i    ]); s2 = s2 + v0 * v0
            var v1 = Float64(x._data[i + 1]); s2 = s2 + v1 * v1
            var v2 = Float64(x._data[i + 2]); s2 = s2 + v2 * v2
            var v3 = Float64(x._data[i + 3]); s2 = s2 + v3 * v3
            var v4 = Float64(x._data[i + 4]); s2 = s2 + v4 * v4
            var v5 = Float64(x._data[i + 5]); s2 = s2 + v5 * v5
            var v6 = Float64(x._data[i + 6]); s2 = s2 + v6 * v6
            var v7 = Float64(x._data[i + 7]); s2 = s2 + v7 * v7
            i += 8
        while i < n:
            var v = Float64(x._data[i]); s2 = s2 + v * v
            i += 1
        var inv = 1.0 / (sqrt(s2) + eps)
        var j = 0
        var lim2 = (n // 8) * 8
        while j < lim2:
            x._data[j    ] = T(Float64(x._data[j    ]) * inv)
            x._data[j + 1] = T(Float64(x._data[j + 1]) * inv)
            x._data[j + 2] = T(Float64(x._data[j + 2]) * inv)
            x._data[j + 3] = T(Float64(x._data[j + 3]) * inv)
            x._data[j + 4] = T(Float64(x._data[j + 4]) * inv)
            x._data[j + 5] = T(Float64(x._data[j + 5]) * inv)
            x._data[j + 6] = T(Float64(x._data[j + 6]) * inv)
            x._data[j + 7] = T(Float64(x._data[j + 7]) * inv)
            j += 8
        while j < n:
            x._data[j] = T(Float64(x._data[j]) * inv)
            j += 1
        return

    var ax = normalize_axis(axis.value(), lenr)
    var left = 1
    var right = 1
    var mid = shp[ax]

    var i0 = 0
    while i0 < ax:
        left = left * shp[i0]
        i0 += 1
    var i1 = ax + 1
    while i1 < lenr:
        right = right * shp[i1]
        i1 += 1

    var l = 0
    while l < left:
        var r = 0
        while r < right:
            var s2b = 0.0
            var m = 0
            while m < mid:
                var idx = (l * mid + m) * right + r
                var v = Float64(x._data[idx]); s2b = s2b + v * v
                m += 1
            var inv2 = 1.0 / (sqrt(s2b) + eps)
            m = 0
            while m < mid:
                var idx2 = (l * mid + m) * right + r
                x._data[idx2] = T(Float64(x._data[idx2]) * inv2)
                m += 1
            r += 1
        l += 1


# -----------------------------------------------------------------------------
# Tensor[Float64] reductions: global min / max
# - Empty tensor: returns 0.0
# - NaN policy: if any element is NaN, return that NaN immediately
# - Micro-opts: local alias for data; 8x unroll
# -----------------------------------------------------------------------------

@always_inline
fn min_t(x: Tensor[Float64]) -> Float64:
    var xs = x._data.copy()
    var n  = len(xs)
    if n == 0:
        return 0.0

    var m = xs[0]
    # m is NaN?
    if not (m == m):
        return m

    var i   = 1
    var lim = (n // 8) * 8
    while i < lim:
        var v0 = xs[i]
        if not (v0 == v0):
            return v0
        if v0 < m:
            m = v0

        var v1 = xs[i + 1]
        if not (v1 == v1):
            return v1
        if v1 < m:
            m = v1

        var v2 = xs[i + 2]
        if not (v2 == v2):
            return v2
        if v2 < m:
            m = v2

        var v3 = xs[i + 3]
        if not (v3 == v3):
            return v3
        if v3 < m:
            m = v3

        var v4 = xs[i + 4]
        if not (v4 == v4):
            return v4
        if v4 < m:
            m = v4

        var v5 = xs[i + 5]
        if not (v5 == v5):
            return v5
        if v5 < m:
            m = v5

        var v6 = xs[i + 6]
        if not (v6 == v6):
            return v6
        if v6 < m:
            m = v6

        var v7 = xs[i + 7]
        if not (v7 == v7):
            return v7
        if v7 < m:
            m = v7

        i = i + 8

    while i < n:
        var v = xs[i]
        if not (v == v):
            return v
        if v < m:
            m = v
        i = i + 1

    return m


@always_inline
fn max_t(x: Tensor[Float64]) -> Float64:
    var xs = x._data.copy()
    var n  = len(xs)
    if n == 0:
        return 0.0

    var M = xs[0]
    # M is NaN?
    if not (M == M):
        return M

    var i   = 1
    var lim = (n // 8) * 8
    while i < lim:
        var v0 = xs[i]
        if not (v0 == v0):
            return v0
        if v0 > M:
            M = v0

        var v1 = xs[i + 1]
        if not (v1 == v1):
            return v1
        if v1 > M:
            M = v1

        var v2 = xs[i + 2]
        if not (v2 == v2):
            return v2
        if v2 > M:
            M = v2

        var v3 = xs[i + 3]
        if not (v3 == v3):
            return v3
        if v3 > M:
            M = v3

        var v4 = xs[i + 4]
        if not (v4 == v4):
            return v4
        if v4 > M:
            M = v4

        var v5 = xs[i + 5]
        if not (v5 == v5):
            return v5
        if v5 > M:
            M = v5

        var v6 = xs[i + 6]
        if not (v6 == v6):
            return v6
        if v6 > M:
            M = v6

        var v7 = xs[i + 7]
        if not (v7 == v7):
            return v7
        if v7 > M:
            M = v7

        i = i + 8

    while i < n:
        var v = xs[i]
        if not (v == v):
            return v
        if v > M:
            M = v
        i = i + 1

    return M


@always_inline
fn min_t(x: Tensor[Float32]) -> Float32:
    var xs = x._data.copy()
    var n  = len(xs)
    if n == 0:
        return 0.0

    var m = xs[0]
    # m is NaN?
    if not (m == m):
        return m

    var i   = 1
    var lim = (n // 8) * 8
    while i < lim:
        var v0 = xs[i]
        if not (v0 == v0):
            return v0
        if v0 < m:
            m = v0

        var v1 = xs[i + 1]
        if not (v1 == v1):
            return v1
        if v1 < m:
            m = v1

        var v2 = xs[i + 2]
        if not (v2 == v2):
            return v2
        if v2 < m:
            m = v2

        var v3 = xs[i + 3]
        if not (v3 == v3):
            return v3
        if v3 < m:
            m = v3

        var v4 = xs[i + 4]
        if not (v4 == v4):
            return v4
        if v4 < m:
            m = v4

        var v5 = xs[i + 5]
        if not (v5 == v5):
            return v5
        if v5 < m:
            m = v5

        var v6 = xs[i + 6]
        if not (v6 == v6):
            return v6
        if v6 < m:
            m = v6

        var v7 = xs[i + 7]
        if not (v7 == v7):
            return v7
        if v7 < m:
            m = v7

        i = i + 8

    while i < n:
        var v = xs[i]
        if not (v == v):
            return v
        if v < m:
            m = v
        i = i + 1

    return m


@always_inline
fn max_t(x: Tensor[Float32]) -> Float32:
    var xs = x._data.copy()
    var n  = len(xs)
    if n == 0:
        return 0.0

    var M = xs[0]
    # M is NaN?
    if not (M == M):
        return M

    var i   = 1
    var lim = (n // 8) * 8
    while i < lim:
        var v0 = xs[i]
        if not (v0 == v0):
            return v0
        if v0 > M:
            M = v0

        var v1 = xs[i + 1]
        if not (v1 == v1):
            return v1
        if v1 > M:
            M = v1

        var v2 = xs[i + 2]
        if not (v2 == v2):
            return v2
        if v2 > M:
            M = v2

        var v3 = xs[i + 3]
        if not (v3 == v3):
            return v3
        if v3 > M:
            M = v3

        var v4 = xs[i + 4]
        if not (v4 == v4):
            return v4
        if v4 > M:
            M = v4

        var v5 = xs[i + 5]
        if not (v5 == v5):
            return v5
        if v5 > M:
            M = v5

        var v6 = xs[i + 6]
        if not (v6 == v6):
            return v6
        if v6 > M:
            M = v6

        var v7 = xs[i + 7]
        if not (v7 == v7):
            return v7
        if v7 > M:
            M = v7

        i = i + 8

    while i < n:
        var v = xs[i]
        if not (v == v):
            return v
        if v > M:
            M = v
        i = i + 1

    return M


@always_inline
fn min_t(x: Tensor[Int]) -> Int:
    var xs = x._data.copy()
    var n  = len(xs)
    if n == 0:
        return 0

    var m = xs[0]
    # m is NaN?
    if not (m == m):
        return m

    var i   = 1
    var lim = (n // 8) * 8
    while i < lim:
        var v0 = xs[i]
        if not (v0 == v0):
            return v0
        if v0 < m:
            m = v0

        var v1 = xs[i + 1]
        if not (v1 == v1):
            return v1
        if v1 < m:
            m = v1

        var v2 = xs[i + 2]
        if not (v2 == v2):
            return v2
        if v2 < m:
            m = v2

        var v3 = xs[i + 3]
        if not (v3 == v3):
            return v3
        if v3 < m:
            m = v3

        var v4 = xs[i + 4]
        if not (v4 == v4):
            return v4
        if v4 < m:
            m = v4

        var v5 = xs[i + 5]
        if not (v5 == v5):
            return v5
        if v5 < m:
            m = v5

        var v6 = xs[i + 6]
        if not (v6 == v6):
            return v6
        if v6 < m:
            m = v6

        var v7 = xs[i + 7]
        if not (v7 == v7):
            return v7
        if v7 < m:
            m = v7

        i = i + 8

    while i < n:
        var v = xs[i]
        if not (v == v):
            return v
        if v < m:
            m = v
        i = i + 1

    return m


@always_inline
fn max_t(x: Tensor[Int]) -> Int:
    var xs = x._data.copy()
    var n  = len(xs)
    if n == 0:
        return 0

    var M = xs[0]
    # M is NaN?
    if not (M == M):
        return M

    var i   = 1
    var lim = (n // 8) * 8
    while i < lim:
        var v0 = xs[i]
        if not (v0 == v0):
            return v0
        if v0 > M:
            M = v0

        var v1 = xs[i + 1]
        if not (v1 == v1):
            return v1
        if v1 > M:
            M = v1

        var v2 = xs[i + 2]
        if not (v2 == v2):
            return v2
        if v2 > M:
            M = v2

        var v3 = xs[i + 3]
        if not (v3 == v3):
            return v3
        if v3 > M:
            M = v3

        var v4 = xs[i + 4]
        if not (v4 == v4):
            return v4
        if v4 > M:
            M = v4

        var v5 = xs[i + 5]
        if not (v5 == v5):
            return v5
        if v5 > M:
            M = v5

        var v6 = xs[i + 6]
        if not (v6 == v6):
            return v6
        if v6 > M:
            M = v6

        var v7 = xs[i + 7]
        if not (v7 == v7):
            return v7
        if v7 > M:
            M = v7

        i = i + 8

    while i < n:
        var v = xs[i]
        if not (v == v):
            return v
        if v > M:
            M = v
        i = i + 1

    return M


# ======================= Comparisons -> UInt8 mask =======================

@always_inline
fn cmp_eval[T: ImplicitlyCopyable & Copyable & Movable](a: T, b: T, mode: Int) -> UInt8:
    if mode == 0: return UInt8(1) if a == b else UInt8(0)
    if mode == 1: return UInt8(1) if a != b else UInt8(0)
    if mode == 2: return UInt8(1) if a <  b else UInt8(0)
    if mode == 3: return UInt8(1) if a <= b else UInt8(0)
    if mode == 4: return UInt8(1) if a >  b else UInt8(0)
    return UInt8(1) if a >= b else UInt8(0)

fn compare[T: ImplicitlyCopyable & Copyable & Movable](a: Tensor[T], b: Tensor[T], mode: Int) -> Tensor[UInt8]:
    var ashp = a._shape.copy()
    var bshp = b._shape.copy()
    var oshp = broadcast_shapes(ashp, bshp)

    if len(oshp) == 0:
        var n = len(a._data)
        var m = len(b._data)
        var k = n if n < m else m
        var out = List[UInt8]()
        out.reserve(k)
        var i0 = 0
        while i0 < k:
            out.append(cmp_eval[T](a._data[i0], b._data[i0], mode))
            i0 += 1
        var shp1 = List[Int]()
        shp1.append(k)
        return Tensor[UInt8](out, shp1)

    var an = len(a._data)
    var bn = len(b._data)
    if an == bn and same_shape(ashp, bshp) and is_contiguous(ashp, a._strides) and is_contiguous(bshp, b._strides):
        var n = an
        var out2 = List[UInt8]()
        out2.reserve(n)
        var i = 0
        var lim = (n // 8) * 8
        while i < lim:
            out2.append(cmp_eval[T](a._data[i    ], b._data[i    ], mode))
            out2.append(cmp_eval[T](a._data[i + 1], b._data[i + 1], mode))
            out2.append(cmp_eval[T](a._data[i + 2], b._data[i + 2], mode))
            out2.append(cmp_eval[T](a._data[i + 3], b._data[i + 3], mode))
            out2.append(cmp_eval[T](a._data[i + 4], b._data[i + 4], mode))
            out2.append(cmp_eval[T](a._data[i + 5], b._data[i + 5], mode))
            out2.append(cmp_eval[T](a._data[i + 6], b._data[i + 6], mode))
            out2.append(cmp_eval[T](a._data[i + 7], b._data[i + 7], mode))
            i += 8
        while i < n:
            out2.append(cmp_eval[T](a._data[i], b._data[i], mode))
        return Tensor[UInt8](out2, ashp)

    var a_str = ensure_strides(a._strides, ashp)
    var b_str = ensure_strides(b._strides, bshp)
    var o_str = row_major_strides(oshp)
    var rank_o = len(oshp)
    var rank_a = len(ashp)
    var rank_b = len(bshp)
    var n_out = numel(oshp)
    var out = List[UInt8]()
    out.reserve(n_out)

    var idx = 0
    while idx < n_out:
        var rem = idx
        var oa = 0
        var ob = 0
        var d = 0
        while d < rank_o:
            var step = o_str[d]
            var cur = 0
            if step != 0:
                cur = (rem // step) % oshp[d]
            var ai = rank_a - rank_o + d
            var bi = rank_b - rank_o + d
            if ai >= 0 and ashp[ai] != 1:
                oa = oa + cur * a_str[ai]
            if bi >= 0 and bshp[bi] != 1:
                ob = ob + cur * b_str[bi]
            if step != 0:
                rem = rem % step
            d += 1
        out.append(cmp_eval[T](a._data[oa], b._data[ob], mode))
        idx += 1
    return Tensor[UInt8](out, oshp)




# ======================= Reductions =======================

# Unrolled 1D reducers (Float64 / Float32 / Int)
# Usage:
#   var s = sum1d_unrolled(tensor_f64)   # returns Float64
#   var s = sum1d_unrolled(tensor_f32)   # returns Float32
#   var s = sum1d_unrolled(tensor_int)   # returns Int

@always_inline
fn sum1d_unrolled(m: Tensor[Float64]) -> Float64:
    # Sum elements of a 1D Float64 tensor using 8-way unrolling.
    var n = len(m._data)

    var acc0 = 0.0
    var acc1 = 0.0
    var acc2 = 0.0
    var acc3 = 0.0
    var acc4 = 0.0
    var acc5 = 0.0
    var acc6 = 0.0
    var acc7 = 0.0

    var idx = 0
    var n_aligned = (n // 8) * 8

    while idx < n_aligned:
        acc0 = acc0 + m._data[idx + 0]
        acc1 = acc1 + m._data[idx + 1]
        acc2 = acc2 + m._data[idx + 2]
        acc3 = acc3 + m._data[idx + 3]
        acc4 = acc4 + m._data[idx + 4]
        acc5 = acc5 + m._data[idx + 5]
        acc6 = acc6 + m._data[idx + 6]
        acc7 = acc7 + m._data[idx + 7]
        idx += 8

    var total = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))

    # Remainder loop
    while idx < n:
        total = total + m._data[idx]
        idx += 1

    return total


@always_inline
fn sum1d_unrolled(m: Tensor[Float32]) -> Float32:
    # Sum elements of a 1D Float32 tensor using 8-way unrolling.
    var n = len(m._data)

    # Use Float32 zeros explicitly to avoid widening to Float64.
    var acc0 = Float32(0.0)
    var acc1 = Float32(0.0)
    var acc2 = Float32(0.0)
    var acc3 = Float32(0.0)
    var acc4 = Float32(0.0)
    var acc5 = Float32(0.0)
    var acc6 = Float32(0.0)
    var acc7 = Float32(0.0)

    var idx = 0
    var n_aligned = (n // 8) * 8

    while idx < n_aligned:
        acc0 = acc0 + m._data[idx + 0]
        acc1 = acc1 + m._data[idx + 1]
        acc2 = acc2 + m._data[idx + 2]
        acc3 = acc3 + m._data[idx + 3]
        acc4 = acc4 + m._data[idx + 4]
        acc5 = acc5 + m._data[idx + 5]
        acc6 = acc6 + m._data[idx + 6]
        acc7 = acc7 + m._data[idx + 7]
        idx += 8

    var total = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))

    # Remainder loop
    while idx < n:
        total = total + m._data[idx]
        idx += 1

    return total


@always_inline
fn sum1d_unrolled(m: Tensor[Int]) -> Int:
    # Sum elements of a 1D Int tensor using 8-way unrolling.
    var n = len(m._data)

    var acc0 = 0
    var acc1 = 0
    var acc2 = 0
    var acc3 = 0
    var acc4 = 0
    var acc5 = 0
    var acc6 = 0
    var acc7 = 0

    var idx = 0
    var n_aligned = (n // 8) * 8

    while idx < n_aligned:
        acc0 = acc0 + m._data[idx + 0]
        acc1 = acc1 + m._data[idx + 1]
        acc2 = acc2 + m._data[idx + 2]
        acc3 = acc3 + m._data[idx + 3]
        acc4 = acc4 + m._data[idx + 4]
        acc5 = acc5 + m._data[idx + 5]
        acc6 = acc6 + m._data[idx + 6]
        acc7 = acc7 + m._data[idx + 7]
        idx += 8

    var total = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))

    # Remainder loop
    while idx < n:
        total = total + m._data[idx]
        idx += 1

    return total


# ---------- SUM for Float64 ----------
fn sum(x: Tensor[Float64], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
    var shp = x._shape.copy()
    var rank = len(shp)

    # WHOLE-TENSOR SUM
    if axis is None:
        var s = 0.0
        if (len(shp) == 1 or is_row_major_contiguous(shp, x._strides)):
            var n = len(x._data)
            var i = 0
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
        else:
            var idx = List[Int]()
            var k = 0
            while k < rank:
                idx.append(0)
                k = k + 1
            var done = False
            while not done:
                var off = 0
                var d = 0
                while d < rank:
                    off = off + idx[d] * x._strides[d]
                    d = d + 1
                s = s + x._data[off]

                var r = rank - 1
                while r >= 0:
                    idx[r] = idx[r] + 1
                    if idx[r] < shp[r]:
                        break
                    idx[r] = 0
                    if r == 0:
                        done = True
                        break
                    r = r - 1

        var out_list = List[Float64]()
        out_list.reserve(1)
        out_list.append(s)

        var out_shape = List[Int]()
        if keepdims:
            var t = 0
            while t < rank:
                out_shape.append(1)
                t = t + 1
        else:
            out_shape.append(1)

        return Tensor[Float64](out_shape, out_list)

    # AXIS REDUCTION
    var ax = normalize_axis(axis.value(), rank)
    var reduce_n: Int
    if rank == 0:
        reduce_n = 1
    else:
        reduce_n = shp[ax]

    if is_row_major_contiguous(shp, x._strides):
        var out_shape2 = shape_drop_axis(shp, ax)

        var inner = 1
        var i_in = ax + 1
        while i_in < rank:
            inner = inner * shp[i_in]
            i_in = i_in + 1

        var outer = 1
        var i_out = 0
        while i_out < ax:
            outer = outer * shp[i_out]
            i_out = i_out + 1

        var base_stride = inner
        var block = reduce_n * inner

        var outv = List[Float64]()
        outv.reserve(outer * inner)

        var o = 0
        while o < outer * inner:
            var base = (o // inner) * block + (o % inner)
            var s2 = 0.0

            var k2 = 0
            var lim2 = (reduce_n // 8) * 8
            while k2 < lim2:
                s2 = s2 + x._data[base + (k2    ) * base_stride]
                s2 = s2 + x._data[base + (k2 + 1) * base_stride]
                s2 = s2 + x._data[base + (k2 + 2) * base_stride]
                s2 = s2 + x._data[base + (k2 + 3) * base_stride]
                s2 = s2 + x._data[base + (k2 + 4) * base_stride]
                s2 = s2 + x._data[base + (k2 + 5) * base_stride]
                s2 = s2 + x._data[base + (k2 + 6) * base_stride]
                s2 = s2 + x._data[base + (k2 + 7) * base_stride]
                k2 = k2 + 8
            while k2 < reduce_n:
                s2 = s2 + x._data[base + k2 * base_stride]
                k2 = k2 + 1

            outv.append(s2)
            o = o + 1

        var tout = Tensor[Float64](out_shape2, outv)
        if keepdims:
            var kd = List[Int]()
            var r = 0
            while r < rank:
                if r == ax:
                    kd.append(1)
                else:
                    kd.append(shp[r])
                r = r + 1
            return tout.reshape(kd)
        return tout.copy()

    # Generic strided axis reduction
    var out_shape_g = shape_drop_axis(shp, ax)
    var out_n = numel(out_shape_g)

    var outer_shape = List[Int]()
    var outer_strides = List[Int]()
    var i = 0
    while i < rank:
        if i != ax:
            outer_shape.append(shp[i])
            outer_strides.append(x._strides[i])
        i = i + 1
    var stride_red = x._strides[ax]

    var out_f64 = List[Float64]()
    out_f64.reserve(out_n)

    var idx2 = List[Int]()
    var k3 = 0
    while k3 < len(outer_shape):
        idx2.append(0)
        k3 = k3 + 1

    var running = True
    while True:
        var base2 = 0
        var d2 = 0
        while d2 < len(outer_shape):
            base2 = base2 + idx2[d2] * outer_strides[d2]
            d2 = d2 + 1

        var acc = 0.0
        var j = 0
        var lim = (reduce_n // 8) * 8
        while j < lim:
            acc = acc + x._data[base2 + (j    ) * stride_red]
            acc = acc + x._data[base2 + (j + 1) * stride_red]
            acc = acc + x._data[base2 + (j + 2) * stride_red]
            acc = acc + x._data[base2 + (j + 3) * stride_red]
            acc = acc + x._data[base2 + (j + 4) * stride_red]
            acc = acc + x._data[base2 + (j + 5) * stride_red]
            acc = acc + x._data[base2 + (j + 6) * stride_red]
            acc = acc + x._data[base2 + (j + 7) * stride_red]
            j = j + 8
        while j < reduce_n:
            acc = acc + x._data[base2 + j * stride_red]
            j = j + 1

        out_f64.append(acc)

        if len(outer_shape) == 0:
            break
        var pos = len(outer_shape) - 1
        var carry = True
        while pos >= 0 and carry:
            idx2[pos] = idx2[pos] + 1
            if idx2[pos] < outer_shape[pos]:
                carry = False
            else:
                idx2[pos] = 0
                if pos == 0:
                    carry = False
                    running = False
            if pos == 0:
                break
            pos = pos - 1
        if not running:
            break

    var tout_g = Tensor[Float64](out_shape_g, out_f64)
    if keepdims:
        var kd2 = List[Int]()
        var r2 = 0
        while r2 < rank:
            if r2 == ax:
                kd2.append(1)
            else:
                kd2.append(shp[r2])
            r2 = r2 + 1
        return tout_g.reshape(kd2)
    return tout_g.copy()


# ---------- SUM for Int ----------
@always_inline
fn sum(x: Tensor[Int], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Int]:
    var shp = x._shape.copy()
    var rank = len(shp)

    # WHOLE-TENSOR SUM
    if axis is None:
        var s = 0
        if (len(shp) == 1 or is_row_major_contiguous(shp, x._strides)):
            var n = len(x._data)
            var i = 0
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
        else:
            var idx = List[Int]()
            var k = 0
            while k < rank:
                idx.append(0)
                k = k + 1
            var done = False
            while not done:
                var off = 0
                var d = 0
                while d < rank:
                    off = off + idx[d] * x._strides[d]
                    d = d + 1
                s = s + x._data[off]

                var r = rank - 1
                while r >= 0:
                    idx[r] = idx[r] + 1
                    if idx[r] < shp[r]:
                        break
                    idx[r] = 0
                    if r == 0:
                        done = True
                        break
                    r = r - 1

        var out_list = List[Int]()
        out_list.reserve(1)
        out_list.append(s)

        var out_shape = List[Int]()
        if keepdims:
            var t = 0
            while t < rank:
                out_shape.append(1)
                t = t + 1
        else:
            out_shape.append(1)

        var st = compute_row_major_strides(out_shape)
        return Tensor[Int](out_list, out_shape, st, 0)

    # AXIS REDUCTION
    var ax = normalize_axis(axis.value(), rank)
    var reduce_n: Int
    if rank == 0:
        reduce_n = 1
    else:
        reduce_n = shp[ax]

    if is_row_major_contiguous(shp, x._strides):
        var out_shape2 = shape_drop_axis(shp, ax)

        var inner = 1
        var i_in = ax + 1
        while i_in < rank:
            inner = inner * shp[i_in]
            i_in = i_in + 1

        var outer = 1
        var i_out = 0
        while i_out < ax:
            outer = outer * shp[i_out]
            i_out = i_out + 1

        var base_stride = inner
        var block = reduce_n * inner

        var outv = List[Int]()
        outv.reserve(outer * inner)

        var o = 0
        while o < outer * inner:
            var base = (o // inner) * block + (o % inner)
            var s2 = 0

            var k2 = 0
            var lim2 = (reduce_n // 8) * 8
            while k2 < lim2:
                s2 = s2 + x._data[base + (k2    ) * base_stride]
                s2 = s2 + x._data[base + (k2 + 1) * base_stride]
                s2 = s2 + x._data[base + (k2 + 2) * base_stride]
                s2 = s2 + x._data[base + (k2 + 3) * base_stride]
                s2 = s2 + x._data[base + (k2 + 4) * base_stride]
                s2 = s2 + x._data[base + (k2 + 5) * base_stride]
                s2 = s2 + x._data[base + (k2 + 6) * base_stride]
                s2 = s2 + x._data[base + (k2 + 7) * base_stride]
                k2 = k2 + 8
            while k2 < reduce_n:
                s2 = s2 + x._data[base + k2 * base_stride]
                k2 = k2 + 1

            outv.append(s2)
            o = o + 1

        var st2 = compute_row_major_strides(out_shape2)
        var tout = Tensor[Int](outv, out_shape2, st2, 0)
        if keepdims:
            var kd = List[Int]()
            var r = 0
            while r < rank:
                if r == ax:
                    kd.append(1)
                else:
                    kd.append(shp[r])
                r = r + 1
            return tout.reshape(kd)
        return tout.copy()

    # Generic strided axis reduction
    var out_shape_g = shape_drop_axis(shp, ax)
    var out_n = numel(out_shape_g)

    var outer_shape = List[Int]()
    var outer_strides = List[Int]()
    var i = 0
    while i < rank:
        if i != ax:
            outer_shape.append(shp[i])
            outer_strides.append(x._strides[i])
        i = i + 1
    var stride_red = x._strides[ax]

    var out_int = List[Int]()
    out_int.reserve(out_n)

    var idx2 = List[Int]()
    var k3 = 0
    while k3 < len(outer_shape):
        idx2.append(0)
        k3 = k3 + 1

    var running = True
    while True:
        var base2 = 0
        var d2 = 0
        while d2 < len(outer_shape):
            base2 = base2 + idx2[d2] * outer_strides[d2]
            d2 = d2 + 1

        var acc = 0
        var j = 0
        var lim = (reduce_n // 8) * 8
        while j < lim:
            acc = acc + x._data[base2 + (j    ) * stride_red]
            acc = acc + x._data[base2 + (j + 1) * stride_red]
            acc = acc + x._data[base2 + (j + 2) * stride_red]
            acc = acc + x._data[base2 + (j + 3) * stride_red]
            acc = acc + x._data[base2 + (j + 4) * stride_red]
            acc = acc + x._data[base2 + (j + 5) * stride_red]
            acc = acc + x._data[base2 + (j + 6) * stride_red]
            acc = acc + x._data[base2 + (j + 7) * stride_red]
            j = j + 8
        while j < reduce_n:
            acc = acc + x._data[base2 + j * stride_red]
            j = j + 1

        out_int.append(acc)

        if len(outer_shape) == 0:
            break
        var pos = len(outer_shape) - 1
        var carry = True
        while pos >= 0 and carry:
            idx2[pos] = idx2[pos] + 1
            if idx2[pos] < outer_shape[pos]:
                carry = False
            else:
                idx2[pos] = 0
                if pos == 0:
                    carry = False
                    running = False
            if pos == 0:
                break
            pos = pos - 1
        if not running:
            break

    var stg = compute_row_major_strides(out_shape_g)
    var tout_g = Tensor[Int](out_int, out_shape_g, stg, 0)
    if keepdims:
        var kd2 = List[Int]()
        var r2 = 0
        while r2 < rank:
            if r2 == ax:
                kd2.append(1)
            else:
                kd2.append(shp[r2])
            r2 = r2 + 1
        return tout_g.reshape(kd2)
    return tout_g.copy()

# ---------- SUM for Float32 ----------
@always_inline
fn sum(x: Tensor[Float32], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float32]:
    var shp = x._shape.copy()
    var rank = len(shp)

    # WHOLE-TENSOR SUM
    if axis is None:
        var s: Float32 = 0.0
        if (len(shp) == 1 or is_row_major_contiguous(shp, x._strides)):
            var n = len(x._data)
            var i = 0
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
        else:
            var idx = List[Int]()
            var k = 0
            while k < rank:
                idx.append(0)
                k = k + 1
            var done = False
            while not done:
                var off = 0
                var d = 0
                while d < rank:
                    off = off + idx[d] * x._strides[d]
                    d = d + 1
                s = s + x._data[off]

                var r = rank - 1
                while r >= 0:
                    idx[r] = idx[r] + 1
                    if idx[r] < shp[r]:
                        break
                    idx[r] = 0
                    if r == 0:
                        done = True
                        break
                    r = r - 1

        var out_list = List[Float32]()
        out_list.reserve(1)
        out_list.append(s)

        var out_shape = List[Int]()
        if keepdims:
            var t = 0
            while t < rank:
                out_shape.append(1)
                t = t + 1
        else:
            out_shape.append(1)

        return Tensor[Float32](out_shape, out_list)

    # AXIS REDUCTION
    var ax = normalize_axis(axis.value(), rank)
    var reduce_n: Int
    if rank == 0:
        reduce_n = 1
    else:
        reduce_n = shp[ax]

    if is_row_major_contiguous(shp, x._strides):
        var out_shape2 = shape_drop_axis(shp, ax)

        var inner = 1
        var i_in = ax + 1
        while i_in < rank:
            inner = inner * shp[i_in]
            i_in = i_in + 1

        var outer = 1
        var i_out = 0
        while i_out < ax:
            outer = outer * shp[i_out]
            i_out = i_out + 1

        var base_stride = inner
        var block = reduce_n * inner

        var outv = List[Float32]()
        outv.reserve(outer * inner)

        var o = 0
        while o < outer * inner:
            var base = (o // inner) * block + (o % inner)
            var s2: Float32 = 0.0

            var k2 = 0
            var lim2 = (reduce_n // 8) * 8
            while k2 < lim2:
                s2 = s2 + x._data[base + (k2    ) * base_stride]
                s2 = s2 + x._data[base + (k2 + 1) * base_stride]
                s2 = s2 + x._data[base + (k2 + 2) * base_stride]
                s2 = s2 + x._data[base + (k2 + 3) * base_stride]
                s2 = s2 + x._data[base + (k2 + 4) * base_stride]
                s2 = s2 + x._data[base + (k2 + 5) * base_stride]
                s2 = s2 + x._data[base + (k2 + 6) * base_stride]
                s2 = s2 + x._data[base + (k2 + 7) * base_stride]
                k2 = k2 + 8
            while k2 < reduce_n:
                s2 = s2 + x._data[base + k2 * base_stride]
                k2 = k2 + 1

            outv.append(s2)
            o = o + 1

        var tout = Tensor[Float32](out_shape2, outv)
        if keepdims:
            var kd = List[Int]()
            var r = 0
            while r < rank:
                if r == ax:
                    kd.append(1)
                else:
                    kd.append(shp[r])
                r = r + 1
            return tout.reshape(kd)
        return tout.copy()

    # Generic strided axis reduction
    var out_shape_g = shape_drop_axis(shp, ax)
    var out_n = numel(out_shape_g)

    var outer_shape = List[Int]()
    var outer_strides = List[Int]()
    var i = 0
    while i < rank:
        if i != ax:
            outer_shape.append(shp[i])
            outer_strides.append(x._strides[i])
        i = i + 1
    var stride_red = x._strides[ax]

    var out_f32 = List[Float32]()
    out_f32.reserve(out_n)

    var idx2 = List[Int]()
    var k3 = 0
    while k3 < len(outer_shape):
        idx2.append(0)
        k3 = k3 + 1

    var running = True
    while True:
        var base2 = 0
        var d2 = 0
        while d2 < len(outer_shape):
            base2 = base2 + idx2[d2] * outer_strides[d2]
            d2 = d2 + 1

        var acc: Float32 = 0.0
        var j = 0
        var lim = (reduce_n // 8) * 8
        while j < lim:
            acc = acc + x._data[base2 + (j    ) * stride_red]
            acc = acc + x._data[base2 + (j + 1) * stride_red]
            acc = acc + x._data[base2 + (j + 2) * stride_red]
            acc = acc + x._data[base2 + (j + 3) * stride_red]
            acc = acc + x._data[base2 + (j + 4) * stride_red]
            acc = acc + x._data[base2 + (j + 5) * stride_red]
            acc = acc + x._data[base2 + (j + 6) * stride_red]
            acc = acc + x._data[base2 + (j + 7) * stride_red]
            j = j + 8
        while j < reduce_n:
            acc = acc + x._data[base2 + j * stride_red]
            j = j + 1

        out_f32.append(acc)

        if len(outer_shape) == 0:
            break
        var pos = len(outer_shape) - 1
        var carry = True
        while pos >= 0 and carry:
            idx2[pos] = idx2[pos] + 1
            if idx2[pos] < outer_shape[pos]:
                carry = False
            else:
                idx2[pos] = 0
                if pos == 0:
                    carry = False
                    running = False
            if pos == 0:
                break
            pos = pos - 1
        if not running:
            break

    var tout_g = Tensor[Float32](out_shape_g, out_f32)
    if keepdims:
        var kd2 = List[Int]()
        var r2 = 0
        while r2 < rank:
            if r2 == ax:
                kd2.append(1)
            else:
                kd2.append(shp[r2])
            r2 = r2 + 1
        return tout_g.reshape(kd2)
    return tout_g.copy()


# ======================= 1D Unrolled STD =======================

@always_inline
fn std1d_unrolled(m: Tensor[Float64], ddof: Int = 0) -> Float64:
    # Computes population/sample std for 1D Float64 via two-pass (sum & sumsq), 8-way unrolled.
    var n = len(m._data)
    #assert(n > ddof and "std: degrees of freedom must be < N")

    # ---- Pass 1: sum ----
    var s0 = 0.0; var s1 = 0.0; var s2 = 0.0; var s3 = 0.0
    var s4 = 0.0; var s5 = 0.0; var s6 = 0.0; var s7 = 0.0
    var i = 0
    var n8 = (n // 8) * 8
    while i < n8:
        s0 = s0 + m._data[i + 0]; s1 = s1 + m._data[i + 1]
        s2 = s2 + m._data[i + 2]; s3 = s3 + m._data[i + 3]
        s4 = s4 + m._data[i + 4]; s5 = s5 + m._data[i + 5]
        s6 = s6 + m._data[i + 6]; s7 = s7 + m._data[i + 7]
        i += 8
    var sumv = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
    while i < n:
        sumv = sumv + m._data[i]
        i += 1
    var mean = sumv / Float64(n)

    # ---- Pass 2: sum of squared diffs ----
    var v0 = 0.0; var v1 = 0.0; var v2 = 0.0; var v3 = 0.0
    var v4 = 0.0; var v5 = 0.0; var v6 = 0.0; var v7 = 0.0
    i = 0
    while i < n8:
        var a0 = m._data[i + 0] - mean; v0 = v0 + a0 * a0
        var a1 = m._data[i + 1] - mean; v1 = v1 + a1 * a1
        var a2 = m._data[i + 2] - mean; v2 = v2 + a2 * a2
        var a3 = m._data[i + 3] - mean; v3 = v3 + a3 * a3
        var a4 = m._data[i + 4] - mean; v4 = v4 + a4 * a4
        var a5 = m._data[i + 5] - mean; v5 = v5 + a5 * a5
        var a6 = m._data[i + 6] - mean; v6 = v6 + a6 * a6
        var a7 = m._data[i + 7] - mean; v7 = v7 + a7 * a7
        i += 8
    var ssd = ((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7))
    while i < n:
        var d = m._data[i] - mean
        ssd = ssd + d * d
        i += 1

    var denom = Float64(n - ddof)
    var varv = ssd / denom
    return sqrt(varv)


@always_inline
fn std1d_unrolled(m: Tensor[Float32], ddof: Int = 0) -> Float32:
    # 1D Float32 two-pass with 8-way unroll; returns Float32.
    var n = len(m._data)
    #assert(n > ddof and "std: degrees of freedom must be < N")

    var z = Float32(0.0)
    var s0 = z; var s1 = z; var s2 = z; var s3 = z
    var s4 = z; var s5 = z; var s6 = z; var s7 = z
    var i = 0
    var n8 = (n // 8) * 8
    while i < n8:
        s0 = s0 + m._data[i + 0]; s1 = s1 + m._data[i + 1]
        s2 = s2 + m._data[i + 2]; s3 = s3 + m._data[i + 3]
        s4 = s4 + m._data[i + 4]; s5 = s5 + m._data[i + 5]
        s6 = s6 + m._data[i + 6]; s7 = s7 + m._data[i + 7]
        i += 8
    var sumv = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
    while i < n:
        sumv = sumv + m._data[i]
        i += 1
    var mean = sumv / Float32(n)

    var v0 = z; var v1 = z; var v2 = z; var v3 = z
    var v4 = z; var v5 = z; var v6 = z; var v7 = z
    i = 0
    while i < n8:
        var a0 = m._data[i + 0] - mean; v0 = v0 + a0 * a0
        var a1 = m._data[i + 1] - mean; v1 = v1 + a1 * a1
        var a2 = m._data[i + 2] - mean; v2 = v2 + a2 * a2
        var a3 = m._data[i + 3] - mean; v3 = v3 + a3 * a3
        var a4 = m._data[i + 4] - mean; v4 = v4 + a4 * a4
        var a5 = m._data[i + 5] - mean; v5 = v5 + a5 * a5
        var a6 = m._data[i + 6] - mean; v6 = v6 + a6 * a6
        var a7 = m._data[i + 7] - mean; v7 = v7 + a7 * a7
        i += 8
    var ssd = ((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7))
    while i < n:
        var d = m._data[i] - mean
        ssd = ssd + d * d
        i += 1

    var denom = Float32(n - ddof)
    var varv = ssd / denom
    return Float32(sqrt(Float64(varv)))


@always_inline
fn std1d_unrolled(m: Tensor[Int], ddof: Int = 0) -> Float64:
    # 1D Int → Float64 result (higher precision). Two-pass, 8-way unrolled.
    var n = len(m._data)
    #assert(n > ddof and "std: degrees of freedom must be < N")

    var s0 = 0; var s1 = 0; var s2 = 0; var s3 = 0
    var s4 = 0; var s5 = 0; var s6 = 0; var s7 = 0
    var i = 0
    var n8 = (n // 8) * 8
    while i < n8:
        s0 = s0 + m._data[i + 0]; s1 = s1 + m._data[i + 1]
        s2 = s2 + m._data[i + 2]; s3 = s3 + m._data[i + 3]
        s4 = s4 + m._data[i + 4]; s5 = s5 + m._data[i + 5]
        s6 = s6 + m._data[i + 6]; s7 = s7 + m._data[i + 7]
        i += 8
    var sumv_i = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))
    while i < n:
        sumv_i = sumv_i + m._data[i]
        i += 1
    var mean = Float64(sumv_i) / Float64(n)

    var v0 = 0.0; var v1 = 0.0; var v2 = 0.0; var v3 = 0.0
    var v4 = 0.0; var v5 = 0.0; var v6 = 0.0; var v7 = 0.0
    i = 0
    while i < n8:
        var a0 = Float64(m._data[i + 0]) - mean; v0 = v0 + a0 * a0
        var a1 = Float64(m._data[i + 1]) - mean; v1 = v1 + a1 * a1
        var a2 = Float64(m._data[i + 2]) - mean; v2 = v2 + a2 * a2
        var a3 = Float64(m._data[i + 3]) - mean; v3 = v3 + a3 * a3
        var a4 = Float64(m._data[i + 4]) - mean; v4 = v4 + a4 * a4
        var a5 = Float64(m._data[i + 5]) - mean; v5 = v5 + a5 * a5
        var a6 = Float64(m._data[i + 6]) - mean; v6 = v6 + a6 * a6
        var a7 = Float64(m._data[i + 7]) - mean; v7 = v7 + a7 * a7
        i += 8
    var ssd = ((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7))
    while i < n:
        var d = Float64(m._data[i]) - mean
        ssd = ssd + d * d
        i += 1

    var denom = Float64(n - ddof)
    var varv = ssd / denom
    return sqrt(varv)

# ======================= Axis-wise STD for Float64 =======================

fn std(x: Tensor[Float64], axis: Optional[Int] = None, keepdims: Bool = False, ddof: Int = 0) -> Tensor[Float64]:
    # Mirrors your sum(x, axis, keepdims) but computes std with two-pass mean/variance.
    var shp = x._shape.copy()
    var rank = len(shp)
    #assert(ddof >= 0 and "std: ddof must be >= 0")

    # ---------- WHOLE-TENSOR ----------
    if axis is None:
        var n_total = len(x._data)
        #assert(n_total > ddof and "std: ddof >= N")

        if (len(shp) == 1 or is_row_major_contiguous(shp, x._strides)):
            # Pass 1: mean (16-way unrolled like your sum)
            var s = 0.0
            var i = 0
            var lim = (n_total // 16) * 16
            while i < lim:
                s = s + x._data[i     ] + x._data[i + 1 ] + x._data[i + 2 ] + x._data[i + 3 ]
                s = s + x._data[i + 4 ] + x._data[i + 5 ] + x._data[i + 6 ] + x._data[i + 7 ]
                s = s + x._data[i + 8 ] + x._data[i + 9 ] + x._data[i + 10] + x._data[i + 11]
                s = s + x._data[i + 12] + x._data[i + 13] + x._data[i + 14] + x._data[i + 15]
                i = i + 16
            while i < n_total:
                s = s + x._data[i]
                i = i + 1
            var mean = s / Float64(n_total)

            # Pass 2: sum of squared diffs (same unroll)
            var q = 0.0
            i = 0
            while i < lim:
                var d0 = x._data[i     ] - mean; q = q + d0 * d0
                var d1 = x._data[i + 1 ] - mean; q = q + d1 * d1
                var d2 = x._data[i + 2 ] - mean; q = q + d2 * d2
                var d3 = x._data[i + 3 ] - mean; q = q + d3 * d3
                var d4 = x._data[i + 4 ] - mean; q = q + d4 * d4
                var d5 = x._data[i + 5 ] - mean; q = q + d5 * d5
                var d6 = x._data[i + 6 ] - mean; q = q + d6 * d6
                var d7 = x._data[i + 7 ] - mean; q = q + d7 * d7
                var d8 = x._data[i + 8 ] - mean; q = q + d8 * d8
                var d9 = x._data[i + 9 ] - mean; q = q + d9 * d9
                var dA = x._data[i + 10] - mean; q = q + dA * dA
                var dB = x._data[i + 11] - mean; q = q + dB * dB
                var dC = x._data[i + 12] - mean; q = q + dC * dC
                var dD = x._data[i + 13] - mean; q = q + dD * dD
                var dE = x._data[i + 14] - mean; q = q + dE * dE
                var dF = x._data[i + 15] - mean; q = q + dF * dF
                i = i + 16
            while i < n_total:
                var d = x._data[i] - mean
                q = q + d * d
                i = i + 1

            var denom = Float64(n_total - ddof)
            var std_scalar = sqrt(q / denom)

            var out_data = List[Float64](); out_data.reserve(1); out_data.append(std_scalar)
            var out_shape = List[Int]()
            if keepdims:
                var t = 0;
                while t < rank: out_shape.append(1); t = t + 1
            else:
                out_shape.append(1)
            return Tensor[Float64](out_shape, out_data)

        # Generic (non-contiguous) whole-tensor: iterate indices to get mean then ssd
        var idx = List[Int](); var k = 0;
        while k < rank: idx.append(0); k = k + 1
        var sumv = 0.0; var count = 0
        var done = False
        while not done:
            var off = 0; var d = 0
            while d < rank: off = off + idx[d] * x._strides[d]; d = d + 1
            sumv = sumv + x._data[off]; count = count + 1
            var r = rank - 1
            while r >= 0:
                idx[r] = idx[r] + 1
                if idx[r] < shp[r]: break
                idx[r] = 0
                if r == 0: done = True; break
                r = r - 1
        #assert(count == n_total and "index walk mismatch")
        var mean_g = sumv / Float64(n_total)

        # Pass 2
        var idx2 = List[Int](); var k2 = 0;
        while k2 < rank: idx2.append(0); k2 = k2 + 1
        var q2 = 0.0; done = False
        while not done:
            var off2 = 0; var d2 = 0
            while d2 < rank: off2 = off2 + idx2[d2] * x._strides[d2]; d2 = d2 + 1
            var dv = x._data[off2] - mean_g
            q2 = q2 + dv * dv
            var r2 = rank - 1
            while r2 >= 0:
                idx2[r2] = idx2[r2] + 1
                if idx2[r2] < shp[r2]: break
                idx2[r2] = 0
                if r2 == 0: done = True; break
                r2 = r2 - 1
        var std_scalar_g = sqrt(q2 / Float64(n_total - ddof))
        var out_data_g = List[Float64](); out_data_g.append(std_scalar_g)
        var out_shape_g = List[Int]()
        if keepdims:
            var t2 = 0;
            while t2 < rank: out_shape_g.append(1); t2 = t2 + 1
        else:
            out_shape_g.append(1)
        return Tensor[Float64](out_shape_g, out_data_g)

    # ---------- AXIS REDUCTION ----------
    var ax = normalize_axis(axis.value(), rank)
    var reduce_n =shp[ax]
    if rank == 0: reduce_n =1
    #assert(reduce_n > ddof and "std: ddof must be < size along axis")

    if is_row_major_contiguous(shp, x._strides):
        # Pack outer/inner like your sum() and do two-pass along the reduced axis.
        var out_shape2 = shape_drop_axis(shp, ax)

        var inner = 1; var i_in = ax + 1
        while i_in < rank: inner = inner * shp[i_in]; i_in = i_in + 1
        var outer = 1; var i_out = 0
        while i_out < ax: outer = outer * shp[i_out]; i_out = i_out + 1

        var base_stride = inner
        var block = reduce_n * inner

        # ---- Pass 1: per-position mean ----
        var means = List[Float64](); means.reserve(outer * inner)
        var o = 0
        while o < outer * inner:
            var base = (o // inner) * block + (o % inner)
            var s = 0.0
            var k0 = 0; var lim0 = (reduce_n // 8) * 8
            while k0 < lim0:
                s = s + x._data[base + (k0    ) * base_stride]
                s = s + x._data[base + (k0 + 1) * base_stride]
                s = s + x._data[base + (k0 + 2) * base_stride]
                s = s + x._data[base + (k0 + 3) * base_stride]
                s = s + x._data[base + (k0 + 4) * base_stride]
                s = s + x._data[base + (k0 + 5) * base_stride]
                s = s + x._data[base + (k0 + 6) * base_stride]
                s = s + x._data[base + (k0 + 7) * base_stride]
                k0 = k0 + 8
            while k0 < reduce_n:
                s = s + x._data[base + k0 * base_stride]
                k0 = k0 + 1
            means.append(s / Float64(reduce_n))
            o = o + 1

        # ---- Pass 2: per-position SSD ----
        var outv = List[Float64](); outv.reserve(outer * inner)
        o = 0
        while o < outer * inner:
            var base = (o // inner) * block + (o % inner)
            var mu = means[o]
            var q = 0.0
            var k1 = 0; var lim1 = (reduce_n // 8) * 8
            while k1 < lim1:
                var d0 = x._data[base + (k1    ) * base_stride] - mu; q = q + d0 * d0
                var d1 = x._data[base + (k1 + 1) * base_stride] - mu; q = q + d1 * d1
                var d2 = x._data[base + (k1 + 2) * base_stride] - mu; q = q + d2 * d2
                var d3 = x._data[base + (k1 + 3) * base_stride] - mu; q = q + d3 * d3
                var d4 = x._data[base + (k1 + 4) * base_stride] - mu; q = q + d4 * d4
                var d5 = x._data[base + (k1 + 5) * base_stride] - mu; q = q + d5 * d5
                var d6 = x._data[base + (k1 + 6) * base_stride] - mu; q = q + d6 * d6
                var d7 = x._data[base + (k1 + 7) * base_stride] - mu; q = q + d7 * d7
                k1 = k1 + 8
            while k1 < reduce_n:
                var d = x._data[base + k1 * base_stride] - mu
                q = q + d * d
                k1 = k1 + 1
            outv.append(sqrt(q / Float64(reduce_n - ddof)))
            o = o + 1

        var tout = Tensor[Float64](out_shape2, outv)
        if keepdims:
            var kd = List[Int](); var r = 0
            while r < rank:
                if r == ax: kd.append(1) else: kd.append(shp[r])
                r = r + 1
            return tout.reshape(kd)
        return tout.copy()

    # ---------- Generic strided axis reduction ----------
    var out_shape_g = shape_drop_axis(shp, ax)
    var out_n = numel(out_shape_g)

    var outer_shape = List[Int](); var outer_strides = List[Int]()
    var i = 0
    while i < rank:
        if i != ax:
            outer_shape.append(shp[i])
            outer_strides.append(x._strides[i])
        i = i + 1
    var stride_red = x._strides[ax]

    # Pass 1: means
    var means_g = List[Float64](); means_g.reserve(out_n)
    var idx1 = List[Int](); var t1 = 0
    while t1 < len(outer_shape): idx1.append(0); t1 = t1 + 1

    var running = True
    while True:
        var base = 0; var d = 0
        while d < len(outer_shape):
            base = base + idx1[d] * outer_strides[d]
            d = d + 1
        var s = 0.0
        var j = 0; var lim = (reduce_n // 8) * 8
        while j < lim:
            s = s + x._data[base + (j    ) * stride_red]
            s = s + x._data[base + (j + 1) * stride_red]
            s = s + x._data[base + (j + 2) * stride_red]
            s = s + x._data[base + (j + 3) * stride_red]
            s = s + x._data[base + (j + 4) * stride_red]
            s = s + x._data[base + (j + 5) * stride_red]
            s = s + x._data[base + (j + 6) * stride_red]
            s = s + x._data[base + (j + 7) * stride_red]
            j = j + 8
        while j < reduce_n:
            s = s + x._data[base + j * stride_red]
            j = j + 1
        means_g.append(s / Float64(reduce_n))

        if len(outer_shape) == 0: break
        var pos = len(outer_shape) - 1
        var carry = True
        while pos >= 0 and carry:
            idx1[pos] = idx1[pos] + 1
            if idx1[pos] < outer_shape[pos]:
                carry = False
            else:
                idx1[pos] = 0
                if pos == 0:
                    carry = False; running = False
            if pos == 0: break
            pos = pos - 1
        if not running: break

    # Pass 2: SSDs -> std
    var outv_g = List[Float64](); outv_g.reserve(out_n)
    var idx2 = List[Int](); var t2 = 0
    while t2 < len(outer_shape): idx2.append(0); t2 = t2 + 1

    var kpos = 0
    running = True
    while True:
        var base2 = 0; var d2 = 0
        while d2 < len(outer_shape):
            base2 = base2 + idx2[d2] * outer_strides[d2]
            d2 = d2 + 1
        var mu = means_g[kpos]
        var q = 0.0
        var j2 = 0; var lim2 = (reduce_n // 8) * 8
        while j2 < lim2:
            var t0 = x._data[base2 + (j2    ) * stride_red] - mu; q = q + t0 * t0
            var t1v = x._data[base2 + (j2 + 1) * stride_red] - mu; q = q + t1v * t1v
            var t2v = x._data[base2 + (j2 + 2) * stride_red] - mu; q = q + t2v * t2v
            var t3v = x._data[base2 + (j2 + 3) * stride_red] - mu; q = q + t3v * t3v
            var t4v = x._data[base2 + (j2 + 4) * stride_red] - mu; q = q + t4v * t4v
            var t5v = x._data[base2 + (j2 + 5) * stride_red] - mu; q = q + t5v * t5v
            var t6v = x._data[base2 + (j2 + 6) * stride_red] - mu; q = q + t6v * t6v
            var t7v = x._data[base2 + (j2 + 7) * stride_red] - mu; q = q + t7v * t7v
            j2 = j2 + 8
        while j2 < reduce_n:
            var dv = x._data[base2 + j2 * stride_red] - mu
            q = q + dv * dv
            j2 = j2 + 1

        outv_g.append(sqrt(q / Float64(reduce_n - ddof)))
        kpos = kpos + 1

        if len(outer_shape) == 0: break
        var pos2 = len(outer_shape) - 1
        var carry2 = True
        while pos2 >= 0 and carry2:
            idx2[pos2] = idx2[pos2] + 1
            if idx2[pos2] < outer_shape[pos2]:
                carry2 = False
            else:
                idx2[pos2] = 0
                if pos2 == 0:
                    carry2 = False; running = False
            if pos2 == 0: break
            pos2 = pos2 - 1
        if not running: break

    var tout_g = Tensor[Float64](out_shape_g, outv_g)
    if keepdims:
        var kd2 = List[Int](); var r2 = 0
        while r2 < rank:
            if r2 == ax: kd2.append(1) else: kd2.append(shp[r2])
            r2 = r2 + 1
        return tout_g.reshape(kd2)
    return tout_g.copy()

# ======================= Axis-wise STD for Int (output Float64) =======================

@always_inline
fn std(x: Tensor[Int], axis: Optional[Int] = None, keepdims: Bool = False, ddof: Int = 0) -> Tensor[Float64]:
    # Two-pass mean/variance. Accumulate in Float64; return Tensor[Float64].
    var shp = x._shape.copy()
    var rank = len(shp)
    #assert(ddof >= 0 and "std: ddof must be >= 0")

    # ---------- WHOLE-TENSOR ----------
    if axis is None:
        var n_total = len(x._data)
        #assert(n_total > ddof and "std: ddof >= N")

        if (len(shp) == 1 or is_row_major_contiguous(shp, x._strides)):
            # Pass 1: mean (16-way unroll)
            var s = 0.0
            var i = 0
            var lim = (n_total // 16) * 16
            while i < lim:
                s = s + Float64(x._data[i     ]) + Float64(x._data[i + 1 ]) + Float64(x._data[i + 2 ]) + Float64(x._data[i + 3 ])
                s = s + Float64(x._data[i + 4 ]) + Float64(x._data[i + 5 ]) + Float64(x._data[i + 6 ]) + Float64(x._data[i + 7 ])
                s = s + Float64(x._data[i + 8 ]) + Float64(x._data[i + 9 ]) + Float64(x._data[i + 10]) + Float64(x._data[i + 11])
                s = s + Float64(x._data[i + 12]) + Float64(x._data[i + 13]) + Float64(x._data[i + 14]) + Float64(x._data[i + 15])
                i = i + 16
            while i < n_total:
                s = s + Float64(x._data[i])
                i = i + 1
            var mean = s / Float64(n_total)

            # Pass 2: sum of squared diffs (16-way)
            var q = 0.0
            i = 0
            while i < lim:
                var d0 = Float64(x._data[i     ]) - mean; q = q + d0 * d0
                var d1 = Float64(x._data[i + 1 ]) - mean; q = q + d1 * d1
                var d2 = Float64(x._data[i + 2 ]) - mean; q = q + d2 * d2
                var d3 = Float64(x._data[i + 3 ]) - mean; q = q + d3 * d3
                var d4 = Float64(x._data[i + 4 ]) - mean; q = q + d4 * d4
                var d5 = Float64(x._data[i + 5 ]) - mean; q = q + d5 * d5
                var d6 = Float64(x._data[i + 6 ]) - mean; q = q + d6 * d6
                var d7 = Float64(x._data[i + 7 ]) - mean; q = q + d7 * d7
                var d8 = Float64(x._data[i + 8 ]) - mean; q = q + d8 * d8
                var d9 = Float64(x._data[i + 9 ]) - mean; q = q + d9 * d9
                var dA = Float64(x._data[i + 10]) - mean; q = q + dA * dA
                var dB = Float64(x._data[i + 11]) - mean; q = q + dB * dB
                var dC = Float64(x._data[i + 12]) - mean; q = q + dC * dC
                var dD = Float64(x._data[i + 13]) - mean; q = q + dD * dD
                var dE = Float64(x._data[i + 14]) - mean; q = q + dE * dE
                var dF = Float64(x._data[i + 15]) - mean; q = q + dF * dF
                i = i + 16
            while i < n_total:
                var d = Float64(x._data[i]) - mean
                q = q + d * d
                i = i + 1

            var denom = Float64(n_total - ddof)
            var std_scalar = sqrt(q / denom)

            var out_data = List[Float64](); out_data.reserve(1); out_data.append(std_scalar)
            var out_shape = List[Int]()
            if keepdims:
                var t = 0
                while t < rank:
                    out_shape.append(1)
                    t = t + 1
            else:
                out_shape.append(1)
            return Tensor[Float64](out_shape, out_data)
        else:
            # Generic strided: Pass 1 (mean)
            var idx = List[Int](); var k = 0
            while k < rank:
                idx.append(0)
                k = k + 1
            var sumv = 0.0
            var count = 0
            var done = False
            while not done:
                var off = 0
                var d = 0
                while d < rank:
                    off = off + idx[d] * x._strides[d]
                    d = d + 1
                sumv = sumv + Float64(x._data[off]); count = count + 1

                var r = rank - 1
                while r >= 0:
                    idx[r] = idx[r] + 1
                    if idx[r] < shp[r]: break
                    idx[r] = 0
                    if r == 0: done = True; break
                    r = r - 1
            #assert(count == n_total and "index walk mismatch")
            var mean_g = sumv / Float64(n_total)

            # Pass 2 (SSD)
            var idx2 = List[Int](); var k2 = 0
            while k2 < rank:
                idx2.append(0)
                k2 = k2 + 1
            var q2 = 0.0; done = False
            while not done:
                var off2 = 0; var d2 = 0
                while d2 < rank:
                    off2 = off2 + idx2[d2] * x._strides[d2]
                    d2 = d2 + 1
                var dv = Float64(x._data[off2]) - mean_g
                q2 = q2 + dv * dv
                var r2 = rank - 1
                while r2 >= 0:
                    idx2[r2] = idx2[r2] + 1
                    if idx2[r2] < shp[r2]: break
                    idx2[r2] = 0
                    if r2 == 0: done = True; break
                    r2 = r2 - 1
            var std_scalar_g = sqrt(q2 / Float64(n_total - ddof))
            var out_data_g = List[Float64](); out_data_g.append(std_scalar_g)
            var out_shape_g = List[Int]()
            if keepdims:
                var t2 = 0
                while t2 < rank:
                    out_shape_g.append(1)
                    t2 = t2 + 1
            else:
                out_shape_g.append(1)
            return Tensor[Float64](out_shape_g, out_data_g)

    # ---------- AXIS REDUCTION ----------
    var ax = normalize_axis(axis.value(), rank)
    var reduce_n =shp[ax]
    if rank == 0: reduce_n =1
    #assert(reduce_n > ddof and "std: ddof must be < size along axis")

    if is_row_major_contiguous(shp, x._strides):
        var out_shape2 = shape_drop_axis(shp, ax)

        var inner = 1; var i_in = ax + 1
        while i_in < rank: inner = inner * shp[i_in]; i_in = i_in + 1
        var outer = 1; var i_out = 0
        while i_out < ax: outer = outer * shp[i_out]; i_out = i_out + 1

        var base_stride = inner
        var block = reduce_n * inner

        # Pass 1: means
        var means = List[Float64](); means.reserve(outer * inner)
        var o = 0
        while o < outer * inner:
            var base = (o // inner) * block + (o % inner)
            var s = 0.0
            var k0 = 0; var lim0 = (reduce_n // 8) * 8
            while k0 < lim0:
                s = s + Float64(x._data[base + (k0    ) * base_stride])
                s = s + Float64(x._data[base + (k0 + 1) * base_stride])
                s = s + Float64(x._data[base + (k0 + 2) * base_stride])
                s = s + Float64(x._data[base + (k0 + 3) * base_stride])
                s = s + Float64(x._data[base + (k0 + 4) * base_stride])
                s = s + Float64(x._data[base + (k0 + 5) * base_stride])
                s = s + Float64(x._data[base + (k0 + 6) * base_stride])
                s = s + Float64(x._data[base + (k0 + 7) * base_stride])
                k0 = k0 + 8
            while k0 < reduce_n:
                s = s + Float64(x._data[base + k0 * base_stride])
                k0 = k0 + 1
            means.append(s / Float64(reduce_n))
            o = o + 1

        # Pass 2: SSD -> std
        var outv = List[Float64](); outv.reserve(outer * inner)
        o = 0
        while o < outer * inner:
            var base2 = (o // inner) * block + (o % inner)
            var mu = means[o]
            var q = 0.0
            var k1 = 0; var lim1 = (reduce_n // 8) * 8
            while k1 < lim1:
                var d0 = Float64(x._data[base2 + (k1    ) * base_stride]) - mu; q = q + d0 * d0
                var d1 = Float64(x._data[base2 + (k1 + 1) * base_stride]) - mu; q = q + d1 * d1
                var d2 = Float64(x._data[base2 + (k1 + 2) * base_stride]) - mu; q = q + d2 * d2
                var d3 = Float64(x._data[base2 + (k1 + 3) * base_stride]) - mu; q = q + d3 * d3
                var d4 = Float64(x._data[base2 + (k1 + 4) * base_stride]) - mu; q = q + d4 * d4
                var d5 = Float64(x._data[base2 + (k1 + 5) * base_stride]) - mu; q = q + d5 * d5
                var d6 = Float64(x._data[base2 + (k1 + 6) * base_stride]) - mu; q = q + d6 * d6
                var d7 = Float64(x._data[base2 + (k1 + 7) * base_stride]) - mu; q = q + d7 * d7
                k1 = k1 + 8
            while k1 < reduce_n:
                var d = Float64(x._data[base2 + k1 * base_stride]) - mu
                q = q + d * d
                k1 = k1 + 1
            outv.append(sqrt(q / Float64(reduce_n - ddof)))
            o = o + 1

        var tout = Tensor[Float64](out_shape2, outv)
        if keepdims:
            var kd = List[Int](); var r = 0
            while r < rank:
                if r == ax: kd.append(1) else: kd.append(shp[r])
                r = r + 1
            return tout.reshape(kd)
        return tout.copy()

    # ---------- Generic strided axis reduction ----------
    var out_shape_g = shape_drop_axis(shp, ax)
    var out_n = numel(out_shape_g)

    var outer_shape = List[Int](); var outer_strides = List[Int]()
    var i2 = 0
    while i2 < rank:
        if i2 != ax:
            outer_shape.append(shp[i2])
            outer_strides.append(x._strides[i2])
        i2 = i2 + 1
    var stride_red = x._strides[ax]

    # Pass 1: means
    var means_g = List[Float64](); means_g.reserve(out_n)
    var idx1 = List[Int](); var t1 = 0
    while t1 < len(outer_shape): idx1.append(0); t1 = t1 + 1

    var running = True
    while True:
        var base = 0; var d = 0
        while d < len(outer_shape):
            base = base + idx1[d] * outer_strides[d]
            d = d + 1
        var s = 0.0
        var j = 0; var lim = (reduce_n // 8) * 8
        while j < lim:
            s = s + Float64(x._data[base + (j    ) * stride_red])
            s = s + Float64(x._data[base + (j + 1) * stride_red])
            s = s + Float64(x._data[base + (j + 2) * stride_red])
            s = s + Float64(x._data[base + (j + 3) * stride_red])
            s = s + Float64(x._data[base + (j + 4) * stride_red])
            s = s + Float64(x._data[base + (j + 5) * stride_red])
            s = s + Float64(x._data[base + (j + 6) * stride_red])
            s = s + Float64(x._data[base + (j + 7) * stride_red])
            j = j + 8
        while j < reduce_n:
            s = s + Float64(x._data[base + j * stride_red])
            j = j + 1
        means_g.append(s / Float64(reduce_n))

        if len(outer_shape) == 0: break
        var pos = len(outer_shape) - 1
        var carry = True
        while pos >= 0 and carry:
            idx1[pos] = idx1[pos] + 1
            if idx1[pos] < outer_shape[pos]:
                carry = False
            else:
                idx1[pos] = 0
                if pos == 0:
                    carry = False; running = False
            if pos == 0: break
            pos = pos - 1
        if not running: break

    # Pass 2: SSDs -> std
    var outv_g = List[Float64](); outv_g.reserve(out_n)
    var idx2 = List[Int](); var t2 = 0
    while t2 < len(outer_shape): idx2.append(0); t2 = t2 + 1

    var kpos = 0
    running = True
    while True:
        var base2 = 0; var d2 = 0
        while d2 < len(outer_shape):
            base2 = base2 + idx2[d2] * outer_strides[d2]
            d2 = d2 + 1
        var mu = means_g[kpos]
        var q = 0.0
        var j2 = 0; var lim2 = (reduce_n // 8) * 8
        while j2 < lim2:
            var t0 = Float64(x._data[base2 + (j2    ) * stride_red]) - mu; q = q + t0 * t0
            var t1v = Float64(x._data[base2 + (j2 + 1) * stride_red]) - mu; q = q + t1v * t1v
            var t2v = Float64(x._data[base2 + (j2 + 2) * stride_red]) - mu; q = q + t2v * t2v
            var t3v = Float64(x._data[base2 + (j2 + 3) * stride_red]) - mu; q = q + t3v * t3v
            var t4v = Float64(x._data[base2 + (j2 + 4) * stride_red]) - mu; q = q + t4v * t4v
            var t5v = Float64(x._data[base2 + (j2 + 5) * stride_red]) - mu; q = q + t5v * t5v
            var t6v = Float64(x._data[base2 + (j2 + 6) * stride_red]) - mu; q = q + t6v * t6v
            var t7v = Float64(x._data[base2 + (j2 + 7) * stride_red]) - mu; q = q + t7v * t7v
            j2 = j2 + 8
        while j2 < reduce_n:
            var dv = Float64(x._data[base2 + j2 * stride_red]) - mu
            q = q + dv * dv
            j2 = j2 + 1

        outv_g.append(sqrt(q / Float64(reduce_n - ddof)))
        kpos = kpos + 1

        if len(outer_shape) == 0: break
        var pos2 = len(outer_shape) - 1
        var carry2 = True
        while pos2 >= 0 and carry2:
            idx2[pos2] = idx2[pos2] + 1
            if idx2[pos2] < outer_shape[pos2]:
                carry2 = False
            else:
                idx2[pos2] = 0
                if pos2 == 0:
                    carry2 = False; running = False
            if pos2 == 0: break
            pos2 = pos2 - 1
        if not running: break

    var tout_g = Tensor[Float64](out_shape_g, outv_g)
    if keepdims:
        var kd2 = List[Int](); var r3 = 0
        while r3 < rank:
            if r3 == ax: kd2.append(1) else: kd2.append(shp[r3])
            r3 = r3 + 1
        return tout_g.reshape(kd2)
    return tout_g.copy()

# ======================= Axis-wise STD for Float32 (output Float64) =======================

@always_inline
fn std(x: Tensor[Float32], axis: Optional[Int] = None, keepdims: Bool = False, ddof: Int = 0) -> Tensor[Float64]:
    # Two-pass mean/variance. Accumulate in Float64; return Tensor[Float64].
    var shp = x._shape.copy()
    var rank = len(shp)
    #assert(ddof >= 0 and "std: ddof must be >= 0")

    # ---------- WHOLE-TENSOR ----------
    if axis is None:
        var n_total = len(x._data)
        #assert(n_total > ddof and "std: ddof >= N")

        if (len(shp) == 1 or is_row_major_contiguous(shp, x._strides)):
            # Pass 1: mean (16-way unroll)
            var s = 0.0
            var i = 0
            var lim = (n_total // 16) * 16
            while i < lim:
                s = s + Float64(x._data[i     ]) + Float64(x._data[i + 1 ]) + Float64(x._data[i + 2 ]) + Float64(x._data[i + 3 ])
                s = s + Float64(x._data[i + 4 ]) + Float64(x._data[i + 5 ]) + Float64(x._data[i + 6 ]) + Float64(x._data[i + 7 ])
                s = s + Float64(x._data[i + 8 ]) + Float64(x._data[i + 9 ]) + Float64(x._data[i + 10]) + Float64(x._data[i + 11])
                s = s + Float64(x._data[i + 12]) + Float64(x._data[i + 13]) + Float64(x._data[i + 14]) + Float64(x._data[i + 15])
                i = i + 16
            while i < n_total:
                s = s + Float64(x._data[i])
                i = i + 1
            var mean = s / Float64(n_total)

            # Pass 2: sum of squared diffs
            var q = 0.0
            i = 0
            while i < lim:
                var d0 = Float64(x._data[i     ]) - mean; q = q + d0 * d0
                var d1 = Float64(x._data[i + 1 ]) - mean; q = q + d1 * d1
                var d2 = Float64(x._data[i + 2 ]) - mean; q = q + d2 * d2
                var d3 = Float64(x._data[i + 3 ]) - mean; q = q + d3 * d3
                var d4 = Float64(x._data[i + 4 ]) - mean; q = q + d4 * d4
                var d5 = Float64(x._data[i + 5 ]) - mean; q = q + d5 * d5
                var d6 = Float64(x._data[i + 6 ]) - mean; q = q + d6 * d6
                var d7 = Float64(x._data[i + 7 ]) - mean; q = q + d7 * d7
                var d8 = Float64(x._data[i + 8 ]) - mean; q = q + d8 * d8
                var d9 = Float64(x._data[i + 9 ]) - mean; q = q + d9 * d9
                var dA = Float64(x._data[i + 10]) - mean; q = q + dA * dA
                var dB = Float64(x._data[i + 11]) - mean; q = q + dB * dB
                var dC = Float64(x._data[i + 12]) - mean; q = q + dC * dC
                var dD = Float64(x._data[i + 13]) - mean; q = q + dD * dD
                var dE = Float64(x._data[i + 14]) - mean; q = q + dE * dE
                var dF = Float64(x._data[i + 15]) - mean; q = q + dF * dF
                i = i + 16
            while i < n_total:
                var d = Float64(x._data[i]) - mean
                q = q + d * d
                i = i + 1

            var std_scalar = sqrt(q / Float64(n_total - ddof))

            var out_data = List[Float64](); out_data.reserve(1); out_data.append(std_scalar)
            var out_shape = List[Int]()
            if keepdims:
                var t = 0
                while t < rank:
                    out_shape.append(1)
                    t = t + 1
            else:
                out_shape.append(1)
            return Tensor[Float64](out_shape, out_data)
        else:
            # Generic strided: Pass 1 (mean)
            var idx = List[Int](); var k = 0
            while k < rank:
                idx.append(0)
                k = k + 1
            var sumv = 0.0
            var count = 0
            var done = False
            while not done:
                var off = 0; var d = 0
                while d < rank:
                    off = off + idx[d] * x._strides[d]
                    d = d + 1
                sumv = sumv + Float64(x._data[off]); count = count + 1
                var r = rank - 1
                while r >= 0:
                    idx[r] = idx[r] + 1
                    if idx[r] < shp[r]: break
                    idx[r] = 0
                    if r == 0: done = True; break
                    r = r - 1
            var mean_g = sumv / Float64(n_total)

            # Pass 2 (SSD)
            var idx2 = List[Int](); var k2 = 0
            while k2 < rank:
                idx2.append(0)
                k2 = k2 + 1
            var q2 = 0.0; done = False
            while not done:
                var off2 = 0; var d2 = 0
                while d2 < rank:
                    off2 = off2 + idx2[d2] * x._strides[d2]
                    d2 = d2 + 1
                var dv = Float64(x._data[off2]) - mean_g
                q2 = q2 + dv * dv
                var r2 = rank - 1
                while r2 >= 0:
                    idx2[r2] = idx2[r2] + 1
                    if idx2[r2] < shp[r2]: break
                    idx2[r2] = 0
                    if r2 == 0: done = True; break
                    r2 = r2 - 1
            var std_scalar_g = sqrt(q2 / Float64(n_total - ddof))
            var out_data_g = List[Float64](); out_data_g.append(std_scalar_g)
            var out_shape_g = List[Int]()
            if keepdims:
                var t2 = 0
                while t2 < rank:
                    out_shape_g.append(1)
                    t2 = t2 + 1
            else:
                out_shape_g.append(1)
            return Tensor[Float64](out_shape_g, out_data_g)

    # ---------- AXIS REDUCTION ----------
    var ax = normalize_axis(axis.value(), rank)
    var reduce_n =shp[ax]
    if rank == 0: reduce_n =1
    #assert(reduce_n > ddof and "std: ddof must be < size along axis")

    if is_row_major_contiguous(shp, x._strides):
        var out_shape2 = shape_drop_axis(shp, ax)

        var inner = 1; var i_in = ax + 1
        while i_in < rank: inner = inner * shp[i_in]; i_in = i_in + 1
        var outer = 1; var i_out = 0
        while i_out < ax: outer = outer * shp[i_out]; i_out = i_out + 1

        var base_stride = inner
        var block = reduce_n * inner

        # Pass 1: means
        var means = List[Float64](); means.reserve(outer * inner)
        var o = 0
        while o < outer * inner:
            var base = (o // inner) * block + (o % inner)
            var s = 0.0
            var k0 = 0; var lim0 = (reduce_n // 8) * 8
            while k0 < lim0:
                s = s + Float64(x._data[base + (k0    ) * base_stride])
                s = s + Float64(x._data[base + (k0 + 1) * base_stride])
                s = s + Float64(x._data[base + (k0 + 2) * base_stride])
                s = s + Float64(x._data[base + (k0 + 3) * base_stride])
                s = s + Float64(x._data[base + (k0 + 4) * base_stride])
                s = s + Float64(x._data[base + (k0 + 5) * base_stride])
                s = s + Float64(x._data[base + (k0 + 6) * base_stride])
                s = s + Float64(x._data[base + (k0 + 7) * base_stride])
                k0 = k0 + 8
            while k0 < reduce_n:
                s = s + Float64(x._data[base + k0 * base_stride])
                k0 = k0 + 1
            means.append(s / Float64(reduce_n))
            o = o + 1

        # Pass 2: SSD -> std
        var outv = List[Float64](); outv.reserve(outer * inner)
        o = 0
        while o < outer * inner:
            var base2 = (o // inner) * block + (o % inner)
            var mu = means[o]
            var q = 0.0
            var k1 = 0; var lim1 = (reduce_n // 8) * 8
            while k1 < lim1:
                var d0 = Float64(x._data[base2 + (k1    ) * base_stride]) - mu; q = q + d0 * d0
                var d1 = Float64(x._data[base2 + (k1 + 1) * base_stride]) - mu; q = q + d1 * d1
                var d2 = Float64(x._data[base2 + (k1 + 2) * base_stride]) - mu; q = q + d2 * d2
                var d3 = Float64(x._data[base2 + (k1 + 3) * base_stride]) - mu; q = q + d3 * d3
                var d4 = Float64(x._data[base2 + (k1 + 4) * base_stride]) - mu; q = q + d4 * d4
                var d5 = Float64(x._data[base2 + (k1 + 5) * base_stride]) - mu; q = q + d5 * d5
                var d6 = Float64(x._data[base2 + (k1 + 6) * base_stride]) - mu; q = q + d6 * d6
                var d7 = Float64(x._data[base2 + (k1 + 7) * base_stride]) - mu; q = q + d7 * d7
                k1 = k1 + 8
            while k1 < reduce_n:
                var d = Float64(x._data[base2 + k1 * base_stride]) - mu
                q = q + d * d
                k1 = k1 + 1
            outv.append(sqrt(q / Float64(reduce_n - ddof)))
            o = o + 1

        var tout = Tensor[Float64](out_shape2, outv)
        if keepdims:
            var kd = List[Int](); var r = 0
            while r < rank:
                if r == ax: kd.append(1) else: kd.append(shp[r])
                r = r + 1
            return tout.reshape(kd)
        return tout.copy()

    # ---------- Generic strided axis reduction ----------
    var out_shape_g = shape_drop_axis(shp, ax)
    var out_n = numel(out_shape_g)

    var outer_shape = List[Int](); var outer_strides = List[Int]()
    var i2 = 0
    while i2 < rank:
        if i2 != ax:
            outer_shape.append(shp[i2])
            outer_strides.append(x._strides[i2])
        i2 = i2 + 1
    var stride_red = x._strides[ax]

    # Pass 1: means
    var means_g = List[Float64](); means_g.reserve(out_n)
    var idx1 = List[Int](); var t1 = 0
    while t1 < len(outer_shape): idx1.append(0); t1 = t1 + 1

    var running = True
    while True:
        var base = 0; var d = 0
        while d < len(outer_shape):
            base = base + idx1[d] * outer_strides[d]
            d = d + 1
        var s = 0.0
        var j = 0; var lim = (reduce_n // 8) * 8
        while j < lim:
            s = s + Float64(x._data[base + (j    ) * stride_red])
            s = s + Float64(x._data[base + (j + 1) * stride_red])
            s = s + Float64(x._data[base + (j + 2) * stride_red])
            s = s + Float64(x._data[base + (j + 3) * stride_red])
            s = s + Float64(x._data[base + (j + 4) * stride_red])
            s = s + Float64(x._data[base + (j + 5) * stride_red])
            s = s + Float64(x._data[base + (j + 6) * stride_red])
            s = s + Float64(x._data[base + (j + 7) * stride_red])
            j = j + 8
        while j < reduce_n:
            s = s + Float64(x._data[base + j * stride_red])
            j = j + 1
        means_g.append(s / Float64(reduce_n))

        if len(outer_shape) == 0: break
        var pos = len(outer_shape) - 1
        var carry = True
        while pos >= 0 and carry:
            idx1[pos] = idx1[pos] + 1
            if idx1[pos] < outer_shape[pos]:
                carry = False
            else:
                idx1[pos] = 0
                if pos == 0:
                    carry = False; running = False
            if pos == 0: break
            pos = pos - 1
        if not running: break

    # Pass 2: SSDs -> std
    var outv_g = List[Float64](); outv_g.reserve(out_n)
    var idx2 = List[Int](); var t2 = 0
    while t2 < len(outer_shape): idx2.append(0); t2 = t2 + 1

    var kpos = 0
    running = True
    while True:
        var base2 = 0; var d2 = 0
        while d2 < len(outer_shape):
            base2 = base2 + idx2[d2] * outer_strides[d2]
            d2 = d2 + 1
        var mu = means_g[kpos]
        var q = 0.0
        var j2 = 0; var lim2 = (reduce_n // 8) * 8
        while j2 < lim2:
            var t0 = Float64(x._data[base2 + (j2    ) * stride_red]) - mu; q = q + t0 * t0
            var t1v = Float64(x._data[base2 + (j2 + 1) * stride_red]) - mu; q = q + t1v * t1v
            var t2v = Float64(x._data[base2 + (j2 + 2) * stride_red]) - mu; q = q + t2v * t2v
            var t3v = Float64(x._data[base2 + (j2 + 3) * stride_red]) - mu; q = q + t3v * t3v
            var t4v = Float64(x._data[base2 + (j2 + 4) * stride_red]) - mu; q = q + t4v * t4v
            var t5v = Float64(x._data[base2 + (j2 + 5) * stride_red]) - mu; q = q + t5v * t5v
            var t6v = Float64(x._data[base2 + (j2 + 6) * stride_red]) - mu; q = q + t6v * t6v
            var t7v = Float64(x._data[base2 + (j2 + 7) * stride_red]) - mu; q = q + t7v * t7v
            j2 = j2 + 8
        while j2 < reduce_n:
            var dv = Float64(x._data[base2 + j2 * stride_red]) - mu
            q = q + dv * dv
            j2 = j2 + 1

        outv_g.append(sqrt(q / Float64(reduce_n - ddof)))
        kpos = kpos + 1

        if len(outer_shape) == 0: break
        var pos2 = len(outer_shape) - 1
        var carry2 = True
        while pos2 >= 0 and carry2:
            idx2[pos2] = idx2[pos2] + 1
            if idx2[pos2] < outer_shape[pos2]:
                carry2 = False
            else:
                idx2[pos2] = 0
                if pos2 == 0:
                    carry2 = False; running = False
            if pos2 == 0: break
            pos2 = pos2 - 1
        if not running: break

    var tout_g = Tensor[Float64](out_shape_g, outv_g)
    if keepdims:
        var kd2 = List[Int](); var r3 = 0
        while r3 < rank:
            if r3 == ax: kd2.append(1) else: kd2.append(shp[r3])
            r3 = r3 + 1
        return tout_g.reshape(kd2)
    return tout_g.copy()


# Generic mean that computes in Float64 for stability, then constructs T
# ---------- MEAN for Float64 ----------
fn mean(x: Tensor[Float64], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float64]:
    var shp = x._shape.copy()
    var rank = len(shp)

    # whole-tensor mean
    if axis is None:
        var n = numel(shp)
        var out_shape = List[Int]()
        if keepdims:
            out_shape = _keepdims_shape_all(shp)
        else:
            out_shape.reserve(1)
            out_shape.append(1)

        var data = List[Float64]()
        data.reserve(1)

        if n == 0:
            data.append(0.0)
            return Tensor[Float64](out_shape, data)

        var s = 0.0
        if len(shp) == 1 or is_row_major_contiguous(shp, x._strides):
            var n0 = len(x._data)
            var i0 = 0
            var lim = (n0 // 16) * 16
            while i0 < lim:
                s = s + x._data[i0 +  0]; s = s + x._data[i0 +  1]
                s = s + x._data[i0 +  2]; s = s + x._data[i0 +  3]
                s = s + x._data[i0 +  4]; s = s + x._data[i0 +  5]
                s = s + x._data[i0 +  6]; s = s + x._data[i0 +  7]
                s = s + x._data[i0 +  8]; s = s + x._data[i0 +  9]
                s = s + x._data[i0 + 10]; s = s + x._data[i0 + 11]
                s = s + x._data[i0 + 12]; s = s + x._data[i0 + 13]
                s = s + x._data[i0 + 14]; s = s + x._data[i0 + 15]
                i0 = i0 + 16
            while i0 < n0:
                s = s + x._data[i0]
                i0 = i0 + 1
        else:
            var idx = List[Int]()
            idx.reserve(rank)
            var i = 0
            while i < n:
                unravel_index(i, shp, idx)
                var li = lin_index(idx, x._strides)
                s = s + x._data[li]
                i = i + 1

        var m = s / Float64(n)
        data.append(m)
        return Tensor[Float64](out_shape, data)

    # axis reduction
    var ax = normalize_axis(axis.value(), rank)
    var reduce_n = 1
    if rank != 0:
        reduce_n = shp[ax]

    if is_row_major_contiguous(shp, x._strides):
        var out_shape = shape_drop_axis(shp, ax)

        var inner = 1
        var i_in = ax + 1
        while i_in < rank:
            inner = inner * shp[i_in]
            i_in = i_in + 1

        var outer = 1
        var i_out = 0
        while i_out < ax:
            outer = outer * shp[i_out]
            i_out = i_out + 1

        var base_stride = inner
        var block = reduce_n * inner

        var outv = List[Float64]()
        outv.reserve(outer * inner)

        var o = 0
        while o < outer * inner:
            var base = (o // inner) * block + (o % inner)
            var s2 = 0.0

            var k2 = 0
            var lim2 = (reduce_n // 8) * 8
            while k2 < lim2:
                s2 = s2 + x._data[base + (k2    ) * base_stride]
                s2 = s2 + x._data[base + (k2 + 1) * base_stride]
                s2 = s2 + x._data[base + (k2 + 2) * base_stride]
                s2 = s2 + x._data[base + (k2 + 3) * base_stride]
                s2 = s2 + x._data[base + (k2 + 4) * base_stride]
                s2 = s2 + x._data[base + (k2 + 5) * base_stride]
                s2 = s2 + x._data[base + (k2 + 6) * base_stride]
                s2 = s2 + x._data[base + (k2 + 7) * base_stride]
                k2 = k2 + 8
            while k2 < reduce_n:
                s2 = s2 + x._data[base + k2 * base_stride]
                k2 = k2 + 1

            var mu = 0.0
            if reduce_n > 0:
                mu = s2 / Float64(reduce_n)
            outv.append(mu)
            o = o + 1

        var tout = Tensor[Float64](out_shape, outv)
        if keepdims:
            var kd = _keepdims_shape(shp, ax)
            return tout.reshape(kd)
        return tout.copy()

    # generic strided axis reduction
    var out_shape_g = shape_drop_axis(shp, ax)
    var out_n = numel(out_shape_g)

    var outer_shape = List[Int]()
    var outer_strides = List[Int]()
    var i3 = 0
    while i3 < rank:
        if i3 != ax:
            outer_shape.append(shp[i3])
            outer_strides.append(x._strides[i3])
        i3 = i3 + 1
    var stride_red = x._strides[ax]

    var outv2 = List[Float64]()
    outv2.reserve(out_n)

    var idx2 = List[Int]()
    var k3 = 0
    while k3 < len(outer_shape):
        idx2.append(0)
        k3 = k3 + 1

    var running = True
    while True:
        var base2 = 0
        var d2 = 0
        while d2 < len(outer_shape):
            base2 = base2 + idx2[d2] * outer_strides[d2]
            d2 = d2 + 1

        var acc = 0.0
        var j = 0
        var lim = (reduce_n // 8) * 8
        while j < lim:
            acc = acc + x._data[base2 + (j    ) * stride_red]
            acc = acc + x._data[base2 + (j + 1) * stride_red]
            acc = acc + x._data[base2 + (j + 2) * stride_red]
            acc = acc + x._data[base2 + (j + 3) * stride_red]
            acc = acc + x._data[base2 + (j + 4) * stride_red]
            acc = acc + x._data[base2 + (j + 5) * stride_red]
            acc = acc + x._data[base2 + (j + 6) * stride_red]
            acc = acc + x._data[base2 + (j + 7) * stride_red]
            j = j + 8
        while j < reduce_n:
            acc = acc + x._data[base2 + j * stride_red]
            j = j + 1

        var mu2 = 0.0
        if reduce_n > 0:
            mu2 = acc / Float64(reduce_n)
        outv2.append(mu2)

        if len(outer_shape) == 0:
            break
        var pos = len(outer_shape) - 1
        var carry = True
        while pos >= 0 and carry:
            idx2[pos] = idx2[pos] + 1
            if idx2[pos] < outer_shape[pos]:
                carry = False
            else:
                idx2[pos] = 0
                if pos == 0:
                    carry = False
                    running = False
            if pos == 0:
                break
            pos = pos - 1
        if not running:
            break

    var tout_g = Tensor[Float64](out_shape_g, outv2)
    if keepdims:
        var kd2 = _keepdims_shape(shp, ax)
        return tout_g.reshape(kd2)
    return tout_g.copy()


@always_inline
fn mean_axes_f64(
    x0: Tensor[Float64], axes_in: List[Int], keepdims: Bool = False
) -> Tensor[Float64]:
    var x = x0.copy()
    var r = len(x._shape)

    # normalize & dedup
    var axes = List[Int]()
    axes.reserve(len(axes_in))
    var seen = List[Int]()
    var i = 0
    while i < len(axes_in):
        var a = axes_in[i]
        if a < 0: a = a + r
        # bounds check
        if a >= 0 and a < r:
            # dedup
            var dup = False
            var j = 0
            while j < len(seen):
                if seen[j] == a: dup = True; break
                j += 1
            if not dup:
                axes.append(a)
                seen.append(a)
        i += 1

    # sort descending
    var n = len(axes)
    var j2 = 0
    while j2 < n:
        var m = j2
        var k = j2 + 1
        while k < n:
            if axes[k] > axes[m]: m = k
            k += 1
        var tmp = axes[j2]; axes[j2] = axes[m]; axes[m] = tmp
        j2 += 1

    # reduce one-by-one, adjusting later axes when keepdims == False
    var t = 0
    while t < len(axes):
        var a = axes[t]
        var opt = Optional[Int](a)
        x = mean(x, opt, keepdims)
        if not keepdims:

            var u = t + 1
            while u < len(axes):
                if axes[u] > a:
                    axes[u] = axes[u] - 1
                u += 1
        t += 1

    return x.copy()

 # ========= variance for Float64 (Welford, 4-arg ctor) =========
@always_inline
fn variance(
    x: Tensor[Float64],
    axis: Optional[Int] = None,
    unbiased: Bool = True,
    keepdims: Bool = False
) -> Tensor[Float64]:
    var shp = x._shape.copy()
    var rank = len(shp)

    # ---- WHOLE TENSOR ----
    if axis is None:
        var n = len(x._data)
        var out_list = List[Float64]()
        out_list.reserve(1)

        if n == 0:
            out_list.append(0.0)
        else:
            var mean_ = 0.0
            var m2 = 0.0
            var i = 0
            while i < n:
                var v = x._data[i]
                var delta = v - mean_
                mean_ = mean_ + (delta / Float64(i + 1))
                var t = v - mean_
                m2 = m2 + (delta * t)
                i = i + 1
            var denom = if unbiased and n > 1: Float64(n - 1) else: Float64(n)
            var vv = if denom > 0.0: m2 / denom else: 0.0
            out_list.append(vv)

        var out_shape = List[Int]()
        if keepdims:
            var j = 0
            while j < rank:
                out_shape.append(1)
                j = j + 1
        else:
            out_shape.append(1)

        var st = compute_row_major_strides(out_shape)
        return Tensor[Float64](out_list, out_shape, st, 0)

    # ---- AXIS REDUCTION ----
    var ax = normalize_axis(axis.value(), rank)
    var out_shape2 = shape_drop_axis(shp, ax)

    var reduce_n = if rank == 0: 1 else: shp[ax]

    var inner = 1
    var i_in = ax + 1
    while i_in < rank:
        inner = inner * shp[i_in]
        i_in = i_in + 1

    var outer = 1
    var i_out = 0
    while i_out < ax:
        outer = outer * shp[i_out]
        i_out = i_out + 1
    outer = outer * inner

    var base_stride = inner
    var block = reduce_n * inner

    var out_vals = List[Float64]()
    out_vals.reserve(outer)

    var o = 0
    while o < outer:
        var base = (o // inner) * block + (o % inner)

        var mean2 = 0.0
        var m22 = 0.0
        var k = 0
        while k < reduce_n:
            var v2 = x._data[base + k * base_stride]
            var delta2 = v2 - mean2
            mean2 = mean2 + (delta2 / Float64(k + 1))
            var t2 = v2 - mean2
            m22 = m22 + (delta2 * t2)
            k = k + 1

        var denom2 = if unbiased and reduce_n > 1: Float64(reduce_n - 1) else: Float64(reduce_n)
        var vv2 = if denom2 > 0.0: m22 / denom2 else: 0.0
        out_vals.append(vv2)
        o = o + 1

    var st2 = compute_row_major_strides(out_shape2)
    var tout = Tensor[Float64](out_vals, out_shape2, st2, 0)

    if keepdims:
        var kd = List[Int]()
        var r = 0
        while r < rank:
            if r == ax: kd.append(1) else: kd.append(shp[r])
            r = r + 1
        return tout.reshape(kd)

    return tout

# ========= variance for Float32 (accumulate in f64, return Float64) =========
@always_inline
fn variance(
    x: Tensor[Float32],
    axis: Optional[Int] = None,
    unbiased: Bool = True,
    keepdims: Bool = False
) -> Tensor[Float64]:
    var shp = x._shape.copy()
    var rank = len(shp)

    # ---- WHOLE TENSOR ----
    if axis is None:
        var n = len(x._data)
        var out_list = List[Float64]()
        out_list.reserve(1)

        if n == 0:
            out_list.append(0.0)
        else:
            var mean_ = 0.0
            var m2 = 0.0
            var i = 0
            while i < n:
                var v = Float64(x._data[i])
                var delta = v - mean_
                mean_ = mean_ + (delta / Float64(i + 1))
                var t = v - mean_
                m2 = m2 + (delta * t)
                i = i + 1
            var denom = if unbiased and n > 1: Float64(n - 1) else: Float64(n)
            var vv = if denom > 0.0: m2 / denom else: 0.0
            out_list.append(vv)

        var out_shape = List[Int]()
        if keepdims:
            var j = 0
            while j < rank:
                out_shape.append(1)
                j = j + 1
        else:
            out_shape.append(1)

        var st = compute_row_major_strides(out_shape)
        return Tensor[Float64](out_list, out_shape, st, 0)

    # ---- AXIS REDUCTION ----
    var ax = normalize_axis(axis.value(), rank)
    var out_shape2 = shape_drop_axis(shp, ax)

    var reduce_n = if rank == 0: 1 else: shp[ax]

    var inner = 1
    var i_in = ax + 1
    while i_in < rank:
        inner = inner * shp[i_in]
        i_in = i_in + 1

    var outer = 1
    var i_out = 0
    while i_out < ax:
        outer = outer * shp[i_out]
        i_out = i_out + 1
    outer = outer * inner

    var base_stride = inner
    var block = reduce_n * inner

    var out_vals = List[Float64]()
    out_vals.reserve(outer)

    var o = 0
    while o < outer:
        var base = (o // inner) * block + (o % inner)

        var mean2 = 0.0
        var m22 = 0.0
        var k = 0
        while k < reduce_n:
            var v2 = Float64(x._data[base + k * base_stride])
            var delta2 = v2 - mean2
            mean2 = mean2 + (delta2 / Float64(k + 1))
            var t2 = v2 - mean2
            m22 = m22 + (delta2 * t2)
            k = k + 1

        var denom2 = if unbiased and reduce_n > 1: Float64(reduce_n - 1) else: Float64(reduce_n)
        var vv2 = if denom2 > 0.0: m22 / denom2 else: 0.0
        out_vals.append(vv2)
        o = o + 1

    var st2 = compute_row_major_strides(out_shape2)
    var tout = Tensor[Float64](out_vals, out_shape2, st2, 0)

    if keepdims:
        var kd = List[Int]()
        var r = 0
        while r < rank:
            if r == ax: kd.append(1) else: kd.append(shp[r])
            r = r + 1
        return tout.reshape(kd)

    return tout




# -------- reduce_max for Float32 (unchanged, constructor-style where needed) --------
fn reduce_max_f32(x: Tensor[Float32]) -> Float32:
    var n = len(x._data)
    if n == 0:
        return 0.0
    var mv = x._data[0]
    var i = 1
    while i < n:
        var v = x._data[i]
        if v > mv:
            mv = v
        i += 1
    return mv

fn reduce_min_f32(x: Tensor[Float32]) -> Float32:
    var n = len(x._data)
    if n == 0:
        return 0.0
    var mv = x._data[0]
    var i = 1
    while i < n:
        var v = x._data[i]
        if v < mv:
            mv = v
        i += 1
    return mv

fn max(x: Tensor[Float32], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float32]:
    var shp = x._shape.copy()
    var rank = len(shp)

    # Reduce over all elements -> scalar tensor [1]
    if axis is None:
        var mval = reduce_max_f32(x)
        var out = Tensor[Float32]([mval], [1])
        if keepdims:
            # shape of ones with same rank
            var kd = List[Int]()
            var i = 0
            while i < rank:
                kd.append(1)
                i += 1
            return out.reshape(kd)
        return out

    # Reduce along a specific axis
    var ax = normalize_axis(axis.value(), rank)
    var reduce_n = shp[ax]

    # elements to the right of axis (stride within a block)
    var inner = 1
    var i_in = ax + 1
    while i_in < rank:
        inner = inner * shp[i_in]
        i_in += 1

    # output shape after dropping the axis
    var out_shape = shape_drop_axis(shp, ax)
    var outer = numel(out_shape)

    var outv = List[Float32]()
    outv.reserve(outer)

    var base_stride = inner
    var block = reduce_n * inner

    var o = 0
    while o < outer:
        # map flat output index to base position in input
        var base = (o // inner) * block + (o % inner)

        var mv = x._data[base]
        var k = 1
        while k < reduce_n:
            var v = x._data[base + k * base_stride]
            if v > mv:
                mv = v
            k += 1
        outv.append(mv)
        o += 1

    var tout = Tensor[Float32](outv, out_shape)
    if keepdims:
        return tout.reshape(keepdims_shape(shp, ax))
    return tout




@always_inline
fn variance(
    x: Tensor[Int],
    axis: Optional[Int] = None,
    unbiased: Bool = True,
    keepdims: Bool = False
) -> Tensor[Float32]:
    var shp = x._shape.copy()
    var rank = len(shp)

    # ---------- WHOLE TENSOR ----------
    if axis is None:
        var out_list = List[Float32]()
        out_list.reserve(1)

        # Logical element count
        var n_total = 1
        if rank == 0:
            n_total = 1
        else:
            var r0 = 0
            n_total = 1
            while r0 < rank:
                n_total = n_total * shp[r0]
                r0 = r0 + 1

        if n_total == 0:
            out_list.append(0.0)
        else:
            if is_row_major_contiguous(shp, x._strides):
                # Welford (contiguous fast-path)
                var mean_ = 0.0
                var m2 = 0.0
                var i = 0
                while i < n_total:
                    var v = Float32(x._data[i])
                    var delta = v - mean_
                    mean_ = mean_ + (delta / Float32(i + 1))
                    var t = v - mean_
                    m2 = m2 + (delta * t)
                    i = i + 1
                var denom = if unbiased and n_total > 1: Float32(n_total - 1) else: Float32(n_total)
                out_list.append(if denom > 0.0: m2 / denom else: 0.0)
            else:
                # Welford (generic strided)
                var idx = List[Int]()
                idx.reserve(rank)
                var d = 0
                while d < rank:
                    idx.append(0)
                    d = d + 1

                var count = 0
                var mean2 = 0.0
                var m22 = 0.0
                var done = False
                while not done:
                    var off = 0
                    var k = 0
                    while k < rank:
                        off = off + idx[k] * x._strides[k]
                        k = k + 1
                    var v2 = Float32(x._data[off])
                    var delta2 = v2 - mean2
                    mean2 = mean2 + (delta2 / Float64(count + 1))
                    var t2 = v2 - mean2
                    m22 = m22 + (delta2 * t2)
                    count = count + 1

                    var r = rank - 1
                    while r >= 0:
                        idx[r] = idx[r] + 1
                        if idx[r] < shp[r]: break
                        idx[r] = 0
                        if r == 0: done = True; break
                        r = r - 1

                var denom2 = if unbiased and count > 1: Float32(count - 1) else: Float32(count)
                out_list.append(if denom2 > 0.0: m22 / denom2 else: 0.0)

        var out_shape = List[Int]()
        if keepdims:
            var j = 0
            while j < rank:
                out_shape.append(1)
                j = j + 1
        else:
            out_shape.append(1)

        var st = compute_row_major_strides(out_shape)
        return Tensor[Float32](out_list, out_shape, st, 0)

    # ---------- AXIS REDUCTION ----------
    var ax = normalize_axis(axis.value(), rank)
    var reduce_n = if rank == 0: 1 else: shp[ax]
    var out_shape2 = shape_drop_axis(shp, ax)

    if is_row_major_contiguous(shp, x._strides):
        # Pack outer/inner (contiguous)
        var inner = 1
        var i_in = ax + 1
        while i_in < rank:
            inner = inner * shp[i_in]
            i_in = i_in + 1

        var outer = 1
        var i_out = 0
        while i_out < ax:
            outer = outer * shp[i_out]
            i_out = i_out + 1
        outer = outer * inner

        var base_stride = inner
        var block = reduce_n * inner

        var out_vals = List[Float64]()
        out_vals.reserve(outer)

        var o = 0
        while o < outer:
            var base = (o // inner) * block + (o % inner)

            var mean3 = 0.0
            var m23 = 0.0
            var k2 = 0
            while k2 < reduce_n:
                var vv = Float64(x._data[base + k2 * base_stride])
                var delta = vv - mean3
                mean3 = mean3 + (delta / Float64(k2 + 1))
                var t = vv - mean3
                m23 = m23 + (delta * t)
                k2 = k2 + 1
            var denom = if unbiased and reduce_n > 1: Float64(reduce_n - 1) else: Float64(reduce_n)
            out_vals.append(if denom > 0.0: m23 / denom else: 0.0)

            o = o + 1

        var st2 = compute_row_major_strides(out_shape2)
        var tout = Tensor[Float64](out_vals, out_shape2, st2, 0)

        if keepdims:
            var kd = List[Int]()
            var r2 = 0
            while r2 < rank:
                if r2 == ax: kd.append(1) else: kd.append(shp[r2])
                r2 = r2 + 1
            return tout.reshape(kd)
        return tout
    else:
        # Generic strided
        var outer_shape = List[Int]()
        var outer_strides = List[Int]()
        var i = 0
        while i < rank:
            if i != ax:
                outer_shape.append(shp[i])
                outer_strides.append(x._strides[i])
            i = i + 1
        var stride_red = x._strides[ax]

        var out_n = 1
        var oi = 0
        while oi < len(outer_shape):
            out_n = out_n * outer_shape[oi]
            oi = oi + 1

        var out_vals_g = List[Float64]()
        out_vals_g.reserve(out_n)

        var idxo = List[Int]()
        idxo.reserve(len(outer_shape))
        var u = 0
        while u < len(outer_shape):
            idxo.append(0)
            u = u + 1

        var done2 = False
        while True:
            var base2 = 0
            var d3 = 0
            while d3 < len(outer_shape):
                base2 = base2 + idxo[d3] * outer_strides[d3]
                d3 = d3 + 1

            var mean4 = 0.0
            var m24 = 0.0
            var cnt = 0
            while cnt < reduce_n:
                var vv2 = Float64(x._data[base2 + cnt * stride_red])
                var dl = vv2 - mean4
                mean4 = mean4 + (dl / Float64(cnt + 1))
                var tt = vv2 - mean4
                m24 = m24 + (dl * tt)
                cnt = cnt + 1
            var denom4 = if unbiased and reduce_n > 1: Float64(reduce_n - 1) else: Float64(reduce_n)
            out_vals_g.append(if denom4 > 0.0: m24 / denom4 else: 0.0)

            if len(outer_shape) == 0: break
            var p = len(outer_shape) - 1
            var carry = True
            while p >= 0 and carry:
                idxo[p] = idxo[p] + 1
                if idxo[p] < outer_shape[p]:
                    carry = False
                else:
                    idxo[p] = 0
                    if p == 0:
                        done2 = True
                if p == 0: break
                p = p - 1
            if done2: break

        var stg = compute_row_major_strides(out_shape2)
        var toutg = Tensor[Float64](out_vals_g, out_shape2, stg, 0)

        if keepdims:
            var kd2 = List[Int]()
            var r3 = 0
            while r3 < rank:
                if r3 == ax: kd2.append(1) else: kd2.append(shp[r3])
                r3 = r3 + 1
            return toutg.reshape(kd2)
        return toutg


fn min(x: Tensor[Float32], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[Float32]:
    var shp = x._shape.copy()
    var rank = len(shp)

    # کاهش روی همهٔ عناصر → اسکالر [1]
    if axis is None:
        var mval = reduce_min_f64(x)
        var out = Tensor[Float32]([mval], [1])
        if keepdims:
            var kd = List[Int]()
            var i = 0
            while i < rank:
                kd.append(1)
                i += 1
            return out.reshape(kd)
        return out

    # کاهش روی یک محور مشخص
    var ax = normalize_axis(axis.value(), rank)
    var reduce_n = shp[ax]

    # تعداد عناصر سمت راست محور (گام درون بلاک)
    var inner = 1
    var i_in = ax + 1
    while i_in < rank:
        inner = inner * shp[i_in]
        i_in += 1

    # شکل خروجی بعد از حذف محور
    var out_shape = shape_drop_axis(shp, ax)
    var outer = numel(out_shape)

    var outv = List[Float64]()
    outv.reserve(outer)

    var base_stride = inner
    var block = reduce_n * inner

    var o = 0
    while o < outer:
        # نگاشت ایندکس تخت خروجی به موقعیت پایه در ورودی
        var base = (o // inner) * block + (o % inner)

        var mv = x._data[base]
        var k = 1
        while k < reduce_n:
            var v = x._data[base + k * base_stride]
            if v < mv:
                mv = v
            k += 1
        outv.append(mv)
        o += 1

    var tout = Tensor[Float64](outv, out_shape)
    if keepdims:
        return tout.reshape(keepdims_shape(shp, ax))
    return tout


fn argmax(x: Tensor[Float32], axis: Optional[Int] = None) -> Tensor[Int]:
    var shp = x._shape.copy()
    var rank = len(shp)

    # حالت بدون محور: اندیسِ بیشینه روی کل آرایه‌ی تخت‌شده
    if axis is None:
        var n = len(x._data)
        if n == 0:
            # قرارداد: اگر تهی بود، 0 برمی‌گردانیم با شکل [1]
            return scalar_int(0)                # اسکالر rank-0

        var best = 0
        var bestv = x._data[0]
        var i = 1
        while i < n:
            var v = x._data[i]
            if v > bestv:
                bestv = v
                best = i
            i += 1
        return scalar_int(best)

    # حالت با محور مشخص
    var ax = normalize_axis(axis.value(), rank)
    var out_shape = shape_drop_axis(shp, ax)
    var reduce_n = shp[ax]

    # تعداد عناصر سمت راست محور
    var inner = 1
    var i_in = ax + 1
    while i_in < rank:
        inner = inner * shp[i_in]
        i_in += 1

    var outer = numel(out_shape)
    var out_idx = List[Int]()
    out_idx.reserve(outer)

    var base_stride = inner
    var block = reduce_n * inner

    var o = 0
    while o < outer:
        var base = (o // inner) * block + (o % inner)

        var besti = 0
        var bestv2 = x._data[base]
        var k = 1
        while k < reduce_n:
            var v = x._data[base + k * base_stride]
            if v > bestv2:
                bestv2 = v
                besti = k
            k += 1

        out_idx.append(besti)
        o += 1

    return Tensor[Int](shape=out_shape, flat=out_idx)


fn argmin(x: Tensor[Float32], axis: Optional[Int] = None) -> Tensor[Int]:
    var shp = x._shape.copy()
    var rank = len(shp)

    # حالت بدون محور: کمینه روی کل آرایه‌ی تخت‌شده
    if axis is None:
        var n = len(x._data)
        if n == 0:
            # قرارداد: اگر تهی بود، 0 برمی‌گردانیم با شکل [1]
            return scalar_int(0)

        var best = 0
        var bestv = x._data[0]
        var i = 1
        while i < n:
            var v = x._data[i]
            if v < bestv:
                bestv = v
                best = i
            i += 1
        return scalar_int(best)

    # حالت با محور مشخص
    var ax = normalize_axis(axis.value(), rank)
    var out_shape = shape_drop_axis(shp, ax)
    var reduce_n = shp[ax]

    # تعداد عناصر سمت راست محور
    var inner = 1
    var i_in = ax + 1
    while i_in < rank:
        inner = inner * shp[i_in]
        i_in += 1

    var outer = numel(out_shape)
    var out_idx = List[Int]()
    out_idx.reserve(outer)

    var base_stride = inner
    var block = reduce_n * inner

    var o = 0
    while o < outer:
        var base = (o // inner) * block + (o % inner)

        var besti = 0
        var bestv2 = x._data[base]
        var k = 1
        while k < reduce_n:
            var v = x._data[base + k * base_stride]
            if v < bestv2:
                bestv2 = v
                besti = k
            k += 1

        out_idx.append(besti)
        o += 1

    return Tensor[Int](shape=out_shape, flat=out_idx)


@always_inline
fn _keepdims_shape_all(shp: List[Int]) -> List[Int]:
    var out = List[Int]()
    var r = len(shp)
    if r == 0:
        out.append(1)
        return out.copy()
    var i = 0
    while i < r:
        out.append(1)
        i += 1
    return out.copy()

@always_inline
fn _keepdims_shape(shp: List[Int], ax: Int) -> List[Int]:
    var r = len(shp)
    var out = List[Int]()
    if r == 0:
        out.append(1)
        return out.copy()
    var i = 0
    while i < r:
        out.append(1 if i == ax else shp[i])
        i += 1
    return out.copy()



@always_inline
fn math_min(a: Int, b: Int) -> Int:
    return a if a <= b else b

@always_inline
fn math_max(a: Int, b: Int) -> Int:
    return a if a >= b else b




# -----------------------------------------------------------------------------
# f(x) = x^2 + 2x  (elementwise)
# -----------------------------------------------------------------------------
@always_inline
fn f_vec(x: Tensor[Float64]) -> Tensor[Float64]:
    var x2 = x.mul(x)
    return x2.add(x.mul_scalar(2.0))

# -----------------------------------------------------------------------------
# Analytic Jacobian: diag(2*x + 2)
# -----------------------------------------------------------------------------
@always_inline
fn analytic_jacobian(x: Tensor[Float64]) -> Tensor[Float64]:
    var x1d = x.flatten()
    var n = x1d.numel()

    var diag_vals = x1d.mul_scalar(2.0).add_scalar(2.0)  # (n,)
    var jbuf = List[Float64]()
    jbuf.reserve(n * n)

    var k = 0
    var total = n * n
    while k < total:
        jbuf.append(0.0)
        k += 1

    var dv = diag_vals._data.copy()
    var i = 0
    while i < n:
        jbuf[i * n + i] = dv[i]
        i += 1

    return from_list_float64(jbuf).reshape([n, n])

# -----------------------------------------------------------------------------
# Numeric Jacobian via central differences (central diff per column)
# J_ij ≈ (f_i(x + eps * e_j) - f_i(x - eps * e_j)) / (2 * eps)
# -----------------------------------------------------------------------------
fn numeric_jacobian(x: Tensor[Float64], eps: Float64 = 1e-6) -> Tensor[Float64]:
    var x1d = x.flatten()
    var n = x1d.numel()

    var jbuf = List[Float64]()
    jbuf.reserve(n * n)
    var total = n * n
    var k = 0
    while k < total: jbuf.append(0.0); k += 1

    var denom = 2.0 * eps
    var xw = x1d.copy()

    var j = 0
    while j < n:
        var xj0 = xw._data[j]              # read
        xw._data[j] = xj0 + eps            # +eps
        var y_plus = f_vec(xw)
        xw._data[j] = xj0 - eps            # -eps
        var y_minus = f_vec(xw)
        xw._data[j] = xj0                  # restore

        var g = y_plus.sub(y_minus).div_scalar(denom)
        var gd = g._data.copy()

        var i = 0
        while i < n:
            jbuf[i * n + j] = gd[i]
            i += 1
        j += 1
    return from_list_float64(jbuf).reshape([n, n])


# -----------------------------------------------------------------------------
# Complex128 element type (pure Mojo, no FFI)
# -----------------------------------------------------------------------------

struct Complex128(ImplicitlyCopyable, Copyable, Movable):
    var re: Float64
    var im: Float64

    fn __init__(out self, re: Float64, im: Float64):
        self.re = re
        self.im = im

    # Make it explicitly Copyable (required by Tensor[T] constraints)
    fn __copyinit__(out self, other: Complex128):
        self.re = other.re
        self.im = other.im

    @always_inline
    fn real(self) -> Float64:
        return self.re

    @always_inline
    fn imag(self) -> Float64:
        return self.im

# (Optional) Tiny sqrt for Float64 (Newton's method). Used by hypot below.
@always_inline
fn _sqrt_f64(x: Float64) -> Float64:
    if x <= 0.0:
        return 0.0
    var y = x
    var half = 0.5 * x
    # ~6 iterations are enough for 1e-12-ish relative error.
    var i = 0
    while i < 6:
        y = 0.5 * (y + (x / y))
        i += 1
    return y

# Stable hypot without overflow/underflow: sqrt(a*a + b*b)
@always_inline
fn _hypot_f64(a: Float64, b: Float64) -> Float64:
    var aa =a
    if a < 0.0: aa =-a
    var bb =b
    if b < 0.0: bb =-b
    if aa < bb:
        var t = aa; aa = bb; bb = t
    if aa == 0.0:
        return 0.0
    var r = bb / aa
    return _sqrt_f64(aa * aa * (1.0 + r * r))

@always_inline
fn _shapes_equal(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    var i = 0
    var n = len(a)
    while i < n:
        if a[i] != b[i]:
            return False
        i += 1
    return True

# -----------------------------------------------------------------------------
# tensor.complex(zr, zi) -> Tensor[Complex128]
# No 'assert' usage; manual guards. On mismatch returns an empty tensor [0]-shape.
# -----------------------------------------------------------------------------
@always_inline
fn complex(zr: Tensor[Float64], zi: Tensor[Float64]) -> Tensor[Complex128]:
    if not _shapes_equal(zr._shape, zi._shape):
        var empty_data = List[Complex128]()
        var empty_shape = List[Int](); empty_shape.append(0)
        return Tensor[Complex128](empty_data, empty_shape)

    var n = len(zr._data)
    var out = List[Complex128]()
    out.reserve(n)

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(Complex128(zr._data[i    ], zi._data[i    ]))
        out.append(Complex128(zr._data[i + 1], zi._data[i + 1]))
        out.append(Complex128(zr._data[i + 2], zi._data[i + 2]))
        out.append(Complex128(zr._data[i + 3], zi._data[i + 3]))
        out.append(Complex128(zr._data[i + 4], zi._data[i + 4]))
        out.append(Complex128(zr._data[i + 5], zi._data[i + 5]))
        out.append(Complex128(zr._data[i + 6], zi._data[i + 6]))
        out.append(Complex128(zr._data[i + 7], zi._data[i + 7]))
        i += 8
    while i < n:
        out.append(Complex128(zr._data[i], zi._data[i]))
        i += 1

    return Tensor[Complex128](out, zr._shape)

# -----------------------------------------------------------------------------
# Free functions on Tensor[Complex128]
# -----------------------------------------------------------------------------
@always_inline
fn complex_real(z: Tensor[Complex128]) -> Tensor[Float64]:
    var n = len(z._data)
    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(z._data[i    ].re)
        out.append(z._data[i + 1].re)
        out.append(z._data[i + 2].re)
        out.append(z._data[i + 3].re)
        out.append(z._data[i + 4].re)
        out.append(z._data[i + 5].re)
        out.append(z._data[i + 6].re)
        out.append(z._data[i + 7].re)
        i += 8
    while i < n:
        out.append(z._data[i].re)
        i += 1

    return Tensor[Float64](out, z._shape)

@always_inline
fn complex_imag(z: Tensor[Complex128]) -> Tensor[Float64]:
    var n = len(z._data)
    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(z._data[i    ].im)
        out.append(z._data[i + 1].im)
        out.append(z._data[i + 2].im)
        out.append(z._data[i + 3].im)
        out.append(z._data[i + 4].im)
        out.append(z._data[i + 5].im)
        out.append(z._data[i + 6].im)
        out.append(z._data[i + 7].im)
        i += 8
    while i < n:
        out.append(z._data[i].im)
        i += 1

    return Tensor[Float64](out, z._shape)

@always_inline
fn complex_abs(z: Tensor[Complex128]) -> Tensor[Float64]:
    var n = len(z._data)
    var out = List[Float64]()
    out.reserve(n)

    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        var c0 = z._data[i    ]; out.append(_hypot_f64(c0.re, c0.im))
        var c1 = z._data[i + 1]; out.append(_hypot_f64(c1.re, c1.im))
        var c2 = z._data[i + 2]; out.append(_hypot_f64(c2.re, c2.im))
        var c3 = z._data[i + 3]; out.append(_hypot_f64(c3.re, c3.im))
        var c4 = z._data[i + 4]; out.append(_hypot_f64(c4.re, c4.im))
        var c5 = z._data[i + 5]; out.append(_hypot_f64(c5.re, c5.im))
        var c6 = z._data[i + 6]; out.append(_hypot_f64(c6.re, c6.im))
        var c7 = z._data[i + 7]; out.append(_hypot_f64(c7.re, c7.im))
        i += 8
    while i < n:
        var c = z._data[i]
        out.append(_hypot_f64(c.re, c.im))
        i += 1

    return Tensor[Float64](out, z._shape)





# ------------------------------
# Method on Tensor[Int]
# ------------------------------
@always_inline
fn one_hot_core_indices(indices: List[Int], shp: List[Int], depth: Int) -> Tensor[Int]:
    # Build output shape = shp + [depth]
    var out_shape = shp.copy()#copy_ints(shp)
    out_shape.append(depth)

    # Compute sizes
    var n = 1
    var i = 0
    while i < len(shp):
        n = n * shp[i]
        i += 1

    # Prepare flat output buffer of size n * depth filled with zeros
    var out_n = n * depth
    var out_data = List[Int]()
    out_data.reserve(out_n)
    var z = 0
    var k = 0
    var lim = (out_n // 8) * 8
    while k < lim:
        out_data.append(z); out_data.append(z); out_data.append(z); out_data.append(z)
        out_data.append(z); out_data.append(z); out_data.append(z); out_data.append(z)
        k += 8
    while k < out_n:
        out_data.append(z)
        k += 1

    # Scatter 1s where 0 <= idx < depth
    i = 0
    while i < n and i < len(indices):
        var cls = indices[i]
        if cls >= 0 and cls < depth:
            out_data[i * depth + cls] = 1
        i += 1

    var out_strides = compute_row_major_strides(out_shape)
    return Tensor[Int](out_data, out_shape, out_strides, 0)



# ---- Local helpers (no-throw, no-assert) ------------------------------------
@always_inline
fn _is_rank2(shp: List[Int]) -> Bool:
    var r = len(shp)
    return r == 2

# Normalize Int index: [N,1] -> [N] via reshape; [N] stays as-is; otherwise copy.
@always_inline
fn _squeeze_if_2d1(x_shape: List[Int], y: Tensor[Int]) -> Tensor[Int]:
    var r = len(y._shape)
    if r == 2:
        var n = y._shape[0]
        var m = y._shape[1]
        if m == 1:
            return y.reshape([n])
        if len(x_shape) > 0 and n == x_shape[0] and m == 1:
            return y.reshape([n])
        return y.copy()
    if r == 1:
        return y.copy()
    return y.copy()

@always_inline
fn _squeeze_if_2d1_f32(x_shape: List[Int], y: Tensor[Float32]) -> Tensor[Float32]:
    var r = len(y._shape)
    if r == 2:
        var n = y._shape[0]
        var m = y._shape[1]
        if m == 1:
            return y.reshape([n])
        if len(x_shape) > 0 and n == x_shape[0] and m == 1:
            return y.reshape([n])
        return y.copy()
    if r == 1:
        return y.copy()
    return y.copy()

@always_inline
fn _squeeze_if_2d1_f64(x_shape: List[Int], y: Tensor[Float64]) -> Tensor[Float64]:
    var r = len(y._shape)
    if r == 2:
        var n = y._shape[0]
        var m = y._shape[1]
        if m == 1:
            return y.reshape([n])
        if len(x_shape) > 0 and n == x_shape[0] and m == 1:
            return y.reshape([n])
        return y.copy()
    if r == 1:
        return y.copy()
    return y.copy()

@always_inline
fn _clamp_index(j: Int, c: Int) -> Int:
    if c <= 0:
        return 0
    var jj = j
    if jj < 0: jj = 0
    if jj >= c: jj = c - 1
    return jj


@always_inline
fn _same_shape(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b): return False
    var k = 0
    while k < len(a):
        if a[k] != b[k]: return False
        k += 1
    return True

# Squeeze helpers with out-params only (no tuple returns)

@always_inline
fn _squeeze_trailing1_like_outer_Int(
    outer: List[Int],
    y: Tensor[Int],
    mut out_y: Tensor[Int],
    mut ok: Bool
) -> None:
    if _same_shape(y._shape, outer):
        out_y = y.copy()
        ok = True
        return
    var r_y = len(y._shape)
    if r_y == len(outer) + 1 and y._shape[r_y - 1] == 1:
        out_y = y.reshape(outer.copy())
        ok = True
        return
    out_y = y.copy()
    ok = False

@always_inline
fn _squeeze_trailing1_like_outer_F32(
    outer: List[Int],
    y: Tensor[Float32],
    mut out_y: Tensor[Float32],
    mut ok: Bool
) -> None:
    if _same_shape(y._shape, outer):
        out_y = y.copy()
        ok = True
        return
    var r_y = len(y._shape)
    if r_y == len(outer) + 1 and y._shape[r_y - 1] == 1:
        out_y = y.reshape(outer.copy())
        ok = True
        return
    out_y = y.copy()
    ok = False

@always_inline
fn _squeeze_trailing1_like_outer_F64(
    outer: List[Int],
    y: Tensor[Float64],
    mut out_y: Tensor[Float64],
    mut ok: Bool
) -> None:
    if _same_shape(y._shape, outer):
        out_y = y.copy()
        ok = True
        return
    var r_y = len(y._shape)
    if r_y == len(outer) + 1 and y._shape[r_y - 1] == 1:
        out_y = y.reshape(outer.copy())
        ok = True
        return
    out_y = y.copy()
    ok = False

# ------------------------------------------------------------
# dim = 1 (target axis = shape[1]; outer = dim0 + dims 2..r-1)
# ------------------------------------------------------------

@always_inline
fn scatter_add_dim1_int(x: Tensor[Int], index: Tensor[Int], src: Tensor[Int]) -> Tensor[Int]:
    var r = len(x._shape)
    if r < 2: return x.copy()

    var outer_shape = List[Int]()
    outer_shape.append(x._shape[0])
    var k = 2
    while k < r:
        outer_shape.append(x._shape[k]); k += 1

    var idxT = index.copy()
    var ok1 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, index, idxT, ok1)
    if not ok1: return x.copy()

    var srcT = src.copy()
    var ok2 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, src, srcT, ok2)
    if not ok2: return x.copy()

    var out = x.copy()
    var off = out._offset
    var s = out._strides.copy()
    var N1 = out._shape[1]

    var tail_rank = r - 2
    if tail_rank == 0:
        var N0 = out._shape[0]
        var i0 = 0
        while i0 < N0:
            var j = _clamp_index(idxT._data[i0], N1)
            var pos0 = off + i0 * s[0] + j * s[1]
            out._data[pos0] = out._data[pos0] + srcT._data[i0]
            i0 += 1
        return out.copy()

    var tail_shape = List[Int]()
    k = 2
    while k < r:
        tail_shape.append(out._shape[k]); k += 1

    var N0 = out._shape[0]
    var tail_count = 1
    k = 0
    while k < tail_rank:
        tail_count = tail_count * tail_shape[k]; k += 1

    var mul = List[Int]()
    mul.reserve(tail_rank)
    k = 0
    while k < tail_rank: mul.append(1); k += 1
    if tail_rank > 0:
        var acc = 1
        var i = tail_rank - 1
        while True:
            mul[i] = acc; acc = acc * tail_shape[i]
            if i == 0: break
            i -= 1

    var outer_count = N0 * tail_count
    var lin = 0
    while lin < outer_count:
        var a = 0
        var rem = lin
        if tail_count > 0:
            a = rem // tail_count
            rem = rem % tail_count

        var pos_base = off + a * s[0]
        k = 0
        while k < tail_rank:
            var idx_tail = (rem // mul[k]) % tail_shape[k]
            pos_base = pos_base + idx_tail * s[k + 2]
            k += 1

        var j = _clamp_index(idxT._data[lin], N1)
        var pos = pos_base + j * s[1]
        out._data[pos] = out._data[pos] + srcT._data[lin]
        lin += 1
    return out.copy()

@always_inline
fn scatter_add_dim1_f32(x: Tensor[Float32], index: Tensor[Int], src: Tensor[Float32]) -> Tensor[Float32]:
    var r = len(x._shape)
    if r < 2: return x.copy()

    var outer_shape = List[Int]()
    outer_shape.append(x._shape[0])
    var k = 2
    while k < r:
        outer_shape.append(x._shape[k]); k += 1

    var idxT = index.copy()
    var ok1 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, index, idxT, ok1)
    if not ok1: return x.copy()

    var srcT = src.copy()
    var ok2 = False
    _squeeze_trailing1_like_outer_F32(outer_shape, src, srcT, ok2)
    if not ok2: return x.copy()

    var out = x.copy()
    var off = out._offset
    var s = out._strides.copy()
    var N1 = out._shape[1]

    var tail_rank = r - 2
    if tail_rank == 0:
        var N0 = out._shape[0]
        var i0 = 0
        while i0 < N0:
            var j = _clamp_index(idxT._data[i0], N1)
            var pos0 = off + i0 * s[0] + j * s[1]
            out._data[pos0] = out._data[pos0] + srcT._data[i0]
            i0 += 1
        return out.copy()

    var tail_shape = List[Int]()
    k = 2
    while k < r:
        tail_shape.append(out._shape[k]); k += 1

    var N0 = out._shape[0]
    var tail_count = 1
    k = 0
    while k < tail_rank:
        tail_count = tail_count * tail_shape[k]; k += 1

    var mul = List[Int]()
    mul.reserve(tail_rank)
    k = 0
    while k < tail_rank: mul.append(1); k += 1
    if tail_rank > 0:
        var acc = 1
        var i = tail_rank - 1
        while True:
            mul[i] = acc; acc = acc * tail_shape[i]
            if i == 0: break
            i -= 1

    var outer_count = N0 * tail_count
    var lin = 0
    while lin < outer_count:
        var a = 0
        var rem = lin
        if tail_count > 0:
            a = rem // tail_count
            rem = rem % tail_count

        var pos_base = off + a * s[0]
        k = 0
        while k < tail_rank:
            var idx_tail = (rem // mul[k]) % tail_shape[k]
            pos_base = pos_base + idx_tail * s[k + 2]
            k += 1

        var j = _clamp_index(idxT._data[lin], N1)
        var pos = pos_base + j * s[1]
        out._data[pos] = out._data[pos] + srcT._data[lin]
        lin += 1
    return out.copy()

@always_inline
fn scatter_add_dim1_f64(x: Tensor[Float64], index: Tensor[Int], src: Tensor[Float64]) -> Tensor[Float64]:
    var r = len(x._shape)
    if r < 2: return x.copy()

    var outer_shape = List[Int]()
    outer_shape.append(x._shape[0])
    var k = 2
    while k < r:
        outer_shape.append(x._shape[k]); k += 1

    var idxT = index.copy()
    var ok1 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, index, idxT, ok1)
    if not ok1: return x.copy()

    var srcT = src.copy()
    var ok2 = False
    _squeeze_trailing1_like_outer_F64(outer_shape, src, srcT, ok2)
    if not ok2: return x.copy()

    var out = x.copy()
    var off = out._offset
    var s = out._strides.copy()
    var N1 = out._shape[1]

    var tail_rank = r - 2
    if tail_rank == 0:
        var N0 = out._shape[0]
        var i0 = 0
        while i0 < N0:
            var j = _clamp_index(idxT._data[i0], N1)
            var pos0 = off + i0 * s[0] + j * s[1]
            out._data[pos0] = out._data[pos0] + srcT._data[i0]
            i0 += 1
        return out.copy()

    var tail_shape = List[Int]()
    k = 2
    while k < r:
        tail_shape.append(out._shape[k]); k += 1

    var N0 = out._shape[0]
    var tail_count = 1
    k = 0
    while k < tail_rank:
        tail_count = tail_count * tail_shape[k]; k += 1

    var mul = List[Int]()
    mul.reserve(tail_rank)
    k = 0
    while k < tail_rank: mul.append(1); k += 1
    if tail_rank > 0:
        var acc = 1
        var i = tail_rank - 1
        while True:
            mul[i] = acc; acc = acc * tail_shape[i]
            if i == 0: break
            i -= 1

    var outer_count = N0 * tail_count
    var lin = 0
    while lin < outer_count:
        var a = 0
        var rem = lin
        if tail_count > 0:
            a = rem // tail_count
            rem = rem % tail_count

        var pos_base = off + a * s[0]
        k = 0
        while k < tail_rank:
            var idx_tail = (rem // mul[k]) % tail_shape[k]
            pos_base = pos_base + idx_tail * s[k + 2]
            k += 1

        var j = _clamp_index(idxT._data[lin], N1)
        var pos = pos_base + j * s[1]
        out._data[pos] = out._data[pos] + srcT._data[lin]
        lin += 1
    return out.copy()

# ------------------------------------------------------------
# dim = 0 (target axis = shape[0]; outer = dims 1..r-1)
# ------------------------------------------------------------

@always_inline
fn scatter_add_dim0_int(x: Tensor[Int], index: Tensor[Int], src: Tensor[Int]) -> Tensor[Int]:
    var r = len(x._shape)
    if r < 2: return x.copy()

    var outer_shape = List[Int]()
    var k = 1
    while k < r:
        outer_shape.append(x._shape[k]); k += 1

    var idxT = index.copy()
    var ok1 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, index, idxT, ok1)
    if not ok1: return x.copy()

    var srcT = src.copy()
    var ok2 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, src, srcT, ok2)
    if not ok2: return x.copy()

    var out = x.copy()
    var off = out._offset
    var s = out._strides.copy()
    var N0 = out._shape[0]

    var outer_rank = len(outer_shape)
    var outer_count = 1
    k = 0
    while k < outer_rank:
        outer_count = outer_count * outer_shape[k]; k += 1

    var mul = List[Int]()
    mul.reserve(outer_rank)
    k = 0
    while k < outer_rank: mul.append(1); k += 1
    if outer_rank > 0:
        var acc = 1
        var i = outer_rank - 1
        while True:
            mul[i] = acc; acc = acc * outer_shape[i]
            if i == 0: break
            i -= 1

    var coords = List[Int]()
    coords.reserve(outer_rank)
    k = 0
    while k < outer_rank: coords.append(0); k += 1

    var lin = 0
    while lin < outer_count:
        var t = lin
        k = 0
        while k < outer_rank:
            var m = mul[k]
            coords[k] = (t // m) % outer_shape[k]
            k += 1

        var j = _clamp_index(idxT._data[lin], N0)

        var pos = off + j * s[0]
        k = 0
        while k < outer_rank:
            pos = pos + coords[k] * s[k + 1]
            k += 1

        out._data[pos] = out._data[pos] + srcT._data[lin]
        lin += 1
    return out.copy()

@always_inline
fn scatter_add_dim0_f32(x: Tensor[Float32], index: Tensor[Int], src: Tensor[Float32]) -> Tensor[Float32]:
    var r = len(x._shape)
    if r < 2: return x.copy()

    var outer_shape = List[Int]()
    var k = 1
    while k < r:
        outer_shape.append(x._shape[k]); k += 1

    var idxT = index.copy()
    var ok1 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, index, idxT, ok1)
    if not ok1: return x.copy()

    var srcT = src.copy()
    var ok2 = False
    _squeeze_trailing1_like_outer_F32(outer_shape, src, srcT, ok2)
    if not ok2: return x.copy()

    var out = x.copy()
    var off = out._offset
    var s = out._strides.copy()
    var N0 = out._shape[0]

    var outer_rank = len(outer_shape)
    var outer_count = 1
    k = 0
    while k < outer_rank:
        outer_count = outer_count * outer_shape[k]; k += 1

    var mul = List[Int]()
    mul.reserve(outer_rank)
    k = 0
    while k < outer_rank: mul.append(1); k += 1
    if outer_rank > 0:
        var acc = 1
        var i = outer_rank - 1
        while True:
            mul[i] = acc; acc = acc * outer_shape[i]
            if i == 0: break
            i -= 1

    var coords = List[Int]()
    coords.reserve(outer_rank)
    k = 0
    while k < outer_rank: coords.append(0); k += 1

    var lin = 0
    while lin < outer_count:
        var t = lin
        k = 0
        while k < outer_rank:
            var m = mul[k]
            coords[k] = (t // m) % outer_shape[k]
            k += 1

        var j = _clamp_index(idxT._data[lin], N0)

        var pos = off + j * s[0]
        k = 0
        while k < outer_rank:
            pos = pos + coords[k] * s[k + 1]
            k += 1

        out._data[pos] = out._data[pos] + srcT._data[lin]
        lin += 1
    return out.copy()

@always_inline
fn scatter_add_dim0_f64(x: Tensor[Float64], index: Tensor[Int], src: Tensor[Float64]) -> Tensor[Float64]:
    var r = len(x._shape)
    if r < 2: return x.copy()

    var outer_shape = List[Int]()
    var k = 1
    while k < r:
        outer_shape.append(x._shape[k]); k += 1

    var idxT = index.copy()
    var ok1 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, index, idxT, ok1)
    if not ok1: return x.copy()

    var srcT = src.copy()
    var ok2 = False
    _squeeze_trailing1_like_outer_F64(outer_shape, src, srcT, ok2)
    if not ok2: return x.copy()

    var out = x.copy()
    var off = out._offset
    var s = out._strides.copy()
    var N0 = out._shape[0]

    var outer_rank = len(outer_shape)
    var outer_count = 1
    k = 0
    while k < outer_rank:
        outer_count = outer_count * outer_shape[k]; k += 1

    var mul = List[Int]()
    mul.reserve(outer_rank)
    k = 0
    while k < outer_rank: mul.append(1); k += 1
    if outer_rank > 0:
        var acc = 1
        var i = outer_rank - 1
        while True:
            mul[i] = acc; acc = acc * outer_shape[i]
            if i == 0: break
            i -= 1

    var coords = List[Int]()
    coords.reserve(outer_rank)
    k = 0
    while k < outer_rank: coords.append(0); k += 1

    var lin = 0
    while lin < outer_count:
        var t = lin
        k = 0
        while k < outer_rank:
            var m = mul[k]
            coords[k] = (t // m) % outer_shape[k]
            k += 1

        var j = _clamp_index(idxT._data[lin], N0)

        var pos = off + j * s[0]
        k = 0
        while k < outer_rank:
            pos = pos + coords[k] * s[k + 1]
            k += 1

        out._data[pos] = out._data[pos] + srcT._data[lin]
        lin += 1
    return out.copy()

# ------------------------------------------------------------
# dim = 3 (target axis = shape[3]; outer = dims 0,1,2)
# ------------------------------------------------------------

@always_inline
fn scatter_add_dim3_int(x: Tensor[Int], index: Tensor[Int], src: Tensor[Int]) -> Tensor[Int]:
    var r = len(x._shape)
    if r < 4: return x.copy()

    var outer_shape = List[Int]()
    outer_shape.append(x._shape[0]); outer_shape.append(x._shape[1]); outer_shape.append(x._shape[2])

    var idxT = index.copy()
    var ok1 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, index, idxT, ok1)
    if not ok1: return x.copy()

    var srcT = src.copy()
    var ok2 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, src, srcT, ok2)
    if not ok2: return x.copy()

    var out = x.copy()
    var off = out._offset
    var s = out._strides.copy()
    var Ndim = out._shape[3]

    var aN = outer_shape[0]; var bN = outer_shape[1]; var cN = outer_shape[2]
    var mul0 = bN * cN
    var mul1 = cN
    var outer_count = aN * bN * cN

    var lin = 0
    while lin < outer_count:
        var a = lin // mul0
        var b = (lin // mul1) % bN
        var c = lin % cN

        var j = _clamp_index(idxT._data[lin], Ndim)
        var pos = off + a * s[0] + b * s[1] + c * s[2] + j * s[3]
        out._data[pos] = out._data[pos] + srcT._data[lin]
        lin += 1
    return out.copy()

@always_inline
fn scatter_add_dim3_f32(x: Tensor[Float32], index: Tensor[Int], src: Tensor[Float32]) -> Tensor[Float32]:
    var r = len(x._shape)
    if r < 4: return x.copy()

    var outer_shape = List[Int]()
    outer_shape.append(x._shape[0]); outer_shape.append(x._shape[1]); outer_shape.append(x._shape[2])

    var idxT = index.copy()
    var ok1 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, index, idxT, ok1)
    if not ok1: return x.copy()

    var srcT = src.copy()
    var ok2 = False
    _squeeze_trailing1_like_outer_F32(outer_shape, src, srcT, ok2)
    if not ok2: return x.copy()

    var out = x.copy()
    var off = out._offset
    var s = out._strides.copy()
    var Ndim = out._shape[3]

    var aN = outer_shape[0]; var bN = outer_shape[1]; var cN = outer_shape[2]
    var mul0 = bN * cN
    var mul1 = cN
    var outer_count = aN * bN * cN

    var lin = 0
    while lin < outer_count:
        var a = lin // mul0
        var b = (lin // mul1) % bN
        var c = lin % cN

        var j = _clamp_index(idxT._data[lin], Ndim)
        var pos = off + a * s[0] + b * s[1] + c * s[2] + j * s[3]
        out._data[pos] = out._data[pos] + srcT._data[lin]
        lin += 1
    return out.copy()

@always_inline
fn scatter_add_dim3_f64(x: Tensor[Float64], index: Tensor[Int], src: Tensor[Float64]) -> Tensor[Float64]:
    var r = len(x._shape)
    if r < 4: return x.copy()

    var outer_shape = List[Int]()
    outer_shape.append(x._shape[0]); outer_shape.append(x._shape[1]); outer_shape.append(x._shape[2])

    var idxT = index.copy()
    var ok1 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, index, idxT, ok1)
    if not ok1: return x.copy()

    var srcT = src.copy()
    var ok2 = False
    _squeeze_trailing1_like_outer_F64(outer_shape, src, srcT, ok2)
    if not ok2: return x.copy()

    var out = x.copy()
    var off = out._offset
    var s = out._strides.copy()
    var Ndim = out._shape[3]

    var aN = outer_shape[0]; var bN = outer_shape[1]; var cN = outer_shape[2]
    var mul0 = bN * cN
    var mul1 = cN
    var outer_count = aN * bN * cN

    var lin = 0
    while lin < outer_count:
        var a = lin // mul0
        var b = (lin // mul1) % bN
        var c = lin % cN

        var j = _clamp_index(idxT._data[lin], Ndim)
        var pos = off + a * s[0] + b * s[1] + c * s[2] + j * s[3]
        out._data[pos] = out._data[pos] + srcT._data[lin]
        lin += 1
    return out.copy()

# ------------------------------------------------------------
# dim = 4 (target axis = shape[4]; outer = dims 0,1,2,3)
# ------------------------------------------------------------

@always_inline
fn scatter_add_dim4_int(x: Tensor[Int], index: Tensor[Int], src: Tensor[Int]) -> Tensor[Int]:
    var r = len(x._shape)
    if r < 5: return x.copy()

    var outer_shape = List[Int]()
    outer_shape.append(x._shape[0]); outer_shape.append(x._shape[1]); outer_shape.append(x._shape[2]); outer_shape.append(x._shape[3])

    var idxT = index.copy()
    var ok1 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, index, idxT, ok1)
    if not ok1: return x.copy()

    var srcT = src.copy()
    var ok2 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, src, srcT, ok2)
    if not ok2: return x.copy()

    var out = x.copy()
    var off = out._offset
    var s = out._strides.copy()
    var Ndim = out._shape[4]

    var aN = outer_shape[0]; var bN = outer_shape[1]; var cN = outer_shape[2]; var dN = outer_shape[3]
    var mul0 = bN * cN * dN
    var mul1 = cN * dN
    var mul2 = dN
    var outer_count = aN * bN * cN * dN

    var lin = 0
    while lin < outer_count:
        var a = lin // mul0
        var b = (lin // mul1) % bN
        var c = (lin // mul2) % cN
        var d = lin % dN

        var j = _clamp_index(idxT._data[lin], Ndim)
        var pos = off + a * s[0] + b * s[1] + c * s[2] + d * s[3] + j * s[4]
        out._data[pos] = out._data[pos] + srcT._data[lin]
        lin += 1
    return out.copy()

@always_inline
fn scatter_add_dim4_f32(x: Tensor[Float32], index: Tensor[Int], src: Tensor[Float32]) -> Tensor[Float32]:
    var r = len(x._shape)
    if r < 5: return x.copy()

    var outer_shape = List[Int]()
    outer_shape.append(x._shape[0]); outer_shape.append(x._shape[1]); outer_shape.append(x._shape[2]); outer_shape.append(x._shape[3])

    var idxT = index.copy()
    var ok1 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, index, idxT, ok1)
    if not ok1: return x.copy()

    var srcT = src.copy()
    var ok2 = False
    _squeeze_trailing1_like_outer_F32(outer_shape, src, srcT, ok2)
    if not ok2: return x.copy()

    var out = x.copy()
    var off = out._offset
    var s = out._strides.copy()
    var Ndim = out._shape[4]

    var aN = outer_shape[0]; var bN = outer_shape[1]; var cN = outer_shape[2]; var dN = outer_shape[3]
    var mul0 = bN * cN * dN
    var mul1 = cN * dN
    var mul2 = dN
    var outer_count = aN * bN * cN * dN

    var lin = 0
    while lin < outer_count:
        var a = lin // mul0
        var b = (lin // mul1) % bN
        var c = (lin // mul2) % cN
        var d = lin % dN

        var j = _clamp_index(idxT._data[lin], Ndim)
        var pos = off + a * s[0] + b * s[1] + c * s[2] + d * s[3] + j * s[4]
        out._data[pos] = out._data[pos] + srcT._data[lin]
        lin += 1
    return out.copy()

@always_inline
fn scatter_add_dim4_f64(x: Tensor[Float64], index: Tensor[Int], src: Tensor[Float64]) -> Tensor[Float64]:
    var r = len(x._shape)
    if r < 5: return x.copy()

    var outer_shape = List[Int]()
    outer_shape.append(x._shape[0]); outer_shape.append(x._shape[1]); outer_shape.append(x._shape[2]); outer_shape.append(x._shape[3])

    var idxT = index.copy()
    var ok1 = False
    _squeeze_trailing1_like_outer_Int(outer_shape, index, idxT, ok1)
    if not ok1: return x.copy()

    var srcT = src.copy()
    var ok2 = False
    _squeeze_trailing1_like_outer_F64(outer_shape, src, srcT, ok2)
    if not ok2: return x.copy()

    var out = x.copy()
    var off = out._offset
    var s = out._strides.copy()
    var Ndim = out._shape[4]

    var aN = outer_shape[0]; var bN = outer_shape[1]; var cN = outer_shape[2]; var dN = outer_shape[3]
    var mul0 = bN * cN * dN
    var mul1 = cN * dN
    var mul2 = dN
    var outer_count = aN * bN * cN * dN

    var lin = 0
    while lin < outer_count:
        var a = lin // mul0
        var b = (lin // mul1) % bN
        var c = (lin // mul2) % cN
        var d = lin % dN

        var j = _clamp_index(idxT._data[lin], Ndim)
        var pos = off + a * s[0] + b * s[1] + c * s[2] + d * s[3] + j * s[4]
        out._data[pos] = out._data[pos] + srcT._data[lin]
        lin += 1
    return out.copy()


@always_inline
fn _prod_shape(xs: List[Int]) -> Int:
    var p = 1
    var i = 0
    while i < len(xs):
        p = p * xs[i]
        i += 1
    return p

@always_inline
fn _is_vec_shape(shp: List[Int]) -> Bool:
    return len(shp) == 1

@always_inline
fn _is_mat_shape(shp: List[Int]) -> Bool:
    return len(shp) == 2

@always_inline
fn _clone_contiguous_f64(x: Tensor[Float64]) -> Tensor[Float64]:
    var shp = x._shape.copy()
    var r   = len(shp)
    var out = tensor.zeros_f64(shp)
    var expect = compute_row_major_strides(shp)
    var n = 1
    var i = 0
    while i < r:
        n = n * shp[i]
        i += 1
    var L = 0
    while L < n:
        var src_off = x._offset
        var d = 0
        while d < r:
            var idx_d = (L // expect[d]) % shp[d]
            src_off = src_off + idx_d * x._strides[d]
            d += 1
        out._data[out._offset + L] = x._data[src_off]
        L += 1
    return out.copy()

@always_inline
fn _ensure_contiguous_f64(x: Tensor[Float64]) -> Tensor[Float64]:
    var r = len(x._shape)
    var expect = compute_row_major_strides(x._shape)
    var is_contig = True
    var i = 0
    while i < r:
        if x._strides[i] != expect[i]:
            is_contig = False
            break
        i += 1
    if is_contig:
        return x.copy()
    return _clone_contiguous_f64(x).copy()

@always_inline
fn _matmul_square_f64(A: Tensor[Float64], B: Tensor[Float64]) -> Tensor[Float64]:
    var n = A._shape[0]
    var C = tensor.zeros_f64([n, n])
    var sAr = A._strides[0]; var sAc = A._strides[1]
    var sBr = B._strides[0]; var sBc = B._strides[1]
    var sCr = C._strides[0]; var sCc = C._strides[1]
    var i = 0
    while i < n:
        var k = 0
        while k < n:
            var aik = A._data[A._offset + i * sAr + k * sAc]
            var j = 0
            while j < n:
                var idx = C._offset + i * sCr + j * sCc
                var bkj = B._data[B._offset + k * sBr + j * sBc]
                C._data[idx] = C._data[idx] + aik * bkj
                j += 1
            k += 1
        i += 1
    return C.copy()

@always_inline
fn qr(xx: Tensor[Float64]) -> (Tensor[Float64], Tensor[Float64]):
    var rA = len(xx._shape)
    if rA < 2:
        var Q1 = tensor.eye_f64(1).to_float64()
        var R1 = tensor.zeros_f64([1, 1])
        return (Q1.copy(), R1.copy())
    var n0 = xx._shape[rA - 2]
    var n1 = xx._shape[rA - 1]
    if n0 != n1:
        var Qbad = xx.copy()
        var Rbad = tensor.zeros_f64([n1, n1])
        return (Qbad.copy(), Rbad.copy())
    var n = n0
    if rA == 2:
        var Ai = _ensure_contiguous_f64(xx).reshape([n, n])
        var Q  = tensor.zeros_f64([n, n])
        var R  = tensor.zeros_f64([n, n])
        var s0 = Ai._strides[0]
        var s1 = Ai._strides[1]
        var k = 0
        while k < n:
            var r = 0
            while r < n:
                var val = Ai._data[Ai._offset + r * s0 + k * s1]
                Q._data[Q._offset + r * s0 + k * s1] = val
                r += 1
            var j = 0
            while j < k:
                var dot = 0.0
                var t = 0
                while t < n:
                    var qjt = Q._data[Q._offset + t * s0 + j * s1]
                    var vt  = Q._data[Q._offset + t * s0 + k * s1]
                    dot = dot + qjt * vt
                    t += 1
                R._data[R._offset + j * s0 + k * s1] = dot
                t = 0
                while t < n:
                    var idx_vk = Q._offset + t * s0 + k * s1
                    var qjt2   = Q._data[Q._offset + t * s0 + j * s1]
                    Q._data[idx_vk] = Q._data[idx_vk] - dot * qjt2
                    t += 1
                j += 1
            var norm2 = 0.0
            var u = 0
            while u < n:
                var vv = Q._data[Q._offset + u * s0 + k * s1]
                norm2 = norm2 + vv * vv
                u += 1
            var rkk = 0.0
            if norm2 > 0.0:
                rkk = sqrt64(norm2)
            R._data[R._offset + k * s0 + k * s1] = rkk
            if rkk != 0.0:
                var inv_rkk = 1.0 / rkk
                var t2 = 0
                while t2 < n:
                    var idx = Q._offset + t2 * s0 + k * s1
                    Q._data[idx] = Q._data[idx] * inv_rkk
                    t2 += 1
            k += 1
        return (Q.copy(), R.copy())
    var batch_shape = List[Int]()
    var i = 0
    while i < rA - 2:
        batch_shape.append(xx._shape[i])
        i += 1
    var B = _prod_shape(batch_shape)
    var A3 = xx.copy()
    if rA == 2:
        A3 = xx.reshape([1, n, n])
    else:
        A3 = xx.reshape([B, n, n])
    var Q3 = tensor.zeros_f64([B, n, n])
    var R3 = tensor.zeros_f64([B, n, n])
    var bi = 0
    while bi < B:
        var Ai0 = A3[bi]
        var Ai  = _ensure_contiguous_f64(Ai0).reshape([n, n])
        var Qi = tensor.zeros_f64([n, n])
        var Ri = tensor.zeros_f64([n, n])
        var s0 = Ai._strides[0]
        var s1 = Ai._strides[1]
        var k = 0
        while k < n:
            var r = 0
            while r < n:
                var val = Ai._data[Ai._offset + r * s0 + k * s1]
                Qi._data[Qi._offset + r * s0 + k * s1] = val
                r += 1
            var j = 0
            while j < k:
                var dot = 0.0
                var t = 0
                while t < n:
                    var qjt = Qi._data[Qi._offset + t * s0 + j * s1]
                    var vt  = Qi._data[Qi._offset + t * s0 + k * s1]
                    dot = dot + qjt * vt
                    t += 1
                Ri._data[Ri._offset + j * s0 + k * s1] = dot
                t = 0
                while t < n:
                    var idx_vk = Qi._offset + t * s0 + k * s1
                    var qjt2   = Qi._data[Qi._offset + t * s0 + j * s1]
                    Qi._data[idx_vk] = Qi._data[idx_vk] - dot * qjt2
                    t += 1
                j += 1
            var norm2 = 0.0
            var u = 0
            while u < n:
                var vv = Qi._data[Qi._offset + u * s0 + k * s1]
                norm2 = norm2 + vv * vv
                u += 1
            var rkk = 0.0
            if norm2 > 0.0:
                rkk = sqrt64(norm2)
            Ri._data[Ri._offset + k * s0 + k * s1] = rkk
            if rkk != 0.0:
                var inv_rkk = 1.0 / rkk
                var t2 = 0
                while t2 < n:
                    var idx = Qi._offset + t2 * s0 + k * s1
                    Qi._data[idx] = Qi._data[idx] * inv_rkk
                    t2 += 1
            k += 1
        Q3[bi] = Qi
        R3[bi] = Ri
        bi += 1
    if len(batch_shape) == 0:
        return (Q3[0].copy(), R3[0].copy())
    else:
        var qshp = batch_shape.copy(); qshp.append(n); qshp.append(n)
        var rshp = batch_shape.copy(); rshp.append(n); rshp.append(n)
        return (Q3.reshape(qshp), R3.reshape(rshp))

@always_inline
fn inv(xx: Tensor[Float64]) -> Tensor[Float64]:
    var rA = len(xx._shape)
    if rA < 2:
        return tensor.eye_f64(1).to_float64()
    var n0 = xx._shape[rA - 2]
    var n1 = xx._shape[rA - 1]
    if n0 != n1:
        return xx.copy()
    var n = n0
    if rA == 2:
        var Ai = _ensure_contiguous_f64(xx).reshape([n, n])
        var M   = Ai.copy()
        var Inv = tensor.eye_f64(n).to_float64()
        var s0 = M._strides[0]
        var s1 = M._strides[1]
        var k = 0
        while k < n:
            var piv = k
            var best = abs64(M._data[M._offset + k * s0 + k * s1])
            var r = k + 1
            while r < n:
                var v = abs64(M._data[M._offset + r * s0 + k * s1])
                if v > best:
                    best = v
                    piv = r
                r += 1
            if piv != k:
                var c = 0
                while c < n:
                    var a0 = M._data[M._offset + k   * s0 + c * s1]
                    var a1 = M._data[M._offset + piv * s0 + c * s1]
                    M._data[M._offset + k   * s0 + c * s1] = a1
                    M._data[M._offset + piv * s0 + c * s1] = a0
                    var b0 = Inv._data[Inv._offset + k   * s0 + c * s1]
                    var b1 = Inv._data[Inv._offset + piv * s0 + c * s1]
                    Inv._data[Inv._offset + k   * s0 + c * s1] = b1
                    Inv._data[Inv._offset + piv * s0 + c * s1] = b0
                    c += 1
            var diag = M._data[M._offset + k * s0 + k * s1]
            if diag == 0.0:
                return tensor.eye_f64(n).to_float64()
            var inv_diag = 1.0 / diag
            var c2 = 0
            while c2 < n:
                M  ._data[M  ._offset + k * s0 + c2 * s1] = M  ._data[M  ._offset + k * s0 + c2 * s1] * inv_diag
                Inv._data[Inv._offset + k * s0 + c2 * s1] = Inv._data[Inv._offset + k * s0 + c2 * s1] * inv_diag
                c2 += 1
            var rr = 0
            while rr < n:
                if rr != k:
                    var factor = M._data[M._offset + rr * s0 + k * s1]
                    if factor != 0.0:
                        var cc = 0
                        while cc < n:
                            var rrcc = M  ._offset + rr * s0 + cc * s1
                            var rkcc = M  ._offset + k  * s0 + cc * s1
                            var irrcc= Inv._offset + rr * s0 + cc * s1
                            var irkcc= Inv._offset + k  * s0 + cc * s1
                            M  ._data[rrcc] = M  ._data[rrcc] - factor * M  ._data[rkcc]
                            Inv._data[irrcc]= Inv._data[irrcc]- factor * Inv._data[irkcc]
                            cc += 1
                rr += 1
            k += 1
        return Inv.copy()
    var batch_shape = List[Int]()
    var i = 0
    while i < rA - 2:
        batch_shape.append(xx._shape[i])
        i += 1
    var B = _prod_shape(batch_shape)
    var A3 = xx.copy()
    if rA == 2:
        A3 = xx.reshape([1, n, n])
    else:
        A3 = xx.reshape([B, n, n])
    var out = tensor.zeros_f64([B, n, n])
    var bi = 0
    while bi < B:
        var Ai0 = A3[bi]
        var Ai  = _ensure_contiguous_f64(Ai0).reshape([n, n])
        var M   = Ai.copy()
        var Inv = tensor.eye_f64(n).to_float64()
        var s0 = M._strides[0]
        var s1 = M._strides[1]
        var k = 0
        while k < n:
            var piv = k
            var best = abs64(M._data[M._offset + k * s0 + k * s1])
            var r = k + 1
            while r < n:
                var v = abs64(M._data[M._offset + r * s0 + k * s1])
                if v > best:
                    best = v
                    piv = r
                r += 1
            if piv != k:
                var c = 0
                while c < n:
                    var a0 = M._data[M._offset + k   * s0 + c * s1]
                    var a1 = M._data[M._offset + piv * s0 + c * s1]
                    M._data[M._offset + k   * s0 + c * s1] = a1
                    M._data[M._offset + piv * s0 + c * s1] = a0
                    var b0 = Inv._data[Inv._offset + k   * s0 + c * s1]
                    var b1 = Inv._data[Inv._offset + piv * s0 + c * s1]
                    Inv._data[Inv._offset + k   * s0 + c * s1] = b1
                    Inv._data[Inv._offset + piv * s0 + c * s1] = b0
                    c += 1
            var diag = M._data[M._offset + k * s0 + k * s1]
            if diag == 0.0:
                out[bi] = tensor.eye_f64(n).to_float64()
                break
            var inv_diag = 1.0 / diag
            var c2 = 0
            while c2 < n:
                M  ._data[M  ._offset + k * s0 + c2 * s1] = M  ._data[M  ._offset + k * s0 + c2 * s1] * inv_diag
                Inv._data[Inv._offset + k * s0 + c2 * s1] = Inv._data[Inv._offset + k * s0 + c2 * s1] * inv_diag
                c2 += 1
            var rr = 0
            while rr < n:
                if rr != k:
                    var factor = M._data[M._offset + rr * s0 + k * s1]
                    if factor != 0.0:
                        var cc = 0
                        while cc < n:
                            var rrcc = M  ._offset + rr * s0 + cc * s1
                            var rkcc = M  ._offset + k  * s0 + cc * s1
                            var irrcc= Inv._offset + rr * s0 + cc * s1
                            var irkcc= Inv._offset + k  * s0 + cc * s1
                            M  ._data[rrcc] = M  ._data[rrcc] - factor * M  ._data[rkcc]
                            Inv._data[irrcc]= Inv._data[irrcc]- factor * Inv._data[irkcc]
                            cc += 1
                rr += 1
            k += 1
        out[bi] = Inv
        bi += 1
    if len(batch_shape) == 0:
        return out[0].copy()
    else:
        var shp = batch_shape.copy()
        shp.append(n)
        shp.append(n)
        return out.reshape(shp)

@always_inline
fn _eig_sym_qr(C_in: Tensor[Float64], iters: Int = 64) -> (Tensor[Float64], Tensor[Float64]):
    var n = C_in._shape[0]
    var C = _ensure_contiguous_f64(C_in).reshape([n, n])
    var V = tensor.eye_f64(n).to_float64()
    var t = 0
    while t < iters:
        var qr_pair = C.qr()
        var Q = qr_pair[0].copy()
        var R = qr_pair[1].copy()
        C = _matmul_square_f64(R, Q)
        V = _matmul_square_f64(V, Q)
        t += 1
    return (V.copy(), C.copy())

@always_inline
fn _permute_cols_f64(M: Tensor[Float64], perm: List[Int]) -> Tensor[Float64]:
    var n = M._shape[0]
    var out = tensor.zeros_f64([n, n])
    var sr = M._strides[0]; var sc = M._strides[1]
    var or_ = out._strides[0]; var oc = out._strides[1]
    var j = 0
    while j < n:
        var src = perm[j]
        var i = 0
        while i < n:
            out._data[out._offset + i * or_ + j * oc] = M._data[M._offset + i * sr + src * sc]
            i += 1
        j += 1
    return out.copy()

@always_inline
fn _scale_cols_f64(M: Tensor[Float64], scale: Tensor[Float64]) -> Tensor[Float64]:
    var n  = M._shape[0]
    var nc = M._shape[1]
    var out = M.copy()
    var sr = out._strides[0]; var sc = out._strides[1]
    var ss = scale._strides[0]
    var j = 0
    while j < nc:
        var f = scale._data[scale._offset + j * ss]
        var i = 0
        while i < n:
            var idx = out._offset + i * sr + j * sc
            out._data[idx] = out._data[idx] * f
            i += 1
        j += 1
    return out.copy()

@always_inline
fn svd(xx: Tensor[Float64]) -> (Tensor[Float64], Tensor[Float64], Tensor[Float64]):
    var rA = len(xx._shape)
    if rA < 2:
        var U1 = tensor.eye_f64(1).to_float64()
        var S1 = tensor.from_list_float64([1.0])
        var Vh1 = tensor.eye_f64(1).to_float64()
        return (U1.copy(), S1.copy(), Vh1.copy())
    var n0 = xx._shape[rA - 2]
    var n1 = xx._shape[rA - 1]
    if n0 != n1:
        var n = n0
        var Ubad = tensor.eye_f64(n).to_float64()
        var Sbad = tensor.zeros_f64([n])
        var Vhbad = tensor.eye_f64(n1).to_float64()
        return (Ubad.copy(), Sbad.copy(), Vhbad.copy())
    var n = n0
    if rA == 2:
        var Ai = _ensure_contiguous_f64(xx).reshape([n, n])
        var AT = Ai.transpose([1, 0])
        var C  = AT.matmul(Ai)
        var ev = _eig_sym_qr(C, 64)
        var V  = ev[0].copy()
        var Dm = ev[1].copy()
        var lam = tensor.zeros_f64([n])
        var sr = Dm._strides[0]; var sc = Dm._strides[1]
        var s1 = lam._strides[0]
        var j = 0
        while j < n:
            var lj = Dm._data[Dm._offset + j * sr + j * sc]
            if lj < 0.0:
                lj = 0.0
            lam._data[lam._offset + j * s1] = lj
            j += 1
        var Svec = tensor.zeros_f64([n])
        j = 0
        while j < n:
            var v = lam._data[lam._offset + j * s1]
            Svec._data[Svec._offset + j * s1] = sqrt64(v)
            j += 1
        var perm = List[Int]()
        var jj = 0
        while jj < n:
            perm.append(jj)
            jj += 1
        var a = 0
        while a < n:
            var b = a + 1
            while b < n:
                var sa = Svec._data[Svec._offset + a * s1]
                var sb = Svec._data[Svec._offset + b * s1]
                if sb > sa:
                    var tmp = perm[a]; perm[a] = perm[b]; perm[b] = tmp
                    var tv = Svec._data[Svec._offset + a * s1]
                    Svec._data[Svec._offset + a * s1] = Svec._data[Svec._offset + b * s1]
                    Svec._data[Svec._offset + b * s1] = tv
                b += 1
            a += 1
        var Vs = _permute_cols_f64(V, perm)
        var AV = Ai.matmul(Vs)
        var inv_sigma = tensor.zeros_f64([n])
        j = 0
        while j < n:
            var s = Svec._data[Svec._offset + j * s1]
            var invs = 0.0
            if s != 0.0:
                invs = 1.0 / s
            inv_sigma._data[inv_sigma._offset + j * s1] = invs
            j += 1
        var U  = _scale_cols_f64(AV, inv_sigma)
        var Vh = Vs.transpose([1, 0])
        return (U.copy(), Svec.copy(), Vh.copy())
    var batch_shape = List[Int]()
    var i = 0
    while i < rA - 2:
        batch_shape.append(xx._shape[i])
        i += 1
    var B = _prod_shape(batch_shape)
    var A3 = xx.copy()
    if rA == 2:
        A3 = xx.reshape([1, n, n])
    else:
        A3 = xx.reshape([B, n, n])
    var U3  = tensor.zeros_f64([B, n, n])
    var S3v = tensor.zeros_f64([B, n])
    var Vh3 = tensor.zeros_f64([B, n, n])
    var bi = 0
    while bi < B:
        var Ai = _ensure_contiguous_f64(A3[bi]).reshape([n, n])
        var AT = Ai.transpose([1, 0])
        var C  = AT.matmul(Ai)
        var ev = _eig_sym_qr(C, 64)
        var V  = ev[0].copy()
        var Dm = ev[1].copy()
        var lam = tensor.zeros_f64([n])
        var sr = Dm._strides[0]; var sc = Dm._strides[1]
        var s1 = lam._strides[0]
        var j = 0
        while j < n:
            var lj = Dm._data[Dm._offset + j * sr + j * sc]
            if lj < 0.0:
                lj = 0.0
            lam._data[lam._offset + j * s1] = lj
            j += 1
        var Svec = tensor.zeros_f64([n])
        j = 0
        while j < n:
            var v = lam._data[lam._offset + j * s1]
            Svec._data[Svec._offset + j * s1] = sqrt64(v)
            j += 1
        var perm = List[Int]()
        var jj = 0
        while jj < n:
            perm.append(jj)
            jj += 1
        var a = 0
        while a < n:
            var b = a + 1
            while b < n:
                var sa = Svec._data[Svec._offset + a * s1]
                var sb = Svec._data[Svec._offset + b * s1]
                if sb > sa:
                    var tmp = perm[a]; perm[a] = perm[b]; perm[b] = tmp
                    var tv = Svec._data[Svec._offset + a * s1]
                    Svec._data[Svec._offset + a * s1] = Svec._data[Svec._offset + b * s1]
                    Svec._data[Svec._offset + b * s1] = tv
                b += 1
            a += 1
        var Vs = _permute_cols_f64(V, perm)
        var AV = Ai.matmul(Vs)
        var inv_sigma = tensor.zeros_f64([n])
        j = 0
        while j < n:
            var s = Svec._data[Svec._offset + j * s1]
            var invs = 0.0
            if s != 0.0:
                invs = 1.0 / s
            inv_sigma._data[inv_sigma._offset + j * s1] = invs
            j += 1
        var U  = _scale_cols_f64(AV, inv_sigma)
        var Vh = Vs.transpose([1, 0])
        U3[bi]  = U
        S3v[bi] = Svec
        Vh3[bi] = Vh
        bi += 1
    if len(batch_shape) == 0:
        return (U3[0].copy(), S3v[0].copy(), Vh3[0].copy())
    else:
        var ush = batch_shape.copy(); ush.append(n); ush.append(n)
        var ssh = batch_shape.copy(); ssh.append(n)
        var vsh = batch_shape.copy(); vsh.append(n); vsh.append(n)
        return (U3.reshape(ush), S3v.reshape(ssh), Vh3.reshape(vsh))

@always_inline
fn cholesky(xx: Tensor[Float64]) -> Tensor[Float64]:
    var rA = len(xx._shape)
    if rA < 2:
        return tensor.eye_f64(1).to_float64()
    var n0 = xx._shape[rA - 2]
    var n1 = xx._shape[rA - 1]
    if n0 != n1:
        return xx.copy()
    var n = n0
    if rA == 2:
        var Ai = _ensure_contiguous_f64(xx).reshape([n, n])
        var L  = tensor.zeros_f64([n, n])
        var sr = Ai._strides[0]
        var sc = Ai._strides[1]
        var lr = L._strides[0]
        var lc = L._strides[1]
        var jitter = 1e-12
        var irow = 0
        while irow < n:
            var jcol = 0
            while jcol <= irow:
                var s = Ai._data[Ai._offset + irow * sr + jcol * sc]
                var k = 0
                while k < jcol:
                    var Lik = L._data[L._offset + irow * lr + k * lc]
                    var Ljk = L._data[L._offset + jcol * lr + k * lc]
                    s = s - Lik * Ljk
                    k += 1
                if irow == jcol:
                    var d = s
                    if d <= 0.0:
                        d = d + jitter
                        if d < 0.0:
                            d = 0.0
                    var val = sqrt64(d)
                    L._data[L._offset + irow * lr + jcol * lc] = val
                else:
                    var Ljj = L._data[L._offset + jcol * lr + jcol * lc]
                    var lij = 0.0
                    if Ljj != 0.0:
                        lij = s / Ljj
                    L._data[L._offset + irow * lr + jcol * lc] = lij
                jcol += 1
            irow += 1
        return L.copy()
    var batch_shape = List[Int]()
    var i = 0
    while i < rA - 2:
        batch_shape.append(xx._shape[i])
        i += 1
    var B = _prod_shape(batch_shape)
    var A3 = xx.copy()
    if rA == 2:
        A3 = xx.reshape([1, n, n])
    else:
        A3 = xx.reshape([B, n, n])
    var out = tensor.zeros_f64([B, n, n])
    var bi = 0
    while bi < B:
        var Ai = _ensure_contiguous_f64(A3[bi]).reshape([n, n])
        var L  = tensor.zeros_f64([n, n])
        var sr = Ai._strides[0]
        var sc = Ai._strides[1]
        var lr = L._strides[0]
        var lc = L._strides[1]
        var jitter = 1e-12
        var irow = 0
        while irow < n:
            var jcol = 0
            while jcol <= irow:
                var s = Ai._data[Ai._offset + irow * sr + jcol * sc]
                var k = 0
                while k < jcol:
                    var Lik = L._data[L._offset + irow * lr + k * lc]
                    var Ljk = L._data[L._offset + jcol * lr + k * lc]
                    s = s - Lik * Ljk
                    k += 1
                if irow == jcol:
                    var d = s
                    if d <= 0.0:
                        d = d + jitter
                        if d < 0.0:
                            d = 0.0
                    var val = sqrt64(d)
                    L._data[L._offset + irow * lr + jcol * lc] = val
                else:
                    var Ljj = L._data[L._offset + jcol * lr + jcol * lc]
                    var lij = 0.0
                    if Ljj != 0.0:
                        lij = s / Ljj
                    L._data[L._offset + irow * lr + jcol * lc] = lij
                jcol += 1
            irow += 1
        out[bi] = L
        bi += 1
    if len(batch_shape) == 0:
        return out[0].copy()
    else:
        var shp = batch_shape.copy()
        shp.append(n)
        shp.append(n)
        return out.reshape(shp)

@always_inline
fn solve(xx: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    var rA = len(xx._shape)
    if rA < 2:
        return b.copy()
    var n0 = xx._shape[rA - 2]
    var n1 = xx._shape[rA - 1]
    if n0 != n1:
        return b.copy()
    var n = n0
    if rA == 2:
        var rb = len(b._shape)
        if rb == 1:
            var Ainv = xx.inv()
            var out1 = Ainv.matmul_vec(b)
            return out1.copy()
        elif rb == 2:
            var Ainv = xx.inv()
            var out2 = Ainv.matmul(b)
            return out2.copy()
        else:
            return b.copy()
    var batch_shape = List[Int]()
    var i = 0
    while i < rA - 2:
        batch_shape.append(xx._shape[i])
        i += 1
    var B = _prod_shape(batch_shape)
    var A3 = xx.copy()
    if rA == 2:
        A3 = xx.reshape([1, n, n])
    else:
        A3 = xx.reshape([B, n, n])
    var rb = len(b._shape)
    var is_vec_b = False
    var k_rhs = 1
    if rb == 0:
        return b.copy()
    elif rb == 1:
        if rA == 2:
            is_vec_b = True
            k_rhs = 1
            var b2 = b.reshape([1, n])
            var Ainv = A3[0].inv()
            var out1 = Ainv.matmul_vec(b2[0])
            return out1.copy()
        else:
            return b.copy()
    elif rb >= 2:
        var expect_vec_rank = rA - 1
        var expect_mat_rank = rA
        if rb == expect_vec_rank:
            is_vec_b = True
            k_rhs = 1
            var b2 = b.reshape([B, n])
            var out = tensor.zeros_f64([B, n])
            var bi = 0
            while bi < B:
                var Ai = A3[bi]
                var Ainv = Ai.inv()
                var xi = Ainv.matmul_vec(b2[bi])
                out[bi] = xi
                bi += 1
            if len(batch_shape) == 0:
                return out[0].copy()
            else:
                var shp = batch_shape.copy()
                shp.append(n)
                return out.reshape(shp)
        elif rb == expect_mat_rank:
            is_vec_b = False
            k_rhs = b._shape[rb - 1]
            var b3 = b.reshape([B, n, k_rhs])
            var out = tensor.zeros_f64([B, n, k_rhs])
            var bi = 0
            while bi < B:
                var Ai = A3[bi]
                var Ainv = Ai.inv()
                var xi  = Ainv.matmul(b3[bi])
                out[bi] = xi
                bi += 1
            if len(batch_shape) == 0:
                return out[0].copy()
            else:
                var shp = batch_shape.copy()
                shp.append(n)
                shp.append(k_rhs)
                return out.reshape(shp)
        else:
            return b.copy()
    else:
        return b.copy()



# ----------------------------- scalar reciprocal ------------------------------
@always_inline
fn reciprocal_scalar(x: Float64, eps: Float64 = 0.0) -> Float64:
    # If |x| < eps, clamp to +/-eps to avoid div-by-zero
    var v = x
    var a = v
    if eps > 0.0:
        if v >= 0.0 and v < eps: v = eps
        if v < 0.0 and -v < eps: v = -eps
    return 1.0 / v

# ----------------------------- tensor reciprocal ------------------------------
fn reciprocal(x: tensor.Tensor[Float64], eps: Float64 = 0.0) -> tensor.Tensor[Float64]:
    # Elementwise y[i] = 1 / clamp(x[i], eps)
    var shp = x.shape()
    var y = tensor.zeros(shp)
    var n = len(x._data)
    var i = 0
    if eps <= 0.0:
        while i < n:
            y._data[i] = 1.0 / x._data[i]
            i = i + 1
    else:
        while i < n:
            var v = x._data[i]
            if v >= 0.0 and v < eps: v = eps
            if v < 0.0 and -v < eps: v = -eps
            y._data[i] = 1.0 / v
            i = i + 1
    return y.copy()

# ----------------------------- helper: safe_div -------------------------------
# Optional helper if you still prefer a.mul(tensor.reciprocal(s)):
fn safe_div(a: tensor.Tensor[Float64], s: tensor.Tensor[Float64], eps: Float64 = 0.0) -> tensor.Tensor[Float64]:
    return a.mul(reciprocal(s, eps))

fn safe_div_scalar(a: tensor.Tensor[Float64], s: Float64, eps: Float64 = 0.0) -> tensor.Tensor[Float64]:
    return a.mul_scalar(reciprocal_scalar(s, eps))



# Row-major 2D @ 2D → 2D, safe fallback on bad shapes
fn matmul2d(a: tensor.Tensor[Float64], b: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
    # Shapes
    var ashp = a.shape()   # [m, k] expected
    var bshp = b.shape()   # [k, n] expected
    var ar = len(ashp)
    var br = len(bshp)

    # Defaults for fallback
    var m = 0
    var k_a = 0
    var k_b = 0
    var n = 0
    if ar == 2:
        m = ashp[0]
        k_a = ashp[1]
    if br == 2:
        k_b = bshp[0]
        n = bshp[1]

    # Validate ranks and inner dim; on failure return zeros([m, n]) to be shape-friendly
    if ar != 2 or br != 2 or k_a != k_b:
        return tensor.zeros([m, n])

    # Fast path: standard triple loop (row-major)
    var out = tensor.zeros([m, n])

    var i = 0
    while i < m:
        var j = 0
        while j < n:
            var acc = 0.0
            var t = 0
            # a[i, t] * b[t, j]
            # a row offset: i*k_a
            # b col offset uses stride n
            var a_row_base = i * k_a
            while t < k_a:
                acc = acc + a._data[a_row_base + t] * b._data[t * n + j]
                t = t + 1
            out._data[i * n + j] = acc
            j = j + 1
        i = i + 1
    return out.copy()
