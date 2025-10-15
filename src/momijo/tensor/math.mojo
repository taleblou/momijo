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
# ====================== Softmax (stable, last-dim) ======================
@always_inline
fn _softmax_lastdim_f64(x: Tensor[Float64]) -> Tensor[Float64]:
    # Assumes row-major contiguous storage and applies softmax along the last axis.
    var shape = x._shape.copy()
    var rank  = len(shape)
    assert(rank >= 1)

    var last = shape[rank - 1]
    assert(last > 0)

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

# Optional: 1D fast-path (calls the same impl but keeps intent clear)
@always_inline
fn _softmax1d_f64(x: Tensor[Float64]) -> Tensor[Float64]:
    assert(len(x._shape) == 1)
    return _softmax_lastdim_f64(x)

# ====================== Public API (axis = -1 only) ======================
@always_inline
fn softmax(x: Tensor[Float64], axis: Int = -1) -> Tensor[Float64]:
    # Only last-dimension supported here; extend as needed.
    assert(axis == -1)   # extend for other axes if/when needed
    return _softmax_lastdim_f64(x)

@always_inline
fn softmax(x: Tensor[Float32], axis: Int = -1) -> Tensor[Float64]:
    var xf64 = to_float64(x)
    return softmax(xf64, axis)

@always_inline
fn softmax(x: Tensor[Int], axis: Int = -1) -> Tensor[Float64]:
    var xf64 = to_float64(x)
    return softmax(xf64, axis)
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

# ---- public unary (per-dtype) ----
# Import the tensor-level converters from your cast module 
@always_inline
fn apply_unary(x: Tensor[Float64], uop_id: Int) -> Tensor[Float64]:
    # Already Float64; apply directly in Float64 space.
    return apply_unary_impl[Float64](x, uop_id, to_float64_of, f64_to)

@always_inline
fn apply_unary(x: Tensor[Float32], uop_id: Int) -> Tensor[Float64]:
    # Convert input to Float64, then run the Float64 implementation.
    var xf64 = to_float64(x)
    return apply_unary_impl[Float64](xf64, uop_id, to_float64_of, f64_to)

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
fn neg_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 0)
@always_inline
fn neg_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 0)


# -------- ABS (op 1) --------
@always_inline
fn abs_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 1)
@always_inline
fn abs_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 1)
@always_inline
fn abs_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 1)

# -------- SQRT (op 2) --------
@always_inline
fn sqrt_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 2)
@always_inline
fn sqrt_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 2)
@always_inline
fn sqrt_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 2)


# -------- EXP (op 3) --------
@always_inline
fn exp_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 3)
@always_inline
fn exp_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 3)
@always_inline
fn exp_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 3)


# -------- LOG (op 4) --------
@always_inline
fn log_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 4)
@always_inline
fn log_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 4)
@always_inline
fn log_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 4)


# -------- SIN (op 5) --------
@always_inline
fn sin_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 5)
@always_inline
fn sin_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 5)
@always_inline
fn sin_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 5)

# -------- COS (op 6) --------
@always_inline
fn cos_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 6)
@always_inline
fn cos_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 6)
@always_inline
fn cos_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 6)


# -------- TAN (op 7) --------
@always_inline
fn tan_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 7)
@always_inline
fn tan_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 7)
@always_inline
fn tan_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 7)


# -------- RELU (op 8) --------
@always_inline
fn relu_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 8)
@always_inline
fn relu_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 8)
@always_inline
fn relu_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 8)

# -------- EXPM1 (op 9) --------
@always_inline
fn expm1_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 9)
@always_inline
fn expm1_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 9)
@always_inline
fn expm1_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 9)


# -------- LOG1P (op 10) --------
@always_inline
fn log1p_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 10)
@always_inline
fn log1p_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 10)
@always_inline
fn log1p_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 10)


# -------- FLOOR (op 11) --------
@always_inline
fn floor_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 11)
@always_inline
fn floor_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 11)
@always_inline
fn floor_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 11)


# -------- CEIL (op 12) --------
@always_inline
fn ceil_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 12)
@always_inline
fn ceil_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 12)
@always_inline
fn ceil_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 12)


# -------- ROUND (op 13) --------
@always_inline
fn round_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 13)
@always_inline
fn round_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 13)
@always_inline
fn round_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 13)

# -------- SIGN (op 14) --------
@always_inline
fn sign_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 14)
@always_inline
fn sign_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 14)
@always_inline
fn sign_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 14)

# -------- SIGMOID (op 15) --------
@always_inline
fn sigmoid_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 15)
@always_inline
fn sigmoid_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 15)
@always_inline
fn sigmoid_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 15)

# -------- TANH (op 16) --------
@always_inline
fn tanh_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 16)
@always_inline
fn tanh_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 16)
@always_inline
fn tanh_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 16)

# -------- SiLU (op 17) --------
@always_inline
fn silu_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 17)
@always_inline
fn silu_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 17)
@always_inline
fn silu_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 17)

# -------- GELU (op 18) --------
@always_inline
fn gelu_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 18)
@always_inline
fn gelu_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 18)
@always_inline
fn gelu_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 18)

# -------- ELU (alpha=1) (op 19) --------
@always_inline
fn elu_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 19)
@always_inline
fn elu_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 19)
@always_inline
fn elu_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 19)

# -------- SELU (op 20) --------
@always_inline
fn selu_t(x: Tensor[Float64]) -> Tensor[Float64]:
    return apply_unary(x, 20)
@always_inline
fn selu_t(x: Tensor[Float32]) -> Tensor[Float64]:
    return apply_unary(x, 20)
@always_inline
fn selu_t(x: Tensor[Int]) -> Tensor[Float64]:
    return apply_unary(x, 20)

# ====================== Binary: comparisons (mask Float64) ======================

@always_inline
fn _cmp_eval_impl[T: ImplicitlyCopyable & Copyable & Movable](
    a: T, b: T, cmp_id: Int, to_f64: fn (T) -> Float64
) -> Float64:
    var af = to_f64(a)
    var bf = to_f64(b)
    # cmp_id: 0==, 1!=, 2<, 3<=, 4>, 5>=
    if cmp_id == 0:   return 1.0 if af == bf else 0.0
    if cmp_id == 1:   return 1.0 if af != bf else 0.0
    if cmp_id == 2:   return 1.0 if af <  bf else 0.0
    if cmp_id == 3:   return 1.0 if af <= bf else 0.0
    if cmp_id == 4:   return 1.0 if af >  bf else 0.0
    # default: >=
    return 1.0 if af >= bf else 0.0

@always_inline
fn _apply_compare_impl[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    y: Tensor[T],
    cmp_id: Int,
    to_f64: fn (T) -> Float64
) -> Tensor[Float64]:
    var n = len(x._data)              # assumes same-sized flattened buffers
    var out = List[Float64]()
    out.reserve(n)
    var i = 0
    var lim = (n // 8) * 8
    while i < lim:
        out.append(_cmp_eval_impl[T](x._data[i    ], y._data[i    ], cmp_id, to_f64))
        out.append(_cmp_eval_impl[T](x._data[i + 1], y._data[i + 1], cmp_id, to_f64))
        out.append(_cmp_eval_impl[T](x._data[i + 2], y._data[i + 2], cmp_id, to_f64))
        out.append(_cmp_eval_impl[T](x._data[i + 3], y._data[i + 3], cmp_id, to_f64))
        out.append(_cmp_eval_impl[T](x._data[i + 4], y._data[i + 4], cmp_id, to_f64))
        out.append(_cmp_eval_impl[T](x._data[i + 5], y._data[i + 5], cmp_id, to_f64))
        out.append(_cmp_eval_impl[T](x._data[i + 6], y._data[i + 6], cmp_id, to_f64))
        out.append(_cmp_eval_impl[T](x._data[i + 7], y._data[i + 7], cmp_id, to_f64))
        i += 8
    while i < n:
        out.append(_cmp_eval_impl[T](x._data[i], y._data[i], cmp_id, to_f64))
        i += 1
    return Tensor[Float64](out, x._shape)

# Per-dtype dispatch (mirrors your unary style)
@always_inline
fn _apply_compare(x: Tensor[Float64], y: Tensor[Float64], cmp_id: Int) -> Tensor[Int]:
    # Run impl at T=Float64, then cast mask (0.0/1.0) to Int (0/1).
    var mf64 = _apply_compare_impl[Float64](x, y, cmp_id, to_float64_of)
    return to_int(mf64)

@always_inline
fn _apply_compare(x: Tensor[Float32], y: Tensor[Float32], cmp_id: Int) -> Tensor[Int]:
    # Run impl at T=Float32, then cast mask to Int.
    var mf64 = _apply_compare_impl[Float32](x, y, cmp_id, to_float64_of)
    return to_int(mf64)

@always_inline
fn _apply_compare(x: Tensor[Int], y: Tensor[Int], cmp_id: Int) -> Tensor[Int]:
    # Run impl at T=Int, then cast mask to Int.
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
fn div_t(a: Tensor[Float32], b: Tensor[Float32]) -> Tensor[Float64]:
    return apply_broadcast3(a, b, 3)

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
fn mod_t(a: Tensor[Float32], b: Tensor[Float32]) -> Tensor[Float64]:
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
fn landnot_t(x: Tensor[Float32], y: Tensor[FloFloat32at64]) -> Tensor[Bool]:
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



@always_inline
fn _not_to_bool_impl[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T], to_f64: fn (T) -> Float64
) -> Tensor[Bool]:
    # Build a Bool mask where output[i] = True if x[i] == 0 else False
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

    var strides = compute_row_major_strides(x._shape)
    return Tensor[Bool](data=out, shape=x._shape, strides=strides)

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
    return Tensor[Bool](data=out, shape=x._shape, strides=strides)

 
 




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
# ---------- Std ----------
fn std[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    axis: Optional[Int] = None,
    unbiased: Bool = True,
    keepdims: Bool = False
) -> Tensor[T]:
    # Use the specialized variance(...) that returns Tensor[Float64]
    var v: Tensor[Float64] = variance(x, axis, unbiased, keepdims)

    var outd = List[T]()
    outd.reserve(len(v._data))

    var i = 0
    var n = len(v._data)
    while i < n:
        var s = v._data[i]
        if s < 0.0:
            s = 0.0
        outd.append(T(sqrt(s)))
        i = i + 1

    return Tensor[T](v._shape, outd)

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
    var shp = x._shape
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
    var shp = x._shape
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
fn min_t(self: Tensor[Float64]) -> Float64:
    var xs = self._data
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
fn max_t(self: Tensor[Float64]) -> Float64:
    var xs = self._data.copy()
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
fn min_t(self: Tensor[Float32]) -> Float32:
    var xs = self._data
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
fn max_t(self: Tensor[Float32]) -> Float32:
    var xs = self._data.copy()
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
fn min_t(self: Tensor[Int]) -> Int:
    var xs = self._data.copy()
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
fn max_t(self: Tensor[Int]) -> Int:
    var xs = self._data.copy()
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

 
# ========= variance for Float64 =========
fn variance(
    x: Tensor[Float64],
    axis: Optional[Int] = None,
    unbiased: Bool = True,
    keepdims: Bool = False
) -> Tensor[Float64]:
    var shp = x._shape.copy()
    var rank = len(shp)

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
            var denom = Float64(n)
            if unbiased and n > 1:
                denom = Float64(n - 1)
            var vv = 0.0
            if denom > 0.0:
                vv = m2 / denom
            out_list.append(vv)

        var out_shape = List[Int]()
        if keepdims:
            var j = 0
            while j < rank:
                out_shape.append(1)
                j = j + 1
        else:
            out_shape.append(1)

        return Tensor[Float64](out_shape, out_list)

    var ax = normalize_axis(axis.value(), rank)
    var out_shape2 = shape_drop_axis(shp, ax)

    var reduce_n = 1
    if rank != 0:
        reduce_n = shp[ax]

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

        var denom2 = Float64(reduce_n)
        if unbiased and reduce_n > 1:
            denom2 = Float64(reduce_n - 1)
        var vv2 = 0.0
        if denom2 > 0.0:
            vv2 = m22 / denom2
        out_vals.append(vv2)
        o = o + 1

    var tout = Tensor[Float64](out_shape2, out_vals)
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
    return tout


# -------- standard deviation (uses Float64 path, then converts to T) --------
fn std[T: ImplicitlyCopyable & Copyable & Movable](
    x: Tensor[T],
    axis: Optional[Int] = None,
    unbiased: Bool = True,
    keepdims: Bool = False
) -> Tensor[T]:
    var v = variance[T](x, axis, unbiased, keepdims)
    var outd = List[T]()
    outd.reserve(len(v._data))
    var i = 0
    var n = len(v._data)
    while i < n:
        var s = v._data[i]
        if s < 0.0:
            s = 0.0
        outd.append(T(sqrt(s)))
        i += 1
    return Tensor[T](outd, v._shape)

# -------- reduce_max for Float64 (unchanged, constructor-style where needed) --------
fn reduce_max_f64(x: Tensor[Float64]) -> Float64:
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

fn reduce_min_f64(x: Tensor[Float64]) -> Float64:
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

fn max[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[T]:
    var shp = x._shape.copy()
    var rank = len(shp)
    var xc = x.astype[Float64]()

    if axis is None:
        var mval = reduce_max_f64(xc)
        var out = Tensor[Float64]([mval], [1]).astype[T]()
        if keepdims:
            var kd = List[Int]()
            var i = 0
            while i < rank:
                kd.append(1)
                i += 1
            return out.reshape(kd)
        return out

    var ax = normalize_axis(axis.value(), rank)
    var reduce_n = shp[ax]

    var inner = 1
    var i_in = ax + 1
    while i_in < rank:
        inner = inner * shp[i_in]
        i_in += 1

    var out_shape = shape_drop_axis(shp, ax)
    var outer = numel(out_shape)
    var outv = List[Float64]()
    outv.reserve(outer)
    var base_stride = inner
    var block = reduce_n * inner

    var o = 0
    while o < outer:
        var base = (o // inner) * block + (o % inner)
        var mv = xc._data[base]
        var k = 1
        while k < reduce_n:
            var v = xc._data[base + k * base_stride]
            if v > mv:
                mv = v
            k += 1
        outv.append(mv)
        o += 1

    var tout = Tensor[Float64](outv, out_shape).astype[T]()
    if keepdims:
        return tout.reshape(keepdims_shape(shp, ax))
    return tout

fn min[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None, keepdims: Bool = False) -> Tensor[T]:
    var shp = x._shape.copy()
    var rank = len(shp)
    var xc = x.astype[Float64]()

    if axis is None:
        var mval = reduce_min_f64(xc)
        var out = Tensor[Float64]([mval], [1]).astype[T]()
        if keepdims:
            var kd = List[Int]()
            var i = 0
            while i < rank:
                kd.append(1)
                i += 1
            return out.reshape(kd)
        return out

    var ax = normalize_axis(axis.value(), rank)
    var reduce_n = shp[ax]

    var inner = 1
    var i_in = ax + 1
    while i_in < rank:
        inner = inner * shp[i_in]
        i_in += 1

    var out_shape = shape_drop_axis(shp, ax)
    var outer = numel(out_shape)
    var outv = List[Float64]()
    outv.reserve(outer)
    var base_stride = inner
    var block = reduce_n * inner

    var o = 0
    while o < outer:
        var base = (o // inner) * block + (o % inner)
        var mv = xc._data[base]
        var k = 1
        while k < reduce_n:
            var v = xc._data[base + k * base_stride]
            if v < mv:
                mv = v
            k += 1
        outv.append(mv)
        o += 1

    var tout = Tensor[Float64](outv, out_shape).astype[T]()
    if keepdims:
        return tout.reshape(keepdims_shape(shp, ax))
    return tout

fn argmax[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None) -> Tensor[Int]:
    var shp = x._shape.copy()
    var rank = len(shp)
    var xc = x.astype[Float64]()

    if axis is None:
        var n = len(xc._data)
        var best = 0
        var bestv = xc._data[0] if n > 0 else 0.0
        var i = 1
        while i < n:
            var v = xc._data[i]
            if v > bestv:
                bestv = v
                best = i
            i += 1
        return Tensor[Int]([best], [1])

    var ax = normalize_axis(axis.value(), rank)
    var out_shape = shape_drop_axis(shp, ax)
    var reduce_n = shp[ax]

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
        var bestv2 = xc._data[base]
        var k = 1
        while k < reduce_n:
            var v = xc._data[base + k * base_stride]
            if v > bestv2:
                bestv2 = v
                besti = k
            k += 1
        out_idx.append(besti)
        o += 1
    return Tensor[Int](out_idx, out_shape)

fn argmin[T: ImplicitlyCopyable & Copyable & Movable](x: Tensor[T], axis: Optional[Int] = None) -> Tensor[Int]:
    var shp = x._shape.copy()
    var rank = len(shp)
    var xc = x.astype[Float64]()

    if axis is None:
        var n = len(xc._data)
        var best = 0
        var bestv = xc._data[0] if n > 0 else 0.0
        var i = 1
        while i < n:
            var v = xc._data[i]
            if v < bestv:
                bestv = v
                best = i
            i += 1
        return Tensor[Int]([best], [1])

    var ax = normalize_axis(axis.value(), rank)
    var out_shape = shape_drop_axis(shp, ax)
    var reduce_n = shp[ax]

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
        var bestv2 = xc._data[base]
        var k = 1
        while k < reduce_n:
            var v = xc._data[base + k * base_stride]
            if v < bestv2:
                bestv2 = v
                besti = k
            k += 1
        out_idx.append(besti)
        o += 1
    return Tensor[Int](out_idx, out_shape)

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

 
