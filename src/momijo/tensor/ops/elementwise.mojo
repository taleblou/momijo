# Project:      Momijo
# Module:       src.momijo.tensor.ops.elementwise
# File:         elementwise.mojo
# Path:         src/momijo/tensor/ops/elementwise.mojo
#
# Description:  Core tensor/ndarray components: shapes/strides, broadcasting rules,
#               element-wise ops, and foundational kernels.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Key functions: argmax_index, argmin_index, __module_name__, _same_len_min, ew_add, ew_sub, ew_mul, ew_div ...
#   - Uses generic functions/types with explicit trait bounds.


from math import exp, log, sqrt, tanh
from momijo.tensor.registry import add as registry_add
from momijo.tensor.tensor import Tensor

fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]
            idx = i
        i += 1
    return idx
fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]
            idx = i
        i += 1
    return idx

fn ensure_not_empty[T: Copyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0
fn __module_name__() -> String:
    return String("momijo/tensor/ops/elementwise.mojo")

# ---------- Elementwise (List[Float64]) ----------

@always_inline
fn _same_len_min(a_len: Int, b_len: Int) -> Int:
    if a_len < b_len:
        return a_len
    return b_len
fn ew_add(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var n = _same_len_min(len(a), len(b))
    var out = List[Float64]()
    var i = 0
    while i < n:
        out.append(a[i] + b[i])
        i += 1
    return out
fn ew_sub(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var n = _same_len_min(len(a), len(b))
    var out = List[Float64]()
    var i = 0
    while i < n:
        out.append(a[i] - b[i])
        i += 1
    return out
fn ew_mul(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var n = _same_len_min(len(a), len(b))
    var out = List[Float64]()
    var i = 0
    while i < n:
        out.append(a[i] * b[i])
        i += 1
    return out
fn ew_div(a: List[Float64], b: List[Float64], eps: Float64 = 0.0) -> List[Float64]:
    var n = _same_len_min(len(a), len(b))
    var out = List[Float64]()
    var i = 0
    while i < n:
        var denom = b[i]
        if eps != 0.0:
            if denom >= 0.0:
                denom = denom + eps
            else:
                denom = denom - eps
        out.append(a[i] / denom)
        i += 1
    return out
fn ew_max(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var n = _same_len_min(len(a), len(b))
    var out = List[Float64]()
    var i = 0
    while i < n:
        var ai = a[i]
        var bi = b[i]
        var m = ai
        if bi > ai:
            m = bi
        out.append(m)
        i += 1
    return out
fn ew_min(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var n = _same_len_min(len(a), len(b))
    var out = List[Float64]()
    var i = 0
    while i < n:
        var ai = a[i]
        var bi = b[i]
        var m = ai
        if bi < ai:
            m = bi
        out.append(m)
        i += 1
    return out
fn ew_clamp(xs: List[Float64], lo: Float64, hi: Float64) -> List[Float64]:
    var out = List[Float64]()
    var i = 0
    while i < len(xs):
        var v = xs[i]
        if v < lo:
            v = lo
        if v > hi:
            v = hi
        out.append(v)
        i += 1
    return out
fn ew_relu(xs: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    var i = 0
    while i < len(xs):
        var v = xs[i]
        if v < 0.0:
            v = 0.0
        out.append(v)
        i += 1
    return out
fn ew_abs(xs: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    var i = 0
    while i < len(xs):
        var v = xs[i]
        if v < 0.0:
            v = -v
        out.append(v)
        i += 1
    return out
fn ew_sign(xs: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    var i = 0
    while i < len(xs):
        var v = xs[i]
        var s = 0.0
        if v > 0.0:
            s = 1.0
        else:
            if v < 0.0:
                s = -1.0
        out.append(s)
        i += 1
    return out
fn ew_sigmoid(xs: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    var i = 0
    while i < len(xs):
        var x = xs[i]
        if x >= 0.0:
            var z = exp(-x)
            out.append(1.0 / (1.0 + z))
        else:
            var z = exp(x)
            out.append(z / (1.0 + z))
        i += 1
    return out
fn ew_tanh(xs: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    var i = 0
    while i < len(xs):
        out.append(tanh(xs[i]))
        i += 1
    return out
fn ew_exp(xs: List[Float64]) -> List[Float64]:
    var out = List[Float64]()
    var i = 0
    while i < len(xs):
        out.append(exp(xs[i]))
        i += 1
    return out
fn ew_log(xs: List[Float64], eps: Float64 = 0.0) -> List[Float64]:
    var out = List[Float64]()
    var i = 0
    while i < len(xs):
        var v = xs[i]
        if eps != 0.0:
            if v >= 0.0:
                v = v + eps
            else:
                v = v - eps
        out.append(log(v))
        i += 1
    return out
fn ew_pow(xs: List[Float64], p: Int) -> List[Float64]:
    var out = List[Float64]()
    var i = 0
    while i < len(xs):
        var base = xs[i]
        if p == 0:
            out.append(1.0)
        else:
            var neg = False
            var e = p
            if e < 0:
                neg = True
                e = -e
            var acc = 1.0
            var k = 0
            while k < e:
                acc = acc * base
                k += 1
            if neg:
                acc = 1.0 / acc
            out.append(acc)
        i += 1
    return out

# ---------- Reductions & normalizations ----------
fn sum_list(xs: List[Float64]) -> Float64:
    var s = 0.0
    var i = 0
    while i < len(xs):
        s += xs[i]
        i += 1
    return s
fn mean_list(xs: List[Float64]) -> Float64:
    var n = len(xs)
    if n == 0:
        return 0.0
    return sum_list(xs) / Float64(n)
fn var_list(xs: List[Float64], ddof: Int = 0) -> Float64:
    var n = len(xs)
    var denom = n - ddof
    if n == 0 or denom <= 0:
        return 0.0
    var mu = mean_list(xs)
    var acc = 0.0
    var i = 0
    while i < n:
        var d = xs[i] - mu
        acc += d * d
        i += 1
    return acc / Float64(denom)
fn std_list(xs: List[Float64], ddof: Int = 0) -> Float64:
    return sqrt(var_list(xs, ddof))
fn normalize_l2(xs: List[Float64], eps: Float64 = 1e-12) -> List[Float64]:
    var s2 = 0.0
    var i = 0
    while i < len(xs):
        s2 += xs[i] * xs[i]
        i += 1
    var denom = sqrt(s2) + eps
    var out = List[Float64]()
    i = 0
    while i < len(xs):
        out.append(xs[i] / denom)
        i += 1
    return out
fn softmax(xs: List[Float64]) -> List[Float64]:
    var n = len(xs)
    if n == 0:
        return List[Float64]()
    var m_idx = argmax_index(xs)
    var m = xs[m_idx]
    var exps = List[Float64]()
    var sum_e = 0.0
    var i = 0
    while i < n:
        var e = exp(xs[i] - m)
        exps.append(e)
        sum_e += e
        i += 1
    var out = List[Float64]()
    i = 0
    while i < n:
        out.append(exps[i] / sum_e)
        i += 1
    return out

# ---------- Tensor-level wrappers (registry) ----------

fn add_f64(a: F64Tensor, b: F64Tensor) -> F64Tensor:
    #   return registry_add[Float64](a, b)
    return registry_add(a, b)

# A tiny built-in smoke test (optional hook)
fn __self_test__() -> Bool:
    var a = [1.0, -2.0, 3.0, -4.0]
    var b = [0.5, 0.5, 0.5, 0.5]
    var c = ew_add(a, b)
    if len(c) != 4: return False
    if argmax_index(a) != 2: return False
    if argmin_index(a) != 3: return False
    var sm = softmax([0.0, 0.0, 0.0])
    if len(sm) != 3: return False
    return True