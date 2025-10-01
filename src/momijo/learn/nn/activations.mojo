# Project:      Momijo
# Module:       learn.nn.activations
# File:         nn/activations.mojo
# Path:         src/momijo/learn/nn/activations.mojo
#
# Description:  Activation functions for Momijo Learn.
#               Backend-agnostic scalar, 1D, and simple 2D implementations with
#               numerically stable softmax and tanh-based GELU approximation.
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
#   - Provided activations (scalar & vector forms unless noted):
#     relu, leaky_relu, relu6, sigmoid, tanh, gelu (tanh-approx), silu(swish),
#     elu, selu, softmax (1D + 2D with dim), hard_sigmoid, hard_swish
#   - Softmax is numerically stable via max-shift.
#   - Uses internal exp/tanh approximations; replace with tensor kernels later.
#   - Overload scheme:
#       f(x: Float64) -> Float64
#       f(xs: List[Float64]) -> List[Float64]
#       softmax(x2d: List[List[Float64]], dim: Int) -> List[List[Float64]]

from collections.list import List

# --------------------------------------------
# Internal helpers (backend-free approximations)
# --------------------------------------------

fn _clamp(x: Float64, lo: Float64, hi: Float64) -> Float64:
    var v = x
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v

# Reasonable exp approximation (bounded input; repeated squaring)
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

# Elementwise map for List[Float64]
fn _map(xs: List[Float64], f) -> List[Float64]:
    var out = List[Float64]()
    out.reserve(Int(xs.size()))
    var i = 0
    while i < Int(xs.size()):
        out.push_back(f(xs[i]))
        i = i + 1
    return out

# --------------------------------------------
# ReLU family
# --------------------------------------------

fn relu(x: Float64) -> Float64:
    if x > 0.0:
        return x
    return 0.0

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

# --------------------------------------------
# Sigmoid / Tanh / SiLU (Swish)
# --------------------------------------------

fn sigmoid(x: Float64) -> Float64:
    var z = _exp_approx(-x)
    return 1.0 / (1.0 + z)

fn sigmoid(xs: List[Float64]) -> List[Float64]:
    return _map(xs, sigmoid)

fn tanh(x: Float64) -> Float64:
    return _tanh_approx(x)

fn tanh(xs: List[Float64]) -> List[Float64]:
    return _map(xs, tanh)

# SiLU / Swish: x * sigmoid(x)
fn silu(x: Float64) -> Float64:
    return x * sigmoid(x)

fn silu(xs: List[Float64]) -> List[Float64]:
    return _map(xs, silu)

# --------------------------------------------
# GELU (tanh-based approximation)
# gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x^3)))
# --------------------------------------------

fn gelu(x: Float64) -> Float64:
    var s = 0.7978845608028654  # sqrt(2/pi)
    var x3 = x * x * x
    var inner = s * (x + 0.044715 * x3)
    var t = _tanh_approx(inner)
    return 0.5 * x * (1.0 + t)

fn gelu(xs: List[Float64]) -> List[Float64]:
    return _map(xs, gelu)

# --------------------------------------------
# ELU / SELU
# --------------------------------------------

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

# SELU constants (from the original paper)
# lambda ≈ 1.0507009873554805, alpha ≈ 1.6732632423543772
fn selu(x: Float64) -> Float64:
    var lmbd = 1.0507009873554805
    var alpha = 1.6732632423543772
    if x >= 0.0:
        return lmbd * x
    return lmbd * (alpha * (_exp_approx(x) - 1.0))

fn selu(xs: List[Float64]) -> List[Float64]:
    return _map(xs, selu)

# --------------------------------------------
# Hard variants
# --------------------------------------------

fn hard_sigmoid(x: Float64) -> Float64:
    # 0 if x < -3, 1 if x > 3, else x/6 + 0.5
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

# --------------------------------------------
# Softmax (stable)
#   - 1D: softmax(List[Float64])
#   - 2D: softmax(List[List[Float64]], dim)
#       dim == 1 (default/-1): row-wise over last axis
#       dim == 0: column-wise over first axis
# --------------------------------------------

fn _softmax_1d(xs: List[Float64]) -> List[Float64]:
    var n = Int(xs.size())
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

# Scalar convenience (softmax over single element is 1.0)
fn softmax(x: Float64, dim: Int = -1) -> Float64:
    return 1.0

# 2D softmax with dimension control
fn softmax(x2d: List[List[Float64]], dim: Int = -1) -> List[List[Float64]]:
    var rows = Int(x2d.size())
    if rows == 0:
        return List[List[Float64]]()

    # Determine last-axis (dim == -1 or 1) -> row-wise
    if dim == -1 or dim == 1:
        var out = List[List[Float64]]()
        out.reserve(rows)
        var r = 0
        while r < rows:
            out.push_back(_softmax_1d(x2d[r]))
            r = r + 1
        return out

    # dim == 0 -> column-wise (apply softmax down each column)
    if dim == 0:
        # Determine max row length (ragged safety: treat missing as -inf by skipping)
        var cols = 0
        var r = 0
        while r < rows:
            var len = Int(x2d[r].size())
            if len > cols:
                cols = len
            r = r + 1

        var out2 = List[List[Float64]]()
        out2.reserve(rows)
        r = 0
        while r < rows:
            var row_out = List[Float64]()
            row_out.reserve(Int(x2d[r].size()))
            var c = 0
            while c < Int(x2d[r].size()):
                row_out.push_back(0.0)
                c = c + 1
            out2.push_back(row_out)
            r = r + 1

        var c = 0
        while c < cols:
            # collect column c
            var col_vals = List[Float64]()
            var row_ids = List[Int]()
            row_ids.reserve(rows)
            var rr = 0
            while rr < rows:
                if c < Int(x2d[rr].size()):
                    col_vals.push_back(x2d[rr][c])
                    row_ids.push_back(rr)
                rr = rr + 1
            # softmax on collected column
            var col_sm = _softmax_1d(col_vals)
            # scatter back
            var k = 0
            while k < Int(col_sm.size()):
                var rid = row_ids[k]
                out2[rid][c] = col_sm[k]
                k = k + 1
            c = c + 1

        return out2

    # Unsupported dim -> default to row-wise
    return softmax(x2d, -1)
