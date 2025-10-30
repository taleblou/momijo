# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.nn.regularizers
# File:         src/momijo/learn/nn/regularizers.mojo
#
# Description:
#   L1/L2 (weight decay) penalties and ElasticNet helpers.
#   - Scalar/List API (backend-free)
#   - Tensor API using Momijo facade: from momijo.tensor import tensor
#   - L2 uses 0.5 * λ * Σ w^2 by default (toggle via use_half_factor=false)
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# Internal helpers (List[Float64]) — backend-free
# -----------------------------------------------------------------------------

fn _sum_abs(xs: List[Float64]) -> Float64:
    var s = 0.0
    var n = Int(xs.size())
    var i = 0
    while i < n:
        var v = xs[i]
        if v < 0.0: v = -v
        s = s + v
        i = i + 1
    return s

fn _sum_sq(xs: List[Float64]) -> Float64:
    var s = 0.0
    var n = Int(xs.size())
    var i = 0
    while i < n:
        var v = xs[i]
        s = s + (v * v)
        i = i + 1
    return s

fn _sum_abs_nested(blocks: List[List[Float64]]) -> Float64:
    var s = 0.0
    var m = Int(blocks.size())
    var i = 0
    while i < m:
        s = s + _sum_abs(blocks[i])
        i = i + 1
    return s

fn _sum_sq_nested(blocks: List[List[Float64]]) -> Float64:
    var s = 0.0
    var m = Int(blocks.size())
    var i = 0
    while i < m:
        s = s + _sum_sq(blocks[i])
        i = i + 1
    return s

# -----------------------------------------------------------------------------
# Public API (List[Float64])
# -----------------------------------------------------------------------------

# L1: λ * Σ |w|
fn l1_penalty(params: List[Float64], weight: Float64) -> Float64:
    if weight == 0.0 or params.size() == 0:
        return 0.0
    return weight * _sum_abs(params)

# L2: 0.5 * λ * Σ w^2  (set use_half_factor=false for λ * Σ w^2)
fn l2_penalty(params: List[Float64], weight: Float64, use_half_factor: Bool = True) -> Float64:
    if weight == 0.0 or params.size() == 0:
        return 0.0
    var base = weight * _sum_sq(params)
    if use_half_factor:
        return 0.5 * base
    return base

# ElasticNet: λ1 * Σ |w| + 0.5 * λ2 * Σ w^2  (half-factor optional)
fn elastic_net_penalty(params: List[Float64], l1: Float64, l2: Float64, use_half_factor: Bool = True) -> Float64:
    var s = 0.0
    if l1 != 0.0:
        s = s + l1_penalty(params, l1)
    if l2 != 0.0:
        s = s + l2_penalty(params, l2, use_half_factor)
    return s

# -----------------------------------------------------------------------------
# Vectorized Tensor helpers (Float32/Float64) — uses elementwise ops & reductions
# NOTE: We provide dedicated overloads for Float32 and Float64 to avoid
#       dtype-casting ambiguity and to keep accumulation numerically stable.
#       Final returned scalar is Float64 (policy: accumulate in Float64).
# -----------------------------------------------------------------------------

# L1 sums

fn _l1_sum_f32(w: tensor.Tensor[Float32]) -> Float64:
    # Vectorized: sum(abs(w)) in Float32, widen to Float64 for the return
    var s32 = tensor.sum(tensor.abs(w))
    return Float64(s32)

fn _l1_sum_f64(w: tensor.Tensor[Float64]) -> Float64:
    var s64 = tensor.sum(tensor.abs(w))
    return s64

# L2 sums (Σ w^2)

fn _l2_sum_f32(w: tensor.Tensor[Float32]) -> Float64:
    var s32 = tensor.sum(tensor.square(w))
    return Float64(s32)

fn _l2_sum_f64(w: tensor.Tensor[Float64]) -> Float64:
    var s64 = tensor.sum(tensor.square(w))
    return s64

# Lists of tensors

fn _l1_sum_f32_list(blocks: List[tensor.Tensor[Float32]]) -> Float64:
    var s = 0.0
    var n = Int(blocks.size())
    var i = 0
    while i < n:
        s = s + _l1_sum_f32(blocks[i])
        i = i + 1
    return s

fn _l1_sum_f64_list(blocks: List[tensor.Tensor[Float64]]) -> Float64:
    var s = 0.0
    var n = Int(blocks.size())
    var i = 0
    while i < n:
        s = s + _l1_sum_f64(blocks[i])
        i = i + 1
    return s

fn _l2_sum_f32_list(blocks: List[tensor.Tensor[Float32]]) -> Float64:
    var s = 0.0
    var n = Int(blocks.size())
    var i = 0
    while i < n:
        s = s + _l2_sum_f32(blocks[i])
        i = i + 1
    return s

fn _l2_sum_f64_list(blocks: List[tensor.Tensor[Float64]]) -> Float64:
    var s = 0.0
    var n = Int(blocks.size())
    var i = 0
    while i < n:
        s = s + _l2_sum_f64(blocks[i])
        i = i + 1
    return s

# -----------------------------------------------------------------------------
# Tensor API (single tensor) — vectorized reductions
# -----------------------------------------------------------------------------

# L1: λ * Σ |w|
fn l1_penalty_tensor_f32(w: tensor.Tensor[Float32], weight: Float64) -> Float64:
    if weight == 0.0:
        return 0.0
    return weight * _l1_sum_f32(w)

fn l1_penalty_tensor_f64(w: tensor.Tensor[Float64], weight: Float64) -> Float64:
    if weight == 0.0:
        return 0.0
    return weight * _l1_sum_f64(w)

# L2: 0.5 * λ * Σ w^2  (set use_half_factor=false for λ * Σ w^2)
fn l2_penalty_tensor_f32(w: tensor.Tensor[Float32], weight: Float64, use_half_factor: Bool = True) -> Float64:
    if weight == 0.0:
        return 0.0
    var base = weight * _l2_sum_f32(w)
    if use_half_factor:
        return 0.5 * base
    return base

fn l2_penalty_tensor_f64(w: tensor.Tensor[Float64], weight: Float64, use_half_factor: Bool = True) -> Float64:
    if weight == 0.0:
        return 0.0
    var base = weight * _l2_sum_f64(w)
    if use_half_factor:
        return 0.5 * base
    return base

# ElasticNet: λ1 * Σ |w| + 0.5 * λ2 * Σ w^2
fn elastic_net_penalty_tensor_f32(w: tensor.Tensor[Float32], l1: Float64, l2: Float64, use_half_factor: Bool = True) -> Float64:
    var s = 0.0
    if l1 != 0.0:
        s = s + l1_penalty_tensor_f32(w, l1)
    if l2 != 0.0:
        s = s + l2_penalty_tensor_f32(w, l2, use_half_factor)
    return s

fn elastic_net_penalty_tensor_f64(w: tensor.Tensor[Float64], l1: Float64, l2: Float64, use_half_factor: Bool = True) -> Float64:
    var s = 0.0
    if l1 != 0.0:
        s = s + l1_penalty_tensor_f64(w, l1)
    if l2 != 0.0:
        s = s + l2_penalty_tensor_f64(w, l2, use_half_factor)
    return s

# -----------------------------------------------------------------------------
# Tensor API (lists of tensors) — vectorized reductions
# -----------------------------------------------------------------------------

# L1: λ * Σ |w|
fn l1_penalty_tensors_f32(blocks: List[tensor.Tensor[Float32]], weight: Float64) -> Float64:
    if weight == 0.0 or blocks.size() == 0:
        return 0.0
    return weight * _l1_sum_f32_list(blocks)

fn l1_penalty_tensors_f64(blocks: List[tensor.Tensor[Float64]], weight: Float64) -> Float64:
    if weight == 0.0 or blocks.size() == 0:
        return 0.0
    return weight * _l1_sum_f64_list(blocks)

# L2: 0.5 * λ * Σ w^2  (set use_half_factor=false for λ * Σ w^2)
fn l2_penalty_tensors_f32(blocks: List[tensor.Tensor[Float32]], weight: Float64, use_half_factor: Bool = True) -> Float64:
    if weight == 0.0 or blocks.size() == 0:
        return 0.0
    var base = weight * _l2_sum_f32_list(blocks)
    if use_half_factor:
        return 0.5 * base
    return base

fn l2_penalty_tensors_f64(blocks: List[tensor.Tensor[Float64]], weight: Float64, use_half_factor: Bool = True) -> Float64:
    if weight == 0.0 or blocks.size() == 0:
        return 0.0
    var base = weight * _l2_sum_f64_list(blocks)
    if use_half_factor:
        return 0.5 * base
    return base

# ElasticNet: λ1 * Σ |w| + 0.5 * λ2 * Σ w^2
fn elastic_net_penalty_tensors_f32(blocks: List[tensor.Tensor[Float32]], l1: Float64, l2: Float64, use_half_factor: Bool = True) -> Float64:
    var s = 0.0
    if l1 != 0.0:
        s = s + l1_penalty_tensors_f32(blocks, l1)
    if l2 != 0.0:
        s = s + l2_penalty_tensors_f32(blocks, l2, use_half_factor)
    return s

fn elastic_net_penalty_tensors_f64(blocks: List[tensor.Tensor[Float64]], l1: Float64, l2: Float64, use_half_factor: Bool = True) -> Float64:
    var s = 0.0
    if l1 != 0.0:
        s = s + l1_penalty_tensors_f64(blocks, l1)
    if l2 != 0.0:
        s = s + l2_penalty_tensors_f64(blocks, l2, use_half_factor)
    return s
