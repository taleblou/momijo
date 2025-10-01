# Project:      Momijo
# Module:       learn.nn.regularizers
# File:         nn/regularizers.mojo
# Path:         src/momijo/learn/nn/regularizers.mojo
#
# Description:  L1/L2 (weight decay) penalties and ElasticNet helper for models.
#               Backend-agnostic scalar implementation that sums over parameter
#               lists. You can later wire these to tensor params by exposing
#               flat Float64 views of weights.
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
#   - Functions (scalar lists): l1_penalty, l2_penalty, elastic_net_penalty
#   - Helpers: _sum_abs, _sum_sq
#   - Convention: L2 penalty uses 0.5 * lambda * Σ w^2 (common in literature).
#     If you prefer TF/PyTorch-style with no 0.5 factor, set use_half_factor=False.

from collections.list import List

# -----------------------------------------------------------------------------
# Internal helpers (scalar-only for now)
# -----------------------------------------------------------------------------

fn _sum_abs(xs: List[Float64]) -> Float64:
    var s = 0.0
    for i in range(xs.size()):
        var v = xs[i]
        if v < 0.0:
            v = -v
        s = s + v
    return s

fn _sum_sq(xs: List[Float64]) -> Float64:
    var s = 0.0
    for i in range(xs.size()):
        var v = xs[i]
        s = s + (v * v)
    return s

# Flatten and accumulate for nested parameter blocks (e.g., per-layer arrays)
fn _sum_abs_nested(blocks: List[List[Float64]]) -> Float64:
    var s = 0.0
    for b in range(blocks.size()):
        s = s + _sum_abs(blocks[b])
    return s

fn _sum_sq_nested(blocks: List[List[Float64]]) -> Float64:
    var s = 0.0
    for b in range(blocks.size()):
        s = s + _sum_sq(blocks[b])
    return s

# -----------------------------------------------------------------------------
# Public API (scalar lists)
# -----------------------------------------------------------------------------

/// L1 penalty:  λ * Σ |w|
fn l1_penalty(params: List[Float64], weight: Float64) -> Float64:
    if weight == 0.0 or params.size() == 0:
        return 0.0
    return weight * _sum_abs(params)

/// L2 penalty (weight decay):  0.5 * λ * Σ w^2  (set use_half_factor=false for λ * Σ w^2)
fn l2_penalty(
    params: List[Float64],
    weight: Float64,
    use_half_factor: Bool = True
) -> Float64:
    if weight == 0.0 or params.size() == 0:
        return 0.0
    var base = weight * _sum_sq(params)
    if use_half_factor:
        return 0.5 * base
    return base

/// ElasticNet:  λ1 * Σ |w| + 0.5 * λ2 * Σ w^2  (half-factor optional)
fn elastic_net_penalty(
    params: List[Float64],
    l1: Float64,
    l2: Float64,
    use_half_factor: Bool = True
) -> Float64:
    var s = 0.0
    if l1 != 0.0:
        s = s + l1_penalty(params, l1)
    if l2 != 0.0:
        s = s + l2_penalty(params, l2, use_half_factor)
    return s

# -----------------------------------------------------------------------------
# Overloads for nested lists (e.g., layer-wise parameter blocks)
# -----------------------------------------------------------------------------

/// L1 penalty over nested parameter blocks
fn l1_penalty_nested(blocks: List[List[Float64]], weight: Float64) -> Float64:
    if weight == 0.0 or blocks.size() == 0:
        return 0.0
    return weight * _sum_abs_nested(blocks)

/// L2 penalty over nested parameter blocks
fn l2_penalty_nested(
    blocks: List[List[Float64]],
    weight: Float64,
    use_half_factor: Bool = True
) -> Float64:
    if weight == 0.0 or blocks.size() == 0:
        return 0.0
    var base = weight * _sum_sq_nested(blocks)
    if use_half_factor:
        return 0.5 * base
    return base

/// ElasticNet over nested parameter blocks
fn elastic_net_penalty_nested(
    blocks: List[List[Float64]],
    l1: Float64,
    l2: Float64,
    use_half_factor: Bool = True
) -> Float64:
    var s = 0.0
    if l1 != 0.0:
        s = s + l1_penalty_nested(blocks, l1)
    if l2 != 0.0:
        s = s + l2_penalty_nested(blocks, l2, use_half_factor)
    return s
