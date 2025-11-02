# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.losses.ranking
# File:         src/momijo/learn/losses/ranking.mojo
#
# Description:
#   Ranking losses for Momijo Learn.
#   - hinge_loss: binary margin-based loss for classification.
#   - triplet_loss: metric learning with anchor/positive/negative embeddings.
#   - triplet_loss_from_dist: triplet loss using precomputed distances.
#   Backend-agnostic: implemented for numeric Lists. Tensor adapters are provided
#   for 1D and 2D cases (see bottom section).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# Notes:
#   - Labels accepted for hinge_loss: {0, 1} or {-1, +1}. Internally 0→-1.
#   - Reduction: "mean" (default) or "sum"; any other string → "mean".
#   - L2 distance uses a small Newton–Raphson sqrt to avoid extra deps.
#   - All functions are pure and raise no exceptions (use assert for shape checks).
#   - Tensor adapters assume 1D indexing t[i] for vectors and a row accessor
#     for 2D. Edit `_tensor2_row_to_list_f64` if your indexing differs.

from collections.list import List
from momijo.tensor.tensor import Tensor   # explicit, no wildcard

# -----------------------------------------------------------------------------
# Utilities (backend-agnostic)
# -----------------------------------------------------------------------------

@always_inline
fn _max0(x: Float64) -> Float64:
    if x > 0.0:
        return x
    return 0.0

@always_inline
fn _to_sign(label: Int) -> Int:
    # Map {0,1} -> {-1,+1}; keep {-1,+1} as-is; anything >0 => +1, else -1.
    if label == 0:
        return -1
    if label == 1:
        return 1
    if label > 0:
        return 1
    return -1

# Minimal sqrt without std math dependency (Newton–Raphson). Assumes x >= 0.
fn _sqrt_nr(x: Float64, iters: Int = 8) -> Float64:
    if x <= 0.0:
        return 0.0
    var guess = x
    if guess < 1.0:
        guess = 1.0
    var k = 0
    while k < iters:
        guess = 0.5 * (guess + x / guess)
        k = k + 1
    return guess

# True L2 distance between two Float64 vectors.
fn _l2_distance(x: List[Float64], y: List[Float64]) -> Float64:
    var n = len(x)
    assert(n == len(y))
    var s = 0.0
    var i = 0
    while i < n:
        var d = x[i] - y[i]
        s = s + d * d
        i = i + 1
    return _sqrt_nr(s, 8)

# Reduction over a list of values. "sum" → sum; anything else → mean.
fn _reduce(values: List[Float64], reduction: String) -> Float64:
    var n = len(values)
    if n == 0:
        return 0.0
    var s = 0.0
    var i = 0
    while i < n:
        s = s + values[i]
        i = i + 1
    if reduction == String("sum"):
        return s
    return s / Float64(n)

# -----------------------------------------------------------------------------
# Hinge Loss — List backend
# -----------------------------------------------------------------------------
# scores: prediction scores (length N)
# labels: {0,1} or {-1,+1} (length N)
# loss_i = max(0, margin - y_i * s_i)
# reduction: "mean" (default) or "sum"
fn hinge_loss(
    scores: List[Float64],
    labels: List[Int],
    margin: Float64 = 1.0,
    reduction: String = String("mean")
) -> Float64:
    var n = len(scores)
    assert(n == len(labels))
    var vals = List[Float64]()
    vals.reserve(n)

    var i = 0
    while i < n:
        var y = Float64(_to_sign(labels[i]))
        var s = scores[i]
        var v = margin - y * s
        vals.append(_max0(v))
        i = i + 1

    return _reduce(vals, reduction)

# -----------------------------------------------------------------------------
# Triplet Loss (embeddings) — List backend
# -----------------------------------------------------------------------------
# anchors/positives/negatives: List of embeddings (List[Float64]) with equal batch size
# d_ap = L2(a_i, p_i), d_an = L2(a_i, n_i)
# loss_i = max(0, d_ap - d_an + margin)
fn triplet_loss(
    anchors: List[List[Float64]],
    positives: List[List[Float64]],
    negatives: List[List[Float64]],
    margin: Float64 = 1.0,
    reduction: String = String("mean")
) -> Float64:
    var n = len(anchors)
    assert(n == len(positives))
    assert(n == len(negatives))

    var vals = List[Float64]()
    vals.reserve(n)

    var i = 0
    while i < n:
        var a = anchors[i]
        var p = positives[i]
        var nvec = negatives[i]
        assert(len(a) == len(p))
        assert(len(a) == len(nvec))

        var d_ap = _l2_distance(a, p)
        var d_an = _l2_distance(a, nvec)
        var v = d_ap - d_an + margin
        vals.append(_max0(v))
        i = i + 1

    return _reduce(vals, reduction)

# -----------------------------------------------------------------------------
# Triplet Loss (from precomputed distances) — List backend
# -----------------------------------------------------------------------------
# ap_dists: d(a_i, p_i) for i in [0..N)
# an_dists: d(a_i, n_i) for i in [0..N)
# loss_i = max(0, ap - an + margin)
fn triplet_loss_from_dist(
    ap_dists: List[Float64],
    an_dists: List[Float64],
    margin: Float64 = 1.0,
    reduction: String = String("mean")
) -> Float64:
    var n = len(ap_dists)
    assert(n == len(an_dists))

    var vals = List[Float64]()
    vals.reserve(n)

    var i = 0
    while i < n:
        var ap = ap_dists[i]
        var an = an_dists[i]
        var v = ap - an + margin
        vals.append(_max0(v))
        i = i + 1

    return _reduce(vals, reduction)

# =============================================================================
# Tensor adapters
# ============================================================================= 
# They convert Tensors to Lists and delegate to the List backend, keeping
# the loss implementations simple and testable.
#
# 1) 1D adapters (hinge_loss and triplet_loss_from_dist)
# 2) 2D adapter for triplet_loss with [N, D] embeddings

# --- common tiny helpers for Tensor indexing/length (1D) ----------------------

@always_inline
fn _tlen1[T](t: Tensor[T]) -> Int:
    # Replace with your project's canonical 1D length accessor if different.
    return len(t)

@always_inline
fn _tget1[T](t: Tensor[T], i: Int) -> T:
    # Replace with your project's canonical 1D indexing if different.
    return t[i]

fn _tensor1_to_list_f64(t: Tensor[Float64]) -> List[Float64]:
    var n = _tlen1(t)
    var out = List[Float64]()
    out.reserve(n)
    var i = 0
    while i < n:
        out.append(_tget1(t, i))
        i = i + 1
    return out

fn _tensor1_to_list_i32(t: Tensor[Int32]) -> List[Int]:
    var n = _tlen1(t)
    var out = List[Int]()
    out.reserve(n)
    var i = 0
    while i < n:
        out.append(Int(_tget1(t, i)))
        i = i + 1
    return out

# --- 1D Tensor adapters -------------------------------------------------------

fn hinge_loss(
    scores: Tensor[Float64],
    labels: Tensor[Int32],
    margin: Float64 = 1.0,
    reduction: String = String("mean")
) -> Float64:
    var s_list = _tensor1_to_list_f64(scores)
    var y_list = _tensor1_to_list_i32(labels)
    return hinge_loss(s_list, y_list, margin, reduction)

fn triplet_loss_from_dist(
    ap_dists: Tensor[Float64],
    an_dists: Tensor[Float64],
    margin: Float64 = 1.0,
    reduction: String = String("mean")
) -> Float64:
    var ap_list = _tensor1_to_list_f64(ap_dists)
    var an_list = _tensor1_to_list_f64(an_dists)
    return triplet_loss_from_dist(ap_list, an_list, margin, reduction)

# --- 2D helpers for row access ------------------------------------------------
# Convert row i of a 2D Float64 Tensor [N, D] into List[Float64].
# Default path uses value_slice(i) if your Tensor exposes it.
# If your Tensor uses t[i, j] indexing, replace the body with the commented
# alternative below (and add a _tcols2(t) helper accordingly).

fn _tensor2_row_to_list_f64(t: Tensor[Float64], i: Int) -> List[Float64]:
    # --- Path A: using value_slice(i) -> List[Float64] (preferred by Momijo arrays)
    var row = t.value_slice(i)        # adjust if your API returns Tensor; then iterate and push
    var n = len(row)
    var out = List[Float64]()
    out.reserve(n)
    var k = 0
    while k < n:
        out.append(row[k])
        k = k + 1
    return out
 
    # var d = _tcols2(t)              # implement a helper returning D
    # var out = List[Float64]()
    # out.reserve(d)
    # var j = 0
    # while j < d:
    #     out.append(t[i, j])      # or t.index2(i, j)
    #     j = j + 1
    # return out

# --- 2D Tensor adapter for triplet_loss --------------------------------------
# Input shapes: anchors[N, D], positives[N, D], negatives[N, D]
# Uses row-wise conversion to List[Float64] and delegates to the List backend.

fn triplet_loss(
    anchors: Tensor[Float64],
    positives: Tensor[Float64],
    negatives: Tensor[Float64],
    margin: Float64 = 1.0,
    reduction: String = String("mean")
) -> Float64:
    var n = len(anchors)                  # number of rows; align with your Tensor's len()
    assert(n == len(positives))
    assert(n == len(negatives))

    var a_list = List[List[Float64]]()
    var p_list = List[List[Float64]]()
    var n_list = List[List[Float64]]()
    a_list.reserve(n); p_list.reserve(n); n_list.reserve(n)

    var i = 0
    while i < n:
        a_list.append(_tensor2_row_to_list_f64(anchors, i))
        p_list.append(_tensor2_row_to_list_f64(positives, i))
        n_list.append(_tensor2_row_to_list_f64(negatives, i))
        i = i + 1

    return triplet_loss(a_list, p_list, n_list, margin, reduction)
