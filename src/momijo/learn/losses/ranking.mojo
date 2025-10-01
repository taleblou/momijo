# Project:      Momijo
# Module:       learn.losses.ranking
# File:         losses/ranking.mojo
# Path:         src/momijo/learn/losses/ranking.mojo
#
# Description:  Ranking losses for Momijo Learn. Provides binary hinge loss for
#               margin-based classification and Triplet Loss for metric learning.
#               Backend-agnostic: works on numeric Lists; later you can wire it
#               to momijo.tensor for vectorized ops.
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
#   - Functions:
#     * hinge_loss(scores, labels, margin=1.0, reduction="mean")
#       labels in {0,1} or {-1,1}; maps 0→-1 internally.
#     * triplet_loss(anchors, positives, negatives, margin=1.0, reduction="mean")
#       anchors/positives/negatives are batched embeddings: List[List[Float64]]
#       distance: L2; loss_i = max(0, d(a,p) - d(a,n) + margin)
#     * triplet_loss_from_dist(ap_dists, an_dists, margin=1.0, reduction="mean")
#       uses precomputed distances d(a,p) and d(a,n).
#   - Input checks (sizes) via simple asserts.
#   - Reduction: "mean" (default) or "sum".

from collections.list import List

# ------------------------------------------------------------
# Utilities (backend-agnostic)
# ------------------------------------------------------------

fn _max0(x: Float64) -> Float64:
    if x > 0.0:
        return x
    return 0.0

fn _to_sign(label: Int) -> Int:
    # Map {0,1} -> {-1, +1}; keep {-1, +1} as-is; anything >0 => +1, else -1.
    if label == 0:
        return -1
    if label == 1:
        return 1
    if label > 0:
        return 1
    return -1

fn _l2_distance(x: List[Float64], y: List[Float64]) -> Float64:
    var n = Int(x.size())
    assert(n == Int(y.size()))
    var s = 0.0
    var i = 0
    while i < n:
        var d = x[i] - y[i]
        s = s + d * d
        i = i + 1
    # If you later add sqrt(), you can return sqrt(s).
    # For ranking margin it's valid to use squared L2 as distance as well,
    # but common TripletLoss uses L2 (sqrt). We'll keep true L2 here by a simple Newton step if needed.
    # To avoid dependency on math.sqrt, we'll implement a minimal sqrt when >=0:
    if s <= 0.0:
        return 0.0
    # Newton-Raphson sqrt with a few iterations (safety, no std math dependency).
    var z = s
    var guess = s
    if guess < 1.0:
        guess = 1.0
    var k = 0
    while k < 8:
        guess = 0.5 * (guess + z / guess)
        k = k + 1
    return guess

fn _reduce(values: List[Float64], reduction: String) -> Float64:
    var n = Int(values.size())
    if n == 0:
        return 0.0
    var s = 0.0
    var i = 0
    while i < n:
        s = s + values[i]
        i = i + 1
    if reduction == String("sum"):
        return s
    # default: mean
    return s / Float64(n)

# ------------------------------------------------------------
# Hinge Loss
# ------------------------------------------------------------
# scores: prediction scores (margin scores), length N
# labels: Int labels in {0,1} or {-1,1}, length N
# loss_i = max(0, margin - y_i * s_i)
# reduction: "mean" (default) or "sum"
fn hinge_loss(
    scores: List[Float64],
    labels: List[Int],
    margin: Float64 = 1.0,
    reduction: String = String("mean")
) -> Float64:
    var n = Int(scores.size())
    assert(n == Int(labels.size()))
    var vals = List[Float64]()
    vals.reserve(n)
    var i = 0
    while i < n:
        var y = Float64(_to_sign(labels[i]))
        var s = scores[i]
        var v = margin - y * s
        vals.push_back(_max0(v))
        i = i + 1
    return _reduce(vals, reduction)

# ------------------------------------------------------------
# Triplet Loss (embeddings)
# ------------------------------------------------------------
# anchors/positives/negatives: List of embeddings (List[Float64]) of equal batch size
# d_ap = L2(a_i, p_i), d_an = L2(a_i, n_i)
# loss_i = max(0, d_ap - d_an + margin)
fn triplet_loss(
    anchors: List[List[Float64]],
    positives: List[List[Float64]],
    negatives: List[List[Float64]],
    margin: Float64 = 1.0,
    reduction: String = String("mean")
) -> Float64:
    var n = Int(anchors.size())
    assert(n == Int(positives.size()))
    assert(n == Int(negatives.size()))

    var vals = List[Float64]()
    vals.reserve(n)

    var i = 0
    while i < n:
        var a = anchors[i]
        var p = positives[i]
        var nvec = negatives[i]
        assert(Int(a.size()) == Int(p.size()))
        assert(Int(a.size()) == Int(nvec.size()))
        var d_ap = _l2_distance(a, p)
        var d_an = _l2_distance(a, nvec)
        var v = d_ap - d_an + margin
        vals.push_back(_max0(v))
        i = i + 1

    return _reduce(vals, reduction)

# ------------------------------------------------------------
# Triplet Loss (from precomputed distances)
# ------------------------------------------------------------
# ap_dists: d(a_i, p_i) for i in [0..N)
# an_dists: d(a_i, n_i) for i in [0..N)
# loss_i = max(0, ap - an + margin)
fn triplet_loss_from_dist(
    ap_dists: List[Float64],
    an_dists: List[Float64],
    margin: Float64 = 1.0,
    reduction: String = String("mean")
) -> Float64:
    var n = Int(ap_dists.size())
    assert(n == Int(an_dists.size()))
    var vals = List[Float64]()
    vals.reserve(n)

    var i = 0
    while i < n:
        var ap = ap_dists[i]
        var an = an_dists[i]
        var v = ap - an + margin
        vals.push_back(_max0(v))
        i = i + 1

    return _reduce(vals, reduction)
