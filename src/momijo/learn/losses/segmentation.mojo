# Project:      Momijo
# Module:       learn.losses.segmentation
# File:         losses/segmentation.mojo
# Path:         src/momijo/learn/losses/segmentation.mojo
#
# Description:  Segmentation losses (Dice / IoU) for Momijo Learn.
#               Backend-agnostic reference implementation over List[Float64].
#               Replace with tensor-optimized kernels when momijo.tensor is ready.
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
#   - Functions (binary): dice_loss, iou_loss
#   - Functions (multi-class / multi-label, one-hot per class): dice_loss_mc, iou_loss_mc
#   - Helper utilities: _clamp01, _sum, _sum_mul, _validate_same_length

from collections.list import List

# -----------------------------
# Internal utilities
# -----------------------------

fn _clamp01(x: Float64) -> Float64:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

fn _sum(xs: List[Float64]) -> Float64:
    var s = 0.0
    for i in range(len(xs)):
        s = s + xs[i]
    return s

fn _sum_mul(a: List[Float64], b: List[Float64]) -> Float64:
    assert(len(a) == len(b))
    var s = 0.0
    for i in range(len(a)):
        s = s + (a[i] * b[i])
    return s

fn _validate_same_length(pred: List[Float64], target: List[Float64]):
    assert(len(pred) == len(target))

fn _clamp_list(mut xs: List[Float64]):
    for i in range(len(xs)):
        xs[i] = _clamp01(xs[i])

# -----------------------------
# Binary Dice / IoU
# -----------------------------
# Inputs:
#   pred   — probabilities in [0,1] (will be clamped)
#   target — ground truth in [0,1] or {0,1}
# Returns:
#   loss as Float64
#
# dice = (2 * intersection + eps) / (sum_p + sum_t + eps)
# loss = 1 - dice

fn dice_loss(pred: List[Float64], target: List[Float64], eps: Float64 = 1e-6) -> Float64:
    _validate_same_length(pred, target)

    # Copy and clamp (avoid mutating caller's buffers)
    var p = List[Float64](len(pred))
    var t = List[Float64](len(target))
    for i in range(len(pred)):
        p.push_back(_clamp01(pred[i]))
        t.push_back(_clamp01(target[i]))

    var inter = _sum_mul(p, t)
    var sum_p = _sum(p)
    var sum_t = _sum(t)

    var dice = (2.0 * inter + eps) / (sum_p + sum_t + eps)
    return 1.0 - dice

# IoU = (intersection + eps) / (union + eps)
# union = sum_p + sum_t - intersection
# loss = 1 - IoU

fn iou_loss(pred: List[Float64], target: List[Float64], eps: Float64 = 1e-6) -> Float64:
    _validate_same_length(pred, target)

    var p = List[Float64](len(pred))
    var t = List[Float64](len(target))
    for i in range(len(pred)):
        p.push_back(_clamp01(pred[i]))
        t.push_back(_clamp01(target[i]))

    var inter = _sum_mul(p, t)
    var sum_p = _sum(p)
    var sum_t = _sum(t)
    var union_val = sum_p + sum_t - inter

    var iou = (inter + eps) / (union_val + eps)
    return 1.0 - iou

# -----------------------------
# Multi-class / Multi-label (one-hot per class)
# -----------------------------
# Shapes (logical):
#   pred_c[k][i] and target_c[k][i] for class k and element i
# Inputs:
#   preds_per_class  — List of classes; each class is List[Float64] probs
#   targets_per_class — same shape, one-hot (or soft labels) per class
# reduction:
#   "mean" — average across classes (default)
#   "sum"  — sum across classes
#   "none" — returns per-class losses (use dice_loss_mc_vec / iou_loss_mc_vec)
#
# Notes:
#   - All class vectors must have identical length across classes and match targets.

fn dice_loss_mc_vec(preds_per_class: List[List[Float64]],
                    targets_per_class: List[List[Float64]],
                    eps: Float64 = 1e-6) -> List[Float64]:
    assert(len(preds_per_class) == len(targets_per_class))
    var losses = List[Float64]()
    for k in range(len(preds_per_class)):
        losses.push_back(dice_loss(preds_per_class[k], targets_per_class[k], eps))
    return losses

fn iou_loss_mc_vec(preds_per_class: List[List[Float64]],
                   targets_per_class: List[List[Float64]],
                   eps: Float64 = 1e-6) -> List[Float64]:
    assert(len(preds_per_class) == len(targets_per_class))
    var losses = List[Float64]()
    for k in range(len(preds_per_class)):
        losses.push_back(iou_loss(preds_per_class[k], targets_per_class[k], eps))
    return losses

fn _reduce_mean(xs: List[Float64]) -> Float64:
    var n = len(xs)
    if n == 0:
        return 0.0
    var s = 0.0
    for i in range(n):
        s = s + xs[i]
    return s / Float64(n)

fn dice_loss_mc(preds_per_class: List[List[Float64]],
                targets_per_class: List[List[Float64]],
                eps: Float64 = 1e-6,
                reduction: String = String("mean")) -> Float64:
    var vec = dice_loss_mc_vec(preds_per_class, targets_per_class, eps)
    if reduction == String("sum"):
        return _sum(vec)
    elif reduction == String("none"):
        # In "none", we return mean as a scalar placeholder to keep signature stable;
        # prefer using dice_loss_mc_vec if you need per-class values.
        return _reduce_mean(vec)
    else:
        return _reduce_mean(vec)

fn iou_loss_mc(preds_per_class: List[List[Float64]],
               targets_per_class: List[List[Float64]],
               eps: Float64 = 1e-6,
               reduction: String = String("mean")) -> Float64:
    var vec = iou_loss_mc_vec(preds_per_class, targets_per_class, eps)
    if reduction == String("sum"):
        return _sum(vec)
    elif reduction == String("none"):
        return _reduce_mean(vec)
    else:
        return _reduce_mean(vec)
