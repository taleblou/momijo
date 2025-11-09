# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.losses.segmentation
# File:         src/momijo/learn/losses/segmentation.mojo
#
# Description:
#   Segmentation losses (Dice / IoU) for Momijo Learn.
#   Part A: backend-agnostic reference over List[Float32].
#   Part B: tensor-optimized overloads using momijo.tensor (Float32/Float32).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# Notes:
#   - Binary: dice_loss, iou_loss
#   - Multi-class: dice_loss_mc, iou_loss_mc
#   - Vector (per-class): dice_loss_mc_vec, iou_loss_mc_vec
#   - Helpers: _clamp01, _sum, _sum_mul, _reduce_mean, _validate_same_length
#   - Tensor overloads rely on: clip, sum, elementwise *, scalar math.

from collections.list import List
from momijo.tensor import tensor   # <— enables tensor overloads (Float32/Float32)

# =============================================================================
# Part A — List[Float32] reference implementation (backend-agnostic)
# =============================================================================

@always_inline
fn _clamp01(x: Float32) -> Float32:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

@always_inline
fn _sum(xs: List[Float32]) -> Float32:
    var s = 0.0
    var i = 0
    while i < len(xs):
        s = s + xs[i]
        i = i + 1
    return s

@always_inline
fn _sum_mul(a: List[Float32], b: List[Float32]) -> Float32:
    assert(len(a) == len(b))
    var s = 0.0
    var i = 0
    while i < len(a):
        s = s + (a[i] * b[i])
        i = i + 1
    return s

@always_inline
fn _validate_same_length(pred: List[Float32], target: List[Float32]):
    assert(len(pred) == len(target))

# ---- Binary Dice (List) ------------------------------------------------------

fn dice_loss(pred: List[Float32], target: List[Float32], eps: Float32 = 1e-6) -> Float32:
    _validate_same_length(pred, target)

    # Clamp copies (no mutation of caller buffers)
    var p = List[Float32]()
    var t = List[Float32]()
    var i = 0
    while i < len(pred):
        p.append(_clamp01(pred[i]))
        t.append(_clamp01(target[i]))
        i = i + 1

    var inter = _sum_mul(p, t)
    var sum_p = _sum(p)
    var sum_t = _sum(t)
    var dice = (2.0 * inter + eps) / (sum_p + sum_t + eps)
    return 1.0 - dice

# ---- Binary IoU (List) -------------------------------------------------------

fn iou_loss(pred: List[Float32], target: List[Float32], eps: Float32 = 1e-6) -> Float32:
    _validate_same_length(pred, target)

    var p = List[Float32]()
    var t = List[Float32]()
    var i = 0
    while i < len(pred):
        p.append(_clamp01(pred[i]))
        t.append(_clamp01(target[i]))
        i = i + 1

    var inter = _sum_mul(p, t)
    var sum_p = _sum(p)
    var sum_t = _sum(t)
    var union_val = sum_p + sum_t - inter
    var iou = (inter + eps) / (union_val + eps)
    return 1.0 - iou

# ---- Multi-class / Multi-label (List) ----------------------------------------

@always_inline
fn _reduce_mean(xs: List[Float32]) -> Float32:
    var n = len(xs)
    if n == 0: return 0.0
    var s = 0.0
    var i = 0
    while i < n:
        s = s + xs[i]
        i = i + 1
    return s / Float32(n)

fn dice_loss_mc_vec(preds_per_class: List[List[Float32]],
                    targets_per_class: List[List[Float32]],
                    eps: Float32 = 1e-6) -> List[Float32]:
    assert(len(preds_per_class) == len(targets_per_class))
    var losses = List[Float32]()
    var k = 0
    while k < len(preds_per_class):
        losses.append(dice_loss(preds_per_class[k], targets_per_class[k], eps))
        k = k + 1
    return losses

fn iou_loss_mc_vec(preds_per_class: List[List[Float32]],
                   targets_per_class: List[List[Float32]],
                   eps: Float32 = 1e-6) -> List[Float32]:
    assert(len(preds_per_class) == len(targets_per_class))
    var losses = List[Float32]()
    var k = 0
    while k < len(preds_per_class):
        losses.append(iou_loss(preds_per_class[k], targets_per_class[k], eps))
        k = k + 1
    return losses

fn dice_loss_mc(preds_per_class: List[List[Float32]],
                targets_per_class: List[List[Float32]],
                eps: Float32 = 1e-6,
                reduction: String = "mean") -> Float32:
    var vec = dice_loss_mc_vec(preds_per_class, targets_per_class, eps)
    if reduction == "sum":
        return _sum(vec)
    elif reduction == "none":
        return _reduce_mean(vec)  # placeholder; use *_mc_vec for per-class
    else:
        return _reduce_mean(vec)

fn iou_loss_mc(preds_per_class: List[List[Float32]],
               targets_per_class: List[List[Float32]],
               eps: Float32 = 1e-6,
               reduction: String = "mean") -> Float32:
    var vec = iou_loss_mc_vec(preds_per_class, targets_per_class, eps)
    if reduction == "sum":
        return _sum(vec)
    elif reduction == "none":
        return _reduce_mean(vec)
    else:
        return _reduce_mean(vec)

# =============================================================================
# Part B — Tensor overloads (Float32/Float32)
# Assumptions (per Momijo standards & tensor module):
# - clip(x, lo, hi) exists for Tensor[Float32]/Tensor[Float32]
# - sum(x) returns scalar of same element type
# - Elementwise multiply uses '*'
# - Scalar math works with Float32/Float32
# =============================================================================

# ---- Binary Dice (Tensor Float32) --------------------------------------------

fn dice_loss(pred: tensor.Tensor[Float32],
             target: tensor.Tensor[Float32],
             eps: Float32 = 1e-6) -> Float32:
    var p = tensor.clip(pred, 0.0, 1.0)
    var t = tensor.clip(target, 0.0, 1.0)
    var inter = tensor.sum(p * t)
    var sum_p = tensor.sum(p)
    var sum_t = tensor.sum(t)
    var dice = (2.0 * inter + eps) / (sum_p + sum_t + eps)
    return 1.0 - dice

# ---- Binary Dice (Tensor Float32) --------------------------------------------

fn dice_loss(pred: tensor.Tensor[Float32],
             target: tensor.Tensor[Float32],
             eps: Float32 = 1e-6) -> Float32:
    var p = tensor.clip(pred, Float32(0.0), Float32(1.0))
    var t = tensor.clip(target, Float32(0.0), Float32(1.0))
    var inter = Float32(tensor.sum(p * t))
    var sum_p = Float32(tensor.sum(p))
    var sum_t = Float32(tensor.sum(t))
    var dice = (2.0 * inter + eps) / (sum_p + sum_t + eps)
    return 1.0 - dice

# ---- Binary IoU (Tensor Float32) --------------------------------------------

fn iou_loss(pred: tensor.Tensor[Float32],
            target: tensor.Tensor[Float32],
            eps: Float32 = 1e-6) -> Float32:
    var p = tensor.clip(pred, 0.0, 1.0)
    var t = tensor.clip(target, 0.0, 1.0)
    var inter = tensor.sum(p * t)
    var sum_p = tensor.sum(p)
    var sum_t = tensor.sum(t)
    var union_val = sum_p + sum_t - inter
    var iou = (inter + eps) / (union_val + eps)
    return 1.0 - iou

# ---- Binary IoU (Tensor Float32) --------------------------------------------

fn iou_loss(pred: tensor.Tensor[Float32],
            target: tensor.Tensor[Float32],
            eps: Float32 = 1e-6) -> Float32:
    var p = tensor.clip(pred, Float32(0.0), Float32(1.0))
    var t = tensor.clip(target, Float32(0.0), Float32(1.0))
    var inter = Float32(tensor.sum(p * t))
    var sum_p = Float32(tensor.sum(p))
    var sum_t = Float32(tensor.sum(t))
    var union_val = sum_p + sum_t - inter
    var iou = (inter + eps) / (union_val + eps)
    return 1.0 - iou

# ---- Multi-class (Tensor Float32) --------------------------------------------

fn dice_loss_mc_vec(preds_per_class: List[tensor.Tensor[Float32]],
                    targets_per_class: List[tensor.Tensor[Float32]],
                    eps: Float32 = 1e-6) -> List[Float32]:
    assert(len(preds_per_class) == len(targets_per_class))
    var losses = List[Float32]()
    var k = 0
    while k < len(preds_per_class):
        losses.append(dice_loss(preds_per_class[k], targets_per_class[k], eps))
        k = k + 1
    return losses

fn iou_loss_mc_vec(preds_per_class: List[tensor.Tensor[Float32]],
                   targets_per_class: List[tensor.Tensor[Float32]],
                   eps: Float32 = 1e-6) -> List[Float32]:
    assert(len(preds_per_class) == len(targets_per_class))
    var losses = List[Float32]()
    var k = 0
    while k < len(preds_per_class):
        losses.append(iou_loss(preds_per_class[k], targets_per_class[k], eps))
        k = k + 1
    return losses

fn dice_loss_mc(preds_per_class: List[tensor.Tensor[Float32]],
                targets_per_class: List[tensor.Tensor[Float32]],
                eps: Float32 = 1e-6,
                reduction: String = "mean") -> Float32:
    var vec = dice_loss_mc_vec(preds_per_class, targets_per_class, eps)
    if reduction == "sum":
        return _sum(vec)
    elif reduction == "none":
        return _reduce_mean(vec)
    else:
        return _reduce_mean(vec)

fn iou_loss_mc(preds_per_class: List[tensor.Tensor[Float32]],
               targets_per_class: List[tensor.Tensor[Float32]],
               eps: Float32 = 1e-6,
               reduction: String = "mean") -> Float32:
    var vec = iou_loss_mc_vec(preds_per_class, targets_per_class, eps)
    if reduction == "sum":
        return _sum(vec)
    elif reduction == "none":
        return _reduce_mean(vec)
    else:
        return _reduce_mean(vec)

# ---- Multi-class (Tensor Float32) --------------------------------------------

fn dice_loss_mc_vec(preds_per_class: List[tensor.Tensor[Float32]],
                    targets_per_class: List[tensor.Tensor[Float32]],
                    eps: Float32 = 1e-6) -> List[Float32]:
    assert(len(preds_per_class) == len(targets_per_class))
    var losses = List[Float32]()
    var k = 0
    while k < len(preds_per_class):
        losses.append(dice_loss(preds_per_class[k], targets_per_class[k], eps))
        k = k + 1
    return losses

fn iou_loss_mc_vec(preds_per_class: List[tensor.Tensor[Float32]],
                   targets_per_class: List[tensor.Tensor[Float32]],
                   eps: Float32 = 1e-6) -> List[Float32]:
    assert(len(preds_per_class) == len(targets_per_class))
    var losses = List[Float32]()
    var k = 0
    while k < len(preds_per_class):
        losses.append(iou_loss(preds_per_class[k], targets_per_class[k], eps))
        k = k + 1
    return losses

fn dice_loss_mc(preds_per_class: List[tensor.Tensor[Float32]],
                targets_per_class: List[tensor.Tensor[Float32]],
                eps: Float32 = 1e-6,
                reduction: String = "mean") -> Float32:
    var vec = dice_loss_mc_vec(preds_per_class, targets_per_class, eps)
    if reduction == "sum":
        return _sum(vec)
    elif reduction == "none":
        return _reduce_mean(vec)
    else:
        return _reduce_mean(vec)

fn iou_loss_mc(preds_per_class: List[tensor.Tensor[Float32]],
               targets_per_class: List[tensor.Tensor[Float32]],
               eps: Float32 = 1e-6,
               reduction: String = "mean") -> Float32:
    var vec = iou_loss_mc_vec(preds_per_class, targets_per_class, eps)
    if reduction == "sum":
        return _sum(vec)
    elif reduction == "none":
        return _reduce_mean(vec)
    else:
        return _reduce_mean(vec)
