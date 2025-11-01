# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/metrics/metrics.mojo
# Description: Minimal accuracy approximation via soft argmax.

from collections.list import List
from momijo.tensor.tensor import Tensor
from momijo.learn.losses.losses import softmax

fn _row_sum(x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
    var C = x.shape()[1]
    return tensor.matmul(x, tensor.ones([C,1]))

fn accuracy_approx(logits: tensor.Tensor[Float64], target_onehot: tensor.Tensor[Float64]) -> Float64:
    var probs = softmax(logits)
    var k = 64.0
    var w = tensor.exp(probs * k)
    var p = w / _row_sum(w)  # [N,C], ~one-hot
    var correct = _row_sum(p * target_onehot)
    var N = probs.shape()[0]
    var total = tensor.matmul(tensor.ones([N,1]).transpose(), correct)
    return total.item() / Float64(N)



# --- helpers ---------------------------------------------------------

fn _scalar(v: Float64) -> Tensor[Float64]:
    var data = List[Float64]()
    data.append(v)
    return Tensor[Float64](data, [1])

fn _reduce_mean(x: Tensor[Float64]) -> Tensor[Float64]:
    return x.mean(None, False)

fn _reduce_sum(x: Tensor[Float64]) -> Tensor[Float64]:
    return x.sum(None, False)

# --- classification metrics -----------------------------------------

# accuracy: if logits, we take argmax over 'axis'; if already labels, pass same axis length 1.
fn accuracy(pred: Tensor[Float64], target: Tensor[Float64], axis: Int = -1) -> Tensor[Float64]:
    var yhat = pred.argmax(Optional[Int](axis))
    var eq = yhat.equal(target)
    return _reduce_mean(eq)

# precision, recall, f1 for multi-class by macro-averaging.
fn precision_recall_f1(pred: Tensor[Float64], target: Tensor[Float64], axis: Int = -1, eps: Float64 = 1e-12) -> Tensor[Float64]:
    var yhat = pred.argmax(Optional[Int](axis))
    var y = target
    # Assume labels are in [0..C-1]. Determine C by max label across y and yhat.
    var ymax = y.max(None, False)
    var hmax = yhat.max(None, False)
    var c_max = ymax.add(hmax).max(None, False)._data[0]
    var C = Int(c_max) + 1 if c_max >= 0.0 else 1

    var prf = List[Float64]()
    var c = 0
    while c < C:
        var cls = _scalar(Float64(c))
        var tp = _reduce_sum( yhat.equal(cls).logical_and(y.equal(cls)) )._data[0]
        var fp = _reduce_sum( yhat.equal(cls).logical_and(y.ne(cls)) )._data[0]
        var fn = _reduce_sum( yhat.ne(cls).logical_and(y.equal(cls)) )._data[0]
        var prec = tp / (tp + fp + eps)
        var rec  = tp / (tp + fn + eps)
        var f1 = 2.0 * prec * rec / (prec + rec + eps)
        prf.append(prec); prf.append(rec); prf.append(f1)
        c += 1

    # macro-average
    var pc = 0.0; var rc = 0.0; var fc = 0.0
    var i = 0
    while i < len(prf):
        pc = pc + prf[i]; rc = rc + prf[i+1]; fc = fc + prf[i+2]
        i += 3
    var denom = Float64(C)
    var out = List[Float64]()
    out.append(pc / denom); out.append(rc / denom); out.append(fc / denom)
    return Tensor[Float64](out, [3])

# --- segmentation / set metrics -------------------------------------

# IoU (Jaccard) for binary masks. For multi-class, call per-class and average.
fn iou(pred_mask: Tensor[Float64], target_mask: Tensor[Float64], eps: Float64 = 1e-6) -> Tensor[Float64]:
    var p = pred_mask
    var t = target_mask
    var inter = _reduce_sum(p.logical_and(t))
    var uni = _reduce_sum(p.logical_or(t))
    var score = inter.add(_scalar(eps)).div(uni.add(_scalar(eps)))
    return score

# --- image quality ---------------------------------------------------

# Peak Signal-to-Noise Ratio; inputs are in [0,1] or specify max_val.
fn psnr(img: Tensor[Float64], ref: Tensor[Float64], max_val: Float64 = 1.0, eps: Float64 = 1e-12) -> Tensor[Float64]:
    var diff = img.sub(ref)
    var mse = _reduce_mean(diff.mul(diff))._data[0]
    var denom = mse + eps
    var ratio = (max_val * max_val) / denom
    # 10 * log10(ratio) = 10 * ln(ratio)/ln(10)
    var val = 10.0 * (Float64.log(ratio) / 2.302585092994046)
    return _scalar(val)

# Structural Similarity Index (SSIM) — simple luminance/contrast/structure formulation.
# Assumes inputs normalized to [0,1]. Windowing can be added later.
fn ssim(x: Tensor[Float64], y: Tensor[Float64], k1: Float64 = 0.01, k2: Float64 = 0.03, win: Int = 11) -> Tensor[Float64]:
    var L = 1.0
    var c1 = (k1 * L) * (k1 * L)
    var c2 = (k2 * L) * (k2 * L)
    var mu_x = _reduce_mean(x)._data[0]
    var mu_y = _reduce_mean(y)._data[0]
    var xm = x.sub(_scalar(mu_x))
    var ym = y.sub(_scalar(mu_y))
    var var_x = _reduce_mean(xm.mul(xm))._data[0]
    var var_y = _reduce_mean(ym.mul(ym))._data[0]
    var cov_xy = _reduce_mean(xm.mul(ym))._data[0]
    var num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    var den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    var s = num / den
    return _scalar(s)
