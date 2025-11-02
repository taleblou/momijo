# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/losses/losses.mojo
# Description: Stable softmax/log_softmax and loss functions.

from collections.list import List
from momijo.tensor import tensor 

# --- helpers ---

@always_inline
fn _numel(shape: List[Int]) -> Int:
    var p = 1
    var i = 0
    var n = len(shape)
    while i < n:
        p = p * shape[i]
        i = i + 1
    return p

# Extract a Float64 scalar from a 1-element tensor (e.g. shape [1,1] or [1])
@always_inline
fn _scalar(t: tensor.Tensor[Float64]) -> Float64:
    # Assumes t has exactly one element
    return t._data[0]

# Row-wise sum: (N,C) @ (C,1) => (N,1)
fn _row_sum(x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
    var C = x.shape()[1]
    var ones_c1 = tensor.ones([C, 1])
    return x.matmul(ones_c1)

# Soft row-wise max via softmax-weighted expectation (smooth approx)
fn _row_max_approx(x: tensor.Tensor[Float64], k: Float64 = 64.0) -> tensor.Tensor[Float64]:
    var kx = x * k
    var w = tensor.exp(kx)
    var Z = _row_sum(w) + 1e-12     # avoid divide-by-zero
    var p = w / Z
    return _row_sum(x * p)

# Stable softmax per row
fn softmax(logits: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
    var m = _row_max_approx(logits)
    var z = logits - m
    var e = tensor.exp(z)
    var s = _row_sum(e) + 1e-12
    return e / s

# Stable log_softmax per row
fn log_softmax(logits: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
    var m = _row_max_approx(logits)
    var z = logits - m
    var e = tensor.exp(z)
    var s = _row_sum(e) + 1e-12
    var lse = tensor.log(s) + m
    return logits - lse

# Cross-entropy from probabilities (targets one-hot)
fn cross_entropy_from_probs(
    probs: tensor.Tensor[Float64],
    target_onehot: tensor.Tensor[Float64]
) -> Float64:
    var eps = 1e-12
    var logp = tensor.log(probs + eps)
    var mul = target_onehot * logp            # (N,C)
    var row = _row_sum(mul)                   # (N,1)
    var N = probs.shape()[0]
    var ones_n1_T = tensor.ones([N, 1]).transpose()  # (1,N)
    var total = ones_n1_T.matmul(row)         # (1,1)
    return - _scalar(total) / Float64(N)

# Cross-entropy directly from logits (more stable)
fn cross_entropy_from_logits(
    logits: tensor.Tensor[Float64],
    target_onehot: tensor.Tensor[Float64]
) -> Float64:
    var lsm = log_softmax(logits)             # (N,C)
    var mul = target_onehot * lsm
    var row = _row_sum(mul)                   # (N,1)
    var N = logits.shape()[0]
    var ones_n1_T = tensor.ones([N, 1]).transpose()
    var total = ones_n1_T.matmul(row)         # (1,1)
    return - _scalar(total) / Float64(N)

# Softmax CE with gradient wrt logits
fn softmax_cross_entropy(
    logits: tensor.Tensor[Float64],
    target_onehot: tensor.Tensor[Float64]
) -> (Float64, tensor.Tensor[Float64]):
    var probs = softmax(logits)
    var loss = cross_entropy_from_probs(probs, target_onehot)
    var N = logits.shape()[0]
    var grad = (probs - target_onehot) / Float64(N)
    return (loss, grad)

# MSE over all elements (mean over N*C)
fn mse_loss(
    y_pred: tensor.Tensor[Float64],
    y_true: tensor.Tensor[Float64]
) -> Float64:
    var diff = y_pred - y_true
    var sq = diff * diff                      # (N,C)
    var row = _row_sum(sq)                    # (N,1)
    var N = y_pred.shape()[0]
    var C = y_pred.shape()[1]
    var ones_n1_T = tensor.ones([N, 1]).transpose()  # (1,N)
    var total = ones_n1_T.matmul(row)         # (1,1)
    return _scalar(total) / Float64(N * C)   



# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Wrap a scalar Float64 into a rank-1 Tensor[Float64] with shape [1]
fn _scalar(v: Float64) -> tensor.Tensor[Float64]:
    var data = List[Float64]()
    data.append(v)
    return tensor.Tensor[Float64](data, [1])

# Numerically safe log with lower clamp
fn _safe_log(x: tensor.Tensor[Float64], eps: Float64) -> tensor.Tensor[Float64]:
    # Clamp to [eps, +inf) to avoid -inf on log(0)
    return (x.clip(eps, 1e308)).log()

# Reduce helpers (mean/sum over all elements)
# Convention: library is expected to return a scalar tensor (shape [1]) for full reductions.
fn _reduce_mean(x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
    # mean over all dims; axis=None, keepdim=False (implementation should produce a scalar tensor)
    return x.mean(None, False)

fn _reduce_sum(x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
    return x.sum(None, False)

# Ones-like helper with a scalar fill
fn _filled_like(x: tensor.Tensor[Float64], v: Float64) -> tensor.Tensor[Float64]:
    return tensor.Tensor[Float64](x.shape(), v)

# -----------------------------------------------------------------------------
# Regression losses
# -----------------------------------------------------------------------------

# Mean Squared Error
fn mse(pred: tensor.Tensor[Float64], target: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
    var diff = pred.sub(target)
    var sq = diff.mul(diff)
    return _reduce_mean(sq)

# Mean Absolute Error
fn mae(pred: tensor.Tensor[Float64], target: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
    var diff = pred.sub(target).abs()
    return _reduce_mean(diff)

# Huber loss (a.k.a. SmoothL1 with parameter delta).
# For |x| <= delta => 0.5 * x^2; else => delta * (|x| - 0.5 * delta)
fn huber(pred: tensor.Tensor[Float64], target: tensor.Tensor[Float64], delta: Float64 = 1.0) -> tensor.Tensor[Float64]:
    var diff = pred.sub(target).abs()
    var quad = diff.clip(0.0, delta)                 # min(diff, delta)
    var lin = diff.sub(quad)                         # max(diff - delta, 0)
    var loss = quad.mul(quad).mul(0.5).add(lin.mul(delta).sub(_scalar(0.5 * delta * delta)))
    return _reduce_mean(loss)

# Smooth L1 with beta (matches PyTorch when beta > 0)
# loss = 0.5*diff^2/beta if |diff| < beta else |diff| - 0.5*beta
fn smooth_l1(pred: tensor.Tensor[Float64], target: tensor.Tensor[Float64], beta: Float64 = 1.0) -> tensor.Tensor[Float64]:
    var diff = pred.sub(target).abs()
    var use_quad = diff.less(beta)
    var quad = diff.mul(diff).mul(0.5 / beta)
    var lin = diff.sub(0.5 * beta)
    var loss = tensor.where(use_quad, quad, lin)
    return _reduce_mean(loss)

# -----------------------------------------------------------------------------
# Classification / probabilistic losses
# -----------------------------------------------------------------------------

# Binary cross-entropy for probabilities in [0, 1]
fn bce(pred: tensor.Tensor[Float64], target: tensor.Tensor[Float64], eps: Float64 = 1e-12) -> tensor.Tensor[Float64]:
    var p = pred.clip(eps, 1.0 - eps)
    var t = target
    var one_p = _filled_like(p, 1.0).sub(p)
    var one_t = _filled_like(t, 1.0).sub(t)
    var term1 = t.mul(_safe_log(p, eps))
    var term2 = one_t.mul(_safe_log(one_p, eps))
    var loss = term1.add(term2).mul(-1.0)
    return _reduce_mean(loss)

# KL divergence KL(p || q) with p, q probability distributions
fn kl_div(p: tensor.Tensor[Float64], q: tensor.Tensor[Float64], eps: Float64 = 1e-12) -> tensor.Tensor[Float64]:
    var pp = p.clip(eps, 1.0)
    var qq = q.clip(eps, 1.0)
    var ratio = pp.div(qq)
    var loss = pp.mul(_safe_log(ratio, eps))
    return _reduce_sum(loss)

# Negative Log-Likelihood: expects log-probabilities and class indices (Float64, will be rounded)
# axis: class axis (default last)
fn nll_loss(log_probs: tensor.Tensor[Float64], target_index: tensor.Tensor[Float64], axis: Int = -1) -> tensor.Tensor[Float64]:
    # Gather log p(y) via one-hot style masking with broadcasting on the class axis.
    var probs = log_probs
    var c = probs.shape()[probs.ndim() - 1] if axis == -1 else probs.shape()[axis]

    # Build class range tensor [0..c-1]
    var tmp = List[Float64]()
    var i = 0
    while i < c:
        tmp.append(Float64(i))
        i += 1
    var classes = tensor.Tensor[Float64](tmp, [c])

    # Compare classes against target indices (broadcasted), mask and reduce
    var onehot = probs.equal(classes)
    var picked = onehot.mul(probs)                      # mask: keep only target class log-prob
    var loss = picked.sum(axis, False).mul(-1.0)        # -log p(y)
    return _reduce_mean(loss)

# Cross-entropy for logits
fn cross_entropy(logits: tensor.Tensor[Float64], target_index: tensor.Tensor[Float64], axis: Int = -1, eps: Float64 = 1e-12) -> tensor.Tensor[Float64]:
    var lsm = logits.log_softmax(axis)
    return nll_loss(lsm, target_index, axis)

# Focal loss (multi-class) on logits with gamma and optional alpha in [0, 1]
fn focal_loss(
    logits: tensor.Tensor[Float64],
    target_index: tensor.Tensor[Float64],
    gamma: Float64 = 2.0,
    alpha: Optional[Float64] = None,
    axis: Int = -1,
    eps: Float64 = 1e-12
) -> tensor.Tensor[Float64]:
    var sm = logits.softmax(axis).clip(eps, 1.0)

    # Select probability of the true class via one-hot masking
    var c = sm.shape()[sm.ndim() - 1] if axis == -1 else sm.shape()[axis]
    var tmp = List[Float64]()
    var i = 0
    while i < c:
        tmp.append(Float64(i))
        i += 1
    var classes = tensor.Tensor[Float64](tmp, [c])

    var onehot = sm.equal(classes)
    var pt = onehot.mul(sm).sum(axis, False)            # true-class probability

    # Modulation factor: (1 - p_t)^gamma
    var mod = _filled_like(pt, 1.0).sub(pt).pow_scalar(gamma)

    # Base CE (mean scalar [1])
    var ce = cross_entropy(logits, target_index, axis, eps)

    # Alpha weighting (optional)
    var alpha_w = 1.0
    if not (alpha is None):
        alpha_w = alpha.value()

    var loss = ce.mul(alpha_w).mul(mod.mean(None, False))
    return loss

# Dice loss for probabilities (binary or per-class averaged if channel axis used externally)
fn dice_loss(pred: tensor.Tensor[Float64], target: tensor.Tensor[Float64], eps: Float64 = 1e-6) -> tensor.Tensor[Float64]:
    var p = pred
    var t = target
    var inter = _reduce_sum(p.mul(t))
    var denom = _reduce_sum(p).add(_reduce_sum(t)).add(_scalar(eps))
    var dice = inter.mul(2.0).div(denom)
    var loss = _scalar(1.0).sub(dice)
    return loss

# -----------------------------------------------------------------------------
# Similarity losses
# -----------------------------------------------------------------------------

# Triplet loss: max(0, d(a,p) - d(a,n) + margin) with squared L2 distance
fn triplet_loss(
    anchor: tensor.Tensor[Float64],
    positive: tensor.Tensor[Float64],
    negative: tensor.Tensor[Float64],
    margin: Float64 = 1.0
) -> tensor.Tensor[Float64]:
    var ap = anchor.sub(positive)
    var an = anchor.sub(negative)
    var dap = _reduce_sum(ap.mul(ap))
    var dan = _reduce_sum(an.mul(an))
    var raw = dap.sub(dan).add(_scalar(margin))
    var zero = _scalar(0.0)
    var loss = tensor.where(raw.greater(zero), raw, zero)
    return _reduce_mean(loss)

# Cosine embedding loss: target ∈ {1, -1}; margin for negative pairs
fn cosine_embedding_loss(
    x1: tensor.Tensor[Float64],
    x2: tensor.Tensor[Float64],
    target: tensor.Tensor[Float64],
    margin: Float64 = 0.0,
    eps: Float64 = 1e-12
) -> tensor.Tensor[Float64]:
    var dot = _reduce_sum(x1.mul(x2))
    var n1 = _reduce_sum(x1.mul(x1)).add(_scalar(eps)).sqrt()
    var n2 = _reduce_sum(x2.mul(x2)).add(_scalar(eps)).sqrt()
    var cos = dot.div(n1.mul(n2))             # [-1, 1]
    var one = _scalar(1.0)
    var pos_loss = one.sub(cos)               # for target = 1
    var neg_loss = (cos.sub(_scalar(margin))).clip(0.0, 1e308)  # for target = -1
    var is_pos = target.greater(_scalar(0.0))
    var loss = tensor.where(is_pos, pos_loss, neg_loss)
    return _reduce_mean(loss)
