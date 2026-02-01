# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/losses/losses.mojo
# Description: Stable softmax/log_softmax and loss functions.

from collections.list import List
from momijo.tensor import tensor
from momijo.tensor.math import softmax,log_softmax

from momijo.tensor import tensor
from collections.list import List

# --- helpers ---

from momijo.tensor import tensor
from collections.list import List

# ---------- debug helpers (tiny & safe) ----------
@always_inline
fn _numel(shape: List[Int]) -> Int:
    var p = 1; var i = 0; var n = len(shape)
    while i < n: p = p * shape[i]; i += 1
    return p

@always_inline
fn _shapes_match(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b): return False
    var i = 0
    while i < len(a):
        if a[i] != b[i]: return False
        i += 1
    return True

fn _clip_tensor(x: tensor.Tensor[Float32], lo: Float32, hi: Float32) -> tensor.Tensor[Float32]:
    var y = x.copy()
    var n = _numel(y.shape()); var i = 0
    while i < n:
        var v = y._data[i]
        if v < lo: v = lo
        if v > hi: v = hi
        y._data[i] = v
        i += 1
    return y.copy()

fn cross_entropy_from_probs(
    probs_in: tensor.Tensor[Float32],
    target_onehot: tensor.Tensor[Float32]
) -> Float32:
    var ps = probs_in.shape(); var ts = target_onehot.shape()
    if len(ps) != 2 or len(ts) != 2: return Float32(0.0)
    var N = ps[0]; var C = ps[1]
    if N <= 0 or C <= 0: return Float32(0.0)
    if ts[0] != N or ts[1] != C: return Float32(0.0)
    var eps = Float32(1e-7)
    var probs = _clip_tensor(probs_in, eps, Float32(1.0) - eps)
    var loss = - (target_onehot * probs.log()).sum_all() / Float32(N)
    return loss

fn cross_entropy_from_logits(
    logits: tensor.Tensor[Float32],
    target_onehot: tensor.Tensor[Float32]
) -> Float32:
    var ls = logits.shape(); var ts = target_onehot.shape()
    if len(ls) != 2 or len(ts) != 2: return Float32(0.0)
    var N = ls[0]; var C = ls[1]
    if N <= 0 or C <= 0: return Float32(0.0)
    if ts[0] != N or ts[1] != C: return Float32(0.0)
    var lsm = log_softmax(logits)
    return - (target_onehot * lsm).sum_all() / Float32(N)

fn softmax_cross_entropy(
    logits: tensor.Tensor[Float32],
    target_onehot: tensor.Tensor[Float32]
) -> (Float32, tensor.Tensor[Float32]):
    var ls = logits.shape(); var ts = target_onehot.shape()
    if len(ls) != 2 or len(ts) != 2: return (Float32(0.0), logits.copy())
    var N = ls[0]; var C = ls[1]
    if N <= 0 or C <= 0: return (Float32(0.0), logits.copy())
    if ts[0] != N or ts[1] != C: return (Float32(0.0), logits.copy())
    var probs = softmax(logits)
    var loss = cross_entropy_from_probs(probs, target_onehot)
    var invN = Float32(1.0) / Float32(N)
    var grad = (probs - target_onehot) * invN
    return (loss, grad.copy())   # no copy for speed; add .copy() if your API requires

fn mse_loss(
    y_pred: tensor.Tensor[Float32],
    y_true: tensor.Tensor[Float32]
) -> Float32:
    var sp = y_pred.shape(); var st = y_true.shape()
    if not _shapes_match(sp, st): return Float32(0.0)
    var total = ((y_pred - y_true) * (y_pred - y_true)).sum_all()
    var n = _numel(sp)
    if n <= 0: return Float32(0.0)
    return total / Float32(n)
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


# Numerically safe log with lower clamp
fn _safe_log(x: tensor.Tensor[Float32], eps: Float32) -> tensor.Tensor[Float32]:
    # Clamp to [eps, +inf) to avoid -inf on log(0)
    return (x.clip(eps, 1e308)).log()

# Reduce helpers (mean/sum over all elements)
# Convention: library is expected to return a scalar tensor (shape [1]) for full reductions.
fn _reduce_mean(x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
    # API supports mean() without args → full reduction
    return x.mean()

@always_inline
fn _reduce_sum(x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
    # API supports sum() without args → full reduction
    return x.sum()

# Ones-like helper with a scalar fill
fn _filled_like(x: tensor.Tensor[Float32], v: Float32) -> tensor.Tensor[Float32]:
    return tensor.Tensor[Float32](x.shape(), v)

# -----------------------------------------------------------------------------
# Regression losses
# -----------------------------------------------------------------------------

# Mean Squared Error
fn mse(pred: tensor.Tensor[Float32], target: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
    var diff = pred.sub(target)
    var sq = diff.mul(diff)
    return _reduce_mean(sq)

# Mean Absolute Error
fn mae(pred: tensor.Tensor[Float32], target: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
    var diff = pred.sub(target).abs()
    return _reduce_mean(diff)

# Huber loss (a.k.a. SmoothL1 with parameter delta).
# For |x| <= delta => 0.5 * x^2; else => delta * (|x| - 0.5 * delta)
fn huber(pred: tensor.Tensor[Float32], target: tensor.Tensor[Float32], delta: Float32 = 1.0) -> tensor.Tensor[Float32]:
    var diff = pred.sub(target).abs()
    var quad = diff.clip(0.0, delta)                 # min(diff, delta)
    var lin = diff.sub(quad)                         # max(diff - delta, 0)
    var loss = quad.mul(quad).mul(0.5).add(lin.mul(delta).sub(_scalar(0.5 * delta * delta)))
    return _reduce_mean(loss)

# Smooth L1 with beta (matches PyTorch when beta > 0)
# loss = 0.5*diff^2/beta if |diff| < beta else |diff| - 0.5*beta
fn smooth_l1(pred: tensor.Tensor[Float32], target: tensor.Tensor[Float32], beta: Float32 = 1.0) -> tensor.Tensor[Float32]:
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
fn bce(pred: tensor.Tensor[Float32], target: tensor.Tensor[Float32], eps: Float32 = 1e-12) -> tensor.Tensor[Float32]:
    var p = pred.clip(eps, 1.0 - eps)
    var t = target
    var one_p = _filled_like(p, 1.0).sub(p)
    var one_t = _filled_like(t, 1.0).sub(t)
    var term1 = t.mul(_safe_log(p, eps))
    var term2 = one_t.mul(_safe_log(one_p, eps))
    var loss = term1.add(term2).mul(-1.0)
    return _reduce_mean(loss)

# KL divergence KL(p || q) with p, q probability distributions
fn kl_div(p: tensor.Tensor[Float32], q: tensor.Tensor[Float32], eps: Float32 = 1e-12) -> tensor.Tensor[Float32]:
    var pp = p.clip(eps, 1.0)
    var qq = q.clip(eps, 1.0)
    var ratio = pp.div(qq)
    var loss = pp.mul(_safe_log(ratio, eps))
    return _reduce_sum(loss)

# Negative Log-Likelihood: expects log-probabilities and class indices (Float32, will be rounded)
# axis: class axis (default last)
fn nll_loss(log_probs: tensor.Tensor[Float32], target_index: tensor.Tensor[Float32], axis: Int = -1) -> tensor.Tensor[Float32]:
    # Gather log p(y) via one-hot style masking with broadcasting on the class axis.
    var probs = log_probs
    var c = probs.shape()[probs.ndim() - 1] if axis == -1 else probs.shape()[axis]

    # Build class range tensor [0..c-1]
    var tmp = List[Float32]()
    var i = 0
    while i < c:
        tmp.append(Float32(i))
        i += 1
    var classes = tensor.Tensor[Float32](tmp, [c])

    # Compare classes against target indices (broadcasted), mask and reduce
    var onehot = probs.equal(classes)
    var picked = onehot.mul(probs)                      # mask: keep only target class log-prob
    var loss = picked.sum(axis, False).mul(-1.0)        # -log p(y)
    return _reduce_mean(loss)

# Cross-entropy for logits
fn cross_entropy(logits: tensor.Tensor[Float32], target_index: tensor.Tensor[Float32], axis: Int = -1, eps: Float32 = 1e-12) -> tensor.Tensor[Float32]:
    var lsm = logits.log_softmax(axis)
    return nll_loss(lsm, target_index, axis)

# Focal loss (multi-class) on logits with gamma and optional alpha in [0, 1]
fn focal_loss(
    logits: tensor.Tensor[Float32],
    target_index: tensor.Tensor[Float32],
    gamma: Float32 = 2.0,
    alpha: Optional[Float32] = None,
    axis: Int = -1,
    eps: Float32 = 1e-12
) -> tensor.Tensor[Float32]:
    var sm = logits.softmax(axis).clip(eps, 1.0)

    # Select probability of the true class via one-hot masking
    var c = sm.shape()[sm.ndim() - 1] if axis == -1 else sm.shape()[axis]
    var tmp = List[Float32]()
    var i = 0
    while i < c:
        tmp.append(Float32(i))
        i += 1
    var classes = tensor.Tensor[Float32](tmp, [c])

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
fn dice_loss(pred: tensor.Tensor[Float32], target: tensor.Tensor[Float32], eps: Float32 = 1e-6) -> tensor.Tensor[Float32]:
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

# Tripvar loss: max(0, d(a,p) - d(a,n) + margin) with squared L2 distance
fn triplet_loss(
    anchor: tensor.Tensor[Float32],
    positive: tensor.Tensor[Float32],
    negative: tensor.Tensor[Float32],
    margin: Float32 = 1.0
) -> tensor.Tensor[Float32]:
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
    x1: tensor.Tensor[Float32],
    x2: tensor.Tensor[Float32],
    target: tensor.Tensor[Float32],
    margin: Float32 = 0.0,
    eps: Float32 = 1e-12
) -> tensor.Tensor[Float32]:
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
