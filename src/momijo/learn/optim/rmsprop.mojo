# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.optim.rmsprop
# File:         src/momijo/learn/optim/rmsprop.mojo
#
# Description:
#   RMSprop optimizer with optional momentum, weight decay, and centered variant.
#   Backend-agnostic: works on a placeholder scalar param type for demos, and
#   optionally on Momijo tensor facade when available.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor   # tensor facade (dtype factories + ops)

# -----------------------------------------------------------------------------
# Small numeric helpers (fallbacks)
# -----------------------------------------------------------------------------

@always_inline
fn _sqrt_scalar(x: Float64) -> Float64:
    if x <= 0.0:
        return 0.0
    var g = x
    var i = 0
    # Newton-Raphson iterations (sufficient for optimizer denom)
    while i < 12:
        g = 0.5 * (g + x / g)
        i = i + 1
    return g

@always_inline
fn _clamp_min(x: Float64, lo: Float64) -> Float64:
    var v = x
    if v < lo:
        v = lo
    return v

@always_inline
fn _numel(shape: List[Int]) -> Int:
    var p = 1
    var i = 0
    var n = len(shape)
    while i < n:
        p = p * shape[i]
        i = i + 1
    return p

# -----------------------------------------------------------------------------
# Placeholder parameter type (Float64 demo path)
# -----------------------------------------------------------------------------

struct OptParam:
    var value: Float64
    var grad: Float64

    fn __init__(out self, value: Float64, grad: Float64 = 0.0):
        self.value = value
        self.grad = grad

# -----------------------------------------------------------------------------
# RMSprop optimizer
# -----------------------------------------------------------------------------

struct RMSprop:
    # Hyperparameters
    var lr: Float64
    var alpha: Float64
    var eps: Float64
    var weight_decay: Float64
    var momentum: Float64
    var centered: Bool
    var decoupled: Bool

    # Demo state (parallel to scalar params)
    var params: List[OptParam]
    var square_avg: List[Float64]   # E[g^2]
    var grad_avg: List[Float64]     # momentum buffer
    var mean_grad: List[Float64]    # E[g] for centered variant

    # Tensor state (optional)
    var params_t: tensor.Tensor[Float64]
    var grads_t:  tensor.Tensor[Float64]
    var square_avg_t: tensor.Tensor[Float64]
    var grad_avg_t:   tensor.Tensor[Float64]
    var mean_grad_t:  tensor.Tensor[Float64]
    var _tensor_mode: Bool

    fn __init__(
        out self,
        lr: Float64 = 0.001,
        alpha: Float64 = 0.99,
        eps: Float64 = 1e-8,
        weight_decay: Float64 = 0.0,
        momentum: Float64 = 0.0,
        centered: Bool = False,
        decoupled: Bool = False
    ):
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.decoupled = decoupled

        self.params = List[OptParam]()
        self.square_avg = List[Float64]()
        self.grad_avg = List[Float64]()
        self.mean_grad = List[Float64]()

        self.params_t = tensor.Tensor[Float64]()
        self.grads_t  = tensor.Tensor[Float64]()
        self.square_avg_t = tensor.Tensor[Float64]()
        self.grad_avg_t   = tensor.Tensor[Float64]()
        self.mean_grad_t  = tensor.Tensor[Float64]()
        self._tensor_mode = False

    # -------------------------------------------------------------------------
    # Registration (scalar demo path)
    # -------------------------------------------------------------------------

    fn set_params(mut self, ps: List[OptParam]):
        self.params = ps

        self.square_avg = List[Float64]()
        self.grad_avg   = List[Float64]()
        self.mean_grad  = List[Float64]()
        var n = len(ps)
        var i = 0
        while i < n:
            self.square_avg.append(0.0)
            self.grad_avg.append(0.0)
            self.mean_grad.append(0.0)
            i = i + 1

        self._tensor_mode = False

    # -------------------------------------------------------------------------
    # Registration (tensor path)
    # -------------------------------------------------------------------------

    fn set_params_tensor(mut self, params: tensor.Tensor[Float64], grads: tensor.Tensor[Float64]):
        # require identical shapes
        var sp = params.shape()
        var sg = grads.shape()
        var k = 0
        assert(len(sp) == len(sg))
        while k < len(sp):
            assert(sp[k] == sg[k])
            k = k + 1

        self.params_t = params
        self.grads_t  = grads

        # Allocate optimizer buffers as zeros with matching shape/dtype
        var f64 = tensor.Float64()
        self.square_avg_t = tensor.zeros(sp, f64)  # E[g^2]
        self.grad_avg_t   = tensor.zeros(sp, f64)  # momentum buffer
        self.mean_grad_t  = tensor.zeros(sp, f64)  # E[g] (centered)

        self._tensor_mode = True

    # -------------------------------------------------------------------------
    # Zero gradients
    # -------------------------------------------------------------------------

    fn zero_grad(mut self):
        if self._tensor_mode:
            var n = _numel(self.grads_t.shape())
            var g = self.grads_t._data
            var i = 0
            while i < n:
                g[i] = 0.0
                i = i + 1
            return

        var n2 = len(self.params)
        var j = 0
        while j < n2:
            var p = self.params[j]
            p.grad = 0.0
            self.params[j] = p
            j = j + 1

    # -------------------------------------------------------------------------
    # One optimization step
    # -------------------------------------------------------------------------

    fn step(mut self):
        if self._tensor_mode:
            self._step_tensor()
        else:
            self._step_scalar()

    # Scalar (Float64 demo) update
    fn _step_scalar(mut self):
        var n = len(self.params)
        var i = 0

        while i < n:
            var p = self.params[i]
            var g = p.grad

            # Classic L2 as grad penalty (if not decoupled)
            if self.weight_decay != 0.0 and not self.decoupled:
                g = g + self.weight_decay * p.value

            # E[g^2]
            var sq = self.square_avg[i]
            sq = self.alpha * sq + (1.0 - self.alpha) * (g * g)
            self.square_avg[i] = sq

            # denom
            var denom: Float64
            if self.centered:
                var mg = self.mean_grad[i]
                mg = self.alpha * mg + (1.0 - self.alpha) * g
                self.mean_grad[i] = mg
                var var_g = sq - (mg * mg)
                if var_g < 0.0:
                    var_g = 0.0
                denom = _sqrt_scalar(var_g + self.eps)
            else:
                denom = _sqrt_scalar(sq + self.eps)

            # Momentum buffer
            if self.momentum != 0.0:
                var v = self.grad_avg[i]
                v = self.momentum * v + g / denom
                self.grad_avg[i] = v
                p.value = p.value - self.lr * v
            else:
                p.value = p.value - self.lr * (g / denom)

            # Decoupled weight decay (AdamW-style) after grad step
            if self.weight_decay != 0.0 and self.decoupled:
                p.value = p.value * (1.0 - self.lr * self.weight_decay)

            self.params[i] = p
            i = i + 1

    # Tensor update
    fn _step_tensor(mut self):
        var p = self.params_t._data
        var g = self.grads_t._data
        var sq = self.square_avg_t._data
        var v  = self.grad_avg_t._data
        var mg = self.mean_grad_t._data

        var n = _numel(self.params_t.shape())
        var i = 0
        while i < n:
            var gi = g[i]
            var pi = p[i]

            if self.weight_decay != 0.0 and not self.decoupled:
                gi = gi + self.weight_decay * pi

            var sqi = sq[i]
            sqi = self.alpha * sqi + (1.0 - self.alpha) * (gi * gi)
            sq[i] = sqi

            var denom: Float64
            if self.centered:
                var mgi = mg[i]
                mgi = self.alpha * mgi + (1.0 - self.alpha) * gi
                mg[i] = mgi
                var var_g = sqi - (mgi * mgi)
                if var_g < 0.0:
                    var_g = 0.0
                denom = _sqrt_scalar(var_g + self.eps)
            else:
                denom = _sqrt_scalar(sqi + self.eps)

            if self.momentum != 0.0:
                var vi = v[i]
                vi = self.momentum * vi + gi / denom
                v[i] = vi
                pi = pi - self.lr * vi
            else:
                pi = pi - self.lr * (gi / denom)

            if self.weight_decay != 0.0 and self.decoupled:
                pi = pi * (1.0 - self.lr * self.weight_decay)

            p[i] = pi
            i = i + 1

    # Pretty string
    fn __str__(self) -> String:
        var s = "RMSprop(lr=" + String(self.lr)
        s = s + ", alpha=" + String(self.alpha)
        s = s + ", eps=" + String(self.eps)
        s = s + ", wd=" + String(self.weight_decay)
        s = s + ", momentum=" + String(self.momentum)
        s = s + ", centered=" + (String("true") if self.centered else String("false"))
        s = s + ", decoupled=" + (String("true") if self.decoupled else String("false"))
        s = s + ", tensor_mode=" + (String("true") if self._tensor_mode else String("false")) + ")"
        return s
