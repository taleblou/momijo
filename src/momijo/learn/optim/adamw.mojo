# Project:      Momijo
# Module:       learn.optim.adamw
# File:         optim/adamw.mojo
# Path:         src/momijo/learn/optim/adamw.mojo
#
# Description:  AdamW optimizer (decoupled weight decay) for Momijo Learn.
#               Backend-agnostic, with a fallback Float64 implementation so it
#               compiles and works in demos before wiring real tensor types.
#               Provides both stateful slots (register once, call step())
#               and a stateless helper (step_with(params, grads)).
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
#   - Types: AdamW (stateful)
#   - Key fns:
#       * register_params(params, grads)  → initialize slots
#       * step()                          → updates registered params
#       * step_with(params, grads)        → one-off update (stateless)
#       * zero_grad()                     → zeroes registered grads
#   - Math:
#       m_t = β1 m_{t-1} + (1-β1) g_t
#       v_t = β2 v_{t-1} + (1-β2) g_t^2
#       m̂_t = m_t / (1-β1^t), v̂_t = v_t / (1-β2^t)
#       p ← p - lr * ( m̂_t / (sqrt(v̂_t)+eps) ) - lr*wd*p
#     (decoupled weight decay per AdamW)
#   - TODO: When `momijo.tensor` is available, swap Float64 lists with real
#     tensor/parameter handles and in-place ops; keep the same public API.

from collections.list import List

struct AdamW:
    # Hyperparameters
    var lr: Float64
    var weight_decay: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64

    # Slots/state (Float64 fallback for now)
    var _params: List[Float64]
    var _grads: List[Float64]
    var _m: List[Float64]
    var _v: List[Float64]
    var _t: Int
    var _initialized: Bool

    fn __init__(
        out self,
        lr: Float64 = 1e-3,
        weight_decay: Float64 = 1e-2,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.999,
        eps: Float64 = 1e-8
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self._params = List[Float64]()
        self._grads  = List[Float64]()
        self._m      = List[Float64]()
        self._v      = List[Float64]()
        self._t = 0
        self._initialized = False

    # -------------------------------------------------------------------------
    # Registration & housekeeping
    # -------------------------------------------------------------------------

    # Register parameter and gradient views (Float64 fallback).
    # For real tensors, expose a parallel overload in the future that receives
    # parameter/tensor handles and initializes m/v with the same shape.
    fn register_params(mut self, params: List[Float64], grads: List[Float64]):
        assert(Int(params.size()) == Int(grads.size()))
        self._params = params
        self._grads = grads

        self._m = List[Float64]()
        self._v = List[Float64]()
        var n = Int(params.size())
        var i = 0
        while i < n:
            self._m.push_back(0.0)
            self._v.push_back(0.0)
            i = i + 1

        self._t = 0
        self._initialized = True

    fn zero_grad(mut self):
        if not self._initialized:
            return
        var n = Int(self._grads.size())
        var i = 0
        while i < n:
            # Float64 fallback
            self._grads[i] = 0.0
            i = i + 1

    # -------------------------------------------------------------------------
    # Update (stateful)
    # -------------------------------------------------------------------------

    fn step(mut self):
        # Requires prior call to register_params(...)
        if not self._initialized:
            return

        self._t = self._t + 1
        var t = self._t

        var b1 = self.beta1
        var b2 = self.beta2
        var one = 1.0

        # Bias correction terms
        # NOTE: For Float64 fallback we compute scalar powers.
        var bias_correction1 = one - pow_scalar(b1, t)
        var bias_correction2 = one - pow_scalar(b2, t)

        var lr = self.lr
        var wd = self.weight_decay
        var eps = self.eps

        var n = Int(self._params.size())
        var i = 0
        while i < n:
            var p = self._params[i]
            var g = self._grads[i]

            # First & second moments
            var m_prev = self._m[i]
            var v_prev = self._v[i]

            var m_t = b1 * m_prev + (one - b1) * g
            var v_t = b2 * v_prev + (one - b2) * (g * g)

            self._m[i] = m_t
            self._v[i] = v_t

            # Bias-corrected
            var m_hat = m_t / bias_correction1
            var v_hat = v_t / bias_correction2

            # Update (decoupled weight decay)
            var denom = sqrt_scalar(v_hat) + eps
            var step_update = (m_hat / denom)

            # p ← p - lr * step_update - lr*wd*p
            p = p - lr * step_update
            if wd != 0.0:
                p = p - lr * wd * p

            self._params[i] = p
            i = i + 1

    # -------------------------------------------------------------------------
    # Stateless helper: one-off update on provided arrays (no internal state).
    # This reuses AdamW internal slots by temporarily binding. Useful for
    # lightweight demos when you don't want a separate register_params call.
    # -------------------------------------------------------------------------
    fn step_with(mut self, params: List[Float64], grads: List[Float64]):
        if not self._initialized or Int(self._params.size()) != Int(params.size()):
            # (Re)initialize for this batch
            self.register_params(params, grads)
        else:
            # reuse m/v; just replace external views
            self._params = params
            self._grads = grads
        self.step()

    # Provided for AMP GradScaler compatibility. No-op here; GradScaler handles it.
    fn zero_grad_after_step(mut self) -> Bool:
        return True


# -----------------------------------------------------------------------------
# Small scalar helpers (Float64 fallback)
# -----------------------------------------------------------------------------

fn pow_scalar(x: Float64, n: Int) -> Float64:
    var r = 1.0
    var i = 0
    while i < n:
        r = r * x
        i = i + 1
    return r

fn sqrt_scalar(x: Float64) -> Float64:
    # Simple Newton-Raphson for positive x (fallback; replace with math.sqrt)
    if x <= 0.0:
        return 0.0
    var g = x
    var i = 0
    while i < 12:  # a few iterations
        g = 0.5 * (g + x / g)
        i = i + 1
    return g
