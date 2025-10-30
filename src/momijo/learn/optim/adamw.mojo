# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.optim.adamw
# File:         src/momijo/learn/optim/adamw.mojo
#
# Description:
#   AdamW optimizer (decoupled weight decay) for Momijo Learn.
#   - Float64 List fallback: runs in demos with no tensor backend.
#   - Optional tensor overloads using the Momijo tensor facade.
#   - Public API:
#       * register_params(params, grads)                 # Float64 lists
#       * register_params_tensor(params_t, grads_t)      # tensor.Tensor[Float64]
#       * step()                                         # update registered params
#       * step_with(params, grads)                       # one-off stateless helper
#       * zero_grad()                                    # zero registered grads
#       * set_hyperparams(...) / reset_state()
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor   # optional tensor overloads

# -----------------------------------------------------------------------------
# Small scalar helpers (Float64 fallback)
# -----------------------------------------------------------------------------

@always_inline
fn _pow_scalar(x: Float64, n: Int) -> Float64:
    var r = 1.0
    var i = 0
    while i < n:
        r = r * x
        i = i + 1
    return r

@always_inline
fn _sqrt_scalar(x: Float64) -> Float64:
    # Newton-Raphson; sufficient for optimizer math
    if x <= 0.0:
        return 0.0
    var g = x
    var i = 0
    while i < 12:
        g = 0.5 * (g + x / g)
        i = i + 1
    return g

@always_inline
fn _assert_same_len(a: List[Float64], b: List[Float64]) -> None:
    assert(Int(a.size()) == Int(b.size()))

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
# AdamW (stateful)
# -----------------------------------------------------------------------------

struct AdamW:
    # Hyperparameters
    var lr: Float64
    var weight_decay: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64

    # Slots/state for Float64 List fallback
    var _params: List[Float64]
    var _grads: List[Float64]
    var _m: List[Float64]
    var _v: List[Float64]
    var _t: Int
    var _initialized: Bool

    # Tensor-backed state (optional)
    var _params_t: tensor.Tensor[Float64]
    var _grads_t: tensor.Tensor[Float64]
    var _m_t: tensor.Tensor[Float64]
    var _v_t: tensor.Tensor[Float64]
    var _tensor_mode: Bool

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

        # tensor slots: create empty by default
        self._params_t = tensor.Tensor[Float64]()
        self._grads_t  = tensor.Tensor[Float64]()
        self._m_t      = tensor.Tensor[Float64]()
        self._v_t      = tensor.Tensor[Float64]()
        self._tensor_mode = False

    # -------------------------------------------------------------------------
    # Hyperparams & housekeeping
    # -------------------------------------------------------------------------

    fn set_hyperparams(mut self,
                       lr: Float64? = None,
                       weight_decay: Float64? = None,
                       beta1: Float64? = None,
                       beta2: Float64? = None,
                       eps: Float64? = None) -> None:
        if lr is not None:
            self.lr = lr.value()
        if weight_decay is not None:
            self.weight_decay = weight_decay.value()
        if beta1 is not None:
            self.beta1 = beta1.value()
        if beta2 is not None:
            self.beta2 = beta2.value()
        if eps is not None:
            self.eps = eps.value()

    fn reset_state(mut self) -> None:
        # Clear both state variants
        self._m = List[Float64]()
        self._v = List[Float64]()
        self._m_t = tensor.Tensor[Float64]()
        self._v_t = tensor.Tensor[Float64]()
        self._t = 0

        # Keep bindings; re-init slots to zeros if already registered
        if self._initialized and self._tensor_mode:
            var s = self._params_t.shape()
            self._m_t = tensor.Tensor[Float64](s, 0.0)
            self._v_t = tensor.Tensor[Float64](s, 0.0)
        elif self._initialized:
            var n2 = Int(self._params.size())
            var i = 0
            while i < n2:
                self._m.push_back(0.0)
                self._v.push_back(0.0)
                i = i + 1

    # -------------------------------------------------------------------------
    # Registration: Float64 lists
    # -------------------------------------------------------------------------

    fn register_params(mut self, params: List[Float64], grads: List[Float64]) -> None:
        _assert_same_len(params, grads)
        self._params = params
        self._grads  = grads

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
        self._tensor_mode = False

    # Zero grads for the registered backend
    fn zero_grad(mut self) -> None:
        if not self._initialized:
            return
        if self._tensor_mode:
            var g = self._grads_t._data
            var n = _numel(self._grads_t.shape())
            var i = 0
            while i < n:
                g[i] = 0.0
                i = i + 1
            return
        var n2 = Int(self._grads.size())
        var j = 0
        while j < n2:
            self._grads[j] = 0.0
            j = j + 1

    # -------------------------------------------------------------------------
    # Registration: tensor backend (Float64)
    # -------------------------------------------------------------------------

    fn register_params_tensor(mut self,
                              params: tensor.Tensor[Float64],
                              grads: tensor.Tensor[Float64]) -> None:
        # Expect identical shapes
        assert(len(params.shape()) == len(grads.shape()))
        var s = params.shape()
        var t = grads.shape()
        var k = 0
        while k < len(s):
            assert(s[k] == t[k])
            k = k + 1

        self._params_t = params
        self._grads_t  = grads
        self._m_t = tensor.Tensor[Float64](s, 0.0)
        self._v_t = tensor.Tensor[Float64](s, 0.0)

        self._t = 0
        self._initialized = True
        self._tensor_mode = True

    # -------------------------------------------------------------------------
    # Update (stateful)
    # -------------------------------------------------------------------------

    fn step(mut self) -> None:
        if not self._initialized:
            return

        self._t = self._t + 1
        var t = self._t

        var b1 = self.beta1
        var b2 = self.beta2
        var one = 1.0

        var bias_correction1 = one - _pow_scalar(b1, t)
        var bias_correction2 = one - _pow_scalar(b2, t)

        var lr = self.lr
        var wd = self.weight_decay
        var eps = self.eps

        if self._tensor_mode:
            # Tensor path: raw contiguous buffer
            var p = self._params_t._data
            var g = self._grads_t._data
            var m = self._m_t._data
            var v = self._v_t._data
            var n = _numel(self._params_t.shape())

            var i = 0
            while i < n:
                var gi = g[i]
                var m_prev = m[i]
                var v_prev = v[i]

                var m_t = b1 * m_prev + (one - b1) * gi
                var v_t = b2 * v_prev + (one - b2) * (gi * gi)

                m[i] = m_t
                v[i] = v_t

                var m_hat = m_t / bias_correction1
                var v_hat = v_t / bias_correction2
                var denom = _sqrt_scalar(v_hat) + eps
                var step_update = m_hat / denom

                var pi = p[i]
                pi = pi - lr * step_update
                if wd != 0.0:
                    # decoupled weight decay
                    pi = pi - lr * wd * pi
                p[i] = pi

                i = i + 1
            return

        # List[Float64] path
        var n2 = Int(self._params.size())
        var j = 0
        while j < n2:
            var pj = self._params[j]
            var gj = self._grads[j]

            var m_prev2 = self._m[j]
            var v_prev2 = self._v[j]

            var m_t2 = b1 * m_prev2 + (one - b1) * gj
            var v_t2 = b2 * v_prev2 + (one - b2) * (gj * gj)

            self._m[j] = m_t2
            self._v[j] = v_t2

            var m_hat2 = m_t2 / bias_correction1
            var v_hat2 = v_t2 / bias_correction2

            var denom2 = _sqrt_scalar(v_hat2) + eps
            var step_update2 = m_hat2 / denom2

            pj = pj - lr * step_update2
            if wd != 0.0:
                pj = pj - lr * wd * pj

            self._params[j] = pj
            j = j + 1

    # -------------------------------------------------------------------------
    # Stateless helper (convenience)
    # -------------------------------------------------------------------------

    fn step_with(mut self, params: List[Float64], grads: List[Float64]) -> None:
        # Reinitialize if not ready or shape changed or tensor-mode bound
        if not self._initialized or self._tensor_mode or Int(self._params.size()) != Int(params.size()):
            self.register_params(params, grads)
        else:
            self._params = params
            self._grads  = grads
        self.step()

    # AMP compatibility hook (kept for API symmetry with GradScaler)
    fn zero_grad_after_step(mut self) -> Bool:
        return True

    # Optional: pretty string
    fn __str__(self) -> String:
        var s = "AdamW(lr=" + String(self.lr)
        s = s + ", wd=" + String(self.weight_decay)
        s = s + ", beta1=" + String(self.beta1)
        s = s + ", beta2=" + String(self.beta2)
        s = s + ", eps=" + String(self.eps)
        s = s + ", t=" + String(self._t)
        s = s + ", tensor_mode="
        s = s + (String("true") if self._tensor_mode else String("false"))
        s = s + ")"
        return s
