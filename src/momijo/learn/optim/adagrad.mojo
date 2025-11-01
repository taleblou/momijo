# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.learn
# Module:       learn.optim.adagrad
# File:         src/momijo/learn/optim/adagrad.mojo
#
# Description:
#   Adagrad optimizer (backend-agnostic).
#   - Stateless scalar kernel: adagrad_update_scalar(...)
#   - List helpers: step_on_lists(...)
#   - Optional Tensor helpers (Float64/Float32) via facade:
#       * adagrad_update_tensor_f64(...)
#       * adagrad_update_tensor_f32(...)
#       * step_on_tensor_list(...), step_on_tensor_list_f32(...): keep per-parameter accumulators
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor   # Tensor facade (no wildcard)

# -----------------------------------------------------------------------------
# Minimal numerics (no stdlib.math guarantees)
# -----------------------------------------------------------------------------

@always_inline
fn _abs64(x: Float64) -> Float64:
    if x >= 0.0:
        return x
    return -x

# Newton-Raphson sqrt for non-negative x. Returns 0 for x<=0.
fn _sqrt64(x: Float64, eps: Float64 = 1e-12) -> Float64:
    if x <= 0.0:
        return 0.0
    var g = x
    var prev = 0.0
    var iters = 0
    while iters < 20:
        prev = g
        g = 0.5 * (g + x / g)
        if _abs64(g - prev) <= eps:
            break
        iters = iters + 1
    return g

# -----------------------------------------------------------------------------
# Stateless scalar kernel
# -----------------------------------------------------------------------------
# Returns (new_param, new_accumulator)
#
# Formula (with optional L2 decay):
#   g'    = grad + weight_decay * param
#   acc'  = acc + (g')^2
#   step  = lr * g' / (sqrt(acc') + eps)
#   param'= param - step

fn adagrad_update_scalar(
    param: Float64,
    grad: Float64,
    acc: Float64,
    lr: Float64,
    eps: Float64,
    weight_decay: Float64
) -> (Float64, Float64):
    var g = grad
    if weight_decay != 0.0:
        g = g + weight_decay * param

    var acc_new = acc + g * g
    var denom = _sqrt64(acc_new) + eps
    if denom <= 0.0:
        denom = eps
    var step = lr * g / denom
    var param_new = param - step
    return (param_new, acc_new)

# -----------------------------------------------------------------------------
# Optimizer (backend-agnostic core + List helpers + optional Tensor helpers)
# -----------------------------------------------------------------------------

struct Adagrad:
    var lr: Float64
    var eps: Float64
    var weight_decay: Float64
    var initial_accumulator_value: Float64

    # Accumulators for List-based API (one scalar per param index)
    var _accumulators: List[Float64]

    # Accumulators for Tensor-based API (one tensor per parameter tensor)
    var _acc_tensors_f64: List[tensor.Tensor[Float64]]
    var _acc_tensors_f32: List[tensor.Tensor[Float32]]

    fn __init__(
        out self,
        lr: Float64 = 0.01,
        eps: Float64 = 1e-10,
        weight_decay: Float64 = 0.0,
        initial_accumulator_value: Float64 = 0.0
    ):
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self._accumulators = List[Float64]()
        self._acc_tensors_f64 = List[tensor.Tensor[Float64]]()
        self._acc_tensors_f32 = List[tensor.Tensor[Float32]]()

    # -------------------------------------------------------------------------
    # Placeholder hooks for a real Parameter/Tensor container binding
    # -------------------------------------------------------------------------
    fn step(mut self):
        # To be wired when a real Parameter store exists.
        # Iterate parameters, use the tensor kernels below.
        pass

    fn zero_grad(mut self):
        # To be wired to actual parameter grads.
        pass

    # -------------------------------------------------------------------------
    # List API (immediately usable for smoke tests)
    # -------------------------------------------------------------------------
    fn _ensure_acc_capacity(mut self, n: Int):
        var cur = len(self._accumulators)
        if n > cur:
            var i = cur
            while i < n:
                self._accumulators.push_back(self.initial_accumulator_value)
                i = i + 1

    # params[i] <- params[i] - lr * g[i] / (sqrt(acc[i])+eps)
    fn step_on_lists(mut self, mut params: List[Float64], grads: List[Float64]) -> List[Float64]:
        var n = len(params)
        assert(n == len(grads))
        self._ensure_acc_capacity(n)

        var i = 0
        while i < n:
            var p = params[i]
            var g = grads[i]
            var acc = self._accumulators[i]

            var (p_new, acc_new) = adagrad_update_scalar(
                p, g, acc, self.lr, self.eps, self.weight_decay
            )
            params[i] = p_new
            self._accumulators[i] = acc_new
            i = i + 1

        return params

    fn reset_state(mut self):
        self._accumulators = List[Float64]()
        self._acc_tensors_f64 = List[tensor.Tensor[Float64]]()
        self._acc_tensors_f32 = List[tensor.Tensor[Float32]]()

    # -------------------------------------------------------------------------
    # Tensor helpers (Float64 / Float32)
    # Pure-return kernels + stateful convenience wrapper for lists of tensors.
    # -------------------------------------------------------------------------

    # Elementwise Adagrad for Float64 tensors
    fn adagrad_update_tensor_f64(
        self,
        param: tensor.Tensor[Float64],
        grad: tensor.Tensor[Float64],
        acc: tensor.Tensor[Float64]
    ) -> (tensor.Tensor[Float64], tensor.Tensor[Float64]):
        var shp = param.shape()
        assert(shp == grad.shape())
        assert(shp == acc.shape())

        var n = 1
        var i = 0
        var r = len(shp)
        while i < r:
            n = n * shp[i]
            i = i + 1

        var p = param._data
        var g = grad._data
        var a = acc._data

        var out_p = tensor.Tensor[Float64](shp, 0.0)
        var out_a = tensor.Tensor[Float64](shp, 0.0)
        var op = out_p._data
        var oa = out_a._data

        var k = 0
        while k < n:
            var gg = g[k]
            if self.weight_decay != 0.0:
                gg = gg + self.weight_decay * p[k]
            var acc_new = a[k] + gg * gg
            var denom = _sqrt64(acc_new) + self.eps
            if denom <= 0.0:
                denom = self.eps
            var step = self.lr * gg / denom
            op[k] = p[k] - step
            oa[k] = acc_new
            k = k + 1

        return (out_p, out_a)

    # Elementwise Adagrad for Float32 tensors
    fn adagrad_update_tensor_f32(
        self,
        param: tensor.Tensor[Float32],
        grad: tensor.Tensor[Float32],
        acc: tensor.Tensor[Float32]
    ) -> (tensor.Tensor[Float32], tensor.Tensor[Float32]):
        var shp = param.shape()
        assert(shp == grad.shape())
        assert(shp == acc.shape())

        var n = 1
        var i = 0
        var r = len(shp)
        while i < r:
            n = n * shp[i]
            i = i + 1

        var p = param._data
        var g = grad._data
        var a = acc._data

        var out_p = tensor.Tensor[Float32](shp, Float32(0))
        var out_a = tensor.Tensor[Float32](shp, Float32(0))
        var op = out_p._data
        var oa = out_a._data

        var eps_f = Float32(self.eps)
        var lr_f = Float32(self.lr)
        var wd_f = Float32(self.weight_decay)

        var k = 0
        while k < n:
            var gg = g[k]
            if self.weight_decay != 0.0:
                gg = gg + wd_f * p[k]
            # promote to f64 for sqrt stability, then back to f32
            var acc_new_f64 = Float64(a[k]) + Float64(gg) * Float64(gg)
            var denom_f64 = _sqrt64(acc_new_f64) + self.eps
            if denom_f64 <= 0.0:
                denom_f64 = self.eps
            var step_f64 = Float64(lr_f) * Float64(gg) / denom_f64
            op[k] = p[k] - Float32(step_f64)
            oa[k] = Float32(acc_new_f64)
            k = k + 1

        return (out_p, out_a)

    # Stateful convenience: update a list of parameter tensors with grads.
    # - Supports Float64 OR Float32 lists (homogeneous dtypes per call).
    # - Maintains an internal accumulator tensor per parameter (same shape).
    fn step_on_tensor_list(
        mut self,
        mut params_f64: List[tensor.Tensor[Float64]],
        grads_f64: List[tensor.Tensor[Float64]]
    ) -> List[tensor.Tensor[Float64]]:
        var n = len(params_f64)
        assert(n == len(grads_f64))

        # Ensure accumulator slots
        var cur = len(self._acc_tensors_f64)
        if n > cur:
            var i = cur
            while i < n:
                var shp = params_f64[i - cur].shape() if i >= cur else params_f64[i].shape()
                var acc0 = tensor.Tensor[Float64](shp, self.initial_accumulator_value)
                self._acc_tensors_f64.push_back(acc0)
                i = i + 1

        var i2 = 0
        while i2 < n:
            var p = params_f64[i2]
            var g = grads_f64[i2]
            var a = self._acc_tensors_f64[i2]
            var (p_new, a_new) = self.adagrad_update_tensor_f64(p, g, a)
            params_f64[i2] = p_new
            self._acc_tensors_f64[i2] = a_new
            i2 = i2 + 1

        return params_f64

    # Overload for Float32 tensors
    fn step_on_tensor_list_f32(
        mut self,
        mut params_f32: List[tensor.Tensor[Float32]],
        grads_f32: List[tensor.Tensor[Float32]]
    ) -> List[tensor.Tensor[Float32]]:
        var n = len(params_f32)
        assert(n == len(grads_f32))

        var cur = len(self._acc_tensors_f32)
        if n > cur:
            var i = cur
            while i < n:
                var shp = params_f32[i - cur].shape() if i >= cur else params_f32[i].shape()
                var acc0 = tensor.Tensor[Float32](shp, Float32(self.initial_accumulator_value))
                self._acc_tensors_f32.push_back(acc0)
                i = i + 1

        var i2 = 0
        while i2 < n:
            var p = params_f32[i2]
            var g = grads_f32[i2]
            var a = self._acc_tensors_f32[i2]
            var (p_new, a_new) = self.adagrad_update_tensor_f32(p, g, a)
            params_f32[i2] = p_new
            self._acc_tensors_f32[i2] = a_new
            i2 = i2 + 1

        return params_f32
