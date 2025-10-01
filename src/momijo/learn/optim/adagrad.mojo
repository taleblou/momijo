# Project:      Momijo
# Module:       learn.optim.adagrad
# File:         optim/adagrad.mojo
# Path:         src/momijo/learn/optim/adagrad.mojo
#
# Description:  Adagrad optimizer (backend-agnostic scaffolding).
#               Provides stable hyperparameters and stateless update kernels
#               you can wire to momijo.tensor when ready. Includes a minimal
#               sqrt implementation (no external math dep).
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
#   - Types: Adagrad
#   - Key fns: adagrad_update_scalar(...), step_on_lists(...)
#   - Backend-agnostic: replace scalar ops with tensor ops once momijo.tensor is ready.
#   - No globals, no inout; pure-return helpers for safe state threading.

from collections.list import List

# -----------------------------------------------------------------------------
# Minimal numerics (no stdlib.math guarantees)
# -----------------------------------------------------------------------------

fn _abs64(x: Float64) -> Float64:
    if x >= 0.0:
        return x
    return -x

fn _sqrt64(x: Float64, eps: Float64 = 1e-12) -> Float64:
    # Newton-Raphson for non-negative x. Returns 0 for x<=0.
    if x <= 0.0:
        return 0.0
    var g = x
    var prev = 0.0
    var iters = 0
    # 20 iterations is plenty for double precision here
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
# Adagrad per-parameter update for scalar values.
# Returns (new_param, new_accumulator)
#
# param: current parameter value
# grad:  gradient for the parameter
# acc:   historical sum of squared gradients (accumulator)
#
# Formula:
#   acc' = acc + g^2
#   update = lr * g / (sqrt(acc') + eps)
#   if weight_decay != 0: g = g + weight_decay * param
#   param' = param - update

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
    # Guard tiny denom to avoid NaN in extreme cases
    if denom <= 0.0:
        denom = eps
    var step = lr * g / denom
    var param_new = param - step
    return (param_new, acc_new)

# -----------------------------------------------------------------------------
# Optimizer
# -----------------------------------------------------------------------------

struct Adagrad:
    var lr: Float64
    var eps: Float64
    var weight_decay: Float64
    var initial_accumulator_value: Float64

    # Internal state for list-based stepping (one accumulator per param index).
    var _accumulators: List[Float64]

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

    # Ensure we have accumulator slots for n parameters.
    fn _ensure_acc_capacity(mut self, n: Int):
        var cur = Int(self._accumulators.size())
        if n > cur:
            var i = cur
            while i < n:
                self._accumulators.push_back(self.initial_accumulator_value)
                i = i + 1

    # -------------------------------------------------------------------------
    # Generic step() — left as a backend hook:
    # When you wire to momijo.tensor, bind self to real parameters and grads
    # so this no-arg step() can iterate and update them in-place. Until then,
    # prefer using step_on_lists(...) in demos/tests.
    # -------------------------------------------------------------------------
    fn step(mut self):
        # Placeholder: integrate with actual Parameter/Tensor containers.
        # Example (future):
        #   for p in self._params:
        #       var g = p.grad()
        #       var (new_val, new_acc) = adagrad_update_tensor(p.value(), g, acc, ...)
        #       p.set_value(new_val)
        #       acc = new_acc
        pass

    # Clear grads on the real parameter objects (future hook).
    fn zero_grad(mut self):
        # Placeholder: in real backend call p.zero_grad() for each param.
        pass

    # -------------------------------------------------------------------------
    # Immediate, usable API for scalar lists (helps smoke tests without tensors)
    # Updates params in-place and returns updated copy (also updates internal accs)
    # params[i] <- params[i] - lr * grad[i] / (sqrt(acc[i])+eps)
    # -------------------------------------------------------------------------
    fn step_on_lists(mut self, mut params: List[Float64], grads: List[Float64]) -> List[Float64]:
        var n = Int(params.size())
        assert(n == Int(grads.size()))
        self._ensure_acc_capacity(n)

        var i = 0
        while i < n:
            var p = params[i]
            var g = grads[i]
            var acc = self._accumulators[i]

            var (p_new, acc_new) = adagrad_update_scalar(
                p, g, acc, self.lr, self.eps, self.weight_decay
            )
            # write back
            params[i] = p_new
            self._accumulators[i] = acc_new
            i = i + 1

        return params

    # Reset accumulators (useful when re-starting training)
    fn reset_state(mut self):
        self._accumulators = List[Float64]()
