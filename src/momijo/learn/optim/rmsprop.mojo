# Project:      Momijo
# Module:       learn.optim.rmsprop
# File:         optim/rmsprop.mojo
# Path:         src/momijo/learn/optim/rmsprop.mojo
#
# Description:  RMSprop optimizer with optional momentum, weight decay, and centered variant.
#               Backend-agnostic skeleton: operates on a list of parameters that each hold
#               a value and gradient. Replace Float64 with Tensor types once momijo.tensor
#               is wired in (duck-typed API kept stable: step()/zero_grad()).
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
#   - Types: OptParam (placeholder scalar param), RMSprop (optimizer)
#   - Hyperparams:
#       lr            : base learning rate
#       alpha         : EMA decay for squared grads (typ. 0.99)
#       eps           : epsilon for numerical stability
#       weight_decay  : L2 penalty (decoupled = False ⇒ classic, True ⇒ AdamW-style)
#       momentum      : momentum factor (0 to disable)
#       centered      : if True, use grad^2 - (EMA(grad))^2 in denom
#       decoupled     : if True, decouple weight decay from grad (AdamW-style)
#   - Update (classic, non-decoupled):
#       sq = alpha * sq + (1 - alpha) * g^2
#       if centered: mg = alpha * mg + (1 - alpha) * g; denom = sqrt(sq - mg^2 + eps)
#       if momentum: buf = momentum * buf + (g / denom)
#         p -= lr * buf
#       else:
#         p -= lr * (g / denom)
#       if weight_decay and not decoupled: g += weight_decay * p
#     Decoupled weight decay:
#       p *= (1 - lr * weight_decay)
#
#   - This file uses Float64 placeholders; swap to Tensor later (keep method signatures).

from collections.list import List

# -----------------------------------------------------------------------------
# Placeholder parameter type
# Replace Float64 with momijo.tensor.Tensor later (value, grad)
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

    # Parameters and state (parallel to params)
    var params: List[OptParam]
    var square_avg: List[Float64]   # E[g^2]
    var grad_avg: List[Float64]     # momentum buffer (v)
    var mean_grad: List[Float64]    # E[g] for centered variant

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

    # Register parameters (placeholder). In real integration, pass Module/Tensor params.
    fn set_params(mut self, ps: List[OptParam]):
        self.params = ps

        # (Re)initialize state to correct length
        self.square_avg = List[Float64]()
        self.grad_avg = List[Float64]()
        self.mean_grad = List[Float64]()
        var n = Int(ps.size())
        var i = 0
        while i < n:
            self.square_avg.push_back(0.0)
            self.grad_avg.push_back(0.0)
            self.mean_grad.push_back(0.0)
            i = i + 1

    # Zero gradients for all parameters (common optimizer API)
    fn zero_grad(mut self):
        var n = Int(self.params.size())
        var i = 0
        while i < n:
            var p = self.params[i]
            p.grad = 0.0
            self.params[i] = p
            i = i + 1

    # One optimization step.
    # Algorithm matches PyTorch RMSprop semantics (approx.), including options:
    # - momentum (buffered update)
    # - centered (E[g^2] - (E[g])^2)
    # - decoupled weight decay (AdamW-style) vs classic L2 in grad
    fn step(mut self):
        var n = Int(self.params.size())
        var i = 0

        while i < n:
            var p = self.params[i]
            var g = p.grad

            # Optional classic L2: add to gradient (non-decoupled)
            if self.weight_decay != 0.0 and not self.decoupled:
                g = g + self.weight_decay * p.value

            # Update EMA of squared gradients
            var sq = self.square_avg[i]
            sq = self.alpha * sq + (1.0 - self.alpha) * (g * g)
            self.square_avg[i] = sq

            # Centered RMSprop: maintain E[g]
            var denom: Float64
            if self.centered:
                var mg = self.mean_grad[i]
                mg = self.alpha * mg + (1.0 - self.alpha) * g
                self.mean_grad[i] = mg
                var var_g = sq - (mg * mg)
                if var_g < 0.0:
                    # Numerical clamp
                    var_g = 0.0
                denom = (var_g + self.eps) ** 0.5
            else:
                denom = (sq + self.eps) ** 0.5

            # Momentum buffer (if any)
            if self.momentum != 0.0:
                var v = self.grad_avg[i]
                v = self.momentum * v + g / denom
                self.grad_avg[i] = v
                # Parameter update
                p.value = p.value - self.lr * v
            else:
                # Plain RMSprop
                p.value = p.value - self.lr * (g / denom)

            # Decoupled weight decay (AdamW-style) after gradient step
            if self.weight_decay != 0.0 and self.decoupled:
                p.value = p.value * (1.0 - self.lr * self.weight_decay)

            # Write back param
            self.params[i] = p
            i = i + 1
