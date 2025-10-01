# Project:      Momijo
# Module:       learn.optim.sgd
# File:         optim/sgd.mojo
# Path:         src/momijo/learn/optim/sgd.mojo
#
# Description:  Stochastic Gradient Descent (SGD) optimizer with momentum,
#               dampening, weight decay (L2), and optional Nesterov momentum.
#               Backend-agnostic via the ParamSGD trait; any parameter type
#               implementing ParamSGD can be updated by this optimizer.
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
#   - Types: ParamSGD (trait), SGD (optimizer)
#   - Key fns: SGD.step_for(params), SGD.zero_grad_for(params)
#   - Configure: lr, momentum, dampening, weight_decay, nesterov
#   - This file is backend-agnostic; wire actual tensor math in your ParamSGD implementations.

from collections.list import List

# -----------------------------------------------------------------------------
# Contract for any optimizable parameter used by SGD
# -----------------------------------------------------------------------------
# Implement this trait for your parameter type (e.g., a Tensor-backed Parameter).
# The implementer should perform the in-place update using the provided hyperparams.
#
# Required behavior in apply_sgd (typical formula):
#   grad'       = grad + weight_decay * param       # L2 regularization (if weight_decay > 0)
#   v           = momentum * v + (1 - dampening) * grad'
#   step_dir    = (momentum * v + (1 - dampening) * grad') if nesterov else v
#   param      -= lr * step_dir
#   (internal state `v` must be maintained by the parameter implementation)
#
# zero_grad should clear the stored gradient (e.g., set to zeros / None).
#
trait ParamSGD:
    fn apply_sgd(
        mut self,
        lr: Float64,
        momentum: Float64,
        dampening: Float64,
        weight_decay: Float64,
        nesterov: Bool
    )

    fn zero_grad(mut self)


# -----------------------------------------------------------------------------
# SGD Optimizer
# -----------------------------------------------------------------------------
struct SGD:
    var lr: Float64
    var momentum: Float64
    var dampening: Float64
    var weight_decay: Float64
    var nesterov: Bool

    fn __init__(
        out self,
        lr: Float64 = 0.01,
        momentum: Float64 = 0.0,
        dampening: Float64 = 0.0,
        weight_decay: Float64 = 0.0,
        nesterov: Bool = False
    ):
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

    # Backward-compat no-arg step (does nothing by itself).
    # Use step_for(...) to update a parameter list.
    fn step(mut self):
        # Intentionally left as a no-op. Call step_for(params) instead.
        pass

    # Update a list of parameters implementing ParamSGD.
    fn step_for[T: ParamSGD](mut self, params: List[T]):
        var n = Int(params.size())
        var i = 0
        while i < n:
            # NOTE: Element access returns a value; implementers of ParamSGD
            # should mutate underlying storage (e.g., hold a handle to real data).
            var p = params[i]
            p.apply_sgd(self.lr, self.momentum, self.dampening, self.weight_decay, self.nesterov)
            i = i + 1

    # Zero gradients for a list of parameters.
    fn zero_grad_for[T: ParamSGD](mut self, params: List[T]):
        var n = Int(params.size())
        var i = 0
        while i < n:
            var p = params[i]
            p.zero_grad()
            i = i + 1

    # Optional: update hyperparameters at runtime (useful for schedulers).
    fn set_lr(mut self, new_lr: Float64):
        self.lr = new_lr

    fn get_lr(self) -> Float64:
        return self.lr
