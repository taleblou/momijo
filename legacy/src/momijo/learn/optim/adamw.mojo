# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo
# File:         src/momijo/learn/optim/adamw.mojo
# Description:  AdamW optimizer for Linear/Conv2d layers (Float64 tensors).
# Notes:
#   - Decoupled weight decay (AdamW): p = p - lr * m_hat/sqrt(v_hat+eps) - lr * wd * p
#   - Matches Momijo layers: Linear.{weight: [out,in], bias_t: [out]}, Conv2d likewise.
#   - English-only comments. No optionals in signatures.

from collections.list import List
from momijo.tensor import tensor
from momijo.learn.nn.layers import Linear
from momijo.learn.nn.conv import Conv2d

# ----------------------------- small helpers --------------------------------

@always_inline
fn _numel(shape: List[Int]) -> Int:
    var p = 1
    var i = 0
    var n = len(shape)
    while i < n:
        p = p * shape[i]
        i = i + 1
    return p

@always_inline
fn _pow_scalar(x: Float64, n: Int) -> Float64:
    var r = 1.0
    var i = 0
    while i < n:
        r = r * x
        i = i + 1
    return r

# ------------------------------ AdamW state ----------------------------------

struct AdamW:
    # Hyperparameters
    var lr: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var weight_decay: Float64

    # Moments (allocated lazily on first step per layer shape)
    var mW: tensor.Tensor[Float64]
    var vW: tensor.Tensor[Float64]
    var mB: tensor.Tensor[Float64]
    var vB: tensor.Tensor[Float64]
    var t: Int
    var _init: Bool

    fn __init__(
        out self,
        lr: Float64 = 1e-3,
        beta1: Float64 = 0.9,
        beta2: Float64 = 0.999,
        eps: Float64 = 1e-8,
        weight_decay: Float64 = 0.0
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.mW = tensor.zeros([1, 1])
        self.vW = tensor.zeros([1, 1])
        self.mB = tensor.zeros([1])
        self.vB = tensor.zeros([1])
        self.t = 0
        self._init = False

    fn __copyinit__(out self, other: Self):
        self.lr = other.lr
        self.beta1 = other.beta1
        self.beta2 = other.beta2
        self.eps = other.eps
        self.weight_decay = other.weight_decay
        self.mW = other.mW.copy()
        self.vW = other.vW.copy()
        self.mB = other.mB.copy()
        self.vB = other.vB.copy()
        self.t = other.t
        self._init = other._init

    # Optional setter to adjust hyperparameters (no Optional syntax)
    fn set_hyperparams(
        mut self,
        lr: Float64,
        beta1: Float64,
        beta2: Float64,
        eps: Float64,
        weight_decay: Float64
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

    # -------------------------- per-layer steps ------------------------------

    fn step_linear(
        mut self,
        mut layer: Linear,
        dW: tensor.Tensor[Float64],
        db: tensor.Tensor[Float64]
    ):
        # Lazy state init on first use (match grad shapes)
        if not self._init:
            self.mW = tensor.zeros_like(dW); self.vW = tensor.zeros_like(dW)
            self.mB = tensor.zeros_like(db); self.vB = tensor.zeros_like(db)
            self.t = 0
            self._init = True

        self.t = self.t + 1
        var b1 = self.beta1; var b2 = self.beta2
        var bc1 = 1.0 - _pow_scalar(b1, self.t)
        var bc2 = 1.0 - _pow_scalar(b2, self.t)

        # Moments
        self.mW = b1 * self.mW + (1.0 - b1) * dW
        self.vW = b2 * self.vW + (1.0 - b2) * (dW * dW)
        self.mB = b1 * self.mB + (1.0 - b1) * db
        self.vB = b2 * self.vB + (1.0 - b2) * (db * db)

        var mW_hat = self.mW / bc1
        var vW_hat = self.vW / bc2
        var mB_hat = self.mB / bc1
        var vB_hat = self.vB / bc2

        # Adam step (elementwise)
        var stepW = mW_hat / (vW_hat.sqrt() + self.eps)
        var newW = layer.weight - self.lr * stepW
        # Decoupled weight decay
        if self.weight_decay != 0.0:
            newW = newW - self.lr * self.weight_decay * layer.weight
        layer.weight = newW.copy()

        if layer.bias:
            var stepB = mB_hat / (vB_hat.sqrt() + self.eps)
            var newB = layer.bias_t - self.lr * stepB
            if self.weight_decay != 0.0:
                newB = newB - self.lr * self.weight_decay * layer.bias_t
            layer.bias_t = newB.copy()

    fn step_conv2d(
        mut self,
        mut layer: Conv2d,
        dW: tensor.Tensor[Float64],
        db: tensor.Tensor[Float64]
    ):
        if not self._init:
            self.mW = tensor.zeros_like(dW); self.vW = tensor.zeros_like(dW)
            self.mB = tensor.zeros_like(db); self.vB = tensor.zeros_like(db)
            self.t = 0
            self._init = True

        self.t = self.t + 1
        var b1 = self.beta1; var b2 = self.beta2
        var bc1 = 1.0 - _pow_scalar(b1, self.t)
        var bc2 = 1.0 - _pow_scalar(b2, self.t)

        self.mW = b1 * self.mW + (1.0 - b1) * dW
        self.vW = b2 * self.vW + (1.0 - b2) * (dW * dW)
        self.mB = b1 * self.mB + (1.0 - b1) * db
        self.vB = b2 * self.vB + (1.0 - b2) * (db * db)

        var mW_hat = self.mW / bc1
        var vW_hat = self.vW / bc2
        var stepW = mW_hat / (vW_hat.sqrt() + self.eps)

        var newW = layer.weight - self.lr * stepW
        if self.weight_decay != 0.0:
            newW = newW - self.lr * self.weight_decay * layer.weight
        layer.weight = newW

        if layer.bias:
            var mB_hat = self.mB / bc1
            var vB_hat = self.vB / bc2
            var stepB = mB_hat / (vB_hat.sqrt() + self.eps)
            var newB = layer.bias_t - self.lr * stepB
            if self.weight_decay != 0.0:
                newB = newB - self.lr * self.weight_decay * layer.bias_t
            layer.bias_t = newB

    # ------------------------------ utils ------------------------------------

    fn zero_state(mut self):
        # Reset moments/time, keep hyperparameters
        self.mW = tensor.zeros([1, 1]); self.vW = tensor.zeros([1, 1])
        self.mB = tensor.zeros([1]);    self.vB = tensor.zeros([1])
        self.t = 0
        self._init = False

    fn __str__(self) -> String:
        var s = String("AdamW(")
        s = s + "lr=" + String(self.lr)
        s = s + ", beta1=" + String(self.beta1)
        s = s + ", beta2=" + String(self.beta2)
        s = s + ", eps=" + String(self.eps)
        s = s + ", wd=" + String(self.weight_decay)
        s = s + ", t=" + String(self.t)
        s = s + ")"
        return s
