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
fn _pow_scalar(x: Float32, n: Int) -> Float32:
    var r :Float32= 1.0
    var i = 0
    while i < n:
        r = r * x
        i = i + 1
    return r.copy()

# ------------------------------ AdamW state ----------------------------------

struct Adam:
    # Hyperparameters
    var lr: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32

    # Moments (allocated lazily on first step per layer shape)
    var mW: tensor.Tensor[Float32]
    var vW: tensor.Tensor[Float32]
    var mB: tensor.Tensor[Float32]
    var vB: tensor.Tensor[Float32]
    var t: Int
    var _init: Bool

    fn __init__(
        out self,
        lr: Float32 = 1e-3,
        beta1: Float32 = 0.9,
        beta2: Float32 = 0.999,
        eps: Float32 = 1e-8,
        weight_decay: Float32 = 0.0
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
        lr: Float32,
        beta1: Float32,
        beta2: Float32,
        eps: Float32,
        weight_decay: Float32
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
        dW: tensor.Tensor[Float32],
        db: tensor.Tensor[Float32]
    ):

        # --- اطمینان از هم‌خوانی شکل state با گرادیان‌ها؛ در غیر این‌صورت reinit ---
        var need_reinit_W = False
        if not self._init:
            need_reinit_W = True
        else:
            var s_mW = self.mW.shape(); var s_dW = dW.shape()
            if _numel(s_mW) != _numel(s_dW) or _numel(s_mW) == 0:
                need_reinit_W = True
        if need_reinit_W:
            self.mW = tensor.zeros_like(dW)
            self.vW = tensor.zeros_like(dW)

        var need_reinit_B = False
        if layer.bias:
            if not self._init:
                need_reinit_B = True
            else:
                var s_mB = self.mB.shape(); var s_db = db.shape()
                if _numel(s_mB) != _numel(s_db) or _numel(s_mB) == 0:
                    need_reinit_B = True
            if need_reinit_B:
                self.mB = tensor.zeros_like(db)
                self.vB = tensor.zeros_like(db)

        if not self._init:
            self.t = 0
            self._init = True


        self.t = self.t + 1

        var b1 = self.beta1; var b2 = self.beta2
        var bc1 = 1.0 - _pow_scalar(b1, self.t)
        var bc2 = 1.0 - _pow_scalar(b2, self.t)

        # --- ممان‌ها ---
        self.mW = b1 * self.mW + (1.0 - b1) * dW
        self.vW = b2 * self.vW + (1.0 - b2) * (dW * dW)
        if layer.bias:
            self.mB = b1 * self.mB + (1.0 - b1) * db
            self.vB = b2 * self.vB + (1.0 - b2) * (db * db)

        # --- تصحیح بایاس و گام ---
        var mW_hat = self.mW / bc1
        var vW_hat = self.vW / bc2
        var stepW = mW_hat / (vW_hat.sqrt() + self.eps)

        var newW = layer.weight - self.lr * stepW
        if self.weight_decay != 0.0:
            newW = newW - self.lr * self.weight_decay * layer.weight
        layer.weight = newW.copy()

        if layer.bias:
            var mB_hat = self.mB / bc1
            var vB_hat = self.vB / bc2
            var stepB = mB_hat / (vB_hat.sqrt() + self.eps)
            var newB = layer.bias_t - self.lr * stepB
            if self.weight_decay != 0.0:
                newB = newB - self.lr * self.weight_decay * layer.bias_t
            layer.bias_t = newB.copy()
        else:
            print("  (no bias branch)")


    fn step_conv2d(
        mut self,
        mut layer: Conv2d,
        dW: tensor.Tensor[Float32],
        db: tensor.Tensor[Float32]
    ):


        # --- (A) اطمینان از هم‌خوانی شکل state با گرادیان وزن ---
        var need_reinit_W = False
        if not self._init:
            need_reinit_W = True
        else:
            # اگر قبلاً init شده اما شکل‌ها نمی‌خوانند یا numel صفر است
            var s_mW = self.mW.shape()
            var s_dW = dW.shape()
            if _numel(s_mW) != _numel(s_dW) or _numel(s_mW) == 0:
                need_reinit_W = True
        if need_reinit_W:
            self.mW = tensor.zeros_like(dW)
            self.vW = tensor.zeros_like(dW)

        # --- (B) اطمینان از هم‌خوانی شکل state با گرادیان بایاس ---
        var need_reinit_B = False
        if layer.bias:
            if not self._init:
                need_reinit_B = True
            else:
                var s_mB = self.mB.shape()
                var s_db = db.shape()
                if _numel(s_mB) != _numel(s_db) or _numel(s_mB) == 0:
                    need_reinit_B = True
            if need_reinit_B:
                self.mB = tensor.zeros_like(db)
                self.vB = tensor.zeros_like(db)

        # اگر اولین بار هر دو reinit شدن، فلگ init رو ست کن
        if not self._init:
            self.t = 0
            self._init = True

        self.t = self.t + 1

        var b1 = self.beta1; var b2 = self.beta2
        var bc1 = 1.0 - _pow_scalar(b1, self.t)
        var bc2 = 1.0 - _pow_scalar(b2, self.t)

        self.mW = b1 * self.mW + (1.0 - b1) * dW
        self.vW = b2 * self.vW + (1.0 - b2) * (dW * dW)
        var mW_hat = self.mW / bc1
        var vW_hat = self.vW / bc2
        var stepW = mW_hat / (vW_hat.sqrt() + self.eps)

        var newW = layer.weight - self.lr * stepW
        if self.weight_decay != 0.0:
            newW = newW - self.lr * self.weight_decay * layer.weight
        layer.weight = newW.copy()

        if layer.bias:
            self.mB = b1 * self.mB + (1.0 - b1) * db
            self.vB = b2 * self.vB + (1.0 - b2) * (db * db)
            var mB_hat = self.mB / bc1
            var vB_hat = self.vB / bc2
            var stepB = mB_hat / (vB_hat.sqrt() + self.eps)
            var newB = layer.bias_t - self.lr * stepB
            if self.weight_decay != 0.0:
                newB = newB - self.lr * self.weight_decay * layer.bias_t
            layer.bias_t = newB.copy()
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



fn _safe_scalar(x: tensor.Tensor[Float64], tag: String) -> String:
    # سعی می‌کنیم یک خلاصه عددی بدهیم (بدون ایندکس چندبعدی)
    var s = x.shape()
    var nelem = 1
    var i = 0
    while i < len(s):
        nelem = nelem * s[i]
        i += 1
    var info = String("[") + s.__str__() + String("]")
    # اگر تابع sum_all دارید:
    var sumv = x.sum_all()
    info = info + String(" sum=") + String(sumv)
    info = info + String(" n=") + String(nelem)
    return tag + String(": ") + info
