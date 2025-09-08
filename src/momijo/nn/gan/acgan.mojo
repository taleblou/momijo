# Project:      Momijo
# Module:       src.momijo.nn.gan.acgan
# File:         acgan.mojo
# Path:         src/momijo/nn/gan/acgan.mojo
#
# Description:  Neural-network utilities for Momijo integrating with tensors,
#               optimizers, and training/evaluation loops.
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
#   - Structs: Linear, Generator, DiscriminatorAux, ACGAN
#   - Key functions: _abs, _sum1d, _max1d, _exp, _log1p, _log, sigmoid, softmax1d ...
#   - Uses generic functions/types with explicit trait bounds.


fn _abs(x: Float64) -> Float64:
    if x >= 0.0: return x
    return -x
fn _sum1d(xs: List[Float64]) -> Float64:
    var s = 0.0
    for v in xs: s += v
    return s
fn _max1d(xs: List[Float64]) -> Float64:
    var m = -1.7976931348623157e308
    for v in xs:
        if v > m: m = v
    return m
fn _exp(x: Float64) -> Float64:
    var term = 1.0
    var sum = 1.0
    var n = 1
    var k = 1.0
    while n <= 12:
        term *= x / k
        sum += term
        n += 1
        k += 1.0
    return sum
fn _log1p(y: Float64) -> Float64:
    var x = y
    if x <= -0.999999999999: x = -0.999999999999
    var term = x
    var res = term
    var n = 2.0
    var sign = -1.0
    for i in range(9):
        term = term * x
        res += sign * term / n
        n += 1.0
        sign = -sign
    return res
fn _log(x: Float64) -> Float64:
    return _log1p(x - 1.0)
fn sigmoid(x: Float64) -> Float64:
    var e = _exp(-x)
    return 1.0 / (1.0 + e)
fn softmax1d(logits: List[Float64]) -> List[Float64]:
    var m = _max1d(logits)
    var exps = List[Float64]()
    for v in logits: exps.push(_exp(v - m))
    var s = _sum1d(exps)
    var out = List[Float64]()
    if s == 0.0:
        var n = len(exps)
        var p = 1.0 / Float64(n)
        for i in range(n): out.push(p)
        return out
    for e in exps: out.push(e / s)
    return out
fn cross_entropy_logits1d(logits: List[Float64], target_index: Int) -> Float64:
    var n = len(logits)
    if n == 0: return 0.0
    var m = _max1d(logits)
    var exps = List[Float64]()
    for v in logits: exps.push(_exp(v - m))
    var s = _sum1d(exps)
    if s == 0.0: return 0.0
    var logZ = _log(s) + m
    if target_index < 0: target_index = 0
    if target_index >= n: target_index = n - 1
    return -(logits[target_index] - logZ)

# --- Simple linear layer with bias ---
struct Linear:
    var in_features: Int
    var out_features: Int
    var W: List[List[Float64]]  # [out, in]
    var b: List[Float64]        # [out]
fn __init__(out self, in_features: Int, out_features: Int, weight: Float64 = 0.01) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.W = List[List[Float64]]()
        for o in range(out_features):
            var row = List[Float64]()
            for i in range(in_features):
                row.push(weight)  # deterministic tiny init
            self.W.push(row)
        self.b = List[Float64]()
        for o in range(out_features):
            self.b.push(0.0)
fn forward(self, x: List[Float64]) -> List[Float64]:
        # x: [in_features] -> y: [out_features]
        var y = List[Float64]()
        for o in range(self.out_features):
            var acc = self.b[o]
            for i in range(self.in_features):
                acc += self.W[o][i] * x[i]
            y.push(acc)
        return y
fn __copyinit__(out self, other: Self) -> None:
        self.in_features = other.in_features
        self.out_features = other.out_features
        self.W = other.W
        self.b = other.b
fn __moveinit__(out self, deinit other: Self) -> None:
        self.in_features = other.in_features
        self.out_features = other.out_features
        self.W = other.W
        self.b = other.b
# --- Generator: z + class_one_hot -> x_fake (vector) ---
struct Generator:
    var z_dim: Int
    var num_classes: Int
    var hidden: Linear
    var out: Linear
fn __init__(out self, z_dim: Int, num_classes: Int, hidden_dim: Int, out_dim: Int) -> None:
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.hidden = Linear(z_dim + num_classes, hidden_dim)
        self.out = Linear(hidden_dim, out_dim)
fn _concat(self, z: List[Float64], class_index: Int) -> List[Float64]:
        var v = List[Float64]()
        # append z
        for i in range(len(z)): v.push(z[i])
        # one-hot
        var k = self.num_classes
        for i in range(k):
            if i == class_index: v.push(1.0) else: v.push(0.0)
        return v
fn forward(self, z: List[Float64], class_index: Int) -> List[Float64]:
        var h = self.hidden.forward(self._concat(z, class_index))
        # simple nonlinearity (tanh-like via scaled sigmoid)
        var a = List[Float64]()
        for v in h:
            a.push(2.0 * sigmoid(2.0 * v) - 1.0)
        return self.out.forward(a)
fn __copyinit__(out self, other: Self) -> None:
        self.z_dim = other.z_dim
        self.num_classes = other.num_classes
        self.hidden = other.hidden
        self.out = other.out
fn __moveinit__(out self, deinit other: Self) -> None:
        self.z_dim = other.z_dim
        self.num_classes = other.num_classes
        self.hidden = other.hidden
        self.out = other.out
# --- Discriminator + Auxiliary classifier ---
struct DiscriminatorAux:
    var in_dim: Int
    var num_classes: Int
    var feat: Linear
    var adv_head: Linear
    var cls_head: Linear
fn __init__(out self, in_dim: Int, hidden_dim: Int, num_classes: Int) -> None:
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.feat = Linear(in_dim, hidden_dim)
        self.adv_head = Linear(hidden_dim, 1)            # real/fake logit
        self.cls_head = Linear(hidden_dim, num_classes)  # class logits
fn forward(self, x: List[Float64]) -> (Float64, List[Float64]):
        var h = self.feat.forward(x)
        var a = List[Float64]()
        for v in h: a.push(2.0 * sigmoid(2.0 * v) - 1.0)
        var adv_logit = self.adv_head.forward(a)[0]
        var cls_logits = self.cls_head.forward(a)
        return (adv_logit, cls_logits)
fn __copyinit__(out self, other: Self) -> None:
        self.in_dim = other.in_dim
        self.num_classes = other.num_classes
        self.feat = other.feat
        self.adv_head = other.adv_head
        self.cls_head = other.cls_head
fn __moveinit__(out self, deinit other: Self) -> None:
        self.in_dim = other.in_dim
        self.num_classes = other.num_classes
        self.feat = other.feat
        self.adv_head = other.adv_head
        self.cls_head = other.cls_head
# --- Losses for ACGAN ---
fn adversarial_bce_with_logits(logit: Float64, target_is_real: Bool) -> Float64:
    # BCE(logit, y) with y in {0,1} using sigmoid + BCE
    var p = sigmoid(logit)
    var y = 1.0
    if not target_is_real: y = 0.0
    # clamp
    if p < 1e-12: p = 1e-12
    if p > 1.0 - 1e-12: p = 1.0 - 1e-12
    return -(y * _log(p) + (1.0 - y) * _log(1.0 - p))
fn auxiliary_ce_logits(logits: List[Float64], target_class: Int) -> Float64:
    return cross_entropy_logits1d(logits, target_class)

# --- Toy training step (no real optimizer) ---
struct ACGAN:
    var G: Generator
    var D: DiscriminatorAux
    var lambda_cls: Float64
fn __init__(out self, z_dim: Int, num_classes: Int, g_hidden: Int, d_hidden: Int, x_dim: Int, lambda_cls: Float64 = 1.0) -> None:
        self.G = Generator(z_dim, num_classes, g_hidden, x_dim)
        self.D = DiscriminatorAux(x_dim, d_hidden, num_classes)
        self.lambda_cls = lambda_cls
fn gen_step(self, z: List[Float64], y: Int) -> (Float64, Float64):
        # Forward
        var x_fake = self.G.forward(z, y)
        var (d_logit, d_cls) = self.D.forward(x_fake)
        # Losses: make discriminator think fake is real + right class
        var adv = adversarial_bce_with_logits(d_logit, True)
        var aux = auxiliary_ce_logits(d_cls, y)
        # No optimizer here; return losses
        return (adv, aux)
fn dis_step(self, x_real: List[Float64], y_real: Int, z: List[Float64], y_fake: Int) -> (Float64, Float64):
        # Real pass
        var (d_logit_r, d_cls_r) = self.D.forward(x_real)
        var adv_r = adversarial_bce_with_logits(d_logit_r, True)
        var aux_r = auxiliary_ce_logits(d_cls_r, y_real)
        # Fake pass
        var x_fake = self.G.forward(z, y_fake)
        var (d_logit_f, d_cls_f) = self.D.forward(x_fake)
        var adv_f = adversarial_bce_with_logits(d_logit_f, False)
        # Commonly, no aux on fake for D; we keep it simple and ignore aux_f.
        # Total
        return ( (adv_r + adv_f) * 0.5, aux_r )
fn __copyinit__(out self, other: Self) -> None:
        self.G = other.G
        self.D = other.D
        self.lambda_cls = other.lambda_cls
fn __moveinit__(out self, deinit other: Self) -> None:
        self.G = other.G
        self.D = other.D
        self.lambda_cls = other.lambda_cls
# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True

    var z = List[Float64]()
    for i in range(8): z.push(0.5)  # deterministic latent

    var ac = ACGAN(8, 3, 16, 16, 10, 1.0)

    # Generator step
    var (g_adv, g_aux) = ac.gen_step(z, 1)
    ok = ok and (g_adv > 0.0) and (g_aux >= 0.0)

    # Discriminator step
    var x_real = List[Float64]()
    for i in range(10): x_real.push(1.0)
    var (d_adv, d_aux) = ac.dis_step(x_real, 2, z, 0)
    ok = ok and (d_adv > 0.0) and (d_aux >= 0.0)

    return ok