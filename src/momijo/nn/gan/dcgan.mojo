# Project:      Momijo
# Module:       src.momijo.nn.gan.dcgan
# File:         dcgan.mojo
# Path:         src/momijo/nn/gan/dcgan.mojo
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
#   - Structs: Linear, Generator, Discriminator, DCGAN
#   - Key functions: _sum1d, _max1d, _exp, _log1p, _log, sigmoid, tanh_like, act1d ...
#   - Uses generic functions/types with explicit trait bounds.


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
fn tanh_like(x: Float64) -> Float64:
    # 2*sigmoid(2x)-1
    var e = _exp(-2.0 * x)
    return 2.0 / (1.0 + e) - 1.0

# Elementwise activations
fn act1d(x: List[Float64]) -> List[Float64]:
    var y = List[Float64]()
    for v in x: y.push(tanh_like(v))
    return y

# --- Linear layer with bias ---
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
        for o in range(out_features): self.b.push(0.0)
fn forward(self, x: List[Float64]) -> List[Float64]:
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
# --- DCGAN Generator: z -> x ---
struct Generator:
    var l1: Linear
    var l2: Linear
fn __init__(out self, z_dim: Int, hidden_dim: Int, x_dim: Int) -> None:
        self.l1 = Linear(z_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, x_dim)
fn forward(self, z: List[Float64]) -> List[Float64]:
        return self.l2.forward(act1d(self.l1.forward(z)))
fn __copyinit__(out self, other: Self) -> None:
        self.l1 = other.l1
        self.l2 = other.l2
fn __moveinit__(out self, deinit other: Self) -> None:
        self.l1 = other.l1
        self.l2 = other.l2
# --- DCGAN Discriminator: x -> real/fake logit ---
struct Discriminator:
    var l1: Linear
    var l2: Linear
fn __init__(out self, x_dim: Int, hidden_dim: Int) -> None:
        self.l1 = Linear(x_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, 1)
fn forward(self, x: List[Float64]) -> Float64:
        var h = act1d(self.l1.forward(x))
        return self.l2.forward(h)[0]
fn __copyinit__(out self, other: Self) -> None:
        self.l1 = other.l1
        self.l2 = other.l2
fn __moveinit__(out self, deinit other: Self) -> None:
        self.l1 = other.l1
        self.l2 = other.l2
# --- Losses ---
fn bce_with_logits_single(logit: Float64, target_is_real: Bool) -> Float64:
    var p = sigmoid(logit)
    var y = 1.0
    if not target_is_real: y = 0.0
    if p < 1e-12: p = 1e-12
    if p > 1.0 - 1e-12: p = 1.0 - 1.0e-12
    return -(y * _log(p) + (1.0 - y) * _log(1.0 - p))

# --- DCGAN wrapper with simple steps ---
struct DCGAN:
    var G: Generator
    var D: Discriminator
fn __init__(out self, z_dim: Int, x_dim: Int, g_hidden: Int, d_hidden: Int) -> None:
        self.G = Generator(z_dim, g_hidden, x_dim)
        self.D = Discriminator(x_dim, d_hidden)

    # Generator wants D to think fakes are real
fn gen_step(self, z: List[Float64]) -> Float64:
        var x_fake = self.G.forward(z)
        var d_logit = self.D.forward(x_fake)
        return bce_with_logits_single(d_logit, True)

    # Discriminator: average real/fake losses
fn dis_step(self, x_real: List[Float64], z: List[Float64]) -> Float64:
        var d_logit_real = self.D.forward(x_real)
        var loss_real = bce_with_logits_single(d_logit_real, True)
        var x_fake = self.G.forward(z)
        var d_logit_fake = self.D.forward(x_fake)
        var loss_fake = bce_with_logits_single(d_logit_fake, False)
        return 0.5 * (loss_real + loss_fake)
fn __copyinit__(out self, other: Self) -> None:
        self.G = other.G
        self.D = other.D
fn __moveinit__(out self, deinit other: Self) -> None:
        self.G = other.G
        self.D = other.D
# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True

    var z = List[Float64]()
    for i in range(8): z.push(0.5)
    var x_real = List[Float64]()
    for i in range(12): x_real.push(1.0)

    var m = DCGAN(8, 12, 16, 16)
    var g_loss = m.gen_step(z)
    ok = ok and (g_loss >= 0.0)

    var d_loss = m.dis_step(x_real, z)
    ok = ok and (d_loss >= 0.0)

    # repeat to exercise path
    for t in range(3):
        var _g = m.gen_step(z)
        var _d = m.dis_step(x_real, z)
        ok = ok and (_g == _g) and (_d == _d)  # NaN guard placeholder

    return ok