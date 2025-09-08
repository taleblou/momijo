# Project:      Momijo
# Module:       src.momijo.nn.training.loop_gan
# File:         loop_gan.mojo
# Path:         src/momijo/nn/training/loop_gan.mojo
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
#   - Structs: GanParamsG, GanParamsD, GanLog
#   - Key functions: _abs, zeros1d_f, _exp, _log, _sigmoid, _mean_f, batch_z, batch_real ...
#   - Uses generic functions/types with explicit trait bounds.
#   - GPU/device utilities present; validate backend assumptions.


fn _abs(x: Float64) -> Float64:
    if x < 0.0: return -x
    return x
fn zeros1d_f(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y
fn _exp(x: Float64) -> Float64:
    # truncated series, OK for moderate |x|
    var term = 1.0
    var sum = 1.0
    var n = 1
    var k = 1.0
    while n <= 20:
        term *= x / k
        sum += term
        n += 1
        k += 1.0
    return sum
fn _log(x: Float64) -> Float64:
    if x <= 0.0: return -745.0
    var y = 0.0
    for it in range(8):
        var ey = _exp(y)
        y = y + 2.0 * (x - ey) / (x + ey)
    return y
fn _sigmoid(x: Float64) -> Float64:
    var e = _exp(-x)
    return 1.0 / (1.0 + e)
fn _mean_f(x: List[Float64]) -> Float64:
    var n = len(x)
    if n == 0: return 0.0
    var s = 0.0
    for i in range(n): s += x[i]
    return s / Float64(n)

# --------- Data (deterministic) ---------
fn batch_z(batch: Int) -> List[Float64]:
    var B = (batch if batch >= 2 else 2)
    var out = List[Float64]()
    for i in range(B):
        var t = Float64(i) / Float64(B - 1)
        out.push(-1.0 + 2.0 * t)        # uniform grid in [-1,1]
    return out
fn batch_real(batch: Int) -> List[Float64]:
    var B = (batch if batch >= 2 else 2)
    var out = List[Float64]()
    for i in range(B):
        var t = Float64(i) / Float64(B - 1)
        out.push(1.5 + 1.0 * t)         # uniform grid in [1.5, 2.5]
    return out

# --------- Model params ---------
struct GanParamsG:
    var a: Float64
    var b: Float64
fn __init__(out self, a: Float64 = 0.1, b: Float64 = 0.0) -> None:
        self.a = a
        self.b = b
fn __copyinit__(out self, other: Self) -> None:
        self.a = other.a
        self.b = other.b
fn __moveinit__(out self, deinit other: Self) -> None:
        self.a = other.a
        self.b = other.b
struct GanParamsD:
    var w: Float64
    var c: Float64
fn __init__(out self, w: Float64 = 0.0, c: Float64 = 0.0) -> None:
        self.w = w
        self.c = c
fn __copyinit__(out self, other: Self) -> None:
        self.w = other.w
        self.c = other.c
fn __moveinit__(out self, deinit other: Self) -> None:
        self.w = other.w
        self.c = other.c
# --------- Forward ---------
fn g_forward(p: GanParamsG, z: List[Float64]) -> List[Float64]:
    var B = len(z)
    var x = List[Float64]()
    for i in range(B): x.push(p.a * z[i] + p.b)
    return x
fn d_forward(p: GanParamsD, x: List[Float64]) -> List[Float64]:
    var B = len(x)
    var y = List[Float64]()
    for i in range(B):
        var s = p.w * x[i] + p.c
        y.push(_sigmoid(s))
    return y

# --------- Losses ---------
fn d_loss(real_p: List[Float64], fake_p: List[Float64]) -> Float64:
    var B = len(real_p)
    if B == 0: return 0.0
    var s = 0.0
    for i in range(B):
        var pr = real_p[i]
        var pf = fake_p[i]
        s += -_log(pr) - _log(1.0 - pf)
    return s / Float64(B)
fn g_loss(fake_p: List[Float64]) -> Float64:
    var B = len(fake_p)
    if B == 0: return 0.0
    var s = 0.0
    for i in range(B):
        var pf = fake_p[i]
        s += -_log(pf)
    return s / Float64(B)

# --------- One optimization step (SGD) ---------
fn d_step(mut d: GanParamsD, lr_d: Float64, x_real: List[Float64], x_fake: List[Float64]) -> GanParamsD:
    var B = len(x_real)
    if B == 0: return d
    var grad_w = 0.0
    var grad_c = 0.0
    for i in range(B):
        var sr = d.w * x_real[i] + d.c
        var pr = _sigmoid(sr)
        var dr = pr - 1.0               # d/ds of -log(pr)
        grad_w += dr * x_real[i]
        grad_c += dr
        var sf = d.w * x_fake[i] + d.c
        var pf = _sigmoid(sf)
        var df = pf                     # d/ds of -log(1-pf)
        grad_w += df * x_fake[i]
        grad_c += df
    grad_w = grad_w / Float64(B)
    grad_c = grad_c / Float64(B)
    d.w = d.w - lr_d * grad_w
    d.c = d.c - lr_d * grad_c
    return d
fn g_step(mut g: GanParamsG, d: GanParamsD, lr_g: Float64, z: List[Float64]) -> GanParamsG:
    var B = len(z)
    if B == 0: return g
    var grad_a = 0.0
    var grad_b = 0.0
    for i in range(B):
        var x = g.a * z[i] + g.b
        var s = d.w * x + d.c
        var p = _sigmoid(s)
        var dlds = p - 1.0              # for -log(D(G(z)))
        var dldx = dlds * d.w
        grad_a += dldx * z[i]
        grad_b += dldx
    grad_a = grad_a / Float64(B)
    grad_b = grad_b / Float64(B)
    g.a = g.a - lr_g * grad_a
    g.b = g.b - lr_g * grad_b
    return g

# --------- Training utilities ---------
struct GanLog:
    var d_losses: List[Float64]
    var g_losses: List[Float64]
    var mu_real: List[Float64]
    var mu_fake: List[Float64]
fn __init__(out self) -> None:
        self.d_losses = List[Float64]()
        self.g_losses = List[Float64]()
        self.mu_real = List[Float64]()
        self.mu_fake = List[Float64]()
fn __copyinit__(out self, other: Self) -> None:
        self.d_losses = other.d_losses
        self.g_losses = other.g_losses
        self.mu_real = other.mu_real
        self.mu_fake = other.mu_fake
fn __moveinit__(out self, deinit other: Self) -> None:
        self.d_losses = other.d_losses
        self.g_losses = other.g_losses
        self.mu_real = other.mu_real
        self.mu_fake = other.mu_fake
fn train_gan(epochs: Int = 300, batch_size: Int = 64, lr_d: Float64 = 0.05, lr_g: Float64 = 0.02) -> (GanParamsG, GanParamsD, GanLog):
    var g = GanParamsG(0.1, 0.0)
    var d = GanParamsD(0.0, 0.0)
    var log = GanLog()
    var B = (batch_size if batch_size >= 2 else 2)
    for ep in range(epochs):
        var z = batch_z(B)
        var xr = batch_real(B)
        var xf = g_forward(g, z)
        # D step
        var pr = d_forward(d, xr)
        var pf = d_forward(d, xf)
        var ld = d_loss(pr, pf)
        d = d_step(d, lr_d, xr, xf)
        # G step (recompute with updated D)
        xf = g_forward(g, z)
        pf = d_forward(d, xf)
        var lg = g_loss(pf)
        g = g_step(g, d, lr_g, z)
        # logging
        log.d_losses.push(ld)
        log.g_losses.push(lg)
        log.mu_real.push(_mean_f(xr))
        log.mu_fake.push(_mean_f(xf))
    return (g, d, log)

# --------- Self-test ---------
fn _self_test() -> Bool:
    var (g, d, log) = train_gan(220, 64, 0.05, 0.02)
    # expect generator near (a=0.5, b=2.0)
    var ok_a = _abs(g.a - 0.5) < 0.25
    var ok_b = _abs(g.b - 2.0) < 0.25
    # losses finite and last g-loss not exploding
    var last_g = log.g_losses[len(log.g_losses) - 1]
    var finite = (last_g > 0.0) and (last_g < 5.0)
    return ok_a and ok_b and finite