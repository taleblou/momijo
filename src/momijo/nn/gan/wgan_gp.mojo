# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.nn.gan
# File: src/momijo/nn/gan/wgan_gp.mojo

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
fn tanh_like(x: Float64) -> Float64:
    var e = _exp(-2.0 * x)
    return 2.0 / (1.0 + e) - 1.0
fn act1d(x: List[Float64]) -> List[Float64]:
    var y = List[Float64]()
    for v in x: y.push(tanh_like(v))
    return y
fn add1d(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var n = len(a)
    var y = List[Float64]()
    for i in range(n): y.push(a[i] + b[i])
    return y
fn scale1d(a: List[Float64], s: Float64) -> List[Float64]:
    var n = len(a)
    var y = List[Float64]()
    for i in range(n): y.push(a[i] * s)
    return y
fn mix1d(a: List[Float64], b: List[Float64], t: Float64) -> List[Float64]:
    var n = len(a)
    var y = List[Float64]()
    for i in range(n): y.push((1.0 - t) * a[i] + t * b[i])
    return y
fn unit_dir(n: Int) -> List[Float64]:
    var u = List[Float64]()
    var s = 1.0
    for i in range(n):
        u.push(s); s = -s
    var inv = 1.0
    if n > 0: inv = 1.0 / (Float64(n) ** 0.5)
    return scale1d(u, inv)
fn grad_norm_proxy(f_plus: Float64, f_minus: Float64, eps: Float64) -> Float64:
    var directional = (f_plus - f_minus) / (2.0 * eps)
    if directional < 0.0: directional = -directional
    return directional

# --- Linear layer ---
struct Linear:
    var in_features: Int
    var out_features: Int
    var W: List[List[Float64]]
    var b: List[Float64]
fn __init__(out self, in_features: Int, out_features: Int, weight: Float64 = 0.01) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.W = List[List[Float64]]()
        for o in range(out_features):
            var row = List[Float64]()
            for i in range(in_features): row.push(weight)
            self.W.push(row)
        self.b = List[Float64]()
        for o in range(out_features): self.b.push(0.0)
fn forward(self, x: List[Float64]) -> List[Float64]:
        var y = List[Float64]()
        for o in range(self.out_features):
            var acc = self.b[o]
            for i in range(self.in_features): acc += self.W[o][i] * x[i]
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
# --- Generator & Critic ---
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
struct Critic:
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
# --- WGAN-GP wrapper ---
struct WGAN_GP:
    var G: Generator
    var C: Critic
    var lambda_gp: Float64
    var eps_fd: Float64
fn __init__(out self, z_dim: Int, x_dim: Int, g_hidden: Int, c_hidden: Int, lambda_gp: Float64 = 10.0, eps_fd: Float64 = 1e-3) -> None:
        self.G = Generator(z_dim, g_hidden, x_dim)
        self.C = Critic(x_dim, c_hidden)
        self.lambda_gp = lambda_gp
        self.eps_fd = eps_fd
fn gen_step(self, z: List[Float64]) -> Float64:
        var x_fake = self.G.forward(z)
        var c_fake = self.C.forward(x_fake)
        return -c_fake
fn _gp(self, x_real: List[Float64], x_fake: List[Float64], t: Float64) -> Float64:
        var x_hat = mix1d(x_real, x_fake, t)
        var u = unit_dir(len(x_hat))
        var x_plus = add1d(x_hat, scale1d(u, self.eps_fd))
        var x_minus = add1d(x_hat, scale1d(u, -self.eps_fd))
        var f_plus = self.C.forward(x_plus)
        var f_minus = self.C.forward(x_minus)
        var g = grad_norm_proxy(f_plus, f_minus, self.eps_fd)
        var diff = g - 1.0
        return self.lambda_gp * diff * diff
fn dis_step(self, x_real: List[Float64], z: List[Float64], t: Float64 = 0.5) -> (Float64, Float64, Float64, Float64):
        var c_real = self.C.forward(x_real)
        var x_fake = self.G.forward(z)
        var c_fake = self.C.forward(x_fake)
        var gp = self._gp(x_real, x_fake, t)
        var loss_c = (c_fake - c_real) + gp
        return (loss_c, c_real, c_fake, gp)
fn __copyinit__(out self, other: Self) -> None:
        self.G = other.G
        self.C = other.C
        self.lambda_gp = other.lambda_gp
        self.eps_fd = other.eps_fd
fn __moveinit__(out self, deinit other: Self) -> None:
        self.G = other.G
        self.C = other.C
        self.lambda_gp = other.lambda_gp
        self.eps_fd = other.eps_fd
# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True

    var z = List[Float64]()
    for i in range(6): z.push(0.3)
    var x_real = List[Float64]()
    for i in range(10): x_real.push(1.0)

    var m = WGAN_GP(6, 10, 12, 12, 10.0, 1e-3)

    var g_loss = m.gen_step(z)
    ok = ok and (g_loss == g_loss)

    var (d_loss, cr, cf, gp) = m.dis_step(x_real, z, 0.4)
    ok = ok and (d_loss == d_loss) and (gp >= 0.0)

    for t in range(3):
        var (dl, rr, ff, gg) = m.dis_step(x_real, z, 0.5)
        var gg_loss = m.gen_step(z)
        ok = ok and (dl == dl) and (gg_loss == gg_loss)

    return ok