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
# File: src/momijo/nn/gan/dragan.mojo

fn _sum1d(xs: List[Float64]) -> Float64:
    var s = 0.0
    for v in xs: s += v
    return s
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

# deterministic pseudo-noise direction (+1,-1,+1,...) for pedagogy
fn unit_dir(n: Int) -> List[Float64]:
    var u = List[Float64]()
    var s = 1.0
    for i in range(n):
        u.push(s)
        s = -s
    # normalize to unit L2 (approximate by 1/sqrt(n))
    var inv = 1.0
    if n > 0:
        inv = 1.0 / (Float64(n) ** 0.5)  # NOTE: Mojo pow for Float64 literal; replace with exp/log if needed
    return scale1d(u, inv)

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
                row.push(weight)
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
# --- Generator & Discriminator ---
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

# Finite-difference directional-derivative proxy of ||grad D(x)||
fn grad_norm_proxy(d: Discriminator, x: List[Float64], eps: Float64) -> Float64:
    var n = len(x)
    if n == 0: return 0.0
    var u = unit_dir(n)                # unit direction
    var x_plus = add1d(x, scale1d(u, eps))
    var x_minus = add1d(x, scale1d(u, -eps))
    var f_plus = d.forward(x_plus)
    var f_minus = d.forward(x_minus)
    var directional = (f_plus - f_minus) / (2.0 * eps)
    if directional < 0.0: directional = -directional
    return directional  # proxy for ||grad||

# --- DRAGAN wrapper ---
struct DRAGAN:
    var G: Generator
    var D: Discriminator
    var lambda_gp: Float64
    var perturb_scale: Float64
    var eps_fd: Float64
fn __init__(out self, z_dim: Int, x_dim: Int, g_hidden: Int, d_hidden: Int, lambda_gp: Float64 = 10.0, perturb_scale: Float64 = 0.5, eps_fd: Float64 = 1e-3) -> None:
        self.G = Generator(z_dim, g_hidden, x_dim)
        self.D = Discriminator(x_dim, d_hidden)
        self.lambda_gp = lambda_gp
        self.perturb_scale = perturb_scale
        self.eps_fd = eps_fd
fn gen_step(self, z: List[Float64]) -> Float64:
        var x_fake = self.G.forward(z)
        var d_logit = self.D.forward(x_fake)
        return bce_with_logits_single(d_logit, True)
fn dis_step(self, x_real: List[Float64], z: List[Float64]) -> (Float64, Float64, Float64):
        # Standard GAN loss
        var d_logit_r = self.D.forward(x_real)
        var loss_real = bce_with_logits_single(d_logit_r, True)
        var x_fake = self.G.forward(z)
        var d_logit_f = self.D.forward(x_fake)
        var loss_fake = bce_with_logits_single(d_logit_f, False)

        # DRAGAN gradient penalty around perturbed real data
        # x_hat = x_real + noise * perturb_scale  (deterministic +/- direction)
        var noise = unit_dir(len(x_real))
        var x_hat = add1d(x_real, scale1d(noise, self.perturb_scale))
        var gnorm = grad_norm_proxy(self.D, x_hat, self.eps_fd)
        var gp = self.lambda_gp * (gnorm - 1.0) * (gnorm - 1.0)

        var d_loss = loss_real + loss_fake + gp
        return (d_loss, gp, loss_real + loss_fake)
fn __copyinit__(out self, other: Self) -> None:
        self.G = other.G
        self.D = other.D
        self.lambda_gp = other.lambda_gp
        self.perturb_scale = other.perturb_scale
        self.eps_fd = other.eps_fd
fn __moveinit__(out self, deinit other: Self) -> None:
        self.G = other.G
        self.D = other.D
        self.lambda_gp = other.lambda_gp
        self.perturb_scale = other.perturb_scale
        self.eps_fd = other.eps_fd
# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True

    var z = List[Float64]()
    for i in range(8): z.push(0.2)
    var x_real = List[Float64]()
    for i in range(12): x_real.push(1.0)

    var m = DRAGAN(8, 12, 16, 16, 5.0, 0.3, 1e-3)

    var g_loss = m.gen_step(z)
    ok = ok and (g_loss >= 0.0)

    var (d_loss, gp, base) = m.dis_step(x_real, z)
    ok = ok and (d_loss >= 0.0) and (gp >= 0.0) and (base >= 0.0)

    # Run a few more
    for t in range(3):
        var (d2, gp2, b2) = m.dis_step(x_real, z)
        ok = ok and (d2 == d2) and (gp2 == gp2) and (b2 == b2)  # NaN guard

    return ok