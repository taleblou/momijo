# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.gan.began
# Path:   src/momijo/nn/gan/began.mojo
#
# Boundary Equilibrium GAN (BEGAN) — compact, dependency-light scaffold for pedagogy.
# Reference idea: Autoencoder-based discriminator; balance via k_t and \gamma.
# This scaffold uses Lists and Float64 math to avoid heavy tensor deps.
#
# Momijo style:
# - No global vars or `export`. Use `var` (not `let`).
# - Constructors use: fn __init__(out self, ...)
# - Prefer `mut/out` over `inout`.

# --- Minimal helpers ---
fn _abs(x: Float64) -> Float64:
    if x >= 0.0: return x
    return -x

fn _sum1d(xs: List[Float64]) -> Float64:
    var s = 0.0
    for v in xs: s += v
    return s

fn _mean_abs_err(a: List[Float64], b: List[Float64]) -> Float64:
    var n = len(a)
    if n == 0 or n != len(b): return 0.0
    var s = 0.0
    for i in range(n):
        s += _abs(a[i] - b[i])
    return s / Float64(n)

# --- Tiny linear layer with bias ---
struct Linear:
    var in_features: Int
    var out_features: Int
    var W: List[List[Float64]]  # [out, in]
    var b: List[Float64]        # [out]

    fn __init__(out self, in_features: Int, out_features: Int, weight: Float64 = 0.01):
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
        var y = List[Float64]()
        for o in range(self.out_features):
            var acc = self.b[o]
            for i in range(self.in_features):
                acc += self.W[o][i] * x[i]
            y.push(acc)
        return y

# --- Simple nonlinearity (tanh-like via scaled sigmoid approximation) ---
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

fn _sigmoid(x: Float64) -> Float64:
    var e = _exp(-x)
    return 1.0 / (1.0 + e)

fn _act1d(x: List[Float64]) -> List[Float64]:
    var y = List[Float64]()
    for v in x:
        y.push(2.0 * _sigmoid(2.0 * v) - 1.0)
    return y

# --- Autoencoder discriminator (AE) ---
struct AE:
    var enc: Linear
    var dec: Linear

    fn __init__(out self, in_dim: Int, hidden_dim: Int):
        self.enc = Linear(in_dim, hidden_dim)
        self.dec = Linear(hidden_dim, in_dim)

    fn encode(self, x: List[Float64]) -> List[Float64]:
        return _act1d(self.enc.forward(x))

    fn decode(self, h: List[Float64]) -> List[Float64]:
        return self.dec.forward(_act1d(h))

    fn recon(self, x: List[Float64]) -> List[Float64]:
        return self.decode(self.encode(x))

    fn recon_loss(self, x: List[Float64]) -> Float64:
        var r = self.recon(x)
        return _mean_abs_err(r, x)  # L1 as in BEGAN paper

# --- Generator (maps z -> x) ---
struct Generator:
    var hid: Linear
    var out: Linear

    fn __init__(out self, z_dim: Int, hidden_dim: Int, x_dim: Int):
        self.hid = Linear(z_dim, hidden_dim)
        self.out = Linear(hidden_dim, x_dim)

    fn forward(self, z: List[Float64]) -> List[Float64]:
        return self.out.forward(_act1d(self.hid.forward(z)))

# --- BEGAN container with balance term k_t ---
struct BEGAN:
    var G: Generator
    var D: AE
    var gamma: Float64      # balance target (e.g., 0.5..0.7)
    var lambda_k: Float64   # step size for k_t update
    var k_t: Float64        # balance variable in [0,1]

    fn __init__(out self, z_dim: Int, x_dim: Int, g_hidden: Int, d_hidden: Int, gamma: Float64 = 0.5, lambda_k: Float64 = 0.001):
        self.G = Generator(z_dim, g_hidden, x_dim)
        self.D = AE(x_dim, d_hidden)
        self.gamma = gamma
        self.lambda_k = lambda_k
        self.k_t = 0.0

    # One generator step: minimize D's reconstruction on fake (no optimizer here)
    fn gen_step(self, z: List[Float64]) -> Float64:
        var x_fake = self.G.forward(z)
        var l_g = self.D.recon_loss(x_fake)
        return l_g

    # One discriminator step: L_D = L_real - k_t * L_fake, update k_t
    fn dis_step(mut self, x_real: List[Float64], z: List[Float64]) -> (Float64, Float64, Float64, Float64):
        var L_real = self.D.recon_loss(x_real)
        var x_fake = self.G.forward(z)
        var L_fake = self.D.recon_loss(x_fake)

        var L_D = L_real - self.k_t * L_fake
        var balance = self.gamma * L_real - L_fake

        # Update k_t
        self.k_t = self.k_t + self.lambda_k * balance
        if self.k_t < 0.0: self.k_t = 0.0
        if self.k_t > 1.0: self.k_t = 1.0

        # Convergence measure (as in paper): M = L_real + |balance|
        var measure = L_real + (_abs(balance))
        return (L_D, L_real, L_fake, measure)

# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True

    var z = List[Float64]()
    for i in range(8): z.push(0.1)
    var x_real = List[Float64]()
    for i in range(12): x_real.push(1.0)

    var model = BEGAN(8, 12, 16, 10, 0.5, 0.01)

    # Initial steps
    var g_loss = model.gen_step(z)
    ok = ok and (g_loss >= 0.0)

    var (d_loss, Lr, Lf, M) = model.dis_step(x_real, z)
    ok = ok and (Lr >= 0.0) and (Lf >= 0.0) and (M >= 0.0)

    # Run a few more discriminator updates to exercise k_t
    for t in range(5):
        var (_d, _Lr, _Lf, _M) = model.dis_step(x_real, z)
        ok = ok and (_d == _d)  # NaN check (trivial equality holds for non-NaN floats in Mojo? using placeholder)

    return ok

 