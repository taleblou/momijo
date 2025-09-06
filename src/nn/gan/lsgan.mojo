# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.gan.lsgan
# Path:   src/momijo/nn/gan/lsgan.mojo
#
# LSGAN (Least Squares GAN) — compact, dependency-light scaffold.
# Uses Lists and Float64 math; replace internals with tensor ops later.
#
# Momijo style:
# - No global vars, no `export`. Use `var` (not `let`).
# - Constructors: fn __init__(out self, ...)
# - Prefer `mut/out` over `inout`.

# --- Helpers ---
fn _sum1d(xs: List[Float64]) -> Float64:
    var s = 0.0
    for v in xs: s += v
    return s

fn act_tanh_like(x: List[Float64]) -> List[Float64]:
    # 2*sigmoid(2x)-1 via truncated exp series (keep deps minimal)
    var y = List[Float64]()
    for v in x:
        # exp(-2v) approx with 12-term series
        var term = 1.0
        var e = 1.0
        var n = 1
        var k = 1.0
        var z = -2.0 * v
        while n <= 12:
            term *= z / k
            e += term
            n += 1
            k += 1.0
        y.push(2.0 / (1.0 + e) - 1.0)
    return y

# --- Linear layer with bias ---
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
        for o in range(out_features): self.b.push(0.0)

    fn forward(self, x: List[Float64]) -> List[Float64]:
        var y = List[Float64]()
        for o in range(self.out_features):
            var acc = self.b[o]
            for i in range(self.in_features):
                acc += self.W[o][i] * x[i]
            y.push(acc)
        return y

# --- LSGAN networks ---
struct Generator:
    var l1: Linear
    var l2: Linear

    fn __init__(out self, z_dim: Int, hidden_dim: Int, x_dim: Int):
        self.l1 = Linear(z_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, x_dim)

    fn forward(self, z: List[Float64]) -> List[Float64]:
        return self.l2.forward(act_tanh_like(self.l1.forward(z)))

struct Discriminator:
    var l1: Linear
    var l2: Linear

    fn __init__(out self, x_dim: Int, hidden_dim: Int):
        self.l1 = Linear(x_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, 1)

    fn forward(self, x: List[Float64]) -> Float64:
        var h = act_tanh_like(self.l1.forward(x))
        return self.l2.forward(h)[0]

# --- Losses (Least Squares) ---
# For a logit y = D(x), target a in {0,1} (or other labels), L = (y - a)^2
fn ls_loss_single(y: Float64, target: Float64) -> Float64:
    var d = y - target
    return d * d

# Typical label choices from LSGAN paper:
#  - For D: real->1, fake->0 (or 1 and 0), average over both
#  - For G: wants D(x_fake)=1
# We keep [0,1] defaults and expose as parameters in the wrapper.

struct LSGAN:
    var G: Generator
    var D: Discriminator
    var a_fake: Float64   # D target for fake (usually 0)
    var b_real: Float64   # D target for real (usually 1)
    var c_gen: Float64    # G target for fake (usually 1)

    fn __init__(out self, z_dim: Int, x_dim: Int, g_hidden: Int, d_hidden: Int, a_fake: Float64 = 0.0, b_real: Float64 = 1.0, c_gen: Float64 = 1.0):
        self.G = Generator(z_dim, g_hidden, x_dim)
        self.D = Discriminator(x_dim, d_hidden)
        self.a_fake = a_fake
        self.b_real = b_real
        self.c_gen = c_gen

    # Generator step: wants D(x_fake) close to c_gen
    fn gen_step(self, z: List[Float64]) -> Float64:
        var x_fake = self.G.forward(z)
        var y_fake = self.D.forward(x_fake)
        return ls_loss_single(y_fake, self.c_gen)

    # Discriminator step: average of real and fake squared losses
    fn dis_step(self, x_real: List[Float64], z: List[Float64]) -> Float64:
        var y_real = self.D.forward(x_real)
        var loss_real = ls_loss_single(y_real, self.b_real)

        var x_fake = self.G.forward(z)
        var y_fake = self.D.forward(x_fake)
        var loss_fake = ls_loss_single(y_fake, self.a_fake)

        return 0.5 * (loss_real + loss_fake)

# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True

    var z = List[Float64]()
    for i in range(8): z.push(0.4)
    var x_real = List[Float64]()
    for i in range(12): x_real.push(1.0)

    var m = LSGAN(8, 12, 16, 16, 0.0, 1.0, 1.0)

    var g_loss = m.gen_step(z)
    ok = ok and (g_loss >= 0.0)

    var d_loss = m.dis_step(x_real, z)
    ok = ok and (d_loss >= 0.0)

    # exercise
    for t in range(3):
        var _g = m.gen_step(z)
        var _d = m.dis_step(x_real, z)
        ok = ok and (_g == _g) and (_d == _d)  # NaN guard

    return ok
 