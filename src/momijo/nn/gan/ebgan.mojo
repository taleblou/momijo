# Project:      Momijo
# Module:       src.momijo.nn.gan.ebgan
# File:         ebgan.mojo
# Path:         src/momijo/nn/gan/ebgan.mojo
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
#   - Structs: Linear, AE, Generator, EBGAN
#   - Key functions: _abs, _l1, _exp, _sigmoid, _act1d, __init__, forward, __copyinit__ ...
#   - Uses generic functions/types with explicit trait bounds.


fn _abs(x: Float64) -> Float64:
    if x >= 0.0: return x
    return -x
fn _l1(a: List[Float64], b: List[Float64]) -> Float64:
    var n = len(a)
    if n == 0 or n != len(b): return 0.0
    var s = 0.0
    for i in range(n):
        s += _abs(a[i] - b[i])
    return s / Float64(n)
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
    # tanh-like via scaled sigmoid
    var y = List[Float64]()
    for v in x:
        y.push(2.0 * _sigmoid(2.0 * v) - 1.0)
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
# --- Autoencoder Discriminator (energy = recon loss) ---
struct AE:
    var enc: Linear
    var dec: Linear
fn __init__(out self, in_dim: Int, hidden_dim: Int) -> None:
        self.enc = Linear(in_dim, hidden_dim)
        self.dec = Linear(hidden_dim, in_dim)
fn encode(self, x: List[Float64]) -> List[Float64]:
        return _act1d(self.enc.forward(x))
fn decode(self, h: List[Float64]) -> List[Float64]:
        return self.dec.forward(_act1d(h))
fn recon(self, x: List[Float64]) -> List[Float64]:
        return self.decode(self.encode(x))
fn energy(self, x: List[Float64]) -> Float64:
        return _l1(self.recon(x), x)  # L1 energy
fn __copyinit__(out self, other: Self) -> None:
        self.enc = other.enc
        self.dec = other.dec
fn __moveinit__(out self, deinit other: Self) -> None:
        self.enc = other.enc
        self.dec = other.dec
# --- Generator ---
struct Generator:
    var hid: Linear
    var out: Linear
fn __init__(out self, z_dim: Int, hidden_dim: Int, x_dim: Int) -> None:
        self.hid = Linear(z_dim, hidden_dim)
        self.out = Linear(hidden_dim, x_dim)
fn forward(self, z: List[Float64]) -> List[Float64]:
        return self.out.forward(_act1d(self.hid.forward(z)))
fn __copyinit__(out self, other: Self) -> None:
        self.hid = other.hid
        self.out = other.out
fn __moveinit__(out self, deinit other: Self) -> None:
        self.hid = other.hid
        self.out = other.out
# --- EBGAN wrapper ---
struct EBGAN:
    var G: Generator
    var D: AE
    var margin: Float64
fn __init__(out self, z_dim: Int, x_dim: Int, g_hidden: Int, d_hidden: Int, margin: Float64 = 1.0) -> None:
        self.G = Generator(z_dim, g_hidden, x_dim)
        self.D = AE(x_dim, d_hidden)
        self.margin = margin

    # Generator loss: minimize energy on fake
fn gen_step(self, z: List[Float64]) -> Float64:
        var x_fake = self.G.forward(z)
        return self.D.energy(x_fake)

    # Discriminator loss: L_D = energy(real) + max(0, margin - energy(fake))
fn dis_step(self, x_real: List[Float64], z: List[Float64]) -> (Float64, Float64, Float64):
        var e_real = self.D.energy(x_real)
        var x_fake = self.G.forward(z)
        var e_fake = self.D.energy(x_fake)
        var hinge = self.margin - e_fake
        if hinge < 0.0: hinge = 0.0
        var loss = e_real + hinge
        return (loss, e_real, e_fake)
fn __copyinit__(out self, other: Self) -> None:
        self.G = other.G
        self.D = other.D
        self.margin = other.margin
fn __moveinit__(out self, deinit other: Self) -> None:
        self.G = other.G
        self.D = other.D
        self.margin = other.margin
# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True

    var z = List[Float64]()
    for i in range(8): z.push(0.1)
    var x_real = List[Float64]()
    for i in range(12): x_real.push(1.0)

    var m = EBGAN(8, 12, 16, 10, 0.5)

    var g_loss = m.gen_step(z)
    ok = ok and (g_loss >= 0.0)

    var (d_loss, e_r, e_f) = m.dis_step(x_real, z)
    ok = ok and (d_loss >= 0.0) and (e_r >= 0.0) and (e_f >= 0.0)

    # Run a few more steps
    for t in range(3):
        var _g = m.gen_step(z)
        var (_d, _er, _ef) = m.dis_step(x_real, z)
        ok = ok and (_g == _g) and (_d == _d)  # NaN guards

    return ok