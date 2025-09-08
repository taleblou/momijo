# Project:      Momijo
# Module:       src.momijo.nn.gan.penalties
# File:         penalties.mojo
# Path:         src/momijo/nn/gan/penalties.mojo
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
#   - Structs: Penalties, DummyD
#   - Key functions: sub1d, mix1d, unit_dir, grad_norm_proxy_1d, __init__, wgan_gp, dragan_gp, r1_proxy ...
#   - Uses generic functions/types with explicit trait bounds.


from gpu.host import dim
from momijo.core.error import module
from momijo.dataframe.helpers import between, diff, sqrt, t
from momijo.ir.dialects.annotations import unit
from momijo.nn.gan.dragan import DRAGAN
from momijo.nn.gan.wgan import WGAN
from momijo.nn.parameter import data, grad
from momijo.tensor.shape import dim
from momijo.util.random_facade import random
from momijo.utils.random import sample
from momijo.utils.result import f, g
from pathlib import Path
from pathlib.path import Path

# NOTE: Removed duplicate definition of `add1d`; use `from momijo.nn.gan.wgan import add1d`

fn sub1d(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var n = len(a)
    var y = List[Float64]()
    for i in range(n): y.push(a[i] - b[i])
    return y
# NOTE: Removed duplicate definition of `scale1d`; use `from momijo.nn.gan.wgan import scale1d`
fn mix1d(a: List[Float64], b: List[Float64], t: Float64) -> List[Float64]:
    # (1-t)*a + t*b
    var n = len(a)
    var y = List[Float64]()
    for i in range(n): y.push((1.0 - t) * a[i] + t * b[i])
    return y

# Deterministic pseudo-random unit direction (+1,-1,+1,...), normalized ~1/sqrt(n)
fn unit_dir(n: Int) -> List[Float64]:
    var u = List[Float64]()
    var s = 1.0
    for i in range(n):
        u.push(s); s = -s
    var inv = 1.0
    if n > 0: inv = 1.0 / (Float64(n) ** 0.5)  # simple normalization
    return scale1d(u, inv)

# Directional derivative proxy of ||grad f(x)|| using central difference along u
fn grad_norm_proxy_1d(fx: Float64, f_plus: Float64, f_minus: Float64, eps: Float64) -> Float64:
    var directional = (f_plus - f_minus) / (2.0 * eps)
    if directional < 0.0: directional = -directional
    return directional

# --- Minimal Discriminator interface (duck-typed) ---
# We expect any D used here to have: fn forward(self, x: List[Float64]) -> Float64

# --- Penalty pack ---
struct Penalties:
    var lambda_gp: Float64
    var eps_fd: Float64     # finite-difference step (e.g., 1e-3)
    var perturb_scale: Float64  # for DRAGAN-style around data manifold
fn __init__(out self, lambda_gp: Float64 = 10.0, eps_fd: Float64 = 1e-3, perturb_scale: Float64 = 0.5) -> None:
        self.lambda_gp = lambda_gp
        self.eps_fd = eps_fd
        self.perturb_scale = perturb_scale

    # WGAN-GP like: sample x_hat on line between real/fake, penalize (||grad D(x_hat)|| - 1)^2
fn wgan_gp(self, D, x_real: List[Float64], x_fake: List[Float64], t: Float64 = 0.5) -> Float64:
        var x_hat = mix1d(x_real, x_fake, t)
        var u = unit_dir(len(x_hat))
        var x_plus = add1d(x_hat, scale1d(u, self.eps_fd))
        var x_minus = add1d(x_hat, scale1d(u, -self.eps_fd))
        var f = D.forward(x_hat)
        var f_plus = D.forward(x_plus)
        var f_minus = D.forward(x_minus)
        var g = grad_norm_proxy_1d(f, f_plus, f_minus, self.eps_fd)
        var diff = g - 1.0
        return self.lambda_gp * diff * diff

    # DRAGAN-like: around perturbed real data
fn dragan_gp(self, D, x_real: List[Float64]) -> Float64:
        var noise = unit_dir(len(x_real))
        var x_hat = add1d(x_real, scale1d(noise, self.perturb_scale))
        var u = unit_dir(len(x_hat))
        var x_plus = add1d(x_hat, scale1d(u, self.eps_fd))
        var x_minus = add1d(x_hat, scale1d(u, -self.eps_fd))
        var f = D.forward(x_hat)
        var f_plus = D.forward(x_plus)
        var f_minus = D.forward(x_minus)
        var g = grad_norm_proxy_1d(f, f_plus, f_minus, self.eps_fd)
        var diff = g - 1.0
        return self.lambda_gp * diff * diff

    # R1-like (on real): penalize ||grad D(x_real)||^2 via FD proxy
fn r1_proxy(self, D, x_real: List[Float64]) -> Float64:
        var u = unit_dir(len(x_real))
        var x_plus = add1d(x_real, scale1d(u, self.eps_fd))
        var x_minus = add1d(x_real, scale1d(u, -self.eps_fd))
        var f = D.forward(x_real)
        var f_plus = D.forward(x_plus)
        var f_minus = D.forward(x_minus)
        var g = grad_norm_proxy_1d(f, f_plus, f_minus, self.eps_fd)
        return self.lambda_gp * g * g

    # R2-like (on fake): penalize ||grad D(x_fake)||^2 via FD proxy
fn r2_proxy(self, D, x_fake: List[Float64]) -> Float64:
        var u = unit_dir(len(x_fake))
        var x_plus = add1d(x_fake, scale1d(u, self.eps_fd))
        var x_minus = add1d(x_fake, scale1d(u, -self.eps_fd))
        var f = D.forward(x_fake)
        var f_plus = D.forward(x_plus)
        var f_minus = D.forward(x_minus)
        var g = grad_norm_proxy_1d(f, f_plus, f_minus, self.eps_fd)
        return self.lambda_gp * g * g
fn __copyinit__(out self, other: Self) -> None:
        self.lambda_gp = other.lambda_gp
        self.eps_fd = other.eps_fd
        self.perturb_scale = other.perturb_scale
fn __moveinit__(out self, deinit other: Self) -> None:
        self.lambda_gp = other.lambda_gp
        self.eps_fd = other.eps_fd
        self.perturb_scale = other.perturb_scale
# --- Tiny dummy discriminator for tests ---
struct DummyD:
    var w: List[Float64]
fn __init__(out self, dim: Int) -> None:
        self.w = List[Float64]()
        for i in range(dim): self.w.push(0.01)  # tiny slope
fn forward(self, x: List[Float64]) -> Float64:
        var s = 0.0
        for i in range(len(x)): s += self.w[i] * x[i]
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.w = other.w
fn __moveinit__(out self, deinit other: Self) -> None:
        self.w = other.w
# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True

    var xr = List[Float64]()
    var xf = List[Float64]()
    for i in range(10):
        xr.push(1.0)
        xf.push(0.2)

    var D = DummyD(10)
    var pen = Penalties(10.0, 1e-3, 0.5)

    var wgp = pen.wgan_gp(D, xr, xf, 0.3)
    var dgp = pen.dragan_gp(D, xr)
    var r1  = pen.r1_proxy(D, xr)
    var r2  = pen.r2_proxy(D, xf)

    ok = ok and (wgp >= 0.0) and (dgp >= 0.0) and (r1 >= 0.0) and (r2 >= 0.0)
    return ok