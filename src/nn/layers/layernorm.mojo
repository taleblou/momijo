# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.layernorm
# Path:   src/momijo/nn/layernorm.mojo
#
# Minimal Layer Normalization (LN) for pedagogy/smoke tests.
# Works on List[Float64] as a single vector [D] and List[List[Float64]] as a
# batch of vectors [N, D]. Normalizes over the last dimension (size D) with
# optional per-feature affine (gamma/beta).
#
# Momijo style:
# - No global vars, no `export`. Use `var` (not `let`).
# - Constructors: fn __init__(out self, ...)
# - Prefer `mut/out` over `inout`.

# --- Helpers ---
fn _zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y

fn _ones1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(1.0)
    return y

fn _zeros2d(r: Int, c: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(r):
        var row = List[Float64]()
        for j in range(c): row.push(0.0)
        y.push(row)
    return y

fn _mean_1d(x: List[Float64]) -> Float64:
    var n = len(x)
    if n == 0: return 0.0
    var s = 0.0
    for i in range(n): s += x[i]
    return s / Float64(n)

fn _var_1d(x: List[Float64], mean: Float64) -> Float64:
    var n = len(x)
    if n == 0: return 0.0
    var s = 0.0
    for i in range(n):
        var d = x[i] - mean
        s += d * d
    return s / Float64(n)

# crude sqrt approximation via two Newton steps
fn _sqrt_pos(x: Float64) -> Float64:
    if x <= 0.0: return 0.0
    var s = x
    s = 0.5 * (s + x / s)
    s = 0.5 * (s + x / s)
    return s

fn _norm_apply_affine_vec(x: List[Float64], gamma: List[Float64], beta: List[Float64], eps: Float64) -> List[Float64]:
    var D = len(x)
    var out = _zeros1d(D)
    var m = _mean_1d(x)
    var v = _var_1d(x, m)
    var denom = _sqrt_pos(v + eps)
    if denom == 0.0: denom = 1.0
    for i in range(D):
        var xhat = (x[i] - m) / denom
        out[i] = gamma[i] * xhat + beta[i]
    return out

# --- LayerNorm over last dimension ---
struct LayerNorm:
    var normalized_shape: Int  # D
    var eps: Float64
    var affine: Bool
    var gamma: List[Float64]   # [D]
    var beta: List[Float64]    # [D]

    fn __init__(out self, normalized_shape: Int, eps: Float64 = 1e-5, affine: Bool = True):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.affine = affine
        self.gamma = _ones1d(normalized_shape)
        self.beta = _zeros1d(normalized_shape)

    fn set_affine(mut self, enable: Bool):
        self.affine = enable
        if not enable:
            var D = self.normalized_shape
            self.gamma = _ones1d(D)
            self.beta = _zeros1d(D)

    # Forward for vector [D]
    fn forward_vec(self, x: List[Float64]) -> List[Float64]:
        var D = self.normalized_shape
        # If affine disabled, use identity on-the-fly
        var g = self.gamma
        var b = self.beta
        if not self.affine:
            g = _ones1d(D)
            b = _zeros1d(D)
        return _norm_apply_affine_vec(x, g, b, self.eps)

    # Forward for matrix [N, D] (row-wise LN on last dim D)
    fn forward_batch(self, x: List[List[Float64]]) -> List[List[Float64]]:
        var N = len(x)
        if N == 0: return x
        var D = self.normalized_shape
        var y = _zeros2d(N, D)

        var g = self.gamma
        var b = self.beta
        if not self.affine:
            g = _ones1d(D)
            b = _zeros1d(D)

        for n in range(N):
            y[n] = _norm_apply_affine_vec(x[n], g, b, self.eps)
        return y

    # Alias: generic `forward` assumes 2D [N,D]
    fn forward(self, x: List[List[Float64]]) -> List[List[Float64]]:
        return self.forward_batch(x)

# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # Vector case
    var D = 5
    var v = _zeros1d(D)
    for i in range(D): v[i] = 0.2 * Float64(i + 1)
    var ln = LayerNorm(D, 1e-5, True)
    var yv = ln.forward_vec(v)
    ok = ok and (len(yv) == D)

    # Check effect of affine
    ln.gamma[0] = 2.0
    ln.beta[0] = -1.0
    var yv2 = ln.forward_vec(v)
    ok = ok and (len(yv2) == D)

    # Batch case [N,D]
    var N = 3
    var X = _zeros2d(N, D)
    for n in range(N):
        for i in range(D):
            X[n][i] = Float64(n + 1) * 0.1 + 0.01 * Float64(i)
    var Y = ln.forward_batch(X)
    ok = ok and (len(Y) == N) and (len(Y[0]) == D)

    # Disable affine and run again
    ln.set_affine(False)
    var Y2 = ln.forward_batch(X)
    ok = ok and (len(Y2) == N) and (len(Y2[0]) == D)

    return ok
 
