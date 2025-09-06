# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.dropout
# Path:   src/momijo/nn/dropout.mojo
#
# Minimal, dependency-light Inverted Dropout for pedagogy/smoke tests.
# Works on List[Float64] (1D) and List[List[Float64]] (2D). Uses a
# deterministic pseudo-random mask (index-hash) to avoid RNG deps.
#
# Momijo style:
# - No global vars, no `export`. Use `var` (not `let`).
# - Constructors: fn __init__(out self, ...)
# - Prefer `mut/out` over `inout`.

# --- Helpers ---
fn _clamp01(x: Float64) -> Float64:
    var y = x
    if y < 0.0: y = 0.0
    if y > 1.0: y = 1.0
    return y

fn _zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y

fn _zeros2d(r: Int, c: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(r):
        var row = List[Float64]()
        for j in range(c): row.push(0.0)
        y.push(row)
    return y

# Simple integer hash -> [0,1) using a prime modulus.
# Avoids true RNG deps but gives deterministic, index-dependent masks.
fn _rand01_idx(i: Int, j: Int, seed: Int) -> Float64:
    var m = 9973  # small prime modulus for pedagogy
    var v = ((i + 1) * 1103515245 + (j + 1) * 12345 + seed * 2654435761) % m
    if v < 0: v = v + m
    return Float64(v) / Float64(m - 1)

# --- Dropout (inverted) ---
struct Dropout:
    var p: Float64       # drop probability in [0,1)
    var training: Bool
    var seed: Int        # for deterministic masks

    fn __init__(out self, p: Float64 = 0.5, training: Bool = True, seed: Int = 1337):
        var pp = _clamp01(p)
        if pp >= 1.0: pp = 0.999999  # avoid div by zero in scale
        self.p = pp
        self.training = training
        self.seed = seed

    fn set_p(mut self, p: Float64):
        var pp = _clamp01(p)
        if pp >= 1.0: pp = 0.999999
        self.p = pp

    fn set_seed(mut self, seed: Int):
        self.seed = seed

    fn train_mode(mut self):
        self.training = True

    fn eval_mode(mut self):
        self.training = False

    # Inverted dropout scaling factor
    fn _scale(self) -> Float64:
        return 1.0 / (1.0 - self.p)

    # 1D forward
    fn forward1d(self, x: List[Float64]) -> List[Float64]:
        var n = len(x)
        if not self.training or self.p <= 0.0:
            return x
        var y = _zeros1d(n)
        var s = self._scale()
        for i in range(n):
            var r = _rand01_idx(i, 0, self.seed)
            if r < self.p:
                y[i] = 0.0
            else:
                y[i] = x[i] * s
        return y

    # 2D forward (row-major), index-hash uses (i,j)
    fn forward2d(self, x: List[List[Float64]]) -> List[List[Float64]]:
        var r = len(x)
        if r == 0: return x
        var c = len(x[0])
        if not self.training or self.p <= 0.0:
            return x
        var y = _zeros2d(r, c)
        var s = self._scale()
        for i in range(r):
            for j in range(c):
                var rnd = _rand01_idx(i, j, self.seed)
                if rnd < self.p:
                    y[i][j] = 0.0
                else:
                    y[i][j] = x[i][j] * s
        return y

# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # 1D
    var v = List[Float64]()
    for i in range(10): v.push(Float64(i))
    var d = Dropout(0.5, True, 42)
    var y = d.forward1d(v)
    ok = ok and (len(y) == 10)
    # deterministic: same seed -> same mask
    var y2 = d.forward1d(v)
    var same = True
    for i in range(10):
        if y[i] != y2[i]: same = False
    ok = ok and same

    # switch to eval: should pass-through
    d.eval_mode()
    var y_eval = d.forward1d(v)
    var pass_through = True
    for i in range(10):
        if y_eval[i] != v[i]: pass_through = False
    ok = ok and pass_through

    # 2D
    var M = 3; var N = 4
    var X = _zeros2d(M, N)
    for i in range(M):
        for j in range(N):
            X[i][j] = 0.1 * Float64(i * N + j + 1)
    var d2 = Dropout(0.25, True, 7)
    var Y = d2.forward2d(X)
    ok = ok and (len(Y) == M) and (len(Y[0]) == N)

    return ok
 