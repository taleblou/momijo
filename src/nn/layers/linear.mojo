# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.linear
# Path:   src/momijo/nn/linear.mojo
#
# Minimal Linear (fully-connected / dense) layer for pedagogy/smoke tests.
# Works on List[Float64] (1D) and List[List[Float64]] (2D batch). Affine transform:
#   y = x @ W^T + b
# with weight W shape [out, in] and bias [out].
#
# Momijo style:
# - No global vars, no `export`. Use `var` (not `let`).
# - Constructors: fn __init__(out self, ...)
# - Prefer `mut/out` over `inout`.

# --- Helpers ---
fn zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y

fn zeros2d(r: Int, c: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(r):
        var row = List[Float64]()
        for j in range(c): row.push(0.0)
        y.push(row)
    return y

fn matvec(W: List[List[Float64]], x: List[Float64], b: List[Float64]) -> List[Float64]:
    # W: [out,in], x: [in], b: [out]
    var outdim = len(W)
    var y = zeros1d(outdim)
    for o in range(outdim):
        var acc = b[o]
        var indim = len(x)
        for i in range(indim):
            acc += W[o][i] * x[i]
        y[o] = acc
    return y

fn matmat(W: List[List[Float64]], X: List[List[Float64]], b: List[Float64]) -> List[List[Float64]]:
    # W: [out,in], X: [N,in], b: [out] -> Y: [N,out]
    var N = len(X)
    var outdim = len(W)
    var Y = zeros2d(N, outdim)
    for n in range(N):
        Y[n] = matvec(W, X[n], b)
    return Y

# --- Linear ---
struct Linear:
    var in_features: Int
    var out_features: Int
    var weight: List[List[Float64]]  # [out,in]
    var bias: List[Float64]          # [out]

    fn __init__(out self, in_features: Int, out_features: Int, bias: Bool = True, w_init: Float64 = 0.01):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = zeros2d(out_features, in_features)
        for o in range(out_features):
            for i in range(in_features):
                self.weight[o][i] = w_init
        self.bias = zeros1d(out_features) if bias else zeros1d(out_features)

    fn forward_vec(self, x: List[Float64]) -> List[Float64]:
        return matvec(self.weight, x, self.bias)

    fn forward_batch(self, X: List[List[Float64]]) -> List[List[Float64]]:
        return matmat(self.weight, X, self.bias)

    # Alias
    fn forward(self, X: List[List[Float64]]) -> List[List[Float64]]:
        return self.forward_batch(X)

# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    var lin = Linear(4, 3, True, 0.5)

    # Single vector
    var x = zeros1d(4)
    for i in range(4): x[i] = Float64(i + 1)
    var y = lin.forward_vec(x)
    ok = ok and (len(y) == 3)

    # Batch
    var X = zeros2d(2, 4)
    for n in range(2):
        for i in range(4): X[n][i] = Float64(n + i)
    var Y = lin.forward_batch(X)
    ok = ok and (len(Y) == 2) and (len(Y[0]) == 3)

    return ok
 