# Project:      Momijo
# Module:       src.momijo.nn.layers.groupnorm
# File:         groupnorm.mojo
# Path:         src/momijo/nn/layers/groupnorm.mojo
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
#   - Structs: GroupNorm
#   - Key functions: _zeros1d, _ones1d, _zeros3d, _zeros4d, _sqrt_pos, _groupnorm_chw, __init__, set_affine ...
#   - Uses generic functions/types with explicit trait bounds.


fn _zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y
fn _ones1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(1.0)
    return y
fn _zeros3d(c: Int, h: Int, w: Int) -> List[List[List[Float64]]]:
    var y = List[List[List[Float64]]]()
    for i in range(c):
        var m = List[List[Float64]]()
        for r in range(h):
            var row = List[Float64]()
            for s in range(w): row.push(0.0)
            m.push(row)
        y.push(m)
    return y
fn _zeros4d(n: Int, c: Int, h: Int, w: Int) -> List[List[List[List[Float64]]]]:
    var y = List[List[List[List[Float64]]]]()
    for i in range(n):
        y.push(_zeros3d(c, h, w))
    return y

# crude sqrt approximation via two Newton iterations
fn _sqrt_pos(x: Float64) -> Float64:
    if x <= 0.0: return 0.0
    var s = x
    s = 0.5 * (s + x / s)
    s = 0.5 * (s + x / s)
    return s

# --- Core per-sample GN on [C,H,W] ---
# Splits C into G groups (last group may be a bit larger if C % G != 0).
# For each group, compute mean/var over all elements in that group's channels and HxW,
# then normalize and apply per-channel affine (gamma/beta).
fn _groupnorm_chw(x: List[List[List[Float64]]], num_groups: Int, gamma: List[Float64], beta: List[Float64], eps: Float64) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var y = _zeros3d(C, H, W)

    var G = num_groups
    if G <= 0: G = 1
    if G > C: G = C

    var base = C / G        # integer
    var rem = C % G         # remainder

    var c_start = 0
    for g in range(G):
        var gsize = base
        if g < rem: gsize += 1
        var c_end = c_start + gsize  # exclusive
        # mean/var over group
        var count = 0
        var sumv = 0.0
        var sumsq = 0.0
        for c in range(c_start, c_end):
            for i in range(H):
                for j in range(W):
                    var v = x[c][i][j]
                    sumv += v
                    sumsq += v * v
                    count += 1
        if count == 0:
            c_start = c_end
            continue
        var mean = sumv / Float64(count)
        var varg = sumsq / Float64(count) - mean * mean
        var denom = _sqrt_pos(varg + eps)
        if denom == 0.0: denom = 1.0  # safety

        for c in range(c_start, c_end):
            var gmm = gamma[c]
            var bbb = beta[c]
            for i in range(H):
                for j in range(W):
                    var xhat = (x[c][i][j] - mean) / denom
                    y[c][i][j] = gmm * xhat + bbb
        c_start = c_end
    return y

# --- Public GN module ---
struct GroupNorm:
    var num_groups: Int
    var num_channels: Int
    var eps: Float64
    var affine: Bool
    var gamma: List[Float64]  # per-channel
    var beta: List[Float64]   # per-channel
fn __init__(out self, num_groups: Int, num_channels: Int, eps: Float64 = 1e-5, affine: Bool = True) -> None:
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        # Always allocate gamma/beta; caller may ignore by setting gamma=1, beta=0
        self.gamma = _ones1d(num_channels)
        self.beta = _zeros1d(num_channels)
fn set_affine(mut self, enable: Bool) -> None:
        self.affine = enable
        if not enable:
            # reset to identity when disabling
            var C = self.num_channels
            self.gamma = _ones1d(C)
            self.beta = _zeros1d(C)

    # forward for [C,H,W]
fn forward_chw(self, x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
        var g = self.num_groups
        var C = self.num_channels
        # if affine disabled, use identity gamma/beta on-the-fly
        var gamma_use = self.gamma
        var beta_use = self.beta
        if not self.affine:
            gamma_use = _ones1d(C)
            beta_use = _zeros1d(C)
        return _groupnorm_chw(x, g, gamma_use, beta_use, self.eps)

    # forward for [N,C,H,W]
fn forward_nchw(self, x: List[List[List[List[Float64]]]]) -> List[List[List[List[Float64]]]]:
        var N = len(x)
        if N == 0: return x
        var C = len(x[0])
        var H = 0
        var W = 0
        if C > 0:
            H = len(x[0][0])
            if H > 0: W = len(x[0][0][0])
        var y = _zeros4d(N, C, H, W)
        for n in range(N):
            y[n] = self.forward_chw(x[n])
        return y
fn __copyinit__(out self, other: Self) -> None:
        self.num_groups = other.num_groups
        self.num_channels = other.num_channels
        self.eps = other.eps
        self.affine = other.affine
        self.gamma = other.gamma
        self.beta = other.beta
fn __moveinit__(out self, deinit other: Self) -> None:
        self.num_groups = other.num_groups
        self.num_channels = other.num_channels
        self.eps = other.eps
        self.affine = other.affine
        self.gamma = other.gamma
        self.beta = other.beta
# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    var C = 5
    var H = 3
    var W = 4

    # Build a simple CHW tensor with varying values
    var x = _zeros3d(C, H, W)
    for c in range(C):
        for i in range(H):
            for j in range(W):
                x[c][i][j] = Float64(c + 1) * 0.1 + 0.01 * Float64(i * W + j)

    # 1) GN with G=1 (LayerNorm-like over all channels)
    var gn1 = GroupNorm(1, C, 1e-5, True)
    var y1 = gn1.forward_chw(x)
    ok = ok and (len(y1) == C) and (len(y1[0]) == H) and (len(y1[0][0]) == W)

    # 2) GN with G=C (InstanceNorm-like per channel)
    var gn2 = GroupNorm(C, C, 1e-5, True)
    var y2 = gn2.forward_chw(x)
    ok = ok and (len(y2) == C)

    # 3) GN on NCHW
    var N = 2
    var xb = _zeros4d(N, C, H, W)
    xb[0] = x
    xb[1] = x
    var gn3 = GroupNorm(2, C, 1e-5, False)  # affine=False -> identity gamma/beta
    var yb = gn3.forward_nchw(xb)
    ok = ok and (len(yb) == N) and (len(yb[0]) == C)

    # 4) Affine parameters effect: set gamma=2, beta=-1 for a channel and check change
    gn1.gamma[0] = 2.0
    gn1.beta[0] = -1.0
    var y1b = gn1.forward_chw(x)
    # Only channel 0 gets different affine now; shapes stay intact
    ok = ok and (len(y1b) == C) and (len(y1b[0]) == H) and (len(y1b[0][0]) == W)

    return ok