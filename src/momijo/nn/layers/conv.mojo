# Project:      Momijo
# Module:       src.momijo.nn.layers.conv
# File:         conv.mojo
# Path:         src/momijo/nn/layers/conv.mojo
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
#   - Structs: Conv2d
#   - Key functions: zeros1d, zeros2d, zeros3d, zeros4d, pad2d_chw, conv2d_single, conv2d_batch, __init__ ...
#   - Uses generic functions/types with explicit trait bounds.


fn zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y
fn zeros2d(h: Int, w: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(h):
        var row = List[Float64]()
        for j in range(w): row.push(0.0)
        y.push(row)
    return y
fn zeros3d(c: Int, h: Int, w: Int) -> List[List[List[Float64]]]:
    var y = List[List[List[Float64]]]()
    for ch in range(c):
        y.push(zeros2d(h, w))
    return y
fn zeros4d(n: Int, c: Int, h: Int, w: Int) -> List[List[List[List[Float64]]]]:
    var y = List[List[List[List[Float64]]]]()
    for i in range(n):
        y.push(zeros3d(c, h, w))
    return y

# --- Padding (zero) ---
# x: [C,H,W] -> pad_h, pad_w applied on both sides
fn pad2d_chw(x: List[List[List[Float64]]], pad_h: Int, pad_w: Int) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var Hp = H + 2 * pad_h
    var Wp = W + 2 * pad_w
    var y = zeros3d(C, Hp, Wp)
    for c in range(C):
        for i in range(H):
            for j in range(W):
                y[c][i + pad_h][j + pad_w] = x[c][i][j]
    return y

# --- Single image conv: x[C,H,W], w[O,C,kh,kw] -> y[O,Hout,Wout] ---
fn conv2d_single(x: List[List[List[Float64]]],
                 w: List[List[List[List[Float64]]]],
                 b: List[Float64],
                 stride_h: Int, stride_w: Int,
                 pad_h: Int, pad_w: Int) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return List[List[List[Float64]]]()
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])

    var O = len(w)
    if O == 0: return List[List[List[Float64]]]()
    var kh = 0
    var kw = 0
    if len(w[0]) > 0:
        kh = len(w[0][0])
        kw = len(w[0][0][0])

    var xp = pad2d_chw(x, pad_h, pad_w)
    var Hp = H + 2 * pad_h
    var Wp = W + 2 * pad_w

    var Hout = 0
    var Wout = 0
    if stride_h > 0 and stride_w > 0 and Hp >= kh and Wp >= kw:
        Hout = (Hp - kh) / stride_h + 1
        Wout = (Wp - kw) / stride_w + 1

    var y = List[List[List[Float64]]]()
    for o in range(O):
        y.push(zeros2d(Hout, Wout))

    for o in range(O):
        for i in range(Hout):
            for j in range(Wout):
                var acc = 0.0
                for c in range(C):
                    for r in range(kh):
                        for s in range(kw):
                            var ii = i * stride_h + r
                            var jj = j * stride_w + s
                            acc += w[o][c][r][s] * xp[c][ii][jj]
                acc += b[o]
                y[o][i][j] = acc
    return y

# --- Batch conv: x[N,C,H,W] -> y[N,O,Hout,Wout] ---
fn conv2d_batch(x: List[List[List[List[Float64]]]],
                w: List[List[List[List[Float64]]]],
                b: List[Float64],
                stride_h: Int, stride_w: Int,
                pad_h: Int, pad_w: Int) -> List[List[List[List[Float64]]]]:
    var N = len(x)
    if N == 0: return List[List[List[List[Float64]]]]()
    var C = len(x[0])
    var H = 0
    var W = 0
    if C > 0:
        H = len(x[0][0])
        if H > 0: W = len(x[0][0][0])

    var O = len(w)
    var kh = 0
    var kw = 0
    if O > 0 and len(w[0]) > 0:
        kh = len(w[0][0])
        kw = len(w[0][0][0])

    var Hp = H + 2 * pad_h
    var Wp = W + 2 * pad_w
    var Hout = 0
    var Wout = 0
    if stride_h > 0 and stride_w > 0 and Hp >= kh and Wp >= kw:
        Hout = (Hp - kh) / stride_h + 1
        Wout = (Wp - kw) / stride_w + 1

    var y = zeros4d(N, O, Hout, Wout)
    for n in range(N):
        var yn = conv2d_single(x[n], w, b, stride_h, stride_w, pad_h, pad_w)
        y[n] = yn
    return y

# --- Conv2d module ---
struct Conv2d:
    var in_channels: Int
    var out_channels: Int
    var kh: Int
    var kw: Int
    var stride_h: Int
    var stride_w: Int
    var pad_h: Int
    var pad_w: Int
    var weight: List[List[List[List[Float64]]]]  # [O][C][kh][kw]
    var bias: List[Float64]
fn __init__(out self, in_channels: Int, out_channels: Int, kernel_h: Int, kernel_w: Int, stride_h: Int = 1, stride_w: Int = 1, pad_h: Int = 0, pad_w: Int = 0, bias: Bool = True, w_init: Float64 = 0.01) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh = kernel_h
        self.kw = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w

        # deterministic tiny init
        self.weight = List[List[List[List[Float64]]]]()
        for o in range(out_channels):
            var oc = List[List[List[Float64]]]()
            for c in range(in_channels):
                var kk = List[List[Float64]]()
                for r in range(kernel_h):
                    var row = List[Float64]()
                    for s in range(kernel_w):
                        row.push(w_init)
                    kk.push(row)
                oc.push(kk)
            self.weight.push(oc)

        self.bias = zeros1d(out_channels) if bias else zeros1d(out_channels)
fn forward_single(self, x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
        # x: [C,H,W] -> [O,Hout,Wout]
        return conv2d_single(x, self.weight, self.bias, self.stride_h, self.stride_w, self.pad_h, self.pad_w)
fn forward_batch(self, x: List[List[List[List[Float64]]]]) -> List[List[List[List[Float64]]]]:
        # x: [N,C,H,W] -> [N,O,Hout,Wout]
        return conv2d_batch(x, self.weight, self.bias, self.stride_h, self.stride_w, self.pad_h, self.pad_w)
fn __copyinit__(out self, other: Self) -> None:
        self.in_channels = other.in_channels
        self.out_channels = other.out_channels
        self.kh = other.kh
        self.kw = other.kw
        self.stride_h = other.stride_h
        self.stride_w = other.stride_w
        self.pad_h = other.pad_h
        self.pad_w = other.pad_w
        self.weight = other.weight
        self.bias = other.bias
fn __moveinit__(out self, deinit other: Self) -> None:
        self.in_channels = other.in_channels
        self.out_channels = other.out_channels
        self.kh = other.kh
        self.kw = other.kw
        self.stride_h = other.stride_h
        self.stride_w = other.stride_w
        self.pad_h = other.pad_h
        self.pad_w = other.pad_w
        self.weight = other.weight
        self.bias = other.bias
# --- Simple nonlinearity to pair with conv in tests ---
fn relu1d(x: List[Float64]) -> List[Float64]:
    var y = List[Float64]()
    for v in x: y.push(0.0 if v < 0.0 else v)
    return y
fn relu3d(x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var y = zeros3d(C, H, W)
    for c in range(C):
        for i in range(H):
            for j in range(W):
                var v = x[c][i][j]
                if v < 0.0: v = 0.0
                y[c][i][j] = v
    return y

# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # Make a 1-sample batch, 2-channel input, small image
    var N = 1
    var C = 2
    var H = 4
    var W = 4
    var x = zeros4d(N, C, H, W)
    # deterministic pattern
    for c in range(C):
        for i in range(H):
            for j in range(W):
                x[0][c][i][j] = Float64(c + 1) * 0.1 + Float64(i * W + j) * 0.01

    var conv = Conv2d(2, 3, 3, 3, 1, 1, 1, 1, True, 0.05)
    var y = conv.forward_batch(x)
    ok = ok and (len(y) == N)
    ok = ok and (len(y[0]) == 3)  # out_channels
    ok = ok and (len(y[0][0]) == H) and (len(y[0][0][0]) == W)  # same spatial due to pad=1, stride=1, k=3

    # Single-image forward
    var y1 = conv.forward_single(x[0])
    ok = ok and (len(y1) == 3)

    # A conv + relu pass
    var y_relu = relu3d(y1)
    ok = ok and (len(y_relu) == 3)

    return ok