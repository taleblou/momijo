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
# Project: momijo.nn.layers
# File: src/momijo/nn/layers/conv_transpose2d.mojo

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

# Output size formula (same as PyTorch/TF conv_transpose):
# H_out = (H_in - 1) * stride_h - 2*pad_h + kh + out_pad_h
# W_out = (W_in - 1) * stride_w - 2*pad_w + kw + out_pad_w

# --- Single image conv_transpose: x[C_in,H,W], w[C_in,C_out,kh,kw] -> y[C_out,Hout,Wout] ---
fn conv_transpose2d_single(x: List[List[List[Float64]]],
                           w: List[List[List[List[Float64]]]],
                           b: List[Float64],
                           stride_h: Int, stride_w: Int,
                           pad_h: Int, pad_w: Int,
                           out_pad_h: Int, out_pad_w: Int) -> List[List[List[Float64]]]:
    var C_in = len(x)
    if C_in == 0: return List[List[List[Float64]]]()
    var H_in = len(x[0])
    var W_in = 0
    if H_in > 0: W_in = len(x[0][0])

    var C_out = 0
    var kh = 0
    var kw = 0
    if C_in > 0 and len(w[0]) > 0:
        C_out = len(w[0])
        kh = len(w[0][0])
        kw = len(w[0][0][0])

    var H_out = (H_in - 1) * stride_h - 2 * pad_h + kh + out_pad_h
    var W_out = (W_in - 1) * stride_w - 2 * pad_w + kw + out_pad_w
    if H_out < 0: H_out = 0
    if W_out < 0: W_out = 0

    var y = zeros3d(C_out, H_out, W_out)

    for c_in in range(C_in):
        for i in range(H_in):
            for j in range(W_in):
                var v = x[c_in][i][j]
                # top-left location in output this input pixel contributes to
                var base_i = i * stride_h - pad_h
                var base_j = j * stride_w - pad_w
                for c_out in range(C_out):
                    for r in range(kh):
                        var oi = base_i + r
                        if oi < 0 or oi >= H_out: continue
                        for s in range(kw):
                            var oj = base_j + s
                            if oj < 0 or oj >= W_out: continue
                            y[c_out][oi][oj] += v * w[c_in][c_out][r][s]

    # add bias
    var has_bias = len(b) == C_out
    if has_bias:
        for c_out in range(C_out):
            for i in range(H_out):
                for j in range(W_out):
                    y[c_out][i][j] += b[c_out]
    return y

# --- Batch conv_transpose: x[N,C_in,H,W] -> y[N,C_out,Hout,Wout] ---
fn conv_transpose2d_batch(x: List[List[List[List[Float64]]]],
                          w: List[List[List[List[Float64]]]],
                          b: List[Float64],
                          stride_h: Int, stride_w: Int,
                          pad_h: Int, pad_w: Int,
                          out_pad_h: Int, out_pad_w: Int) -> List[List[List[List[Float64]]]]:
    var N = len(x)
    if N == 0: return List[List[List[List[Float64]]]]()

    # Infer output dims from the first sample
    var y0 = conv_transpose2d_single(x[0], w, b, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w)
    var C_out = len(y0)
    var H_out = 0
    var W_out = 0
    if C_out > 0:
        H_out = len(y0[0])
        if H_out > 0: W_out = len(y0[0][0])

    var y = zeros4d(N, C_out, H_out, W_out)
    y[0] = y0
    for n in range(1, N):
        y[n] = conv_transpose2d_single(x[n], w, b, stride_h, stride_w, pad_h, pad_w, out_pad_h, out_pad_w)
    return y

# --- ConvTranspose2d module ---
struct ConvTranspose2d:
    var in_channels: Int
    var out_channels: Int
    var kh: Int
    var kw: Int
    var stride_h: Int
    var stride_w: Int
    var pad_h: Int
    var pad_w: Int
    var out_pad_h: Int
    var out_pad_w: Int
    var weight: List[List[List[List[Float64]]]]  # [C_in][C_out][kh][kw]
    var bias: List[Float64]
fn __init__(out self, in_channels: Int, out_channels: Int, kernel_h: Int, kernel_w: Int, stride_h: Int = 1, stride_w: Int = 1, pad_h: Int = 0, pad_w: Int = 0, out_pad_h: Int = 0, out_pad_w: Int = 0, bias: Bool = True, w_init: Float64 = 0.01) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kh = kernel_h
        self.kw = kernel_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.out_pad_h = out_pad_h
        self.out_pad_w = out_pad_w

        # deterministic tiny init
        self.weight = List[List[List[List[Float64]]]]()
        for ci in range(in_channels):
            var oc = List[List[List[Float64]]]()
            for co in range(out_channels):
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
        return conv_transpose2d_single(x, self.weight, self.bias, self.stride_h, self.stride_w, self.pad_h, self.pad_w, self.out_pad_h, self.out_pad_w)
fn forward_batch(self, x: List[List[List[List[Float64]]]]) -> List[List[List[List[Float64]]]]:
        return conv_transpose2d_batch(x, self.weight, self.bias, self.stride_h, self.stride_w, self.pad_h, self.pad_w, self.out_pad_h, self.out_pad_w)
fn __copyinit__(out self, other: Self) -> None:
        self.in_channels = other.in_channels
        self.out_channels = other.out_channels
        self.kh = other.kh
        self.kw = other.kw
        self.stride_h = other.stride_h
        self.stride_w = other.stride_w
        self.pad_h = other.pad_h
        self.pad_w = other.pad_w
        self.out_pad_h = other.out_pad_h
        self.out_pad_w = other.out_pad_w
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
        self.out_pad_h = other.out_pad_h
        self.out_pad_w = other.out_pad_w
        self.weight = other.weight
        self.bias = other.bias
# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # Single sample: in_channels=2, out_channels=3, H=W=3, k=3, stride=2, pad=1
    var C_in = 2
    var C_out = 3
    var H = 3
    var W = 3
    var x = zeros3d(C_in, H, W)
    for c in range(C_in):
        for i in range(H):
            for j in range(W):
                x[c][i][j] = Float64(c + 1) * 0.1 + Float64(i * W + j) * 0.01

    var deconv = ConvTranspose2d(C_in, C_out, 3, 3, 2, 2, 1, 1, 0, 0, True, 0.05)
    var y = deconv.forward_single(x)

    # Expected output spatial: (3-1)*2 - 2*1 + 3 + 0 = 5  -> 5x5
    ok = ok and (len(y) == C_out)
    ok = ok and (len(y[0]) == 5) and (len(y[0][0]) == 5)

    # Batch path
    var N = 2
    var xb = zeros4d(N, C_in, H, W)
    xb[0] = x
    xb[1] = x
    var yb = deconv.forward_batch(xb)
    ok = ok and (len(yb) == N) and (len(yb[0]) == C_out) and (len(yb[0][0]) == 5) and (len(yb[0][0][0]) == 5)

    # Try output padding (to break ties when stride>1)
    var deconv2 = ConvTranspose2d(C_in, C_out, 3, 3, 2, 2, 1, 1, 1, 1, True, 0.05)
    var y2 = deconv2.forward_single(x)
    # Expected: 6x6
    ok = ok and (len(y2[0]) == 6) and (len(y2[0][0]) == 6)

    return ok