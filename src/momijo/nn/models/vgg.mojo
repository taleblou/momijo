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
# Project: momijo.nn.models
# File: src/momijo/nn/models/vgg.mojo

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
fn relu(x: Float64) -> Float64:
    if x < 0.0: return 0.0
    return x
fn relu_chw(x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var y = zeros3d(C, H, W)
    for c in range(C):
        for i in range(H):
            for j in range(W):
                y[c][i][j] = relu(x[c][i][j])
    return y
fn pad2d_chw(x: List[List[List[Float64]]], ph: Int, pw: Int) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var Hp = H + 2 * ph
    var Wp = W + 2 * pw
    var y = zeros3d(C, Hp, Wp)
    for c in range(C):
        for i in range(H):
            for j in range(W):
                y[c][i + ph][j + pw] = x[c][i][j]
    return y

# 3x3 Conv single image: w [O][C][3][3], b[O]
fn conv3x3_single(x: List[List[List[Float64]]],
                  w: List[List[List[List[Float64]]]],
                  b: List[Float64],
                  stride: Int, pad: Int) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return List[List[List[Float64]]]()
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var xp = pad2d_chw(x, pad, pad)
    var Hp = H + 2 * pad
    var Wp = W + 2 * pad
    var O = len(w)
    var Hout = 0; var Wout = 0
    if stride > 0 and Hp >= 3 and Wp >= 3:
        Hout = (Hp - 3) / stride + 1
        Wout = (Wp - 3) / stride + 1
    var y = zeros3d(O, Hout, Wout)
    for o in range(O):
        for i in range(Hout):
            for j in range(Wout):
                var acc = 0.0
                for c in range(C):
                    for r in range(3):
                        for s in range(3):
                            var ii = i * stride + r
                            var jj = j * stride + s
                            acc += w[o][c][r][s] * xp[c][ii][jj]
                y[o][i][j] = acc + b[o]
    return y

# MaxPool 2x2 s2
fn maxpool2x2_single(x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0]); var W = 0
    if H > 0: W = len(x[0][0])
    var Hout = H / 2
    var Wout = W / 2
    var y = zeros3d(C, Hout, Wout)
    for c in range(C):
        for i in range(Hout):
            for j in range(Wout):
                var m = -1.7976931348623157e308
                for r in range(2):
                    for s in range(2):
                        var ii = i * 2 + r
                        var jj = j * 2 + s
                        var v = x[c][ii][jj]
                        if v > m: m = v
                y[c][i][j] = m
    return y

# Global Average Pool over HxW
fn gap_chw(x: List[List[List[Float64]]]) -> List[Float64]:
    var C = len(x)
    var H = 0
    var W = 0
    if C > 0:
        H = len(x[0])
        if H > 0: W = len(x[0][0])
    var out = zeros1d(C)
    var denom = Float64(H * W)
    if denom <= 0.0: denom = 1.0
    for c in range(C):
        var s = 0.0
        for i in range(H):
            for j in range(W):
                s += x[c][i][j]
        out[c] = s / denom
    return out

# Linear
struct Linear:
    var in_features: Int
    var out_features: Int
    var W: List[List[Float64]]  # [out,in]
    var b: List[Float64]
fn __init__(out self, in_features: Int, out_features: Int, w_init: Float64 = 0.01) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.W = zeros2d(out_features, in_features)
        self.b = zeros1d(out_features)
        for o in range(out_features):
            for i in range(in_features):
                self.W[o][i] = w_init
fn forward_vec(self, x: List[Float64]) -> List[Float64]:
        var y = zeros1d(self.out_features)
        for o in range(self.out_features):
            var s = 0.0
            for i in range(self.in_features):
                s += self.W[o][i] * x[i]
            y[o] = s + self.b[o]
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
# --- VGG Block (n convs of 3x3, then MaxPool) ---
struct VGGBlock:
    var in_ch: Int
    var out_ch: Int
    var nconv: Int
    var ws: List[List[List[List[List[Float64]]]]]  # [nconv][O][C][3][3]
    var bs: List[List[Float64]]                    # [nconv][O]
fn __init__(out self, in_ch: Int, out_ch: Int, nconv: Int) -> None:
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nconv = nconv
        self.ws = List[List[List[List[List[Float64]]]]]()
        self.bs = List[List[Float64]]()
        var cin = in_ch
        for k in range(nconv):
            var cout = out_ch
            var Wk = List[List[List[List[Float64]]]]()
            var bk = zeros1d(cout)
            for o in range(cout):
                var oc = List[List[List[Float64]]]()
                var k3 = List[List[Float64]]()
                for r in range(3):
                    var row = List[Float64]()
                    for s in range(3): row.push(0.01)
                    k3.push(row)
                for c in range(cin): oc.push(k3)
                Wk.push(oc)
            self.ws.push(Wk)
            self.bs.push(bk)
            cin = out_ch  # next layer input chans
fn forward_chw(self, x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
        var y = x
        var cin = self.in_ch
        for k in range(self.nconv):
            y = conv3x3_single(y, self.ws[k], self.bs[k], 1, 1)
            y = relu_chw(y)
            cin = self.out_ch
        y = maxpool2x2_single(y)
        return y
fn __copyinit__(out self, other: Self) -> None:
        self.in_ch = other.in_ch
        self.out_ch = other.out_ch
        self.nconv = other.nconv
        self.ws = other.ws
        self.bs = other.bs
fn __moveinit__(out self, deinit other: Self) -> None:
        self.in_ch = other.in_ch
        self.out_ch = other.out_ch
        self.nconv = other.nconv
        self.ws = other.ws
        self.bs = other.bs
# --- VGG Tiny ---
struct VGGNetTiny:
    var num_classes: Int
    var b1: VGGBlock
    var b2: VGGBlock
    var b3: VGGBlock
    var classifier: Linear  # (after GAP): 256 -> num_classes
fn __init__(out self, num_classes: Int = 10) -> None:
        self.num_classes = num_classes
        self.b1 = VGGBlock(3, 64, 1)
        self.b2 = VGGBlock(64, 128, 1)
        self.b3 = VGGBlock(128, 256, 2)
        self.classifier = Linear(256, num_classes)
fn forward_chw(self, x: List[List[List[Float64]]]) -> List[Float64]:
        var y = self.b1.forward_chw(x)
        y = self.b2.forward_chw(y)
        y = self.b3.forward_chw(y)
        var feat = gap_chw(y)
        var logits = self.classifier.forward_vec(feat)
        return logits
fn forward_nchw(self, x: List[List[List[List[Float64]]]]) -> List[List[Float64]]:
        var N = len(x)
        var out = List[List[Float64]]()
        for n in range(N):
            out.push(self.forward_chw(x[n]))
        return out
fn __copyinit__(out self, other: Self) -> None:
        self.num_classes = other.num_classes
        self.b1 = other.b1
        self.b2 = other.b2
        self.b3 = other.b3
        self.classifier = other.classifier
fn __moveinit__(out self, deinit other: Self) -> None:
        self.num_classes = other.num_classes
        self.b1 = other.b1
        self.b2 = other.b2
        self.b3 = other.b3
        self.classifier = other.classifier
# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True
    var N = 1; var C = 3; var H = 32; var W = 32
    var x = zeros4d(N, C, H, W)
    for i in range(H):
        for j in range(W):
            x[0][0][i][j] = 0.01 * Float64(i * W + j)
            x[0][1][i][j] = 0.02 * Float64(i * W + j)
            x[0][2][i][j] = 0.03 * Float64(i * W + j)
    var net = VGGNetTiny(10)
    var y = net.forward_nchw(x)
    ok = ok and (len(y) == N) and (len(y[0]) == 10)
    return ok