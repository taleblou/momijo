# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.resnet
# Path:   src/momijo/nn/resnet.mojo
#
# Minimal ResNet (BasicBlock) for pedagogy/smoke tests.
# List-based Float64 implementation; self-contained (no external deps).
# Supports single image CHW and batch NCHW via simple loops.
#
# Architecture (Tiny):
#   stem: 3x3 s=2 -> BN -> ReLU -> MaxPool 2x2 s=2
#   layer1: BasicBlock(in=64, out=64, stride=1) x1
#   layer2: BasicBlock(in=64, out=128, stride=2) x1
#   layer3: BasicBlock(in=128, out=256, stride=2) x1
#   layer4: BasicBlock(in=256, out=512, stride=2) x1
#   head: GAP -> Linear(num_classes)
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

# --- Convolutions ---
# 3x3 Conv: w: [O][C][3][3], b:[O]
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

# 1x1 Conv with stride: w: [O][C][1][1]
fn conv1x1_single(x: List[List[List[Float64]]],
                  w: List[List[List[List[Float64]]]],
                  b: List[Float64],
                  stride: Int) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return List[List[List[Float64]]]()
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var O = len(w)
    var Hout = (H - 1) / stride + 1
    var Wout = (W - 1) / stride + 1
    var y = zeros3d(O, Hout, Wout)
    for o in range(O):
        for i in range(Hout):
            for j in range(Wout):
                var acc = 0.0
                var ii = i * stride
                var jj = j * stride
                for c in range(C):
                    acc += w[o][c][0][0] * x[c][ii][jj]
                y[o][i][j] = acc + b[o]
    return y

# MaxPool 2x2 stride 2
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

# --- BatchNorm2d-lite (instance over HxW) ---
struct BatchNorm2dLite:
    var num_features: Int
    var eps: Float64
    var gamma: List[Float64]
    var beta: List[Float64]

    fn __init__(out self, num_features: Int, eps: Float64 = 1e-5):
        self.num_features = num_features
        self.eps = eps
        self.gamma = zeros1d(num_features)
        self.beta = zeros1d(num_features)
        for c in range(num_features):
            self.gamma[c] = 1.0
            self.beta[c] = 0.0

    fn forward_chw(self, x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
        var C = len(x)
        if C == 0: return x
        var H = len(x[0])
        var W = 0
        if H > 0: W = len(x[0][0])
        var y = zeros3d(C, H, W)
        for c in range(C):
            var sumv = 0.0
            var sumsq = 0.0
            var cnt = H * W
            if cnt <= 0: cnt = 1
            for i in range(H):
                for j in range(W):
                    var v = x[c][i][j]
                    sumv += v
                    sumsq += v * v
            var mean = sumv / Float64(cnt)
            var varg = sumsq / Float64(cnt) - mean * mean
            var denom = varg + self.eps
            var s = denom
            s = 0.5 * (s + denom / s)
            s = 0.5 * (s + denom / s)
            if s == 0.0: s = 1.0
            for i in range(H):
                for j in range(W):
                    var xhat = (x[c][i][j] - mean) / s
                    y[c][i][j] = self.gamma[c] * xhat + self.beta[c]
        return y

# --- Global Average Pool over HxW ---
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

# --- Linear classifier ---
struct Linear:
    var in_features: Int
    var out_features: Int
    var W: List[List[Float64]]  # [out,in]
    var b: List[Float64]

    fn __init__(out self, in_features: Int, out_features: Int, w_init: Float64 = 0.01):
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

# --- Basic Residual Block ---
struct BasicBlock:
    var in_ch: Int
    var out_ch: Int
    var stride: Int
    var has_proj: Bool
    # weights
    var w1: List[List[List[List[Float64]]]]; var b1: List[Float64]; var bn1: BatchNorm2dLite
    var w2: List[List[List[List[Float64]]]]; var b2: List[Float64]; var bn2: BatchNorm2dLite
    var wproj: List[List[List[List[Float64]]]]; var bproj: List[Float64]; var bnproj: BatchNorm2dLite

    fn __init__(out self, in_ch: Int, out_ch: Int, stride: Int):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.has_proj = (stride != 1) or (in_ch != out_ch)
        # conv1 3x3
        self.w1 = List[List[List[List[Float64]]]]()
        self.b1 = zeros1d(out_ch)
        for o in range(out_ch):
            var oc = List[List[List[Float64]]]()
            var k = List[List[Float64]]()
            for r in range(3):
                var row = List[Float64]()
                for s in range(3): row.push(0.01)
                k.push(row)
            for c in range(in_ch): oc.push(k)
            self.w1.push(oc)
        self.bn1 = BatchNorm2dLite(out_ch)
        # conv2 3x3
        self.w2 = List[List[List[List[Float64]]]]()
        self.b2 = zeros1d(out_ch)
        for o in range(out_ch):
            var oc2 = List[List[List[Float64]]]()
            var k2 = List[List[Float64]]()
            for r in range(3):
                var row2 = List[Float64]()
                for s in range(3): row2.push(0.01)
                k2.push(row2)
            for c in range(out_ch): oc2.push(k2)
            self.w2.push(oc2)
        self.bn2 = BatchNorm2dLite(out_ch)
        # projection 1x1 if needed
        self.wproj = List[List[List[List[Float64]]]]()
        self.bproj = zeros1d(out_ch)
        self.bnproj = BatchNorm2dLite(out_ch)
        if self.has_proj:
            for o in range(out_ch):
                var ocp = List[List[List[Float64]]]()
                var rowp = List[List[Float64]]([[0.01]])
                for c in range(in_ch): ocp.push(rowp)
                self.wproj.push(ocp)

    fn forward_chw(self, x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
        var y = conv3x3_single(x, self.w1, self.b1, self.stride, 1)
        y = self.bn1.forward_chw(y)
        y = relu_chw(y)
        y = conv3x3_single(y, self.w2, self.b2, 1, 1)
        y = self.bn2.forward_chw(y)
        var res = x
        if self.has_proj:
            res = conv1x1_single(x, self.wproj, self.bproj, self.stride)
            res = self.bnproj.forward_chw(res)
        # add
        var C = len(y); var H = 0; var W = 0
        if C > 0:
            H = len(y[0]); 
            if H > 0: W = len(y[0][0])
        var out = zeros3d(C, H, W)
        for c in range(C):
            for i in range(H):
                for j in range(W):
                    out[c][i][j] = y[c][i][j] + res[c][i][j]
        return relu_chw(out)

# --- ResNet Tiny ---
struct ResNetTiny:
    var num_classes: Int
    # stem
    var stem_w: List[List[List[List[Float64]]]]; var stem_b: List[Float64]; var stem_bn: BatchNorm2dLite
    # layers
    var l1: BasicBlock
    var l2: BasicBlock
    var l3: BasicBlock
    var l4: BasicBlock
    # head
    var head: Linear

    fn __init__(out self, num_classes: Int = 10):
        self.num_classes = num_classes
        # stem: 3x3 s2, in=3 -> 64
        var in_ch = 3; var out_ch = 64
        self.stem_w = List[List[List[List[Float64]]]]()
        self.stem_b = zeros1d(out_ch)
        for o in range(out_ch):
            var oc = List[List[List[Float64]]]()
            var k = List[List[Float64]]()
            for r in range(3):
                var row = List[Float64]()
                for s in range(3): row.push(0.01)
                k.push(row)
            for c in range(in_ch): oc.push(k)
            self.stem_w.push(oc)
        self.stem_bn = BatchNorm2dLite(out_ch)
        # layers
        self.l1 = BasicBlock(64, 64, 1)
        self.l2 = BasicBlock(64, 128, 2)
        self.l3 = BasicBlock(128, 256, 2)
        self.l4 = BasicBlock(256, 512, 2)
        self.head = Linear(512, num_classes)

    fn forward_chw(self, x: List[List[List[Float64]]]) -> List[Float64]:
        # stem
        var y = conv3x3_single(x, self.stem_w, self.stem_b, 2, 1)
        y = self.stem_bn.forward_chw(y)
        y = relu_chw(y)
        y = maxpool2x2_single(y)  # size /2
        # layers
        y = self.l1.forward_chw(y)
        y = self.l2.forward_chw(y)
        y = self.l3.forward_chw(y)
        y = self.l4.forward_chw(y)
        # head
        var feat = gap_chw(y)
        var logits = self.head.forward_vec(feat)
        return logits

    fn forward_nchw(self, x: List[List[List[List[Float64]]]]) -> List[List[Float64]]:
        var N = len(x)
        var out = List[List[Float64]]()
        for n in range(N):
            out.push(self.forward_chw(x[n]))
        return out

# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True
    var N = 1; var C = 3; var H = 64; var W = 64
    var x = zeros4d(N, C, H, W)
    for i in range(H):
        for j in range(W):
            x[0][0][i][j] = 0.01 * Float64(i * W + j)
            x[0][1][i][j] = 0.02 * Float64(i * W + j)
            x[0][2][i][j] = 0.03 * Float64(i * W + j)
    var net = ResNetTiny(10)
    var y = net.forward_nchw(x)
    ok = ok and (len(y) == N) and (len(y[0]) == 10)
    return ok
 