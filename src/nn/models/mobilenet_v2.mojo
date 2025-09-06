# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.mobilenet_v2
# Path:   src/momijo/nn/mobilenet_v2.mojo
#
# Minimal MobileNetV2 (encoder + classifier) for pedagogy/smoke tests.
# No external deps; list-based Float64 math. This is an educational
# implementation, not optimized. Batch support provided by looping over N.
#
# Building blocks:
#  - ReLU6
#  - Conv2d (3x3 & 1x1) + zero padding + stride (no dilation/groups general)
#  - DepthwiseConv2d (per-channel 3x3)
#  - BatchNorm2d-lite (per-channel over HxW; instance-style, no running stats)
#  - InvertedResidual (expansion -> depthwise -> projection [+ residual])
#  - Global Average Pool (per-channel mean over HxW)
#  - Linear classifier (features -> num_classes)
#
# Shapes: CHW for single image; NCHW for batches.

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

fn relu6(x: Float64) -> Float64:
    var v = x
    if v < 0.0: v = 0.0
    if v > 6.0: v = 6.0
    return v

fn relu6_chw(x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var y = zeros3d(C, H, W)
    for c in range(C):
        for i in range(H):
            for j in range(W):
                y[c][i][j] = relu6(x[c][i][j])
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

# --- Pointwise 1x1 Conv (single image) ---
# w: [O][C][1][1], b:[O]
fn conv1x1_single(x: List[List[List[Float64]]],
                  w: List[List[List[List[Float64]]]],
                  b: List[Float64]) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return List[List[List[Float64]]]()
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var O = len(w)
    var y = zeros3d(O, H, W)
    for o in range(O):
        for i in range(H):
            for j in range(W):
                var acc = 0.0
                for c in range(C):
                    acc += w[o][c][0][0] * x[c][i][j]
                y[o][i][j] = acc + b[o]
    return y

# --- Standard 3x3 Conv (no groups) single image ---
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
    var kh = 3; var kw = 3
    var Hout = 0; var Wout = 0
    if stride > 0 and Hp >= kh and Wp >= kw:
        Hout = (Hp - kh) / stride + 1
        Wout = (Wp - kw) / stride + 1
    var y = zeros3d(O, Hout, Wout)
    for o in range(O):
        for i in range(Hout):
            for j in range(Wout):
                var acc = 0.0
                for c in range(C):
                    for r in range(kh):
                        for s in range(kw):
                            var ii = i * stride + r
                            var jj = j * stride + s
                            acc += w[o][c][r][s] * xp[c][ii][jj]
                y[o][i][j] = acc + b[o]
    return y

# --- Depthwise 3x3 Conv (per-channel) single image ---
# wdw: [C][3][3], b:[C]
fn depthwise3x3_single(x: List[List[List[Float64]]],
                       wdw: List[List[List[Float64]]],
                       b: List[Float64],
                       stride: Int, pad: Int) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var xp = pad2d_chw(x, pad, pad)
    var Hp = H + 2 * pad
    var Wp = W + 2 * pad
    var kh = 3; var kw = 3
    var Hout = 0; var Wout = 0
    if stride > 0 and Hp >= kh and Wp >= kw:
        Hout = (Hp - kh) / stride + 1
        Wout = (Wp - kw) / stride + 1
    var y = zeros3d(C, Hout, Wout)
    for c in range(C):
        for i in range(Hout):
            for j in range(Wout):
                var acc = 0.0
                for r in range(kh):
                    for s in range(kw):
                        var ii = i * stride + r
                        var jj = j * stride + s
                        acc += wdw[c][r][s] * xp[c][ii][jj]
                y[c][i][j] = acc + b[c]
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

# --- Inverted Residual Block ---
struct InvertedResidual:
    var in_ch: Int
    var out_ch: Int
    var stride: Int
    var expand_ch: Int
    # weights
    var w_exp: List[List[List[List[Float64]]]]; var b_exp: List[Float64]
    var bn1: BatchNorm2dLite
    var w_dw: List[List[Float64]]; var b_dw: List[Float64]  # depthwise [C][3][3]
    var bn2: BatchNorm2dLite
    var w_proj: List[List[List[List[Float64]]]]; var b_proj: List[Float64]
    var bn3: BatchNorm2dLite

    fn __init__(out self, in_ch: Int, out_ch: Int, stride: Int, expand_ratio: Int):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        if expand_ratio < 1: expand_ratio = 1
        self.expand_ch = in_ch * expand_ratio

        # expansion 1x1
        self.w_exp = List[List[List[List[Float64]]]]()
        self.b_exp = zeros1d(self.expand_ch)
        for o in range(self.expand_ch):
            var oc = List[List[List[Float64]]]()
            var row = List[List[Float64]]([[0.01]])
            for c in range(in_ch):
                oc.push(row)  # 1x1
            self.w_exp.push(oc)
        self.bn1 = BatchNorm2dLite(self.expand_ch)

        # depthwise 3x3
        self.w_dw = List[List[Float64]]()
        self.b_dw = zeros1d(self.expand_ch)
        for c in range(self.expand_ch):
            var k = List[Float64]()
            # 3x3 kernel filled with small constants
            for r in range(3):
                for s in range(3):
                    k.push(0.01)
            self.w_dw.push(k)   # flattened [9]
        self.bn2 = BatchNorm2dLite(self.expand_ch)

        # projection 1x1 -> out_ch
        self.w_proj = List[List[List[List[Float64]]]]()
        self.b_proj = zeros1d(out_ch)
        for o in range(out_ch):
            var oc = List[List[List[Float64]]]()
            var rowp = List[List[Float64]]([[0.01]])
            for c in range(self.expand_ch):
                oc.push(rowp)
            self.w_proj.push(oc)
        self.bn3 = BatchNorm2dLite(out_ch)

    fn _depthwise3x3(self, x: List[List[List[Float64]]], stride: Int, pad: Int) -> List[List[List[Float64]]]:
        # adapt flattened kernel to [C][3][3]
        var C = len(x)
        var kd = List[List[List[Float64]]]()
        for c in range(C):
            var k = List[Float64]()
            for i in range(9): k.push(self.w_dw[c][i])
            var m = List[List[Float64]]()
            var idx = 0
            for r in range(3):
                var row = List[Float64]()
                for s in range(3):
                    row.push(k[idx]); idx += 1
                m.push(row)
            kd.push(m)
        return depthwise3x3_single(x, kd, self.b_dw, stride, pad)

    fn forward_chw(self, x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
        var use_res = (self.stride == 1) and (self.in_ch == self.out_ch)
        # 1) expand
        var y = conv1x1_single(x, self.w_exp, self.b_exp)
        y = self.bn1.forward_chw(y)
        y = relu6_chw(y)
        # 2) depthwise
        y = self._depthwise3x3(y, self.stride, 1)
        y = self.bn2.forward_chw(y)
        y = relu6_chw(y)
        # 3) project
        y = conv1x1_single(y, self.w_proj, self.b_proj)
        y = self.bn3.forward_chw(y)
        if use_res:
            # elementwise add
            var C = len(y); var H = 0; var W = 0
            if C > 0:
                H = len(y[0]); 
                if H > 0: W = len(y[0][0])
            var out = zeros3d(C, H, W)
            for c in range(C):
                for i in range(H):
                    for j in range(W):
                        out[c][i][j] = y[c][i][j] + x[c][i][j]
            return out
        return y

# --- Tiny MobileNetV2 backbone ---
struct MobileNetV2Tiny:
    var num_classes: Int
    # stem
    var stem_w: List[List[List[List[Float64]]]]; var stem_b: List[Float64]
    var stem_bn: BatchNorm2dLite
    # blocks
    var b1: InvertedResidual
    var b2: InvertedResidual
    var b3: InvertedResidual
    # head
    var head_w: List[List[List[List[Float64]]]]; var head_b: List[Float64]
    var head_bn: BatchNorm2dLite
    var classifier: Linear

    fn __init__(out self, num_classes: Int = 10):
        self.num_classes = num_classes
        # stem: 3x3 s2, in=3 -> 32
        var in_ch = 3; var out_ch = 32
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
        # blocks (subset of standard config)
        # (t, c, n, s): (1,16,1,1), (6,24,2,2), (6,32,3,2) -> here 3 blocks total for brevity
        self.b1 = InvertedResidual(32, 16, 1, 1)  # t=1, s=1
        self.b2 = InvertedResidual(16, 24, 2, 6)  # t=6, s=2
        self.b3 = InvertedResidual(24, 32, 2, 6)  # t=6, s=2
        # head: 1x1 to 128
        var head_out = 128
        self.head_w = List[List[List[List[Float64]]]]()
        self.head_b = zeros1d(head_out)
        for o in range(head_out):
            var oc2 = List[List[List[Float64]]]()
            var rowp = List[List[Float64]]([[0.01]])
            for c in range(32): oc2.push(rowp)
            self.head_w.push(oc2)
        self.head_bn = BatchNorm2dLite(head_out)
        self.classifier = Linear(head_out, num_classes)

    fn forward_chw(self, x: List[List[List[Float64]]]) -> List[Float64]:
        # stem
        var y = conv3x3_single(x, self.stem_w, self.stem_b, 2, 1)
        y = self.stem_bn.forward_chw(y)
        y = relu6_chw(y)
        # blocks
        y = self.b1.forward_chw(y)
        y = self.b2.forward_chw(y)
        y = self.b3.forward_chw(y)
        # head
        y = conv1x1_single(y, self.head_w, self.head_b)
        y = self.head_bn.forward_chw(y)
        y = relu6_chw(y)
        # GAP -> classifier
        var feat = gap_chw(y)
        var logits = self.classifier.forward_vec(feat)
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
    var N = 1; var C = 3; var H = 32; var W = 32
    var x = zeros4d(N, C, H, W)
    for i in range(H):
        for j in range(W):
            x[0][0][i][j] = 0.01 * Float64(i * W + j)
            x[0][1][i][j] = 0.02 * Float64(i * W + j)
            x[0][2][i][j] = 0.03 * Float64(i * W + j)
    var net = MobileNetV2Tiny(10)
    var y = net.forward_nchw(x)
    ok = ok and (len(y) == N) and (len(y[0]) == 10)
    return ok
 
