# MIT License
# Copyright (c) 2025
# SPDX-License-Identifier: MIT
#
# Module: momijo.nn.pooling
# Path:   src/momijo/nn/pooling.mojo
#
# Minimal 2D pooling utilities (MaxPool2d & AvgPool2d) for pedagogy/smoke tests.
# List-based implementation (Float64) supporting:
#  - Single image [C,H,W] and batch [N,C,H,W]
#  - Kernel size (kh, kw), stride (sh, sw), and zero-padding (ph, pw)
#  - No dilation, no ceil_mode (floor output size)
#
# Momijo style:
# - No global vars, no `export`. Use `var` (not `let`).
# - Constructors: fn __init__(out self, ...)
# - Prefer `mut/out` over `inout`.

# --- Helpers ---
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

fn pad2d_chw(x: List[List[List[Float64]]], pad_h: Int, pad_w: Int, pad_value: Float64 = 0.0) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var Hp = H + 2 * pad_h
    var Wp = W + 2 * pad_w
    var y = zeros3d(C, Hp, Wp)
    for c in range(C):
        for i in range(Hp):
            for j in range(Wp):
                y[c][i][j] = pad_value
    for c in range(C):
        for i in range(H):
            for j in range(W):
                y[c][i + pad_h][j + pad_w] = x[c][i][j]
    return y

fn _max(a: Float64, b: Float64) -> Float64:
    if a >= b: return a
    return b

# --- Core pooling on single image [C,H,W] ---
fn maxpool2d_single(x: List[List[List[Float64]]], kh: Int, kw: Int, sh: Int, sw: Int, ph: Int, pw: Int) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var xp = pad2d_chw(x, ph, pw, -1.7976931348623157e308)  # -INF for max
    var Hp = H + 2 * ph
    var Wp = W + 2 * pw
    var Hout = 0
    var Wout = 0
    if sh > 0 and sw > 0 and Hp >= kh and Wp >= kw:
        Hout = (Hp - kh) / sh + 1
        Wout = (Wp - kw) / sw + 1
    var y = zeros3d(C, Hout, Wout)
    for c in range(C):
        for i in range(Hout):
            for j in range(Wout):
                var m = -1.7976931348623157e308
                for r in range(kh):
                    for s in range(kw):
                        var ii = i * sh + r
                        var jj = j * sw + s
                        m = _max(m, xp[c][ii][jj])
                y[c][i][j] = m
    return y

fn avgpool2d_single(x: List[List[List[Float64]]], kh: Int, kw: Int, sh: Int, sw: Int, ph: Int, pw: Int, count_include_pad: Bool = False) -> List[List[List[Float64]]]:
    var C = len(x)
    if C == 0: return x
    var H = len(x[0])
    var W = 0
    if H > 0: W = len(x[0][0])
    var pad_val = 0.0
    var xp = pad2d_chw(x, ph, pw, pad_val)
    var Hp = H + 2 * ph
    var Wp = W + 2 * pw
    var Hout = 0
    var Wout = 0
    if sh > 0 and sw > 0 and Hp >= kh and Wp >= kw:
        Hout = (Hp - kh) / sh + 1
        Wout = (Wp - kw) / sw + 1
    var y = zeros3d(C, Hout, Wout)
    for c in range(C):
        for i in range(Hout):
            for j in range(Wout):
                var sumv = 0.0
                var denom = 0.0
                for r in range(kh):
                    for s in range(kw):
                        var ii = i * sh + r
                        var jj = j * sw + s
                        sumv += xp[c][ii][jj]
                        if count_include_pad:
                            denom += 1.0
                        else:
                            # Only count if inside original (un-padded) region
                            var inside_i = (ii >= ph) and (ii < ph + H)
                            var inside_j = (jj >= pw) and (jj < pw + W)
                            if inside_i and inside_j: denom += 1.0
                if denom == 0.0: denom = 1.0
                y[c][i][j] = sumv / denom
    return y

# --- Batch wrappers [N,C,H,W] ---
fn maxpool2d_batch(x: List[List[List[List[Float64]]]], kh: Int, kw: Int, sh: Int, sw: Int, ph: Int, pw: Int) -> List[List[List[List[Float64]]]]:
    var N = len(x)
    if N == 0: return x
    var y0 = maxpool2d_single(x[0], kh, kw, sh, sw, ph, pw)
    var C = len(y0)
    var Hout = 0
    var Wout = 0
    if C > 0:
        Hout = len(y0[0])
        if Hout > 0: Wout = len(y0[0][0])
    var y = zeros4d(N, C, Hout, Wout)
    y[0] = y0
    for n in range(1, N):
        y[n] = maxpool2d_single(x[n], kh, kw, sh, sw, ph, pw)
    return y

fn avgpool2d_batch(x: List[List[List[List[Float64]]]], kh: Int, kw: Int, sh: Int, sw: Int, ph: Int, pw: Int, count_include_pad: Bool = False) -> List[List[List[List[Float64]]]]:
    var N = len(x)
    if N == 0: return x
    var y0 = avgpool2d_single(x[0], kh, kw, sh, sw, ph, pw, count_include_pad)
    var C = len(y0)
    var Hout = 0
    var Wout = 0
    if C > 0:
        Hout = len(y0[0])
        if Hout > 0: Wout = len(y0[0][0])
    var y = zeros4d(N, C, Hout, Wout)
    y[0] = y0
    for n in range(1, N):
        y[n] = avgpool2d_single(x[n], kh, kw, sh, sw, ph, pw, count_include_pad)
    return y

# --- Pooling modules ---
struct MaxPool2d:
    var kh: Int
    var kw: Int
    var sh: Int
    var sw: Int
    var ph: Int
    var pw: Int

    fn __init__(out self, kernel_h: Int, kernel_w: Int, stride_h: Int = 0, stride_w: Int = 0, pad_h: Int = 0, pad_w: Int = 0):
        self.kh = kernel_h
        self.kw = kernel_w
        self.sh = stride_h if stride_h > 0 else kernel_h
        self.sw = stride_w if stride_w > 0 else kernel_w
        self.ph = pad_h
        self.pw = pad_w

    fn forward_chw(self, x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
        return maxpool2d_single(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw)

    fn forward_nchw(self, x: List[List[List[List[Float64]]]]) -> List[List[List[List[Float64]]]]:
        return maxpool2d_batch(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw)

struct AvgPool2d:
    var kh: Int
    var kw: Int
    var sh: Int
    var sw: Int
    var ph: Int
    var pw: Int
    var count_include_pad: Bool

    fn __init__(out self, kernel_h: Int, kernel_w: Int, stride_h: Int = 0, stride_w: Int = 0, pad_h: Int = 0, pad_w: Int = 0, count_include_pad: Bool = False):
        self.kh = kernel_h
        self.kw = kernel_w
        self.sh = stride_h if stride_h > 0 else kernel_h
        self.sw = stride_w if stride_w > 0 else kernel_w
        self.ph = pad_h
        self.pw = pad_w
        self.count_include_pad = count_include_pad

    fn forward_chw(self, x: List[List[List[Float64]]]) -> List[List[List[Float64]]]:
        return avgpool2d_single(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw, self.count_include_pad)

    fn forward_nchw(self, x: List[List[List[List[Float64]]]]) -> List[List[List[List[Float64]]]]:
        return avgpool2d_batch(x, self.kh, self.kw, self.sh, self.sw, self.ph, self.pw, self.count_include_pad)

# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # Build a simple NCHW tensor: N=1, C=2, H=W=4
    var N = 1
    var C = 2
    var H = 4
    var W = 4
    var x = zeros4d(N, C, H, W)
    for c in range(C):
        for i in range(H):
            for j in range(W):
                x[0][c][i][j] = Float64(c + 1) * 0.1 + 0.01 * Float64(i * W + j)

    # MaxPool: k=2, s=2, p=0 -> H_out=W_out=2
    var mp = MaxPool2d(2, 2, 2, 2, 0, 0)
    var y_max = mp.forward_nchw(x)
    ok = ok and (len(y_max) == N) and (len(y_max[0]) == C) and (len(y_max[0][0]) == 2) and (len(y_max[0][0][0]) == 2)

    # AvgPool: k=2, s=2, p=0 -> 2x2
    var ap = AvgPool2d(2, 2, 2, 2, 0, 0, False)
    var y_avg = ap.forward_nchw(x)
    ok = ok and (len(y_avg) == N) and (len(y_avg[0]) == C) and (len(y_avg[0][0]) == 2) and (len(y_avg[0][0][0]) == 2)

    # With padding: k=3, s=2, p=1 -> H_out = floor((H + 2p - k)/s) + 1 = floor((4+2-3)/2)+1 = 2
    var mp2 = MaxPool2d(3, 3, 2, 2, 1, 1)
    var y_max2 = mp2.forward_nchw(x)
    ok = ok and (len(y_max2[0][0]) == 2) and (len(y_max2[0][0][0]) == 2)

    return ok
 
