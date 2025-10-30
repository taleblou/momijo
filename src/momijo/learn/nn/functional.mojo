# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.nn.functional
# File:         src/momijo/learn/nn/functional.mojo
#
# Description:
#   Functional (stateless) neural ops inspired by torch.nn.functional.
#   - List-based reference ops over 4D NCHW (Float64) for clarity/correctness demos.
#   - Tensor-based ops using momijo.tensor facade:
#       * im2col / col2im (+ indices variant)
#       * conv1d / conv2d (naive + im2col) / depthwise / pointwise / separable
#       * max_pool2d (reference on List-based NCHW)
#   Shapes:
#     * NCHW: [N, C, H, W]
#     * OIHW: [O, I, KH, KW]
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
struct Conv2dConfig:
    var stride_h: Int
    var stride_w: Int
    var pad_h: Int
    var pad_w: Int
    var dil_h: Int
    var dil_w: Int

    fn __init__(
        out self,
        stride_h: Int = 1,
        stride_w: Int = 1,
        pad_h: Int = 0,
        pad_w: Int = 0,
        dil_h: Int = 1,
        dil_w: Int = 1
    ):
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.dil_h = dil_h
        self.dil_w = dil_w

# -----------------------------------------------------------------------------
# Helpers for List-based NCHW (Float64)
# -----------------------------------------------------------------------------
# Type spelled explicitly: List[List[List[List[Float64]]]]

fn _zeros_4d(n: Int, c: Int, h: Int, w: Int) -> List[List[List[List[Float64]]]]:
    var out = List[List[List[List[Float64]]]]()
    var ni = 0
    while ni < n:
        var cs = List[List[List[Float64]]]()
        var ci = 0
        while ci < c:
            var hs = List[List[Float64]]()
            var yi = 0
            while yi < h:
                var ws = List[Float64]()
                var xi = 0
                while xi < w:
                    ws.append(0.0)
                    xi = xi + 1
                hs.append(ws)
                yi = yi + 1
            cs.append(hs)
            ci = ci + 1
        out.append(cs)
        ni = ni + 1
    return out

fn _pad2d_nchw(
    x: List[List[List[List[Float64]]]],
    pad_h: Int,
    pad_w: Int
) -> List[List[List[List[Float64]]]]:
    if pad_h == 0 and pad_w == 0:
        return x

    var n = Int(x.size())
    var c = Int(x[0].size())
    var h = Int(x[0][0].size())
    var w = Int(x[0][0][0].size())

    var hp = h + 2 * pad_h
    var wp = w + 2 * pad_w
    var y = _zeros_4d(n, c, hp, wp)

    var ni = 0
    while ni < n:
        var ci = 0
        while ci < c:
            var yi = 0
            while yi < h:
                var xi = 0
                while xi < w:
                    y[ni][ci][yi + pad_h][xi + pad_w] = x[ni][ci][yi][xi]
                    xi = xi + 1
                yi = yi + 1
            ci = ci + 1
        ni = ni + 1
    return y

# -----------------------------------------------------------------------------
# conv2d (reference on Lists)
# -----------------------------------------------------------------------------
# x: [N,C_in,H_in,W_in], weight: [C_out,C_in,KH,KW], bias: [C_out] or empty

fn conv2d(
    x: List[List[List[List[Float64]]]],
    weight: List[List[List[List[Float64]]]],
    bias: List[Float64] = List[Float64](),
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1
) -> List[List[List[List[Float64]]]]:
    var cfg = Conv2dConfig(stride, stride, padding, padding, dilation, dilation)

    var N = Int(x.size())
    var C_in = Int(x[0].size())
    var H_in = Int(x[0][0].size())
    var W_in = Int(x[0][0][0].size())

    var C_out = Int(weight.size())
    assert C_out > 0
    assert C_in == Int(weight[0].size())

    var KH = Int(weight[0][0].size())
    var KW = Int(weight[0][0][0].size())

    var x_pad = _pad2d_nchw(x, cfg.pad_h, cfg.pad_w)
    var H_pad = Int(x_pad[0][0].size())
    var W_pad = Int(x_pad[0][0][0].size())

    var H_out = Int(((H_pad - (cfg.dil_h * (KH - 1) + 1)) // cfg.stride_h) + 1)
    var W_out = Int(((W_pad - (cfg.dil_w * (KW - 1) + 1)) // cfg.stride_w) + 1)
    assert H_out > 0
    assert W_out > 0

    var y = _zeros_4d(N, C_out, H_out, W_out)

    var n = 0
    while n < N:
        var co = 0
        while co < C_out:
            var oy = 0
            while oy < H_out:
                var ox = 0
                while ox < W_out:
                    var acc = 0.0
                    var ci = 0
                    while ci < C_in:
                        var ky = 0
                        while ky < KH:
                            var kx = 0
                            while kx < KW:
                                var iy = oy * cfg.stride_h + ky * cfg.dil_h
                                var ix = ox * cfg.stride_w + kx * cfg.dil_w
                                acc = acc + x_pad[n][ci][iy][ix] * weight[co][ci][ky][kx]
                                kx = kx + 1
                            ky = ky + 1
                        ci = ci + 1
                    if bias.size() == C_out:
                        acc = acc + bias[co]
                    y[n][co][oy][ox] = acc
                    ox = ox + 1
                oy = oy + 1
            co = co + 1
        n = n + 1

    return y

# -----------------------------------------------------------------------------
# max_pool2d (reference on Lists)
# -----------------------------------------------------------------------------

fn max_pool2d(
    x: List[List[List[List[Float64]]]],
    kernel_size: Int,
    stride: Int = 0,
    padding: Int = 0,
    ceil_mode: Bool = False
) -> List[List[List[List[Float64]]]]:
    assert kernel_size > 0
    var s = stride
    if s <= 0:
        s = kernel_size

    var x_pad = _pad2d_nchw(x, padding, padding)

    var N = Int(x_pad.size())
    var C = Int(x_pad[0].size())
    var H = Int(x_pad[0][0].size())
    var W = Int(x_pad[0][0][0].size())

    var H_out = 0
    var W_out = 0
    if ceil_mode:
        H_out = Int(((H - kernel_size) + (s - 1)) // s + 1)
        W_out = Int(((W - kernel_size) + (s - 1)) // s + 1)
    else:
        H_out = Int(((H - kernel_size) // s) + 1)
        W_out = Int(((W - kernel_size) // s) + 1)
    assert H_out > 0
    assert W_out > 0

    var y = _zeros_4d(N, C, H_out, W_out)

    var n = 0
    while n < N:
        var c = 0
        while c < C:
            var oy = 0
            while oy < H_out:
                var ox = 0
                while ox < W_out:
                    var maxv = -1.7976931348623157e308  # Float64 min-ish
                    var ky = 0
                    while ky < kernel_size:
                        var kx = 0
                        while kx < kernel_size:
                            var iy = oy * s + ky
                            var ix = ox * s + kx
                            if iy < H and ix < W:
                                var v = x_pad[n][c][iy][ix]
                                if v > maxv:
                                    maxv = v
                            kx = kx + 1
                        ky = ky + 1
                    y[n][c][oy][ox] = maxv
                    ox = ox + 1
                oy = oy + 1
            c = c + 1
        n = n + 1

    return y

# -----------------------------------------------------------------------------
# Tensor utilities
# -----------------------------------------------------------------------------

@always_inline
fn _out_dim(in_size: Int, k: Int, stride: Int, pad: Int) -> Int:
    var s = stride
    if s <= 0:
        s = 1
    var o = in_size + 2 * pad - k
    if o < 0:
        o = 0
    return o // s + 1

@always_inline
fn _get4[T: Copyable](x: tensor.Tensor[T], n: Int, c: Int, h: Int, w: Int) -> T:
    var sh = x.shape()
    var N = sh[0]; var C = sh[1]; var H = sh[2]; var W = sh[3]
    var inside = 1
    if n < 0 or n >= N: inside = 0
    if c < 0 or c >= C: inside = 0
    if h < 0 or h >= H: inside = 0
    if w < 0 or w >= W: inside = 0
    if inside == 0:
        return T(0)
    var idx = ((n * C + c) * H + h) * W + w
    return x._data[idx]

@always_inline
fn _set4[T: Copyable & Movable](x: tensor.Tensor[T], n: Int, c: Int, h: Int, w: Int, v: T):
    var sh = x.shape()
    var idx = ((n * sh[1] + c) * sh[2] + h) * sh[3] + w
    x._data[idx] = v

# -----------------------------------------------------------------------------
# im2col / col2im
# -----------------------------------------------------------------------------
# im2col output: [N, C*kH*kW, out_h*out_w]

fn im2col[T: Copyable](x: tensor.Tensor[T], kH: Int, kW: Int, stride: Int = 1, padding: Int = 0) -> tensor.Tensor[T]:
    var s = x.shape()
    var N = s[0]; var C = s[1]; var H = s[2]; var W = s[3]
    var out_h = _out_dim(H, kH, stride, padding)
    var out_w = _out_dim(W, kW, stride, padding)
    var KC = C * kH * kW
    var L = out_h * out_w

    var data = List[T]()
    data.reserve(N * KC * L)
    var n = 0
    while n < N:
        var oh = 0
        while oh < out_h:
            var ih0 = oh * stride - padding
            var ow = 0
            while ow < out_w:
                var iw0 = ow * stride - padding
                var c = 0
                while c < C:
                    var kh = 0
                    while kh < kH:
                        var ih = ih0 + kh
                        var kw = 0
                        while kw < kW:
                            var iw = iw0 + kw
                            data.append(_get4(x, n, c, ih, iw))
                            kw = kw + 1
                        kh = kh + 1
                    c = c + 1
                ow = ow + 1
            oh = oh + 1
        n = n + 1
    return tensor.Tensor[T](data, [N, KC, L])

# Linear indices variant, using Int tensor
fn im2col_indices(x: tensor.Tensor[Int], kH: Int, kW: Int, stride: Int = 1, padding: Int = 0) -> tensor.Tensor[Int]:
    var s = x.shape()
    var N = s[0]; var C = s[1]; var H = s[2]; var W = s[3]
    var out_h = _out_dim(H, kH, stride, padding)
    var out_w = _out_dim(W, kW, stride, padding)
    var KC = C * kH * kW
    var L = out_h * out_w

    var data = List[Int]()
    data.reserve(N * KC * L)
    var n = 0
    while n < N:
        var oh = 0
        while oh < out_h:
            var ih0 = oh * stride - padding
            var ow = 0
            while ow < out_w:
                var iw0 = ow * stride - padding
                var c = 0
                while c < C:
                    var kh = 0
                    while kh < kH:
                        var ih = ih0 + kh
                        var kw = 0
                        while kw < kW:
                            var iw = iw0 + kw
                            var inside = 1
                            if ih < 0 or ih >= H: inside = 0
                            if iw < 0 or iw >= W: inside = 0
                            var lin = 0
                            if inside == 1:
                                lin = ((n * C + c) * H + ih) * W + iw
                            data.append(lin)
                            kw = kw + 1
                        kh = kh + 1
                    c = c + 1
                ow = ow + 1
            oh = oh + 1
        n = n + 1
    return tensor.Tensor[Int](data, [N, KC, L])

fn col2im[T: Copyable & Movable](cols: tensor.Tensor[T], x_shape: List[Int], kH: Int, kW: Int, stride: Int = 1, padding: Int = 0) -> tensor.Tensor[T]:
    var N = x_shape[0]; var C = x_shape[1]; var H = x_shape[2]; var W = x_shape[3]
    var out_h = _out_dim(H, kH, stride, padding)
    var out_w = _out_dim(W, kW, stride, padding)
    var KC = C * kH * kW
    var L = out_h * out_w

    var out = tensor.Tensor[T]([N, C, H, W], T(0))
    var n = 0
    while n < N:
        var oh = 0
        while oh < out_h:
            var ih0 = oh * stride - padding
            var ow = 0
            while ow < out_w:
                var iw0 = ow * stride - padding
                var c = 0
                while c < C:
                    var kh = 0
                    while kh < kH:
                        var ih = ih0 + kh
                        var kw = 0
                        while kw < kW:
                            var iw = iw0 + kw
                            var v = cols._data[(n * KC + (c * kH + kh) * kW + kw) * L + (oh * out_w + ow)]
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                var cur = _get4(out, n, c, ih, iw)
                                _set4(out, n, c, ih, iw, cur + v)
                            kw = kw + 1
                        kh = kh + 1
                    c = c + 1
                ow = ow + 1
            oh = oh + 1
        n = n + 1
    return out

fn col2im_indices[T: Copyable & Movable](cols: tensor.Tensor[T], idxs: tensor.Tensor[Int], x_shape: List[Int]) -> tensor.Tensor[T]:
    var N = x_shape[0]; var C = x_shape[1]; var H = x_shape[2]; var W = x_shape[3]
    var out = tensor.Tensor[T]([N, C, H, W], T(0))
    var sh = cols.shape()
    var N2 = sh[0]; var K = sh[1]; var L = sh[2]

    var n = 0
    while n < N2:
        var k = 0
        while k < K:
            var l = 0
            while l < L:
                var lin = idxs._data[(n * K + k) * L + l]
                if lin > 0:
                    var nn = lin // (C * H * W)
                    var rem = lin - nn * (C * H * W)
                    var cc = rem // (H * W)
                    rem = rem - cc * (H * W)
                    var hh = rem // W
                    var ww = rem - hh * W
                    var cur = _get4(out, nn, cc, hh, ww)
                    var addv = cols._data[(n * K + k) * L + l]
                    _set4(out, nn, cc, hh, ww, cur + addv)
                l = l + 1
            k = k + 1
        n = n + 1
    return out

# -----------------------------------------------------------------------------
# conv2d family (Tensor)
# -----------------------------------------------------------------------------

fn conv2d_naive[T: Copyable & Movable](x: tensor.Tensor[T], w: tensor.Tensor[T], stride: Int = 1, padding: Int = 0) -> tensor.Tensor[T]:
    var xs = x.shape(); var ws = w.shape()
    var N = xs[0]; var C = xs[1]; var H = xs[2]; var W = xs[3]
    var F = ws[0]; var Cw = ws[1]; var kH = ws[2]; var kW = ws[3]
    assert C == Cw

    var out_h = _out_dim(H, kH, stride, padding)
    var out_w = _out_dim(W, kW, stride, padding)
    var y = tensor.Tensor[T]([N, F, out_h, out_w], T(0))

    var n = 0
    while n < N:
        var f = 0
        while f < F:
            var oh = 0
            while oh < out_h:
                var ih0 = oh * stride - padding
                var ow = 0
                while ow < out_w:
                    var iw0 = ow * stride - padding
                    var acc = T(0)
                    var c = 0
                    while c < C:
                        var kh = 0
                        while kh < kH:
                            var ih = ih0 + kh
                            var kw = 0
                            while kw < kW:
                                var iw = iw0 + kw
                                var xv = _get4(x, n, c, ih, iw)
                                var wv = w._data[((f * C + c) * kH + kh) * kW + kw]
                                acc = acc + xv * wv
                                kw = kw + 1
                            kh = kh + 1
                        c = c + 1
                    y._data[((n * F + f) * out_h + oh) * out_w + ow] = acc
                    ow = ow + 1
                oh = oh + 1
            f = f + 1
        n = n + 1
    return y

fn conv2d[T: Copyable & Movable](x: tensor.Tensor[T], w: tensor.Tensor[T], stride: Int = 1, padding: Int = 0) -> tensor.Tensor[T]:
    var ws = w.shape()
    var F = ws[0]; var C = ws[1]; var kH = ws[2]; var kW = ws[3]
    var cols = im2col(x, kH, kW, stride, padding)  # [N, C*kH*kW, L]
    var sh = x.shape()
    var out_h = _out_dim(sh[2], kH, stride, padding)
    var out_w = _out_dim(sh[3], kW, stride, padding)
    var L = out_h * out_w
    var K = C * kH * kW

    var wflat = tensor.Tensor[T]([F, K], T(0))
    var f = 0
    while f < F:
        var c = 0
        while c < C:
            var kh = 0
            while kh < kH:
                var kw = 0
                while kw < kW:
                    var src = ((f * C + c) * kH + kh) * kW + kw
                    var dst = f * K + (c * kH + kh) * kW + kw
                    wflat._data[dst] = w._data[src]
                    kw = kw + 1
                kh = kh + 1
            c = c + 1
        f = f + 1
    var y = tensor.Tensor[T]([sh[0], F, out_h, out_w], T(0))

    var n = 0
    while n < sh[0]:
        var l = 0
        while l < L:
            var f2 = 0
            while f2 < F:
                var acc = T(0)
                var k = 0
                var lim = (K // 8) * 8
                while k < lim:
                    acc = acc + wflat._data[f2 * K + k    ] * cols._data[(n * K + k    ) * L + l]
                    acc = acc + wflat._data[f2 * K + k + 1] * cols._data[(n * K + k + 1) * L + l]
                    acc = acc + wflat._data[f2 * K + k + 2] * cols._data[(n * K + k + 2) * L + l]
                    acc = acc + wflat._data[f2 * K + k + 3] * cols._data[(n * K + k + 3) * L + l]
                    acc = acc + wflat._data[f2 * K + k + 4] * cols._data[(n * K + k + 4) * L + l]
                    acc = acc + wflat._data[f2 * K + k + 5] * cols._data[(n * K + k + 5) * L + l]
                    acc = acc + wflat._data[f2 * K + k + 6] * cols._data[(n * K + k + 6) * L + l]
                    acc = acc + wflat._data[f2 * K + k + 7] * cols._data[(n * K + k + 7) * L + l]
                    k = k + 8
                while k < K:
                    acc = acc + wflat._data[f2 * K + k] * cols._data[(n * K + k) * L + l]
                    k = k + 1
                y._data[((n * F + f2) * out_h + (l // out_w)) * out_w + (l % out_w)] = acc
                f2 = f2 + 1
            l = l + 1
        n = n + 1
    return y

# conv1d via reshape to 2d with H=1
fn conv1d[T: Copyable & Movable](x: tensor.Tensor[T], w: tensor.Tensor[T], stride: Int = 1, padding: Int = 0) -> tensor.Tensor[T]:
    var xs = x.shape()  # expect [N,C,W]
    var N = xs[0]; var C = xs[1]; var W = xs[2]
    var x2 = tensor.Tensor[T]([N, C, 1, W], T(0))
    var n = 0
    while n < N:
        var c = 0
        while c < C:
            var w0 = 0
            while w0 < W:
                x2._data[((n * C + c) * 1 + 0) * W + w0] = x._data[(n * C + c) * W + w0]
                w0 = w0 + 1
            c = c + 1
        n = n + 1
    var ws = w.shape()  # [F,C,kW]
    var F = ws[0]; var kW = ws[2]
    var w2 = tensor.Tensor[T]([F, C, 1, kW], T(0))
    var f = 0
    while f < F:
        var c2 = 0
        while c2 < C:
            var kw = 0
            while kw < kW:
                w2._data[((f * C + c2) * 1 + 0) * kW + kw] = w._data[(f * C + c2) * kW + kw]
                kw = kw + 1
            c2 = c2 + 1
        f = f + 1
    var y2 = conv2d(x2, w2, stride, padding)
    var ys = y2.shape()
    var outW = ys[3]
    var y = tensor.Tensor[T]([ys[0], ys[1], outW], T(0))
    var n2 = 0
    while n2 < ys[0]:
        var f3 = 0
        while f3 < ys[1]:
            var ww = 0
            while ww < outW:
                y._data[(n2 * ys[1] + f3) * outW + ww] = y2._data[((n2 * ys[1] + f3) * ys[2] + 0) * outW + ww]
                ww = ww + 1
            f3 = f3 + 1
        n2 = n2 + 1
    return y

# Depthwise conv2d: Input [N,C,H,W], Kernel [C,kH,kW] or [C,1,kH,kW] -> [N,C,out_h,out_w]
fn conv2d_depthwise[T: Copyable & Movable](x: tensor.Tensor[T], k: tensor.Tensor[T], stride: Int = 1, padding: Int = 0) -> tensor.Tensor[T]:
    var xs = x.shape()
    var N = xs[0]; var C = xs[1]; var H = xs[2]; var W = xs[3]
    var ks = k.shape()
    var kH = 0; var kW = 0
    if len(ks) == 3:
        kH = ks[1]; kW = ks[2]
    else:
        kH = ks[2]; kW = ks[3]
    var out_h = _out_dim(H, kH, stride, padding)
    var out_w = _out_dim(W, kW, stride, padding)
    var y = tensor.Tensor[T]([N, C, out_h, out_w], T(0))

    var n = 0
    while n < N:
        var c = 0
        while c < C:
            var oh = 0
            while oh < out_h:
                var ih0 = oh * stride - padding
                var ow = 0
                while ow < out_w:
                    var iw0 = ow * stride - padding
                    var acc = T(0)
                    var kh = 0
                    while kh < kH:
                        var ih = ih0 + kh
                        var kw = 0
                        while kw < kW:
                            var iw = iw0 + kw
                            var xv = _get4(x, n, c, ih, iw)
                            var wv = T(0)
                            if len(ks) == 3:
                                wv = k._data[(c * kH + kh) * kW + kw]
                            else:
                                wv = k._data[((c * 1 + 0) * kH + kh) * kW + kw]
                            acc = acc + xv * wv
                            kw = kw + 1
                        kh = kh + 1
                    y._data[((n * C + c) * out_h + oh) * out_w + ow] = acc
                    ow = ow + 1
                oh = oh + 1
            c = c + 1
        n = n + 1
    return y

# Pointwise conv2d: 1x1 conv, Kernel [F,C,1,1] -> [N,F,H,W]
fn conv2d_pointwise[T: Copyable & Movable](x: tensor.Tensor[T], w: tensor.Tensor[T]) -> tensor.Tensor[T]:
    var xs = x.shape()
    var N = xs[0]; var C = xs[1]; var H = xs[2]; var W = xs[3]
    var ws = w.shape()
    var F = ws[0]
    var y = tensor.Tensor[T]([N, F, H, W], T(0))

    var n = 0
    while n < N:
        var f = 0
        while f < F:
            var h = 0
            while h < H:
                var w0 = 0
                while w0 < W:
                    var acc = T(0)
                    var c = 0
                    while c < C:
                        var xv = _get4(x, n, c, h, w0)
                        var ww = w._data[f * C + c]
                        acc = acc + xv * ww
                        c = c + 1
                    y._data[((n * F + f) * H + h) * W + w0] = acc
                    w0 = w0 + 1
                h = h + 1
            f = f + 1
        n = n + 1
    return y

# Separable conv: depthwise followed by pointwise
fn separable_conv2d[T: Copyable & Movable](x: tensor.Tensor[T], depthwise: tensor.Tensor[T], pointwise: tensor.Tensor[T], stride: Int = 1, padding: Int = 0) -> tensor.Tensor[T]:
    var dw = conv2d_depthwise(x, depthwise, stride, padding)
    return conv2d_pointwise(dw, pointwise)
