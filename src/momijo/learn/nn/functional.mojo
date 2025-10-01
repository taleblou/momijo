# Project:      Momijo
# Module:       learn.nn.functional
# File:         nn/functional.mojo
# Path:         src/momijo/learn/nn/functional.mojo
#
# Description:  Functional (stateless) neural ops for Momijo Learn, inspired by
#               PyTorch's torch.nn.functional. Includes reference CPU implementations
#               for conv2d and max_pool2d over 4D NCHW lists. Later, these can be
#               wired to momijo.tensor fast paths.
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
#   - Types: NCHW4D (alias), Conv2dConfig
#   - Key fns: conv2d, max_pool2d
#   - Semantics: NCHW layout; stride/padding/dilation supported; groups=1
#   - This is a correctness-first reference; optimize/replace with tensor backends later.

from collections.list import List

# --------------------------
# Aliases & small utilities
# --------------------------

# Simple NCHW container with Float64 scalars:
alias NCHW4D = List[List[List[List[Float64]]]]  # [N][C][H][W]
alias OIHW4D = List[List[List[List[Float64]]]]  # [O][I][KH][KW]

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

# --------------------------
# Helpers for List-based NCHW
# --------------------------

fn _zeros_4d(n: Int, c: Int, h: Int, w: Int) -> NCHW4D:
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
                    ws.push_back(0.0)
                    xi = xi + 1
                hs.push_back(ws)
                yi = yi + 1
            cs.push_back(hs)
            ci = ci + 1
        out.push_back(cs)
        ni = ni + 1
    return out

fn _pad2d_nchw(x: NCHW4D, pad_h: Int, pad_w: Int) -> NCHW4D:
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

# --------------------------
# conv2d (reference implementation)
# --------------------------
# Args:
#   x:      [N, C_in, H_in, W_in]  (NCHW)
#   weight: [C_out, C_in, KH, KW]  (OIHW)
#   bias:   [C_out] or None
#   stride: Int or tuple-like (handled by stride_h/stride_w)
#   padding:Int or tuple-like (handled by pad_h/pad_w)
#   dilation:Int or tuple-like (handled by dil_h/dil_w)
# Notes:
#   - groups == 1 only (for simplicity here).
#   - Float64 math for numerical stability in reference path.

fn conv2d(
    x: NCHW4D,
    weight: OIHW4D,
    bias: List[Float64] = List[Float64](),   # empty ⇒ no bias
    stride: Int = 1,
    padding: Int = 0,
    dilation: Int = 1
) -> NCHW4D:
    # Map scalar args to config (square stride/pad/dilation)
    var cfg = Conv2dConfig(stride, stride, padding, padding, dilation, dilation)

    # Shapes
    var N = Int(x.size())
    var C_in = Int(x[0].size())
    var H_in = Int(x[0][0].size())
    var W_in = Int(x[0][0][0].size())

    var C_out = Int(weight.size())
    assert C_out > 0
    assert C_in == Int(weight[0].size())

    var KH = Int(weight[0][0].size())
    var KW = Int(weight[0][0][0].size())

    # Padding
    var x_pad = _pad2d_nchw(x, cfg.pad_h, cfg.pad_w)
    var H_pad = Int(x_pad[0][0].size())
    var W_pad = Int(x_pad[0][0][0].size())

    # Output spatial sizes
    var H_out = Int( ((H_pad - (cfg.dil_h * (KH - 1) + 1)) // cfg.stride_h) + 1 )
    var W_out = Int( ((W_pad - (cfg.dil_w * (KW - 1) + 1)) // cfg.stride_w) + 1 )
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
                    # bias
                    if bias.size() == C_out:
                        acc = acc + bias[co]
                    y[n][co][oy][ox] = acc
                    ox = ox + 1
                oy = oy + 1
            co = co + 1
        n = n + 1

    return y

# --------------------------
# max_pool2d (reference implementation)
# --------------------------
# Args:
#   x: [N, C, H, W]
#   kernel_size: Int (square kernel)
#   stride: Int (defaults to kernel_size)
#   padding: Int (zero-padding before pool)
#   ceil_mode: Bool (if true, uses ceil for output size; here we mimic torch default False)
# Notes:
#   - Dilation is uncommon in max pooling and omitted for simplicity here.

fn max_pool2d(
    x: NCHW4D,
    kernel_size: Int,
    stride: Int = 0,
    padding: Int = 0,
    ceil_mode: Bool = False
) -> NCHW4D:
    assert kernel_size > 0
    var s = (stride if stride > 0 else kernel_size)

    # Optional padding
    var x_pad = _pad2d_nchw(x, padding, padding)

    var N = Int(x_pad.size())
    var C = Int(x_pad[0].size())
    var H = Int(x_pad[0][0].size())
    var W = Int(x_pad[0][0][0].size())

    # Output spatial dims
    var H_out = 0
    var W_out = 0
    if ceil_mode:
        # ceil_div((H - K) , s) + 1
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
                    var maxv = -1.7976931348623157e+308  # ~ -inf for Float64
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
