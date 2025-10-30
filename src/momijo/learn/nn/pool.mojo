# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.nn.pooling
# File:         src/momijo/learn/nn/pooling.mojo
#
# Description:
#   2D pooling utilities for 4D tensors in NCHW layout:
#     - maxpool2d / avgpool2d with configurable kernel, stride, and padding
#     - global_avg_pool2d / global_max_pool2d over HxW
#     - adaptive_avg_pool2d to (out_h, out_w)
#     - max_unpool2d (with indices) and avg_unpool2d
#   Notes:
#     - Generic over T (element type). Requires T to support +, /, comparison.
#     - Backend-agnostic; uses Momijo tensor facade: `from momijo.tensor import tensor`.
#     - Indices for unpool are flat positions over the target y-shape (NCHW).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor   # facade import

# --------------------------------- utils ---------------------------------

@always_inline
fn _prod(shape: List[Int]) -> Int:
    var p = 1
    var i = 0
    var r = len(shape)
    while i < r:
        p = p * shape[i]
        i += 1
    return p

@always_inline
fn _out_dim(in_size: Int, k: Int, stride: Int, pad: Int) -> Int:
    var s = stride
    if s <= 0:
        s = 1
    var o = in_size + 2 * pad - k
    if o < 0:
        o = 0
    return (o // s) + 1

@always_inline
fn _check_4d(name: String, x_shape: List[Int]) -> None:
    assert len(x_shape) == 4, name + ": expected 4D NCHW tensor"

@always_inline
fn _get4[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T], n: Int, c: Int, h: Int, w: Int) -> T:
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
fn _set4[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T], n: Int, c: Int, h: Int, w: Int, v: T) -> None:
    var sh = x.shape()
    var idx = ((n * sh[1] + c) * sh[2] + h) * sh[3] + w
    x._data[idx] = v

# -------------------------------- core pooling ----------------------------

# mode: 0 = max, 1 = avg
fn pool2d_core[T: ImplicitlyCopyable & Copyable & Movable](
    x: tensor.Tensor[T],
    k: Int,
    stride: Int = 0,
    padding: Int = 0,
    mode: Int = 0
) -> tensor.Tensor[T]:
    var xs = x.shape()
    _check_4d("pool2d_core", xs)
    var N = xs[0]; var C = xs[1]; var H = xs[2]; var W = xs[3]
    var st = stride
    if st <= 0:
        st = k
    var out_h = _out_dim(H, k, st, padding)
    var out_w = _out_dim(W, k, st, padding)
    var y = tensor.Tensor[T]([N, C, out_h, out_w], T(0))

    var n = 0
    while n < N:
        var c = 0
        while c < C:
            var oh = 0
            while oh < out_h:
                var ih0 = oh * st - padding
                var ow = 0
                while ow < out_w:
                    var iw0 = ow * st - padding
                    if mode == 0:
                        # max
                        var best = _get4(x, n, c, ih0, iw0)
                        var kh = 0
                        while kh < k:
                            var kw = 0
                            while kw < k:
                                var v = _get4(x, n, c, ih0 + kh, iw0 + kw)
                                if v > best:
                                    best = v
                                kw += 1
                            kh += 1
                        _set4(y, n, c, oh, ow, best)
                    else:
                        # avg
                        var acc = T(0)
                        var cnt = 0
                        var kh2 = 0
                        while kh2 < k:
                            var kw2 = 0
                            while kw2 < k:
                                var ih = ih0 + kh2
                                var iw = iw0 + kw2
                                if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                    acc = acc + _get4(x, n, c, ih, iw)
                                    cnt += 1
                                kw2 += 1
                            kh2 += 1
                        if cnt == 0:
                            _set4(y, n, c, oh, ow, T(0))
                        else:
                            _set4(y, n, c, oh, ow, acc / T(cnt))
                    ow += 1
                oh += 1
            c += 1
        n += 1
    return y

# Also return argmax indices (flat in input NCHW space) for unpooling.
fn maxpool2d_with_indices[T: ImplicitlyCopyable & Copyable & Movable](
    x: tensor.Tensor[T],
    k: Int,
    stride: Int = 0,
    padding: Int = 0
) -> (tensor.Tensor[T], tensor.Tensor[Int]):
    var xs = x.shape()
    _check_4d("maxpool2d_with_indices", xs)
    var N = xs[0]; var C = xs[1]; var H = xs[2]; var W = xs[3]
    var st = stride
    if st <= 0:
        st = k
    var out_h = _out_dim(H, k, st, padding)
    var out_w = _out_dim(W, k, st, padding)

    var y = tensor.Tensor[T]([N, C, out_h, out_w], T(0))
    var idxs = tensor.Tensor[Int]([N, C, out_h, out_w],  Int(-1))

    var n = 0
    while n < N:
        var c = 0
        while c < C:
            var oh = 0
            while oh < out_h:
                var ih0 = oh * st - padding
                var ow = 0
                while ow < out_w:
                    var iw0 = ow * st - padding
                    var best = _get4(x, n, c, ih0, iw0)
                    var best_pos = -1
                    var kh = 0
                    while kh < k:
                        var kw = 0
                        while kw < k:
                            var ih = ih0 + kh
                            var iw = iw0 + kw
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                var v = _get4(x, n, c, ih, iw)
                                if v > best or best_pos == -1:
                                    best = v
                                    best_pos = ((n * C + c) * H + ih) * W + iw
                            kw += 1
                        kh += 1
                    _set4(y, n, c, oh, ow, best)
                    idxs._data[((n * C + c) * out_h + oh) * out_w + ow] = best_pos
                    ow += 1
                oh += 1
            c += 1
        n += 1
    return (y, idxs)

# -------------------------------- public API ------------------------------

fn maxpool2d[T: ImplicitlyCopyable & Copyable & Movable](
    x: tensor.Tensor[T], k: Int, stride: Int = 0, padding: Int = 0
) -> tensor.Tensor[T]:
    return pool2d_core(x, k, stride, padding, 0)

fn avgpool2d[T: ImplicitlyCopyable & Copyable & Movable](
    x: tensor.Tensor[T], k: Int, stride: Int = 0, padding: Int = 0
) -> tensor.Tensor[T]:
    return pool2d_core(x, k, stride, padding, 1)

# Global pools over HxW
fn global_avg_pool2d[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T]) -> tensor.Tensor[T]:
    var xs = x.shape()
    _check_4d("global_avg_pool2d", xs)
    var N = xs[0]; var C = xs[1]; var H = xs[2]; var W = xs[3]
    var y = tensor.Tensor[T]([N, C, 1, 1], T(0))
    var n = 0
    while n < N:
        var c = 0
        while c < C:
            var acc = T(0)
            var i = 0
            var total = H * W
            while i < total:
                var h = i // W
                var w = i - h * W
                acc = acc + _get4(x, n, c, h, w)
                i += 1
            if total > 0:
                _set4(y, n, c, 0, 0, acc / T(total))
            else:
                _set4(y, n, c, 0, 0, T(0))
            c += 1
        n += 1
    return y

fn global_max_pool2d[T: ImplicitlyCopyable & Copyable & Movable](x: tensor.Tensor[T]) -> tensor.Tensor[T]:
    var xs = x.shape()
    _check_4d("global_max_pool2d", xs)
    var N = xs[0]; var C = xs[1]; var H = xs[2]; var W = xs[3]
    var y = tensor.Tensor[T]([N, C, 1, 1], T(0))
    var n = 0
    while n < N:
        var c = 0
        while c < C:
            var best = _get4(x, n, c, 0, 0)
            var i = 1
            var total = H * W
            while i < total:
                var h = i // W
                var w = i - h * W
                var v = _get4(x, n, c, h, w)
                if v > best:
                    best = v
                i += 1
            _set4(y, n, c, 0, 0, best)
            c += 1
        n += 1
    return y

# Adaptive average pooling to (out_h, out_w)
fn adaptive_avg_pool2d[T: ImplicitlyCopyable & Copyable & Movable](
    x: tensor.Tensor[T], out_h: Int, out_w: Int
) -> tensor.Tensor[T]:
    var xs = x.shape()
    _check_4d("adaptive_avg_pool2d", xs)
    var N = xs[0]; var C = xs[1]; var H = xs[2]; var W = xs[3]
    var oh = out_h; var ow = out_w
    if oh <= 0: oh = 1
    if ow <= 0: ow = 1
    var y = tensor.Tensor[T]([N, C, oh, ow], T(0))

    var n = 0
    while n < N:
        var c = 0
        while c < C:
            var i = 0
            while i < oh:
                var h0 = (i * H) // oh
                var h1 = ((i + 1) * H + oh - 1) // oh
                if h1 > H: h1 = H
                var j = 0
                while j < ow:
                    var w0 = (j * W) // ow
                    var w1 = ((j + 1) * W + ow - 1) // ow
                    if w1 > W: w1 = W
                    var acc = T(0)
                    var cnt = 0
                    var h = h0
                    while h < h1:
                        var wv = w0
                        while wv < w1:
                            acc = acc + _get4(x, n, c, h, wv)
                            cnt += 1
                            wv += 1
                        h += 1
                    if cnt > 0:
                        _set4(y, n, c, i, j, acc / T(cnt))
                    else:
                        _set4(y, n, c, i, j, T(0))
                    j += 1
                i += 1
            c += 1
        n += 1
    return y

# ------------------------------- unpooling -------------------------------

# Max unpool with indices: y_shape is [N,C,H,W] of target.
# Indices are flat offsets over NCHW of the target space (as produced by maxpool2d_with_indices).
fn max_unpool2d[T: ImplicitlyCopyable & Copyable & Movable](
    x: tensor.Tensor[T],
    indices: tensor.Tensor[Int],
    y_shape: List[Int]
) -> tensor.Tensor[T]:
    var N = y_shape[0]; var C = y_shape[1]; var H = y_shape[2]; var W = y_shape[3]
    var y = tensor.Tensor[T]([N, C, H, W], T(0))

    var xs = x.shape()
    _check_4d("max_unpool2d:x", xs)
    var ishp = indices.shape()
    _check_4d("max_unpool2d:indices", ishp)

    var n = 0
    while n < xs[0]:
        var c = 0
        while c < xs[1]:
            var h = 0
            while h < xs[2]:
                var w0 = 0
                while w0 < xs[3]:
                    var idx = ((n * xs[1] + c) * xs[2] + h) * xs[3] + w0
                    var pos = indices._data[idx]
                    if pos >= 0:
                        var nn = pos // (C * H * W)
                        var rem = pos - nn * (C * H * W)
                        var cc = rem // (H * W)
                        rem = rem - cc * (H * W)
                        var hh = rem // W
                        var ww = rem - hh * W
                        _set4(y, nn, cc, hh, ww, x._data[idx])
                    w0 += 1
                h += 1
            c += 1
        n += 1
    return y

# Avg unpool: uniform scatter of pooled values back to the window (requires k/stride/pad)
fn avg_unpool2d[T: ImplicitlyCopyable & Copyable & Movable](
    x: tensor.Tensor[T],
    k: Int,
    stride: Int = 0,
    padding: Int = 0,
    y_shape: List[Int]
) -> tensor.Tensor[T]:
    var N = y_shape[0]; var C = y_shape[1]; var H = y_shape[2]; var W = y_shape[3]
    var y = tensor.Tensor[T]([N, C, H, W], T(0))
    var st = stride
    if st <= 0:
        st = k

    var xs = x.shape()
    _check_4d("avg_unpool2d:x", xs)
    var n = 0
    while n < xs[0]:
        var c = 0
        while c < xs[1]:
            var oh = 0
            while oh < xs[2]:
                var ih0 = oh * st - padding
                var ow = 0
                while ow < xs[3]:
                    var iw0 = ow * st - padding
                    var cnt = 0
                    var kh = 0
                    while kh < k:
                        var kw = 0
                        while kw < k:
                            var ih = ih0 + kh
                            var iw = iw0 + kw
                            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                cnt += 1
                            kw += 1
                        kh += 1
                    var share = T(0)
                    if cnt > 0:
                        share = x._data[((n * xs[1] + c) * xs[2] + oh) * xs[3] + ow] / T(cnt)
                    var kh2 = 0
                    while kh2 < k:
                        var kw2 = 0
                        while kw2 < k:
                            var ih2 = ih0 + kh2
                            var iw2 = iw0 + kw2
                            if ih2 >= 0 and ih2 < H and iw2 >= 0 and iw2 < W:
                                var cur = _get4(y, n, c, ih2, iw2)
                                _set4(y, n, c, ih2, iw2, cur + share)
                            kw2 += 1
                        kh2 += 1
                    ow += 1
                oh += 1
            c += 1
        n += 1
    return y
