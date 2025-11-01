# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.audio.signal
# File:         src/momijo/learn/audio/signal.mojo
#
# Description:
#   Minimal audio utilities: hann_window, frame_signal, dft_power, spectrogram_power.
#   Float64-based, no external deps. Compatible with Momijo style.

from collections.list import List
from momijo.tensor import tensor

fn _wrap_pi(x: Float64) -> Float64:
    var PI  = 3.141592653589793
    var TAU = 6.283185307179586
    var y = x - TAU * Float64(Int(x / TAU))
    if y >  PI:  y = y - TAU
    if y < -PI:  y = y + TAU
    return y

fn _cos(x: Float64) -> Float64:
    var xx = _wrap_pi(x)
    var z = xx * xx
    var c = 1.0
    c = c - (z * 0.5)
    c = c + (z * z) * (1.0 / 24.0)
    c = c - (z * z * z) * (1.0 / 720.0)
    return c

fn _sin(x: Float64) -> Float64:
    var PI_2 = 1.5707963267948966
    return _cos(x - PI_2)

fn hann_window(win_len: Int) -> tensor.Tensor[Float64]:
    var wshape = List[Int](); wshape.append(win_len)
    var w = tensor.zeros(wshape)
    if win_len <= 1:
        if win_len == 1: w._data[0] = 1.0
        return w.copy()
    var n = 0
    var PI = 3.141592653589793
    while n < win_len:
        var a = 2.0 * PI * Float64(n) / Float64(win_len - 1)
        w._data[n] = 0.5 - 0.5 * _cos(a)
        n = n + 1
    return w.copy()

fn frame_signal(x: tensor.Tensor[Float64], frame_len: Int, hop: Int) -> List[tensor.Tensor[Float64]]:
    var L = x.shape()[0]
    var out = List[tensor.Tensor[Float64]]()
    if L < frame_len: 
        return out.copy()
    var n_frames = (L - frame_len) // hop + 1
    var i = 0
    while i < n_frames:
        var start = i * hop
        var fshape = List[Int](); fshape.append(frame_len)
        var f = tensor.zeros(fshape)
        var j = 0
        while j < frame_len:
            f._data[j] = x._data[start + j]
            j = j + 1
        out.append(f.copy())
        i = i + 1
    return out.copy()

fn dft_power(frame: tensor.Tensor[Float64], n_fft: Int) -> tensor.Tensor[Float64]:
    var L = frame.shape()[0]
    var half = n_fft // 2
    var out_shape = List[Int](); out_shape.append(half + 1)
    var p = tensor.zeros(out_shape)

    var k = 0
    var PI = 3.141592653589793
    while k <= half:
        var re = 0.0
        var im = 0.0
        var n = 0
        while n < L:
            var ang = 2.0 * PI * Float64(k * n) / Float64(n_fft)
            var cn = _cos(ang)
            var sn = _sin(ang)
            var xn = frame._data[n]
            re = re + xn * cn
            im = im - xn * sn
            n = n + 1
        p._data[k] = re * re + im * im
        k = k + 1
    return p.copy()

fn spectrogram_power(
    wav: tensor.Tensor[Float64],
    n_fft: Int,
    hop_length: Int,
    win_len: Int
) -> tensor.Tensor[Float64]:
    var frames = frame_signal(wav, win_len, hop_length)
    var n_frames = len(frames)
    var n_bins = n_fft // 2 + 1

    var w = hann_window(win_len)

    var osh = List[Int](); osh.append(n_bins); osh.append(n_frames)
    var spec = tensor.zeros(osh)

    var t = 0
    while t < n_frames:
        var f = frames[t].copy()
        var i = 0
        while i < win_len:
            f._data[i] = f._data[i] * w._data[i]
            i = i + 1

        var p = dft_power(f, n_fft)
        var b = 0
        while b < n_bins:
            spec._data[b * n_frames + t] = p._data[b]
            b = b + 1
        t = t + 1
    return spec.copy()
