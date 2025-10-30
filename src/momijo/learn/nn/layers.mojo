# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.nn.layers
# File:         src/momijo/learn/nn/layers.mojo
#
# Description:
#   Core neural network layers for Momijo Learn. Public-facing layer types with
#   stable constructors and forward() APIs. The math is backend-agnostic for now;
#   wire real vectorized kernels later. Parameters use momijo.tensor so they can
#   be summarized and counted properly.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor
from momijo.learn.nn.module import Module
from momijo.learn.utils.summary import Summarizer
from momijo.learn.utils.randomness import RNG, rng_from_seed

# -----------------------------------------------------------------------------
# Small numeric / tensor helpers (pure Mojo; row-major contiguous assumption)
# -----------------------------------------------------------------------------

@always_inline
fn _numel(shape: List[Int]) -> Int:
    var p = 1
    var i = 0
    var n = len(shape)
    while i < n:
        p = p * shape[i]
        i = i + 1
    return p

@always_inline
fn _zeros_f64(shape: List[Int]) -> tensor.Tensor[Float64]:
    return tensor.Tensor[Float64](shape, 0.0)

@always_inline
fn _ones_f64(shape: List[Int]) -> tensor.Tensor[Float64]:
    var t = tensor.Tensor[Float64](shape, 0.0)
    var d = t._data
    var n = _numel(shape)
    var i = 0
    while i < n:
        d[i] = 1.0
        i = i + 1
    return t

fn _copy_tensor(mut dst: tensor.Tensor[Float64], src: tensor.Tensor[Float64]):
    var n = _numel(src.shape())
    var sd = src._data
    var dd = dst._data
    var i = 0
    while i < n:
        dd[i] = sd[i]
        i = i + 1

# Fill tensor with uniform values in [low, high) using RNG.
fn _fill_uniform_(mut t: tensor.Tensor[Float64], low: Float64, high: Float64, rng: Pointer[RNG]):
    var d = t._data
    var n = _numel(t.shape())
    var i = 0
    while i < n:
        d[i] = rng.value.uniform(low, high)
        i = i + 1

# fan_in/fan_out for Linear and Conv2d weights
fn _calc_fans(shape: List[Int]) -> (Int, Int):
    var n = len(shape)
    if n == 2:
        var fan_out = shape[0]
        var fan_in  = shape[1]
        return (fan_in, fan_out)
    if n >= 4:
        var receptive = 1
        var i = 2
        while i < n:
            receptive = receptive * shape[i]
            i = i + 1
        var fan_out = shape[0] * receptive
        var fan_in  = shape[1] * receptive
        return (fan_in, fan_out)
    var prod = 1
    var j = 0
    while j < n:
        prod = prod * shape[j]
        j = j + 1
    return (prod, prod)

# Xavier/Glorot uniform: bound = sqrt(6 / (fan_in + fan_out))
fn _xavier_uniform_(mut t: tensor.Tensor[Float64], rng: Pointer[RNG]):
    var shp = t.shape()
    var fans = _calc_fans(shp)
    var fan_in = fans[0]
    var fan_out = fans[1]
    var denom = fan_in + fan_out
    if denom <= 0:
        return
    var bound = Float64.sqrt(6.0 / Float64(denom))
    _fill_uniform_(t, -bound, bound, rng)

# Kaiming/He uniform for ReLU: bound = sqrt(6 / fan_in)
fn _kaiming_uniform_relu_(mut t: tensor.Tensor[Float64], rng: Pointer[RNG]):
    var shp = t.shape()
    var fans = _calc_fans(shp)
    var fan_in = fans[0]
    if fan_in <= 0:
        return
    var bound = Float64.sqrt(6.0 / Float64(fan_in))
    _fill_uniform_(t, -bound, bound, rng)

# Flatten shape utility: merges dims [start..end] (inclusive) into one.
fn _flatten_shape(shape: List[Int], start_dim: Int, end_dim: Int) -> List[Int]:
    var rank = len(shape)
    var sd = start_dim
    var ed = end_dim
    if sd < 0: sd = rank + sd
    if ed < 0: ed = rank + ed
    if sd < 0: sd = 0
    if ed >= rank: ed = rank - 1
    if sd > ed:
        var tmp = sd; sd = ed; ed = tmp

    var out = List[Int]()
    var i = 0
    while i < sd:
        out.push_back(shape[i])
        i = i + 1

    var prod = 1
    var k = sd
    while k <= ed:
        prod = prod * shape[k]
        k = k + 1
    out.push_back(prod)

    var j = ed + 1
    while j < rank:
        out.push_back(shape[j])
        j = j + 1

    return out

# -----------------------------------------------------------------------------
# Linear
# -----------------------------------------------------------------------------

struct Linear(Module):
    var in_features: Int
    var out_features: Int
    var bias: Bool

    # Parameters: weight [out_features, in_features], bias [out_features] (optional)
    var weight: tensor.Tensor[Float64]
    var bias_t: tensor.Tensor[Float64]   # empty when bias=False

    fn __init__(out self, in_features: Int, out_features: Int, bias: Bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        var wshape = List[Int]()
        wshape.push_back(out_features)
        wshape.push_back(in_features)
        self.weight = _zeros_f64(wshape)

        var bshape = List[Int]()
        if bias:
            bshape.push_back(out_features)
            self.bias_t = _zeros_f64(bshape)
        else:
            bshape.push_back(0)
            self.bias_t = _zeros_f64(bshape)

        self.reset_parameters()

    fn reset_parameters(mut self):
        # Xavier uniform for weights, bias = zeros.
        var rng = rng_from_seed(12345)
        _xavier_uniform_(self.weight, &rng)
        var bn = _numel(self.bias_t.shape())
        var j = 0
        while j < bn:
            self.bias_t._data[j] = 0.0
            j = j + 1

    fn forward(self, x):
        # Implements y = x @ W^T + b for tensor.Tensor[Float64] input.
        # Shapes:
        #   x: [N, in_features]
        #   W: [out_features, in_features]
        #   b: [out_features] (optional)
        # Returns:
        #   y: [N, out_features]
        var shp = x.shape()
        if len(shp) != 2 or shp[1] != self.in_features:
            # Pass-through for unsupported shapes/dtypes.
            return x

        var N = shp[0]
        var D_in = shp[1]
        var D_out = self.out_features

        var yshape = List[Int]()
        yshape.push_back(N)
        yshape.push_back(D_out)
        var y = tensor.Tensor[Float64](yshape, 0.0)

        var xd = x._data
        var yd = y._data
        var wd = self.weight._data
        var bd = self.bias_t._data

        var n = 0
        while n < N:
            var o = 0
            while o < D_out:
                var acc = 0.0
                var d = 0
                while d < D_in:
                    # x[n, d] index in row-major: n*D_in + d
                    var xv = xd[n * D_in + d]
                    # W[o, d] index in row-major: o*D_in + d
                    acc = acc + xv * wd[o * D_in + d]
                    d = d + 1
                if self.bias:
                    acc = acc + bd[o]
                yd[n * D_out + o] = acc
                o = o + 1
            n = n + 1
        return y

    fn __str__(self) -> String:
        var s = String("Linear(")
        s = s + "in_features=" + String(self.in_features)
        s = s + ", out_features=" + String(self.out_features)
        s = s + ", bias=" + (String("True") if self.bias else String("False")) + ")"
        return s

    fn summarize(self, s: Pointer[Summarizer]):
        var in_shape = List[Int]()
        in_shape.push_back(-1)
        in_shape.push_back(self.in_features)
        var out_shape = List[Int]()
        out_shape.push_back(-1)
        out_shape.push_back(self.out_features)
        var ps = List[tensor.Tensor[Float64]]()
        ps.push_back(self.weight)
        var bs = List[tensor.Tensor[Float64]]()
        if self.bias:
            bs.push_back(self.bias_t)
        s.value.add_params_f64("Linear", in_shape, out_shape, ps, bs, True)

# -----------------------------------------------------------------------------
# Conv2d (NCHW, groups supported, naive loops)
# -----------------------------------------------------------------------------

@always_inline
fn _idx_nchw(n: Int, c: Int, h: Int, w: Int, C: Int, H: Int, W: Int) -> Int:
    # Row-major contiguous: ((((n*C)+c)*H)+h)*W + w
    return (((n * C) + c) * H + h) * W + w

struct Conv2d(Module):
    var in_channels: Int
    var out_channels: Int
    var kernel_size: Int
    var stride: Int
    var padding: Int
    var dilation: Int
    var groups: Int
    var bias: Bool

    # Parameters:
    # weight: [out_channels, in_channels/groups, k, k]
    # bias:   [out_channels] (optional)
    var weight: tensor.Tensor[Float64]
    var bias_t: tensor.Tensor[Float64]

    fn __init__(
        out self,
        in_channels: Int,
        out_channels: Int,
        kernel_size: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = True
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        var icg = in_channels // groups

        var wshape = List[Int]()
        wshape.push_back(out_channels)
        wshape.push_back(icg)
        wshape.push_back(kernel_size)
        wshape.push_back(kernel_size)
        self.weight = _zeros_f64(wshape)

        var bshape = List[Int]()
        if bias:
            bshape.push_back(out_channels)
            self.bias_t = _zeros_f64(bshape)
        else:
            bshape.push_back(0)
            self.bias_t = _zeros_f64(bshape)

        self.reset_parameters()

    fn reset_parameters(mut self):
        # Kaiming uniform (ReLU) for weights, bias = zeros.
        var rng = rng_from_seed(54321)
        _kaiming_uniform_relu_(self.weight, &rng)
        var bn = _numel(self.bias_t.shape())
        var j = 0
        while j < bn:
            self.bias_t._data[j] = 0.0
            j = j + 1

    fn forward(self, x):
        # Naive NCHW convolution with groups, stride, padding, dilation.
        # Only implemented for tensor.Tensor[Float64]; other types pass-through.
        var shp = x.shape()
        if len(shp) != 4 or shp[1] != self.in_channels:
            return x

        var N = shp[0]; var C = shp[1]; var H = shp[2]; var W = shp[3]
        var K  = self.kernel_size
        var S  = self.stride
        var P  = self.padding
        var D  = self.dilation
        var G  = self.groups
        var OC = self.out_channels
        var ICG = C // G                 # in-channels per group
        var OCG = OC // G                # out-channels per group

        # Output spatial dims (floor division)
        var OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
        var OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
        if OH <= 0 or OW <= 0:
            return x

        var yshape = List[Int]()
        yshape.push_back(N); yshape.push_back(OC); yshape.push_back(OH); yshape.push_back(OW)
        var y = tensor.Tensor[Float64](yshape, 0.0)

        var xd = x._data
        var yd = y._data
        var wd = self.weight._data
        var bd = self.bias_t._data

        var n = 0
        while n < N:
            var oc = 0
            while oc < OC:
                var g = oc // OCG
                var ic_base = g * ICG
                var oh = 0
                while oh < OH:
                    var ow = 0
                    while ow < OW:
                        var acc = 0.0
                        var icg = 0
                        while icg < ICG:
                            var ic = ic_base + icg
                            var kh = 0
                            while kh < K:
                                var ih = oh * S - P + kh * D
                                var kw = 0
                                while kw < K:
                                    var iw = ow * S - P + kw * D
                                    if ih >= 0 and ih < H and iw >= 0 and iw < W:
                                        var x_idx = _idx_nchw(n, ic, ih, iw, C, H, W)
                                        # weight index: [oc, icg, kh, kw]
                                        var w_idx = (((oc * ICG) + icg) * K + kh) * K + kw
                                        acc = acc + xd[x_idx] * wd[w_idx]
                                    kw = kw + 1
                                kh = kh + 1
                            icg = icg + 1
                        if self.bias:
                            acc = acc + bd[oc]
                        yd[_idx_nchw(n, oc, oh, ow, OC, OH, OW)] = acc
                        ow = ow + 1
                    oh = oh + 1
                oc = oc + 1
            n = n + 1
        return y

    fn __str__(self) -> String:
        var s = String("Conv2d(")
        s = s + "in_channels=" + String(self.in_channels)
        s = s + ", out_channels=" + String(self.out_channels)
        s = s + ", kernel_size=" + String(self.kernel_size)
        s = s + ", stride=" + String(self.stride)
        s = s + ", padding=" + String(self.padding)
        s = s + ", dilation=" + String(self.dilation)
        s = s + ", groups=" + String(self.groups)
        s = s + ", bias=" + (String("True") if self.bias else String("False")) + ")"
        return s

    fn summarize(self, s: Pointer[Summarizer]):
        var in_shape = List[Int]()
        in_shape.push_back(-1); in_shape.push_back(self.in_channels); in_shape.push_back(-1); in_shape.push_back(-1)
        var out_shape = List[Int]()
        out_shape.push_back(-1); out_shape.push_back(self.out_channels); out_shape.push_back(-1); out_shape.push_back(-1)
        var ps = List[tensor.Tensor[Float64]]()
        ps.push_back(self.weight)
        var bs = List[tensor.Tensor[Float64]]()
        if self.bias:
            bs.push_back(self.bias_t)
        s.value.add_params_f64("Conv2d", in_shape, out_shape, ps, bs, True)

# -----------------------------------------------------------------------------
# BatchNorm2d (NCHW; training/eval behavior with running stats)
# -----------------------------------------------------------------------------

struct BatchNorm2d(Module):
    var num_features: Int
    var eps: Float64
    var momentum: Float64
    var affine: Bool
    var track_running_stats: Bool

    var weight: tensor.Tensor[Float64]
    var bias_t: tensor.Tensor[Float64]
    var running_mean: tensor.Tensor[Float64]
    var running_var: tensor.Tensor[Float64]

    fn __init__(
        out self,
        num_features: Int,
        eps: Float64 = 1e-5,
        momentum: Float64 = 0.1,
        affine: Bool = True,
        track_running_stats: Bool = True
    ):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        var cshape = List[Int](); cshape.push_back(num_features)

        if affine:
            self.weight = _ones_f64(cshape)
            self.bias_t = _zeros_f64(cshape)
        else:
            var z = List[Int](); z.push_back(0)
            self.weight = _zeros_f64(z)
            self.bias_t = _zeros_f64(z)

        if track_running_stats:
            self.running_mean = _zeros_f64(cshape)
            self.running_var  = _ones_f64(cshape)
        else:
            var z2 = List[Int](); z2.push_back(0)
            self.running_mean = _zeros_f64(z2)
            self.running_var  = _zeros_f64(z2)

        self.reset_parameters()

    fn reset_parameters(mut self):
        if self.affine:
            var n = _numel(self.weight.shape())
            var i = 0
            while i < n:
                self.weight._data[i] = 1.0
                i = i + 1
            var m = _numel(self.bias_t.shape())
            var j = 0
            while j < m:
                self.bias_t._data[j] = 0.0
                j = j + 1
        if self.track_running_stats:
            var c = _numel(self.running_mean.shape())
            var k = 0
            while k < c:
                self.running_mean._data[k] = 0.0
                self.running_var._data[k] = 1.0
                k = k + 1

    fn forward(self, x, training: Bool = True):
        # Only implements tensor.Tensor[Float64] with shape [N, C, H, W].
        var shp = x.shape()
        if len(shp) != 4 or shp[1] != self.num_features:
            return x

        var N = shp[0]; var C = shp[1]; var H = shp[2]; var W = shp[3]
        var count = Float64(N * H * W)
        if count <= 0.0:
            return x

        var xd = x._data
        var gamma = self.weight._data
        var beta  = self.bias_t._data
        var rm    = self.running_mean._data
        var rv    = self.running_var._data

        # Allocate output
        var yshape = List[Int](); yshape.push_back(N); yshape.push_back(C); yshape.push_back(H); yshape.push_back(W)
        var y = tensor.Tensor[Float64](yshape, 0.0)
        var yd = y._data

        var c = 0
        while c < C:
            # Compute mean/var for channel c (over N,H,W) in training; else use running stats.
            var mean_c = 0.0
            var var_c  = 1.0

            if training:
                var sum = 0.0
                var n = 0
                while n < N:
                    var h = 0
                    while h < H:
                        var w = 0
                        while w < W:
                            sum = sum + xd[_idx_nchw(n, c, h, w, C, H, W)]
                            w = w + 1
                        h = h + 1
                    n = n + 1
                mean_c = sum / count

                var sq = 0.0
                n = 0
                while n < N:
                    var h2 = 0
                    while h2 < H:
                        var w2 = 0
                        while w2 < W:
                            var v = xd[_idx_nchw(n, c, h2, w2, C, H, W)] - mean_c
                            sq = sq + v * v
                            w2 = w2 + 1
                        h2 = h2 + 1
                    n = n + 1
                var_c = sq / count

                if self.track_running_stats:
                    # Update running stats: new = (1-m)*old + m*batch
                    var m = self.momentum
                    rm[c] = (1.0 - m) * rm[c] + m * mean_c
                    rv[c] = (1.0 - m) * rv[c] + m * var_c
            else:
                if self.track_running_stats:
                    mean_c = rm[c]
                    var_c  = rv[c]
                else:
                    mean_c = 0.0
                    var_c  = 1.0

            var inv = 1.0 / Float64.sqrt(var_c + self.eps)
            var g = 1.0
            var b = 0.0
            if self.affine:
                g = gamma[c]
                b = beta[c]

            var n3 = 0
            while n3 < N:
                var h3 = 0
                while h3 < H:
                    var w3 = 0
                    while w3 < W:
                        var xval = xd[_idx_nchw(n3, c, h3, w3, C, H, W)]
                        var z = (xval - mean_c) * inv
                        var outv = z * g + b
                        yd[_idx_nchw(n3, c, h3, w3, C, H, W)] = outv
                        w3 = w3 + 1
                    h3 = h3 + 1
                n3 = n3 + 1
            c = c + 1
        return y

    fn __str__(self) -> String:
        var s = String("BatchNorm2d(")
        s = s + "num_features=" + String(self.num_features)
        s = s + ", eps=" + String(self.eps)
        s = s + ", momentum=" + String(self.momentum)
        s = s + ", affine=" + (String("True") if self.affine else String("False"))
        s = s + ", track_running_stats=" + (String("True") if self.track_running_stats else String("False")) + ")"
        return s

    fn summarize(self, s: Pointer[Summarizer]):
        var in_shape = List[Int]()
        in_shape.push_back(-1); in_shape.push_back(self.num_features); in_shape.push_back(-1); in_shape.push_back(-1)
        var out_shape = List[Int]()
        out_shape.push_back(-1); out_shape.push_back(self.num_features); out_shape.push_back(-1); out_shape.push_back(-1)

        var params = List[tensor.Tensor[Float64]]()
        var buffers = List[tensor.Tensor[Float64]]()

        if self.affine:
            params.push_back(self.weight)
            params.push_back(self.bias_t)
        if self.track_running_stats:
            buffers.push_back(self.running_mean)
            buffers.push_back(self.running_var)

        s.value.add_params_f64("BatchNorm2d", in_shape, out_shape, params, buffers, True)

# -----------------------------------------------------------------------------
# Dropout
# -----------------------------------------------------------------------------

struct Dropout(Module):
    var p: Float64
    var inplace: Bool
    var seed: Int

    fn __init__(out self, p: Float64 = 0.5, inplace: Bool = False, seed: Int = 1234567):
        var pp = p
        if pp < 0.0: pp = 0.0
        if pp > 1.0: pp = 1.0
        self.p = pp
        self.inplace = inplace
        self.seed = seed

    fn forward(self, x, training: Bool = True):
        if not training:
            return x

        var keep_prob = 1.0 - self.p
        if keep_prob <= 0.0:
            return x

        var shp = x.shape()
        var n = _numel(shp)
        var rng = rng_from_seed(self.seed)

        if self.inplace:
            var xd = x._data
            var i = 0
            while i < n:
                var r = rng.uniform(0.0, 1.0)
                if r < keep_prob:
                    xd[i] = xd[i] / keep_prob
                else:
                    xd[i] = 0.0
                i = i + 1
            return x
        else:
            var out = tensor.Tensor[Float64](shp, 0.0)
            var xd2 = x._data
            var yd = out._data
            var j = 0
            while j < n:
                var r2 = rng.uniform(0.0, 1.0)
                if r2 < keep_prob:
                    yd[j] = xd2[j] / keep_prob
                else:
                    yd[j] = 0.0
                j = j + 1
            return out

    fn __str__(self) -> String:
        var s = String("Dropout(")
        s = s + "p=" + String(self.p)
        s = s + ", inplace=" + (String("True") if self.inplace else String("False"))
        s = s + ", seed=" + String(self.seed) + ")"
        return s

    fn summarize(self, s: Pointer[Summarizer]):
        var in_shape = List[Int](); in_shape.push_back(-1)
        var out_shape = List[Int](); out_shape.push_back(-1)
        s.value.add_leaf("Dropout", in_shape, out_shape)

# -----------------------------------------------------------------------------
# Flatten
# -----------------------------------------------------------------------------

struct Flatten(Module):
    var start_dim: Int
    var end_dim: Int

    fn __init__(out self, start_dim: Int = 1, end_dim: Int = -1):
        self.start_dim = start_dim
        self.end_dim = end_dim

    fn forward(self, x):
        # Produces a new contiguous tensor with flattened dims [start_dim..end_dim].
        var shp = x.shape()
        var new_shape = _flatten_shape(shp, self.start_dim, self.end_dim)
        var out = tensor.Tensor[Float64](new_shape, 0.0)
        # Row-major element order is preserved; copy 1:1 because reshape is a view.
        # Since we do not have view support here, we copy.
        _copy_tensor(out, x)
        return out

    fn __str__(self) -> String:
        var s = String("Flatten(")
        s = s + "start_dim=" + String(self.start_dim)
        s = s + ", end_dim=" + String(self.end_dim) + ")"
        return s

    fn summarize(self, s: Pointer[Summarizer]):
        var in_shape = List[Int](); in_shape.push_back(-1)
        var out_shape = List[Int](); out_shape.push_back(-1)
        s.value.add_leaf("Flatten", in_shape, out_shape)
