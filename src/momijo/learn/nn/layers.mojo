# MIT License
# SPDX-License-Identifier: MIT
# Project: momijo
# File: src/momijo/learn/nn/layers.mojo
# Description: Core dense and activation layers for Momijo Learn.

from momijo.tensor import tensor
from collections.list import List 
from momijo.learn.nn.module import Module
from momijo.learn.utils.summary import Summarizer
from momijo.learn.utils.randomness import RNG, rng_from_seed
 
from collections.list import List
from momijo.tensor import tensor

from momijo.learn.api.functional import (
    quantize_symmetric_int8,
    matmul_i8_i8_to_i32,
    dequantize_i32_to_f32_add_bias
)
# ---------- math helpers (approx) ----------
@always_inline
fn _sqrt_pos_f64(x: Float64) -> Float64:
    if x <= 0.0: return 0.0
    var y = x
    var i = 0
    while i < 12:            # Newton-Raphson
        y = 0.5 * (y + x / y)
        i = i + 1
    return y

@always_inline
fn _exp_f64(x: Float64) -> Float64:

    var term = 1.0
    var sum = 1.0
    var k = 1
    while k <= 12:
        term = term * x / Float64(k)
        sum = sum + term
        k = k + 1
    return sum


@always_inline
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
# ---------- small tensor helpers ----------
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
    var n = _numel(shape)
    var i = 0
    while i < n:
        t._data[i] = 1.0    
        i = i + 1
    return t.copy()

# elementwise max(x, a)
fn _maximum_scalar(x: tensor.Tensor[Float64], a: Float64) -> tensor.Tensor[Float64]:
    var y = tensor.Tensor[Float64](x.shape(), 0.0)
    var n = _numel(x.shape())
    var i = 0
    while i < n:
        var v = x._data[i]
        y._data[i] = v if v > a else a
        i = i + 1
    return y.copy()
# elementwise max(x, a)
fn _maximum_scalar(x: tensor.GradTensor, a: Float64) -> tensor.GradTensor:
    var y = tensor.Tensor[Float64](x.shape(), 0.0)
    var n = _numel(x.shape())
    var i = 0
    while i < n:
        var v = x._data[i]
        y._data[i] = v if v > a else a
        i = i + 1
    return y.copy()

# ---------- activations ----------
struct ReLU(Copyable, Movable):
    fn __init__(out self): pass
    fn __copyinit__(out self, other: Self): pass
    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        return _maximum_scalar(x, 0.0)
    fn forward(self, x: tensor.GradTensor) -> tensor.GradTensor:
        return _maximum_scalar(x, 0.0)

struct LeakyReLU(Copyable, Movable):
    var slope: Float64
    fn __init__(out self, slope: Float64 = 0.01): self.slope = slope
    fn __copyinit__(out self, other: Self): self.slope = other.slope
    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var pos = _maximum_scalar(x, 0.0)
        var neg = x - pos
        return pos + (neg * self.slope)

    fn forward(self, x: tensor.GradTensor) -> tensor.GradTensor:
        var pos = _maximum_scalar(x, 0.0)
        var neg = x - pos
        return pos + (neg * self.slope)

struct Sigmoid(Copyable, Movable):
    fn __init__(out self): pass
    fn __copyinit__(out self, other: Self): pass
    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:

        var y = tensor.Tensor[Float64](x.shape(), 0.0)
        var n = _numel(x.shape())
        var i = 0
        while i < n:
            var v = x._data[i]
            var e = _exp_f64(-v)
            y._data[i] = 1.0 / (1.0 + e)
            i = i + 1
        return y.copy()
    fn forward(self, x: tensor.GradTensor) -> tensor.GradTensor:

        var y = tensor.GradTensor(x.shape(), 0.0)
        var n = _numel(x.shape())
        var i = 0
        while i < n:
            var v = x._data[i]
            var e = _exp_f64(-v)
            y._data[i] = 1.0 / (1.0 + e)
            i = i + 1
        return y.copy()

struct Tanh(Copyable, Movable):
    fn __init__(out self): pass
    fn __copyinit__(out self, other: Self): pass
    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        # tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})
        var y = tensor.Tensor[Float64](x.shape(), 0.0)
        var n = _numel(x.shape())
        var i = 0
        while i < n:
            var v = x._data[i]
            var ex = _exp_f64(v)
            var em = _exp_f64(-v)
            y._data[i] = (ex - em) / (ex + em)
            i = i + 1
        return y.copy()
    fn forward(self, x: tensor.GradTensor) -> tensor.GradTensor:
        # tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})
        var y = tensor.GradTensor(x.shape(), 0.0)
        var n = _numel(x.shape())
        var i = 0
        while i < n:
            var v = x._data[i]
            var ex = _exp_f64(v)
            var em = _exp_f64(-v)
            y._data[i] = (ex - em) / (ex + em)
            i = i + 1
        return y.copy()


struct BatchNorm1d(Copyable, Movable):
    var num_features: Int
    var gamma: tensor.Tensor[Float64]
    var beta: tensor.Tensor[Float64]
    var running_mean: tensor.Tensor[Float64]
    var running_var: tensor.Tensor[Float64]
    var eps: Float64
    var momentum: Float64

    fn __init__(out self, num_features: Int, eps: Float64 = 1e-5, momentum: Float64 = 0.1):
        self.num_features = num_features
        self.gamma = _ones_f64([num_features])
        self.beta = _zeros_f64([num_features])
        self.running_mean = _zeros_f64([num_features])
        self.running_var = _ones_f64([num_features])
        self.eps = eps
        self.momentum = momentum

    fn __copyinit__(out self, other: Self):
        self.num_features = other.num_features
        self.gamma = other.gamma.copy()
        self.beta = other.beta.copy()
        self.running_mean = other.running_mean.copy()
        self.running_var = other.running_var.copy()
        self.eps = other.eps
        self.momentum = other.momentum

    fn _col_sum(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        # Sum over rows ⇒ [1, C]
        var shp = x.shape()          # [N, C]
        var N = shp[0]; var C = shp[1]
        var out = tensor.Tensor[Float64]([1, C], 0.0)
        var r = 0
        while r < N:
            var c = 0
            while c < C:
                out._data[c] = out._data[c] + x._data[r * C + c]
                c = c + 1
            r = r + 1
        return out.copy()

    fn forward(mut self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        # x: [N, C]
        var shp = x.shape()
        if len(shp) != 2 or shp[1] != self.num_features:
            return x.copy()

        var N = shp[0]; var C = shp[1]
        if N <= 0: return x.copy()

        var mean = self._col_sum(x) / Float64(N)               # [1, C]
        var xm = x - mean                                      # broadcast row
        var varv = self._col_sum(xm * xm) / Float64(N)         # [1, C]

        # update running stats
        var c = 0
        while c < C:
            self.running_mean._data[c] = (self.running_mean._data[c] * (1.0 - self.momentum)) + (mean._data[c] * self.momentum)
            self.running_var._data[c]  = (self.running_var._data[c]  * (1.0 - self.momentum)) + (varv._data[c] * self.momentum)
            c = c + 1

        # invstd = 1 / sqrt(var + eps)
        var invstd = tensor.Tensor[Float64]([1, C], 0.0)
        c = 0
        while c < C:
            invstd._data[c] = 1.0 / _sqrt_pos_f64(varv._data[c] + self.eps)
            c = c + 1

        return (xm * invstd) * self.gamma + self.beta          # broadcasting row-wise

    fn forward(mut self, x: tensor.GradTensor) -> tensor.GradTensor:
        # x: [N, C]
        var shp = x.shape()
        if len(shp) != 2 or shp[1] != self.num_features:
            return x.copy()

        var N = shp[0]; var C = shp[1]
        if N <= 0: return x.copy()

        var mean = self._col_sum(x) / Float64(N)               # [1, C]
        var xm = x - mean                                      # broadcast row
        var varv = self._col_sum(xm * xm) / Float64(N)         # [1, C]

        # update running stats
        var c = 0
        while c < C:
            self.running_mean._data[c] = (self.running_mean._data[c] * (1.0 - self.momentum)) + (mean._data[c] * self.momentum)
            self.running_var._data[c]  = (self.running_var._data[c]  * (1.0 - self.momentum)) + (varv._data[c] * self.momentum)
            c = c + 1

        # invstd = 1 / sqrt(var + eps)
        var invstd = tensor.GradTensor([1, C], 0.0)
        c = 0
        while c < C:
            invstd._data[c] = 1.0 / _sqrt_pos_f64(varv._data[c] + self.eps)
            c = c + 1

        return (xm * invstd) * self.gamma + self.beta          # broadcasting row-wise

fn _fill_uniform_(mut t: tensor.Tensor[Float64],
                  low: Float64,
                  high: Float64,
                  mut rng: RNG):
    var n = _numel(t.shape())
    var i = 0
    while i < n:
        t._data[i] = rng.uniform(low, high)   
        i = i + 1

fn _xavier_uniform_(mut t: tensor.Tensor[Float64], mut rng: RNG):
    var shp = t.shape()
    var fans = _calc_fans(shp)
    var fan_in = fans[0]
    var fan_out = fans[1]
    var denom = fan_in + fan_out
    if denom <= 0: return
    var bound = _sqrt_pos_f64(6.0 / Float64(denom))
    _fill_uniform_(t, -bound, bound, rng)

fn _kaiming_uniform_relu_(mut t: tensor.Tensor[Float64], mut rng: RNG):
    var shp = t.shape()
    var fans = _calc_fans(shp)
    var fan_in = fans[0]
    if fan_in <= 0: return
    var bound = _sqrt_pos_f64(6.0 / Float64(fan_in))
    _fill_uniform_(t, -bound, bound, rng)


# -----------------------------------------------------------------------------
# Linear
# -----------------------------------------------------------------------------

struct Linear(Copyable, Movable):
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
        wshape.append(out_features)
        wshape.append(in_features)
        self.weight = _zeros_f64(wshape)

        var bshape = List[Int]()
        if bias:
            bshape.append(out_features)
            self.bias_t = _zeros_f64(bshape)
        else:
            bshape.append(0)
            self.bias_t = _zeros_f64(bshape)

        self.reset_parameters()

    fn reset_parameters(mut self):
        # Xavier uniform for weights, bias = zeros.
        var rng = rng_from_seed(12345)
        _xavier_uniform_(self.weight, rng)
        var bn = _numel(self.bias_t.shape())
        var j = 0
        while j < bn:
            self.bias_t._data[j] = 0.0
            j = j + 1

        
    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var shp = x.shape()
        if len(shp) != 2 or shp[1] != self.in_features:
            return x.copy()

        var y = x.matmul(self.weight.transpose())
        if self.bias:
            y = y + self.bias_t
        return y.copy()
        
    fn forward(self, x: tensor.GradTensor) -> tensor.GradTensor:
        var shp = x.shape()
        if len(shp) != 2 or shp[1] != self.in_features:
            return x.copy()

        var y = x.matmul(self.weight.transpose())
        if self.bias:
            y = y + self.bias_t
        return y.copy()


    fn __str__(self) -> String:
        var s = String("Linear(")
        s = s + "in_features=" + String(self.in_features)
        s = s + ", out_features=" + String(self.out_features)
        s = s + ", bias=" + (String("True") if self.bias else String("False")) + ")"
        return s

    fn summarize(self, s: Pointer[Summarizer]):
        var in_shape = List[Int]()
        in_shape.append(-1)
        in_shape.append(self.in_features)
        var out_shape = List[Int]()
        out_shape.append(-1)
        out_shape.append(self.out_features)
        var ps = List[tensor.Tensor[Float64]]()
        ps.append(self.weight)
        var bs = List[tensor.Tensor[Float64]]()
        if self.bias:
            bs.append(self.bias_t)
        s.value.add_params_f64("Linear", in_shape, out_shape, ps, bs, True)

# -----------------------------------------------------------------------------
# Conv2d (NCHW, groups supported, naive loops)
# -----------------------------------------------------------------------------

@always_inline
fn _idx_nchw(n: Int, c: Int, h: Int, w: Int, C: Int, H: Int, W: Int) -> Int:
    # Row-major contiguous: ((((n*C)+c)*H)+h)*W + w
    return (((n * C) + c) * H + h) * W + w
  

fn _zeros(shape: List[Int]) -> tensor.Tensor[Float64]:
    return tensor.zeros(shape)

fn _randn(shape: List[Int], scale: Float64 = 0.01) -> tensor.Tensor[Float64]:
    var t = tensor.randn(shape)
    # scale in-place
    var n = t.numel()
    var i = 0
    while i < n:
        t._data[i] = t._data[i] * scale
        i = i + 1
    return t.copy()

struct Conv2d(Copyable, Movable):
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
        wshape.append(out_channels)
        wshape.append(icg)
        wshape.append(kernel_size)
        wshape.append(kernel_size)
        # small Gaussian init (replace with project Kaiming if available)
        self.weight = _randn(wshape, 0.02)

        var bshape = List[Int]()
        if bias:
            bshape.append(out_channels)
            self.bias_t = _zeros(bshape)
        else:
            bshape.append(0)
            self.bias_t = _zeros(bshape)

    fn __copyinit__(out self, other: Self):
        self.in_channels  = other.in_channels
        self.out_channels = other.out_channels
        self.kernel_size  = other.kernel_size
        self.stride       = other.stride
        self.padding      = other.padding
        self.dilation     = other.dilation
        self.groups       = other.groups
        self.bias         = other.bias
        self.weight = other.weight.copy()
        self.bias_t = other.bias_t.copy()

    fn forward(self, x: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        # Naive NCHW convolution with groups, stride, padding, dilation.
        var shp = x.shape()
        if len(shp) != 4 or shp[1] != self.in_channels:
            # pass-through when incompatible
            return x.copy()

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
            return x.copy()

        var yshape = List[Int]()
        yshape.append(N); yshape.append(OC); yshape.append(OH); yshape.append(OW)
        var y = tensor.zeros(yshape)

        var xd = x._data.copy()
        var yd = y._data.copy()
        var wd = self.weight._data.copy()
        var bd = self.bias_t._data.copy()

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
                                        # x index
                                        var x_idx = _idx_nchw(n, ic, ih, iw, C, H, W)
                                        # w index: [oc, icg, kh, kw] in row-major
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
        return y.copy()

    fn forward(self, x: tensor.GradTensor) -> tensor.GradTensor:
        # Naive NCHW convolution with groups, stride, padding, dilation.
        var shp = x.shape()
        if len(shp) != 4 or shp[1] != self.in_channels:
            # pass-through when incompatible
            return x.copy()

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
            return x.copy()

        var yshape = List[Int]()
        yshape.append(N); yshape.append(OC); yshape.append(OH); yshape.append(OW)
        var y = tensor.zeros(yshape)

        var xd = x._data.copy()
        var yd = y._data.copy()
        var wd = self.weight._data.copy()
        var bd = self.bias_t._data.copy()

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
                                        # x index
                                        var x_idx = _idx_nchw(n, ic, ih, iw, C, H, W)
                                        # w index: [oc, icg, kh, kw] in row-major
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
        return y.copy()

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


# -----------------------------------------------------------------------------
# BatchNorm2d (NCHW; training/eval behavior with running stats)
# -----------------------------------------------------------------------------

struct BatchNorm2d(Copyable, Movable):
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

        var cshape = List[Int](); cshape.append(num_features)

        if affine:
            self.weight = _ones_f64(cshape)
            self.bias_t = _zeros_f64(cshape)
        else:
            var z = List[Int](); z.append(0)
            self.weight = _zeros_f64(z)
            self.bias_t = _zeros_f64(z)

        if track_running_stats:
            self.running_mean = _zeros_f64(cshape)
            self.running_var  = _ones_f64(cshape)
        else:
            var z2 = List[Int](); z2.append(0)
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

    fn forward(self, x: tensor.Tensor[Float64], training: Bool = True) -> tensor.Tensor[Float64]:
        # Only Float64 with [N,C,H,W]
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

        var yshape = List[Int](); yshape.append(N); yshape.append(C); yshape.append(H); yshape.append(W)
        var y = tensor.Tensor[Float64](yshape, 0.0)
        var yd = y._data

        var c = 0
        while c < C:
            var mean_c = 0.0
            var var_c  = 1.0

            if training:
                var sum = 0.0
                var n0 = 0
                while n0 < N:
                    var h0 = 0
                    while h0 < H:
                        var w0 = 0
                        while w0 < W:
                            sum = sum + xd[_idx_nchw(n0, c, h0, w0, C, H, W)]
                            w0 = w0 + 1
                        h0 = h0 + 1
                    n0 = n0 + 1
                mean_c = sum / count

                var sq = 0.0
                var n1 = 0
                while n1 < N:
                    var h1 = 0
                    while h1 < H:
                        var w1 = 0
                        while w1 < W:
                            var v = xd[_idx_nchw(n1, c, h1, w1, C, H, W)] - mean_c
                            sq = sq + v * v
                            w1 = w1 + 1
                        h1 = h1 + 1
                    n1 = n1 + 1
                var_c = sq / count

                if self.track_running_stats:
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

            
            var inv = 1.0 / _sqrt_pos_f64(var_c + self.eps)

            var g = 1.0
            var b = 0.0
            if self.affine:
                g = gamma[c]
                b = beta[c]

            var n2 = 0
            while n2 < N:
                var h2 = 0
                while h2 < H:
                    var w2 = 0
                    while w2 < W:
                        var xval = xd[_idx_nchw(n2, c, h2, w2, C, H, W)]
                        var z = (xval - mean_c) * inv
                        yd[_idx_nchw(n2, c, h2, w2, C, H, W)] = z * g + b
                        w2 = w2 + 1
                    h2 = h2 + 1
                n2 = n2 + 1
            c = c + 1

    return y.copy()


    fn forward(self, x: tensor.GradTensor, training: Bool = True) -> tensor.GradTensor:
        # Only Float64 with [N,C,H,W]
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

        var yshape = List[Int](); yshape.append(N); yshape.append(C); yshape.append(H); yshape.append(W)
        var y = tensor.GradTensor(yshape, 0.0)
        var yd = y._data

        var c = 0
        while c < C:
            var mean_c = 0.0
            var var_c  = 1.0

            if training:
                var sum = 0.0
                var n0 = 0
                while n0 < N:
                    var h0 = 0
                    while h0 < H:
                        var w0 = 0
                        while w0 < W:
                            sum = sum + xd[_idx_nchw(n0, c, h0, w0, C, H, W)]
                            w0 = w0 + 1
                        h0 = h0 + 1
                    n0 = n0 + 1
                mean_c = sum / count

                var sq = 0.0
                var n1 = 0
                while n1 < N:
                    var h1 = 0
                    while h1 < H:
                        var w1 = 0
                        while w1 < W:
                            var v = xd[_idx_nchw(n1, c, h1, w1, C, H, W)] - mean_c
                            sq = sq + v * v
                            w1 = w1 + 1
                        h1 = h1 + 1
                    n1 = n1 + 1
                var_c = sq / count

                if self.track_running_stats:
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

            
            var inv = 1.0 / _sqrt_pos_f64(var_c + self.eps)

            var g = 1.0
            var b = 0.0
            if self.affine:
                g = gamma[c]
                b = beta[c]

            var n2 = 0
            while n2 < N:
                var h2 = 0
                while h2 < H:
                    var w2 = 0
                    while w2 < W:
                        var xval = xd[_idx_nchw(n2, c, h2, w2, C, H, W)]
                        var z = (xval - mean_c) * inv
                        yd[_idx_nchw(n2, c, h2, w2, C, H, W)] = z * g + b
                        w2 = w2 + 1
                    h2 = h2 + 1
                n2 = n2 + 1
            c = c + 1

    return y.copy()

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
        in_shape.append(-1); in_shape.append(self.num_features); in_shape.append(-1); in_shape.append(-1)
        var out_shape = List[Int]()
        out_shape.append(-1); out_shape.append(self.num_features); out_shape.append(-1); out_shape.append(-1)

        var params = List[tensor.Tensor[Float64]]()
        var buffers = List[tensor.Tensor[Float64]]()

        if self.affine:
            params.append(self.weight)
            params.append(self.bias_t)
        if self.track_running_stats:
            buffers.append(self.running_mean)
            buffers.append(self.running_var)

        s.value.add_params_f64("BatchNorm2d", in_shape, out_shape, params, buffers, True)

# -----------------------------------------------------------------------------
# Dropout
# -----------------------------------------------------------------------------

struct Dropout(Copyable, Movable):
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

    fn forward(self, x: tensor.Tensor[Float64], training: Bool = True) -> tensor.Tensor[Float64]:
        if not training:
            return x.copy()

        var keep_prob = 1.0 - self.p
        if keep_prob <= 0.0:
            return x.copy()

        var shp = x.shape()
        var n = _numel(shp)
        var rng = rng_from_seed(self.seed)

        if self.inplace:
            var xd = x._data.copy()
            var i = 0
            while i < n:
                var r = rng.uniform(0.0, 1.0)
                if r < keep_prob:
                    xd[i] = xd[i] / keep_prob
                else:
                    xd[i] = 0.0
                i = i + 1
            return x.copy()
        else:
            var out = tensor.Tensor[Float64](shp, 0.0)
            var xd2 = x._data.copy()
            var yd = out._data.copy()
            var j = 0
            while j < n:
                var r2 = rng.uniform(0.0, 1.0)
                if r2 < keep_prob:
                    yd[j] = xd2[j] / keep_prob
                else:
                    yd[j] = 0.0
                j = j + 1
            return out.copy()

    fn forward(self, x: tensor.GradTensor, training: Bool = True) -> tensor.GradTensor:
        if not training:
            return x.copy()

        var keep_prob = 1.0 - self.p
        if keep_prob <= 0.0:
            return x.copy()

        var shp = x.shape()
        var n = _numel(shp)
        var rng = rng_from_seed(self.seed)

        if self.inplace:
            var xd = x._data.copy()
            var i = 0
            while i < n:
                var r = rng.uniform(0.0, 1.0)
                if r < keep_prob:
                    xd[i] = xd[i] / keep_prob
                else:
                    xd[i] = 0.0
                i = i + 1
            return x.copy()
        else:
            var out = tensor.Tensor[Float64](shp, 0.0)
            var xd2 = x._data.copy()
            var yd = out._data.copy()
            var j = 0
            while j < n:
                var r2 = rng.uniform(0.0, 1.0)
                if r2 < keep_prob:
                    yd[j] = xd2[j] / keep_prob
                else:
                    yd[j] = 0.0
                j = j + 1
            return out.copy()

    fn __str__(self) -> String:
        var s = String("Dropout(")
        s = s + "p=" + String(self.p)
        s = s + ", inplace=" + (String("True") if self.inplace else String("False"))
        s = s + ", seed=" + String(self.seed) + ")"
        return s

    fn summarize(self, s: Pointer[Summarizer]):
        var in_shape = List[Int](); in_shape.append(-1)
        var out_shape = List[Int](); out_shape.append(-1)
        s.value.add_leaf("Dropout", in_shape, out_shape)


# -----------------------------------------------------------------------------
# Flatten
# -----------------------------------------------------------------------------

struct Flatten(Copyable, Movable):
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
        var in_shape = List[Int](); in_shape.append(-1)
        var out_shape = List[Int](); out_shape.append(-1)
        s.value.add_leaf("Flatten", in_shape, out_shape)

 

# ----------------------------- Quantized Linear (INT8 weights + float out) ----
struct QuantLinear(Copyable, Movable):
    var in_features: Int
    var out_features: Int
    var w_q: tensor.Tensor[Int8]        # [out, in]
    var b_f: tensor.Tensor[Float64]     # [out]
    var s_w: Float64
    var s_x: Float64

    fn __init__(out self, in_features: Int, out_features: Int,
                w_q: tensor.Tensor[Int8], b_f: tensor.Tensor[Float64],
                s_w: Float64, s_x: Float64):
        self.in_features = in_features
        self.out_features = out_features
        self.w_q = w_q.copy()
        self.b_f = b_f.copy()
        self.s_w = s_w
        self.s_x = s_x

    fn forward(self, x_f32: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var x_q = quantize_symmetric_int8(x_f32, self.s_x)
        var y_i32 = matmul_i8_i8_to_i32(x_q, self.w_q.transpose())
        var sf = self.s_x * self.s_w
        return dequantize_i32_to_f32_add_bias(y_i32, sf, self.b_f)
    fn forward(self, x_f32: tensor.GradTensor) -> tensor.GradTensor:
        var x_q = quantize_symmetric_int8(x_f32, self.s_x)
        var y_i32 = matmul_i8_i8_to_i32(x_q, self.w_q.transpose())
        var sf = self.s_x * self.s_w
        return dequantize_i32_to_f32_add_bias(y_i32, sf, self.b_f)