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

from momijo.tensor.gpu.runtime import (
    gpu_available,
    memset_f32,
    atomic_add_f32,
    # 1D launchers:
    launch_1d_maxpool_fw,
    launch_1d_maxpool_bw,
    launch_1d_conv2d_fw,
    # kernel types for signature checking
    Kernel1D_MaxPoolFW,
    Kernel1D_MaxPoolBW,
    Kernel1D_Conv2DFW,
    Kernel1D_LinearFW,
    Kernel1D_LinearBW_DX,
    Kernel1D_LinearBW_DW,
    Kernel1D_LinearBW_DB,
    launch_1d_linear_fw,
    launch_1d_linear_bw_dx,
    launch_1d_linear_bw_dw,
    launch_1d_linear_bw_db,
    launch_1d_dropout_fw,
    Kernel1D_DropoutFW
)
# ---------- math helpers (approx) ----------
@always_inline
fn _sqrt_pos_f64(x: Float32) -> Float32:
    if x <= 0.0: return 0.0
    var y = x
    var i = 0
    while i < 12:            # Newton-Raphson
        y = 0.5 * (y + x / y)
        i = i + 1
    return y

@always_inline
fn _exp_f64(x: Float32) -> Float32:

    var term = 1.0
    var sum = 1.0
    var k = 1
    while k <= 12:
        term = term * x / Float32(k)
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
fn _zeros_f64(shape: List[Int]) -> tensor.Tensor[Float32]:
    return tensor.Tensor[Float32](shape, 0.0)

@always_inline
fn _ones_f64(shape: List[Int]) -> tensor.Tensor[Float32]:
    var t = tensor.Tensor[Float32](shape, 0.0)
    var n = _numel(shape)
    var i = 0
    while i < n:
        t._data[i] = 1.0
        i = i + 1
    return t.copy()

# elementwise max(x, a)
fn _maximum_scalar(x: tensor.Tensor[Float32], a: Float32) -> tensor.Tensor[Float32]:
    var y = tensor.Tensor[Float32](x.shape(), 0.0)
    var n = _numel(x.shape())
    var i = 0
    while i < n:
        var v = x._data[i]
        y._data[i] = v if v > a else a
        i = i + 1
    return y.copy()
# elementwise max(x, a)
fn _maximum_scalar(mut ctx: GradContext, x: GradTensor, a: Float32) -> GradTensor:
    return x.maximum_scalar(ctx, a)

# ---------- activations ----------
struct ReLU(Copyable, Movable):
    fn __init__(out self): pass
    fn __copyinit__(out self, other: Self): pass
    fn forward(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        return _maximum_scalar(x, 0.0)
    fn forward(self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        return x.maximum_scalar(ctx, 0.0)

struct LeakyReLU(Copyable, Movable):
    var slope: Float32
    fn __init__(out self, slope: Float32 = 0.01): self.slope = slope
    fn __copyinit__(out self, other: Self): self.slope = other.slope
    fn forward(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var pos = _maximum_scalar(x, 0.0)
        var neg = x - pos
        return pos + (neg * self.slope)
    fn forward(self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        var pos   = x.maximum_scalar(ctx, 0.0)          # max(x, 0)
        var neg   = x.sub(ctx, pos)                     # x - max(x,0)  (<= 0)
        var leak  = neg.mul_scalar(ctx, self.slope)     # slope * negative part
        return pos.add(ctx, leak)                       # max(x,0) + slope * min(x,0)

struct Sigmoid(Copyable, Movable):
    fn __init__(out self): pass
    fn __copyinit__(out self, other: Self): pass
    fn forward(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:

        var y = tensor.Tensor[Float32](x.shape(), 0.0)
        var n = _numel(x.shape())
        var i = 0
        while i < n:
            var v = x._data[i]
            var e = _exp_f64(-v)
            y._data[i] = 1.0 / (1.0 + e)
            i = i + 1
        return y.copy()
    # Sigmoid (autograd-aware): y = 1 / (1 + exp(-x))
    fn forward(self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        var den  = x.neg(ctx).exp(ctx).add_scalar(ctx, 1.0)   # 1 + exp(-x)
        var shp  = ctx.tape.values[den.id]._shape.copy()      # same shape as x
        var one  = GradTensor.from_tensor(ctx, tensor.zeros(shp).add_scalar(1.0), False)
        return one.div_scalar(ctx, den)


struct Tanh(Copyable, Movable):
    fn __init__(out self): pass
    fn __copyinit__(out self, other: Self): pass
    fn forward(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        # tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})
        var y = tensor.Tensor[Float32](x.shape(), 0.0)
        var n = _numel(x.shape())
        var i = 0
        while i < n:
            var v = x._data[i]
            var ex = _exp_f64(v)
            var em = _exp_f64(-v)
            y._data[i] = (ex - em) / (ex + em)
            i = i + 1
        return y.copy()

    # Tanh (autograd-aware): tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
    fn forward(self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        var ex  = x.exp(ctx)               # e^x
        var em  = x.neg(ctx).exp(ctx)      # e^{-x}
        var num = ex.sub(ctx, em)          # e^x - e^{-x}
        var den = ex.add(ctx, em)          # e^x + e^{-x}
        return num.div_scalar(ctx, den)


struct BatchNorm1d(Copyable, Movable):
    var num_features: Int
    var gamma: tensor.Tensor[Float32]
    var beta: tensor.Tensor[Float32]
    var running_mean: tensor.Tensor[Float32]
    var running_var: tensor.Tensor[Float32]
    var eps: Float32
    var momentum: Float32

    fn __init__(out self, num_features: Int, eps: Float32 = 1e-5, momentum: Float32 = 0.1):
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

    fn _col_sum(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        # Sum over rows ⇒ [1, C]
        var shp = x.shape()          # [N, C]
        var N = shp[0]; var C = shp[1]
        var out = tensor.Tensor[Float32]([1, C], 0.0)
        var r = 0
        while r < N:
            var c = 0
            while c < C:
                out._data[c] = out._data[c] + x._data[r * C + c]
                c = c + 1
            r = r + 1
        return out.copy()

    fn forward(mut self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        # x: [N, C]
        var shp = x.shape()
        if len(shp) != 2 or shp[1] != self.num_features:
            return x.copy()

        var N = shp[0]; var C = shp[1]
        if N <= 0: return x.copy()

        var mean = self._col_sum(x) / Float32(N)               # [1, C]
        var xm = x - mean                                      # broadcast row
        var varv = self._col_sum(xm * xm) / Float32(N)         # [1, C]

        # update running stats
        var c = 0
        while c < C:
            self.running_mean._data[c] = (self.running_mean._data[c] * (1.0 - self.momentum)) + (mean._data[c] * self.momentum)
            self.running_var._data[c]  = (self.running_var._data[c]  * (1.0 - self.momentum)) + (varv._data[c] * self.momentum)
            c = c + 1

        # invstd = 1 / sqrt(var + eps)
        var invstd = tensor.Tensor[Float32]([1, C], 0.0)
        c = 0
        while c < C:
            invstd._data[c] = 1.0 / _sqrt_pos_f64(varv._data[c] + self.eps)
            c = c + 1

        return (xm * invstd) * self.gamma + self.beta          # broadcasting row-wise

    fn forward(mut self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        # x: [N, C]
        var shp = ctx.tape.values[x.id]._shape.copy()
        if len(shp) != 2 or shp[1] != self.num_features:
            return x.copy()

        var N = shp[0]
        var C = shp[1]
        if N <= 0:
            return x.copy()

        # column-sum over N → [1, C]
        var sum_cols = x.sum_axis(ctx, 0, True)           # [1, C]
        var mean     = sum_cols.div_scalar(ctx, Float32(N))# [1, C]

        var xm       = x.sub(ctx, mean)                    # [N, C]
        var var_num  = xm.mul(ctx, xm).sum_axis(ctx, 0, True)       # [1, C]
        var varv     = var_num.div_scalar(ctx, Float32(N))          # [1, C]

        # update running stats (no grad)
        var mean_t = ctx.tape.values[mean.id].copy()
        var varv_t = ctx.tape.values[varv.id].copy()
        var c = 0
        while c < C:
            self.running_mean._data[c] = self.running_mean._data[c] * (1.0 - self.momentum) + mean_t._data[c] * self.momentum
            self.running_var._data[c]  = self.running_var._data[c]  * (1.0 - self.momentum) + varv_t._data[c] * self.momentum
            c = c + 1

        # invstd = 1 / sqrt(var + eps)
        var denom  = varv.add_scalar(ctx, self.eps).sqrt(ctx)        # [1, C]
        var one    = GradTensor.from_tensor(ctx, tensor.zeros([1, C]).add_scalar(1.0), False)
        var invstd = one.div_scalar(ctx, denom)                             # [1, C]

        # y = (x - mean) * invstd * gamma + beta
        var y = xm.mul(ctx, invstd)
        y = y.mul(ctx, self.gamma)
        y = y.add(ctx, self.beta)
        return y.copy()

fn _fill_uniform_(mut t: tensor.Tensor[Float32],
                  low: Float32,
                  high: Float32,
                  mut rng: RNG):
    var n = _numel(t.shape())
    var i = 0
    while i < n:
        t._data[i] = rng.uniform(low, high)
        i = i + 1

fn _xavier_uniform_(mut t: tensor.Tensor[Float32], mut rng: RNG):
    var shp = t.shape()
    var fans = _calc_fans(shp)
    var fan_in = fans[0]
    var fan_out = fans[1]
    var denom = fan_in + fan_out
    if denom <= 0: return
    var bound = _sqrt_pos_f64(6.0 / Float32(denom))
    _fill_uniform_(t, -bound, bound, rng)

fn _kaiming_uniform_relu_(mut t: tensor.Tensor[Float32], mut rng: RNG):
    var shp = t.shape()
    var fans = _calc_fans(shp)
    var fan_in = fans[0]
    if fan_in <= 0: return
    var bound = _sqrt_pos_f64(6.0 / Float32(fan_in))
    _fill_uniform_(t, -bound, bound, rng)


# -----------------------------------------------------------------------------
# Linear
# -----------------------------------------------------------------------------


struct Linear(Copyable, Movable):
    var in_features:  Int
    var out_features: Int
    var bias:         Bool
    var weight: tensor.Tensor[Float32]   # [OutF, InF]
    var bias_t: tensor.Tensor[Float32]   # [OutF] or []

    fn __init__(out self, in_features: Int, out_features: Int, bias: Bool = True):
        self.in_features  = in_features
        self.out_features = out_features
        self.bias         = bias
        self.weight = tensor.zeros([out_features, in_features])
        if bias:
            self.bias_t = tensor.zeros([out_features])
        else:
            self.bias_t = tensor.zeros([0])
        self.reset_parameters()

    fn reset_parameters(mut self):
        var wd = self.weight._data.copy()
        var scale: Float32 = 0.01 / Float32(self.in_features)
        var i = 0
        var n = len(wd)
        while i < n:
            wd[i] = scale
            i = i + 1
        self.weight._data = wd.copy()
        if self.bias:
            var bd = self.bias_t._data.copy()
            var j = 0
            var m = len(bd)
            while j < m:
                bd[j] = 0.0
                j = j + 1
            self.bias_t._data = bd.copy()

    @always_inline
    fn _has_gpu(self) -> Bool:
        return gpu_available()

    fn forward_cpu_parallel(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var shp = x.shape()
        if len(shp) != 2:
            return x.copy()
        var N   = shp[0]
        var InF = shp[1]
        if InF != self.in_features:
            return x.copy()
        var OutF = self.out_features
        var y = tensor.zeros([N, OutF])

        var xd = x._data.copy()
        var wd = self.weight._data.copy()
        var bd = self.bias_t._data.copy()
        var yd = y._data.copy()

        var n = 0
        while n < N:
            var x_row = n * InF
            var o = 0
            while o < OutF:
                var w_row = o * InF
                var acc: Float32 = 0.0
                var k = 0
                while k + 7 < InF:
                    acc = acc + xd[x_row + k + 0] * wd[w_row + k + 0]
                    acc = acc + xd[x_row + k + 1] * wd[w_row + k + 1]
                    acc = acc + xd[x_row + k + 2] * wd[w_row + k + 2]
                    acc = acc + xd[x_row + k + 3] * wd[w_row + k + 3]
                    acc = acc + xd[x_row + k + 4] * wd[w_row + k + 4]
                    acc = acc + xd[x_row + k + 5] * wd[w_row + k + 5]
                    acc = acc + xd[x_row + k + 6] * wd[w_row + k + 6]
                    acc = acc + xd[x_row + k + 7] * wd[w_row + k + 7]
                    k = k + 8
                while k + 3 < InF:
                    acc = acc + xd[x_row + k + 0] * wd[w_row + k + 0]
                    acc = acc + xd[x_row + k + 1] * wd[w_row + k + 1]
                    acc = acc + xd[x_row + k + 2] * wd[w_row + k + 2]
                    acc = acc + xd[x_row + k + 3] * wd[w_row + k + 3]
                    k = k + 4
                while k < InF:
                    acc = acc + xd[x_row + k] * wd[w_row + k]
                    k = k + 1
                if self.bias and len(bd) == OutF:
                    acc = acc + bd[o]
                yd[n * OutF + o] = acc
                o = o + 1
            n = n + 1

        y._data = yd.copy()
        return y.copy()

    @staticmethod
    fn _kernel_linear_fw(
        tid: Int,
        x:  List[Float32],
        wT: List[Float32],
        b:  List[Float32],
        mut y:  List[Float32],
        N: Int, InF: Int, OutF: Int,
        has_bias: Bool
    ) -> None:
        var total = N * OutF
        if tid >= total:
            return
        var n = tid // OutF
        var o = tid - n * OutF
        var x_row = n * InF
        var col   = o
        var acc: Float32 = 0.0
        var k = 0
        while k + 7 < InF:
            acc = acc + x[x_row + k + 0] * wT[(k + 0) * OutF + col]
            acc = acc + x[x_row + k + 1] * wT[(k + 1) * OutF + col]
            acc = acc + x[x_row + k + 2] * wT[(k + 2) * OutF + col]
            acc = acc + x[x_row + k + 3] * wT[(k + 3) * OutF + col]
            acc = acc + x[x_row + k + 4] * wT[(k + 4) * OutF + col]
            acc = acc + x[x_row + k + 5] * wT[(k + 5) * OutF + col]
            acc = acc + x[x_row + k + 6] * wT[(k + 6) * OutF + col]
            acc = acc + x[x_row + k + 7] * wT[(k + 7) * OutF + col]
            k = k + 8
        while k + 3 < InF:
            acc = acc + x[x_row + k + 0] * wT[(k + 0) * OutF + col]
            acc = acc + x[x_row + k + 1] * wT[(k + 1) * OutF + col]
            acc = acc + x[x_row + k + 2] * wT[(k + 2) * OutF + col]
            acc = acc + x[x_row + k + 3] * wT[(k + 3) * OutF + col]
            k = k + 4
        while k < InF:
            acc = acc + x[x_row + k] * wT[k * OutF + col]
            k = k + 1
        if has_bias and len(b) == OutF:
            acc = acc + b[o]
        y[n * OutF + o] = acc

    fn forward_gpu(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var shp = x.shape()
        if len(shp) != 2:
            return x.copy()
        var N   = shp[0]
        var InF = shp[1]
        if InF != self.in_features:
            return x.copy()
        var OutF = self.out_features
        var y = tensor.zeros([N, OutF])

        var xd = x._data.copy()
        var wT = List[Float32]()
        wT.reserve(InF * OutF)
        var ic = 0
        while ic < InF:
            var oc = 0
            while oc < OutF:
                wT.append(self.weight._data[oc * InF + ic])
                oc = oc + 1
            ic = ic + 1
        var bd = self.bias_t._data.copy()
        var yd = y._data.copy()

        var total = N * OutF
        var block = 256
        if block > total:
            block = total
        launch_1d_linear_fw(total, block, Self._kernel_linear_fw, xd, wT, bd, yd, N, InF, OutF, self.bias)

        y._data = yd.copy()
        return y.copy()

    fn forward_auto(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        if self._has_gpu():
            return self.forward_gpu(x)
        return self.forward_cpu_parallel(x)

    fn forward(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        return self.forward_auto(x)

    # -------- Backward CPU --------
    fn backward_cpu_parallel(self, x: tensor.Tensor[Float32], dy: tensor.Tensor[Float32])
        -> (tensor.Tensor[Float32], tensor.Tensor[Float32], tensor.Tensor[Float32]):
        var shp = x.shape()
        var shp2 = dy.shape()
        if len(shp) != 2 or len(shp2) != 2:
            return (tensor.zeros_like(self.weight), tensor.zeros_like(self.bias_t), tensor.zeros_like(x))
        var N   = shp[0]
        var InF = shp[1]
        var N2  = shp2[0]
        var OutF = shp2[1]
        if N != N2 or InF != self.in_features or OutF != self.out_features:
            return (tensor.zeros_like(self.weight), tensor.zeros_like(self.bias_t), tensor.zeros_like(x))

        var dx = tensor.zeros([N, InF])
        var dW = tensor.zeros([OutF, InF])
        var db = tensor.zeros([OutF])

        var xd = x._data.copy()
        var dyd = dy._data.copy()
        var wd = self.weight._data.copy()
        var dxd = dx._data.copy()
        var dWd = dW._data.copy()
        var dbd = db._data.copy()

        # db
        var o = 0
        while o < OutF:
            var s: Float32 = 0.0
            var n = 0
            while n < N:
                s = s + dyd[n * OutF + o]
                n = n + 1
            dbd[o] = s
            o = o + 1

        # dx = dy @ W
        var n2 = 0
        while n2 < N:
            var dy_row = n2 * OutF
            var i = 0
            while i < InF:
                var accx: Float32 = 0.0
                var o2 = 0
                while o2 + 7 < OutF:
                    var woff = o2 * InF + i
                    accx = accx + dyd[dy_row + o2 + 0] * wd[woff + 0 * InF]
                    accx = accx + dyd[dy_row + o2 + 1] * wd[woff + 1 * InF]
                    accx = accx + dyd[dy_row + o2 + 2] * wd[woff + 2 * InF]
                    accx = accx + dyd[dy_row + o2 + 3] * wd[woff + 3 * InF]
                    accx = accx + dyd[dy_row + o2 + 4] * wd[woff + 4 * InF]
                    accx = accx + dyd[dy_row + o2 + 5] * wd[woff + 5 * InF]
                    accx = accx + dyd[dy_row + o2 + 6] * wd[woff + 6 * InF]
                    accx = accx + dyd[dy_row + o2 + 7] * wd[woff + 7 * InF]
                    o2 = o2 + 8
                while o2 + 3 < OutF:
                    var woff4 = o2 * InF + i
                    accx = accx + dyd[dy_row + o2 + 0] * wd[woff4 + 0 * InF]
                    accx = accx + dyd[dy_row + o2 + 1] * wd[woff4 + 1 * InF]
                    accx = accx + dyd[dy_row + o2 + 2] * wd[woff4 + 2 * InF]
                    accx = accx + dyd[dy_row + o2 + 3] * wd[woff4 + 3 * InF]
                    o2 = o2 + 4
                while o2 < OutF:
                    accx = accx + dyd[dy_row + o2] * wd[o2 * InF + i]
                    o2 = o2 + 1
                dxd[n2 * InF + i] = accx
                i = i + 1
            n2 = n2 + 1

        # dW = dy^T @ x
        var o3 = 0
        while o3 < OutF:
            var i2 = 0
            while i2 < InF:
                var accw: Float32 = 0.0
                var n3 = 0
                while n3 + 7 < N:
                    accw = accw + dyd[(n3 + 0) * OutF + o3] * xd[(n3 + 0) * InF + i2]
                    accw = accw + dyd[(n3 + 1) * OutF + o3] * xd[(n3 + 1) * InF + i2]
                    accw = accw + dyd[(n3 + 2) * OutF + o3] * xd[(n3 + 2) * InF + i2]
                    accw = accw + dyd[(n3 + 3) * OutF + o3] * xd[(n3 + 3) * InF + i2]
                    accw = accw + dyd[(n3 + 4) * OutF + o3] * xd[(n3 + 4) * InF + i2]
                    accw = accw + dyd[(n3 + 5) * OutF + o3] * xd[(n3 + 5) * InF + i2]
                    accw = accw + dyd[(n3 + 6) * OutF + o3] * xd[(n3 + 6) * InF + i2]
                    accw = accw + dyd[(n3 + 7) * OutF + o3] * xd[(n3 + 7) * InF + i2]
                    n3 = n3 + 8
                while n3 + 3 < N:
                    accw = accw + dyd[(n3 + 0) * OutF + o3] * xd[(n3 + 0) * InF + i2]
                    accw = accw + dyd[(n3 + 1) * OutF + o3] * xd[(n3 + 1) * InF + i2]
                    accw = accw + dyd[(n3 + 2) * OutF + o3] * xd[(n3 + 2) * InF + i2]
                    accw = accw + dyd[(n3 + 3) * OutF + o3] * xd[(n3 + 3) * InF + i2]
                    n3 = n3 + 4
                while n3 < N:
                    accw = accw + dyd[n3 * OutF + o3] * xd[n3 * InF + i2]
                    n3 = n3 + 1
                dWd[o3 * InF + i2] = accw
                i2 = i2 + 1
            o3 = o3 + 1

        dx._data = dxd.copy()
        dW._data = dWd.copy()
        db._data = dbd.copy()
        return (dW.copy(), db.copy(), dx.copy())

    # -------- Backward GPU --------
    @staticmethod
    fn _kernel_linear_bw_dx(
        tid: Int,
        dy: List[Float32],
        w:  List[Float32],
        mut dx: List[Float32],
        N: Int, InF: Int, OutF: Int
    ) -> None:
        var total = N * InF
        if total <= 0 or tid >= total:
            return
        var n = tid // InF
        var i = tid - n * InF
        var dy_row = n * OutF
        var acc: Float32 = 0.0
        var o = 0
        while o + 7 < OutF:
            var woff = o * InF + i
            acc = acc + dy[dy_row + o + 0] * w[woff + 0 * InF]
            acc = acc + dy[dy_row + o + 1] * w[woff + 1 * InF]
            acc = acc + dy[dy_row + o + 2] * w[woff + 2 * InF]
            acc = acc + dy[dy_row + o + 3] * w[woff + 3 * InF]
            acc = acc + dy[dy_row + o + 4] * w[woff + 4 * InF]
            acc = acc + dy[dy_row + o + 5] * w[woff + 5 * InF]
            acc = acc + dy[dy_row + o + 6] * w[woff + 6 * InF]
            acc = acc + dy[dy_row + o + 7] * w[woff + 7 * InF]
            o = o + 8
        while o + 3 < OutF:
            var woff4 = o * InF + i
            acc = acc + dy[dy_row + o + 0] * w[woff4 + 0 * InF]
            acc = acc + dy[dy_row + o + 1] * w[woff4 + 1 * InF]
            acc = acc + dy[dy_row + o + 2] * w[woff4 + 2 * InF]
            acc = acc + dy[dy_row + o + 3] * w[woff4 + 3 * InF]
            o = o + 4
        while o < OutF:
            acc = acc + dy[dy_row + o] * w[o * InF + i]
            o = o + 1
        dx[n * InF + i] = acc

    @staticmethod
    fn _kernel_linear_bw_dw(
        tid: Int,
        dy: List[Float32],
        x:  List[Float32],
        mut dW: List[Float32],
        N: Int, InF: Int, OutF: Int
    ) -> None:
        var total = OutF * InF
        if total <= 0 or tid >= total:
            return
        var o = tid // InF
        var i = tid - o * InF
        var acc: Float32 = 0.0
        var n = 0
        while n + 7 < N:
            acc = acc + dy[(n + 0) * OutF + o] * x[(n + 0) * InF + i]
            acc = acc + dy[(n + 1) * OutF + o] * x[(n + 1) * InF + i]
            acc = acc + dy[(n + 2) * OutF + o] * x[(n + 2) * InF + i]
            acc = acc + dy[(n + 3) * OutF + o] * x[(n + 3) * InF + i]
            acc = acc + dy[(n + 4) * OutF + o] * x[(n + 4) * InF + i]
            acc = acc + dy[(n + 5) * OutF + o] * x[(n + 5) * InF + i]
            acc = acc + dy[(n + 6) * OutF + o] * x[(n + 6) * InF + i]
            acc = acc + dy[(n + 7) * OutF + o] * x[(n + 7) * InF + i]
            n = n + 8
        while n + 3 < N:
            acc = acc + dy[(n + 0) * OutF + o] * x[(n + 0) * InF + i]
            acc = acc + dy[(n + 1) * OutF + o] * x[(n + 1) * InF + i]
            acc = acc + dy[(n + 2) * OutF + o] * x[(n + 2) * InF + i]
            acc = acc + dy[(n + 3) * OutF + o] * x[(n + 3) * InF + i]
            n = n + 4
        while n < N:
            acc = acc + dy[n * OutF + o] * x[n * InF + i]
            n = n + 1
        dW[o * InF + i] = acc

    @staticmethod
    fn _kernel_linear_bw_db(
        tid: Int,
        dy: List[Float32],
        mut db: List[Float32],
        N: Int, OutF: Int
    ) -> None:
        var o = tid
        if o >= OutF:
            return
        var acc: Float32 = 0.0
        var n = 0
        while n + 7 < N:
            acc = acc + dy[(n + 0) * OutF + o]
            acc = acc + dy[(n + 1) * OutF + o]
            acc = acc + dy[(n + 2) * OutF + o]
            acc = acc + dy[(n + 3) * OutF + o]
            acc = acc + dy[(n + 4) * OutF + o]
            acc = acc + dy[(n + 5) * OutF + o]
            acc = acc + dy[(n + 6) * OutF + o]
            acc = acc + dy[(n + 7) * OutF + o]
            n = n + 8
        while n + 3 < N:
            acc = acc + dy[(n + 0) * OutF + o]
            acc = acc + dy[(n + 1) * OutF + o]
            acc = acc + dy[(n + 2) * OutF + o]
            acc = acc + dy[(n + 3) * OutF + o]
            n = n + 4
        while n < N:
            acc = acc + dy[n * OutF + o]
            n = n + 1
        db[o] = acc

    fn backward_gpu(self, x: tensor.Tensor[Float32], dy: tensor.Tensor[Float32])
        -> (tensor.Tensor[Float32], tensor.Tensor[Float32], tensor.Tensor[Float32]):
        var shp = x.shape()
        var shp2 = dy.shape()
        if len(shp) != 2 or len(shp2) != 2:
            return (tensor.zeros_like(self.weight), tensor.zeros_like(self.bias_t), tensor.zeros_like(x))
        var N   = shp[0]
        var InF = shp[1]
        var N2  = shp2[0]
        var OutF = shp2[1]
        if N != N2 or InF != self.in_features or OutF != self.out_features:
            return (tensor.zeros_like(self.weight), tensor.zeros_like(self.bias_t), tensor.zeros_like(x))

        var dx = tensor.zeros([N, InF])
        var dW = tensor.zeros([OutF, InF])
        var db = tensor.zeros([OutF])

        var xd  = x._data.copy()
        var dyd = dy._data.copy()
        var wd  = self.weight._data.copy()

        var dxd = dx._data.copy()
        var dWd = dW._data.copy()
        var dbd = db._data.copy()

        var total_dx = N * InF
        var bs1 = 256
        if bs1 > total_dx:
            bs1 = total_dx
        launch_1d_linear_bw_dx(total_dx, bs1, Self._kernel_linear_bw_dx, dyd, wd, dxd, N, InF, OutF)

        var total_dw = OutF * InF
        var bs2 = 256
        if bs2 > total_dw:
            bs2 = total_dw
        launch_1d_linear_bw_dw(total_dw, bs2, Self._kernel_linear_bw_dw, dyd, xd, dWd, N, InF, OutF)

        var total_db = OutF
        var bs3 = 256
        if bs3 > total_db:
            bs3 = total_db
        launch_1d_linear_bw_db(total_db, bs3, Self._kernel_linear_bw_db, dyd, dbd, N, OutF)

        dx._data = dxd.copy()
        dW._data = dWd.copy()
        db._data = dbd.copy()
        return (dW.copy(), db.copy(), dx.copy())

    fn backward_auto(self, x: tensor.Tensor[Float32], dy: tensor.Tensor[Float32])
        -> (tensor.Tensor[Float32], tensor.Tensor[Float32], tensor.Tensor[Float32]):
        if self._has_gpu():
            return self.backward_gpu(x, dy)
        return self.backward_cpu_parallel(x, dy)

    fn backward(self, x: tensor.Tensor[Float32], dy: tensor.Tensor[Float32])
        -> (tensor.Tensor[Float32], tensor.Tensor[Float32], tensor.Tensor[Float32]):
        return self.backward_auto(x, dy)

    fn forward(self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        var xv = x.value(ctx)
        var y  = self.forward_auto(xv)
        var lid = ctx.tape.add_leaf(y)
        return GradTensor(lid, False)


# -----------------------------------------------------------------------------
# Conv2d (NCHW, groups supported, naive loops)
# -----------------------------------------------------------------------------

@always_inline
fn _idx_nchw(n: Int, c: Int, h: Int, w: Int, C: Int, H: Int, W: Int) -> Int:
    # Row-major contiguous: ((((n*C)+c)*H)+h)*W + w
    return (((n * C) + c) * H + h) * W + w


fn _zeros(shape: List[Int]) -> tensor.Tensor[Float32]:
    return tensor.zeros(shape)

fn _randn(shape: List[Int], scale: Float32 = 0.01) -> tensor.Tensor[Float32]:
    var t = tensor.randn(shape)
    # scale in-place
    var n = t.numel()
    var i = 0
    while i < n:
        t._data[i] = t._data[i] * scale
        i = i + 1
    return t.copy()

# struct Conv2d(Copyable, Movable):
#     var in_channels: Int
#     var out_channels: Int
#     var kernel_size: Int
#     var stride: Int
#     var padding: Int
#     var dilation: Int
#     var groups: Int
#     var bias: Bool

#     # Parameters:
#     # weight: [out_channels, in_channels/groups, k, k]
#     # bias:   [out_channels] (optional)
#     var weight: tensor.Tensor[Float32]
#     var bias_t: tensor.Tensor[Float32]

#     fn __init__(
#         out self,
#         in_channels: Int,
#         out_channels: Int,
#         kernel_size: Int,
#         stride: Int = 1,
#         padding: Int = 0,
#         dilation: Int = 1,
#         groups: Int = 1,
#         bias: Bool = True
#     ):
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#         self.bias = bias

#         var icg = in_channels // groups

#         var wshape = List[Int]()
#         wshape.append(out_channels)
#         wshape.append(icg)
#         wshape.append(kernel_size)
#         wshape.append(kernel_size)
#         # small Gaussian init (replace with project Kaiming if available)
#         self.weight = _randn(wshape, 0.02)

#         var bshape = List[Int]()
#         if bias:
#             bshape.append(out_channels)
#             self.bias_t = _zeros(bshape)
#         else:
#             bshape.append(0)
#             self.bias_t = _zeros(bshape)

#     fn __copyinit__(out self, other: Self):
#         self.in_channels  = other.in_channels
#         self.out_channels = other.out_channels
#         self.kernel_size  = other.kernel_size
#         self.stride       = other.stride
#         self.padding      = other.padding
#         self.dilation     = other.dilation
#         self.groups       = other.groups
#         self.bias         = other.bias
#         self.weight = other.weight.copy()
#         self.bias_t = other.bias_t.copy()

#     fn forward(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
#         # Naive NCHW convolution with groups, stride, padding, dilation.
#         var shp = x.shape()
#         if len(shp) != 4 or shp[1] != self.in_channels:
#             # pass-through when incompatible
#             return x.copy()

#         var N = shp[0]; var C = shp[1]; var H = shp[2]; var W = shp[3]
#         var K  = self.kernel_size
#         var S  = self.stride
#         var P  = self.padding
#         var D  = self.dilation
#         var G  = self.groups
#         var OC = self.out_channels
#         var ICG = C // G                 # in-channels per group
#         var OCG = OC // G                # out-channels per group

#         # Output spatial dims (floor division)
#         var OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
#         var OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
#         if OH <= 0 or OW <= 0:
#             return x.copy()

#         var yshape = List[Int]()
#         yshape.append(N); yshape.append(OC); yshape.append(OH); yshape.append(OW)
#         var y = tensor.zeros(yshape)

#         var xd = x._data.copy()
#         var yd = y._data.copy()
#         var wd = self.weight._data.copy()
#         var bd = self.bias_t._data.copy()

#         var n = 0
#         while n < N:
#             var oc = 0
#             while oc < OC:
#                 var g = oc // OCG
#                 var ic_base = g * ICG
#                 var oh = 0
#                 while oh < OH:
#                     var ow = 0
#                     while ow < OW:
#                         var acc = 0.0
#                         var icg = 0
#                         while icg < ICG:
#                             var ic = ic_base + icg
#                             var kh = 0
#                             while kh < K:
#                                 var ih = oh * S - P + kh * D
#                                 var kw = 0
#                                 while kw < K:
#                                     var iw = ow * S - P + kw * D
#                                     if ih >= 0 and ih < H and iw >= 0 and iw < W:
#                                         # x index
#                                         var x_idx = _idx_nchw(n, ic, ih, iw, C, H, W)
#                                         # w index: [oc, icg, kh, kw] in row-major
#                                         var w_idx = (((oc * ICG) + icg) * K + kh) * K + kw
#                                         acc = acc + xd[x_idx] * wd[w_idx]
#                                     kw = kw + 1
#                                 kh = kh + 1
#                             icg = icg + 1
#                         if self.bias:
#                             acc = acc + bd[oc]
#                         yd[_idx_nchw(n, oc, oh, ow, OC, OH, OW)] = acc
#                         ow = ow + 1
#                     oh = oh + 1
#                 oc = oc + 1
#             n = n + 1
#         return y.copy()

#    @always_inline
#     fn _idx_nchw(n: Int, c: Int, h: Int, w: Int, C: Int, H: Int, W: Int) -> Int:
#         return (((n * C) + c) * H + h) * W + w

#     fn forward(self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
#         # Work on the raw tensor value (not GradTensor internals)
#         var xv  = x.value(ctx)                 # Tensor[Float32]
#         var shp = xv._shape
#         if len(shp) != 4 or shp[1] != self.in_channels:
#             # passthrough on mismatch
#             return x

#         var N = shp[0]; var C = shp[1]; var H = shp[2]; var W = shp[3]
#         var K  = self.kernel_size
#         var S  = self.stride
#         var P  = self.padding
#         var D  = self.dilation
#         var G  = self.groups
#         var OC = self.out_channels
#         var ICG = C // G                 # in-channels per group
#         var OCG = OC // G                # out-channels per group

#         # Output spatial dims
#         var OH = (H + 2 * P - D * (K - 1) - 1) // S + 1
#         var OW = (W + 2 * P - D * (K - 1) - 1) // S + 1
#         if OH <= 0 or OW <= 0:
#             return x

#         var yshape = List[Int]()
#         yshape.append(N); yshape.append(OC); yshape.append(OH); yshape.append(OW)
#         var y = tensor.zeros(yshape)

#         # Raw buffers
#         var xd = xv._data
#         var yd = y._data
#         var wd = self.weight._data                 # weight: Tensor[Float32] [OC, ICG, K, K] row-major
#         var bd = self.bias_t._data                 # bias:   Tensor[Float32] [OC] (may be empty if bias=False)

#         var n = 0
#         while n < N:
#             var oc = 0
#             while oc < OC:
#                 var g = oc // OCG
#                 var ic_base = g * ICG
#                 var oh = 0
#                 while oh < OH:
#                     var ow = 0
#                     while ow < OW:
#                         var acc = 0.0
#                         var icg = 0
#                         while icg < ICG:
#                             var ic = ic_base + icg
#                             var kh = 0
#                             while kh < K:
#                                 var ih = oh * S - P + kh * D
#                                 var kw = 0
#                                 while kw < K:
#                                     var iw = ow * S - P + kw * D
#                                     if ih >= 0 and ih < H and iw >= 0 and iw < W:
#                                         # x index
#                                         var x_idx = _idx_nchw(n, ic, ih, iw, C, H, W)
#                                         # w index: [oc, icg, kh, kw] in row-major
#                                         var w_idx = (((oc * ICG) + icg) * K + kh) * K + kw
#                                         acc = acc + xd[x_idx] * wd[w_idx]
#                                     kw = kw + 1
#                                 kh = kh + 1
#                             icg = icg + 1
#                         if self.bias and len(bd) == OC:
#                             acc = acc + bd[oc]
#                         yd[_idx_nchw(n, oc, oh, ow, OC, OH, OW)] = acc
#                         ow = ow + 1
#                     oh = oh + 1
#                 oc = oc + 1
#             n = n + 1

#         # Return as a leaf GradTensor (no grad through this op for now)
#         var track = (x.requires_grad and ctx.grad_enabled())
#         if not track:
#             var lid = ctx.tape.add_leaf(y)
#             return GradTensor(lid, False)
#         # If you want to track conv later, replace with add_binary/add_ternary and stash hyperparams
#         var lid2 = ctx.tape.add_leaf(y)
#         return GradTensor(lid2, True)


#     fn __str__(self) -> String:
#         var s = String("Conv2d(")
#         s = s + "in_channels=" + String(self.in_channels)
#         s = s + ", out_channels=" + String(self.out_channels)
#         s = s + ", kernel_size=" + String(self.kernel_size)
#         s = s + ", stride=" + String(self.stride)
#         s = s + ", padding=" + String(self.padding)
#         s = s + ", dilation=" + String(self.dilation)
#         s = s + ", groups=" + String(self.groups)
#         s = s + ", bias=" + (String("True") if self.bias else String("False")) + ")"
#         return s


# -----------------------------------------------------------------------------
# BatchNorm2d (NCHW; training/eval behavior with running stats)
# -----------------------------------------------------------------------------

struct BatchNorm2d(Copyable, Movable):
    var num_features: Int
    var eps: Float32
    var momentum: Float32
    var affine: Bool
    var track_running_stats: Bool

    var weight: tensor.Tensor[Float32]
    var bias_t: tensor.Tensor[Float32]
    var running_mean: tensor.Tensor[Float32]
    var running_var: tensor.Tensor[Float32]

    fn __init__(
        out self,
        num_features: Int,
        eps: Float32 = 1e-5,
        momentum: Float32 = 0.1,
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

    fn forward(self, x: tensor.Tensor[Float32], training: Bool = True) -> tensor.Tensor[Float32]:
        # Only Float32 with [N,C,H,W]
        var shp = x.shape()
        if len(shp) != 4 or shp[1] != self.num_features:
            return x

        var N = shp[0]; var C = shp[1]; var H = shp[2]; var W = shp[3]
        var count = Float32(N * H * W)
        if count <= 0.0:
            return x

        var xd = x._data
        var gamma = self.weight._data
        var beta  = self.bias_t._data
        var rm    = self.running_mean._data
        var rv    = self.running_var._data

        var yshape = List[Int](); yshape.append(N); yshape.append(C); yshape.append(H); yshape.append(W)
        var y = tensor.Tensor[Float32](yshape, 0.0)
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

    fn forward(mut self, mut ctx: GradContext, x: GradTensor, training: Bool = True) -> GradTensor:
        # x: [N, C, H, W]
        var xv  = x.value(ctx)
        var shp = xv._shape
        if len(shp) != 4 or shp[1] != self.num_features:
            return x

        var N = shp[0]; var C = shp[1]; var H = shp[2]; var W = shp[3]
        var count = Float32(N * H * W)
        if count <= 0.0:
            return x

        # ---------- compute mean & var ----------
        # mean over (N,H,W) per-channel → [1, C, 1, 1]
        var mean = x.sum_axis(ctx, 0, True).sum_axis(ctx, 2, True).sum_axis(ctx, 3, True).div_scalar(ctx, count)

        var xm   = x.sub(ctx, mean)                                         # [N,C,H,W] - [1,C,1,1]
        var varn = xm.mul(ctx, xm).sum_axis(ctx, 0, True).sum_axis(ctx, 2, True).sum_axis(ctx, 3, True)
        var varv = varn.div_scalar(ctx, count)                               # [1,C,1,1]

        # ---------- update running stats (no grad) ----------
        if self.track_running_stats and training:
            var mean_t = ctx.tape.values[mean.id]                            # Tensor[Float32] [1,C,1,1]
            var varv_t = ctx.tape.values[varv.id]
            var c = 0
            while c < C:
                # index [0,c,0,0]
                var mval = mean_t._data[((0 * C + c) * 1 + 0) * 1 + 0]
                var vval = varv_t._data[((0 * C + c) * 1 + 0) * 1 + 0]
                self.running_mean._data[c] = (1.0 - self.momentum) * self.running_mean._data[c] + self.momentum * mval
                self.running_var._data[c]  = (1.0 - self.momentum)  * self.running_var._data[c]  + self.momentum * vval
                c = c + 1

        # ---------- choose stats for normalization ----------
        var use_mean = mean
        var use_var  = varv
        if not training and self.track_running_stats:
            var rm = GradTensor.from_tensor(ctx, self.running_mean, False)   # [C]
            var rv = GradTensor.from_tensor(ctx, self.running_var, False)    # [C]
            var tgt = List[Int](); tgt.append(1); tgt.append(C); tgt.append(1); tgt.append(1)
            use_mean = rm.reshape(ctx, tgt)                                  # [1,C,1,1]
            use_var  = rv.reshape(ctx, tgt)                                  # [1,C,1,1]

        # invstd = 1 / sqrt(var + eps)
        var denom  = use_var.add_scalar(ctx, self.eps).sqrt(ctx)             # [1,C,1,1]
        var one4   = GradTensor.from_tensor(ctx, tensor.zeros(List[Int]([1, C, 1, 1])).add_scalar(1.0), False)
        var invstd = one4.div_scalar(ctx, denom)                                    # [1,C,1,1]

        # normalize
        var y = x.sub(ctx, use_mean).mul(ctx, invstd)                        # [N,C,H,W]

        # affine (weight/bias از نوع GradTensor در نظر گرفته شده‌اند)
        if self.affine:
            var tgt = List[Int](); tgt.append(1); tgt.append(C); tgt.append(1); tgt.append(1)
            var gr  = self.weight.reshape(ctx, tgt)                          # [1,C,1,1]
            var br  = self.bias_t.reshape(ctx, tgt)                          # [1,C,1,1]
            y = y.mul(ctx, gr).add(ctx, br)

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
        in_shape.append(-1); in_shape.append(self.num_features); in_shape.append(-1); in_shape.append(-1)
        var out_shape = List[Int]()
        out_shape.append(-1); out_shape.append(self.num_features); out_shape.append(-1); out_shape.append(-1)

        var params = List[tensor.Tensor[Float32]]()
        var buffers = List[tensor.Tensor[Float32]]()

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
    var p: Float32
    var inplace: Bool
    var seed: UInt64

    fn __init__(out self, p: Float32 = 0.5, inplace: Bool = False, seed: Int = 1234567):
        var pp = p
        if pp < 0.0:
            pp = 0.0
        if pp > 1.0:
            pp = 1.0
        self.p = pp
        self.inplace = inplace
        self.seed = UInt64(seed)

    @always_inline
    fn _has_gpu(self) -> Bool:
        return gpu_available()

    @staticmethod
    fn _numel(shp: List[Int]) -> Int:
        var m = 1
        var i = 0
        var k = len(shp)
        while i < k:
            m = m * shp[i]
            i = i + 1
        return m

    # stateless per-index PRNG in [0,1) for determinism and parallel safety
    @staticmethod
    fn _rand01(seed: UInt64, i: Int) -> Float32:
        var x = seed ^ UInt64(i & 0x7FFFFFFF)
        x = x ^ (x >> 12)
        x = x * 0x9E3779B97F4A7C15
        x = x ^ (x >> 25)
        x = x * 0xC2B2AE3D27D4EB4F
        x = x ^ (x >> 27)
        var u32 = UInt32(x & 0xFFFFFFFF)
        # 1/2^32
        return Float32(u32) * Float32(1.0 / 4294967296.0)

    # ------------- CPU fast path (loop-unroll "SIMD-like") -------------
    fn forward_cpu_fast(self, x: tensor.Tensor[Float32], training: Bool = True) -> tensor.Tensor[Float32]:
        if not training:
            return x.copy()

        var keep_prob = 1.0 - self.p
        if keep_prob <= 0.0:
            # p == 1 → return zeros with same shape
            return tensor.zeros(x.shape()).copy()

        var shp = x.shape()
        var n = Self._numel(shp)

        if self.inplace:
            var buf = x._data.copy()
            var i = 0

            # unroll by 8
            while i + 7 < n:
                var r0 = Self._rand01(self.seed, i + 0)
                var r1 = Self._rand01(self.seed, i + 1)
                var r2 = Self._rand01(self.seed, i + 2)
                var r3 = Self._rand01(self.seed, i + 3)
                var r4 = Self._rand01(self.seed, i + 4)
                var r5 = Self._rand01(self.seed, i + 5)
                var r6 = Self._rand01(self.seed, i + 6)
                var r7 = Self._rand01(self.seed, i + 7)

                buf[i + 0] = buf[i + 0] / keep_prob if r0 < keep_prob else 0.0
                buf[i + 1] = buf[i + 1] / keep_prob if r1 < keep_prob else 0.0
                buf[i + 2] = buf[i + 2] / keep_prob if r2 < keep_prob else 0.0
                buf[i + 3] = buf[i + 3] / keep_prob if r3 < keep_prob else 0.0
                buf[i + 4] = buf[i + 4] / keep_prob if r4 < keep_prob else 0.0
                buf[i + 5] = buf[i + 5] / keep_prob if r5 < keep_prob else 0.0
                buf[i + 6] = buf[i + 6] / keep_prob if r6 < keep_prob else 0.0
                buf[i + 7] = buf[i + 7] / keep_prob if r7 < keep_prob else 0.0

                i = i + 8
            # tail
            while i < n:
                var r = Self._rand01(self.seed, i)
                buf[i] = buf[i] / keep_prob if r < keep_prob else 0.0
                i = i + 1

            x._data = buf.copy()
            return x.copy()
        else:
            var out = tensor.zeros(shp)
            var xi = x._data.copy()
            var yo = out._data.copy()
            var j = 0

            # unroll by 8
            while j + 7 < n:
                var r0 = Self._rand01(self.seed, j + 0)
                var r1 = Self._rand01(self.seed, j + 1)
                var r2 = Self._rand01(self.seed, j + 2)
                var r3 = Self._rand01(self.seed, j + 3)
                var r4 = Self._rand01(self.seed, j + 4)
                var r5 = Self._rand01(self.seed, j + 5)
                var r6 = Self._rand01(self.seed, j + 6)
                var r7 = Self._rand01(self.seed, j + 7)

                yo[j + 0] = xi[j + 0] / keep_prob if r0 < keep_prob else 0.0
                yo[j + 1] = xi[j + 1] / keep_prob if r1 < keep_prob else 0.0
                yo[j + 2] = xi[j + 2] / keep_prob if r2 < keep_prob else 0.0
                yo[j + 3] = xi[j + 3] / keep_prob if r3 < keep_prob else 0.0
                yo[j + 4] = xi[j + 4] / keep_prob if r4 < keep_prob else 0.0
                yo[j + 5] = xi[j + 5] / keep_prob if r5 < keep_prob else 0.0
                yo[j + 6] = xi[j + 6] / keep_prob if r6 < keep_prob else 0.0
                yo[j + 7] = xi[j + 7] / keep_prob if r7 < keep_prob else 0.0

                j = j + 8
            # tail
            while j < n:
                var r2 = Self._rand01(self.seed, j)
                yo[j] = xi[j] / keep_prob if r2 < keep_prob else 0.0
                j = j + 1

            out._data = yo.copy()
            return out.copy()

    # ------------- GPU kernel (List buffers + stateless PRNG) -------------
    @staticmethod
    fn _kernel_dropout_fw(
        tid: Int,
        x:   List[Float32],
        mut y:   List[Float32],
        keep_prob: Float32,
        seed: UInt64,
        n: Int
    ) -> None:
        if tid >= n:
            return

        var si = seed ^ UInt64(tid & 0x7FFFFFFF)
        si = si ^ (si >> 12)
        si = si * 0x9E3779B97F4A7C15
        si = si ^ (si >> 25)
        si = si * 0xC2B2AE3D27D4EB4F
        si = si ^ (si >> 27)
        var u32 = UInt32(si & 0xFFFFFFFF)
        var r = Float32(u32) * Float32(1.0 / 4294967296.0)

        var v = x[tid]
        if r < keep_prob:
            y[tid] = v / keep_prob
        else:
            y[tid] = 0.0

    # ------------- GPU path (1D launcher) -------------
    fn forward_gpu(self, x: tensor.Tensor[Float32], training: Bool = True) -> tensor.Tensor[Float32]:
        if not training:
            return x.copy()

        var keep_prob = 1.0 - self.p
        if keep_prob <= 0.0:
            return tensor.zeros(x.shape()).copy()

        var shp = x.shape()
        var n = Self._numel(shp)

        var total_threads = n
        var block = 256
        if block > total_threads:
            block = total_threads

        if self.inplace:
            var ybuf = x._data.copy()
            launch_1d_dropout_fw(
                total_threads, block, Self._kernel_dropout_fw,
                x._data, ybuf,
                keep_prob, self.seed, n
            )
            x._data = ybuf.copy()
            return x.copy()
        else:
            var out = tensor.zeros(shp)
            var ybuf2 = out._data.copy()
            launch_1d_dropout_fw(
                total_threads, block, Self._kernel_dropout_fw,
                x._data, ybuf2,
                keep_prob, self.seed, n
            )
            out._data = ybuf2.copy()
            return out.copy()

    # ------------- Auto switch -------------
    fn forward_auto(self, x: tensor.Tensor[Float32], training: Bool = True) -> tensor.Tensor[Float32]:
        if self._has_gpu():
            return self.forward_gpu(x, training)
        return self.forward_cpu_fast(x, training)

    fn forward(self, x: tensor.Tensor[Float32], training: Bool = True) -> tensor.Tensor[Float32]:
        return self.forward_auto(x, training)

    # ------------- Autograd version (no in-place on tape) -------------
    fn forward(self, mut ctx: GradContext, x: GradTensor, training: Bool = True) -> GradTensor:
        if not training or self.p <= 0.0:
            return x.copy()

        var keep_prob = 1.0 - self.p
        if keep_prob <= 0.0:
            return x.mul_scalar(ctx, 0.0)

        var xv = x.value(ctx)
        var shp = xv._shape.copy()
        var n = Self._numel(shp)

        # build deterministic Bernoulli mask with inverted scaling
        var mask = tensor.zeros(shp)
        var md = mask._data.copy()

        var i = 0
        while i < n:
            var r = Self._rand01(self.seed, i)
            md[i] = 1.0 / keep_prob if r < keep_prob else 0.0
            i = i + 1

        mask._data = md.copy()

        # leaf, no grad for mask
        var mgt = GradTensor.from_tensor(ctx, mask, False)
        var y = x.mul(ctx, mgt)
        return y.copy()


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
    var end_dim:   Int

    fn __init__(out self, start_dim: Int = 1, end_dim: Int = -1):
        self.start_dim = start_dim
        self.end_dim   = end_dim

    @always_inline
    fn _normalize_dims(self, shp: List[Int]) -> (Int, Int, Int):
        var rank = len(shp)
        var s = self.start_dim
        var e = self.end_dim

        if s < 0:
            s = rank + s
        if e < 0:
            e = rank + e
        if s < 0:
            s = 0
        if e >= rank:
            e = rank - 1
        return (rank, s, e)

    @always_inline
    fn _compute_new_shape(self, shp: List[Int]) -> List[Int]:
        var norm = self._normalize_dims(shp)
        var rank = norm[0]
        var s    = norm[1]
        var e    = norm[2]

        if s > e:
            return shp.copy()

        # product of [s..e]
        var prod = 1
        var i = s
        while i <= e:
            prod = prod * shp[i]
            i = i + 1

        # build new shape
        var out = List[Int]()
        i = 0
        while i < s:
            out.append(shp[i])
            i = i + 1
        out.append(prod)
        i = e + 1
        while i < rank:
            out.append(shp[i])
            i = i + 1
        return out.copy()

    # --------------------
    # Tensor fast paths
    # --------------------
    @always_inline
    fn forward_cpu(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        # reshape صفر-هزینه (بدون کپی)؛ فقط شکل را عوض می‌کنیم
        var new_shape = self._compute_new_shape(x._shape)
        return tensor.reshape(x, new_shape)

    @always_inline
    fn forward_gpu(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        # برای Flatten کاری روی GPU لازم نیست؛ همان مسیر O(1) است
        var new_shape = self._compute_new_shape(x._shape)
        return tensor.reshape(x, new_shape)

    fn forward_auto(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        # اگر در پروژه‌ات gpu_available() داری، می‌توانی اینجا از آن استفاده کنی
        # از آن‌جا که Flatten محاسبه‌ای ندارد، تفاوتی نمی‌کند.
        return self.forward_cpu(x)

    fn forward(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        return self.forward_auto(x)

    # --------------------
    # Autograd fast paths
    # --------------------
    fn forward(self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        # x.reshape(ctx, ...) خودش گرادیان برگشتی را مدیریت می‌کند (view)
        var shp = ctx.tape.values[x.id]._shape.copy()
        var new_shape = self._compute_new_shape(shp)
        return x.reshape(ctx, new_shape)

    fn forward_cpu(self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        var shp = ctx.tape.values[x.id]._shape.copy()
        var new_shape = self._compute_new_shape(shp)
        return x.reshape(ctx, new_shape)

    fn forward_gpu(self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        var shp = ctx.tape.values[x.id]._shape.copy()
        var new_shape = self._compute_new_shape(shp)
        return x.reshape(ctx, new_shape)

    fn forward_auto(self, mut ctx: GradContext, x: GradTensor) -> GradTensor:
        # Flatten محاسبه ندارد؛ مسیر CPU/GPU یکسان است
        return self.forward_cpu(ctx, x)

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
    var b_f: tensor.Tensor[Float32]     # [out]
    var s_w: Float32
    var s_x: Float32

    fn __init__(out self, in_features: Int, out_features: Int,
                w_q: tensor.Tensor[Int8], b_f: tensor.Tensor[Float32],
                s_w: Float32, s_x: Float32):
        self.in_features = in_features
        self.out_features = out_features
        self.w_q = w_q.copy()
        self.b_f = b_f.copy()
        self.s_w = s_w
        self.s_x = s_x

    fn forward(self, x_f32: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var x_q = quantize_symmetric_int8(x_f32, self.s_x)
        var y_i32 = matmul_i8_i8_to_i32(x_q, self.w_q.transpose())
        var sf = self.s_x * self.s_w
        return dequantize_i32_to_f32_add_bias(y_i32, sf, self.b_f)
    fn forward(self, mut ctx: GradContext, x_f32: GradTensor) -> GradTensor:
        # 1) raw value
        var x = x_f32.value(ctx)   # Tensor[Float32] (no grad ops below)

        # 2) quantize → int8
        var x_q = quantize_symmetric_int8(x, self.s_x)

        # 3) int8×int8 → int32 (وزنِ ترانهاده/سازگار با کرنل)
        var w_qT = self.w_q.transpose()
        var y_i32 = matmul_i8_i8_to_i32(x_q, w_qT)

        # 4) dequant + bias → float32/64
        var sf = self.s_x * self.s_w
        var y_f = dequantize_i32_to_f32_add_bias(y_i32, sf, self.b_f)   # Tensor[Float32]

        # 5) برگرداندن به‌صورت leaf
        var lid = ctx.tape.add_leaf(y_f)
        return GradTensor(lid, False)
