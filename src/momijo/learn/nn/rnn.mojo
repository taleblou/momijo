# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.nn.pooling
# File:         src/momijo/learn/nn/rnn.mojo
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

from momijo.tensor import tensor
from momijo.learn.nn.layers import Linear
from momijo.learn.nn.functional import _hardsigmoid, _hardtanh
# ----------------------------- GRU (1-layer, batch_first) ---------------------
struct GRU:
    var in_size: Int
    var hid_size: Int
    # x-projections
    var Wxz: Linear
    var Wxr: Linear
    var Wxn: Linear
    # h-projections
    var Whz: Linear
    var Whr: Linear
    var Whn: Linear

    fn __init__(out self, input_size: Int, hidden_size: Int):
        self.in_size = input_size
        self.hid_size = hidden_size
        self.Wxz = Linear(input_size, hidden_size)
        self.Wxr = Linear(input_size, hidden_size)
        self.Wxn = Linear(input_size, hidden_size)
        self.Whz = Linear(hidden_size, hidden_size)
        self.Whr = Linear(hidden_size, hidden_size)
        self.Whn = Linear(hidden_size, hidden_size)

    # x: [B,T,C_in], h0 optional None -> zeros
    fn forward(self, x: tensor.Tensor[Float64]) -> (tensor.Tensor[Float64], tensor.Tensor[Float64]):
        var shp = x.shape()
        var B = shp[0]; var T = shp[1]; var C = shp[2]
        if C != self.in_size:
            return (x.copy(), tensor.zeros([1, B, self.hid_size]))

        var y = tensor.zeros([B, T, self.hid_size])
        var h = tensor.zeros([B, self.hid_size])  # current hidden

        var t = 0
        while t < T:
            # xt: [B, C]
            var xt = tensor.zeros([B, C])
            # slice copy
            var b = 0
            while b < B:
                var c = 0
                while c < C:
                    xt._data[b*C + c] = x._data[(b*T + t)*C + c]
                    c = c + 1
                b = b + 1

            # z = sigmoid(xWz + hUz + bz)
            var xz = self.Wxz.forward(xt)
            var hz = self.Whz.forward(h)
            var z = tensor.zeros([B, self.hid_size])
            var i = 0
            while i < z.numel():
                z._data[i] = _hardsigmoid(xz._data[i] + hz._data[i])
                i = i + 1

            # r = sigmoid(xWr + hUr + br)
            var xr = self.Wxr.forward(xt)
            var hr = self.Whr.forward(h)
            var r = tensor.zeros([B, self.hid_size])
            i = 0
            while i < r.numel():
                r._data[i] = _hardsigmoid(xr._data[i] + hr._data[i])
                i = i + 1

            # n̂ = tanh(xWn + (r ⊙ h)Un + bn)
            var xn = self.Wxn.forward(xt)
            var rh = tensor.zeros([B, self.hid_size])
            i = 0
            while i < rh.numel():
                rh._data[i] = r._data[i] * h._data[i]
                i = i + 1
            var hn = self.Whn.forward(rh)
            var n_hat = tensor.zeros([B, self.hid_size])
            i = 0
            while i < n_hat.numel():
                n_hat._data[i] = _hardtanh(xn._data[i] + hn._data[i])
                i = i + 1

            # h' = (1 - z) ⊙ n̂ + z ⊙ h
            var h_new = tensor.zeros([B, self.hid_size])
            i = 0
            while i < h_new.numel():
                var zt = z._data[i]
                var ht = h._data[i]
                var nt = n_hat._data[i]
                h_new._data[i] = (1.0 - zt) * nt + zt * ht
                i = i + 1

            # write y[:,t,:] = h'
            b = 0
            while b < B:
                var j = 0
                while j < self.hid_size:
                    y._data[(b*T + t)*self.hid_size + j] = h_new._data[b*self.hid_size + j]
                    j = j + 1
                b = b + 1

            h = h_new.copy()
            t = t + 1

        # h_out: [1,B,H]
        var h_out = tensor.zeros([1, B, self.hid_size])
        var bi = 0
        while bi < B:
            var j = 0
            while j < self.hid_size:
                h_out._data[(0*B + bi)*self.hid_size + j] = h._data[bi*self.hid_size + j]
                j = j + 1
            bi = bi + 1

        return (y.copy(), h_out.copy())
