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
from momijo.learn.nn.attention import MultiHeadAttention
from momijo.learn.nn.functional import _apply_hardtanh_inplace
# ----------------------------- Encoder/Decoder layers -------------------------
struct EncoderLayer:
    var self_attn: MultiHeadAttention
    var ff1: Linear
    var ff2: Linear

    fn __init__(out self, d_model: Int, nhead: Int):
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.ff1 = Linear(d_model, 4*d_model)
        self.ff2 = Linear(4*d_model, d_model)

    fn forward(self, src: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var attn = self.self_attn.forward(src, src)  # [T,B,C]
        # simple residual
        var out = tensor.zeros(src.shape())
        var n = out.numel(); var i = 0
        while i < n:
            out._data[i] = src._data[i] + attn._data[i]
            i = i + 1
        # feed-forward + residual
        var TB = src.shape()[0] * src.shape()[1]
        var C = src.shape()[2]
        var x2 = tensor.zeros([TB, C])
        i = 0
        while i < TB*C:
            x2._data[i] = out._data[i]
            i = i + 1
        var h1 = self.ff1.forward(x2)
        _apply_hardtanh_inplace(h1)
        var h2 = self.ff2.forward(h1)
        var out2 = tensor.zeros([src.shape()[0], src.shape()[1], C])
        i = 0
        while i < TB*C:
            out2._data[i] = out._data[i] + h2._data[i]
            i = i + 1
        return out2.copy()

struct DecoderLayer:
    var self_attn: MultiHeadAttention
    var cross_attn: MultiHeadAttention
    var ff1: Linear
    var ff2: Linear

    fn __init__(out self, d_model: Int, nhead: Int):
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.ff1 = Linear(d_model, 4*d_model)
        self.ff2 = Linear(4*d_model, d_model)

    fn forward(self, tgt: tensor.Tensor[Float64], memory: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var a1 = self.self_attn.forward(tgt, tgt)
        var r1 = tensor.zeros(tgt.shape())
        var i = 0; var n = r1.numel()
        while i < n:
            r1._data[i] = tgt._data[i] + a1._data[i]
            i = i + 1

        var a2 = self.cross_attn.forward(r1, memory)
        var r2 = tensor.zeros(tgt.shape())
        i = 0
        while i < n:
            r2._data[i] = r1._data[i] + a2._data[i]
            i = i + 1

        var TB = tgt.shape()[0] * tgt.shape()[1]; var C = tgt.shape()[2]
        var x2 = tensor.zeros([TB, C]); i = 0
        while i < TB*C:
            x2._data[i] = r2._data[i]
            i = i + 1

        var h1 = self.ff1.forward(x2)
        _apply_hardtanh_inplace(h1)
        var h2 = self.ff2.forward(h1)

        var out = tensor.zeros([tgt.shape()[0], tgt.shape()[1], C]); i = 0
        while i < TB*C:
            out._data[i] = r2._data[i] + h2._data[i]
            i = i + 1
        return out.copy()

# ----------------------------- Transformer (1 enc / 1 dec) --------------------
struct TinyTransformer:
    var d_model: Int
    var nhead: Int
    var enc: EncoderLayer
    var dec: DecoderLayer

    fn __init__(out self, d_model: Int, nhead: Int):
        self.d_model = d_model
        self.nhead = nhead
        self.enc = EncoderLayer(d_model, nhead)
        self.dec = DecoderLayer(d_model, nhead)

    # src: [T,B,C], tgt: [T2,B,C]
    fn forward(self, src: tensor.Tensor[Float64], tgt: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var memory = self.enc.forward(src)
        var out = self.dec.forward(tgt, memory)
        return out.copy()
 