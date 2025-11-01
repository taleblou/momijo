# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.nn.activations
# File:         src/momijo/learn/nn/activations.mojo
#
# Description:
#   Activation functions for Momijo Learn.
#   - Scalar & List APIs (backend-free) with stable approximations.
#   - Tensor APIs (elementwise + stable softmax/log_softmax along axis).
#   - GELU (tanh-approx), SiLU/Swish, hard variants, shrink/softplus/softsign.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor
from momijo.learn.nn.layers import Linear
from momijo.learn.nn.functional import _rowwise_relu_normalize



@always_inline
fn sqrt64(x: Float64) -> Float64:
    var v = x
    if v <= 0.0:
        return 0.0
    var g = v
    if g < 1.0:
        g = 1.0
    var i = 0
    while i < 6:
        g = 0.5 * (g + v / g)
        i += 1
    return g
    
 
# ----------------------------- Multi-Head Attention ---------------------------
struct MultiHeadAttention:
    var d_model: Int
    var nhead: Int
    var d_head: Int
    var Wq: Linear
    var Wk: Linear
    var Wv: Linear
    var Wo: Linear

    fn __init__(out self, d_model: Int, nhead: Int):
        #assert(d_model % nhead == 0)
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.Wq = Linear(d_model, d_model)
        self.Wk = Linear(d_model, d_model)
        self.Wv = Linear(d_model, d_model)
        self.Wo = Linear(d_model, d_model)

    # x: [T,B,C], y: [S,B,C]  -> attend y with queries from x (generic; encoder self-attn when x=y)
    fn forward(self, x: tensor.Tensor[Float64], y: tensor.Tensor[Float64]) -> tensor.Tensor[Float64]:
        var Tx = x.shape()[0]; var B = x.shape()[1]; var C = x.shape()[2]
        var Ty = y.shape()[0]
        var d_h = self.d_head
        # flatten time+batch for Linear: [T*B, C]
        var xb = tensor.zeros([Tx*B, C]); var yb = tensor.zeros([Ty*B, C])
        var i = 0
        while i < Tx*B:
            var c = 0
            while c < C:
                xb._data[i*C + c] = x._data[i*C + c]
                c = c + 1
            i = i + 1
        i = 0
        while i < Ty*B:
            var c2 = 0
            while c2 < C:
                yb._data[i*C + c2] = y._data[i*C + c2]
                c2 = c2 + 1
            i = i + 1

        var Q = self.Wq.forward(xb)  # [Tx*B, C]
        var K = self.Wk.forward(yb)  # [Ty*B, C]
        var V = self.Wv.forward(yb)  # [Ty*B, C]

        # reshape to [B, T, nhead, d_h] by-view copy
        var q = tensor.zeros([B, Tx, self.nhead, d_h])
        var k = tensor.zeros([B, Ty, self.nhead, d_h])
        var v = tensor.zeros([B, Ty, self.nhead, d_h])
        # fill
        var b = 0
        while b < B:
            var t = 0
            while t < Tx:
                var h = 0
                while h < self.nhead:
                    var j = 0
                    while j < d_h:
                        q._data[(((b*Tx + t)*self.nhead + h)*d_h + j)] = Q._data[(t + b*Tx)*C + (h*d_h + j)]
                        j = j + 1
                    h = h + 1
                t = t + 1
            b = b + 1
        b = 0
        while b < B:
            var t2 = 0
            while t2 < Ty:
                var h2 = 0
                while h2 < self.nhead:
                    var j2 = 0
                    while j2 < d_h:
                        var idx = (((b*Ty + t2)*self.nhead + h2)*d_h + j2)
                        k._data[idx] = K._data[(t2 + b*Ty)*C + (h2*d_h + j2)]
                        v._data[idx] = V._data[(t2 + b*Ty)*C + (h2*d_h + j2)]
                        j2 = j2 + 1
                    h2 = h2 + 1
                t2 = t2 + 1
            b = b + 1

        # scaled dot-product attention per head
        var out = tensor.zeros([B, Tx, self.nhead, d_h])
        b = 0
        while b < B:
            var h3 = 0
            while h3 < self.nhead:
                # scores S = Q * K^T ; Q:[Tx,d_h], K:[Ty,d_h] -> [Tx,Ty]
                var tq = 0
                while tq < Tx:
                    # build row scores for time tq
                    var row = List[Float64]()
                    var tk = 0
                    while tk < Ty:
                        var s = 0.0
                        var j3 = 0
                        while j3 < d_h:
                            var qv = q._data[(((b*Tx + tq)*self.nhead + h3)*d_h + j3)]
                            var kv = k._data[(((b*Ty + tk)*self.nhead + h3)*d_h + j3)]
                            s = s + qv * kv
                            j3 = j3 + 1
                        # scale by sqrt(d_h)
                        s = s / sqrt64(d_h)  # if sqrt not available in your math, drop scaling
                        row.append(s)
                        tk = tk + 1
                    # normalize row with ReLU-normalized "softmax"
                    _rowwise_relu_normalize(row)

                    # weighted sum: sum_j row[j] * V[j]
                    var jv = 0
                    while jv < d_h:
                        var acc = 0.0
                        tk = 0
                        while tk < Ty:
                            var w = row[tk]
                            var vv = v._data[(((b*Ty + tk)*self.nhead + h3)*d_h + jv)]
                            acc = acc + w * vv
                            tk = tk + 1
                        out._data[(((b*Tx + tq)*self.nhead + h3)*d_h + jv)] = acc
                        jv = jv + 1

                    tq = tq + 1
                h3 = h3 + 1
            b = b + 1

        # concat heads -> [B, Tx, C] then project with Wo
        var concat = tensor.zeros([B*Tx, C])
        b = 0
        while b < B:
            var t3 = 0
            while t3 < Tx:
                var h4 = 0
                while h4 < self.nhead:
                    var j4 = 0
                    while j4 < d_h:
                        concat._data[(t3 + b*Tx)*C + (h4*d_h + j4)] =
                            out._data[(((b*Tx + t3)*self.nhead + h4)*d_h + j4)]
                        j4 = j4 + 1
                    h4 = h4 + 1
                t3 = t3 + 1
            b = b + 1

        var merged = self.Wo.forward(concat)  # [B*Tx, C]

        # reshape back to [Tx,B,C]
        var z = tensor.zeros([Tx, B, C])
        var ti = 0
        while ti < Tx:
            var bi2 = 0
            while bi2 < B:
                var ci = 0
                while ci < C:
                    z._data[(ti*B + bi2)*C + ci] = merged._data[(ti + bi2*Tx)*C + ci]
                    ci = ci + 1
                bi2 = bi2 + 1
            ti = ti + 1
        return z.copy()

