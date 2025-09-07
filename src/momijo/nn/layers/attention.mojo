# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.nn.layers
# File: src/momijo/nn/layers/attention.mojo

fn _max(a: Float64, b: Float64) -> Float64:
    if a >= b: return a
    return b
fn _sum1d(xs: List[Float64]) -> Float64:
    var s = 0.0
    for v in xs: s += v
    return s
fn _max1d(xs: List[Float64]) -> Float64:
    var m = -1.7976931348623157e308
    for v in xs:
        if v > m: m = v
    return m
fn _exp(x: Float64) -> Float64:
    # truncated series to avoid heavy deps
    var term = 1.0
    var sum = 1.0
    var n = 1
    var k = 1.0
    while n <= 12:
        term *= x / k
        sum += term
        n += 1
        k += 1.0
    return sum
fn softmax1d(logits: List[Float64]) -> List[Float64]:
    var m = _max1d(logits)
    var exps = List[Float64]()
    for v in logits: exps.push(_exp(v - m))
    var s = _sum1d(exps)
    var out = List[Float64]()
    if s == 0.0:
        var n = len(exps)
        if n == 0: return out
        var p = 1.0 / Float64(n)
        for i in range(n): out.push(p)
        return out
    for e in exps: out.push(e / s)
    return out

# matrix utilities for small 2D Lists (row-major: [rows][cols])
fn zeros2d(rows: Int, cols: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(rows):
        var row = List[Float64]()
        for j in range(cols): row.push(0.0)
        y.push(row)
    return y
fn matmul2d(a: List[List[Float64]], b: List[List[Float64]]) -> List[List[Float64]]:
    var m = len(a)
    if m == 0: return List[List[Float64]]()
    var k = len(a[0])
    var kb = len(b)
    var n = 0
    if kb > 0: n = len(b[0])
    if k != kb: 
        # return zeros to keep deterministic without exceptions
        return zeros2d(m, n)
    var y = zeros2d(m, n)
    for i in range(m):
        for j in range(n):
            var acc = 0.0
            for t in range(k):
                acc += a[i][t] * b[t][j]
            y[i][j] = acc
    return y
fn transpose2d(x: List[List[Float64]]) -> List[List[Float64]]:
    var m = len(x)
    if m == 0: return List[List[Float64]]()
    var n = len(x[0])
    var y = zeros2d(n, m)
    for i in range(m):
        for j in range(n):
            y[j][i] = x[i][j]
    return y
fn scale2d(x: List[List[Float64]], s: Float64) -> List[List[Float64]]:
    var m = len(x)
    var n = 0
    if m > 0: n = len(x[0])
    var y = zeros2d(m, n)
    for i in range(m):
        for j in range(n):
            y[i][j] = x[i][j] * s
    return y
fn row_softmax2d(x: List[List[Float64]]) -> List[List[Float64]]:
    var m = len(x)
    var n = 0
    if m > 0: n = len(x[0])
    var y = zeros2d(m, n)
    for i in range(m):
        var probs = softmax1d(x[i])
        for j in range(n): y[i][j] = probs[j]
    return y

# --- Scaled Dot-Product Attention (single head) ---
# Q: [Lq, d], K: [Lk, d], V: [Lk, dv] -> Out: [Lq, dv], Attn: [Lq, Lk]
fn scaled_dot_product_attention(Q: List[List[Float64]], K: List[List[Float64]], V: List[List[Float64]]) -> (List[List[Float64]], List[List[Float64]]):
    var d = 0
    if len(Q) > 0: d = len(Q[0])
    if d == 0:
        return (List[List[Float64]](), List[List[Float64]]())
    var scores = matmul2d(Q, transpose2d(K))                 # [Lq, Lk]
    var scale = 1.0 / (Float64(d) ** 0.5)                    # 1/sqrt(d)
    var scaled = scale2d(scores, scale)
    var attn = row_softmax2d(scaled)                         # [Lq, Lk]
    var out = matmul2d(attn, V)                              # [Lq, dv]
    return (out, attn)

# --- Linear layer ---
struct Linear:
    var in_features: Int
    var out_features: Int
    var W: List[List[Float64]]  # [out, in]
    var b: List[Float64]        # [out]
fn __init__(out self, in_features: Int, out_features: Int, weight: Float64 = 0.01) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.W = List[List[Float64]]()
        for o in range(out_features):
            var row = List[Float64]()
            for i in range(in_features): row.push(weight)
            self.W.push(row)
        self.b = List[Float64]()
        for o in range(out_features): self.b.push(0.0)
fn forward(self, x: List[List[Float64]]) -> List[List[Float64]]:
        # x: [L, in] ; returns [L, out]  (apply affine row-wise)
        var L = len(x)
        var out = zeros2d(L, self.out_features)
        # Convert W: [out,in] to [in,out] to reuse matmul2d(x, W^T)
        var WT = transpose2d(self.W)  # [in, out]
        out = matmul2d(x, WT)         # [L, out]
        for i in range(L):
            for o in range(self.out_features):
                out[i][o] += self.b[o]
        return out
fn __copyinit__(out self, other: Self) -> None:
        self.in_features = other.in_features
        self.out_features = other.out_features
        self.W = other.W
        self.b = other.b
fn __moveinit__(out self, deinit other: Self) -> None:
        self.in_features = other.in_features
        self.out_features = other.out_features
        self.W = other.W
        self.b = other.b
# --- Multi-Head Attention (list-based scaffold) ---
struct MultiHeadAttention:
    var d_model: Int
    var num_heads: Int
    var d_k: Int
    var d_v: Int
    var W_q: Linear
    var W_k: Linear
    var W_v: Linear
    var W_o: Linear
fn __init__(out self, d_model: Int, num_heads: Int) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        if num_heads <= 0: self.num_heads = 1
        # split evenly for pedagogy
        self.d_k = d_model / self.num_heads
        self.d_v = d_model / self.num_heads
        if self.d_k <= 0: self.d_k = 1
        if self.d_v <= 0: self.d_v = 1
        self.W_q = Linear(d_model, self.num_heads * self.d_k)
        self.W_k = Linear(d_model, self.num_heads * self.d_k)
        self.W_v = Linear(d_model, self.num_heads * self.d_v)
        self.W_o = Linear(self.num_heads * self.d_v, d_model)
fn _split_heads(self, X: List[List[Float64]], head_dim: Int) -> List[List[List[Float64]]]:
        # X: [L, num_heads*head_dim] -> list of heads: H * [L, head_dim]
        var L = len(X)
        var H = self.num_heads
        var heads = List[List[List[Float64]]]()
        var offset = 0
        for h in range(H):
            var Hm = zeros2d(L, head_dim)
            for i in range(L):
                for j in range(head_dim):
                    Hm[i][j] = X[i][offset + j]
            offset += head_dim
            heads.push(Hm)
        return heads
fn _concat_heads(self, heads: List[List[List[Float64]]]) -> List[List[Float64]]:
        # heads: H * [L, dv] -> [L, H*dv]
        var H = len(heads)
        if H == 0: return List[List[Float64]]()
        var L = len(heads[0])
        var dv = 0
        if L > 0: dv = len(heads[0][0])
        var Y = zeros2d(L, H * dv)
        for h in range(H):
            for i in range(L):
                for j in range(dv):
                    Y[i][h*dv + j] = heads[h][i][j]
        return Y

    # Forward: q,k,v shapes are [L, d_model]
fn forward(self, q: List[List[Float64]], k: List[List[Float64]], v: List[List[Float64]]) -> (List[List[Float64]], List[List[List[Float64]]]):
        var Q = self.W_q.forward(q)  # [L, H*d_k]
        var K = self.W_k.forward(k)  # [L, H*d_k]
        var V = self.W_v.forward(v)  # [L, H*d_v]

        var Qh = self._split_heads(Q, self.d_k)
        var Kh = self._split_heads(K, self.d_k)
        var Vh = self._split_heads(V, self.d_v)

        var heads_out = List[List[List[Float64]]]()
        var heads_attn = List[List[List[Float64]]]()

        for h in range(self.num_heads):
            var (o, a) = scaled_dot_product_attention(Qh[h], Kh[h], Vh[h])
            heads_out.push(o)
            heads_attn.push(a)

        var concat = self._concat_heads(heads_out)  # [L, H*d_v]
        var out = self.W_o.forward(concat)          # [L, d_model]
        return (out, heads_attn)
fn __copyinit__(out self, other: Self) -> None:
        self.d_model = other.d_model
        self.num_heads = other.num_heads
        self.d_k = other.d_k
        self.d_v = other.d_v
        self.W_q = other.W_q
        self.W_k = other.W_k
        self.W_v = other.W_v
        self.W_o = other.W_o
fn __moveinit__(out self, deinit other: Self) -> None:
        self.d_model = other.d_model
        self.num_heads = other.num_heads
        self.d_k = other.d_k
        self.d_v = other.d_v
        self.W_q = other.W_q
        self.W_k = other.W_k
        self.W_v = other.W_v
        self.W_o = other.W_o
# --- Smoke tests ---
fn _self_test() -> Bool:
    var ok = True

    # tiny sequence with d_model=8, L=4
    var L = 4
    var dmodel = 8
    var q = zeros2d(L, dmodel)
    var k = zeros2d(L, dmodel)
    var v = zeros2d(L, dmodel)
    for i in range(L):
        for j in range(dmodel):
            q[i][j] = 0.1 * Float64(j + 1)
            k[i][j] = 0.2 * Float64(j + 1)
            v[i][j] = 0.3 * Float64(j + 1)

    # single-head attention via scaled dot-product
    var (o1, a1) = scaled_dot_product_attention(q, k, v)
    ok = ok and (len(o1) == L) and (len(a1) == L)

    # multi-head attention
    var mha = MultiHeadAttention(dmodel, 2)
    var (o2, attns) = mha.forward(q, k, v)
    ok = ok and (len(o2) == L) and (len(o2[0]) == dmodel) and (len(attns) == 2)

    return ok