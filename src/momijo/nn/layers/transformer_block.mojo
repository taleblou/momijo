# Project:      Momijo
# Module:       src.momijo.nn.layers.transformer_block
# File:         transformer_block.mojo
# Path:         src/momijo/nn/layers/transformer_block.mojo
#
# Description:  Neural-network utilities for Momijo integrating with tensors,
#               optimizers, and training/evaluation loops.
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
#   - Structs: Linear, LayerNorm, MultiHeadAttention, FeedForward, TransformerBlock
#   - Key functions: zeros1d, ones1d, zeros2d, add2d, _exp, softmax1d, matmul2d, transpose2d ...
#   - Uses generic functions/types with explicit trait bounds.


fn zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y
fn ones1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(1.0)
    return y
fn zeros2d(r: Int, c: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(r):
        var row = List[Float64]()
        for j in range(c): row.push(0.0)
        y.push(row)
    return y
fn add2d(a: List[List[Float64]], b: List[List[Float64]]) -> List[List[Float64]]:
    var L = len(a)
    if L == 0: return a
    var D = len(a[0])
    var y = zeros2d(L, D)
    for i in range(L):
        for j in range(D):
            y[i][j] = a[i][j] + b[i][j]
    return y
fn _exp(x: Float64) -> Float64:
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
    var m = -1.7976931348623157e308
    for v in logits:
        if v > m: m = v
    var exps = List[Float64]()
    for v in logits: exps.push(_exp(v - m))
    var s = 0.0
    for e in exps: s += e
    var out = List[Float64]()
    if s == 0.0:
        var n = len(exps)
        if n == 0: return out
        var p = 1.0 / Float64(n)
        for i in range(n): out.push(p)
        return out
    for e in exps: out.push(e / s)
    return out
fn matmul2d(a: List[List[Float64]], b: List[List[Float64]]) -> List[List[Float64]]:
    var m = len(a)
    if m == 0: return List[List[Float64]]()
    var k = len(a[0])
    var kb = len(b)
    var n = 0
    if kb > 0: n = len(b[0])
    if k != kb: return zeros2d(m, n)
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
fn relu_row(x: List[Float64]) -> List[Float64]:
    var D = len(x)
    var y = zeros1d(D)
    for i in range(D):
        var v = x[i]
        if v < 0.0: v = 0.0
        y[i] = v
    return y
fn relu2d(x: List[List[Float64]]) -> List[List[Float64]]:
    var L = len(x)
    if L == 0: return x
    var D = len(x[0])
    var y = zeros2d(L, D)
    for i in range(L):
        y[i] = relu_row(x[i])
    return y

# --- Deterministic inverted Dropout ---
fn _clamp01(x: Float64) -> Float64:
    var y = x
    if y < 0.0: y = 0.0
    if y > 1.0: y = 1.0
    return y
fn _rand01_idx(i: Int, j: Int, seed: Int) -> Float64:
    var m = 9973
    var v = ((i + 1) * 1103515245 + (j + 1) * 12345 + seed * 2654435761) % m
    if v < 0: v = v + m
    return Float64(v) / Float64(m - 1)
fn dropout2d(x: List[List[Float64]], p: Float64, training: Bool, seed: Int) -> List[List[Float64]]:
    var pp = _clamp01(p)
    if not training or pp <= 0.0:
        return x
    if pp >= 1.0: pp = 0.999999
    var keep_scale = 1.0 / (1.0 - pp)
    var L = len(x)
    if L == 0: return x
    var D = len(x[0])
    var y = zeros2d(L, D)
    for i in range(L):
        for j in range(D):
            var r = _rand01_idx(i, j, seed)
            if r < pp:
                y[i][j] = 0.0
            else:
                y[i][j] = x[i][j] * keep_scale
    return y

# --- Linear ---
struct Linear:
    var in_features: Int
    var out_features: Int
    var W: List[List[Float64]]  # [out, in]
    var b: List[Float64]        # [out]
fn __init__(out self, in_features: Int, out_features: Int, w_init: Float64 = 0.01) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.W = zeros2d(out_features, in_features)
        self.b = zeros1d(out_features)
        for o in range(out_features):
            for i in range(in_features):
                self.W[o][i] = w_init
fn forward_rows(self, x: List[List[Float64]]) -> List[List[Float64]]:
        var L = len(x)
        if L == 0: return x
        var y = zeros2d(L, self.out_features)
        var WT = transpose2d(self.W)   # [in,out]
        y = matmul2d(x, WT)           # [L,out]
        for i in range(L):
            for o in range(self.out_features):
                y[i][o] += self.b[o]
        return y
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
# --- LayerNorm over last dim ---
struct LayerNorm:
    var D: Int
    var eps: Float64
    var gamma: List[Float64]
    var beta: List[Float64]
fn __init__(out self, D: Int, eps: Float64 = 1e-5) -> None:
        self.D = D
        self.eps = eps
        self.gamma = ones1d(D)
        self.beta = zeros1d(D)
fn forward(self, x: List[List[Float64]]) -> List[List[Float64]]:
        var L = len(x)
        if L == 0: return x
        var y = zeros2d(L, self.D)
        for i in range(L):
            var m = 0.0
            for j in range(self.D): m += x[i][j]
            m /= Float64(self.D)
            var v = 0.0
            for j in range(self.D):
                var d = x[i][j] - m
                v += d * d
            v /= Float64(self.D)
            # sqrt via two Newton steps
            var denom = v + self.eps
            var s = denom
            s = 0.5 * (s + denom / s)
            s = 0.5 * (s + denom / s)
            if s == 0.0: s = 1.0
            for j in range(self.D):
                var xhat = (x[i][j] - m) / s
                y[i][j] = self.gamma[j] * xhat + self.beta[j]
        return y
fn __copyinit__(out self, other: Self) -> None:
        self.D = other.D
        self.eps = other.eps
        self.gamma = other.gamma
        self.beta = other.beta
fn __moveinit__(out self, deinit other: Self) -> None:
        self.D = other.D
        self.eps = other.eps
        self.gamma = other.gamma
        self.beta = other.beta
# --- Scaled Dot-Product Attention (single head) ---
fn scaled_dot_product_attention(Q: List[List[Float64]], K: List[List[Float64]], V: List[List[Float64]]) -> (List[List[Float64]], List[List[Float64]]):
    var d = 0
    if len(Q) > 0: d = len(Q[0])
    if d == 0:
        return (List[List[Float64]](), List[List[Float64]]())
    var scores = matmul2d(Q, transpose2d(K))         # [Lq, Lk]
    var scale = 1.0 / (Float64(d) ** 0.5)            # 1/sqrt(d)
    var scaled = scale2d(scores, scale)
    var Lq = len(scaled)
    var Lk = 0
    if Lq > 0: Lk = len(scaled[0])
    var attn = zeros2d(Lq, Lk)
    for i in range(Lq):
        attn[i] = softmax1d(scaled[i])
    var out = matmul2d(attn, V)                      # [Lq, dv]
    return (out, attn)

# --- Multi-Head Attention ---
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
        self.num_heads = num_heads if num_heads > 0 else 1
        self.d_k = d_model / self.num_heads
        self.d_v = d_model / self.num_heads
        if self.d_k <= 0: self.d_k = 1
        if self.d_v <= 0: self.d_v = 1
        self.W_q = Linear(d_model, self.num_heads * self.d_k)
        self.W_k = Linear(d_model, self.num_heads * self.d_k)
        self.W_v = Linear(d_model, self.num_heads * self.d_v)
        self.W_o = Linear(self.num_heads * self.d_v, d_model)
fn _split_heads(self, X: List[List[Float64]], head_dim: Int) -> List[List[List[Float64]]]:
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

    # Self-attention forward: x as q=k=v with shape [L, d_model]
fn forward(self, x: List[List[Float64]]) -> (List[List[Float64]], List[List[List[Float64]]]):
        var Q = self.W_q.forward_rows(x)  # [L, H*d_k]
        var K = self.W_k.forward_rows(x)  # [L, H*d_k]
        var V = self.W_v.forward_rows(x)  # [L, H*d_v]
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
        var out = self.W_o.forward_rows(concat)     # [L, d_model]
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
# --- Position-wise FeedForward ---
struct FeedForward:
    var l1: Linear
    var l2: Linear
fn __init__(out self, d_model: Int, d_ff: Int) -> None:
        self.l1 = Linear(d_model, d_ff)
        self.l2 = Linear(d_ff, d_model)
fn forward(self, x: List[List[Float64]]) -> List[List[Float64]]:
        var h = self.l1.forward_rows(x)
        var a = relu2d(h)
        return self.l2.forward_rows(a)
fn __copyinit__(out self, other: Self) -> None:
        self.l1 = other.l1
        self.l2 = other.l2
fn __moveinit__(out self, deinit other: Self) -> None:
        self.l1 = other.l1
        self.l2 = other.l2
# --- Transformer Encoder Block ---
struct TransformerBlock:
    var d_model: Int
    var n_heads: Int
    var d_ff: Int
    var dropout_p: Float64
    var training: Bool
    var seed: Int

    var ln1: LayerNorm
    var ln2: LayerNorm
    var mha: MultiHeadAttention
    var ffn: FeedForward
fn __init__(out self, d_model: Int, n_heads: Int, d_ff: Int, dropout_p: Float64 = 0.1, training: Bool = True, seed: Int = 123) -> None:
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_p = dropout_p
        self.training = training
        self.seed = seed
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
fn train_mode(mut self) -> None:
        self.training = True
fn eval_mode(mut self) -> None:
        self.training = False

    # Forward: x[L, d_model] -> (y[L, d_model], attn_per_head)
fn forward(mut self, x: List[List[Float64]]) -> (List[List[Float64]], List[List[List[Float64]]]):
        # Pre-LN
        var x1 = self.ln1.forward(x)
        var (attn_out, attns) = self.mha.forward(x1)
        var attn_out_do = dropout2d(attn_out, self.dropout_p, self.training, self.seed)
        var x_res1 = add2d(x, attn_out_do)

        var x2 = self.ln2.forward(x_res1)
        var ffn_out = self.ffn.forward(x2)
        var ffn_out_do = dropout2d(ffn_out, self.dropout_p, self.training, self.seed + 7)
        var y = add2d(x_res1, ffn_out_do)
        return (y, attns)
fn __copyinit__(out self, other: Self) -> None:
        self.d_model = other.d_model
        self.n_heads = other.n_heads
        self.d_ff = other.d_ff
        self.dropout_p = other.dropout_p
        self.training = other.training
        self.seed = other.seed
        self.ln1 = other.ln1
        self.ln2 = other.ln2
        self.mha = other.mha
        self.ffn = other.ffn
fn __moveinit__(out self, deinit other: Self) -> None:
        self.d_model = other.d_model
        self.n_heads = other.n_heads
        self.d_ff = other.d_ff
        self.dropout_p = other.dropout_p
        self.training = other.training
        self.seed = other.seed
        self.ln1 = other.ln1
        self.ln2 = other.ln2
        self.mha = other.mha
        self.ffn = other.ffn
# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True

    var L = 6
    var D = 8
    var x = zeros2d(L, D)
    for i in range(L):
        for j in range(D):
            x[i][j] = 0.1 * Float64(i + 1) + 0.01 * Float64(j)

    var blk = TransformerBlock(D, 2, 16, 0.2, True, 42)
    var (y, attns) = blk.forward(x)
    ok = ok and (len(y) == L) and (len(y[0]) == D) and (len(attns) == 2)

    blk.eval_mode()
    var (y2, att2) = blk.forward(x)
    ok = ok and (len(y2) == L) and (len(att2) == 2)

    return ok