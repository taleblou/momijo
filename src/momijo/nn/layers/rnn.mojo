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
# File: src/momijo/nn/layers/rnn.mojo

fn zeros1d(n: Int) -> List[Float64]:
    var y = List[Float64]()
    for i in range(n): y.push(0.0)
    return y
fn zeros2d(r: Int, c: Int) -> List[List[Float64]]:
    var y = List[List[Float64]]()
    for i in range(r):
        var row = List[Float64]()
        for j in range(c): row.push(0.0)
        y.push(row)
    return y
fn add1d(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var n = len(a)
    var y = zeros1d(n)
    for i in range(n): y[i] = a[i] + b[i]
    return y
fn mul1d(a: List[Float64], b: List[Float64]) -> List[Float64]:
    var n = len(a)
    var y = zeros1d(n)
    for i in range(n): y[i] = a[i] * b[i]
    return y
fn matvec(W: List[List[Float64]], x: List[Float64]) -> List[Float64]:
    var out = zeros1d(len(W))
    for r in range(len(W)):
        var s = 0.0
        for c in range(len(W[0])):
            s += W[r][c] * x[c]
        out[r] = s
    return out
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
fn sigmoid(x: Float64) -> Float64:
    var e = _exp(-x)
    return 1.0 / (1.0 + e)
fn tanh_like(x: Float64) -> Float64:
    var e = _exp(-2.0 * x)
    return 2.0 / (1.0 + e) - 1.0
fn act_tanh(v: List[Float64]) -> List[Float64]:
    var y = zeros1d(len(v))
    for i in range(len(v)): y[i] = tanh_like(v[i])
    return y
fn act_sigmoid(v: List[Float64]) -> List[Float64]:
    var y = zeros1d(len(v))
    for i in range(len(v)): y[i] = sigmoid(v[i])
    return y

# --- Linear (for parameter grouping) ---
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
fn forward(self, x: List[Float64]) -> List[Float64]:
        var y = matvec(self.W, x)
        for o in range(self.out_features): y[o] += self.b[o]
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
# --- Vanilla RNN (tanh) cell ---
struct RNNCell:
    var input_size: Int
    var hidden_size: Int
    var Wx: Linear
    var Wh: Linear
fn __init__(out self, input_size: Int, hidden_size: Int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wx = Linear(input_size, hidden_size)
        self.Wh = Linear(hidden_size, hidden_size)
fn init_hidden(self) -> List[Float64]:
        return zeros1d(self.hidden_size)
fn forward(self, x_t: List[Float64], h_prev: List[Float64]) -> List[Float64]:
        var a = add1d(self.Wx.forward(x_t), self.Wh.forward(h_prev))
        return act_tanh(a)
fn __copyinit__(out self, other: Self) -> None:
        self.input_size = other.input_size
        self.hidden_size = other.hidden_size
        self.Wx = other.Wx
        self.Wh = other.Wh
fn __moveinit__(out self, deinit other: Self) -> None:
        self.input_size = other.input_size
        self.hidden_size = other.hidden_size
        self.Wx = other.Wx
        self.Wh = other.Wh
# --- GRU cell ---
struct GRUCell:
    var input_size: Int
    var hidden_size: Int
    # Gates: z, r, n (candidate)
    var Wz: Linear; var Uz: Linear
    var Wr: Linear; var Ur: Linear
    var Wn: Linear; var Un: Linear
fn __init__(out self, input_size: Int, hidden_size: Int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wz = Linear(input_size, hidden_size); self.Uz = Linear(hidden_size, hidden_size)
        self.Wr = Linear(input_size, hidden_size); self.Ur = Linear(hidden_size, hidden_size)
        self.Wn = Linear(input_size, hidden_size); self.Un = Linear(hidden_size, hidden_size)
fn init_hidden(self) -> List[Float64]:
        return zeros1d(self.hidden_size)
fn forward(self, x_t: List[Float64], h_prev: List[Float64]) -> List[Float64]:
        var z = act_sigmoid(add1d(self.Wz.forward(x_t), self.Uz.forward(h_prev)))
        var r = act_sigmoid(add1d(self.Wr.forward(x_t), self.Ur.forward(h_prev)))

        var rh = mul1d(r, h_prev)
        var npre = add1d(self.Wn.forward(x_t), self.Un.forward(rh))
        var nvec = act_tanh(npre)

        var one_minus_z = zeros1d(self.hidden_size)
        for i in range(self.hidden_size): one_minus_z[i] = 1.0 - z[i]
        var left = mul1d(one_minus_z, nvec)
        var right = mul1d(z, h_prev)
        return add1d(left, right)
fn __copyinit__(out self, other: Self) -> None:
        self.input_size = other.input_size
        self.hidden_size = other.hidden_size
        self.Wz = other.Wz
        self.Wr = other.Wr
        self.Wn = other.Wn
fn __moveinit__(out self, deinit other: Self) -> None:
        self.input_size = other.input_size
        self.hidden_size = other.hidden_size
        self.Wz = other.Wz
        self.Wr = other.Wr
        self.Wn = other.Wn
# --- LSTM cell ---
struct LSTMCell:
    var input_size: Int
    var hidden_size: Int
    # Gates: i, f, g, o
    var Wi: Linear; var Ui: Linear
    var Wf: Linear; var Uf: Linear
    var Wg: Linear; var Ug: Linear
    var Wo: Linear; var Uo: Linear
fn __init__(out self, input_size: Int, hidden_size: Int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wi = Linear(input_size, hidden_size); self.Ui = Linear(hidden_size, hidden_size)
        self.Wf = Linear(input_size, hidden_size); self.Uf = Linear(hidden_size, hidden_size)
        self.Wg = Linear(input_size, hidden_size); self.Ug = Linear(hidden_size, hidden_size)
        self.Wo = Linear(input_size, hidden_size); self.Uo = Linear(hidden_size, hidden_size)
fn init_hidden(self) -> (List[Float64], List[Float64]):
        return (zeros1d(self.hidden_size), zeros1d(self.hidden_size))  # (h, c)
fn forward(self, x_t: List[Float64], h_prev: List[Float64], c_prev: List[Float64]) -> (List[Float64], List[Float64]):
        var i = act_sigmoid(add1d(self.Wi.forward(x_t), self.Ui.forward(h_prev)))
        var f = act_sigmoid(add1d(self.Wf.forward(x_t), self.Uf.forward(h_prev)))
        var g = act_tanh(add1d(self.Wg.forward(x_t), self.Ug.forward(h_prev)))
        var o = act_sigmoid(add1d(self.Wo.forward(x_t), self.Uo.forward(h_prev)))
        var c_t = zeros1d(self.hidden_size)
        for k in range(self.hidden_size):
            c_t[k] = f[k] * c_prev[k] + i[k] * g[k]
        var h_t = zeros1d(self.hidden_size)
        for k in range(self.hidden_size):
            h_t[k] = o[k] * tanh_like(c_t[k])
        return (h_t, c_t)
fn __copyinit__(out self, other: Self) -> None:
        self.input_size = other.input_size
        self.hidden_size = other.hidden_size
        self.Wi = other.Wi
        self.Wf = other.Wf
        self.Wg = other.Wg
        self.Wo = other.Wo
fn __moveinit__(out self, deinit other: Self) -> None:
        self.input_size = other.input_size
        self.hidden_size = other.hidden_size
        self.Wi = other.Wi
        self.Wf = other.Wf
        self.Wg = other.Wg
        self.Wo = other.Wo
# --- Sequence wrappers (single-layer) ---
struct RNN:
    var cell: RNNCell
fn __init__(out self, input_size: Int, hidden_size: Int) -> None:
        self.cell = RNNCell(input_size, hidden_size)

    # x: [T, input_size] ; returns (outputs [T, hidden_size], h_last)
fn forward(self, x: List[List[Float64]]) -> (List[List[Float64]], List[Float64]):
        var T = len(x)
        var outs = zeros2d(T, self.cell.hidden_size)
        var h = self.cell.init_hidden()
        for t in range(T):
            h = self.cell.forward(x[t], h)
            outs[t] = h
        return (outs, h)
fn __copyinit__(out self, other: Self) -> None:
        self.cell = other.cell
fn __moveinit__(out self, deinit other: Self) -> None:
        self.cell = other.cell
struct GRU:
    var cell: GRUCell
fn __init__(out self, input_size: Int, hidden_size: Int) -> None:
        self.cell = GRUCell(input_size, hidden_size)
fn forward(self, x: List[List[Float64]]) -> (List[List[Float64]], List[Float64]):
        var T = len(x)
        var outs = zeros2d(T, self.cell.hidden_size)
        var h = self.cell.init_hidden()
        for t in range(T):
            h = self.cell.forward(x[t], h)
            outs[t] = h
        return (outs, h)
fn __copyinit__(out self, other: Self) -> None:
        self.cell = other.cell
fn __moveinit__(out self, deinit other: Self) -> None:
        self.cell = other.cell
struct LSTM:
    var cell: LSTMCell
fn __init__(out self, input_size: Int, hidden_size: Int) -> None:
        self.cell = LSTMCell(input_size, hidden_size)
fn forward(self, x: List[List[Float64]]) -> (List[List[Float64]], (List[Float64], List[Float64])):
        var T = len(x)
        var outs = zeros2d(T, self.cell.hidden_size)
        var (h, c) = self.cell.init_hidden()
        for t in range(T):
            var (h1, c1) = self.cell.forward(x[t], h, c)
            h = h1; c = c1
            outs[t] = h
        return (outs, (h, c))
fn __copyinit__(out self, other: Self) -> None:
        self.cell = other.cell
fn __moveinit__(out self, deinit other: Self) -> None:
        self.cell = other.cell
# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True

    # Build a toy sequence T=4, input=3
    var T = 4
    var D = 3
    var X = zeros2d(T, D)
    for t in range(T):
        for i in range(D):
            X[t][i] = 0.1 * Float64(t + 1) + 0.05 * Float64(i)

    # RNN
    var rnn = RNN(D, 5)
    var (Yr, hr) = rnn.forward(X)
    ok = ok and (len(Yr) == T) and (len(Yr[0]) == 5) and (len(hr) == 5)

    # GRU
    var gru = GRU(D, 6)
    var (Yg, hg) = gru.forward(X)
    ok = ok and (len(Yg) == T) and (len(Yg[0]) == 6) and (len(hg) == 6)

    # LSTM
    var lstm = LSTM(D, 7)
    var (Yl, hc) = lstm.forward(X)
    ok = ok and (len(Yl) == T) and (len(Yl[0]) == 7) and (len(hc[0]) == 7) and (len(hc[1]) == 7)

    return ok