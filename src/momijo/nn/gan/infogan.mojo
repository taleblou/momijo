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
# Project: momijo.nn.gan
# File: src/momijo/nn/gan/infogan.mojo

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
fn _log1p(y: Float64) -> Float64:
    var x = y
    if x <= -0.999999999999: x = -0.999999999999
    var term = x
    var res = term
    var n = 2.0
    var sign = -1.0
    for i in range(9):
        term = term * x
        res += sign * term / n
        n += 1.0
        sign = -sign
    return res
fn _log(x: Float64) -> Float64:
    return _log1p(x - 1.0)
fn sigmoid(x: Float64) -> Float64:
    var e = _exp(-x)
    return 1.0 / (1.0 + e)
fn tanh_like(x: Float64) -> Float64:
    var e = _exp(-2.0 * x)
    return 2.0 / (1.0 + e) - 1.0
fn act1d(x: List[Float64]) -> List[Float64]:
    var y = List[Float64]()
    for v in x: y.push(tanh_like(v))
    return y
fn softmax1d(logits: List[Float64]) -> List[Float64]:
    var m = _max1d(logits)
    var exps = List[Float64]()
    for v in logits: exps.push(_exp(v - m))
    var s = _sum1d(exps)
    var out = List[Float64]()
    if s == 0.0:
        var n = len(exps)
        var p = 1.0 / Float64(n)
        for i in range(n): out.push(p)
        return out
    for e in exps: out.push(e / s)
    return out
fn bce_with_logits_single(logit: Float64, target_is_real: Bool) -> Float64:
    var p = sigmoid(logit)
    var y = 1.0
    if not target_is_real: y = 0.0
    if p < 1e-12: p = 1e-12
    if p > 1.0 - 1e-12: p = 1.0 - 1.0e-12
    return -(y * _log(p) + (1.0 - y) * _log(1.0 - p))
fn cross_entropy_logits1d(logits: List[Float64], target_index: Int) -> Float64:
    var n = len(logits)
    if n == 0: return 0.0
    var m = _max1d(logits)
    var exps = List[Float64]()
    for v in logits: exps.push(_exp(v - m))
    var s = _sum1d(exps)
    if s == 0.0: return 0.0
    var logZ = _log(s) + m
    if target_index < 0: target_index = 0
    if target_index >= n: target_index = n - 1
    return -(logits[target_index] - logZ)

(# Gaussian negative log-likelihood for continuous code c ~ N(mu, sigma ^ UInt8(2))) & UInt8(0xFF)
# Input: predicted (mu, logvar) and true scalar c
fn gaussian_nll(mu: Float64, logvar: Float64, c: Float64) -> Float64:
    var inv_var = _exp(-logvar)
    var diff = c - mu
    # constant term (0.5 * log(2*pi)) omitted for pedagogy; add if desired
    return 0.5 * (logvar + diff * diff * inv_var)

# --- Linear layer with bias ---
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
            for i in range(in_features):
                row.push(weight)  # deterministic tiny init
            self.W.push(row)
        self.b = List[Float64]()
        for o in range(out_features): self.b.push(0.0)
fn forward(self, x: List[Float64]) -> List[Float64]:
        var y = List[Float64]()
        for o in range(self.out_features):
            var acc = self.b[o]
            for i in range(self.in_features):
                acc += self.W[o][i] * x[i]
            y.push(acc)
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
# --- Generator: z || c -> x ---
struct Generator:
    var l1: Linear
    var l2: Linear
    var total_in: Int
fn __init__(out self, z_dim: Int, cat_dim: Int, cont_dim: Int, hidden_dim: Int, x_dim: Int) -> None:
        self.total_in = z_dim + cat_dim + cont_dim
        self.l1 = Linear(self.total_in, hidden_dim)
        self.l2 = Linear(hidden_dim, x_dim)
fn _concat(self, z: List[Float64], c_cat_onehot: List[Float64], c_cont: List[Float64]) -> List[Float64]:
        var v = List[Float64]()
        for i in range(len(z)): v.push(z[i])
        for i in range(len(c_cat_onehot)): v.push(c_cat_onehot[i])
        for i in range(len(c_cont)): v.push(c_cont[i])
        return v
fn forward(self, z: List[Float64], c_cat_onehot: List[Float64], c_cont: List[Float64]) -> List[Float64]:
        var h = self.l1.forward(self._concat(z, c_cat_onehot, c_cont))
        var a = act1d(h)
        return self.l2.forward(a)
fn __copyinit__(out self, other: Self) -> None:
        self.l1 = other.l1
        self.l2 = other.l2
        self.total_in = other.total_in
fn __moveinit__(out self, deinit other: Self) -> None:
        self.l1 = other.l1
        self.l2 = other.l2
        self.total_in = other.total_in
# --- Discriminator with shared body and two heads: D(x) & Q(c|x) ---
struct DQ:
    var feat: Linear
    var adv: Linear      # -> 1 logit
    var q_cat: Linear    # -> cat_dim logits
    var q_mu: Linear     # -> cont_dim means
    var q_logv: Linear   # -> cont_dim log-variances
    var cat_dim: Int
    var cont_dim: Int
fn __init__(out self, x_dim: Int, hidden_dim: Int, cat_dim: Int, cont_dim: Int) -> None:
        self.feat = Linear(x_dim, hidden_dim)
        self.adv = Linear(hidden_dim, 1)
        self.q_cat = Linear(hidden_dim, cat_dim)
        self.q_mu = Linear(hidden_dim, cont_dim)
        self.q_logv = Linear(hidden_dim, cont_dim)
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim
fn forward(self, x: List[Float64]) -> (Float64, List[Float64], List[Float64], List[Float64]):
        var h = act1d(self.feat.forward(x))
        var logit = self.adv.forward(h)[0]
        var cat_logits = self.q_cat.forward(h)
        var mu = self.q_mu.forward(h)
        var logv = self.q_logv.forward(h)
        return (logit, cat_logits, mu, logv)
fn __copyinit__(out self, other: Self) -> None:
        self.feat = other.feat
        self.adv = other.adv
        self.q_cat = other.q_cat
        self.q_mu = other.q_mu
        self.q_logv = other.q_logv
        self.cat_dim = other.cat_dim
        self.cont_dim = other.cont_dim
fn __moveinit__(out self, deinit other: Self) -> None:
        self.feat = other.feat
        self.adv = other.adv
        self.q_cat = other.q_cat
        self.q_mu = other.q_mu
        self.q_logv = other.q_logv
        self.cat_dim = other.cat_dim
        self.cont_dim = other.cont_dim
# --- InfoGAN wrapper ---
struct InfoGAN:
    var G: Generator
    var DQnet: DQ
    var cat_dim: Int
    var cont_dim: Int
    var lambda_mi: Float64
fn __init__(out self, z_dim: Int, cat_dim: Int, cont_dim: Int, g_hidden: Int, dq_hidden: Int, x_dim: Int, lambda_mi: Float64 = 1.0) -> None:
        self.G = Generator(z_dim, cat_dim, cont_dim, g_hidden, x_dim)
        self.DQnet = DQ(x_dim, dq_hidden, cat_dim, cont_dim)
        self.cat_dim = cat_dim
        self.cont_dim = cont_dim
        self.lambda_mi = lambda_mi
fn one_hot(self, idx: Int) -> List[Float64]:
        var out = List[Float64]()
        for i in range(self.cat_dim):
            if i == idx: out.push(1.0) else: out.push(0.0)
        return out

    # Mutual information lower bound term (variational):
    # L_MI = E[ -log Q(c_cat | x) ] + E[ GaussianNLL(c_cont | mu(x), logvar(x)) ]
fn mi_loss(self, cat_logits: List[Float64], cat_target: Int, mu: List[Float64], logv: List[Float64], c_cont: List[Float64]) -> Float64:
        var ce = cross_entropy_logits1d(cat_logits, cat_target)
        var nll = 0.0
        var k = len(c_cont)
        for i in range(k):
            nll += gaussian_nll(mu[i], logv[i], c_cont[i])
        if k > 0: nll = nll / Float64(k)
        return ce + nll

    # Generator step: fool D and maximize MI
fn gen_step(self, z: List[Float64], c_cat_idx: Int, c_cont: List[Float64]) -> (Float64, Float64):
        var x_fake = self.G.forward(z, self.one_hot(c_cat_idx), c_cont)
        var (logit, cat_logits, mu, logv) = self.DQnet.forward(x_fake)
        var adv = bce_with_logits_single(logit, True)
        var mi = self.mi_loss(cat_logits, c_cat_idx, mu, logv, c_cont)
        return (adv, self.lambda_mi * mi)

    # Discriminator step: GAN loss on real/fake + MI head supervision (optional, common to share feat)
fn dis_step(self, x_real: List[Float64], z: List[Float64], c_cat_idx: Int, c_cont: List[Float64]) -> (Float64, Float64):
        var (d_logit_r, _, _, _) = self.DQnet.forward(x_real)
        var loss_real = bce_with_logits_single(d_logit_r, True)

        var x_fake = self.G.forward(z, self.one_hot(c_cat_idx), c_cont)
        var (d_logit_f, cat_logits_f, mu_f, logv_f) = self.DQnet.forward(x_fake)
        var loss_fake = bce_with_logits_single(d_logit_f, False)

        var mi = self.mi_loss(cat_logits_f, c_cat_idx, mu_f, logv_f, c_cont)
        var d_loss = 0.5 * (loss_real + loss_fake) + self.lambda_mi * mi * 0.0  # usually MI optimized via Q/G, keep 0 here
        return (d_loss, mi)
fn __copyinit__(out self, other: Self) -> None:
        self.G = other.G
        self.DQnet = other.DQnet
        self.cat_dim = other.cat_dim
        self.cont_dim = other.cont_dim
        self.lambda_mi = other.lambda_mi
fn __moveinit__(out self, deinit other: Self) -> None:
        self.G = other.G
        self.DQnet = other.DQnet
        self.cat_dim = other.cat_dim
        self.cont_dim = other.cont_dim
        self.lambda_mi = other.lambda_mi
# --- Smoke test ---
fn _self_test() -> Bool:
    var ok = True

    var z = List[Float64]()
    for i in range(6): z.push(0.3)
    var c_cont = List[Float64](); c_cont.push(-0.5); c_cont.push(0.25)
    var x_real = List[Float64]()
    for i in range(10): x_real.push(1.0)

    var m = InfoGAN(6, 4, 2, 16, 16, 10, 1.0)

    var (g_adv, g_mi) = m.gen_step(z, 2, c_cont)
    ok = ok and (g_adv >= 0.0) and (g_mi >= 0.0)

    var (d_loss, d_mi) = m.dis_step(x_real, z, 1, c_cont)
    ok = ok and (d_loss >= 0.0) and (d_mi >= 0.0)

    # exercise paths a bit more
    var (g_adv2, g_mi2) = m.gen_step(z, 0, c_cont)
    ok = ok and (g_adv2 == g_adv2) and (g_mi2 == g_mi2)

    return ok