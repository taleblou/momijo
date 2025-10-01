# Project:      Momijo
# Module:       learn.prob.distributions
# File:         prob/distributions.mojo
# Path:         src/momijo/learn/prob/distributions.mojo
#
# Description:  Probability distributions for Momijo Learn: Uniform, Bernoulli,
#               Binomial, Categorical, Normal. Includes a lightweight RNG and
#               a small math shim (ln, sqrt) to avoid external dependencies.
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
#   - Types: RNG, Uniform, Bernoulli, Binomial, Categorical, Normal
#   - Key fns: sample(n, seed), mean(), var(), log_prob(x)
#   - Normal sampling uses CLT (12 uniforms - 6) ≈ N(0,1)  → scaled to (μ, σ^2)
#   - Binomial.log_prob uses Stirling approximation for ln(n!), numerically stable.
#   - Categorical samples indices in [0..K-1]; mean/var = NaN (not defined).

from collections.list import List

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------
var NEG_INF: Float64 = -1.0e300
var LN2:     Float64 = 0.6931471805599453
var LN2PI:   Float64 = 1.8378770664093453      # ln(2π)
var PI:      Float64 = 3.141592653589793

# -----------------------------------------------------------------------------
# Minimal math shim (no external dependency)
# -----------------------------------------------------------------------------
# ln(x):  range reduction to [1,2), series ln(1+t) ≈ t - t^2/2 + t^3/3 - ... (8 terms)
fn _ln_series_1_to_2(x: Float64) -> Float64:
    # pre: x in [1, 2)
    var t = x - 1.0
    var t_pow = t
    var res = t
    res = res - (t_pow * t) * 0.5          # - t^2/2
    t_pow = t_pow * t                      # t^2
    res = res + (t_pow * t) / 3.0          # + t^3/3
    t_pow = t_pow * t                      # t^3
    res = res - (t_pow * t) / 4.0          # - t^4/4
    t_pow = t_pow * t                      # t^4
    res = res + (t_pow * t) / 5.0          # + t^5/5
    t_pow = t_pow * t                      # t^5
    res = res - (t_pow * t) / 6.0
    t_pow = t_pow * t
    res = res + (t_pow * t) / 7.0
    t_pow = t_pow * t
    res = res - (t_pow * t) / 8.0
    return res

fn ln(x: Float64) -> Float64:
    if x <= 0.0:
        return NEG_INF
    # Decompose x = m * 2^k, m in [1,2)
    var m = x
    var k: Int = 0
    while m >= 2.0:
        m = m * 0.5
        k = k + 1
    while m < 1.0:
        m = m * 2.0
        k = k - 1
    return Float64(k) * LN2 + _ln_series_1_to_2(m)

# sqrt via Newton-Raphson
fn sqrt(x: Float64) -> Float64:
    if x <= 0.0:
        return 0.0
    var y = x
    var i = 0
    while i < 16:
        y = 0.5 * (y + x / y)
        i = i + 1
    return y

# -----------------------------------------------------------------------------
# Lightweight RNG: XorShift64*
# -----------------------------------------------------------------------------
struct RNG:
    var state: UInt64

    fn __init__(out self, seed: UInt64):
        var s = seed
        if s == UInt64(0):
            s = UInt64(0x9E3779B97F4A7C15)
        self.state = s

    fn next_u64(mut self) -> UInt64:
        var x = self.state
        x = x ^ (x >> UInt64(12))
        x = x ^ (x << UInt64(25))
        x = x ^ (x >> UInt64(27))
        self.state = x
        return x & UInt64(0xFFFFFFFFFFFFFFFF)

    fn next_f64(mut self) -> Float64:
        var bits = self.next_u64()
        var mant = (bits >> UInt64(11)) & UInt64(0x1FFFFFFFFFFFFF)  # top 53 bits
        return (Float64(mant) + 1.0) / 9007199254740992.0           # in (0,1)

# -----------------------------------------------------------------------------
# Uniform(a, b)
# -----------------------------------------------------------------------------
struct Uniform:
    var a: Float64
    var b: Float64
    var width: Float64

    fn __init__(out self, a: Float64 = 0.0, b: Float64 = 1.0):
        var lo = a
        var hi = b
        if hi < lo:
            # swap to ensure a <= b
            var tmp = lo
            lo = hi
            hi = tmp
        if hi == lo:
            hi = lo + 1e-12
        self.a = lo
        self.b = hi
        self.width = hi - lo

    fn sample(self, n: Int, seed: UInt64 = UInt64(0xC0FFEE1234ABCDEF)) -> List[Float64]:
        var out = List[Float64]()
        out.reserve(n)
        var rng = RNG(seed)
        var i = 0
        while i < n:
            var u = rng.next_f64()
            out.push_back(self.a + self.width * u)
            i = i + 1
        return out

    fn mean(self) -> Float64:
        return 0.5 * (self.a + self.b)

    fn var(self) -> Float64:
        var w = self.width
        return (w * w) / 12.0

    fn log_prob(self, x: Float64) -> Float64:
        if x < self.a or x >= self.b:
            return NEG_INF
        return -ln(self.width)

# -----------------------------------------------------------------------------
# Bernoulli(p)
# -----------------------------------------------------------------------------
struct Bernoulli:
    var p: Float64

    fn __init__(out self, p: Float64):
        var prob = p
        if prob < 0.0:
            prob = 0.0
        if prob > 1.0:
            prob = 1.0
        self.p = prob

    fn sample(self, n: Int, seed: UInt64 = UInt64(0x1EDC6F41ABC98721)) -> List[Int]:
        var out = List[Int]()
        out.reserve(n)
        var rng = RNG(seed)
        var i = 0
        while i < n:
            var u = rng.next_f64()
            out.push_back(1 if u < self.p else 0)
            i = i + 1
        return out

    fn mean(self) -> Float64:
        return self.p

    fn var(self) -> Float64:
        return self.p * (1.0 - self.p)

    fn log_prob(self, k: Int) -> Float64:
        if k == 1:
            if self.p == 0.0:
                return NEG_INF
            return ln(self.p)
        elif k == 0:
            var q = 1.0 - self.p
            if q == 0.0:
                return NEG_INF
            return ln(q)
        return NEG_INF

# -----------------------------------------------------------------------------
# Binomial(n, p)  (sum of n independent Bernoulli)
# -----------------------------------------------------------------------------
fn _ln_factorial(n: Int) -> Float64:
    # Stirling with correction: ln(n!) ≈ n ln n - n + 0.5 ln(2πn) + 1/(12n)
    if n <= 1:
        return 0.0
    var nf = Float64(n)
    return nf * ln(nf) - nf + 0.5 * ln(2.0 * PI * nf) + (1.0 / (12.0 * nf))

fn _ln_n_choose_k(n: Int, k: Int) -> Float64:
    if k < 0 or k > n:
        return NEG_INF
    return _ln_factorial(n) - _ln_factorial(k) - _ln_factorial(n - k)

struct Binomial:
    var n: Int
    var p: Float64

    fn __init__(out self, n: Int, p: Float64):
        var nn = n
        if nn < 0:
            nn = 0
        var prob = p
        if prob < 0.0:
            prob = 0.0
        if prob > 1.0:
            prob = 1.0
        self.n = nn
        self.p = prob

    fn sample(self, m: Int, seed: UInt64 = UInt64(0xABCDEF0123456789)) -> List[Int]:
        # naive: sum of n Bernoulli per draw (sufficient for CPU baseline; optimize later)
        var out = List[Int]()
        out.reserve(m)
        var rng = RNG(seed)
        var i = 0
        while i < m:
            var s = 0
            var t = 0
            while t < self.n:
                if rng.next_f64() < self.p:
                    s = s + 1
                t = t + 1
            out.push_back(s)
            i = i + 1
        return out

    fn mean(self) -> Float64:
        return Float64(self.n) * self.p

    fn var(self) -> Float64:
        return Float64(self.n) * self.p * (1.0 - self.p)

    fn log_prob(self, k: Int) -> Float64:
        if k < 0 or k > self.n:
            return NEG_INF
        if self.p == 0.0:
            return 0.0 if k == 0 else NEG_INF
        if self.p == 1.0:
            return 0.0 if k == self.n else NEG_INF
        var lnC = _ln_n_choose_k(self.n, k)
        var kp = Float64(k) * ln(self.p)
        var nmk = Float64(self.n - k) * ln(1.0 - self.p)
        return lnC + kp + nmk

# -----------------------------------------------------------------------------
# Categorical(probs[0..K-1]) — samples indices
# -----------------------------------------------------------------------------
struct Categorical:
    var probs: List[Float64]
    var cum: List[Float64]

    fn __init__(out self, probs: List[Float64]):
        # normalize to sum=1 and build cumulative
        var ps = List[Float64]()
        var s: Float64 = 0.0
        var i = 0
        while i < Int(probs.size()):
            var v = probs[i]
            if v < 0.0:
                v = 0.0
            ps.push_back(v)
            s = s + v
            i = i + 1
        if s <= 0.0:
            # fallback to uniform over K
            var k = Int(ps.size())
            if k == 0:
                ps.push_back(1.0)
                s = 1.0
            else:
                var j = 0
                while j < k:
                    ps[j] = 1.0
                    j = j + 1
                s = Float64(k)
        # normalize
        var j = 0
        while j < Int(ps.size()):
            ps[j] = ps[j] / s
            j = j + 1
        # build cumulative
        var cm = List[Float64]()
        cm.reserve(Int(ps.size()))
        var acc: Float64 = 0.0
        var t = 0
        while t < Int(ps.size()):
            acc = acc + ps[t]
            cm.push_back(acc)
            t = t + 1
        self.probs = ps
        self.cum = cm

    fn sample(self, n: Int, seed: UInt64 = UInt64(0xDEADBEEFCAFEBABE)) -> List[Int]:
        var out = List[Int]()
        out.reserve(n)
        var rng = RNG(seed)
        var i = 0
        while i < n:
            var u = rng.next_f64()
            var idx = 0
            # linear scan; can be optimized to binary search later
            while idx < Int(self.cum.size()) and u > self.cum[idx]:
                idx = idx + 1
            if idx >= Int(self.cum.size()):
                idx = Int(self.cum.size()) - 1
            out.push_back(idx)
            i = i + 1
        return out

    fn mean(self) -> Float64:
        # Not defined on indices → return NaN
        return 0.0 / 0.0

    fn var(self) -> Float64:
        # Not defined on indices → return NaN
        return 0.0 / 0.0

    fn log_prob(self, index: Int) -> Float64:
        if index < 0 or index >= Int(self.probs.size()):
            return NEG_INF
        var p = self.probs[index]
        if p <= 0.0:
            return NEG_INF
        return ln(p)

# -----------------------------------------------------------------------------
# Normal(μ, σ) with CLT sampler and exact log_prob (no exp needed)
# -----------------------------------------------------------------------------
struct Normal:
    var mean: Float64
    var std: Float64
    var var_val: Float64
    var inv_var: Float64

    fn __init__(out self, mean: Float64, std: Float64):
        var s = std
        if s <= 0.0:
            s = 1e-12
        self.mean = mean
        self.std = s
        self.var_val = s * s
        self.inv_var = 1.0 / self.var_val

    fn sample(self, n: Int, seed: UInt64 = UInt64(0xA5A5A5A5D3C1B2F1)) -> List[Float64]:
        var out = List[Float64]()
        out.reserve(n)
        var rng = RNG(seed)
        var i = 0
        while i < n:
            var s: Float64 = 0.0
            var k = 0
            while k < 12:
                s = s + rng.next_f64()
                k = k + 1
            s = s - 6.0  # ~ N(0,1)
            out.push_back(self.mean + self.std * s)
            i = i + 1
        return out

    fn mean(self) -> Float64:
        return self.mean

    fn var(self) -> Float64:
        return self.var_val

    fn log_prob(self, x: Float64) -> Float64:
        var z = x - self.mean
        # log N(x|μ,σ^2) = -0.5 * [ (z^2/σ^2) + ln(2π) + 2 ln σ ]
        var term = (z * z) * self.inv_var
        return -0.5 * (term + LN2PI + 2.0 * ln(self.std))
