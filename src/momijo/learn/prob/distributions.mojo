# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.prob.distributions
# File:         src/momijo/learn/prob/distributions.mojo
#
# Description:
#   Probability distributions for Momijo Learn: Uniform, Bernoulli, Binomial,
#   Categorical, Normal. Includes a lightweight RNG and a math shim (ln, sqrt)
#   to avoid external dependencies. Provides scalar/List APIs and Tensor APIs.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor   # Tensor facade per Momijo policy

# -----------------------------------------------------------------------------
# Constants via struct (no globals)
# -----------------------------------------------------------------------------
struct _Const:
    @staticmethod
    fn NEG_INF() -> Float32: return -1.0e300
    @staticmethod
    fn LN2()     -> Float32: return 0.6931471805599453
    @staticmethod
    fn LN2PI()   -> Float32: return 1.8378770664093453
    @staticmethod
    fn PI()      -> Float32: return 3.141592653589793

# -----------------------------------------------------------------------------
# Math shim
# -----------------------------------------------------------------------------
@always_inline
fn _ln_series_1_to_2(x: Float32) -> Float32:
    var t = x - 1.0
    var t2 = t * t
    var t3 = t2 * t
    var t4 = t3 * t
    var t5 = t4 * t
    var t6 = t5 * t
    var t7 = t6 * t
    var t8 = t7 * t
    return t \
         - 0.5 * t2 \
         + (1.0 / 3.0) * t3 \
         - 0.25 * t4 \
         + 0.2 * t5 \
         - (1.0 / 6.0) * t6 \
         + (1.0 / 7.0) * t7 \
         - 0.125 * t8

fn ln(x: Float32) -> Float32:
    if x <= 0.0:
        return _Const.NEG_INF()
    var m = x
    var k: Int = 0
    while m >= 2.0:
        m = m * 0.5
        k = k + 1
    while m < 1.0:
        m = m * 2.0
        k = k - 1
    return Float32(k) * _Const.LN2() + _ln_series_1_to_2(m)

fn sqrt(x: Float32) -> Float32:
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

    fn next_f64(mut self) -> Float32:
        # Returns (0,1); avoids 0 exactly.
        var bits = self.next_u64()
        var mant = (bits >> UInt64(11)) & UInt64(0x1FFFFFFFFFFFFF)
        return (Float32(mant) + 1.0) / 9007199254740992.0

# -----------------------------------------------------------------------------
# Small tensor helpers
# -----------------------------------------------------------------------------
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
fn _zeros_like_shape(shape: List[Int]) -> tensor.Tensor[Float32]:
    return tensor.Tensor[Float32](shape, 0.0)

# Fill a tensor with U(0,1) using RNG
fn _fill_uniform01(mut rng: RNG, out: tensor.Tensor[Float32]) -> RNG:
    var d = out._data
    var n = _numel(out.shape())
    var i = 0
    while i < n:
        d[i] = rng.next_f64()
        i = i + 1
    return rng

# -----------------------------------------------------------------------------
# Uniform(a, b)
# -----------------------------------------------------------------------------
struct Uniform:
    var a: Float32
    var b: Float32
    var width: Float32

    fn __init__(out self, a: Float32 = 0.0, b: Float32 = 1.0):
        var lo = a
        var hi = b
        if hi < lo:
            var t = lo; lo = hi; hi = t
        if hi == lo:
            hi = lo + 1e-12
        self.a = lo
        self.b = hi
        self.width = hi - lo

    # -------- List API --------
    fn sample(self, n: Int, seed: UInt64 = UInt64(0xC0FFEE1234ABCDEF)) -> List[Float32]:
        var out = List[Float32]()
        out.reserve(n)
        var rng = RNG(seed)
        var i = 0
        while i < n:
            var u = rng.next_f64()
            out.append(self.a + self.width * u)
            i = i + 1
        return out

    fn mean_value(self) -> Float32:
        return 0.5 * (self.a + self.b)

    fn variance(self) -> Float32:
        var w = self.width
        return (w * w) / 12.0

    fn log_prob(self, x: Float32) -> Float32:
        if x < self.a or x >= self.b:
            return _Const.NEG_INF()
        return -ln(self.width)

    # -------- Tensor API --------
    fn sample_tensor(self, shape: List[Int], seed: UInt64 = UInt64(0xC0FFEE1234ABCDEF)) -> tensor.Tensor[Float32]:
        var out = tensor.Tensor[Float32](shape, 0.0)
        var rng = RNG(seed)
        rng = _fill_uniform01(rng, out)
        var d = out._data
        var n = _numel(shape)
        var i = 0
        while i < n:
            d[i] = self.a + self.width * d[i]
            i = i + 1
        return out

    fn log_prob_tensor(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var out = tensor.Tensor[Float32](x.shape(), 0.0)
        var xd = x._data
        var yd = out._data
        var n = _numel(x.shape())
        var i = 0
        while i < n:
            var v = xd[i]
            if v < self.a or v >= self.b:
                yd[i] = _Const.NEG_INF()
            else:
                yd[i] = -ln(self.width)
            i = i + 1
        return out

# -----------------------------------------------------------------------------
# Bernoulli(p)
# -----------------------------------------------------------------------------
struct Bernoulli:
    var p: Float32

    fn __init__(out self, p: Float32):
        var prob = p
        if prob < 0.0: prob = 0.0
        if prob > 1.0: prob = 1.0
        self.p = prob

    # -------- List API --------
    fn sample(self, n: Int, seed: UInt64 = UInt64(0x1EDC6F41ABC98721)) -> List[Int]:
        var out = List[Int]()
        out.reserve(n)
        var rng = RNG(seed)
        var i = 0
        while i < n:
            var u = rng.next_f64()
            out.append(1 if u < self.p else 0)
            i = i + 1
        return out

    fn mean_value(self) -> Float32:
        return self.p

    fn variance(self) -> Float32:
        return self.p * (1.0 - self.p)

    fn log_prob(self, k: Int) -> Float32:
        if k == 1:
            if self.p == 0.0: return _Const.NEG_INF()
            return ln(self.p)
        elif k == 0:
            var q = 1.0 - self.p
            if q == 0.0: return _Const.NEG_INF()
            return ln(q)
        return _Const.NEG_INF()

    # -------- Tensor API --------
    # Returns 0.0/1.0 tensor (Float32)
    fn sample_tensor(self, shape: List[Int], seed: UInt64 = UInt64(0x1EDC6F41ABC98721)) -> tensor.Tensor[Float32]:
        var out = tensor.Tensor[Float32](shape, 0.0)
        var rng = RNG(seed)
        var d = out._data
        var n = _numel(shape)
        var i = 0
        while i < n:
            var u = rng.next_f64()
            d[i] = 1.0 if u < self.p else 0.0
            i = i + 1
        return out

    fn log_prob_tensor(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var out = tensor.Tensor[Float32](x.shape(), 0.0)
        var xd = x._data
        var yd = out._data
        var n = _numel(x.shape())
        var i = 0
        var p1 = self.p
        var q1 = 1.0 - p1
        while i < n:
            var v = xd[i]
            if v >= 0.5 and v <= 1.5:
                if p1 == 0.0:
                    yd[i] = _Const.NEG_INF()
                else:
                    yd[i] = ln(p1)
            elif v > -0.5 and v < 0.5:
                if q1 == 0.0:
                    yd[i] = _Const.NEG_INF()
                else:
                    yd[i] = ln(q1)
            else:
                yd[i] = _Const.NEG_INF()
            i = i + 1
        return out

# -----------------------------------------------------------------------------
# Binomial(n, p)
# -----------------------------------------------------------------------------
@always_inline
fn _ln_factorial(n: Int) -> Float32:
    if n <= 1: return 0.0
    var nf = Float32(n)
    return nf * ln(nf) - nf + 0.5 * ln(2.0 * _Const.PI() * nf) + (1.0 / (12.0 * nf))

@always_inline
fn _ln_n_choose_k(n: Int, k: Int) -> Float32:
    if k < 0 or k > n: return _Const.NEG_INF()
    return _ln_factorial(n) - _ln_factorial(k) - _ln_factorial(n - k)

struct Binomial:
    var n: Int
    var p: Float32

    fn __init__(out self, n: Int, p: Float32):
        var nn = n
        if nn < 0: nn = 0
        var prob = p
        if prob < 0.0: prob = 0.0
        if prob > 1.0: prob = 1.0
        self.n = nn
        self.p = prob

    # -------- List API --------
    fn sample(self, m: Int, seed: UInt64 = UInt64(0xABCDEF0123456789)) -> List[Int]:
        var out = List[Int]()
        out.reserve(m)
        var rng = RNG(seed)
        var i = 0
        while i < m:
            var s = 0
            var t = 0
            while t < self.n:
                if rng.next_f64() < self.p: s = s + 1
                t = t + 1
            out.append(s)
            i = i + 1
        return out

    fn mean_value(self) -> Float32:
        return Float32(self.n) * self.p

    fn variance(self) -> Float32:
        return Float32(self.n) * self.p * (1.0 - self.p)

    fn log_prob(self, k: Int) -> Float32:
        if k < 0 or k > self.n: return _Const.NEG_INF()
        if self.p == 0.0: return 0.0 if k == 0 else _Const.NEG_INF()
        if self.p == 1.0: return 0.0 if k == self.n else _Const.NEG_INF()
        var lnC = _ln_n_choose_k(self.n, k)
        var kp = Float32(k) * ln(self.p)
        var nmk = Float32(self.n - k) * ln(1.0 - self.p)
        return lnC + kp + nmk

    # -------- Tensor API --------
    # Returns counts (Float32 integers 0..n)
    fn sample_tensor(self, shape: List[Int], seed: UInt64 = UInt64(0xABCDEF0123456789)) -> tensor.Tensor[Float32]:
        var out = tensor.Tensor[Float32](shape, 0.0)
        var rng = RNG(seed)
        var d = out._data
        var n_el = _numel(shape)
        var i = 0
        while i < n_el:
            var s = 0
            var t = 0
            while t < self.n:
                if rng.next_f64() < self.p: s = s + 1
                t = t + 1
            d[i] = Float32(s)
            i = i + 1
        return out

    fn log_prob_tensor(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var out = tensor.Tensor[Float32](x.shape(), 0.0)
        var xd = x._data
        var yd = out._data
        var n_el = _numel(x.shape())
        var i = 0
        while i < n_el:
            var kv = Int(xd[i])
            var lp = self.log_prob(kv)
            yd[i] = lp
            i = i + 1
        return out

# -----------------------------------------------------------------------------
# Categorical(probs[0..K-1])
# -----------------------------------------------------------------------------
struct Categorical:
    var probs: List[Float32]
    var cum: List[Float32]

    fn __init__(out self, probs: List[Float32]):
        var ps = List[Float32]()
        var s: Float32 = 0.0
        var i = 0
        while i < len(probs):
            var v = probs[i]
            if v < 0.0: v = 0.0
            ps.append(v)
            s = s + v
            i = i + 1
        if s <= 0.0:
            var k = len(ps)
            if k == 0:
                ps.append(1.0); s = 1.0
            else:
                var j = 0
                while j < k:
                    ps[j] = 1.0
                    j = j + 1
                s = Float32(k)
        var j2 = 0
        while j2 < len(ps):
            ps[j2] = ps[j2] / s
            j2 = j2 + 1
        var cm = List[Float32]()
        cm.reserve(len(ps))
        var acc: Float32 = 0.0
        var t = 0
        while t < len(ps):
            acc = acc + ps[t]
            cm.append(acc)
            t = t + 1
        self.probs = ps
        self.cum = cm

    # -------- List API --------
    fn sample(self, n: Int, seed: UInt64 = UInt64(0xDEADBEEFCAFEBABE)) -> List[Int]:
        var out = List[Int]()
        out.reserve(n)
        var rng = RNG(seed)
        var i = 0
        while i < n:
            var u = rng.next_f64()
            var idx = 0
            while idx < len(self.cum) and u > self.cum[idx]:
                idx = idx + 1
            if idx >= len(self.cum):
                idx = len(self.cum) - 1
            out.append(idx)
            i = i + 1
        return out

    fn mean_value(self) -> Float32:
        # Not defined for categorical over indices; NaN sentinel:
        return 0.0 / 0.0

    fn variance(self) -> Float32:
        # Not defined for categorical over indices; NaN sentinel:
        return 0.0 / 0.0

    fn log_prob(self, index: Int) -> Float32:
        if index < 0 or index >= len(self.probs):
            return _Const.NEG_INF()
        var p = self.probs[index]
        if p <= 0.0: return _Const.NEG_INF()
        return ln(p)

    # -------- Tensor API --------
    # probs_t: 1D Tensor summing to 1 (will be normalized internally to be safe)
    fn from_probs_tensor(probs_t: tensor.Tensor[Float32]) -> Categorical:
        var k = _numel(probs_t.shape())
        var ps = List[Float32]()
        ps.reserve(k)
        var s: Float32 = 0.0
        var d = probs_t._data
        var i = 0
        while i < k:
            var v = d[i]
            if v < 0.0: v = 0.0
            ps.append(v)
            s = s + v
            i = i + 1
        if s <= 0.0:
            var j = 0
            while j < k:
                ps[j] = 1.0
                j = j + 1
            s = Float32(k)
        var j2 = 0
        while j2 < k:
            ps[j2] = ps[j2] / s
            j2 = j2 + 1
        return Categorical(ps)

    # Returns indices in 0..K-1 as Float32 (integers stored in Float32)
    fn sample_tensor(self, shape: List[Int], seed: UInt64 = UInt64(0xDEADBEEFCAFEBABE)) -> tensor.Tensor[Float32]:
        var out = tensor.Tensor[Float32](shape, 0.0)
        var rng = RNG(seed)
        var d = out._data
        var n_el = _numel(shape)
        var i = 0
        while i < n_el:
            var u = rng.next_f64()
            var idx = 0
            while idx < len(self.cum) and u > self.cum[idx]:
                idx = idx + 1
            if idx >= len(self.cum):
                idx = len(self.cum) - 1
            d[i] = Float32(idx)
            i = i + 1
        return out

    # x: Float32 tensor with integer category indices
    fn log_prob_tensor(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var out = tensor.Tensor[Float32](x.shape(), 0.0)
        var xd = x._data
        var yd = out._data
        var n_el = _numel(x.shape())
        var i = 0
        while i < n_el:
            var index = Int(xd[i])
            var lp = self.log_prob(index)
            yd[i] = lp
            i = i + 1
        return out

# -----------------------------------------------------------------------------
# Normal(μ, σ)
# -----------------------------------------------------------------------------
struct Normal:
    var mu: Float32
    var std: Float32
    var var_val: Float32
    var inv_var: Float32

    fn __init__(out self, mean: Float32, std: Float32):
        var s = std
        if s <= 0.0: s = 1e-12
        self.mu = mean
        self.std = s
        self.var_val = s * s
        self.inv_var = 1.0 / self.var_val

    # -------- List API --------
    fn sample(self, n: Int, seed: UInt64 = UInt64(0xA5A5A5A5D3C1B2F1)) -> List[Float32]:
        # Box-Muller-less CLT approximation (12 uniforms).
        var out = List[Float32]()
        out.reserve(n)
        var rng = RNG(seed)
        var i = 0
        while i < n:
            var s: Float32 = 0.0
            var k = 0
            while k < 12:
                s = s + rng.next_f64()
                k = k + 1
            s = s - 6.0
            out.append(self.mu + self.std * s)
            i = i + 1
        return out

    fn mean_value(self) -> Float32:
        return self.mu

    fn variance(self) -> Float32:
        return self.var_val

    fn log_prob(self, x: Float32) -> Float32:
        var z = x - self.mu
        var term = (z * z) * self.inv_var
        return -0.5 * (term + _Const.LN2PI() + 2.0 * ln(self.std))

    # -------- Tensor API --------
    fn sample_tensor(self, shape: List[Int], seed: UInt64 = UInt64(0xA5A5A5A5D3C1B2F1)) -> tensor.Tensor[Float32]:
        var out = tensor.Tensor[Float32](shape, 0.0)
        var rng = RNG(seed)
        var d = out._data
        var n = _numel(shape)
        var i = 0
        while i < n:
            var s: Float32 = 0.0
            var k = 0
            while k < 12:
                s = s + rng.next_f64()
                k = k + 1
            s = s - 6.0
            d[i] = self.mu + self.std * s
            i = i + 1
        return out

    fn log_prob_tensor(self, x: tensor.Tensor[Float32]) -> tensor.Tensor[Float32]:
        var out = tensor.Tensor[Float32](x.shape(), 0.0)
        var xd = x._data
        var yd = out._data
        var n = _numel(x.shape())
        var i = 0
        var invv = self.inv_var
        var lconst = _Const.LN2PI()
        var ls = ln(self.std)
        while i < n:
            var z = xd[i] - self.mu
            var term = (z * z) * invv
            yd[i] = -0.5 * (term + lconst + 2.0 * ls)
            i = i + 1
        return out
