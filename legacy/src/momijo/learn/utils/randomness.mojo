# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.utils.randomness
# File:         src/momijo/learn/utils/randomness.mojo
#
# Description:
#   Deterministic and fast random utilities for Momijo Learn.
#   - SplitMix64 seeding + Xoshiro256** core PRNG (UInt64)
#   - RNG struct with next_u64/next_f64, randint, uniform, normal, bernoulli
#   - In-place shuffle, permutation, choice (weighted/unweighted)
#   - SeedBundle helper to derive multiple consistent seeds from one base seed
#   - seed_all(seed) kept as a backend-agnostic compatibility shim (no globals)
#
# Design notes:
#     around. This makes determinism and testing straightforward.
#   - All floating output uses open interval (0,1) where appropriate to avoid
#     log(0) issues downstream (e.g. Box–Muller).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List

# -----------------------------------------------------------------------------
# Utilities: mixing, seeding, tiny math
# -----------------------------------------------------------------------------

@always_inline
fn _rotl(x: UInt64, k: Int) -> UInt64:
    var kk = UInt64(k & 63)
    return (x << kk) | (x >> (UInt64(64) - kk))

# SplitMix64: high-quality 64-bit mixer (seed expander)
struct SplitMix64:
    var state: UInt64
    fn __init__(out self, seed: UInt64):
        self.state = seed

    fn next(mut self) -> UInt64:
        self.state = self.state &+ 0x9E3779B97F4A7C15  # wrap-around add
        var z = self.state
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        z = z ^ (z >> 31)
        return z

# Minimal helpers
@always_inline
fn _to_u64(seed: Int) -> UInt64:
    if seed < 0:
        return UInt64(-seed)
    return UInt64(seed)

@always_inline
fn _u64_to_open01(u: UInt64) -> Float32:
    # Map 53 high bits to (0,1). Avoid exact 0 by adding 0.5 ulp.
    var v = u >> 11                              # keep top 53 bits
    var y = Float32(v) * (1.0 / 9007199254740992.0)  # 1/2^53
    if y <= 0.0:
        return 5e-324  # Double min subnormal as tiny positive
    if y >= 1.0:
        return 1.0 - 5e-324
    return y

@always_inline
fn _abs_f64(x: Float32) -> Float32:
    var v = x
    if v < 0.0:
        v = -v
    return v

# -----------------------------------------------------------------------------
# Xoshiro256** core PRNG
# -----------------------------------------------------------------------------

struct Xoshiro256ss:
    var s0: UInt64
    var s1: UInt64
    var s2: UInt64
    var s3: UInt64

    fn __init__(out self, s0: UInt64, s1: UInt64, s2: UInt64, s3: UInt64):
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3

    # Construct from a single 64-bit seed using SplitMix64
    @staticmethod
    fn from_seed(seed: UInt64) -> Xoshiro256ss:
        var sm = SplitMix64(seed)
        var a = sm.next()
        var b = sm.next()
        var c = sm.next()
        var d = sm.next()
        # Avoid all-zero state
        if (a | b | c | d) == 0:
            d = 1
        return Xoshiro256ss(a, b, c, d)

    fn next_u64(mut self) -> UInt64:
        var result = _rotl(self.s1 * 5, 7) * 9
        var t = self.s1 << 17

        self.s2 = self.s2 ^ self.s0
        self.s3 = self.s3 ^ self.s1
        self.s1 = self.s1 ^ self.s2
        self.s0 = self.s0 ^ self.s3
        self.s2 = self.s2 ^ t
        self.s3 = _rotl(self.s3, 45)

        return result

    # Jump for 2^128 steps (for independent sequences)
    fn jump(mut self):
        var J = [UInt64(0x180EC6D33CFD0ABA), UInt64(0xD5A61266F0C9392C),
                 UInt64(0xA9582618E03FC9AA), UInt64(0x39ABDC4529B1661C)]
        var s0 = UInt64(0)
        var s1 = UInt64(0)
        var s2 = UInt64(0)
        var s3 = UInt64(0)
        var i = 0
        while i < 4:
            var b = J[i]
            var bit = 0
            while bit < 64:
                if (b & (UInt64(1) << UInt64(bit))) != 0:
                    s0 = s0 ^ self.s0
                    s1 = s1 ^ self.s1
                    s2 = s2 ^ self.s2
                    s3 = s3 ^ self.s3
                _ = self.next_u64()
                bit = bit + 1
            i = i + 1
        self.s0 = s0; self.s1 = s1; self.s2 = s2; self.s3 = s3

# -----------------------------------------------------------------------------
# RNG facade with distribution helpers
# -----------------------------------------------------------------------------

struct RNG:
    var core: Xoshiro256ss
    var _has_gauss: Bool
    var _gauss: Float32

    fn __init__(out self, seed: UInt64):
        self.core = Xoshiro256ss.from_seed(seed)
        self._has_gauss = False
        self._gauss = 0.0

    @staticmethod
    fn from_int_seed(seed: Int) -> RNG:
        return RNG(_to_u64(seed))

    fn next_u64(mut self) -> UInt64:
        return self.core.next_u64()

    fn next_f64_open01(mut self) -> Float32:
        return _u64_to_open01(self.next_u64())

    # Uniform integers in [low, high) with rejection to avoid modulo bias
    fn randint(mut self, low: Int, high: Int) -> Int:

        if high <= low:
            return low

        var span_u = UInt64(high - low)
        # max_u = 2^64-1
        var max_u = UInt64(0) - UInt64(1)
        # threshold = max_u - ((max_u + 1) % span_u)

        var threshold = max_u - ((max_u + UInt64(1)) % span_u)

        while True:
            var x = self.next_u64()
            if x <= threshold:
                return low + Int(x % span_u)

    # Uniform Float32 in [low, high)
    fn uniform(mut self, low: Float32 = 0.0, high: Float32 = 1.0) -> Float32:

        if high <= low:
            return low
        var r = self.next_f64_open01()
        return low + (high - low) * r

    # Standard normal via Box–Muller (with 1-sample cache)
    fn normal(mut self, mean: Float32 = 0.0, std: Float32 = 1.0) -> Float32:

        if std < 0.0:
            std = -std

        if self._has_gauss:
            self._has_gauss = False
            return mean + std * self._gauss

        var u1 = self.next_f64_open01()
        var u2 = self.next_f64_open01()


        if u1 <= 5e-324:
            u1 = 5e-324

        # r = sqrt(-2 ln u1), theta = 2*pi*u2 (tau ≈ 6.283185307179586)
        var ln_u1 = _log_approx(u1)
        var r = _sqrt_approx(-2.0 * ln_u1)
        var tau = 6.283185307179586
        var angle = tau * u2


        var z0 = r * _cos_sin_combo(angle, True)
        var z1 = r * _cos_sin_combo(angle, False)

        self._gauss = z1
        self._has_gauss = True
        return mean + std * z0


    # Bernoulli(p)
    fn bernoulli(mut self, p: Float32) -> Int:
        var pp = p
        if pp < 0.0: pp = 0.0
        if pp > 1.0: pp = 1.0
        var u = self.next_f64_open01()
        if u < pp: return 1
        return 0

    # Shuffle a list in-place (Fisher–Yates)
    fn shuffle[T: Copyable & Movable & ImplicitlyCopyable](mut self, xs: List[T]):
        var n = len(xs)
        var i = n - 1
        while i > 0:
            var j = self.randint(0, i + 1)
            var tmp = xs[i]
            xs[i] = xs[j]
            xs[j] = tmp
            i = i - 1

    # Return a new permuted list of indices 0..n-1
    fn permutation(mut self, n: Int) -> List[Int]:
        var idx = List[Int]()
        var i = 0
        while i < n:
            idx.append(i)
            i = i + 1
        self.shuffle(idx)
        return idx

   # Choice without replacement from [0..n-1], returns k unique indices.
    fn choice_k(mut self, n: Int, k: Int) -> List[Int]:
        var out = List[Int]()
        if n <= 0:
            return out
        var kk = k
        if kk < 0: kk = 0
        if kk > n: kk = n


        var idx = self.permutation(n)
        var i = 0
        while i < kk:
            out.append(idx[i])
            i = i + 1
        return out

    # Weighted choice (with replacement) over weights w[0..m-1]
    fn weighted_choice(mut self, weights: List[Float32]) -> Int:
        var m = len(weights)
        if m <= 0:
            return 0

        var sumw = 0.0
        var i = 0
        while i < m:
            var w = weights[i]
            if w > 0.0:
                sumw = sumw + w
            i = i + 1

        if sumw <= 0.0:
            return 0

        var r = self.uniform(0.0, sumw)
        var acc = 0.0
        i = 0
        while i < m:
            var w2 = weights[i]
            if w2 > 0.0:
                acc = acc + w2
                if r <= acc:
                    return i
            i = i + 1

        return m - 1

# -----------------------------------------------------------------------------
# Tiny approximations for log/sqrt/cos/sin to avoid external deps
# -----------------------------------------------------------------------------

# Natural log approximation: range reduction + atanh-series (like elsewhere)
fn _log_approx(x: Float32) -> Float32:
    if x <= 0.0:
        return -1.7976931348623157e308
    var ln2 = 0.6931471805599453
    var k = 0
    var m = x
    while m > 1.5:
        m = m * 0.5
        k = k + 1
    while m < 0.75:
        m = m * 2.0
        k = k - 1
    var t = (m - 1.0) / (m + 1.0)
    var t2 = t * t
    var term = t
    var sum = term
    var i = 1
    while i <= 6:
        term = term * t2
        var denom = Float32(2 * i + 1)
        sum = sum + (term / denom)
        i = i + 1
    return 2.0 * sum + Float32(k) * ln2

# Sqrt via 6 steps of Newton–Raphson
fn _sqrt_approx(x: Float32) -> Float32:
    if x <= 0.0:
        return 0.0
    var g = x
    var i = 0
    while i < 6:
        g = 0.5 * (g + x / g)
        i = i + 1
    return g

# Very small cos/sin helper: use a minimized cordic-like Taylor for |x|<=π
# and wrap to [-π, π] first. For Box–Muller we only need either cos or sin.
fn _wrap_pi(x: Float32) -> Float32:
    var pi = 3.141592653589793
    var two_pi = 6.283185307179586
    var v = x
    # crude wrap by subtraction
    while v > pi:
        v = v - two_pi
    while v < -pi:
        v = v + two_pi
    return v

fn _cos_sin_combo(x: Float32, want_cos: Bool) -> Float32:
    var v = _wrap_pi(x)
    # 7th-order Taylor is enough here
    var v2 = v * v
    var v3 = v2 * v
    var v4 = v2 * v2
    var v5 = v4 * v
    var v6 = v3 * v3
    var v7 = v6 * v
    if want_cos:
        # cos ≈ 1 - v^2/2 + v^4/24 - v^6/720
        var c = 1.0
        c = c - (v2 * 0.5)
        c = c + (v4 * (1.0 / 24.0))
        c = c - (v6 * (1.0 / 720.0))
        return c
    # sin ≈ v - v^3/6 + v^5/120 - v^7/5040
    var s = v
    s = s - (v3 * (1.0 / 6.0))
    s = s + (v5 * (1.0 / 120.0))
    s = s - (v7 * (1.0 / 5040.0))
    return s

# -----------------------------------------------------------------------------
# Seed derivation helpers
# -----------------------------------------------------------------------------

struct SeedBundle:
    var global: UInt64
    var data:   UInt64
    var model:  UInt64
    var aug:    UInt64
    var loader: UInt64

    fn __init__(out self, base: UInt64):
        var sm = SplitMix64(base)
        self.global = sm.next()
        self.data   = sm.next()
        self.model  = sm.next()
        self.aug    = sm.next()
        self.loader = sm.next()

    fn to_list(self) -> List[UInt64]:
        var xs = List[UInt64]()
        xs.append(self.global)
        xs.append(self.data)
        xs.append(self.model)
        xs.append(self.aug)
        xs.append(self.loader)
        return xs

# Primary constructor helpers
@always_inline
fn rng_from_seed(seed: Int) -> RNG:
    return RNG(_to_u64(seed))

@always_inline
fn derived_rngs(seed: Int) -> (RNG, RNG, RNG, RNG, RNG):
    var b = SeedBundle(_to_u64(seed))
    return (RNG(b.global), RNG(b.data), RNG(b.model), RNG(b.aug), RNG(b.loader))

# -----------------------------------------------------------------------------
# Compatibility shim: seed_all
# -----------------------------------------------------------------------------
# This function intentionally does NOT set any global state (Momijo policy).
# It exists to mirror common ML libraries. Use it to produce a deterministic

fn seed_all(seed: Int):
    var _ = seed
    # no-op by design to avoid globals
    # Recommended pattern:
    #   var (rng_global, rng_data, rng_model, rng_aug, rng_loader) = derived_rngs(seed)
    # and then pass these RNGs explicitly to places that need randomness.

# -----------------------------------------------------------------------------
# (Optional) convenience: quick samples without managing RNG externally
# -----------------------------------------------------------------------------

fn randint(seed: Int, low: Int, high: Int) -> Int:
    var r = rng_from_seed(seed)
    return r.randint(low, high)

fn uniform(seed: Int, low: Float32 = 0.0, high: Float32 = 1.0) -> Float32:
    var r = rng_from_seed(seed)
    return r.uniform(low, high)

fn normal(seed: Int, mean: Float32 = 0.0, std: Float32 = 1.0) -> Float32:
    var r = rng_from_seed(seed)
    return r.normal(mean, std)

fn bernoulli(seed: Int, p: Float32) -> Int:
    var r = rng_from_seed(seed)
    return r.bernoulli(p)

# Shuffle/permute helpers working on copies
fn permutation(seed: Int, n: Int) -> List[Int]:
    var r = rng_from_seed(seed)
    return r.permutation(n)

fn choice_k(seed: Int, n: Int, k: Int) -> List[Int]:
    var r = rng_from_seed(seed)
    return r.choice_k(n, k)
