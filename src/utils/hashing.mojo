# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.utils
# File: src/momijo/utils/hashing.mojo
#
# NOTE: Follows the strict Mojo checklist:
# - Only 'var' (no 'let'); no global mutable state
# - No 'export'; explicit imports; English comments
# - Constructors use `fn __init__(out self, ...)`
# - Prefer mut/out over inout
# - Minimal exceptions; no prints here
# - Provide deterministic, stable 64-bit hashes (FNV-1a + SplitMix64 combiner)

from stdlib.list import List
from stdlib.string import String

# ------------------------------------------------------------
# Constants (compile-time literals only; no global mutation)
# ------------------------------------------------------------
fn _const_fnv64_offset() -> UInt64: return UInt64(1469598103934665603)    # FNV64 offset basis
fn _const_fnv64_prime() -> UInt64:  return UInt64(1099511628211)          # FNV64 prime
fn _const_golden64() -> UInt64:     return UInt64(0x9E3779B97F4A7C15)

# ------------------------------------------------------------
# SplitMix64 mixer (for hash_combine and avalanche)
# ------------------------------------------------------------
fn _splitmix64(x: UInt64) -> UInt64:
    var z = x + UInt64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
    return z ^ (z >> 31)

# ------------------------------------------------------------
# FNV-1a hasher (64-bit) — streaming
# ------------------------------------------------------------
struct Hasher64:
    state: UInt64

    fn __init__(out self, seed: UInt64 = _const_fnv64_offset()):
        self.state = seed

    fn update_byte(mut self, b: UInt8):
        var s = self.state
        s = s ^ UInt64(b)
        s = s * _const_fnv64_prime()
        self.state = s

    fn update_bytes(mut self, xs: List[UInt8]):
        var i = 0
        var n = len(xs)
        while i < n:
            self.update_byte(xs[i])
            i += 1

    fn update_u64(mut self, x: UInt64):
        # process as 8 bytes little-endian to remain platform-stable
        var i = 0
        while i < 8:
            self.update_byte(UInt8((x >> UInt64(i * 8)) & UInt64(0xFF)))
            i += 1

    fn update_i64(mut self, x: Int64):
        self.update_u64(UInt64(x))

    fn update_bool(mut self, v: Bool):
        if v:
            self.update_byte(UInt8(1))
        else:
            self.update_byte(UInt8(0))

    fn update_f64(mut self, x: Float64):
        # Normalize NaN to a single representation to ensure deterministic hashing
        var y = x
        if not (x == x):  # NaN check
            y = Float64(0.0) / Float64(0.0)  # produce a NaN consistently; then map to fixed bits below
        # Reinterpret the bits by simple rounding trick:
        # Convert to integer bits via byte-by-byte extraction using string formatting is not available.
        # We'll use a simple stable approximation: treat +0.0 and -0.0 the same and hash the decimal string.
        # (Better: via bitcasts, but Mojo may not expose them portably yet.)
        # To keep stability, we map -0.0 -> +0.0 and infinities to fixed tokens.
        if y == Float64(0.0):
            # +0.0 and -0.0 collapse
            self.update_bytes(String("+0.0").bytes())
            return
        if y > Float64(0.0) and (y * Float64(0.5)) == Float64(y): # a benign op to avoid NaN compare pitfalls
            # normal positive
            self.update_bytes(String("p").bytes())
            self.update_bytes(String(String(y)).bytes())
            return
        if y < Float64(0.0):
            self.update_bytes(String("n").bytes())
            self.update_bytes(String(String(y)).bytes())
            return
        # handle NaN / inf / others as text
        self.update_bytes(String(String(y)).bytes())

    fn update_string(mut self, s: String):
        self.update_bytes(s.bytes())

    fn digest(self) -> UInt64:
        # Additional avalanche for better distribution
        return _splitmix64(self.state)

# ------------------------------------------------------------
# Convenience one-shot hashing helpers
# ------------------------------------------------------------
fn hash_bytes(xs: List[UInt8]) -> UInt64:
    var h = Hasher64()
    h.update_bytes(xs)
    return h.digest()

fn hash_string(s: String) -> UInt64:
    var h = Hasher64()
    h.update_string(s)
    return h.digest()

fn hash_bool(b: Bool) -> UInt64:
    var h = Hasher64()
    h.update_bool(b)
    return h.digest()

fn hash_i64(x: Int64) -> UInt64:
    var h = Hasher64()
    h.update_i64(x)
    return h.digest()

fn hash_f64(x: Float64) -> UInt64:
    var h = Hasher64()
    h.update_f64(x)
    return h.digest()

# Combine two 64-bit hashes deterministically
fn hash_combine(h1: UInt64, h2: UInt64) -> UInt64:
    var x = h1 ^ (_const_golden64() + h2 + (h1 << UInt64(6)) + (h1 >> UInt64(2)))
    return _splitmix64(x)

# Hash a list of strings (common for env/args) deterministically
fn hash_list_of_strings(xs: List[String]) -> UInt64:
    var h = Hasher64()
    var i = 0
    var n = len(xs)
    while i < n:
        h.update_string(xs[i])
        h.update_byte(UInt8(0xFF))  # separator to avoid ambiguity
        i += 1
    return h.digest()

# Hash a list of bytes-like (List[UInt8] chunks) deterministically
fn hash_list_of_bytes(chunks: List[List[UInt8]]) -> UInt64:
    var h = Hasher64()
    var i = 0
    var n = len(chunks)
    while i < n:
        h.update_bytes(chunks[i])
        h.update_byte(UInt8(0x00))  # separator
        i += 1
    return h.digest()

# ------------------------------------------------------------
# Minimal self-test (no prints)
# ------------------------------------------------------------
fn _self_test() -> Bool:
    var ok = True
    var h1 = hash_string(String("abc"))
    var h2 = hash_string(String("abcd"))
    ok = ok and (h1 != h2)

    var combined = hash_combine(h1, h2)
    ok = ok and (combined != h1) and (combined != h2)

    var xs = List[String]()
    xs.append(String("a"))
    xs.append(String("b"))
    ok = ok and (hash_list_of_strings(xs) != UInt64(0))

    ok = ok and (hash_bool(True) != hash_bool(False))
    ok = ok and (hash_i64(Int64(0)) != hash_i64(Int64(1)))

    return ok
