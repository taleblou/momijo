# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.utils
# File: src/momijo/utils/thread_pool.mojo
#
# Checklist compliance:
# - Only 'var' (no 'let'); no 'export'; no global mutable state
# - Explicit imports; English comments
# - Prefer mut/out over inout
# - Struct constructors via `fn __init__(out self, ...)`
# - Deterministic behavior; no exceptions; no prints in library
#
# NOTE:
# Mojo's stable stdlib may not expose full OS-thread primitives in all environments.
# This module provides a *compatibility* thread-pool facade with a sequential fallback.
# The API is designed so that a future true multi-threaded backend can replace internals
# without breaking call sites.
#
# Design:
#   - ThreadPool holds only configuration (worker count).
#   - All operations are currently sequential but chunk-aware for API parity.
#   - Provided helpers: parallel_for over index ranges, chunked mapping/reduction utilities,
#     and convenience 'submit_*' facades that run immediately.
#
# IMPORTANT:
#   - Callbacks must be pure (no shared mutable global state).
#   - Indexing uses Int64; ranges are [start, end) exclusive on 'end'.

from stdlib.list import List
from stdlib.string import String

# ---------------------------------------------
# Utilities
# ---------------------------------------------

fn _max(a: Int64, b: Int64) -> Int64:
    if a >= b: return a
    return b

fn _min(a: Int64, b: Int64) -> Int64:
    if a <= b: return a
    return b

# Guess CPU count when not available (portable fallback)
fn cpu_count_guess() -> Int64:
    return Int64(1)

# Split total items into nearly-equal contiguous chunks
fn chunk_ranges(total: Int64, nchunks: Int64) -> List[(Int64, Int64)]:
    var result = List[(Int64, Int64)]()
    var n = total
    var k = nchunks
    if n <= Int64(0) or k <= Int64(0):
        return result
    if k > n: k = n
    var base = n / k
    var extra = n % k
    var start = Int64(0)
    var i = Int64(0)
    while i < k:
        var len = base + (Int64(1) if i < extra else Int64(0))
        var end = start + len
        result.append((start, end))
        start = end
        i += Int64(1)
    return result

# ---------------------------------------------
# ThreadPool (compat facade; sequential backend)
# ---------------------------------------------

struct ThreadPool:
    workers: Int64

    fn __init__(out self, workers: Int64 = Int64(0)):
        var w = workers
        if w <= Int64(0):
            w = cpu_count_guess()
        self.workers = w

    fn __copyinit__(out self, other: Self):
        self.workers = other.workers

    fn set_workers(mut self, workers: Int64):
        var w = workers
        if w <= Int64(0):
            w = cpu_count_guess()
        self.workers = w

    # -----------------------------------------
    # Core: parallel_for over n items (0..n-1)
    # body: fn(i: Int64) -> None
    # -----------------------------------------
    fn parallel_for(self, n: Int64, body: fn(Int64) -> None):
        if n <= Int64(0): return
        # Sequential fallback; honors the contract
        var i = Int64(0)
        while i < n:
            body(i)
            i += Int64(1)

    # Convenience: parallel_for over [start, end)
    fn parallel_for_range(self, start: Int64, end: Int64, body: fn(Int64) -> None):
        var a = start
        var b = end
        if b < a:
            var t = a
            a = b
            b = t
        if a == b: return
        var i = a
        while i < b:
            body(i)
            i += Int64(1)

    # Submit-like API (immediate execution; returns an ID counter)
    fn submit_for(self, n: Int64, body: fn(Int64) -> None) -> Int64:
        self.parallel_for(n, body)
        return Int64(1)

    fn submit_range(self, start: Int64, end: Int64, body: fn(Int64) -> None) -> Int64:
        self.parallel_for_range(start, end, body)
        return Int64(1)

    # Wait/shutdown are no-ops in sequential fallback
    fn wait(self):
        pass

    fn shutdown(self):
        pass

    # -----------------------------------------
    # Map helpers (Float64 / Int64)
    # -----------------------------------------
    fn map_f64(self, xs: List[Float64], f: fn(Float64) -> Float64) -> List[Float64]:
        var n = Int64(len(xs))
        var out = List[Float64]()
        var i = Int64(0)
        while i < n:
            out.append(f(xs[i]))
            i += Int64(1)
        return out

    fn map_i64(self, xs: List[Int64], f: fn(Int64) -> Int64) -> List[Int64]:
        var n = Int64(len(xs))
        var out = List[Int64]()
        var i = Int64(0)
        while i < n:
            out.append(f(xs[i]))
            i += Int64(1)
        return out

    # -----------------------------------------
    # Reduce helpers (sum)
    # -----------------------------------------
    fn reduce_sum_f64(self, xs: List[Float64]) -> Float64:
        var acc = Float64(0.0)
        var i = Int64(0)
        var n = Int64(len(xs))
        while i < n:
            acc = acc + xs[i]
            i += Int64(1)
        return acc

    fn reduce_sum_i64(self, xs: List[Int64]) -> Int64:
        var acc = Int64(0)
        var i = Int64(0)
        var n = Int64(len(xs))
        while i < n:
            acc = acc + xs[i]
            i += Int64(1)
        return acc

    # -----------------------------------------
    # Map-reduce convenience
    # -----------------------------------------
    fn map_reduce_f64(self, xs: List[Float64], f: fn(Float64) -> Float64) -> Float64:
        var i = Int64(0)
        var n = Int64(len(xs))
        var acc = Float64(0.0)
        while i < n:
            acc = acc + f(xs[i])
            i += Int64(1)
        return acc

    fn map_reduce_i64(self, xs: List[Int64], f: fn(Int64) -> Int64) -> Int64:
        var i = Int64(0)
        var n = Int64(len(xs))
        var acc = Int64(0)
        while i < n:
            acc = acc + f(xs[i])
            i += Int64(1)
        return acc

# ---------------------------------------------
# Factories
# ---------------------------------------------

fn thread_pool_default() -> ThreadPool:
    var p = ThreadPool(cpu_count_guess())
    return p

fn thread_pool_with(workers: Int64) -> ThreadPool:
    var p = ThreadPool(workers)
    return p

# ---------------------------------------------
# Self-test (no prints)
# ---------------------------------------------

fn _self_test() -> Bool:
    var ok = True

    var tp = thread_pool_with(Int64(4))

    # parallel_for sanity
    var n = Int64(5)
    var seen = List[Int64]()
    fn collect(i: Int64) -> None:
        seen.append(i)
    tp.parallel_for(n, collect)
    ok = ok and (Int64(len(seen)) == n)

    # map/reduce
    var xs = List[Int64](); xs.append(Int64(1)); xs.append(Int64(2)); xs.append(Int64(3))
    fn inc(x: Int64) -> Int64: return x + Int64(1)
    var ys = tp.map_i64(xs, inc)
    ok = ok and (len(ys) == len(xs))

    var s = tp.reduce_sum_i64(xs)
    ok = ok and (s == Int64(6))

    return ok
