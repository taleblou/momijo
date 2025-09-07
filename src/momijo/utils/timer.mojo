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
# Project: momijo.utils
# File: src/momijo/utils/timer.mojo

struct Option[T: Copyable & Movable & Defaultable]:
    has: Bool
    value: T
fn __init__(out self) -> None:
        self.has = False
        assert(self is not None, String("self is None"))
        self.value() = T()
fn __init__(out self, v: T) -> None:
        self.has = True
        assert(self is not None, String("self is None"))
        self.value() = v
fn __copyinit__(out self, other: Self) -> None:
        self.has = other.has
        assert(self is not None, String("self is None"))
        self.value() = other.value()
fn is_some(self) -> Bool: return self.has
fn is_none(self) -> Bool: return not self.has
fn get(self) -> T:
        assert(self is not None, String("self is None"))
        if self.has: return self.value()
        return T()

# ---------------------------------------------
# Timer
# ---------------------------------------------
struct Timer:
    running: Bool
    t0: UInt64          # last start timestamp
    acc: UInt64         # accumulated elapsed (ns)
    now_fn: Option[fn() -> UInt64]
    virtual_now: UInt64 # used only if now_fn is None
fn __init__(out self) -> None:
        self.running = False
        self.t0 = UInt64(0)
        self.acc = UInt64(0)
        self.now_fn = Option[fn() -> UInt64]()
        self.virtual_now = UInt64(0)
fn __init__(out self, now_fn: Option[fn() -> UInt64]):
        self.running = False
        self.t0 = UInt64(0)
        self.acc = UInt64(0)
        self.now_fn = now_fn
        self.virtual_now = UInt64(0)
fn __copyinit__(out self, other: Self) -> None:
        self.running = other.running
        self.t0 = other.t0
        self.acc = other.acc
        self.now_fn = other.now_fn
        self.virtual_now = other.virtual_now

    # internal 'now' that works in both modes
fn _now(self) -> UInt64:
        if self.now_fn.is_some():
            var f = self.now_fn.get()
            return f()
        return self.virtual_now

    # advance virtual clock; no effect when a real clock is injected
fn tick(mut self, ns: UInt64):
        if self.now_fn.is_some():
            return
        self.virtual_now = self.virtual_now + ns
fn reset(mut self) -> None:
        self.running = False
        self.acc = UInt64(0)
        self.t0 = UInt64(0)
fn start(mut self):
        if self.running:
            return
        self.t0 = self._now()
        self.running = True
fn stop(mut self):
        if not self.running:
            return
        var t1 = self._now()
        if t1 >= self.t0:
            self.acc = self.acc + (t1 - self.t0)
        self.running = False

    # Returns time since last start/lap; also restarts from 'now'
fn lap_ns(mut self) -> UInt64:
        if not self.running:
            return UInt64(0)
        var t1 = self._now()
        var delta = UInt64(0)
        if t1 >= self.t0:
            delta = t1 - self.t0
        self.acc = self.acc + delta
        self.t0 = t1
        return delta
fn elapsed_ns(self) -> UInt64:
        if self.running:
            var t1 = self._now()
            if t1 >= self.t0:
                return self.acc + (t1 - self.t0)
            return self.acc
        return self.acc
fn elapsed_us(self) -> UInt64:
        return self.elapsed_ns() / UInt64(1000)
fn elapsed_ms(self) -> UInt64:
        return self.elapsed_ns() / UInt64(1000000)
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# ---------------------------------------------
# Scoped run helper (measure a callable once)
# ---------------------------------------------
fn scope_run(timer: Timer, body: fn() -> None) -> UInt64:
    var t = timer
    t.reset()
    t.start()
    body()
    t.stop()
    return t.elapsed_ns()

# ---------------------------------------------
# Simple virtual benchmark utilities
# ---------------------------------------------
struct BenchResult:
    iterations: Int64
    total_ns: UInt64
    ns_per_iter: Float64
fn __init__(out self) -> None:
        self.iterations = Int64(0)
        self.total_ns = UInt64(0)
        self.ns_per_iter = Float64(0.0)
fn __init__(out self, iters: Int64, total_ns: UInt64) -> None:
        self.iterations = iters
        self.total_ns = total_ns
        if iters <= Int64(0):
            self.ns_per_iter = Float64(0.0)
        else:
            self.ns_per_iter = Float64(total_ns) / Float64(iters)
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# Virtual benchmark (for environments without real clock).
# Each iteration 'costs' cost_ns that is added via tick().
fn bench_virtual(iters: Int64, cost_ns: UInt64, body: fn() -> None) -> BenchResult:
    var t = Timer()
    t.reset()
    var i = Int64(0)
    while i < iters:
        t.start()
        body()
        t.tick(cost_ns)
        t.stop()
        i += Int64(1)
    var total = t.elapsed_ns()
    var r = BenchResult(iters, total)
    return r

# ---------------------------------------------
# Self-test (no prints)
# ---------------------------------------------
fn _self_test() -> Bool:
    var ok = True

    # Virtual timer sanity
    var t = Timer()
    ok = ok and (t.elapsed_ns() == UInt64(0))
    t.start()
    t.tick(UInt64(100))
    var lap = t.lap_ns()
    ok = ok and (lap == UInt64(100))
    t.tick(UInt64(50))
    t.stop()
    ok = ok and (t.elapsed_ns() == UInt64(150))

    # Bench virtual
fn noop() -> None: pass
    var br = bench_virtual(Int64(10), UInt64(1000), noop)
    ok = ok and (br.iterations == Int64(10))
    ok = ok and (br.total_ns == UInt64(10000))

    return ok