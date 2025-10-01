# Project:      Momijo
# Module:       learn.utils.profiling
# File:         utils/profiling.mojo
# Path:         src/momijo/learn/utils/profiling.mojo
#
# Description:  Lightweight, backend-agnostic profiling utilities for Momijo Learn.
#               Provides a high-resolution Timer that accumulates elapsed time across
#               multiple start/stop intervals, using a caller-supplied monotonic
#               timestamp in nanoseconds. No OS/FFI dependency here—wire your
#               monotonic time source elsewhere and pass it in.
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
#   - Types: Timer
#   - Key fns: start(t_ns), stop(t_ns) -> seconds, lap(t_ns) -> seconds, elapsed(t_ns)
#   - Helpers: ns_to_s, ns_to_ms, s_to_ns
#   - Design: time source is injected by the caller (t_ns = monotonic nanoseconds)


# -----------------------------------------------------------------------------
# Unit converters
# -----------------------------------------------------------------------------

@staticmethod
fn ns_to_s(ns: UInt64) -> Float64:
    return Float64(ns) / 1_000_000_000.0

@staticmethod
fn ns_to_ms(ns: UInt64) -> Float64:
    return Float64(ns) / 1_000_000.0

@staticmethod
fn s_to_ns(seconds: Float64) -> UInt64:
    # Clamp negatives to zero to stay safe.
    var s = seconds
    if s < 0.0:
        s = 0.0
    # Round to nearest integer nanosecond.
    return UInt64(s * 1_000_000_000.0)


# -----------------------------------------------------------------------------
# Timer
# -----------------------------------------------------------------------------
# Accumulates elapsed time between start/stop pairs. Can be started/stopped
# multiple times. "lap" returns time since the last start without stopping.
# All methods expect t_ns = monotonic nanoseconds supplied by the caller.

struct Timer:
    var running: Bool
    var start_ns: UInt64
    var accum_ns: UInt64

    fn __init__(out self):
        self.running = False
        self.start_ns = UInt64(0)
        self.accum_ns = UInt64(0)

    # Reset the timer to zero. If currently running, keeps it running from t_ns.
    fn reset(mut self, t_ns: UInt64, keep_running: Bool = False):
        self.accum_ns = UInt64(0)
        if keep_running and self.running:
            self.start_ns = t_ns
        else:
            self.running = False
            self.start_ns = UInt64(0)

    # Start the timer if not already running.
    fn start(mut self, t_ns: UInt64):
        if not self.running:
            self.running = True
            self.start_ns = t_ns

    # Stop the timer if running and accumulate. Returns the duration of
    # the last running segment in SECONDS. If not running, returns 0.0.
    fn stop(mut self, t_ns: UInt64) -> Float64:
        if not self.running:
            return 0.0
        var seg_ns = t_ns - self.start_ns
        self.accum_ns = self.accum_ns + seg_ns
        self.running = False
        return ns_to_s(seg_ns)

    # Lap time since the last start, without stopping. If not running, returns 0.0.
    fn lap(self, t_ns: UInt64) -> Float64:
        if not self.running:
            return 0.0
        let seg_ns = t_ns - self.start_ns
        return ns_to_s(seg_ns)

    # Total elapsed time in SECONDS (accumulated + current segment if running).
    fn elapsed(self, t_ns: UInt64) -> Float64:
        if self.running:
            let live_ns = t_ns - self.start_ns
            return ns_to_s(self.accum_ns + live_ns)
        return ns_to_s(self.accum_ns)

    # Same as elapsed, but in MILLISECONDS.
    fn elapsed_ms(self, t_ns: UInt64) -> Float64:
        if self.running:
            let live_ns = t_ns - self.start_ns
            return ns_to_ms(self.accum_ns + live_ns)
        return ns_to_ms(self.accum_ns)

    # Ensure a single-shot measure: reset -> start -> stop, returning seconds.
    # This is handy for timing an operation when you already have its start/end stamps.
    fn measure(mut self, t_start_ns: UInt64, t_end_ns: UInt64) -> Float64:
        self.running = False
        self.accum_ns = UInt64(0)
        self.start_ns = t_start_ns
        self.running = True
        return self.stop(t_end_ns)
