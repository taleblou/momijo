# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.utils.profiling
# File:         src/momijo/learn/utils/profiling.mojo
#
# Description:
#   Lightweight, backend-agnostic profiling utilities for Momijo Learn.
#   - Unit converters (ns<->s/ms)
#   - Timer: start/stop/lap/elapsed with external monotonic ns timestamps
#   - ScopedTimer: RAII helper for a Timer
#   - LapStats: aggregate stats for many laps (count/min/max/total/mean)
#   - LabelProfiler: named sections (e.g., "forward"/"backward"/"step")
#   - ScopedSection: RAII helper to profile a named section
#   - SMA: fixed-window simple moving average in nanoseconds
#   - TrainingLoopProfiler: train-loop profiler with standard labels + SMA.
#
# Design:
#   This module never queries the system clock. Pass monotonic t_ns (UInt64)
#   from the caller/platform layer.

from collections.list import List
# Optional integration (kept unused here to avoid hard coupling):
from momijo.tensor import tensor

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
    var s = seconds
    if s < 0.0:
        s = 0.0
    # Clamp to UInt64 range is implicit in typical workloads; add guards if needed.
    return UInt64(s * 1_000_000_000.0)

# -----------------------------------------------------------------------------
# Timer
# -----------------------------------------------------------------------------

struct Timer:
    var running: Bool
    var start_ns: UInt64
    var accum_ns: UInt64

    fn __init__(out self):
        self.running = False
        self.start_ns = UInt64(0)
        self.accum_ns = UInt64(0)

    fn reset(mut self, t_ns: UInt64, keep_running: Bool = False):
        self.accum_ns = UInt64(0)
        if keep_running and self.running:
            self.start_ns = t_ns
        else:
            self.running = False
            self.start_ns = UInt64(0)

    fn start(mut self, t_ns: UInt64):
        if not self.running:
            self.running = True
            self.start_ns = t_ns

    fn stop(mut self, t_ns: UInt64) -> Float64:
        if not self.running:
            return 0.0
        var seg_ns = t_ns - self.start_ns
        self.accum_ns = self.accum_ns + seg_ns
        self.running = False
        return ns_to_s(seg_ns)

    fn lap(self, t_ns: UInt64) -> Float64:
        if not self.running:
            return 0.0
        var seg_ns = t_ns - self.start_ns
        return ns_to_s(seg_ns)

    fn elapsed(self, t_ns: UInt64) -> Float64:
        if self.running:
            var live_ns = t_ns - self.start_ns
            return ns_to_s(self.accum_ns + live_ns)
        return ns_to_s(self.accum_ns)

    fn elapsed_ms(self, t_ns: UInt64) -> Float64:
        if self.running:
            var live_ns = t_ns - self.start_ns
            return ns_to_ms(self.accum_ns + live_ns)
        return ns_to_ms(self.accum_ns)

    fn measure(mut self, t_start_ns: UInt64, t_end_ns: UInt64) -> Float64:
        self.running = False
        self.accum_ns = UInt64(0)
        self.start_ns = t_start_ns
        self.running = True
        return self.stop(t_end_ns)

# -----------------------------------------------------------------------------
# ScopedTimer (RAII helper for Timer)
# -----------------------------------------------------------------------------

struct ScopedTimer:
    var timer_ref: Pointer[Timer]
    var started: Bool

    fn __init__(out self, timer_ref: Pointer[Timer], t_ns: UInt64):
        self.timer_ref = timer_ref
        self.started = False
        if self.timer_ref is not None:
            if not self.timer_ref.value.running:
                self.timer_ref.value.start(t_ns)
                self.started = True

    fn stop(mut self, t_ns: UInt64) -> Float64:
        if self.timer_ref is None:
            return 0.0
        if self.started and self.timer_ref.value.running:
            self.started = False
            return self.timer_ref.value.stop(t_ns)
        return 0.0

    fn __del__(deinit self):
        # No-op: we don't know t_end_ns here and must not guess.
        pass

# -----------------------------------------------------------------------------
# LapStats: aggregate statistics for multiple laps (nanoseconds domain)
# -----------------------------------------------------------------------------

struct LapStats:
    var count: Int
    var total_ns: UInt64
    var min_ns: UInt64
    var max_ns: UInt64

    fn __init__(out self):
        self.count = 0
        self.total_ns = UInt64(0)
        self.min_ns = UInt64(0)
        self.max_ns = UInt64(0)

    fn add_lap(mut self, lap_ns: UInt64):
        self.total_ns = self.total_ns + lap_ns
        if self.count == 0:
            self.min_ns = lap_ns
            self.max_ns = lap_ns
        else:
            if lap_ns < self.min_ns:
                self.min_ns = lap_ns
            if lap_ns > self.max_ns:
                self.max_ns = lap_ns
        self.count = self.count + 1

    fn mean_ns(self) -> Float64:
        if self.count <= 0:
            return 0.0
        return Float64(self.total_ns) / Float64(self.count)

    fn mean_ms(self) -> Float64:
        # Compute in Float to avoid truncation.
        return self.mean_ns() / 1_000_000.0

    fn mean_s(self) -> Float64:
        return self.mean_ns() / 1_000_000_000.0

    fn total_s(self) -> Float64:
        return ns_to_s(self.total_ns)

    fn min_s(self) -> Float64:
        return ns_to_s(self.min_ns)

    fn max_s(self) -> Float64:
        return ns_to_s(self.max_ns)

    fn reset(mut self):
        self.count = 0
        self.total_ns = UInt64(0)
        self.min_ns = UInt64(0)
        self.max_ns = UInt64(0)

    fn __str__(self) -> String:
        var s = "LapStats(count=" + String(self.count)
        s = s + ", total_s=" + String(self.total_s())
        s = s + ", mean_s=" + String(self.mean_s())
        s = s + ", min_s=" + String(self.min_s())
        s = s + ", max_s=" + String(self.max_s()) + ")"
        return s

# -----------------------------------------------------------------------------
# LabelProfiler: named sections aggregator
# -----------------------------------------------------------------------------

struct LabelProfiler:
    var labels: List[String]
    var stats: List[LapStats]

    fn __init__(out self):
        self.labels = List[String]()
        self.stats = List[LapStats]()

    fn _find_index(self, label: String) -> Int:
        var i = 0
        var n = len(self.labels)
        while i < n:
            if self.labels[i] == label:
                return i
            i = i + 1
        return -1

    fn clear(mut self):
        self.labels = List[String]()
        self.stats = List[LapStats]()

    fn add_lap_ns(mut self, label: String, lap_ns: UInt64):
        var idx = self._find_index(label)
        if idx < 0:
            self.labels.push_back(label)
            var st = LapStats()
            st.add_lap(lap_ns)
            self.stats.push_back(st)
            return
        var cur = self.stats[idx]
        cur.add_lap(lap_ns)
        self.stats[idx] = cur

    fn add_lap_s(mut self, label: String, lap_s: Float64):
        var lap_ns = s_to_ns(lap_s)
        self.add_lap_ns(label, lap_ns)

    fn summary(self) -> List[String]:
        var out = List[String]()
        var i = 0
        var n = len(self.labels)
        while i < n:
            var name = self.labels[i]
            var st = self.stats[i]
            var line = String(" - ") + name + ": count=" + String(st.count)
            line = line + ", total_s=" + String(st.total_s())
            line = line + ", mean_s=" + String(st.mean_s())
            line = line + ", min_s=" + String(st.min_s())
            line = line + ", max_s=" + String(st.max_s())
            out.push_back(line)
            i = i + 1
        return out

    fn __str__(self) -> String:
        var lines = self.summary()
        var s = "LabelProfiler:\n"
        var i = 0
        var n = len(lines)
        while i < n:
            s = s + lines[i] + "\n"
            i = i + 1
        return s

# -----------------------------------------------------------------------------
# ScopedSection: RAII helper for named sections
# -----------------------------------------------------------------------------

struct ScopedSection:
    var profiler_ref: Pointer[LabelProfiler]
    var label: String
    var t_start_ns: UInt64
    var open: Bool

    fn __init__(out self, profiler_ref: Pointer[LabelProfiler], label: String, t_start_ns: UInt64):
        self.profiler_ref = profiler_ref
        self.label = label
        self.t_start_ns = t_start_ns
        self.open = True

    fn end(mut self, t_end_ns: UInt64):
        if not self.open:
            return
        self.open = False
        if self.profiler_ref is not None:
            var lap_ns = t_end_ns - self.t_start_ns
            self.profiler_ref.value.add_lap_ns(self.label, lap_ns)

    fn __del__(deinit self):
        # No-op: without a provided end timestamp we cannot record a correct lap.
        pass

# -----------------------------------------------------------------------------
# SMA: fixed-window Simple Moving Average for nanoseconds
# -----------------------------------------------------------------------------

struct SMA:
    var capacity: Int
    var count: Int
    var index: Int
    var sum_ns: UInt64
    var buf: List[UInt64]

    fn __init__(out self, capacity: Int):
        var cap = capacity
        if cap < 1:
            cap = 1
        self.capacity = cap
        self.count = 0
        self.index = 0
        self.sum_ns = UInt64(0)
        self.buf = List[UInt64]()
        var i = 0
        while i < cap:
            self.buf.push_back(UInt64(0))
            i = i + 1

    fn reset(mut self):
        self.count = 0
        self.index = 0
        self.sum_ns = UInt64(0)
        var i = 0
        while i < self.capacity:
            self.buf[i] = UInt64(0)
            i = i + 1

    fn add(mut self, value_ns: UInt64):
        if self.count < self.capacity:
            self.sum_ns = self.sum_ns + value_ns
            self.buf[self.index] = value_ns
            self.index = (self.index + 1) % self.capacity
            self.count = self.count + 1
            return
        var old = self.buf[self.index]
        self.sum_ns = self.sum_ns + value_ns - old
        self.buf[self.index] = value_ns
        self.index = (self.index + 1) % self.capacity

    fn mean_s(self) -> Float64:
        if self.count <= 0:
            return 0.0
        return (Float64(self.sum_ns) / Float64(self.count)) / 1_000_000_000.0

    fn mean_ms(self) -> Float64:
        if self.count <= 0:
            return 0.0
        return (Float64(self.sum_ns) / Float64(self.count)) / 1_000_000.0

# -----------------------------------------------------------------------------
# TrainingLoopProfiler: end-to-end helper for train loops
# -----------------------------------------------------------------------------

struct TrainingLoopProfiler:
    var window: Int
    var labels: List[String]      # fixed order
    var stats: List[LapStats]     # per label cumulative
    var smas: List[SMA]           # per label SMA window
    var current_start_ns: UInt64  # last begin timestamp for active label
    var current_label_idx: Int    # -1 if none active

    fn __init__(out self, window: Int = 50):
        var win = window
        if win < 1:
            win = 1
        self.window = win
        self.labels = List[String]()
        self.stats = List[LapStats]()
        self.smas = List[SMA]()
        self.current_start_ns = UInt64(0)
        self.current_label_idx = -1

        self._add_label("data")
        self._add_label("forward")
        self._add_label("backward")
        self._add_label("step")
        self._add_label("misc")

    fn _add_label(mut self, name: String):
        self.labels.push_back(name)
        var st = LapStats()
        self.stats.push_back(st)
        var sm = SMA(self.window)
        self.smas.push_back(sm)

    fn _find_index(self, label: String) -> Int:
        var i = 0
        var n = len(self.labels)
        while i < n:
            if self.labels[i] == label:
                return i
            i = i + 1
        return -1

    fn begin(mut self, label: String, t_ns: UInt64):
        var idx = self._find_index(label)
        if idx < 0:
            self._add_label(label)
            idx = len(self.labels) - 1
        self.current_label_idx = idx
        self.current_start_ns = t_ns

    fn end(mut self, t_ns: UInt64):
        if self.current_label_idx < 0:
            return
        var lap_ns = t_ns - self.current_start_ns
        var st = self.stats[self.current_label_idx]
        st.add_lap(lap_ns)
        self.stats[self.current_label_idx] = st
        var sm = self.smas[self.current_label_idx]
        sm.add(lap_ns)
        self.smas[self.current_label_idx] = sm
        self.current_label_idx = -1
        self.current_start_ns = UInt64(0)

    fn begin_data(mut self, t_ns: UInt64): self.begin("data", t_ns)
    fn end_data(mut self, t_ns: UInt64): self.end(t_ns)

    fn begin_forward(mut self, t_ns: UInt64): self.begin("forward", t_ns)
    fn end_forward(mut self, t_ns: UInt64): self.end(t_ns)

    fn begin_backward(mut self, t_ns: UInt64): self.begin("backward", t_ns)
    fn end_backward(mut self, t_ns: UInt64): self.end(t_ns)

    fn begin_step(mut self, t_ns: UInt64): self.begin("step", t_ns)
    fn end_step(mut self, t_ns: UInt64): self.end(t_ns)

    fn begin_misc(mut self, t_ns: UInt64): self.begin("misc", t_ns)
    fn end_misc(mut self, t_ns: UInt64): self.end(t_ns)

    fn clear(mut self):
        var i = 0
        var n = len(self.stats)
        while i < n:
            var st = self.stats[i]
            st.reset()
            self.stats[i] = st
            var sm = self.smas[i]
            sm.reset()
            self.smas[i] = sm
            i = i + 1
        self.current_label_idx = -1
        self.current_start_ns = UInt64(0)

    fn summary_lines(self) -> List[String]:
        var out = List[String]()
        var i = 0
        var n = len(self.labels)
        while i < n:
            var name = self.labels[i]
            var st = self.stats[i]
            var sm = self.smas[i]
            var line = String(" - ") + name
            line = line + ": count=" + String(st.count)
            line = line + ", total_s=" + String(st.total_s())
            line = line + ", mean_s=" + String(st.mean_s())
            line = line + ", sma_s=" + String(sm.mean_s())
            out.push_back(line)
            i = i + 1
        return out

    fn __str__(self) -> String:
        var lines = self.summary_lines()
        var s = "TrainingLoopProfiler:\n"
        var i = 0
        var n = len(lines)
        while i < n:
            s = s + lines[i] + "\n"
            i = i + 1
        return s
