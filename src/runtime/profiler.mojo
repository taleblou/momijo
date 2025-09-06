# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.runtime
# File: src/momijo/runtime/profiler.mojo

# Profiler: lightweight runtime profiler to measure op execution times.

@fieldwise_init
struct ProfileEntry:
    var name: String
    var duration_ms: Float64

    fn __init__(out self, name: String, duration_ms: Float64):
        self.name = name
        self.duration_ms = duration_ms

    fn summary(self) -> String:
        return String("ProfileEntry(") + self.name + String(", ") + String(self.duration_ms) + String("ms)")


@fieldwise_init
struct Profiler:
    var entries: List[ProfileEntry]

    fn __init__(out self):
        self.entries = List[ProfileEntry]()

    fn record(mut self, name: String, duration_ms: Float64):
        var e = ProfileEntry(name, duration_ms)
        self.entries.push_back(e)

    fn report(self) -> List[String]:
        var outs = List[String]()
        var i = 0
        while i < len(self.entries):
            outs.push_back(self.entries[i].summary())
            i += 1
        return outs

    fn average_time(self, name: String) -> Float64:
        var total = 0.0
        var count = 0
        var i = 0
        while i < len(self.entries):
            if self.entries[i].name == name:
                total += self.entries[i].duration_ms
                count += 1
            i += 1
        if count == 0:
            return 0.0
        return total / count


fn _self_test() -> Bool:
    var p = Profiler()
    p.record(String("add"), 1.2)
    p.record(String("add"), 0.8)
    var avg = p.average_time(String("add"))
    var ok = True
    if avg < 0.9 or avg > 1.1:
        ok = False
    if len(p.report()) == 0:
        ok = False
    return ok
