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
# File: src/momijo/runtime/telemetry.mojo

# Telemetry: runtime collection of metrics and statistics.

@fieldwise_init
struct Metric:
    var name: String
    var value: Float64

    fn __init__(out self, name: String, value: Float64):
        self.name = name
        self.value = value

    fn summary(self) -> String:
        return String("Metric(") + self.name + String(", ") + String(self.value) + String(")")


@fieldwise_init
struct Telemetry:
    var metrics: List[Metric]

    fn __init__(out self):
        self.metrics = List[Metric]()

    fn record(mut self, name: String, value: Float64):
        var m = Metric(name, value)
        self.metrics.push_back(m)

    fn get(self, name: String) -> Optional[Float64]:
        var i = 0
        while i < len(self.metrics):
            if self.metrics[i].name == name:
                return self.metrics[i].value
            i += 1
        return None

    fn list_metrics(self) -> List[String]:
        var outs = List[String]()
        var i = 0
        while i < len(self.metrics):
            outs.push_back(self.metrics[i].summary())
            i += 1
        return outs


fn _self_test() -> Bool:
    var t = Telemetry()
    t.record(String("latency"), 12.5)
    t.record(String("throughput"), 100.0)
    var ok = True
    if t.get(String("latency")).value() != 12.5:
        ok = False
    if len(t.list_metrics()) < 2:
        ok = False
    return ok
