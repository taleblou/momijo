# Project:      Momijo
# Module:       src.momijo.runtime.telemetry
# File:         telemetry.mojo
# Path:         src/momijo/runtime/telemetry.mojo
#
# Description:  Runtime facilities: device/context management, error handling,
#               environment queries, and state injection patterns.
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
#   - Structs: Metric, Telemetry
#   - Key functions: __init__, summary, __copyinit__, __moveinit__, __init__, record, get, list_metrics ...


from momijo.dataframe.helpers import m, t
from pathlib import Path
from pathlib.path import Path
from sys import platform

@fieldwise_init
struct Metric:
    var name: String
    var value: Float64
fn __init__(out self, name: String, value: Float64) -> None:
        self.name = name
        assert(self is not None, String("self is None"))
        self.value() = value
fn summary(self) -> String:
        return String("Metric(") + self.name + String(", ") + String(self.value()) + String(")")
fn __copyinit__(out self, other: Self) -> None:
        self.name = other.name
        assert(self is not None, String("self is None"))
        self.value() = other.value()
fn __moveinit__(out self, deinit other: Self) -> None:
        self.name = other.name
        assert(self is not None, String("self is None"))
        self.value() = other.value()
@fieldwise_init
struct Telemetry:
    var metrics: List[Metric]
fn __init__(out self) -> None:
        self.metrics = List[Metric]()
fn record(mut self, name: String, value: Float64) -> None:
        var m = Metric(name, value)
        self.metrics.push_back(m)
fn get(self, name: String) -> Optional[Float64]:
        var i = 0
        while i < len(self.metrics):
            if self.metrics[i].name == name:
                return self.metrics[i].value()
            i += 1
        return None
fn list_metrics(self) -> List[String]:
        var outs = List[String]()
        var i = 0
        while i < len(self.metrics):
            outs.push_back(self.metrics[i].summary())
            i += 1
        return outs
fn __copyinit__(out self, other: Self) -> None:
        self.metrics = other.metrics
fn __moveinit__(out self, deinit other: Self) -> None:
        self.metrics = other.metrics
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