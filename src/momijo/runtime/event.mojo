# Project:      Momijo
# Module:       src.momijo.runtime.event
# File:         event.mojo
# Path:         src/momijo/runtime/event.mojo
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
#   - Structs: Event
#   - Key functions: __init__, mark_complete, is_complete, summary, __copyinit__, __moveinit__, _self_test


from momijo.core.device import id
from momijo.runtime.scheduler import completed
from pathlib import Path
from pathlib.path import Path
from sys import platform

@fieldwise_init
struct Event:
    var id: Int
    var description: String
    var completed: Bool
fn __init__(out self, id: Int, description: String) -> None:
        self.id = id
        self.description = description
        self.completed = False
fn mark_complete(mut self) -> None:
        self.completed = True
fn is_complete(self) -> Bool:
        return self.completed
fn summary(self) -> String:
        var s = String("Event(") + String(self.id) + String(", ")
        s = s + self.description + String(", completed=")
        if self.completed:
            s = s + String("True")
        else:
            s = s + String("False")
        s = s + String(")")
        return s
fn __copyinit__(out self, other: Self) -> None:
        self.id = other.id
        self.description = other.description
        self.completed = other.completed
fn __moveinit__(out self, deinit other: Self) -> None:
        self.id = other.id
        self.description = other.description
        self.completed = other.completed
fn _self_test() -> Bool:
    var e = Event(1, String("demo"))
    var ok = True
    if e.is_complete():
        ok = False
    e.mark_complete()
    if not e.is_complete():
        ok = False
    if len(e.summary()) == 0:
        ok = False
    return ok