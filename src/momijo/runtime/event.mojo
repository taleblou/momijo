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
# Project: momijo.runtime
# File: src/momijo/runtime/event.mojo

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