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
# File: src/momijo/runtime/event.mojo

# Event: simple runtime event signaling and status tracking.

@fieldwise_init
struct Event:
    var id: Int
    var description: String
    var completed: Bool

    fn __init__(out self, id: Int, description: String):
        self.id = id
        self.description = description
        self.completed = False

    fn mark_complete(mut self):
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
