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
# File: src/momijo/runtime/scheduler.mojo

# Scheduler: simple runtime scheduler to order and execute tasks.

@fieldwise_init
struct Task:
    var id: Int
    var name: String
    var done: Bool

    fn __init__(out self, id: Int, name: String):
        self.id = id
        self.name = name
        self.done = False

    fn mark_done(mut self):
        self.done = True

    fn summary(self) -> String:
        var s = String("Task(") + String(self.id) + String(", ") + self.name + String(", done=")
        if self.done:
            s = s + String("True")
        else:
            s = s + String("False")
        s = s + String(")")
        return s


@fieldwise_init
struct Scheduler:
    var tasks: List[Task]

    fn __init__(out self):
        self.tasks = List[Task]()

    fn add_task(mut self, id: Int, name: String):
        var t = Task(id, name)
        self.tasks.push_back(t)

    fn mark_done(mut self, id: Int):
        var i = 0
        while i < len(self.tasks):
            if self.tasks[i].id == id:
                self.tasks[i].mark_done()
                return
            i += 1

    fn pending(self) -> List[String]:
        var outs = List[String]()
        var i = 0
        while i < len(self.tasks):
            if not self.tasks[i].done:
                outs.push_back(self.tasks[i].summary())
            i += 1
        return outs

    fn completed(self) -> List[String]:
        var outs = List[String]()
        var i = 0
        while i < len(self.tasks):
            if self.tasks[i].done:
                outs.push_back(self.tasks[i].summary())
            i += 1
        return outs


fn _self_test() -> Bool:
    var s = Scheduler()
    s.add_task(1, String("download"))
    s.add_task(2, String("process"))
    var ok = True
    if len(s.pending()) != 2:
        ok = False
    s.mark_done(1)
    if len(s.completed()) != 1:
        ok = False
    return ok
