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
# File: src/momijo/runtime/stream.mojo

# Stream: runtime abstraction for execution streams (e.g. command queues).

@fieldwise_init
struct Stream:
    var id: Int
    var name: String
    var tasks: List[String]

    fn __init__(out self, id: Int, name: String):
        self.id = id
        self.name = name
        self.tasks = List[String]()

    fn enqueue(mut self, task: String):
        self.tasks.push_back(task)

    fn dequeue(mut self) -> Optional[String]:
        if len(self.tasks) == 0:
            return None
        var t = self.tasks[0]
        # shift left
        var new_tasks = List[String]()
        var i = 1
        while i < len(self.tasks):
            new_tasks.push_back(self.tasks[i])
            i += 1
        self.tasks = new_tasks
        return t

    fn pending(self) -> Int:
        return len(self.tasks)

    fn summary(self) -> String:
        return String("Stream(") + String(self.id) + String(", ") + self.name + String(", pending=") + String(len(self.tasks)) + String(")")


fn _self_test() -> Bool:
    var s = Stream(1, String("main"))
    s.enqueue(String("op1"))
    s.enqueue(String("op2"))
    var ok = True
    if s.pending() != 2:
        ok = False
    var first = s.dequeue().value()
    if first != String("op1"):
        ok = False
    if s.pending() != 1:
        ok = False
    return ok
