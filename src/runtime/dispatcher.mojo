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
# File: src/momijo/runtime/dispatcher.mojo

# Dispatcher: lightweight runtime dispatcher to invoke functions or kernels
# based on a registered id or name.

@fieldwise_init
struct DispatchEntry:
    var id: Int
    var name: String

    fn __init__(out self, id: Int, name: String):
        self.id = id
        self.name = name


@fieldwise_init
struct Dispatcher:
    var entries: List[DispatchEntry]

    fn __init__(out self):
        self.entries = List[DispatchEntry]()

    fn register(mut self, id: Int, name: String):
        var e = DispatchEntry(id, name)
        self.entries.push_back(e)

    fn find_by_id(self, id: Int) -> Optional[String]:
        var i = 0
        while i < len(self.entries):
            if self.entries[i].id == id:
                return self.entries[i].name
            i += 1
        return None

    fn find_by_name(self, name: String) -> Optional[Int]:
        var i = 0
        while i < len(self.entries):
            if self.entries[i].name == name:
                return self.entries[i].id
            i += 1
        return None

    fn count(self) -> Int:
        return len(self.entries)


fn _self_test() -> Bool:
    var d = Dispatcher()
    d.register(1, String("add"))
    d.register(2, String("mul"))
    var ok = True
    if d.count() != 2:
        ok = False
    if d.find_by_id(1).value() != String("add"):
        ok = False
    if d.find_by_name(String("mul")).value() != 2:
        ok = False
    return ok
