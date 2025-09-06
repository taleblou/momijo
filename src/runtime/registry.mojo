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
# File: src/momijo/runtime/registry.mojo

# Registry: global runtime registry for ops or components.

@fieldwise_init
struct RegistryEntry:
    var name: String
    var id: Int

    fn __init__(out self, name: String, id: Int):
        self.name = name
        self.id = id

    fn summary(self) -> String:
        return String("RegistryEntry(") + self.name + String(", ") + String(self.id) + String(")")


@fieldwise_init
struct Registry:
    var entries: List[RegistryEntry]

    fn __init__(out self):
        self.entries = List[RegistryEntry]()

    fn register(mut self, name: String, id: Int):
        var e = RegistryEntry(name, id)
        self.entries.push_back(e)

    fn find_by_name(self, name: String) -> Optional[Int]:
        var i = 0
        while i < len(self.entries):
            if self.entries[i].name == name:
                return self.entries[i].id
            i += 1
        return None

    fn find_by_id(self, id: Int) -> Optional[String]:
        var i = 0
        while i < len(self.entries):
            if self.entries[i].id == id:
                return self.entries[i].name
            i += 1
        return None

    fn list_entries(self) -> List[String]:
        var outs = List[String]()
        var i = 0
        while i < len(self.entries):
            outs.push_back(self.entries[i].summary())
            i += 1
        return outs


fn _self_test() -> Bool:
    var r = Registry()
    r.register(String("op_add"), 1)
    r.register(String("op_mul"), 2)
    var ok = True
    if r.find_by_name(String("op_add")).value() != 1:
        ok = False
    if r.find_by_id(2).value() != String("op_mul"):
        ok = False
    if len(r.list_entries()) < 2:
        ok = False
    return ok
