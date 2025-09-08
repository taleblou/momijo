# Project:      Momijo
# Module:       src.momijo.runtime.dispatcher
# File:         dispatcher.mojo
# Path:         src/momijo/runtime/dispatcher.mojo
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
#   - Structs: DispatchEntry, Dispatcher
#   - Key functions: __init__, __copyinit__, __moveinit__, __init__, register, find_by_id, find_by_name, count ...


@fieldwise_init
struct DispatchEntry:
    var id: Int
    var name: String
fn __init__(out self, id: Int, name: String) -> None:
        self.id = id
        self.name = name
fn __copyinit__(out self, other: Self) -> None:
        self.id = other.id
        self.name = other.name
fn __moveinit__(out self, deinit other: Self) -> None:
        self.id = other.id
        self.name = other.name
@fieldwise_init
struct Dispatcher:
    var entries: List[DispatchEntry]
fn __init__(out self) -> None:
        self.entries = List[DispatchEntry]()
fn register(mut self, id: Int, name: String) -> None:
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
fn __copyinit__(out self, other: Self) -> None:
        self.entries = other.entries
fn __moveinit__(out self, deinit other: Self) -> None:
        self.entries = other.entries
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