# Project:      Momijo
# Module:       learn.api.sequential
# File:         api/sequential.mojo
# Path:         src/momijo/learn/api/sequential.mojo
#
# Description:  Keras/PyTorch-style Sequential container that applies a list
#               of Modules in order. Provides convenience methods for building,
#               summarizing, and (de)serializing model state. This is backend-
#               agnostic; connect layer math to momijo.tensor ops in the Modules.
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
#   - Types: Sequential
#   - Key fns: add, insert, remove_at, clear, len, get, set, forward, summary,
#              state_dict, load_state_dict, to_list
#   - Assumes each Module defines: forward(x), state_dict() -> String,
#     load_state_dict(state: String)

from collections.list import List
from momijo.learn.nn.module import Module

struct Sequential:
    # Ordered list of modules applied one after another.
    var modules: List[Module]

    fn __init__(out self):
        self.modules = List[Module]()

    # Append a module to the end.
    fn add(mut self, m: Module):
        self.modules.push_back(m)

    # Insert a module at a given index [0..len].
    fn insert(mut self, index: Int, m: Module):
        # Clamp index to [0, size]
        var i = index
        if i < 0:
            i = 0
        var n = Int(self.modules.size())
        if i > n:
            i = n
        # Manual insert since List may not expose insert; rebuild tail.
        var tail = List[Module]()
        # Move elements [i..end) to tail
        var k = i
        while k < n:
            tail.push_back(self.modules[k])
            k = k + 1
        # Truncate to i
        # (No direct truncate API assumed; rebuild a new list)
        var head = List[Module]()
        var j = 0
        while j < i:
            head.push_back(self.modules[j])
            j = j + 1
        head.push_back(m)
        # append tail back
        var t = 0
        while t < Int(tail.size()):
            head.push_back(tail[t])
            t = t + 1
        self.modules = head

    # Remove module at index (no-op if out of range).
    fn remove_at(mut self, index: Int):
        var n = Int(self.modules.size())
        if index < 0 or index >= n:
            return
        var out_list = List[Module]()
        var i = 0
        while i < n:
            if i != index:
                out_list.push_back(self.modules[i])
            i = i + 1
        self.modules = out_list

    # Clear all modules.
    fn clear(mut self):
        self.modules = List[Module]()

    # Number of modules.
    fn __len__(self) -> Int:
        return Int(self.modules.size())

    # Get module at index. (Returns a copy; adapt when reference semantics are available.)
    fn get(self, index: Int) -> Module:
        return self.modules[index]

    # Replace module at index.
    fn set(mut self, index: Int, m: Module):
        self.modules[index] = m

    # Forward pass: y = m_n(...m_2(m_1(x))...)
    # This relies on each Module having a callable `forward(x)` function.
    fn forward(self, x):
        var out = x
        var i = 0
        var n = Int(self.modules.size())
        while i < n:
            # Duck-typed call; Module.forward must exist.
            out = self.modules[i].forward(out)
            i = i + 1
        return out

    # Convenience alias, mirroring typical __call__ in other frameworks.
    fn run(self, x):
        return self.forward(x)

    # Human-readable one-line summary (module names and count).
    fn summary(self) -> String:
        var n = Int(self.modules.size())
        var s = String("Sequential(") + String(n) + String(" modules)")
        return s

    # Serialize state as a simplistic JSON-like object by concatenating child states.
    # This is a placeholder; prefer using project-wide MNP checkpoint utilities.
    fn state_dict(self) -> String:
        var n = Int(self.modules.size())
        var s = String("{\"type\":\"Sequential\",\"num_modules\":") + String(n) + String(",\"children\":[")
        var i = 0
        while i < n:
            s = s + self.modules[i].state_dict()
            if i + 1 < n:
                s = s + String(",")
            i = i + 1
        s = s + String("]}")
        return s

    # Load state by distributing chunks to children in order.
    # For now expects a pre-split approach at higher level; this stub keeps API shape.
    fn load_state_dict(mut self, state: String):
        # TODO: integrate with utils.checkpoint to parse and dispatch per child.
        # Placeholder: no-op to keep compatibility until checkpoint utilities are wired.
        pass

    # Return an immutable snapshot of internal list (copy).
    fn to_list(self) -> List[Module]:
        var out_list = List[Module]()
        var i = 0
        var n = Int(self.modules.size())
        while i < n:
            out_list.push_back(self.modules[i])
            i = i + 1
        return out_list
