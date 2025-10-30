# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.api.sequential
# File:         src/momijo/learn/api/sequential.mojo
#
# Description:
#   Keras/PyTorch-style Sequential container that applies a list of Modules
#   in order. Provides basic list-like mutation (add/insert/remove), safe
#   accessors, forward/run alias, simple string summary, and minimal state
#   (de)serialization stubs. Backend-agnostic: each child Module is expected
#   to implement `forward(x)` and optional `state_dict()` / `load_state_dict`.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.learn.nn.module import Module
from momijo.learn.utils.summary import Summarizer

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

@always_inline
fn _clamp_index(i: Int, lo: Int, hi: Int) -> Int:
    # Clamps i into [lo, hi].
    var v = i
    if v < lo:
        v = lo
    if v > hi:
        v = hi
    return v

# -----------------------------------------------------------------------------
# Sequential
# -----------------------------------------------------------------------------

struct Sequential:
    # Ordered list of modules applied one after another.
    var modules: List[Module]

    fn __init__(out self):
        self.modules = List[Module]()

    # -------------------------- mutation --------------------------

    # Append a module at the end.
    fn add(mut self, m: Module):
        self.modules.push_back(m)

    # Insert a module at a given index [0..len].
    # If index < 0 it becomes 0; if index > len it becomes len (append).
    fn insert(mut self, index: Int, m: Module):
        var n = Int(self.modules.size())
        var i = _clamp_index(index, 0, n)

        # Manual insert (rebuild list) to avoid relying on List.insert availability.
        var out_list = List[Module]()
        var k = 0
        while k < i:
            out_list.push_back(self.modules[k])
            k = k + 1
        out_list.push_back(m)
        while k < n:
            out_list.push_back(self.modules[k])
            k = k + 1
        self.modules = out_list

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

    # Replace module at index (no-op if out of range).
    fn set(mut self, index: Int, m: Module):
        var n = Int(self.modules.size())
        if index < 0 or index >= n:
            return
        self.modules[index] = m

    # -------------------------- accessors --------------------------

    # Number of modules.
    fn __len__(self) -> Int:
        return Int(self.modules.size())

    # Get module at index (returns a copy of the value).
    # If out-of-range, returns the last element when available, otherwise a default Module().
    fn get(self, index: Int) -> Module:
        var n = Int(self.modules.size())
        if n == 0:
            # Return a default-constructed Module for safety.
            var dummy = Module()
            return dummy
        var i = _clamp_index(index, 0, n - 1)
        return self.modules[i]

    # Return a shallow copy of the internal list.
    fn to_list(self) -> List[Module]:
        var out_list = List[Module]()
        var i = 0
        var n = Int(self.modules.size())
        while i < n:
            out_list.push_back(self.modules[i])
            i = i + 1
        return out_list

    # -------------------------- execution --------------------------

    # Forward pass: y = m_n(...m_2(m_1(x))...)
    # Relies on each Module exposing `forward(x)`.
    fn forward(self, x):
        var out = x
        var i = 0
        var n = Int(self.modules.size())
        while i < n:
            out = self.modules[i].forward(out)
            i = i + 1
        return out

    # Alias mirroring typical __call__ in other frameworks.
    fn run(self, x):
        return self.forward(x)

    # -------------------------- introspection --------------------------

    # Human-readable one-line summary.
    fn summary(self) -> String:
        var n = Int(self.modules.size())
        var s = String("Sequential(") + String(n) + String(" modules)")
        return s

    # Detailed summarization via Summarizer (builder pattern).
    # Each child is expected to provide `summarize(self, s: Pointer[Summarizer])`.
    fn summarize(self, s: Pointer[Summarizer]):
        var i = 0
        var n = Int(self.modules.size())
        while i < n:
            # Duck-typed: child.summarize(&s) if available.
            self.modules[i].summarize(s)
            i = i + 1

    # -------------------------- (de)serialization --------------------------

    # Minimal, JSON-like aggregation of child states. For production, prefer the
    # project-wide checkpoint utilities (MNP format) and per-module state dicts.
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

    # Placeholder loader. Keep API shape stable; integrate with utils.checkpoint later.
    fn load_state_dict(mut self, state: String):
        # TODO: parse `state`, split into child chunks, and dispatch to each child's
        #       load_state_dict. This stub intentionally does nothing for now.
        pass

    # -------------------------- stringification --------------------------

    # Pretty string with module list (single line).
    fn __str__(self) -> String:
        var n = Int(self.modules.size())
        var s = String("Sequential[")
        var i = 0
        while i < n:
            s = s + self.modules[i].__str__()
            if i + 1 < n:
                s = s + String(", ")
            i = i + 1
        s = s + String("]")
        return s
