# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.nn.module
# File:         src/momijo/learn/nn/module.mojo
#
# Description:
#   PyTorch-like base Module for Momijo Learn. Supports hierarchical composition
#   (submodules), parameter/buffer registration, training/eval mode switching,
#   and flat state_dict export with fully-qualified names (e.g., "layer1.weight").
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List
from momijo.tensor import tensor   # ← only explicit, no wildcards

# -----------------------------
# Minimal JSON string escaping
# -----------------------------
fn _json_escape(s: String) -> String:
    var out = String("")
    var i = 0
    while i < s.__len__():
        var ch = s.__getitem__(i)
        if ch == String("\\"):
            out += String("\\\\")
        elif ch == String("\""):
            out += String("\\\"")
        elif ch == String("\n"):
            out += String("\\n")
        elif ch == String("\r"):
            out += String("\\r")
        elif ch == String("\t"):
            out += String("\\t")
        else:
            out += ch
        i = i + 1
    return out

# -----------------------------
# Small tensor summary (metadata-only)
# -----------------------------
fn _tensor_summary(t: tensor.Tensor[Float64]) -> String:
    # NOTE: print only scalars/String; do not dump raw buffers here.
    var s = String("tensor(")
    s += String("dtype=Float64")
    s += String(", shape=[")
    var nd = t.ndim()
    var d = 0
    while d < nd:
        s += String(t.shape_dim(d))
        if d + 1 < nd:
            s += String(",")
        d = d + 1
    s += String("])")
    return s

# -----------------------------
# Base Module
# -----------------------------
struct Module:
    # Identity
    var name: String
    var training: Bool

    # Hierarchy
    var child_names: List[String]
    var children: List[Module]

    # Learnable params
    var param_names: List[String]
    var param_values: List[tensor.Tensor[Float64]]
    var param_requires_grad: List[Bool]

    # Non-learnable buffers
    var buffer_names: List[String]
    var buffer_values: List[tensor.Tensor[Float64]]

    fn __init__(out self, name: String = String("")):
        self.name = name
        self.training = True

        self.child_names = List[String]()
        self.children = List[Module]()

        self.param_names = List[String]()
        self.param_values = List[tensor.Tensor[Float64]]()
        self.param_requires_grad = List[Bool]()

        self.buffer_names = List[String]()
        self.buffer_values = List[tensor.Tensor[Float64]]()

    # ---------------------------
    # Composition / registration
    # ---------------------------

    fn add_module(mut self, name: String, m: Module):
        var i = 0
        while i < self.child_names.size():
            if self.child_names.__getitem__(i) == name:
                self.children.__setitem__(i, m)
                return
            i = i + 1
        self.child_names.push_back(name)
        self.children.push_back(m)

    fn modules(self) -> List[Module]:
        return self.children

    fn register_parameter(mut self, name: String, value: tensor.Tensor[Float64], requires_grad: Bool = True):
        # Replace if exists; otherwise append
        var i = 0
        while i < self.param_names.size():
            if self.param_names.__getitem__(i) == name:
                self.param_values.__setitem__(i, value)
                self.param_requires_grad.__setitem__(i, requires_grad)
                return
            i = i + 1
        self.param_names.push_back(name)
        self.param_values.push_back(value)
        self.param_requires_grad.push_back(requires_grad)

    fn register_buffer(mut self, name: String, value: tensor.Tensor[Float64]):
        var i = 0
        while i < self.buffer_names.size():
            if self.buffer_names.__getitem__(i) == name:
                self.buffer_values.__setitem__(i, value)
                return
            i = i + 1
        self.buffer_names.push_back(name)
        self.buffer_values.push_back(value)

    # Convenience aliases
    fn add_parameter(mut self, name: String, value: tensor.Tensor[Float64], requires_grad: Bool = True):
        self.register_parameter(name, value, requires_grad)

    fn add_buffer(mut self, name: String, value: tensor.Tensor[Float64]):
        self.register_buffer(name, value)

    # --------------
    # Train / eval
    # --------------

    fn train(mut self) -> Module:
        self.training = True
        var i = 0
        while i < self.children.size():
            var c = self.children.__getitem__(i)
            c.train()
            self.children.__setitem__(i, c)
            i = i + 1
        return self

    fn eval(mut self) -> Module:
        self.training = False
        var i = 0
        while i < self.children.size():
            var c = self.children.__getitem__(i)
            c.eval()
            self.children.__setitem__(i, c)
            i = i + 1
        return self

    fn is_training(self) -> Bool:
        return self.training

    # ------------------
    # Accessors
    # ------------------

    fn parameters(self) -> List[tensor.Tensor[Float64]]:
        return self.param_values

    fn buffers(self) -> List[tensor.Tensor[Float64]]:
        return self.buffer_values

    # ------------------
    # Introspection
    # ------------------

    fn named_parameters(self, prefix: String = String("")) -> List[String]:
        var out = List[String]()

        var i = 0
        while i < self.param_names.size():
            var key = self.param_names.__getitem__(i)
            var full = prefix
            if prefix.__len__() > 0:
                full += String(".")
            full += key
            out.push_back(full)
            i = i + 1

        var j = 0
        while j < self.children.size():
            var cname = self.child_names.__getitem__(j)
            var child = self.children.__getitem__(j)
            var child_prefix = prefix
            if child_prefix.__len__() > 0:
                child_prefix += String(".")
            child_prefix += cname

            var sub = child.named_parameters(child_prefix)
            var k = 0
            while k < sub.size():
                out.push_back(sub.__getitem__(k))
                k = k + 1
            j = j + 1
        return out

    fn named_buffers(self, prefix: String = String("")) -> List[String]:
        var out = List[String]()

        var i = 0
        while i < self.buffer_names.size():
            var key = self.buffer_names.__getitem__(i)
            var full = prefix
            if prefix.__len__() > 0:
                full += String(".")
            full += key
            out.push_back(full)
            i = i + 1

        var j = 0
        while j < self.children.size():
            var cname = self.child_names.__getitem__(j)
            var child = self.children.__getitem__(j)
            var child_prefix = prefix
            if child_prefix.__len__() > 0:
                child_prefix += String(".")
            child_prefix += cname

            var sub = child.named_buffers(child_prefix)
            var k = 0
            while k < sub.size():
                out.push_back(sub.__getitem__(k))
                k = k + 1
            j = j + 1
        return out

    # ------------------
    # Serialization
    # ------------------

    fn _pairs_flat(self, prefix: String = String("")) -> List[String]:
        var lines = List[String]()

        # params
        var i = 0
        while i < self.param_names.size():
            var k = self.param_names.__getitem__(i)
            var v = self.param_values.__getitem__(i)
            var full = prefix
            if prefix.__len__() > 0:
                full += String(".")
            full += k
            var line = String("\"") + _json_escape(full) + String("\":\"") + _json_escape(_tensor_summary(v)) + String("\"")
            lines.push_back(line)
            i = i + 1

        # buffers
        var b = 0
        while b < self.buffer_names.size():
            var k2 = self.buffer_names.__getitem__(b)
            var v2 = self.buffer_values.__getitem__(b)
            var full2 = prefix
            if prefix.__len__() > 0:
                full2 += String(".")
            full2 += k2
            var line2 = String("\"") + _json_escape(full2) + String("\":\"") + _json_escape(_tensor_summary(v2)) + String("\"")
            lines.push_back(line2)
            b = b + 1

        # recurse
        var j = 0
        while j < self.children.size():
            var cname = self.child_names.__getitem__(j)
            var child = self.children.__getitem__(j)
            var child_prefix = prefix
            if child_prefix.__len__() > 0:
                child_prefix += String(".")
            child_prefix += cname

            var sub = child._pairs_flat(child_prefix)
            var k = 0
            while k < sub.size():
                lines.push_back(sub.__getitem__(k))
                k = k + 1
            j = j + 1

        return lines

    fn state_dict(self) -> String:
        var pairs = self._pairs_flat(String(""))
        var sb = String("{")
        var i = 0
        while i < pairs.size():
            sb += pairs.__getitem__(i)
            if i + 1 < pairs.size():
                sb += String(",")
            i = i + 1
        sb += String("}")
        return sb

    fn load_state_dict(mut self, state: String):
        # Placeholder: future work → parse and dispatch into tensors.
        if state.__len__() < 2:
            return
        var first = state.__getitem__(0)
        var last = state.__getitem__(state.__len__() - 1)
        if not (first == String("{") and last == String("}")):
            return
        # TODO: implement parser when a stable tensor (de)serialization is finalized.
        return
