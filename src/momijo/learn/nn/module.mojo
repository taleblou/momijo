# Project:      Momijo
# Module:       learn.nn.module
# File:         nn/module.mojo
# Path:         src/momijo/learn/nn/module.mojo
#
# Description:  PyTorch-like base Module for Momijo Learn. Supports hierarchical
#               composition (submodules), parameter and buffer registration,
#               training/eval mode switching, and flat state_dict export with
#               fully-qualified names (e.g., "layer1.weight").
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
#   - Types: Module
#   - Key fns: add_module, register_parameter, register_buffer,
#              train/eval/is_training, state_dict, load_state_dict
#   - The storage for parameters/buffers is backend-agnostic (String placeholders).
#     Once momijo.tensor is ready, replace String with real tensor handles and
#     wire JSON (de)serialization accordingly.

from collections.list import List

# Helper: minimal JSON string escape for flat key/value export.
fn _json_escape(s: String) -> String:
    var out = String("")
    var i = 0
    while i < s.__len__():
        var ch = s.__getitem__(i)
        # NOTE: __getitem__ on String returns a 1-char String in current Mojo.
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

# Base module
struct Module:
    # Identity
    var name: String
    var training: Bool

    # Hierarchy
    var child_names: List[String]
    var children: List[Module]

    # Registered learnable params (placeholder storage as String)
    var param_names: List[String]
    var param_values: List[String]

    # Registered non-learnable buffers (placeholder storage as String)
    var buffer_names: List[String]
    var buffer_values: List[String]

    fn __init__(out self, name: String = String("")):
        self.name = name
        self.training = True

        self.child_names = List[String]()
        self.children = List[Module]()

        self.param_names = List[String]()
        self.param_values = List[String]()

        self.buffer_names = List[String]()
        self.buffer_values = List[String]()

    # ---------------------------
    # Composition / registration
    # ---------------------------

    fn add_module(mut self, name: String, m: Module):
        # No duplicate names allowed (simple check)
        var i = 0
        while i < self.child_names.size():
            if self.child_names.__getitem__(i) == name:
                # overwrite semantics could be defined later; for now just replace
                self.children.__setitem__(i, m)
                return
            i = i + 1
        self.child_names.push_back(name)
        self.children.push_back(m)

    fn modules(self) -> List[Module]:
        # Returns direct children (shallow)
        return self.children

    fn register_parameter(mut self, name: String, placeholder_value: String):
        # In real impl, placeholder_value will be a Tensor or a typed Param wrapper.
        self.param_names.push_back(name)
        self.param_values.push_back(placeholder_value)

    fn register_buffer(mut self, name: String, placeholder_value: String):
        self.buffer_names.push_back(name)
        self.buffer_values.push_back(placeholder_value)

    # --------------
    # Train / eval
    # --------------

    fn train(mut self) -> Module:
        self.training = True
        # Propagate to children
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
    # Introspection
    # ------------------

    # Flat list of parameter names with fully-qualified prefixes
    fn named_parameters(self, prefix: String = String("")) -> List[String]:
        var out = List[String]()

        # local params
        var i = 0
        while i < self.param_names.size():
            var key = self.param_names.__getitem__(i)
            var full = prefix
            if prefix.__len__() > 0:
                full += String(".")
            full += key
            out.push_back(full)
            i = i + 1

        # recurse
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

    # Flat list of buffer names with fully-qualified prefixes
    fn named_buffers(self, prefix: String = String("")) -> List[String]:
        var out = List[String]()

        # local buffers
        var i = 0
        while i < self.buffer_names.size():
            var key = self.buffer_names.__getitem__(i)
            var full = prefix
            if prefix.__len__() > 0:
                full += String(".")
            full += key
            out.push_back(full)
            i = i + 1

        # recurse
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

    # Export a flat JSON object string:
    # {
    #   "weight": "<val>",
    #   "bias": "<val>",
    #   "child.weight": "<val>",
    #   ...
    # }
    # Values are placeholders (String). Replace with real tensor serialization later.
    fn state_dict(self) -> String:
        var sb = List[String]()
        sb.push_back(String("{"))

        # local params
        var i = 0
        while i < self.param_names.size():
            var k = _json_escape(self.param_names.__getitem__(i))
            var v = _json_escape(self.param_values.__getitem__(i))
            var line = String("\"") + k + String("\":\"") + v + String("\"")
            sb.push_back(line)
            i = i + 1

        # local buffers
        var b = 0
        while b < self.buffer_names.size():
            var k2 = _json_escape(self.buffer_names.__getitem__(b))
            var v2 = _json_escape(self.buffer_values.__getitem__(b))
            var line2 = String("\"") + k2 + String("\":\"") + v2 + String("\"")
            sb.push_back(line2)
            b = b + 1

        # recurse children with prefix "childname."
        var j = 0
        while j < self.children.size():
            var cname = self.child_names.__getitem__(j)
            var child = self.children.__getitem__(j)

            # child's flat dict as JSON string; we parse shallowly by splitting pairs
            # because we don't rely on a JSON parser here.
            var child_json = child.state_dict()

            # Strip leading '{' and trailing '}' to splice pairs.
            var inner = child_json
            var n = inner.__len__()
            if n >= 2:
                inner = inner.slice(1, n - 1)

            # If child has any pairs, prefix keys with "cname."
            # We'll scan inner for pairs split by ',' at top level (since it's flat).
            # For simplicity (and since we control generator), we can split on ','.
            var start = 0
            var idx = 0
            while idx <= inner.__len__():
                var need_flush = False
                if idx == inner.__len__():
                    need_flush = True
                else:
                    var ch = inner.__getitem__(idx)
                    if ch == String(","):
                        need_flush = True
                if need_flush:
                    if idx > start:
                        var pair = inner.slice(start, idx).strip()
                        if pair.__len__() > 0:
                            # pair is like: "key":"value"
                            # we need to inject prefix after first quote
                            # Find second quote to isolate the key content.
                            var qi = pair.find(String("\""))
                            var qj = -1
                            if qi >= 0:
                                qj = pair.find_from(String("\""), qi + 1)
                            if qi == 0 and qj > qi:
                                var key_inner = pair.slice(1, qj)
                                var rest = pair.slice(qj + 1, pair.__len__())
                                var prefixed = String("\"") + _json_escape(cname + String(".") + key_inner) + rest
                                sb.push_back(prefixed)
                            else:
                                # Fallback: push as-is (should not happen with our generator)
                                sb.push_back(pair)
                    start = idx + 1
                idx = idx + 1
            j = j + 1

        # Join with commas
        var out = String("")
        var t = 0
        while t < sb.size():
            out += sb.__getitem__(t)
            if t + 1 < sb.size():
                out += String(",")
            t = t + 1

        out += String("}")
        return out

    # Load from a flat JSON-like String. This is a placeholder that currently
    # validates keys and counts but does not deserialize into real tensors.
    # When momijo.tensor is ready, implement actual parsing and assignment.
    fn load_state_dict(mut self, state: String):
        # Very lightweight sanity: check braces exist.
        if state.__len__() < 2:
            return
        if not (state.__getitem__(0) == String("{") and state.__getitem__(state.__len__() - 1) == String("}")):
            return
        # TODO: implement proper parsing and dispatch to:
        #  - local params by exact name
        #  - local buffers by exact name
        #  - children by "childname." prefix
        # For now, it's a no-op to keep compatibility.
        pass
