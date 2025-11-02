# MIT License
# SPDX-License-Identifier: MIT
# Project:      Momijo
# Module:       learn.hub.registry
# File:         src/momijo/learn/hub/registry.mojo
#
# Description:
#   Model registry for Momijo Learn Hub.
#   - Stores unique (name -> fully-qualified constructor path).
#   - Duplicate-safe registration (updates path if name exists).
#   - Lookup / list / remove / clear APIs.
#   - No module-level globals; uses function-local static instance.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo

from collections.list import List

# -----------------------------------------------------------------------------
# Internal store (name ↔ constructor_path)
# -----------------------------------------------------------------------------

struct _Registry:
    var names: List[String]
    var ctor_paths: List[String]

    fn __init__(out self):
        self.names = List[String]()
        self.ctor_paths = List[String]()

    fn size(self) -> Int:
        return len(self.names)

    fn _index_of(self, name: String) -> Int:
        var i = 0
        var n = len(self.names)
        while i < n:
            if self.names[i] == name:
                return i
            i = i + 1
        return -1

# -----------------------------------------------------------------------------
# Private accessor for the singleton registry (no module-level globals)
# -----------------------------------------------------------------------------

fn _registry() -> _Registry:
    # Function-local static keeps state without global variables.
    static var INST = _Registry()
    return INST

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

# Register a model name with its fully-qualified constructor path.
# Example:
#   register_model("resnet18", "momijo.vision.models.resnet.resnet18")
# Returns true if a new entry was added; false if name existed (path updated if different).
fn register_model(name: String, constructor_path: String) -> Bool:
    var reg = _registry()
    var idx = reg._index_of(name)
    if idx >= 0:
        if reg.ctor_paths[idx] != constructor_path:
            reg.ctor_paths[idx] = constructor_path
        return False
    reg.names.append(name)
    reg.ctor_paths.append(constructor_path)
    return True

# Check if a model name exists.
fn has_model(name: String) -> Bool:
    var reg = _registry()
    return reg._index_of(name) >= 0

# Return the constructor path for a name, or empty string if not found.
fn get_constructor_path(name: String) -> String:
    var reg = _registry()
    var idx = reg._index_of(name)
    if idx >= 0:
        return reg.ctor_paths[idx]
    return String("")

# List all registered model names (copy).
fn list_models() -> List[String]:
    var reg = _registry()
    var out = List[String]()
    var i = 0
    var n = len(reg)
    while i < n:
        out.append(reg.names[i])
        i = i + 1
    return out

# Remove a model by name. Returns true if removed.
fn unregister_model(name: String) -> Bool:
    var reg = _registry()
    var idx = reg._index_of(name)
    if idx < 0:
        return False
    var last = len(reg) - 1
    # Swap-with-last then pop (O(1))
    reg.names[idx] = reg.names[last]
    reg.ctor_paths[idx] = reg.ctor_paths[last]
    reg.names.pop_back()
    reg.ctor_paths.pop_back()
    return True

# Clear the registry; returns number of entries removed.
fn clear_registry() -> Int:
    var reg = _registry()
    var n = len(reg)
    reg.names = List[String]()
    reg.ctor_paths = List[String]()
    return n

# Optional: export a paired view (names, ctor_paths) as copies.
fn dump_registry() -> (List[String], List[String]):
    var reg = _registry()
    var names_copy = List[String]()
    var paths_copy = List[String]()
    var i = 0
    var n = len(reg)
    while i < n:
        names_copy.append(reg.names[i])
        paths_copy.append(reg.ctor_paths[i])
        i = i + 1
    return (names_copy, paths_copy)
