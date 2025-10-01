# Project:      Momijo
# Module:       learn.hub.registry
# File:         hub/registry.mojo
# Path:         src/momijo/learn/hub/registry.mojo
#
# Description:  Model registry for Momijo Learn Hub.
#               Stores fully-qualified constructor paths (as strings) for named models,
#               provides uniqueness checks, lookup APIs, and a clearable singleton store
#               without module-level globals (via function-local static instance).
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
#   - Types: _Registry
#   - Key fns: register_model(name, ctor_path), has_model(name), list_models(),
#              get_constructor_path(name), unregister_model(name), clear_registry()
#   - Policy: No module-level globals; singleton lives as a function-local static.

from collections.list import List

# Internal store (name ↔ constructor_path)
struct _Registry:
    var names: List[String]
    var ctor_paths: List[String]

    fn __init__(out self):
        self.names = List[String]()
        self.ctor_paths = List[String]()

    fn size(self) -> Int:
        return Int(self.names.size())

# Private accessor for the singleton registry.
# Uses a function-local static to avoid module-level global state.
fn _registry() -> _Registry:
    # NOTE: relying on Mojo supporting function-local static initialization.
    # If your linter requires, this can be swapped to a private factory with
    # a hidden tagged-union state.
    static var INST = _Registry()
    return INST

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

# Register a model name with its fully-qualified constructor path.
# Example:
#   register_model(String("resnet18"), String("momijo.vision.models.resnet.resnet18"))
#
# Returns true if a new entry was added; false if the name already existed (no-op).
fn register_model(name: String, constructor_path: String) -> Bool:
    var reg = _registry()
    # Check duplicate
    var i = 0
    while i < reg.size():
        if reg.names[i] == name:
            # Update path if different (idempotent registration)
            if reg.ctor_paths[i] != constructor_path:
                reg.ctor_paths[i] = constructor_path
            return False
        i = i + 1

    # Insert new
    reg.names.push_back(name)
    reg.ctor_paths.push_back(constructor_path)
    return True

# Check if a model name exists in registry.
fn has_model(name: String) -> Bool:
    var reg = _registry()
    var i = 0
    while i < reg.size():
        if reg.names[i] == name:
            return True
        i = i + 1
    return False

# Return the constructor path for a model name, or empty string if not found.
fn get_constructor_path(name: String) -> String:
    var reg = _registry()
    var i = 0
    while i < reg.size():
        if reg.names[i] == name:
            return reg.ctor_paths[i]
        i = i + 1
    return String("")

# List all registered model names (copy).
fn list_models() -> List[String]:
    var reg = _registry()
    # Return a shallow copy to avoid external mutation of internal store
    var out = List[String]()
    var i = 0
    while i < reg.size():
        out.push_back(reg.names[i])
        i = i + 1
    return out

# Remove a model by name. Returns true if removed.
fn unregister_model(name: String) -> Bool:
    var reg = _registry()
    var i = 0
    while i < reg.size():
        if reg.names[i] == name:
            # Compact by swapping with last then popping (O(1))
            var last = reg.size() - 1
            reg.names[i] = reg.names[last]
            reg.ctor_paths[i] = reg.ctor_paths[last]
            # Pop last
            reg.names.pop_back()
            reg.ctor_paths.pop_back()
            return True
        i = i + 1
    return False

# Clear the registry; returns the number of entries removed.
fn clear_registry() -> Int:
    var reg = _registry()
    var n = reg.size()
    # Reinitialize lists
    reg.names = List[String]()
    reg.ctor_paths = List[String]()
    return n

# Optional: get paired view (name, ctor_path) as two parallel lists.
# Useful for debugging or exporting registry.
fn dump_registry() -> (List[String], List[String]):
    var reg = _registry()
    # copies
    var names_copy = List[String]()
    var paths_copy = List[String]()
    var i = 0
    while i < reg.size():
        names_copy.push_back(reg.names[i])
        paths_copy.push_back(reg.ctor_paths[i])
        i = i + 1
    return (names_copy, paths_copy)
