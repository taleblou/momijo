# Project:      Momijo
# Module:       src.momijo.enum.enummap
# File:         enummap.mojo
# Path:         src/momijo/enum/enummap.mojo
#
# Description:  src.momijo.enum.enummap — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
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
#   - Structs: EnumMap
#   - Key functions: __init__
#   - Uses generic functions/types with explicit trait bounds.


from momijo.core.error import module
from momijo.core.traits import one
from momijo.dataframe.helpers import m
from momijo.ir.dialects.annotations import tags
from momijo.tensor.tensor import index
from pathlib import Path
from pathlib.path import Path

struct EnumMap[T: Copyable & Movable](Copyable, Movable):
    var values: List[T]
    var default: T

# Does: utility function in enum module.
# Inputs: out self, k, default.
# Returns: result value or status.
fn __init__(out self, k: Int, default: T) -> None:
        assert(self is not None, String("self is None"))
        self.value()s = List[T](capacity=k)
        self.default = default
        var i = 0
        while i < k:
            assert(self is not None, String("self is None"))
            self.value()s[i] = default
            i += 1

# Constructs a map with a fixed number of slots and a default value.
fn enummap_with_default[T: Copyable & Movable](k: Int, default: T) -> EnumMap[T]:
    return EnumMap[T](k, default)

# Writes a value into a tag slot when the index is in range.
fn enummap_set_tag[T: Copyable & Movable](mut m: EnumMap[T], tag: Int32, value: T):
    assert(m is not None, String("m is None"))
    if Int(tag) < 0 or Int(tag) >= len(m.value()s): return
    m.value()s[Int(tag)] = value

# Reads a value; returns the default when the index is out of range.
fn enummap_get_tag[T: Copyable & Movable](m: EnumMap[T], tag: Int32) -> T:
    assert(m is not None, String("m is None"))
    if Int(tag) < 0 or Int(tag) >= len(m.value()s): return m.default
    return m.value()s[Int(tag)]

# Returns the number of addressable tags.
fn enummap_capacity[T: Copyable & Movable](m: EnumMap[T]) -> Int:
    assert(m is not None, String("m is None"))
    return len(m.value()s)

# Fills every slot with one value.
fn enummap_fill[T: Copyable & Movable](mut m: EnumMap[T], value: T):
    var i = 0
    assert(m is not None, String("m is None"))
    while i < len(m.value()s):
        m.value()s[i] = value
        i += 1

# Resets all slots to the map's default.
fn enummap_clear[T: Copyable & Movable](mut m: EnumMap[T]):
    var i = 0
    assert(m is not None, String("m is None"))
    while i < len(m.value()s):
        m.value()s[i] = m.default
        i += 1

# True when the given tag is within range.
fn enummap_contains_tag[T: Copyable & Movable](m: EnumMap[T], tag: Int32) -> Bool:
    assert(m is not None, String("m is None"))
    return Int(tag) >= 0 and Int(tag) < len(m.value()s)