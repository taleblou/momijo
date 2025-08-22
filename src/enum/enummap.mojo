# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Module: momijo.enum.enummap
# Minimal enum utilities implemented in Mojo.
# Project: momijo.enum
# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Momijo Enum
# This file is part of the Momijo project. See the LICENSE file at the repository root.


# Associates values to small integer tags [0, k). Fixed-size, array-backed.
struct EnumMap[T: Copyable & Movable](Copyable, Movable):
    var values: List[T]
    var default: T

# Does: utility function in enum module.
# Inputs: out self, k, default.
# Returns: result value or status.
    fn __init__(out self, k: Int, default: T):
        self.values = List[T](capacity=k)
        self.default = default
        var i = 0
        while i < k:
            self.values[i] = default
            i += 1

# Constructs a map with a fixed number of slots and a default value.
fn enummap_with_default[T: Copyable & Movable](k: Int, default: T) -> EnumMap[T]:
    return EnumMap[T](k, default)

# Writes a value into a tag slot when the index is in range.
fn enummap_set_tag[T: Copyable & Movable](mut m: EnumMap[T], tag: Int32, value: T):
    if Int(tag) < 0 or Int(tag) >= len(m.values): return
    m.values[Int(tag)] = value

# Reads a value; returns the default when the index is out of range.
fn enummap_get_tag[T: Copyable & Movable](m: EnumMap[T], tag: Int32) -> T:
    if Int(tag) < 0 or Int(tag) >= len(m.values): return m.default
    return m.values[Int(tag)]

# Returns the number of addressable tags.
fn enummap_capacity[T: Copyable & Movable](m: EnumMap[T]) -> Int:
    return len(m.values)

# Fills every slot with one value.
fn enummap_fill[T: Copyable & Movable](mut m: EnumMap[T], value: T):
    var i = 0
    while i < len(m.values):
        m.values[i] = value
        i += 1

# Resets all slots to the map's default.
fn enummap_clear[T: Copyable & Movable](mut m: EnumMap[T]):
    var i = 0
    while i < len(m.values):
        m.values[i] = m.default
        i += 1

# True when the given tag is within range.
fn enummap_contains_tag[T: Copyable & Movable](m: EnumMap[T], tag: Int32) -> Bool:
    return Int(tag) >= 0 and Int(tag) < len(m.values)