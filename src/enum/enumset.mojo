# Module: momijo.enum.enumset
# Minimal enum utilities implemented in Mojo.
# Project: momijo.enum
# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Momijo Enum
# This file is part of the Momijo project. See the LICENSE file at the repository root.


from bit import pop_count

# Tracks membership of small integer tags (0..63) in a compact bitset.
struct EnumSet(Copyable, Movable, Sized):
    var bits: UInt64

# Does: utility function in enum module.
# Inputs: out self.
# Returns: result value or status.
    fn __init__(out self):
        self.bits = 0

# Does: utility function in enum module.
# Inputs: self.
# Returns: result value or status.
    fn __len__(self) -> Int:
        return Int(pop_count(self.bits))

# Returns a new empty set.
fn enumset_empty() -> EnumSet:
    return EnumSet()

# Adds a tag to the set.
fn enumset_insert(mut s: EnumSet, tag: Int32):
    s.bits = s.bits | (UInt64(1) << UInt64(tag))

# Removes a tag from the set.
fn enumset_erase(mut s: EnumSet, tag: Int32):
    s.bits = s.bits & (~(UInt64(1) << UInt64(tag)))

# Checks membership.
fn enumset_contains(s: EnumSet, tag: Int32) -> Bool:
    return (s.bits & (UInt64(1) << UInt64(tag))) != 0

# Set union.
fn enumset_union(a: EnumSet, b: EnumSet) -> EnumSet:
    return EnumSet(bits=(a.bits | b.bits))

# Set intersection.
fn enumset_intersect(a: EnumSet, b: EnumSet) -> EnumSet:
    return EnumSet(bits=(a.bits & b.bits))

# Counts members.
fn enumset_count(s: EnumSet) -> Int:
    return Int(pop_count(s.bits))

# True when empty.
fn enumset_is_empty(s: EnumSet) -> Bool:
    return s.bits == 0

# Clears all members.
fn enumset_clear_all(mut s: EnumSet):
    s.bits = 0

# Adds many tags at once.
fn enumset_insert_many(mut s: EnumSet, tags: List[Int32]):
    var i = 0
    while i < len(tags):
        s.bits = s.bits | (UInt64(1) << UInt64(tags[i]))
        i += 1

# Removes many tags at once.
fn enumset_erase_many(mut s: EnumSet, tags: List[Int32]):
    var i = 0
    while i < len(tags):
        s.bits = s.bits & (~(UInt64(1) << UInt64(tags[i])))
        i += 1

# Returns the raw mask for interop.
fn enumset_to_bits(s: EnumSet) -> UInt64:
    return s.bits

# Builds from a raw mask (truncates to 64 tags).
fn enumset_from_bits(mask: UInt64) -> EnumSet:
    return EnumSet(bits=mask)