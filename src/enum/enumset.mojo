# Module: momijo.enum.enumset
# Minimal enum utilities implemented in Mojo.
# Project: momijo.enum
# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Momijo Enum
# This file is part of the Momijo project. See the LICENSE file at the repository root.


from bit import pop_count
struct EnumSet(Copyable, Movable, Sized):
    var bits: UInt64

    fn __init__(out self, bits: UInt64 = 0):
        # Initializes empty set or from bitmask.
        self.bits = bits

    fn __len__(self) -> Int:
        # Number of members == number of 1-bits.
        var x = self.bits
        var cnt: Int = 0
        while x != 0:
            x = x & (x - UInt64(1))   # clear lowest set bit
            cnt += 1
        return cnt

    fn is_empty(self) -> Bool:
        # True if no members are present.
        return self.bits == 0

    fn clear(mut self):
        # Removes all members.
        self.bits = 0

    fn contains(self, idx: Int) -> Bool:
        # Checks membership by ordinal index.
        # Input: idx — ordinal of the enum value
        # Output: True if present
        if idx < 0 or idx >= 64:
            return False
        return (self.bits & (UInt64(1) << UInt64(idx))) != 0

    fn add(mut self, idx: Int):
        # Inserts a member by ordinal index.
        if idx < 0 or idx >= 64:
            return
        self.bits = self.bits | (UInt64(1) << UInt64(idx))

    fn remove(mut self, idx: Int):
        # Erases a member by ordinal index.
        if idx < 0 or idx >= 64:
            return
        self.bits = self.bits & ~(UInt64(1) << UInt64(idx))

    fn union(a: Self, b: Self) -> Self:
        # Returns set union.
        return EnumSet(bits = (a.bits | b.bits))

    fn intersect(a: Self, b: Self) -> Self:
        # Returns set intersection.
        return EnumSet(bits = (a.bits & b.bits))

    fn difference(a: Self, b: Self) -> Self:
        # Returns set difference a \ b.
        return EnumSet(bits = (a.bits & ~b.bits))

    fn symmetric_difference(a: Self, b: Self) -> Self:
        # Returns symmetric difference.
        return EnumSet(bits = (a.bits ^ b.bits))

    fn is_subset(a: Self, b: Self) -> Bool:
        # True if all members of a are in b.
        return (a.bits & ~b.bits) == 0

    fn is_superset(a: Self, b: Self) -> Bool:
        # True if all members of b are in a.
        return (b.bits & ~a.bits) == 0

    fn equals(a: Self, b: Self) -> Bool:
        # Value equality based on bitmask.
        return a.bits == b.bits

    fn any(self) -> Bool:
        # True if at least one member exists.
        return self.bits != 0

    fn all_up_to(self, n: Int) -> Bool:
        # True if all ordinals in [0, n) are present.
        # Input: n — domain size to check
        if n <= 0:
            return True
        if n > 64:
            return False

        var mask: UInt64
        if n == 64:
            mask = ~UInt64(0)
        else:
            mask = (UInt64(1) << UInt64(n)) - UInt64(1)

        return (self.bits & mask) == mask