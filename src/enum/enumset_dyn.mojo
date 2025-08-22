# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Module: momijo.enum.enumset_dyn
# Minimal enum utilities implemented in Mojo.
# Project: momijo.enum
# MIT License
# Copyright (c) 2025 Morteza Talebou (https://taleblou.ir/)
# Momijo Enum
# This file is part of the Momijo project. See the LICENSE file at the repository root.

#
# Copyright (c) 2025 Morteza Taleblou (https:#taleblou.ir/)
# All rights reserved.
#
from bit import pop_count
from .meta import EnumMetaInfo, all_tags

from momijo.enum.enumset import EnumSet

# Helper: count set bits of a UInt64 (Kernighan)
fn popcount_u64(x: UInt64) -> Int:
    var v = x
    var c: Int = 0
    while v != 0:
        v = v & (v - UInt64(1))
        c += 1
    return c

# Helper: mask of lower m bits (clamped to [0,64])
fn lower_mask(m: Int) -> UInt64:
    if m <= 0:
        return UInt64(0)
    if m >= 64:
        return ~UInt64(0)
    return (UInt64(1) << UInt64(m)) - UInt64(1)

struct DynEnumSet(Copyable, Movable, Sized):
    # Public fields expected by tests
    var k: Int                  # logical domain size in bits (clamped to [0,256])
    var chunks: List[UInt64]    # 4x64-bit words; tests may replace this list

    # Default ctor: k=256, chunks=[0,0,0,0]
    fn __init__(out self):
        self.k = 256
        self.chunks = [UInt64(0), UInt64(0), UInt64(0), UInt64(0)]

    # Ctor with explicit k; initialize zeroed chunks
    fn __init__(out self, k: Int):
        var d = k
        if d < 0:
            d = 0
        if d > 256:
            d = 256
        self.k = d
        self.chunks = [UInt64(0), UInt64(0), UInt64(0), UInt64(0)]

    fn __len__(self) -> Int:
        # sum popcount of all words (supporting any list length; tests usually 4)
        var s: Int = 0
        for w in self.chunks:
            s += popcount_u64(w)
        return s

    fn clear(mut self):
        # zero all words (handle lists shorter/longer than 4 defensively)
        var n = len(self.chunks)
        var i = 0
        while i < n:
            self.chunks[i] = UInt64(0)
            i += 1

    fn in_domain(self, idx: Int) -> Bool:
        if idx < 0:
            return False
        if idx >= self.k:
            return False
        if idx >= 256:
            return False
        return True

    # Map global index to (chunk_index, bit_offset)
    fn which(idx: Int) -> (ci: Int, off: Int):
        if idx < 64:
            return (0, idx)
        elif idx < 128:
            return (1, idx - 64)
        elif idx < 192:
            return (2, idx - 128)
        else:
            return (3, idx - 192)

    fn contains(self, idx: Int) -> Bool:
        if not self.in_domain(idx):
            return False
        var (ci, off) = DynEnumSet.which(idx)
        if ci < 0 or ci >= len(self.chunks):
            return False
        var word = self.chunks[ci]
        return (word & (UInt64(1) << UInt64(off))) != 0

    fn add(mut self, idx: Int):
        if not self.in_domain(idx):
            return
        var (ci, off) = DynEnumSet.which(idx)
        if ci < 0:
            return
        # ensure chunks has at least ci+1 items
        var need = ci + 1
        while len(self.chunks) < need:
            self.chunks.append(UInt64(0))
        self.chunks[ci] = self.chunks[ci] | (UInt64(1) << UInt64(off))

    fn remove(mut self, idx: Int):
        if not self.in_domain(idx):
            return
        var (ci, off) = DynEnumSet.which(idx)
        if ci < 0 or ci >= len(self.chunks):
            return
        self.chunks[ci] = self.chunks[ci] & ~(UInt64(1) << UInt64(off))

    # ----- set ops on aligned words -----

    fn min_k(a: Self, b: Self) -> Int:
        var d = a.k
        if b.k < d:
            d = b.k
        return d

    fn wordwise(a: Self, b: Self, op: Int) -> Self:
        # op: 0=OR, 1=AND, 2=AND_NOT(a\b), 3=XOR
        var out = DynEnumSet(DynEnumSet.min_k(a, b))
        var na = len(a.chunks)
        var nb = len(b.chunks)
        var n = na
        if nb > n:
            n = nb
        # ensure output has n words
        var i = len(out.chunks)
        while i < n:
            out.chunks.append(UInt64(0))
            i += 1
        i = 0
        while i < n:
            var aw: UInt64 = UInt64(0)
            if i < na:
                aw = a.chunks[i]
            var bw: UInt64 = UInt64(0)
            if i < nb:
                bw = b.chunks[i]
            var rw: UInt64 = UInt64(0)
            if op == 0:
                rw = aw | bw
            elif op == 1:
                rw = aw & bw
            elif op == 2:
                rw = aw & ~bw
            else:
                rw = aw ^ bw
            out.chunks[i] = rw
            i += 1
        return out

    fn union(a: Self, b: Self) -> Self:
        return DynEnumSet.wordwise(a, b, 0)

    fn intersect(a: Self, b: Self) -> Self:
        return DynEnumSet.wordwise(a, b, 1)

    fn difference(a: Self, b: Self) -> Self:
        return DynEnumSet.wordwise(a, b, 2)

    fn symmetric_difference(a: Self, b: Self) -> Self:
        return DynEnumSet.wordwise(a, b, 3)

    fn is_subset(a: Self, b: Self) -> Bool:
        var na = len(a.chunks)
        var nb = len(b.chunks)
        var n = na
        if nb > n:
            n = nb
        var i = 0
        while i < n:
            var aw: UInt64 = UInt64(0)
            if i < na:
                aw = a.chunks[i]
            var bw: UInt64 = UInt64(0)
            if i < nb:
                bw = b.chunks[i]
            if (aw & ~bw) != 0:
                return False
            i += 1
        return True

    fn is_superset(a: Self, b: Self) -> Bool:
        var na = len(a.chunks)
        var nb = len(b.chunks)
        var n = na
        if nb > n:
            n = nb
        var i = 0
        while i < n:
            var aw: UInt64 = UInt64(0)
            if i < na:
                aw = a.chunks[i]
            var bw: UInt64 = UInt64(0)
            if i < nb:
                bw = b.chunks[i]
            if (bw & ~aw) != 0:
                return False
            i += 1
        return True

    fn equals(a: Self, b: Self) -> Bool:
        if a.k != b.k:
            return False
        var na = len(a.chunks)
        var nb = len(b.chunks)
        var n = na
        if nb > n:
            n = nb
        var i = 0
        while i < n:
            var aw: UInt64 = UInt64(0)
            if i < na:
                aw = a.chunks[i]
            var bw: UInt64 = UInt64(0)
            if i < nb:
                bw = b.chunks[i]
            if aw != bw:
                return False
            i += 1
        return True

    fn any(self) -> Bool:
        for w in self.chunks:
            if w != 0:
                return True
        return False

    fn all_up_to(self, n: Int) -> Bool:
        # True if all ordinals in [0, n) are present (clamped to domain and 256).
        var need = n
        if need < 0:
            need = 0
        if need > self.k:
            need = self.k
        if need > 256:
            need = 256
        if need == 0:
            return True

        var rem = need
        var i = 0
        while rem > 0 and i < len(self.chunks):
            var take = 0
            if rem >= 64:
                take = 64
            else:
                take = rem
            var mask = lower_mask(take)
            var w = self.chunks[i]
            if (w & mask) != mask:
                return False
            rem -= take
            i += 1
        return rem == 0
# Does: utility function in enum module.
# Inputs: meta.
# Returns: result value or status.
fn enumset_dyn_from_meta(meta: EnumMetaInfo) -> DynEnumSet:
    var k = len(all_tags(meta))
    var n_chunks = (k + 63) # 64
    var chunks = List[UInt64](n_chunks)
    for i in range(0, n_chunks): chunks[i] = 0
    return DynEnumSet(chunks=chunks, k=k)

# Does: utility function in enum module.
# Inputs: mut s, tag.
# Returns: result value or status.
fn enumset_dyn_add(mut s: DynEnumSet, tag: Int):
    if tag < 0 or tag >= s.k: return
    var idx = tag # 64
    var bit = UInt64(1) << UInt64(tag % 64)
    s.chunks[idx] = s.chunks[idx] | bit

# Does: utility function in enum module.
# Inputs: mut s, tag.
# Returns: result value or status.
fn enumset_dyn_remove(mut s: DynEnumSet, tag: Int):
    if tag < 0 or tag >= s.k: return
    var idx = tag # 64
    var bit = UInt64(1) << UInt64(tag % 64)
    s.chunks[idx] = s.chunks[idx] & (~bit)

# Does: utility function in enum module.
# Inputs: s, tag.
# Returns: result value or status.
fn enumset_dyn_has(s: DynEnumSet, tag: Int) -> Bool:
    if tag < 0 or tag >= s.k: return False
    var idx = tag # 64
    var bit = UInt64(1) << UInt64(tag % 64)
    return (s.chunks[idx] & bit) != 0

# Does: utility function in enum module.
# Inputs: s.
# Returns: result value or status.
fn enumset_dyn_count(s: DynEnumSet) -> Int:
    var acc = 0
    for i in range(0, len(s.chunks)):
        acc += Int(pop_count(s.chunks[i]))
    return acc