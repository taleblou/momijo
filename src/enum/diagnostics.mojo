# Module: momijo.enum.diagnostics
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
from .match import Case, RangeCase

# Does: utility function in enum module.
# Inputs: cases.
# Returns: result value or status.
fn find_duplicate_cases(cases: List[Case]) -> List[UInt64]:
    var out = List[UInt64](0)
    for i in range(0, len(cases)):
        for j in range(i+1, len(cases)):
            if cases[i].tag == cases[j].tag:
                out.append(cases[i].tag)
    return out

# Does: utility function in enum module.
# Inputs: ranges.
# Returns: result value or status.
fn find_overlapping_ranges(ranges: List[RangeCase]) -> List[(UInt64, UInt64)]:
    var out = List[(UInt64, UInt64)](0)
    for i in range(0, len(ranges)):
        var a = ranges[i]
        var alo = a.lo if a.lo <= a.hi else a.hi
        var ahi = a.hi if a.hi >= a.lo else a.lo
        for j in range(i+1, len(ranges)):
            var b = ranges[j]
            var blo = b.lo if b.lo <= b.hi else b.hi
            var bhi = b.hi if b.hi >= b.lo else b.lo
            if not (ahi < blo or bhi < alo):
                out.append((UInt64(i), UInt64(j)))
    return out

# Does: utility function in enum module.
# Inputs: cases, ranges.
# Returns: result value or status.
fn find_shadowed_cases(cases: List[Case], ranges: List[RangeCase]) -> List[UInt64]:
    var out = List[UInt64](0)
    for i in range(0, len(cases)):
        var t = cases[i].tag
        var shadow = False
        for r in range(0, len(ranges)):
            var lo = ranges[r].lo if ranges[r].lo <= ranges[r].hi else ranges[r].hi
            var hi = ranges[r].hi if ranges[r].hi >= ranges[r].lo else ranges[r].lo
            if t >= lo and t <= hi:
                shadow = True; break
        if shadow: out.append(t)
    return out

# Does: utility function in enum module.
# Inputs: cases, ranges, u_lo, u_hi.
# Returns: result value or status.
fn coverage_holes(cases: List[Case], ranges: List[RangeCase], u_lo: UInt64, u_hi: UInt64) -> List[UInt64]:
    var out = List[UInt64](0)
    for t in range(Int(u_lo), Int(u_hi)+1):
        var covered = False
        for i in range(0, len(cases)):
            if cases[i].tag == UInt64(t):
                covered = True; break
        if not covered:
            for r in range(0, len(ranges)):
                var lo = ranges[r].lo if ranges[r].lo <= ranges[r].hi else ranges[r].hi
                var hi = ranges[r].hi if ranges[r].hi >= ranges[r].lo else ranges[r].lo
                if UInt64(t) >= lo and UInt64(t) <= hi:
                    covered = True; break
        if not covered:
            out.append(UInt64(t))
    return out

# Does: utility function in enum module.
# Inputs: cases, ranges, u_lo, u_hi.
# Returns: result value or status.
fn assert_exhaustive_or_warn(cases: List[Case], ranges: List[RangeCase], u_lo: UInt64, u_hi: UInt64) -> Bool:
    var holes = coverage_holes(cases, ranges, u_lo, u_hi)
    if len(holes) == 0:
        return True
    print(String("[enumx] Non-exhaustive match; missing tags: ") + String(holes))
    return False

# Does: utility function in enum module.
# Inputs: cases.
# Returns: result value or status.
fn assert_no_duplicates_or_warn(cases: List[Case]) -> Bool:
    var dups = find_duplicate_cases(cases)
    if len(dups) == 0: return True
    print(String("[enumx] Duplicate exact cases: ") + String(dups))
    return False