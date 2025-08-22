# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Module: momijo.enum.profile
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
from .match import Matcher, Case, RangeCase, match_get_build_count
from .jumptable import JT_U16, JT_U32, JT_U64
from bit import pop_count

# Data structure representing a concept in the enum library.
struct MatcherProfile:
    var mode: UInt64
    var n_cases: Int
    var n_ranges: Int
    var density_times_100: UInt64
    var span: UInt64
    var jt_kind: UInt64
    var jt_slots: Int
    var jt_bytes: UInt64
    var bitmask_nodes: Int
    var fast_low8_bits: Int
    var builds_so_far: UInt64

# Does: utility function in enum module.
# Inputs: cases.
# Returns: result value or status.
fn _density_local(cases: List[Case]) -> (UInt64, UInt64, UInt64):
    if len(cases) == 0: return (0, 0, 0)
    var lo = cases[0].tag; var hi = cases[0].tag
    for i in range(1, len(cases)):
        var t = cases[i].tag
        if t < lo: lo = t
        if t > hi: hi = t
    var span = (hi - lo) + 1
    var dens_times_100 = (span * 100) / UInt64(len(cases))
    return (dens_times_100, lo, hi)

# Does: utility function in enum module.
# Inputs: cases, ranges, m.
# Returns: result value or status.
fn collect_profile(cases: List[Case], ranges: List[RangeCase], m: Matcher) -> MatcherProfile:
    var (dens100, lo, hi) = _density_local(cases)
    var span = (hi - lo + 1) if len(cases) > 0 else 0
    var jt_slots = m.dj.jt.span
    var jt_bytes: UInt64 = 0
    if m.dj.jt.kind == JT_U16:
        jt_bytes = UInt64(jt_slots * 2)
    elif m.dj.jt.kind == JT_U32:
        jt_bytes = UInt64(jt_slots * 4)
    else:
        jt_bytes = UInt64(jt_slots * 8)
    var fast_bits = Int(pop_count(m.fast_low8))
    var nodes = 0
    if m.mode == 3:
        nodes = len(m.bt.nodes)
    return MatcherProfile(
        mode=m.mode,
        n_cases=len(cases),
        n_ranges=len(ranges),
        density_times_100=dens100,
        span=span,
        jt_kind=m.dj.jt.kind,
        jt_slots=jt_slots,
        jt_bytes=jt_bytes,
        bitmask_nodes=nodes,
        fast_low8_bits=fast_bits,
        builds_so_far=match_get_build_count()
    )