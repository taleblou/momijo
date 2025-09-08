# Project:      Momijo
# Module:       src.momijo.enum.profile
# File:         profile.mojo
# Path:         src/momijo/enum/profile.mojo
#
# Description:  src.momijo.enum.profile — focused Momijo functionality with a stable public API.
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
#   - Structs: MatcherProfile
#   - Key functions: __init__, __copyinit__, __moveinit__, _density_local, collect_profile
#   - Uses generic functions/types with explicit trait bounds.


from .jumptable import JT_U16, JT_U32
from bit import pop_count
from momijo.core.device import kind
from momijo.core.error import module
from momijo.dataframe.helpers import m, t
from momijo.enum.match import Case, RangeCase
from momijo.enum.matcher import Matcher, RangeCase, match_get_build_count
from pathlib import Path
from pathlib.path import Path

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
fn __init__(out self, mode: UInt64 = 0, n_cases: Int = 0, n_ranges: Int = 0, density_times_100: UInt64 = 0, span: UInt64 = 0, jt_kind: UInt64 = 0, jt_slots: Int = 0, jt_bytes: UInt64 = 0, bitmask_nodes: Int = 0, fast_low8_bits: Int = 0, builds_so_far: UInt64 = 0) -> None:
        self.mode = mode
        self.n_cases = n_cases
        self.n_ranges = n_ranges
        self.density_times_100 = density_times_100
        self.span = span
        self.jt_kind = jt_kind
        self.jt_slots = jt_slots
        self.jt_bytes = jt_bytes
        self.bitmask_nodes = bitmask_nodes
        self.fast_low8_bits = fast_low8_bits
        self.builds_so_far = builds_so_far
fn __copyinit__(out self, other: Self) -> None:
        self.mode = other.mode
        self.n_cases = other.n_cases
        self.n_ranges = other.n_ranges
        self.density_times_100 = other.density_times_100
        self.span = other.span
        self.jt_kind = other.jt_kind
        self.jt_slots = other.jt_slots
        self.jt_bytes = other.jt_bytes
        self.bitmask_nodes = other.bitmask_nodes
        self.fast_low8_bits = other.fast_low8_bits
        self.builds_so_far = other.builds_so_far
fn __moveinit__(out self, deinit other: Self) -> None:
        self.mode = other.mode
        self.n_cases = other.n_cases
        self.n_ranges = other.n_ranges
        self.density_times_100 = other.density_times_100
        self.span = other.span
        self.jt_kind = other.jt_kind
        self.jt_slots = other.jt_slots
        self.jt_bytes = other.jt_bytes
        self.bitmask_nodes = other.bitmask_nodes
        self.fast_low8_bits = other.fast_low8_bits
        self.builds_so_far = other.builds_so_far
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