#
# Copyright (c) 2025 Morteza Taleblou (https://taleblou.ir/)
# All rights reserved.
#
from .jumptable import JumpTable, build_jump_table, jump_lookup
from bit import pop_count

var _enumx_build_count: UInt64 = 0
fn match_get_build_count() -> UInt64:
    return _enumx_build_count

struct Case:
    var tag: UInt64
    var arm: UInt64

struct RangeCase:
    var lo: UInt64
    var hi: UInt64
    var arm: UInt64

# Sparse tree
struct SparseTree:
    var n: Int
    var tags: List[UInt64]
    var arms: List[UInt64]

fn build_sparse_tree(cases: List[Case]) -> SparseTree:
    var n = Int(len(cases))
    var tags = List[UInt64](n)
    var arms = List[UInt64](n)
    for i in range(0, n):
        tags[i] = cases[i].tag
        arms[i] = cases[i].arm
    var i = 1
    while i < n:
        var key_t = tags[i]; var key_a = arms[i]; var j = i - 1
        while j >= 0 and tags[j] > key_t:
            tags[j+1] = tags[j]; arms[j+1] = arms[j]; j -= 1
        tags[j+1] = key_t; arms[j+1] = key_a; i += 1
    return SparseTree(n=n, tags=tags, arms=arms)

fn match_sparse_tree(tag: UInt64, default_arm: UInt64, st: SparseTree) -> UInt64:
    var lo: Int = 0; var hi: Int = st.n - 1
    while lo <= hi:
        var mid = lo + ((hi - lo) // 2)
        var t = st.tags[mid]
        if t == tag: return st.arms[mid]
        if t < tag: lo = mid + 1
        else: hi = mid - 1
    return default_arm

# Range table
struct RangeTable:
    var n: Int
    var los: List[UInt64]
    var his: List[UInt64]
    var arms: List[UInt64]

fn build_range_table(ranges: List[RangeCase]) -> RangeTable:
    var n = Int(len(ranges))
    var los = List[UInt64](n); var his = List[UInt64](n); var arms = List[UInt64](n)
    for i in range(0, n):
        var a = ranges[i]
        if a.hi < a.lo: var tmp = a.lo; a.lo = a.hi; a.hi = tmp
        los[i] = a.lo; his[i] = a.hi; arms[i] = a.arm
    var i = 1
    while i < n:
        var lo = los[i]; var hi = his[i]; var ar = arms[i]; var j = i - 1
        while j >= 0 and los[j] > lo:
            los[j+1] = los[j]; his[j+1] = his[j]; arms[j+1] = arms[j]; j -= 1
        los[j+1] = lo; his[j+1] = hi; arms[j+1] = ar; i += 1
    return RangeTable(n=n, los=los, his=his, arms=arms)

fn match_range_table(tag: UInt64, default_arm: UInt64, rt: RangeTable) -> UInt64:
    var lo: Int = 0; var hi: Int = rt.n - 1; var idx: Int = -1
    while lo <= hi:
        var mid = lo + ((hi - lo) // 2)
        var mlo = rt.los[mid]
        if mlo <= tag: idx = mid; lo = mid + 1
        else: hi = mid - 1
    if idx >= 0 and tag <= rt.his[idx]: return rt.arms[idx]
    return default_arm

# Bitmask cluster
struct BitMaskNode:
    var mask: UInt64
    var has_left: Bool
    var left_tag: UInt64
    var left_arm: UInt64
    var has_right: Bool
    var right_tag: UInt64
    var right_arm: UInt64

struct BitMaskTree:
    var nodes: List[BitMaskNode]
    var default_arm: UInt64

fn _best_single_bit(cases: List[Case]) -> UInt64:
    var best_bit: UInt64 = 0
    var best_score: UInt64 = 0xFFFFFFFFFFFFFFFF
    for b in range(0, 16):
        var left = 0
        for i in range(0, len(cases)):
            if (cases[i].tag >> UInt64(b)) & 1 == 0:
                left += 1
        var right = len(cases) - left
        var diff = left - right if left >= right else right - left
        if UInt64(diff) < best_score:
            best_score = UInt64(diff)
            best_bit = UInt64(b)
    return best_bit

fn build_bitmask_tree(cases: List[Case], default_arm: UInt64, depth_limit: Int = 3) -> BitMaskTree:
    var nodes = List[BitMaskNode](0)
    var work_cases = cases
    for d in range(0, depth_limit):
        if len(work_cases) <= 2:
            break
        var b = _best_single_bit(work_cases)
        var lefts = List[Case](0)
        var rights = List[Case](0)
        for i in range(0, len(work_cases)):
            if ((work_cases[i].tag >> b) & 1) == 0:
                lefts.append(work_cases[i])
            else:
                rights.append(work_cases[i])
        var node = BitMaskNode(mask=(1 << b), has_left=False, left_tag=0, left_arm=default_arm,
                               has_right=False, right_tag=0, right_arm=default_arm)
        if len(lefts) == 1:
            node.has_left = False; node.left_tag = lefts[0].tag; node.left_arm = lefts[0].arm
        else:
            node.has_left = True
        if len(rights) == 1:
            node.has_right = False; node.right_tag = rights[0].tag; node.right_arm = rights[0].arm
        else:
            node.has_right = True
        nodes.append(node)
        work_cases = lefts if len(lefts) > len(rights) else rights
    return BitMaskTree(nodes=nodes, default_arm=default_arm)

fn match_bitmask(tag: UInt64, default_arm: UInt64, bt: BitMaskTree) -> (Bool, UInt64):
    for i in range(0, len(bt.nodes)):
        var n = bt.nodes[i]
        var branch = (tag & n.mask) != 0
        if branch:
            if not n.has_right:
                if tag == n.right_tag: return (True, n.right_arm)
        else:
            if not n.has_left:
                if tag == n.left_tag: return (True, n.left_arm)
    return (False, default_arm)

# Dense JT
struct DenseJT:
    var jt: JumpTable

fn build_dense(cases: List[Case], default_arm: UInt64) -> DenseJT:
    return DenseJT(jt=build_jump_table(cases, default_arm))

fn match_dense(tag: UInt64, default_arm: UInt64, dj: DenseJT) -> UInt64:
    return jump_lookup(tag, dj.jt)

# Matcher
struct Matcher:
    var mode: UInt64   # 0=sparse, 1=ranges, 2=dense, 3=bitmask
    var tree: SparseTree
    var rt: RangeTable
    var dj: DenseJT
    var bt: BitMaskTree
    var default_arm: UInt64
    var fast_low8: UInt64

fn _density(cases: List[Case]) -> (Float64, UInt64, UInt64):
    if len(cases) == 0:
        return (9999.0, 0, 0)
    var lo = cases[0].tag; var hi = cases[0].tag
    for i in range(1, len(cases)):
        var t = cases[i].tag
        if t < lo: lo = t
        if t > hi: hi = t
    var span = (hi - lo) + 1
    var n = UInt64(len(cases))
    var dens = Float64(span) / Float64(n)
    return (dens, lo, hi)

fn build_matcher(cases: List[Case], ranges: List[RangeCase], density_factor: Float64 = 2.0, default_arm: UInt64 = 0) -> Matcher:
    var tree = build_sparse_tree(cases)
    var rt = build_range_table(ranges)
    var dj = DenseJT(jt=build_jump_table([], default_arm))
    var bt = build_bitmask_tree(cases, default_arm, 3)
    var mode: UInt64 = 0

    # low8 bitmap
    var low_mask: UInt64 = 0
    for i in range(0, len(cases)):
        var low = cases[i].tag & 63
        low_mask = low_mask | (1 << low)

    # selection
    if len(cases) > 0:
        var (dens, _, _) = _density(cases)
        if dens <= density_factor and len(cases) >= 4:
            dj = build_dense(cases, default_arm)
            mode = 2
        elif len(bt.nodes) > 0 and len(cases) >= 4:
            mode = 3
        else:
            mode = 0
    if len(ranges) > 0:
        if mode != 2:
            mode = 1 if mode != 3 else 3

    _enumx_build_count += 1
    return Matcher(mode=mode, tree=tree, rt=rt, dj=dj, bt=bt, default_arm=default_arm, fast_low8=low_mask)

fn match_with_selector(tag: UInt64, default_arm: UInt64, m: Matcher) -> UInt64:
    # fast reject for cases (ranges still considered)
    if ((m.fast_low8 >> (tag & 63)) & 1) == 0:
        if m.mode == 1:
            var r0 = match_range_table(tag, default_arm, m.rt)
            if r0 != default_arm: return r0
        if m.mode == 2:
            return match_dense(tag, default_arm, m.dj)
        return default_arm

    if m.mode == 1:
        var r = match_range_table(tag, default_arm, m.rt)
        if r != default_arm: return r
    if m.mode == 2:
        return match_dense(tag, default_arm, m.dj)
    if m.mode == 3:
        var (ok, arm) = match_bitmask(tag, default_arm, m.bt)
        if ok: return arm
        return match_sparse_tree(tag, default_arm, m.tree)
    return match_sparse_tree(tag, default_arm, m.tree)
