# Module: momijo.enum.repr
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
from sys.info import sizeof, alignof
from bit import log2_ceil

const STRAT_EXPLICIT: UInt64 = 0
const STRAT_NICHE: UInt64 = 1
const STRAT_TAGGING: UInt64 = 2

# Data structure representing a concept in the enum library.
struct EnumRepr:
    var k: UInt64
    var tag_bytes: UInt64
    var union_size: UInt64
    var union_align: UInt64
    var strategy: UInt64
    var boxing_threshold: UInt64
    var boxed_mask_low64: UInt64

# Data structure representing a concept in the enum library.
struct PayloadMeta:
    var size: UInt64
    var align: UInt64
    var has_null_niche: Bool
    var has_nonzero_niche: Bool

# Does: utility function in enum module.
# Inputs: —.
# Returns: result value or status.
fn meta_empty() -> PayloadMeta:
    return PayloadMeta(size=0, align=1, has_null_niche=False, has_nonzero_niche=False)

# Does: utility function in enum module.
# Inputs: —.
# Returns: result value or status.
fn meta_ptr_like() -> PayloadMeta:
    var ps = UInt64(sizeof[UnsafePointer[UInt8]]())
    var pa = UInt64(alignof[UnsafePointer[UInt8]]())
    return PayloadMeta(size=ps, align=pa, has_null_niche=True, has_nonzero_niche=False)

# Does: utility function in enum module.
# Inputs: —.
# Returns: result value or status.
fn meta_nonzero64() -> PayloadMeta:
    return PayloadMeta(size=8, align=8, has_null_niche=False, has_nonzero_niche=True)

# Does: utility function in enum module.
# Inputs: size, align.
# Returns: result value or status.
fn meta_from_size_align(size: UInt64, align: UInt64) -> PayloadMeta:
    var ps = UInt64(sizeof[UnsafePointer[UInt8]]())
    var pa = UInt64(alignof[UnsafePointer[UInt8]]())
    var is_ptr = (size == ps and align >= pa)
    return PayloadMeta(size=size, align=align, has_null_niche=is_ptr, has_nonzero_niche=False)

# Does: utility function in enum module.
# Inputs: k.
# Returns: result value or status.
fn min_tag_bytes(k: UInt64) -> UInt64:
    if k <= 1: return 0
    var bits = UInt64(log2_ceil(Int(k-1) + 1))
    var bytes = (bits + 7) / 8
    return bytes if bytes > 1 else 1

# Does: utility function in enum module.
# Inputs: a, b.
# Returns: result value or status.
fn fold_max(a: UInt64, b: UInt64) -> UInt64:
    return a if a >= b else b

# Does: utility function in enum module.
# Inputs: k, sizes, aligns, prefer_niche, prefer_tagging.
# Returns: result value or status.
fn enum_desc(k: UInt64, sizes: List[UInt64], aligns: List[UInt64], prefer_niche: Bool, prefer_tagging: Bool) -> EnumRepr:
    var maxs: UInt64 = 0
    var maxa: UInt64 = 1
    for i in range(0, Int(k)):
        maxs = fold_max(maxs, sizes[i])
        maxa = fold_max(maxa, aligns[i])
    var strat = STRAT_EXPLICIT
    if prefer_niche: strat = STRAT_NICHE
    elif prefer_tagging: strat = STRAT_TAGGING
    return EnumRepr(k=k, tag_bytes=min_tag_bytes(k), union_size=maxs, union_align=maxa, strategy=strat, boxing_threshold=32, boxed_mask_low64=0)

# Does: utility function in enum module.
# Inputs: k, metas, dense_prefer.
# Returns: result value or status.
fn make_enum_repr_auto2(k: UInt64, metas: List[PayloadMeta], dense_prefer: Bool=False) -> EnumRepr:
    var maxs: UInt64 = 0
    var maxa: UInt64 = 1
    var n_empty = 0
    var n_ptr_like = 0
    for i in range(0, Int(k)):
        var m = metas[i]
        maxs = fold_max(maxs, m.size)
        maxa = fold_max(maxa, m.align)
        if m.size == 0: n_empty += 1
        if m.has_null_niche: n_ptr_like += 1
    var strat = STRAT_EXPLICIT
    if (n_empty >= 1) and (n_ptr_like == 1) and (k <= 8):
        strat = STRAT_NICHE
    return EnumRepr(
        k=k,
        tag_bytes=min_tag_bytes(k),
        union_size=maxs,
        union_align=maxa,
        strategy=strat,
        boxing_threshold=64 if maxs > 64 else 32,
        boxed_mask_low64=0
    )