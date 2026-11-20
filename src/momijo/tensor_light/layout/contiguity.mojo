# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR LAYOUT / CONTIGUITY AND OVERLAP
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Layout and Strides
# Module:        momijo.tensor_light.layout.contiguity
# Source:        src/momijo/tensor_light/layout/contiguity.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines contiguity and overlap in the tensor layout and strides layer of
#   the FP32 Light tensor implementation.
#
#   Public declarations include `InternalOverlapKind`, `is_c_contiguous`,
#   `is_f_contiguous`, `internal_overlap_kind`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Compute and validate shape-to-storage layout mappings.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `InternalOverlapKind`,
#     `is_c_contiguous`, `is_f_contiguous`, `internal_overlap_kind`,
#     `has_zero_stride_broadcast` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Preserve correct offsets and spans for arbitrary legal strides.
#   - Detect overflow, invalid extents, and internal overlap.
#   - Never promote unknown provenance to proven non-aliasing.
#
# Performance Objectives:
#   - Prefer metadata-only layout analysis.
#   - Cache or reuse validated stride computations where safe.
#   - Do not recycle memory before lifetime and completion proofs hold.
#
# Maintainability:
#   - Use consistent C/F contiguity terminology.
#   - Test negative, zero, and unusually large strides.
#   - Cover recycled blocks, ownership failures, and overlapping buffers.
#
# ------------------------------------------------------------------------------
# PROJECT INFORMATION
# ------------------------------------------------------------------------------
#
# Authors:
#   Morteza Taleblou
#   Mitra Daneshmand
#
# Organization:  https://github.com/momijo
# Repository:    https://github.com/taleblou/momijo
#
# Maintainer Website:
#   https://taleblou.ir/
#
# Maintainer LinkedIn:
#   https://www.linkedin.com/in/morteza-taleblou
#
# ------------------------------------------------------------------------------
# LICENSING AND ATTRIBUTION
# ------------------------------------------------------------------------------
#
# License:       MIT
# License File:  LICENSE
#
# Copyright and Licensing:
#   This source file is distributed under the MIT License.
#   The SPDX declarations above identify its applicable license and
#   stated copyright holders.
#
#   Redistributions containing this software or substantial portions
#   of it must retain the applicable copyright and permission notices
#   as required by the MIT License.
#
# Attribution:
#   Acknowledgment of the Momijo project in downstream documentation
#   is appreciated but is not an additional licensing requirement.
#
# ==============================================================================

# Conservative overlap analysis; UNKNOWN must not be treated as writable.
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..internal.checked_math import checked_sub, checked_add
from .offset import checked_relative_span

@fieldwise_init
struct InternalOverlapKind(ImplicitlyCopyable, Equatable):
    var code: Int
    comptime NONE = Self(0)
    comptime DEFINITE = Self(1)
    comptime UNKNOWN = Self(2)
    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

def is_c_contiguous[rank: Int](shape: TensorShape[rank], strides: StaticRankArray[rank]) -> Bool:
    # This predicate is public and may receive fabricated cached numel. Do
    # not let a wrapped stride product certify a contiguous/no-overlap view.
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok() or canonical.payload.value().numel() != shape.numel():
        return False
    if shape.is_empty():
        return True
    var expected = 1
    comptime for j in range(rank):
        var i = rank - 1 - j
        var extent = shape.dims.get(i)
        if extent > 1:
            if strides.get(i) != expected:
                return False
            if expected > (Int.MAX // 4) // extent:
                return False
            expected *= extent
    return True

def is_f_contiguous[rank: Int](shape: TensorShape[rank], strides: StaticRankArray[rank]) -> Bool:
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok() or canonical.payload.value().numel() != shape.numel():
        return False
    if shape.is_empty():
        return True
    var expected = 1
    comptime for i in range(rank):
        var extent = shape.dims.get(i)
        if extent > 1:
            if strides.get(i) != expected:
                return False
            if expected > (Int.MAX // 4) // extent:
                return False
            expected *= extent
    return True

# GCD is used only in checked metadata analysis, never an element loop.
# For two active axes, a pairwise collision exists precisely when the
# smallest nonzero index deltas |s_b|/g and |s_a|/g both fit their extents.
# This catches proven overlaps such as shape=(2,2), strides=(2,2) that the
# sorted-stride sufficient uniqueness rule would conservatively call UNKNOWN.
def _stride_gcd_positive(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var remainder = x % y
        x = y
        y = remainder
    return x


def internal_overlap_kind[rank: Int](shape: TensorShape[rank], strides: StaticRankArray[rank]) -> InternalOverlapKind:
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok() or canonical.payload.value().numel() != shape.numel():
        return InternalOverlapKind.UNKNOWN
    if shape.is_empty() or shape.numel() == 1:
        return InternalOverlapKind.NONE
    comptime for i in range(rank):
        if shape.dims.get(i) > 1 and strides.get(i) == 0:
            return InternalOverlapKind.DEFINITE
    if is_c_contiguous(shape, strides) or is_f_contiguous(shape, strides):
        return InternalOverlapKind.NONE
    # Pairwise affine collisions are DEFINITE, not merely UNKNOWN.  Different
    # signs are covered by the absolute magnitudes: choose opposite-signed
    # index deltas when strides have equal signs, and same-signed deltas when
    # they have opposite signs.  Both coordinates remain in range by placing
    # the shorter delta at opposite ends of each axis.  This is a bounded
    # O(rank^2) metadata pass (rank <= 8), never a hot element-loop branch.
    comptime for a in range(rank):
        if shape.dims.get(a) > 1:
            var stride_a = strides.get(a)
            if stride_a != Int.MIN and stride_a != 0:
                if stride_a < 0:
                    stride_a = -stride_a
                comptime for b in range(a + 1, rank):
                    if shape.dims.get(b) > 1:
                        var stride_b = strides.get(b)
                        if stride_b != Int.MIN and stride_b != 0:
                            if stride_b < 0:
                                stride_b = -stride_b
                            var common = _stride_gcd_positive(stride_a, stride_b)
                            # The zero-stride case has already returned
                            # DEFINITE above. Both arguments are > 0 here.
                            var delta_a = stride_b // common
                            var delta_b = stride_a // common
                            if (delta_a <= shape.dims.get(a) - 1
                                and delta_b <= shape.dims.get(b) - 1):
                                return InternalOverlapKind.DEFINITE
    # Positive or negative non-overlapping permuted dense layouts may be
    # proven with a sorted absolute-stride sufficient condition.
    var ordered = Array[Int, 8](fill=0)
    var used = Array[Bool, 8](fill=False)
    var active = 0
    comptime for i in range(rank):
        if shape.dims.get(i) > 1:
            active += 1
    for k in range(active):
        var chosen = -1
        var smallest = Int.MAX
        comptime for i in range(rank):
            if shape.dims.get(i) > 1 and not used[i]:
                var step = strides.get(i)
                if step == Int.MIN:
                    return InternalOverlapKind.UNKNOWN
                if step < 0:
                    step = -step
                if step < smallest:
                    chosen = i
                    smallest = step
        if chosen < 0:
            return InternalOverlapKind.UNKNOWN
        ordered[k] = chosen
        used[chosen] = True
    var occupied_span = 1
    for k in range(active):
        var i = ordered[k]
        var stride = strides.get(i)
        if stride < 0:
            stride = -stride
        if stride < occupied_span:
            return InternalOverlapKind.UNKNOWN
        # Reject overflow before updating the sufficient-condition span.
        # A conservative UNKNOWN is always safer than a false no-alias proof.
        var extra = shape.dims.get(i) - 1
        if stride <= 0:
            return InternalOverlapKind.UNKNOWN
        if extra > (Int.MAX - occupied_span) // stride:
            return InternalOverlapKind.UNKNOWN
        occupied_span += extra * stride
    return InternalOverlapKind.NONE

def has_zero_stride_broadcast[rank: Int](shape: TensorShape[rank], strides: StaticRankArray[rank]) -> Bool:
    # This public metadata predicate may influence overlap/reuse scheduling.
    # A caller can forge the cached `elements` field after construction: do
    # not certify a broadcast from extents belonging to an invalid shape.
    # Other contiguity/overlap queries use this same fail-closed validation.
    var canonical = shape.checked_numel()
    if not canonical.is_ok():
        return False
    comptime for i in range(rank):
        if shape.dims.get(i) > 1 and strides.get(i) == 0:
            return True
    return False


def is_unique_mapping[rank: Int](shape: TensorShape[rank], strides: StaticRankArray[rank]) -> Bool:
    # UNKNOWN is not sufficient evidence for mutable/parallel noalias access.
    return internal_overlap_kind[rank](shape, strides) == InternalOverlapKind.NONE


def is_exhaustive_mapping[rank: Int](shape: TensorShape[rank], strides: StaticRankArray[rank]) -> Bool:
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok() or canonical.payload.value().numel() != shape.numel():
        return False
    if shape.is_empty():
        return True
    if not is_unique_mapping[rank](shape, strides):
        return False
    # Exhaustiveness is a property of the relative mapping. Its logical
    # origin need not be allocation offset zero (e.g. a reversed dense view).
    var physical = checked_relative_span[rank](shape, strides)
    if not physical.is_ok() or physical.payload.value().empty:
        return False
    var width = checked_sub(physical.payload.value().hi, physical.payload.value().lo)
    if not width.is_ok():
        return False
    var count = checked_add(width.payload.value(), 1)
    if not count.is_ok():
        return False
    return count.payload.value() == shape.numel()
