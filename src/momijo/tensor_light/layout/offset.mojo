# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR LAYOUT / PHYSICAL STORAGE OFFSETS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Layout and Strides
# Module:        momijo.tensor_light.layout.offset
# Source:        src/momijo/tensor_light/layout/offset.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines physical storage offsets in the tensor layout and strides layer of
#   the FP32 Light tensor implementation.
#
#   Public declarations include `PhysicalSpan`, `checked_relative_span`,
#   `checked_physical_span`, `physical_offset_unchecked`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Compute and validate shape-to-storage layout mappings.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `PhysicalSpan`,
#     `checked_relative_span`, `checked_physical_span`,
#     `physical_offset_unchecked`, `physical_span` and related symbols.
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

# Signed-stride range calculation with no unchecked pointer arithmetic.
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..internal.checked_math import checked_add, checked_sub, checked_mul
from ..internal.status import TensorStatus, TensorResult

@fieldwise_init
struct PhysicalSpan(ImplicitlyCopyable):
    var lo: Int
    var hi: Int
    var empty: Bool

# Relative bounds are independent of the chosen allocation origin. In
# particular, a reversed view has a negative relative minimum but may be
# perfectly valid once its owner supplies a positive logical-origin offset.
# Do not call checked_physical_span(..., offset=0) to measure its size: that
# function describes an actual storage address, not a relative footprint.
def checked_relative_span[rank: Int](
    shape: TensorShape[rank], strides: StaticRankArray[rank],
) -> TensorResult[PhysicalSpan]:
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok():
        return TensorResult[PhysicalSpan](canonical.status)
    if canonical.payload.value().numel() != shape.numel():
        return TensorResult[PhysicalSpan](TensorStatus.INVALID_SHAPE)
    if canonical.payload.value().is_empty():
        return TensorResult[PhysicalSpan](PhysicalSpan(0, 0, True))
    var lo = 0
    var hi = 0
    comptime for i in range(rank):
        var extent = canonical.payload.value().dims.get(i)
        # Singleton dimensions contribute no distance, even for an
        # unrepresentable negative stride such as Int.MIN.
        if extent > 1:
            var product = checked_mul(extent - 1, strides.get(i))
            if not product.is_ok():
                return TensorResult[PhysicalSpan](product.status)
            var delta = product.payload.value()
            if delta < 0:
                var added = checked_add(lo, delta)
                if not added.is_ok():
                    return TensorResult[PhysicalSpan](added.status)
                lo = added.payload.value()
            else:
                var added = checked_add(hi, delta)
                if not added.is_ok():
                    return TensorResult[PhysicalSpan](added.status)
                hi = added.payload.value()
    return TensorResult[PhysicalSpan](PhysicalSpan(lo, hi, False))

def checked_physical_span[rank: Int](
    shape: TensorShape[rank], strides: StaticRankArray[rank], offset: Int,
) -> TensorResult[PhysicalSpan]:
    # Actual storage coordinates are nonnegative, but relative stride
    # displacement need not be. Use the ONE checked relative-range analysis
    # for both allocation bounds and required-span-size calculations.
    if offset < 0:
        return TensorResult[PhysicalSpan](TensorStatus.INVALID_LAYOUT)
    var relative = checked_relative_span[rank](shape, strides)
    if not relative.is_ok():
        return TensorResult[PhysicalSpan](relative.status)
    if relative.payload.value().empty:
        return TensorResult[PhysicalSpan](PhysicalSpan(offset, offset, True))
    var lo = checked_add(offset, relative.payload.value().lo)
    if not lo.is_ok():
        return TensorResult[PhysicalSpan](lo.status)
    var hi = checked_add(offset, relative.payload.value().hi)
    if not hi.is_ok():
        return TensorResult[PhysicalSpan](hi.status)
    return TensorResult[PhysicalSpan](PhysicalSpan(
        lo.payload.value(), hi.payload.value(), False,
    ))

# Layout-only checked helpers, independent of allocation ownership.
def physical_offset_unchecked[rank: Int](
    shape: TensorShape[rank], strides: StaticRankArray[rank],
    base_offset: Int, indices: StaticRankArray[rank],
) -> Int:
    # PRECONDITION: checked layout and in-bounds coordinates. Called in hot
    # loops only after the owning Tensor proves the physical allocation span.
    var position = base_offset
    comptime for i in range(rank):
        position += indices.get(i) * strides.get(i)
    return position


def physical_span[rank: Int](
    shape: TensorShape[rank], strides: StaticRankArray[rank],
    base_offset: Int,
) -> TensorResult[PhysicalSpan]:
    # checked_physical_span now owns the complete shape/cache validation;
    # do not repeat it at this delegating metadata boundary.
    return checked_physical_span[rank](shape, strides, base_offset)


def min_physical_offset[rank: Int](
    shape: TensorShape[rank], strides: StaticRankArray[rank], base_offset: Int,
) -> TensorResult[Int]:
    var span = physical_span[rank](shape, strides, base_offset)
    if not span.is_ok():
        return TensorResult[Int](span.status)
    return TensorResult[Int](span.payload.value().lo)


def max_physical_offset[rank: Int](
    shape: TensorShape[rank], strides: StaticRankArray[rank], base_offset: Int,
) -> TensorResult[Int]:
    var span = physical_span[rank](shape, strides, base_offset)
    if not span.is_ok():
        return TensorResult[Int](span.status)
    return TensorResult[Int](span.payload.value().hi)


def required_span_size[rank: Int](
    shape: TensorShape[rank], strides: StaticRankArray[rank],
) -> TensorResult[Int]:
    var span = checked_relative_span[rank](shape, strides)
    if not span.is_ok():
        return TensorResult[Int](span.status)
    if span.payload.value().empty:
        return TensorResult[Int](0)
    var difference = checked_sub(span.payload.value().hi, span.payload.value().lo)
    if not difference.is_ok():
        return TensorResult[Int](difference.status)
    return checked_add(difference.payload.value(), 1)
