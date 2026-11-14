# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR ACCESS / SPAN
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Access and References
# Module:        momijo.tensor_light.access.span
# Source:        src/momijo/tensor_light/access/span.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines span in the access and references layer of the FP32 Light tensor
#   implementation.
#
#   Public declarations include `as_c_span`, `as_f_span`, `as_mut_c_span`,
#   `as_mut_f_span`; preconditions and supported targets are determined by
#   their implementations.
#
# Responsibilities:
#   - Provide checked tensor element access without exceeding storage bounds.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `as_c_span`, `as_f_span`,
#     `as_mut_c_span`, `as_mut_f_span`, `as_span` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate index bounds, backing capacity, and access mutability.
#   - Never expose an element reference beyond its owning storage lifetime.
#
# Performance Objectives:
#   - Keep checked metadata setup outside element-wise hot loops.
#   - Avoid copying tensor storage when a verified accessor is sufficient.
#
# Maintainability:
#   - Document borrowing, aliasing, and raw-pointer preconditions.
#   - Test strided, empty, and offset access paths.
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

# One checked zero-copy Span boundary for both C-order and F-order views.
# The view/adapters module owns the canonical Layout validation. Delegating
# here preserves the access/span responsibility without a second unchecked
# implementation or allocation. No raw pointer is constructed or fabricated.


from std.collections import Span
from ..view.view import TensorView, TensorMutView
from ..view.adapters import as_c_span as _checked_c_span
from ..view.adapters import as_f_span as _checked_f_span
from ..view.adapters import mut_as_c_span as _checked_mut_c_span
from ..view.adapters import mut_as_f_span as _checked_mut_f_span
from ..internal.status import TensorResult
from ..layout.memory_format import ContiguousOrder
from ..shape.shape import TensorShape
from ..internal.status import TensorStatus
from ..alias.overlap import host_output_disjoint_from_input
from ..layout.layout_analysis import is_reverse_c_contiguous
from ..iterator.nd_iterator import NDIndexIterator
from std.memory import unsafe_memcpy, unsafe_memmove
from ..storage.host_storage import HostStorage

def as_c_span[rank: Int, origin: Origin](
    view: TensorView[rank, origin],
) -> TensorResult[Span[False, Float32, origin]]:
    return _checked_c_span(view)


def as_f_span[rank: Int, origin: Origin](
    view: TensorView[rank, origin],
) -> TensorResult[Span[False, Float32, origin]]:
    return _checked_f_span(view)


def as_mut_c_span[rank: Int, origin: Origin[mut=True]](
    mut view: TensorMutView[rank, origin],
) -> TensorResult[Span[True, Float32, origin]]:
    return _checked_mut_c_span(view)


def as_mut_f_span[rank: Int, origin: Origin[mut=True]](
    mut view: TensorMutView[rank, origin],
) -> TensorResult[Span[True, Float32, origin]]:
    return _checked_mut_f_span(view)


# Explicit order selection prevents a silent C-vs-F materialization choice.


def as_span[rank: Int, origin: Origin](
    view: TensorView[rank, origin], order: ContiguousOrder,
) -> TensorResult[Span[False, Float32, origin]]:
    if order == ContiguousOrder.C:
        return as_c_span(view)
    if order == ContiguousOrder.F:
        return as_f_span(view)
    return TensorResult[Span[False, Float32, origin]](TensorStatus.INVALID_LAYOUT)


def as_mut_span[rank: Int, origin: Origin[mut=True]](
    mut view: TensorMutView[rank, origin], order: ContiguousOrder,
) -> TensorResult[Span[True, Float32, origin]]:
    if order == ContiguousOrder.C:
        return as_mut_c_span(view)
    if order == ContiguousOrder.F:
        return as_mut_f_span(view)
    return TensorResult[Span[True, Float32, origin]](TensorStatus.INVALID_LAYOUT)


def span_matches_shape[rank: Int, origin: Origin](
    span: Span[False, Float32, origin], shape: TensorShape[rank],
) -> Bool:
    var canonical = TensorShape[rank].checked(shape.dims)
    return (canonical.is_ok()
            and canonical.payload.value().numel() == shape.numel()
            and len(span) == shape.numel())


# Public structural copy adapters use logical C-order iteration even for
# F-contiguous, transposed, reversed, or otherwise strided source views.
# The direct memcpy path is legal only for proven-disjoint physical spans.
# Potentially overlapping C-order copies use memmove; arbitrary strided
# overlaps explicitly snapshot their inputs to preserve logical copy order.


def copy_to_span[rank: Int, src_origin: Origin, dst_origin: Origin[mut=True]](
    source: TensorView[rank, src_origin],
    mut destination: Span[True, Float32, dst_origin],
) -> TensorStatus:
    var verified = source._checked_current_layout()
    if not verified.is_ok():
        return verified.status
    var mapping = verified.payload.value()
    var count = mapping.shape.numel()
    if count != len(destination):
        return TensorStatus.SIZE_MISMATCH
    # The destination may overlap the source backing storage (including
    # addresses not currently selected by a strided view). A no-scratch
    # logical-order copy must never overwrite unread source elements.
    if count == 0:
        return TensorStatus.OK
    # The owner may be larger than this view. Only the proven physical
    # envelope [span_lo, span_hi] can contain a source element; overlap with
    # UNUSED storage must not force a false ALIAS_UNKNOWN. An envelope may
    # also include layout holes, so this remains a conservative proof.
    var source_envelope = source.data[mapping.span_lo:mapping.span_hi + 1]
    var overlaps_source = not host_output_disjoint_from_input(destination, source_envelope)
    if mapping.is_c_contiguous() and mapping.is_exhaustive():
        # Both sides have identical forward logical ordering. memmove is
        # correct for exact and shifted aliases; disjoint ranges use memcpy.
        var contiguous = as_c_span(source)
        if not contiguous.is_ok():
            return contiguous.status
        if overlaps_source:
            unsafe_memmove(
                dest=destination.unsafe_ptr(),
                src=contiguous.payload.value().unsafe_ptr(),
                count=count,
            )
        else:
            unsafe_memcpy(
                dest=destination.unsafe_ptr(),
                src=contiguous.payload.value().unsafe_ptr(),
                count=count,
            )
        return TensorStatus.OK
    if (not overlaps_source and mapping.is_exhaustive()
        and is_reverse_c_contiguous(mapping)):
        # Explicit logical C-order copy of a fully reversed dense view.
        # Both physical envelopes are disjoint, so no staging owner is needed.
        # Unlike a reverse-F traversal this ordering is valid for all ranks.
        var read_pointer = source.data.unsafe_ptr()
        var write_pointer = destination.unsafe_ptr()
        var last = mapping.span_hi
        for i in range(count):
            write_pointer.unsafe_store(i, read_pointer.unsafe_load(last - i)[0])
        return TensorStatus.OK
    if overlaps_source:
        # A strided/reversed/F-order source may have cyclic dependencies with
        # the destination. Snapshot every logical element before ANY write;
        # an overlap is not a reason to silently fail a supported copy API.
        var pending = HostStorage._allocate_for_complete_overwrite(count)
        if not pending.is_ok():
            return pending.status
        var staging = pending.payload.take()
        var scratch = staging.borrow_mut_span()
        var read_cursor = NDIndexIterator[rank](mapping.shape)
        for i in range(count):
            scratch[i] = source.data[
                mapping.physical_offset_unchecked(read_cursor.coordinates)
            ]
            read_cursor.advance()
        for i in range(count):
            destination[i] = scratch[i]
        return TensorStatus.OK
    var cursor = NDIndexIterator[rank](mapping.shape)
    for i in range(count):
        destination[i] = source.data[
            mapping.physical_offset_unchecked(cursor.coordinates)
        ]
        cursor.advance()
    return TensorStatus.OK


def copy_from_span[rank: Int, dst_origin: Origin[mut=True], src_origin: Origin](
    mut destination: TensorMutView[rank, dst_origin],
    source: Span[False, Float32, src_origin],
) -> TensorStatus:
    var verified = destination._checked_current_layout()
    if not verified.is_ok():
        return verified.status
    var mapping = verified.payload.value()
    if not mapping.is_proven_unique():
        return TensorStatus.OVERLAPPING
    var count = mapping.shape.numel()
    if len(source) != count:
        return TensorStatus.SIZE_MISMATCH
    if count == 0:
        return TensorStatus.OK
    # The mutable view may address only a small physical envelope of a
    # larger owning allocation. The proven envelope, not the entire owner,
    # is sufficient for a conservative no-overlap check against the source.
    var target_envelope = destination.data[mapping.span_lo:mapping.span_hi + 1]
    var overlaps_destination = not host_output_disjoint_from_input(target_envelope, source)
    if mapping.is_c_contiguous() and mapping.is_exhaustive():
        var contiguous = as_mut_c_span(destination)
        if not contiguous.is_ok():
            return contiguous.status
        if overlaps_destination:
            unsafe_memmove(
                dest=contiguous.payload.value().unsafe_ptr(),
                src=source.unsafe_ptr(),
                count=count,
            )
        else:
            unsafe_memcpy(
                dest=contiguous.payload.value().unsafe_ptr(),
                src=source.unsafe_ptr(),
                count=count,
            )
        return TensorStatus.OK
    if (not overlaps_destination and mapping.is_exhaustive()
        and is_reverse_c_contiguous(mapping)):
        # Physically dense, logically reverse-C destination. The source and
        # destination are proven disjoint by their actual host byte ranges.
        var read_pointer = source.unsafe_ptr()
        var write_pointer = destination.data.unsafe_ptr()
        var last = mapping.span_hi
        for i in range(count):
            write_pointer.unsafe_store(last - i, read_pointer.unsafe_load(i)[0])
        return TensorStatus.OK
    if overlaps_destination:
        # Arbitrary destination strides can overwrite input elements before
        # their logical turn. Keep the complete input in a private owned
        # allocation until all scatter writes have completed.
        var pending = HostStorage._allocate_for_complete_overwrite(count)
        if not pending.is_ok():
            return pending.status
        var staging = pending.payload.take()
        var scratch = staging.borrow_mut_span()
        for i in range(count):
            scratch[i] = source[i]
        var write_cursor = NDIndexIterator[rank](mapping.shape)
        for i in range(count):
            destination.data[
                mapping.physical_offset_unchecked(write_cursor.coordinates)
            ] = scratch[i]
            write_cursor.advance()
        return TensorStatus.OK
    var cursor = NDIndexIterator[rank](mapping.shape)
    for i in range(count):
        destination.data[
            mapping.physical_offset_unchecked(cursor.coordinates)
        ] = source[i]
        cursor.advance()
    return TensorStatus.OK
