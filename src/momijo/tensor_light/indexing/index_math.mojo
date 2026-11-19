# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR INDEXING / INDEX ARITHMETIC
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Indexing
# Module:        momijo.tensor_light.indexing.index_math
# Source:        src/momijo/tensor_light/indexing/index_math.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines index arithmetic in the tensor indexing layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `basic_index_scalar_offset`,
#   `basic_index_layout`; preconditions and supported targets are determined
#   by their implementations.
#
# Responsibilities:
#   - Normalize and validate index or slicing operations.
#   - Preserve documented behavior for bounds, strides, view provenance, and
#     write safety.
#   - Provide source-declared interfaces such as `basic_index_scalar_offset`,
#     `basic_index_layout`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Normalize and bounds-check indices before memory access.
#   - Preserve documented negative-index and slicing semantics.
#   - Check index bounds before converting indices into physical offsets.
#
# Performance Objectives:
#   - Perform checked offset computation outside repeated element access.
#   - Avoid unnecessary index-tensor copies.
#   - Reuse normalized index ranges when the operation permits.
#
# Maintainability:
#   - Centralize slicing and index normalization.
#   - Test empty, negative-step, out-of-range, and high-rank cases.
#   - Test invalid indices, zero extents, and boundary conditions.
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

# Checked mixed basic indexing: integers, slices, newaxis, and ellipsis.
# All index descriptors are supplied in a fixed Array (no heap pack).
# AOT output rank is inferred for typed terms, or explicitly supplied for runtime-tagged BasicIndex.
from ..layout.layout import TensorLayout
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus
from ..internal.checked_math import checked_add, checked_mul
from .index import normalize_index
from .slice import normalize_slice
from .slice_spec import BasicIndex, BasicIndexKind, _summarize_basic_index_pack


def basic_index_scalar_offset[rank: Int](layout: TensorLayout[rank], indices: StaticRankArray[rank]) -> TensorResult[Int]:
    return layout.physical_offset(indices)


def basic_index_layout[src_rank: Int, out_rank: Int, count: Int](
    source: TensorLayout[src_rank],
    terms: Array[BasicIndex, count],
    storage_capacity: Int,
) -> TensorResult[TensorLayout[out_rank]]:
    comptime assert src_rank >= 0 and src_rank <= 8, "source rank must be 0..8"
    comptime assert out_rank >= 0 and out_rank <= 8, "output rank must be 0..8"
    comptime assert count >= 1 and count <= 17, "basic index pack must have 1..17 terms"

    var checked_source = source.validated_snapshot(storage_capacity)
    if not checked_source.is_ok():
        return TensorResult[TensorLayout[out_rank]](checked_source.status)
    var trusted = checked_source.payload.value()

    var inspected_terms = _summarize_basic_index_pack(terms)
    if not inspected_terms.is_ok():
        return TensorResult[TensorLayout[out_rank]](inspected_terms.status)
    var summary = inspected_terms.payload.value()
    var consumed = summary.consumed_source_axes
    var produced = summary.produced_output_axes
    if consumed > src_rank:
        return TensorResult[TensorLayout[out_rank]](TensorStatus.INVALID_AXIS)
    var implicit_slices = src_rank - consumed
    if produced + implicit_slices != out_rank:
        return TensorResult[TensorLayout[out_rank]](TensorStatus.INVALID_RANK)

    var output_dims = StaticRankArray[out_rank](0)
    var output_strides = StaticRankArray[out_rank](0)
    # Defer a stride overflow until the COMPLETE indexed result is known.
    # A later zero-length slice may make an otherwise oversized stride
    # unreachable, so early rejection would incorrectly reject a safe view.
    var pending_stride_overflow = False
    # A preceding valid integer/slice can overflow an intermediate shifted
    # origin even when a LATER slice makes the output empty. Record the first
    # offset failure, then resolve it only after the complete output shape is
    # known. No memory is accessed during this metadata-only construction.
    var pending_offset_error = TensorStatus.OK
    var read_axis = 0
    var write_axis = 0
    var final_offset = trusted.offset
    var saw_ellipsis = False

    comptime for i in range(count):
        ref term = terms[i]
        if term.kind == BasicIndexKind.ELLIPSIS:
            saw_ellipsis = True
            for j in range(implicit_slices):
                output_dims.set(write_axis, trusted.shape.dims.get(read_axis))
                output_strides.set(write_axis, trusted.strides.get(read_axis))
                write_axis += 1
                read_axis += 1
        elif term.kind == BasicIndexKind.NEW_AXIS:
            output_dims.set(write_axis, 1)
            output_strides.set(write_axis, 0)
            write_axis += 1
        elif term.kind == BasicIndexKind.INTEGER:
            var idx = normalize_index(term.integer_value, trusted.shape.dims.get(read_axis))
            if not idx.is_ok():
                return TensorResult[TensorLayout[out_rank]](idx.status)
            # Validate the integer axis even if another axis is empty.
            # An empty mapping has NO reachable element; its legal strides
            # may be arbitrarily large. Avoid overflowing a pointless
            # physical-origin shift; the canonical empty result keeps the
            # validated source origin below.
            if not trusted.shape.is_empty() and pending_offset_error == TensorStatus.OK:
                var delta = checked_mul(idx.payload.value(), trusted.strides.get(read_axis))
                if not delta.is_ok():
                    pending_offset_error = delta.status
                else:
                    var shifted = checked_add(final_offset, delta.payload.value())
                    if not shifted.is_ok():
                        pending_offset_error = shifted.status
                    else:
                        final_offset = shifted.payload.value()
            read_axis += 1
        else:
            var selected = normalize_slice(term.slice_value, trusted.shape.dims.get(read_axis))
            if not selected.is_ok():
                return TensorResult[TensorLayout[out_rank]](selected.status)
            var sliced = selected.payload.value()
            output_dims.set(write_axis, sliced.length)
            # A zero- or singleton-length axis never varies its offset. A
            # zero stride is canonical and avoids a spurious Int overflow for
            # an unobservable oversized slice step.
            if sliced.length <= 1:
                output_strides.set(write_axis, 0)
            else:
                var stride_result = checked_mul(trusted.strides.get(read_axis), sliced.step)
                if not stride_result.is_ok():
                    pending_stride_overflow = True
                    output_strides.set(write_axis, 0)
                else:
                    output_strides.set(write_axis, stride_result.payload.value())
            if sliced.length != 0 and not trusted.shape.is_empty() and pending_offset_error == TensorStatus.OK:
                var delta = checked_mul(sliced.start, trusted.strides.get(read_axis))
                if not delta.is_ok():
                    pending_offset_error = delta.status
                else:
                    var shifted = checked_add(final_offset, delta.payload.value())
                    if not shifted.is_ok():
                        pending_offset_error = shifted.status
                    else:
                        final_offset = shifted.payload.value()
            read_axis += 1
            write_axis += 1

    if not saw_ellipsis:
        for i in range(implicit_slices):
            output_dims.set(write_axis, trusted.shape.dims.get(read_axis))
            output_strides.set(write_axis, trusted.strides.get(read_axis))
            write_axis += 1
            read_axis += 1

    if read_axis != src_rank or write_axis != out_rank:
        return TensorResult[TensorLayout[out_rank]](TensorStatus.INVALID_RANK)
    var result_shape = TensorShape[out_rank].checked(output_dims)
    if not result_shape.is_ok():
        return TensorResult[TensorLayout[out_rank]](result_shape.status)
    if result_shape.payload.value().is_empty():
        # All offset and stride arithmetic is unobservable on an empty view.
        # This also covers earlier indices/slices followed by a zero slice.
        final_offset = trusted.offset
    else:
        if pending_stride_overflow:
            return TensorResult[TensorLayout[out_rank]](TensorStatus.OVERFLOW)
        if pending_offset_error != TensorStatus.OK:
            return TensorResult[TensorLayout[out_rank]](pending_offset_error)
    return TensorLayout[out_rank].checked(result_shape.payload.value(), output_strides, final_offset, storage_capacity)
