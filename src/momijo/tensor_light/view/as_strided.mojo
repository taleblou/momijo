# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR VIEW / STRIDED TENSOR VIEWS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Views
# Module:        momijo.tensor_light.view.as_strided
# Source:        src/momijo/tensor_light/view/as_strided.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines strided tensor views in the tensor views layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `as_strided`, `slice_axis`, `slice`, `narrow`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Build safe tensor views and layout-only transformations.
#   - Preserve documented behavior for bounds, strides, view provenance, and
#     write safety.
#   - Provide source-declared interfaces such as `as_strided`, `slice_axis`,
#     `slice`, `narrow`, `flip`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate shape/stride/span invariants for borrowed views.
#   - Enforce write restrictions for broadcast or overlapping mappings.
#
# Performance Objectives:
#   - Prefer zero-copy metadata transforms when valid.
#   - Avoid view materialization unless the API explicitly requires it.
#
# Maintainability:
#   - Keep view semantics consistent with owning-tensor operations.
#   - Test permutations, negative strides, and zero-length axes.
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

# Checked strided metadata views; never allocate, copy, or hide materialization.
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..layout.layout import TensorLayout
from ..internal.checked_math import checked_add, checked_mul
from ..internal.status import TensorStatus, TensorResult
from ..indexing.slice import SliceSpec, normalize_slice

def as_strided[old_rank: Int, new_rank: Int](base: TensorLayout[old_rank], shape: TensorShape[new_rank], strides: StaticRankArray[new_rank], offset: Int, capacity: Int) -> TensorResult[TensorLayout[new_rank]]:
    var checked_base = base.validated_snapshot(capacity)
    if not checked_base.is_ok():
        return TensorResult[TensorLayout[new_rank]](checked_base.status)
    return TensorLayout[new_rank].checked(shape, strides, offset, capacity)

def slice_axis[rank: Int](base: TensorLayout[rank], axis: Int, spec: SliceSpec, capacity: Int) -> TensorResult[TensorLayout[rank]]:
    var checked_base = base.validated_snapshot(capacity)
    if not checked_base.is_ok():
        return TensorResult[TensorLayout[rank]](checked_base.status)
    var trusted = checked_base.payload.value()
    var a = axis
    if a < 0:
        if a < -rank:
            return TensorResult[TensorLayout[rank]](TensorStatus.INVALID_AXIS)
        a += rank
    if a >= rank:
        return TensorResult[TensorLayout[rank]](TensorStatus.INVALID_AXIS)
    var normalized = normalize_slice(spec, trusted.shape.dims.get(a))
    if not normalized.is_ok():
        return TensorResult[TensorLayout[rank]](normalized.status)
    var selected = normalized.payload.value()
    var dims = trusted.shape.dims.copy()
    dims.set(a, selected.length)
    var new_shape = TensorShape[rank].checked(dims)
    if not new_shape.is_ok():
        return TensorResult[TensorLayout[rank]](new_shape.status)
    var next_strides = trusted.strides.copy()
    if selected.length <= 1 or new_shape.payload.value().is_empty():
        # No pair of reachable indices differs on this axis. Preserve only
        # semantic addresses and avoid multiplication of an irrelevant stride.
        next_strides.set(a, 0)
    else:
        var next_step = checked_mul(next_strides.get(a), selected.step)
        if not next_step.is_ok():
            return TensorResult[TensorLayout[rank]](next_step.status)
        next_strides.set(a, next_step.payload.value())
    # The RESULT may be empty because some *other* axis has extent zero.
    # No element can be addressed in that case, so keep a valid origin rather
    # than shifting an empty layout (which may overflow for huge strides).
    var offset = trusted.offset
    if not new_shape.payload.value().is_empty():
        var delta = checked_mul(trusted.strides.get(a), selected.start)
        if not delta.is_ok():
            return TensorResult[TensorLayout[rank]](delta.status)
        var shifted = checked_add(trusted.offset, delta.payload.value())
        if not shifted.is_ok():
            return TensorResult[TensorLayout[rank]](shifted.status)
        offset = shifted.payload.value()
    return TensorLayout[rank].checked(new_shape.payload.value(), next_strides, offset, capacity)


# Public vocabulary: all three operations are strictly metadata-only.
# `as_strided` remains checked; there is no unsafe bypass of owner bounds.
def slice[rank: Int](
    base: TensorLayout[rank], axis: Int, spec: SliceSpec, capacity: Int,
) -> TensorResult[TensorLayout[rank]]:
    return slice_axis[rank](base, axis, spec, capacity)


def narrow[rank: Int](
    base: TensorLayout[rank], axis: Int, start: Int, length: Int,
    capacity: Int,
) -> TensorResult[TensorLayout[rank]]:
    var checked_base = base.validated_snapshot(capacity)
    if not checked_base.is_ok():
        return TensorResult[TensorLayout[rank]](checked_base.status)
    var a = axis
    if a < 0:
        if a < -rank:
            return TensorResult[TensorLayout[rank]](TensorStatus.INVALID_AXIS)
        a += rank
    if a >= rank:
        return TensorResult[TensorLayout[rank]](TensorStatus.INVALID_AXIS)
    var extent = checked_base.payload.value().shape.dims.get(a)
    if start < 0 or length < 0 or start > extent or length > extent - start:
        return TensorResult[TensorLayout[rank]](TensorStatus.OUT_OF_BOUNDS)
    # The subtraction guard above proves that start+length cannot overflow.
    return slice_axis[rank](
        checked_base.payload.value(), a,
        SliceSpec(start, start + length, 1), capacity,
    )


def flip[rank: Int](
    base: TensorLayout[rank], axis: Int, capacity: Int,
) -> TensorResult[TensorLayout[rank]]:
    return base.flip(axis, capacity)
