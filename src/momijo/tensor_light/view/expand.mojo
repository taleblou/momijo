# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR VIEW / BROADCAST VIEWS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Views
# Module:        momijo.tensor_light.view.expand
# Source:        src/momijo/tensor_light/view/expand.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines broadcast views in the tensor views layer of the FP32 Light tensor
#   implementation.
#
#   Public declarations include `broadcast_layout`, `squeeze_axes`, `expand`,
#   `broadcast_to`; preconditions and supported targets are determined by
#   their implementations.
#
# Responsibilities:
#   - Build safe tensor views and layout-only transformations.
#   - Preserve documented behavior for shape compatibility, rank boundaries,
#     and empty dimensions.
#   - Provide source-declared interfaces such as `broadcast_layout`,
#     `squeeze_axes`, `expand`, `broadcast_to`.
#   - Handle scalar, empty, and rank-dependent edge cases explicitly.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate shape/stride/span invariants for borrowed views.
#   - Enforce write restrictions for broadcast or overlapping mappings.
#   - Handle scalar, rank-zero, and zero-length dimensions without silent
#     mismatch.
#
# Performance Objectives:
#   - Prefer zero-copy metadata transforms when valid.
#   - Avoid view materialization unless the API explicitly requires it.
#   - Reuse validated output-shape metadata before data traversal.
#
# Maintainability:
#   - Keep view semantics consistent with owning-tensor operations.
#   - Test permutations, negative strides, and zero-length axes.
#   - Exercise incompatible dimensions and rank-limit regressions.
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

# Zero-stride read-only broadcast view metadata; writes require uniqueness proof.
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..layout.layout import TensorLayout, expand_layout
from ..internal.status import TensorStatus, TensorResult

def broadcast_layout[src_rank: Int, dst_rank: Int](
    source: TensorLayout[src_rank], target: TensorShape[dst_rank], capacity: Int,
) -> TensorResult[TensorLayout[dst_rank]] where src_rank >= 0 and dst_rank >= src_rank and dst_rank <= 8:
    # One authoritative validation, overflow and zero-stride broadcast policy.
    return expand_layout[src_rank, dst_rank](source, target, capacity)


# A compile-time count of explicitly requested singleton axes is removed
# from the metadata in one pass. The axis VALUES may be runtime Ints, but
# output rank remains statically rank-count. Negative axes are permitted;
# repeated or non-singleton axes are rejected before constructing a view.
# An empty axis pack is a validated metadata identity, including for rank 0.
def squeeze_axes[rank: Int, count: Int](
    source: TensorLayout[rank], axes: StaticRankArray[count], capacity: Int,
) -> TensorResult[TensorLayout[rank - count]] where rank >= 0 and rank <= 8 and count >= 0 and count <= rank:
    var proof = source.validated_snapshot(capacity)
    if not proof.is_ok():
        return TensorResult[TensorLayout[rank - count]](proof.status)
    var trusted = proof.payload.value()
    var removed = Array[Bool, 8](fill=False)
    for i in range(count):
        var axis = axes.get(i)
        if axis < 0:
            if axis < -rank:
                return TensorResult[TensorLayout[rank - count]](TensorStatus.INVALID_AXIS)
            axis += rank
        if axis >= rank:
            return TensorResult[TensorLayout[rank - count]](TensorStatus.INVALID_AXIS)
        if removed[axis]:
            return TensorResult[TensorLayout[rank - count]](TensorStatus.INVALID_AXIS)
        if trusted.shape.dims.get(axis) != 1:
            return TensorResult[TensorLayout[rank - count]](TensorStatus.SIZE_MISMATCH)
        removed[axis] = True
    var dims = StaticRankArray[rank - count](0)
    var strides = StaticRankArray[rank - count](0)
    var output_axis = 0
    comptime for axis in range(rank):
        if not removed[axis]:
            dims.set(output_axis, trusted.shape.dims.get(axis))
            strides.set(output_axis, trusted.strides.get(axis))
            output_axis += 1
    var new_shape = TensorShape[rank - count].checked(dims)
    if not new_shape.is_ok():
        return TensorResult[TensorLayout[rank - count]](new_shape.status)
    return TensorLayout[rank - count].checked(
        new_shape.payload.value(), strides, trusted.offset, capacity,
    )


# The names below intentionally do not define a second algorithm.
def expand[src_rank: Int, dst_rank: Int](
    source: TensorLayout[src_rank], target: TensorShape[dst_rank], capacity: Int,
) -> TensorResult[TensorLayout[dst_rank]] where src_rank >= 0 and dst_rank >= src_rank and dst_rank <= 8:
    return broadcast_layout[src_rank, dst_rank](source, target, capacity)


def broadcast_to[src_rank: Int, dst_rank: Int](
    source: TensorLayout[src_rank], target: TensorShape[dst_rank], capacity: Int,
) -> TensorResult[TensorLayout[dst_rank]] where src_rank >= 0 and dst_rank >= src_rank and dst_rank <= 8:
    return broadcast_layout[src_rank, dst_rank](source, target, capacity)


# The documented view/expand module also exposes singleton rank transforms.
# Keep their implementation in layout/layout.mojo and preserve the existing
# view/unsqueeze module as the only thin delegate: no repeated stride rules.
from .unsqueeze import unsqueeze, squeeze_axis
