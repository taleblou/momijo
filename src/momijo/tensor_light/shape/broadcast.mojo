# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR SHAPE / BROADCASTING
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Shapes
# Module:        momijo.tensor_light.shape.broadcast
# Source:        src/momijo/tensor_light/shape/broadcast.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines broadcasting in the tensor shapes layer of the FP32 Light tensor
#   implementation.
#
#   Public declarations include `broadcast_shape`, `can_broadcast_shapes`,
#   `is_broadcast_axis`, `broadcast_many_shapes`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Construct, normalize, or validate tensor shape metadata.
#   - Preserve documented behavior for shape compatibility, rank boundaries,
#     and empty dimensions.
#   - Provide source-declared interfaces such as `broadcast_shape`,
#     `can_broadcast_shapes`, `is_broadcast_axis`, `broadcast_many_shapes`.
#   - Handle scalar, empty, and rank-dependent edge cases explicitly.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate extents, axes, rank, and element-count overflow.
#   - Preserve rank-zero and zero-sized-shape semantics.
#   - Handle scalar, rank-zero, and zero-length dimensions without silent
#     mismatch.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Keep shape arithmetic in metadata-only code.
#   - Avoid allocating shape arrays unnecessarily in hot paths.
#   - Reuse validated output-shape metadata before data traversal.
#
# Maintainability:
#   - Share one broadcasting and axis normalization policy.
#   - Test invalid ranks, mismatched shapes, and inferred dimensions.
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

# NumPy-compatible right-aligned shape broadcast for static operand ranks.
from .shape import TensorShape
from .shape_storage import StaticRankArray
from ..internal.status import TensorStatus, TensorResult

def broadcast_shape[a_rank: Int, b_rank: Int, out_rank: Int](a: TensorShape[a_rank], b: TensorShape[b_rank]) -> TensorResult[TensorShape[out_rank]]:
    comptime assert out_rank <= 8 and out_rank >= a_rank and out_rank >= b_rank, "invalid output broadcast rank"
    # A mutable TensorShape may carry stale cached numel. Never treat such
    # metadata as a valid operand when constructing a broadcast result.
    var checked_a = TensorShape[a_rank].checked(a.dims)
    if not checked_a.is_ok():
        return TensorResult[TensorShape[out_rank]](checked_a.status)
    var checked_b = TensorShape[b_rank].checked(b.dims)
    if not checked_b.is_ok():
        return TensorResult[TensorShape[out_rank]](checked_b.status)
    if checked_a.payload.value().numel() != a.numel() or checked_b.payload.value().numel() != b.numel():
        return TensorResult[TensorShape[out_rank]](TensorStatus.INVALID_SHAPE)
    var dims = StaticRankArray[out_rank](1)
    comptime for i in range(out_rank):
        # These axis positions determine which source code can be instantiated.
        # They must be compile-time values, not mutable runtime locals.
        comptime ai = i - (out_rank - a_rank)
        comptime bi = i - (out_rank - b_rank)
        var da = 1
        var db = 1
        comptime if ai >= 0:
            da = a.dims.get(ai)
        comptime if bi >= 0:
            db = b.dims.get(bi)
        if da == db:
            dims.set(i, da)
        elif da == 1:
            dims.set(i, db)
        elif db == 1:
            dims.set(i, da)
        else:
            return TensorResult[TensorShape[out_rank]](TensorStatus.SIZE_MISMATCH)
    return TensorShape[out_rank].checked(dims)


def can_broadcast_shapes[a_rank: Int, b_rank: Int, out_rank: Int](
    a: TensorShape[a_rank], b: TensorShape[b_rank]
) -> Bool:
    var output = broadcast_shape[a_rank, b_rank, out_rank](a, b)
    return output.is_ok()


def is_broadcast_axis[source_rank: Int, out_rank: Int](
    source: TensorShape[source_rank], target: TensorShape[out_rank], axis: Int
) -> TensorResult[Bool]:
    comptime assert out_rank >= source_rank and out_rank <= 8, "invalid broadcast rank"
    var a = axis
    if a < 0:
        if a < -out_rank:
            return TensorResult[Bool](TensorStatus.INVALID_AXIS)
        a += out_rank
    if a >= out_rank:
        return TensorResult[Bool](TensorStatus.INVALID_AXIS)
    var output = broadcast_shape[source_rank, out_rank, out_rank](source, target)
    if not output.is_ok():
        return TensorResult[Bool](output.status)
    if not output.payload.value().same_extents(target):
        return TensorResult[Bool](TensorStatus.SIZE_MISMATCH)
    var old_axis = a - (out_rank - source_rank)
    if old_axis < 0:
        return TensorResult[Bool](target.dims.get(a) != 1)
    return TensorResult[Bool](source.dims.get(old_axis) == 1 and target.dims.get(a) != 1)

# Fixed-count broadcasting is resolved before the elementwise inner loop.
def broadcast_many_shapes[rank: Int, count: Int](shapes: Array[TensorShape[rank], count]) -> TensorResult[TensorShape[rank]]:
    comptime assert rank >= 0 and rank <= 8 and count >= 1, "invalid shape collection"
    var accumulated = TensorShape[rank].checked(shapes[0].dims)
    if not accumulated.is_ok():
        return accumulated^
    if accumulated.payload.value().numel() != shapes[0].numel():
        return TensorResult[TensorShape[rank]](TensorStatus.INVALID_SHAPE)
    for i in range(1, count):
        var next = broadcast_shape[rank, rank, rank](accumulated.payload.value(), shapes[i])
        if not next.is_ok():
            return next^
        accumulated = next^
    return accumulated^
