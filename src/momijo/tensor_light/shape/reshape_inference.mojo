# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR SHAPE / RESHAPE INFERENCE
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Shapes
# Module:        momijo.tensor_light.shape.reshape_inference
# Source:        src/momijo/tensor_light/shape/reshape_inference.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines reshape inference in the tensor shapes layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `infer_reshape_shape`, `validate_same_numel`,
#   `find_inferred_dimension`, `resolve_inferred_extent`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Construct, normalize, or validate tensor shape metadata.
#   - Preserve documented behavior for shape compatibility, rank boundaries,
#     and empty dimensions.
#   - Provide source-declared interfaces such as `infer_reshape_shape`,
#     `validate_same_numel`, `find_inferred_dimension`,
#     `resolve_inferred_extent`.
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

# A single -1 dimension may be inferred; ambiguous zero-product inference is rejected.
from .shape import TensorShape
from .shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus
from ..internal.checked_math import checked_mul_nonnegative

def infer_reshape_shape[rank: Int](source_numel: Int, proposed: StaticRankArray[rank]) -> TensorResult[TensorShape[rank]]:
    comptime assert rank >= 0 and rank <= 8, "invalid target rank"
    if source_numel < 0 or source_numel > Int.MAX // 4:
        return TensorResult[TensorShape[rank]](TensorStatus.INVALID_SHAPE)
    # Scan all dimensions first: a later zero may make the entire requested
    # shape empty, even when preceding nonzero factors would overflow Int.
    var inferred_axis = -1
    var any_zero = False
    comptime for i in range(rank):
        var d = proposed.get(i)
        if d == -1:
            if inferred_axis != -1:
                return TensorResult[TensorShape[rank]](TensorStatus.INVALID_SHAPE)
            inferred_axis = i
        elif d < 0:
            return TensorResult[TensorShape[rank]](TensorStatus.INVALID_SHAPE)
        elif d == 0:
            any_zero = True
    var dims = proposed.copy()
    if any_zero:
        if inferred_axis >= 0:
            # -1 and any zero-size factor have infinitely many compatible
            # inferred extents: reject rather than silently choosing one.
            return TensorResult[TensorShape[rank]](TensorStatus.INVALID_SHAPE)
        var empty_shape = TensorShape[rank].checked(dims)
        if not empty_shape.is_ok():
            return empty_shape
        if source_numel != 0:
            return TensorResult[TensorShape[rank]](TensorStatus.SIZE_MISMATCH)
        return empty_shape
    if inferred_axis >= 0 and source_numel == 0:
        # There are no explicit zero factors and exactly one inferred axis.
        # Its only possible value is zero. Avoid multiplying potentially huge
        # known factors: an EMPTY shape is valid even if that intermediate
        # product cannot fit in Int. This is not the ambiguous case where
        # another explicitly requested extent is already zero.
        dims.set(inferred_axis, 0)
        return TensorShape[rank].checked(dims)
    var known = 1
    comptime for i in range(rank):
        var d = proposed.get(i)
        if d >= 0:
            var product = checked_mul_nonnegative(known, d)
            if not product.is_ok():
                return TensorResult[TensorShape[rank]](product.status)
            known = product.payload.value()
    if inferred_axis >= 0:
        if known == 0:
            return TensorResult[TensorShape[rank]](TensorStatus.INVALID_SHAPE)
        if source_numel % known != 0:
            return TensorResult[TensorShape[rank]](TensorStatus.SIZE_MISMATCH)
        dims.set(inferred_axis, source_numel // known)
    var final_shape = TensorShape[rank].checked(dims)
    if not final_shape.is_ok():
        return final_shape
    if final_shape.payload.value().numel() != source_numel:
        return TensorResult[TensorShape[rank]](TensorStatus.SIZE_MISMATCH)
    return final_shape


def validate_same_numel[source_rank: Int, target_rank: Int](
    source: TensorShape[source_rank], target: TensorShape[target_rank]
) -> TensorStatus:
    var a = TensorShape[source_rank].checked(source.dims)
    if not a.is_ok():
        return a.status
    var b = TensorShape[target_rank].checked(target.dims)
    if not b.is_ok():
        return b.status
    if a.payload.value().numel() != source.numel() or b.payload.value().numel() != target.numel():
        return TensorStatus.INVALID_SHAPE
    if a.payload.value().numel() != b.payload.value().numel():
        return TensorStatus.SIZE_MISMATCH
    return TensorStatus.OK

# The following two checked helpers share the same policy as infer_reshape_shape.
def find_inferred_dimension[rank: Int](proposed: StaticRankArray[rank]) -> TensorResult[Int]:
    comptime assert rank >= 0 and rank <= 8, "invalid target rank"
    var found = -1
    comptime for i in range(rank):
        var dim = proposed.get(i)
        if dim == -1:
            if found >= 0:
                return TensorResult[Int](TensorStatus.INVALID_SHAPE)
            found = i
        elif dim < 0:
            return TensorResult[Int](TensorStatus.INVALID_SHAPE)
    return TensorResult[Int](found)


def resolve_inferred_extent(source_numel: Int, known_product: Int) -> TensorResult[Int]:
    if source_numel < 0 or source_numel > Int.MAX // 4 or known_product < 0:
        return TensorResult[Int](TensorStatus.INVALID_SHAPE)
    if known_product == 0:
        return TensorResult[Int](TensorStatus.INVALID_SHAPE)
    if source_numel % known_product != 0:
        return TensorResult[Int](TensorStatus.SIZE_MISMATCH)
    return TensorResult[Int](source_numel // known_product)
