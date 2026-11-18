# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR INTERNAL / VALIDATION AND RESULTS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Internal Validation Utilities
# Module:        momijo.tensor_light.internal.validation
# Source:        src/momijo/tensor_light/internal/validation.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor validation and results in the internal validation utilities
#   layer of the FP32 Light tensor implementation.
#
#   Public declarations include `validate_storage_bounds`, `validate_rank`,
#   `validate_same_shape`, `validate_representability`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Provide reusable checked primitives and status contracts.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `validate_storage_bounds`,
#     `validate_rank`, `validate_same_shape`, `validate_representability`,
#     `validate_same_numel`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Detect integer overflow and invalid metadata before dereference.
#   - Return explicit statuses for unsupported or invalid states.
#
# Performance Objectives:
#   - Use simple inlinable checked primitives where appropriate.
#   - Keep validation work proportional to required metadata.
#
# Maintainability:
#   - Reuse one error vocabulary across dependent modules.
#   - Test boundary arithmetic and invalid-state propagation.
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

# Checked-boundary validation only; unchecked kernels trust the resulting plan.
from ..layout.layout import TensorLayout
from ..shape.shape import TensorShape
from .status import TensorStatus

def validate_storage_bounds[rank: Int](
    layout: TensorLayout[rank], capacity: Int
) -> TensorStatus:
    # Revalidate shape, stride, offset, physical allocation bound and cached
    # structural flags at the checked boundary, once outside the hot loop.
    var rebuilt = layout.validated_snapshot(capacity)
    return rebuilt.status

def validate_rank[rank: Int]() -> TensorStatus:
    # Static-rank admissibility: rank 0 is a scalar, 8 is the upper bound.
    comptime if rank < 0 or rank > 8:
        return TensorStatus.INVALID_RANK
    return TensorStatus.OK


def validate_same_shape[rank: Int](a: TensorShape[rank], b: TensorShape[rank]) -> TensorStatus:
    var ac = TensorShape[rank].checked(a.dims)
    if not ac.is_ok():
        return ac.status
    var bc = TensorShape[rank].checked(b.dims)
    if not bc.is_ok():
        return bc.status
    if ac.payload.value().numel() != a.numel() or bc.payload.value().numel() != b.numel():
        return TensorStatus.INVALID_SHAPE
    if not ac.payload.value().same_extents(bc.payload.value()):
        return TensorStatus.SHAPE_MISMATCH
    return TensorStatus.OK


def validate_representability(condition: Bool) -> TensorStatus:
    if condition:
        return TensorStatus.OK
    return TensorStatus.NOT_REPRESENTABLE

# Count-only compatibility is weaker than validate_same_shape(): reshaping
# may change rank and individual extents without changing logical element count.
# Each public TensorShape has mutable metadata, so independently revalidate
# both cached products before treating either as a trusted element count.
def validate_same_numel[left_rank: Int, right_rank: Int](
    left: TensorShape[left_rank], right: TensorShape[right_rank],
) -> TensorStatus:
    var left_count = left.checked_numel()
    if not left_count.is_ok():
        return left_count.status
    var right_count = right.checked_numel()
    if not right_count.is_ok():
        return right_count.status
    if left_count.payload.value() != right_count.payload.value():
        return TensorStatus.SIZE_MISMATCH
    return TensorStatus.OK
