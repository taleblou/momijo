# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR SHAPE / AXIS NORMALIZATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Shapes
# Module:        momijo.tensor_light.shape.axis
# Source:        src/momijo/tensor_light/shape/axis.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines axis normalization in the tensor shapes layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `normalize_axis`, `validate_axis`,
#   `normalize_axis_pair`, `validate_axes_unique`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Normalize signed axis indices using the compile-time tensor rank.
#   - Validate axis pairs and unique sets of axes.
#   - Expose selection checks for fixed-rank axis metadata.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Enforce the source-defined supported rank range of 0 through 8.
#   - Reject out-of-range and duplicate axes when uniqueness is required.
#   - Preserve documented behavior for equal axis pairs and rank-zero cases.
#
# Performance Objectives:
#   - Use compile-time rank specialization and fixed-size axis storage.
#   - Avoid heap allocation and floating-point work in axis checks.
#
# Maintainability:
#   - Keep the axis normalization contract consistent across views and
#     reductions.
#   - Test negative axes, duplicate sets, equal pairs, and zero/high-rank
#     boundaries.
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

# Runtime axes are normalized once at a checked entry point.
from ..internal.status import TensorResult, TensorStatus
from .shape_storage import StaticRankArray

def normalize_axis[rank: Int](axis: Int) -> TensorResult[Int]:
    comptime assert rank >= 0 and rank <= 8, "invalid tensor rank"
    if axis < 0:
        if axis < -rank:
            return TensorResult[Int](TensorStatus.INVALID_AXIS)
        return TensorResult[Int](axis + rank)
    if axis >= rank:
        return TensorResult[Int](TensorStatus.INVALID_AXIS)
    return TensorResult[Int](axis)


def validate_axis[rank: Int](axis: Int) -> TensorStatus:
    var checked = normalize_axis[rank](axis)
    return checked.status


def normalize_axis_pair[rank: Int](a: Int, b: Int) -> TensorResult[StaticRankArray[2]]:
    # Equal normalized axes are permitted (e.g. a no-op transpose).
    var first = normalize_axis[rank](a)
    if not first.is_ok():
        return TensorResult[StaticRankArray[2]](first.status)
    var second = normalize_axis[rank](b)
    if not second.is_ok():
        return TensorResult[StaticRankArray[2]](second.status)
    var pair = StaticRankArray[2](0)
    pair.set(0, first.payload.value())
    pair.set(1, second.payload.value())
    return TensorResult[StaticRankArray[2]](pair)


def validate_axes_unique[rank: Int, count: Int](axes: Array[Int, count]) -> TensorStatus:
    comptime assert rank >= 0 and rank <= 8 and count >= 0 and count <= 8, "axis bounds"
    if count > rank:
        return TensorStatus.INVALID_AXIS
    var seen = Array[Bool, 8](fill=False)
    for i in range(count):
        var normalized = normalize_axis[rank](axes[i])
        if not normalized.is_ok():
            return normalized.status
        var at = normalized.payload.value()
        if seen[at]:
            return TensorStatus.INVALID_AXIS
        seen[at] = True
    return TensorStatus.OK


def axis_is_selected[rank: Int, count: Int](axis: Int, axes: Array[Int, count]) -> TensorResult[Bool]:
    var axis_result = normalize_axis[rank](axis)
    if not axis_result.is_ok():
        return TensorResult[Bool](axis_result.status)
    var valid = validate_axes_unique[rank, count](axes)
    if valid != TensorStatus.OK:
        return TensorResult[Bool](valid)
    for i in range(count):
        var candidate = normalize_axis[rank](axes[i])
        if candidate.payload.value() == axis_result.payload.value():
            return TensorResult[Bool](True)
    return TensorResult[Bool](False)
