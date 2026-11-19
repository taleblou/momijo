# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR VIEW / RESHAPING VIEWS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Views
# Module:        momijo.tensor_light.view.reshape
# Source:        src/momijo/tensor_light/view/reshape.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines reshaping views in the tensor views layer of the FP32 Light tensor
#   implementation.
#
#   Public declarations include `reshape_c_view`, `reshape_f_view`,
#   `flatten_view`, `ravel_view`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Build safe tensor views and layout-only transformations.
#   - Preserve documented behavior for shape compatibility, rank boundaries,
#     and empty dimensions.
#   - Provide source-declared interfaces such as `reshape_c_view`,
#     `reshape_f_view`, `flatten_view`, `ravel_view`, `ravel_f_view` and
#     related symbols.
#   - Handle scalar, empty, and rank-dependent edge cases explicitly.
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

# Zero-copy C/F reshape for any affine layout whose iteration-order chunks
# can be regrouped without changing element addresses. Never materializes.
# Rank is AOT-static (0..8); metadata uses fixed-capacity local arrays.
from ..shape.shape import TensorShape, flatten_shape
from ..shape.shape_storage import StaticRankArray
from ..layout.layout import TensorLayout
from ..layout.layout_analysis import _reshape_view_layout_checked
from ..internal.status import TensorResult


# All affine representability logic is owned by layout/layout_analysis.mojo.
# This view layer simply returns its validated result without materialization.
def reshape_c_view[src_rank: Int, dst_rank: Int](
    src: TensorLayout[src_rank], desired: TensorShape[dst_rank], capacity: Int
) -> TensorResult[TensorLayout[dst_rank]]:
    return _reshape_view_layout_checked[src_rank, dst_rank, True](src, desired, capacity)


def reshape_f_view[src_rank: Int, dst_rank: Int](
    src: TensorLayout[src_rank], desired: TensorShape[dst_rank], capacity: Int
) -> TensorResult[TensorLayout[dst_rank]]:
    return _reshape_view_layout_checked[src_rank, dst_rank, False](src, desired, capacity)


# Layout-level public view adapters. These return only checked metadata;
# the caller retains ownership of the original storage and its borrow origin.
# No allocation, fallback copy or implicit device transfer is possible.
def flatten_view[src_rank: Int, start_dim: Int, end_dim: Int](
    src: TensorLayout[src_rank], capacity: Int,
) -> TensorResult[TensorLayout[src_rank - (end_dim - start_dim)]]:
    var flattened = flatten_shape[src_rank, start_dim, end_dim](src.shape)
    if not flattened.is_ok():
        return TensorResult[TensorLayout[src_rank - (end_dim - start_dim)]](
            flattened.status
        )
    return reshape_c_view[src_rank, src_rank - (end_dim - start_dim)](
        src, flattened.payload.value(), capacity,
    )


def ravel_view[src_rank: Int](
    src: TensorLayout[src_rank], capacity: Int,
) -> TensorResult[TensorLayout[1]]:
    # Use the validated source shape to reject forged cached numel before
    # requesting rank-one linearization of a borrowed storage mapping.
    var count = src.shape.checked_numel()
    if not count.is_ok():
        return TensorResult[TensorLayout[1]](count.status)
    var target = TensorShape[1].checked(
        StaticRankArray[1](count.payload.value())
    )
    if not target.is_ok():
        return TensorResult[TensorLayout[1]](target.status)
    return reshape_c_view[src_rank, 1](
        src, target.payload.value(), capacity,
    )


def ravel_f_view[src_rank: Int](
    src: TensorLayout[src_rank], capacity: Int,
) -> TensorResult[TensorLayout[1]]:
    var count = src.shape.checked_numel()
    if not count.is_ok():
        return TensorResult[TensorLayout[1]](count.status)
    var target = TensorShape[1].checked(
        StaticRankArray[1](count.payload.value())
    )
    if not target.is_ok():
        return TensorResult[TensorLayout[1]](target.status)
    return reshape_f_view[src_rank, 1](
        src, target.payload.value(), capacity,
    )


# Canonical zero-copy API name; failed representability is returned as a
# status by the same exact C-layout analysis, never materialized silently.
def reshape_view[src_rank: Int, dst_rank: Int](
    source: TensorLayout[src_rank], target: TensorShape[dst_rank], capacity: Int,
) -> TensorResult[TensorLayout[dst_rank]]:
    return reshape_c_view[src_rank, dst_rank](source, target, capacity)
