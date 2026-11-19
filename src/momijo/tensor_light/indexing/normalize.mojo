# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR INDEXING / INDEX NORMALIZATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Indexing
# Module:        momijo.tensor_light.indexing.normalize
# Source:        src/momijo/tensor_light/indexing/normalize.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines index normalization in the tensor indexing layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `normalize_integer_indices`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Normalize a fixed-rank collection of integer indices against TensorShape
#     extents.
#   - Validate shape metadata and cached element count before using
#     dimensions.
#   - Return normalized indices or an explicit error status.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Reject inconsistent shape metadata and out-of-range integer indices.
#   - Preserve the source-defined fixed-rank and negative-index contracts.
#
# Performance Objectives:
#   - Use compile-time rank specialization for element-independent index
#     checking.
#   - Avoid tensor storage allocation, copies, and per-element dispatch.
#
# Maintainability:
#   - Keep integer-index normalization consistent with shape and index
#     modules.
#   - Test rank boundaries, negative indices, invalid dimensions, and
#     cached-count mismatches.
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

# Checked fixed-rank all-integer indexing. Heterogeneous packs are separate.
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus
from .index import normalize_index

def normalize_integer_indices[rank: Int](shape: TensorShape[rank], raw: StaticRankArray[rank]) -> TensorResult[StaticRankArray[rank]]:
    # This function is also an expert-low-level public entry point; validate
    # the cached count once before using its dimensions for index bounds.
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok():
        return TensorResult[StaticRankArray[rank]](canonical.status)
    if canonical.payload.value().numel() != shape.numel():
        return TensorResult[StaticRankArray[rank]](TensorStatus.INVALID_SHAPE)
    var out = StaticRankArray[rank](0)
    comptime for i in range(rank):
        var result = normalize_index(raw.get(i), shape.dims.get(i))
        if not result.is_ok():
            return TensorResult[StaticRankArray[rank]](result.status)
        out.set(i, result.payload.value())
    return TensorResult[StaticRankArray[rank]](out)
