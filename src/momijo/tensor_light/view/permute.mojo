# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR VIEW / AXIS PERMUTATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Views
# Module:        momijo.tensor_light.view.permute
# Source:        src/momijo/tensor_light/view/permute.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines axis permutations in the tensor views layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `permute`, `validate_permutation`,
#   `invert_permutation`; preconditions and supported targets are determined
#   by their implementations.
#
# Responsibilities:
#   - Build safe tensor views and layout-only transformations.
#   - Preserve documented behavior for shape compatibility, rank boundaries,
#     and empty dimensions.
#   - Provide source-declared interfaces such as `permute`,
#     `validate_permutation`, `invert_permutation`.
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

# Checked rank-preserving metadata permutation.
from ..layout.layout import TensorLayout
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus

def permute[rank: Int](source: TensorLayout[rank], axes: StaticRankArray[rank], capacity: Int) -> TensorResult[TensorLayout[rank]]:
    return source.permute(axes, capacity)


# A valid permutation contains every source axis exactly once. Fixed-rank
# arrays ensure no heap work and permit compile-time unrolling in callers.
def validate_permutation[rank: Int](axes: StaticRankArray[rank]) -> TensorStatus:
    comptime assert rank >= 0 and rank <= 8, "permutation rank must be 0..8"
    var seen = Array[Bool, 8](fill=False)
    comptime for i in range(rank):
        var axis = axes.get(i)
        if axis < 0 or axis >= rank:
            return TensorStatus.INVALID_AXIS
        if seen[axis]:
            return TensorStatus.INVALID_AXIS
        seen[axis] = True
    return TensorStatus.OK


def invert_permutation[rank: Int](axes: StaticRankArray[rank]) -> TensorResult[StaticRankArray[rank]]:
    var valid = validate_permutation[rank](axes)
    if valid != TensorStatus.OK:
        return TensorResult[StaticRankArray[rank]](valid)
    var inverse = StaticRankArray[rank](0)
    comptime for i in range(rank):
        inverse.set(axes.get(i), i)
    return TensorResult[StaticRankArray[rank]](inverse)
