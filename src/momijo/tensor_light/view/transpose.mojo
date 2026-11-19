# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR VIEW / TRANSPOSITION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Views
# Module:        momijo.tensor_light.view.transpose
# Source:        src/momijo/tensor_light/view/transpose.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor transposition in the tensor views layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `swapaxes`, `movedim`, `transpose`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Build safe tensor views and layout-only transformations.
#   - Preserve documented behavior for bounds, strides, view provenance, and
#     write safety.
#   - Provide source-declared interfaces such as `swapaxes`, `movedim`,
#     `transpose`.
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

# Metadata-only transpose and axis movement.
from ..layout.layout import TensorLayout
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus

def swapaxes[rank: Int](base: TensorLayout[rank], a: Int, b: Int, capacity: Int) -> TensorResult[TensorLayout[rank]]:
    return base.transpose(a, b, capacity)

def movedim[rank: Int](base: TensorLayout[rank], source: Int, destination: Int, capacity: Int) -> TensorResult[TensorLayout[rank]]:
    var a = source
    var b = destination
    if a < 0:
        if a < -rank:
            return TensorResult[TensorLayout[rank]](TensorStatus.INVALID_AXIS)
        a += rank
    if b < 0:
        if b < -rank:
            return TensorResult[TensorLayout[rank]](TensorStatus.INVALID_AXIS)
        b += rank
    if a >= rank or b >= rank:
        return TensorResult[TensorLayout[rank]](TensorStatus.INVALID_AXIS)
    var order = StaticRankArray[rank](0)
    comptime for i in range(rank):
        order.set(i, i)
    var shifted = order.get(a)
    if a < b:
        for i in range(a, b):
            order.set(i, order.get(i + 1))
    elif a > b:
        for i in range(a, b, -1):
            order.set(i, order.get(i - 1))
    order.set(b, shifted)
    return base.permute(order, capacity)


def transpose[rank: Int](
    source: TensorLayout[rank], first: Int, second: Int, capacity: Int,
) -> TensorResult[TensorLayout[rank]]:
    return source.transpose(first, second, capacity)
