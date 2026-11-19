# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / VIEW OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / View Operations
# Module:        momijo.tensor_light.ops.view
# Source:        src/momijo/tensor_light/ops/view.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines view transformations for axis changes, shape changes, and
#   broadcasting.
#
#   Public declarations include `transpose`, `reshape_view`, `permute`,
#   `swapaxes`; preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for bounds, strides, view provenance, and
#     write safety.
#   - Provide source-declared interfaces such as `transpose`, `reshape_view`,
#     `permute`, `swapaxes`, `movedim` and related symbols.
#   - Apply view transforms without copying elements when a view is valid.
#   - Represent broadcasting with shape/stride metadata.
#   - Validate reshape compatibility before producing a view.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate operand shape, dtype, and storage access.
#   - Preserve operation-specific broadcasting and result contracts.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid materializing broadcasts unless output storage needs it.
#   - Prefer specialized or fused paths only when semantics match.
#
# Maintainability:
#   - Keep public operations aligned with underlying kernel semantics.
#   - Cover edge cases and maintain relevant regression tests.
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

# Public view transforms delegate to Tensor's structural metadata layer.
from ..core.tensor import Tensor
from ..storage.host_storage import HostStorage
from ..shape.shape_storage import StaticRankArray
from ..shape.shape import TensorShape
from ..view.view import TensorView
from ..internal.status import TensorResult

def transpose[rank: Int](ref input: Tensor[rank, HostStorage], a: Int, b: Int) -> TensorResult[TensorView[rank, origin_of(input.impl.storage)]]:
    return input.transpose(a, b)

def reshape_view[rank: Int, dst_rank: Int](ref input: Tensor[rank, HostStorage], requested: TensorShape[dst_rank]) -> TensorResult[TensorView[dst_rank, origin_of(input.impl.storage)]]:
    return input.reshape_view[dst_rank](requested)


# Public functional aliases are strictly thin delegates to the PRIMARY
# Tensor facade. They never allocate, implicitly transfer, copy, or resolve
# different CPU/GPU algorithms. A zero-copy transform either returns a
# validated borrowed view or the documented representability failure.

def permute[rank: Int](
    ref input: Tensor[rank, HostStorage], axes: StaticRankArray[rank],
) -> TensorResult[TensorView[rank, origin_of(input.impl.storage)]]:
    return input.permute(axes)


def swapaxes[rank: Int](
    ref input: Tensor[rank, HostStorage], first: Int, second: Int,
) -> TensorResult[TensorView[rank, origin_of(input.impl.storage)]]:
    return input.swapaxes(first, second)


def movedim[rank: Int](
    ref input: Tensor[rank, HostStorage], source: Int, destination: Int,
) -> TensorResult[TensorView[rank, origin_of(input.impl.storage)]]:
    return input.movedim(source, destination)


def unsqueeze[rank: Int, axis: Int](
    ref input: Tensor[rank, HostStorage],
) -> TensorResult[TensorView[rank + 1, origin_of(input.impl.storage)]] where rank < 8:
    return input.unsqueeze_view[axis]()


def squeeze_axis[rank: Int, axis: Int](
    ref input: Tensor[rank, HostStorage],
) -> TensorResult[TensorView[rank - 1, origin_of(input.impl.storage)]] where rank > 0:
    return input.squeeze_axis_view[axis]()


def squeeze_axes[rank: Int, count: Int](
    ref input: Tensor[rank, HostStorage], axes: StaticRankArray[count],
) -> TensorResult[TensorView[rank - count, origin_of(input.impl.storage)]] where count >= 0 and count <= rank:
    return input.squeeze_axes[count](axes)


def expand[rank: Int, new_rank: Int](
    ref input: Tensor[rank, HostStorage], target: TensorShape[new_rank],
) -> TensorResult[TensorView[new_rank, origin_of(input.impl.storage)]] where new_rank >= rank and new_rank <= 8:
    return input.expand_view[new_rank](target)


def broadcast_to[rank: Int, new_rank: Int](
    ref input: Tensor[rank, HostStorage], target: TensorShape[new_rank],
) -> TensorResult[TensorView[new_rank, origin_of(input.impl.storage)]] where new_rank >= rank and new_rank <= 8:
    return input.broadcast_to[new_rank](target)


def flatten_view[rank: Int, start_dim: Int, end_dim: Int](
    ref input: Tensor[rank, HostStorage],
) -> TensorResult[TensorView[rank - (end_dim - start_dim), origin_of(input.impl.storage)]] where rank > 0:
    return input.flatten_view[start_dim, end_dim]()


def ravel_view[rank: Int](
    ref input: Tensor[rank, HostStorage],
) -> TensorResult[TensorView[1, origin_of(input.impl.storage)]]:
    return input.ravel_view()


def ravel_f_view[rank: Int](
    ref input: Tensor[rank, HostStorage],
) -> TensorResult[TensorView[1, origin_of(input.impl.storage)]]:
    return input.ravel_f_view()


def flip[rank: Int](
    ref input: Tensor[rank, HostStorage], axis: Int,
) -> TensorResult[TensorView[rank, origin_of(input.impl.storage)]]:
    return input.flip(axis)


def reshape_f_view[rank: Int, dst_rank: Int](
    ref input: Tensor[rank, HostStorage], requested: TensorShape[dst_rank],
) -> TensorResult[TensorView[dst_rank, origin_of(input.impl.storage)]]:
    return input.reshape_f_view[dst_rank](requested)
