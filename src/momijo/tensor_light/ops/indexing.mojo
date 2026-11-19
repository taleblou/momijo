# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / INDEXING OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Indexing Operations
# Module:        momijo.tensor_light.ops.indexing
# Source:        src/momijo/tensor_light/ops/indexing.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines checked tensor indexing, gather, scatter, and related index
#   transforms.
#
#   Public declarations include `basic_getitem`, `basic_setitem`, `gather`,
#   `ScatterConflictPolicy`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `basic_getitem`,
#     `basic_setitem`, `gather`, `ScatterConflictPolicy`, `scatter` and
#     related symbols.
#   - Validate gather, scatter, and slicing index domains.
#   - Implement source-defined scatter conflict/overwrite behavior.
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

# Public basic-index helpers. Integer array indexing is rank-static.

from ..core.tensor import Tensor
from ..storage.storage import TensorStorage, MutableTensorStorage
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus
from std.collections import Span
from ..shape.shape import TensorShape
from ..shape.axis import normalize_axis
from ..factory.creation import _complete_overwrite_output
from ..storage.host_storage import HostStorage
from ..iterator.nd_iterator import NDIndexIterator
from ..indexing.index import normalize_index
from ..view.view import TensorView

def basic_getitem[rank: Int, S: TensorStorage](tensor: Tensor[rank, S], index: StaticRankArray[rank]) -> TensorResult[Float32]:
    return tensor.get(index)

def basic_setitem[rank: Int, S: MutableTensorStorage](mut tensor: Tensor[rank, S], index: StaticRankArray[rank], value: Float32) -> TensorStatus:
    return tensor.set(index, value)


# Integer indices are an auxiliary Int Span, not a second tensor dtype.
# Gather always materializes; all bounds are checked before writing output.


def gather[rank: Int, S: TensorStorage, origin: Origin](
    source: Tensor[rank, S], axis: Int, indices: Span[False, Int, origin]
) -> TensorResult[Tensor[rank, HostStorage]]:
    comptime if identical(S, HostStorage):
        return source.gather(axis, indices)
    if not source.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    var checked_source = source._checked_current_layout()
    if not checked_source.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked_source.status)
    var axis_result = normalize_axis[rank](axis)
    if not axis_result.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](axis_result.status)
    var selected_axis = axis_result.payload.value()
    var extent = checked_source.payload.value().shape.dims.get(selected_axis)
    # Validate the entire index vector before allocating/materializing output.
    for i in range(len(indices)):
        var at = normalize_index(indices[i], extent)
        if not at.is_ok():
            return TensorResult[Tensor[rank, HostStorage]](at.status)
    var dims = checked_source.payload.value().shape.dims.copy()
    dims.set(selected_axis, len(indices))
    var output_shape = TensorShape[rank].checked(dims)
    if not output_shape.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](output_shape.status)
    var made = _complete_overwrite_output[rank](output_shape.payload.value())
    if not made.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](made.status)
    var output = made.payload.take()
    # HostStorage has a proven live span. The source layout and EVERY index
    # were checked before materialization; no per-element TensorResult needed.
    var safe_source = checked_source.payload.value()
    var indexed_pitch = safe_source.strides.get(selected_axis)
    var steps = safe_source.strides.copy()
    steps.set(selected_axis, 0)
    # Generic host-readable Storage retains its checked physical access,
    # but the source Layout proof and destination bounds are established once.
    var destination = output.impl.storage.borrow_mut_span()
    var cursor = NDIndexIterator[rank](output_shape.payload.value())
    var linear = 0
    while not cursor.done():
        var selected = indices[cursor.coordinates.get(selected_axis)]
        if selected < 0:
            selected += extent
        var physical = safe_source.offset + selected * indexed_pitch
        comptime for i in range(rank):
            physical += cursor.coordinates.get(i) * steps.get(i)
        var element = source.impl.storage.read_host(physical)
        if not element.is_ok():
            return TensorResult[Tensor[rank, HostStorage]](element.status)
        destination[linear] = element.payload.value()
        linear += 1
        cursor.advance()
    return output._finish_complete_overwrite()


# Sequential last-write semantics are well-defined for repeated index values;
# GPU scatter must use an explicit conflict policy and target-specific lowering.
@fieldwise_init
struct ScatterConflictPolicy(Copyable, Equatable):
    var code: Int
    comptime REPLACE_LAST = Self(0)

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code


def scatter[rank: Int, origin: Origin](
    mut destination: Tensor[rank, HostStorage],
    axis: Int,
    indices: Span[False, Int, origin],
    source: Tensor[rank, HostStorage],
    policy: ScatterConflictPolicy,
) -> TensorStatus:
    # The owning Tensor is the sole HostStorage scatter implementation.
    return destination.scatter_from(axis, indices, source, policy.code)


# A basic slice, never a materialization or dynamic-rank index operation.

def narrow[rank: Int](
    ref input: Tensor[rank, HostStorage], axis: Int, start: Int, length: Int,
) -> TensorResult[TensorView[rank, origin_of(input.impl.storage)]]:
    return input.narrow(axis, start, length)
