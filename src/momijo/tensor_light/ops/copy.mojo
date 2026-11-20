# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / COPYING
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Copying
# Module:        momijo.tensor_light.ops.copy
# Source:        src/momijo/tensor_light/ops/copy.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor copying, contiguity conversion, or explicit host/device
#   copy entry points.
#
#   Public declarations include `clone`, `reshape_copy`, `ravel_copy`,
#   `flatten_copy`; preconditions and supported targets are determined by
#   their implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `clone`, `reshape_copy`,
#     `ravel_copy`, `flatten_copy`, `to_device` and related symbols.
#   - Keep source/destination shape and ownership validation explicit.
#   - Provide host/device copy entry points where implemented.
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

# Explicit deep copy: no view-or-copy ambiguity.
from ..core.tensor import Tensor
from ..shape.shape import TensorShape
from ..layout.memory_format import ContiguousOrder
from ..storage.storage import TensorStorage
from ..storage.host_storage import HostStorage
from ..factory.creation import _complete_overwrite_output
from ..iterator.nd_iterator import NDIndexIterator
from ..internal.status import TensorResult, TensorStatus
from ..view.view import TensorView, TensorMutView
from ..kernels.cpu.copy import cpu_copy_strided, cpu_fill_strided, cpu_zero_strided
from max.gpu.host import DeviceContext
from ..storage.device_storage import DeviceStorage

def clone[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Tensor[rank, HostStorage]]:
    if not input.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    var checked_input = input._checked_current_layout()
    if not checked_input.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked_input.status)
    comptime if S == HostStorage:
        return rebind[Tensor[rank, HostStorage]](input).clone()
    # The generic host-readable source is traversed in full logical C-order.
    var made = _complete_overwrite_output[rank](checked_input.payload.value().shape)
    if not made.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](made.status)
    var result = made.payload.take()
    var src_layout = checked_input.payload.value()
    var destination = result.impl.storage.borrow_mut_span()
    var cursor = NDIndexIterator[rank](src_layout.shape)
    while not cursor.done():
        var offset = src_layout.physical_offset_unchecked(cursor.coordinates)
        var value = input.impl.storage.read_host(offset)
        if not value.is_ok():
            return TensorResult[Tensor[rank, HostStorage]](value.status)
        destination[cursor.produced] = value.payload.value()
        cursor.advance()
    return (result^)._finish_complete_overwrite()


# Materialization is explicit; view-only operations are in ops/view.mojo.
def reshape_copy[rank: Int, out_rank: Int](
    ref source: Tensor[rank, HostStorage], shape: TensorShape[out_rank]
) -> TensorResult[Tensor[out_rank, HostStorage]]:
    return source.reshape_copy[out_rank](shape)


def ravel_copy[rank: Int](
    ref source: Tensor[rank, HostStorage]
) -> TensorResult[Tensor[1, HostStorage]]:
    return source.ravel_copy()


def flatten_copy[rank: Int, start_dim: Int, end_dim: Int](
    ref source: Tensor[rank, HostStorage]
) -> TensorResult[Tensor[rank - (end_dim - start_dim), HostStorage]] where rank >= 1:
    return source.flatten_copy[start_dim, end_dim]()


# The FP32 owning Tensor remains the primary API. Module-level transfer names
# are thin delegates to Tensor.to_device()/to_host(), never distinct backends.
def to_device[rank: Int](
    ref source: Tensor[rank, HostStorage], ctx: DeviceContext
) -> TensorResult[Tensor[rank, DeviceStorage]]:
    return source.to_device(ctx)


def to_host[rank: Int](
    ref source: Tensor[rank, DeviceStorage], ctx: DeviceContext
) -> TensorResult[Tensor[rank, HostStorage]]:
    return source.to_host(ctx)

# Checked zero-copy-view-to-view CPU materialization. This is an explicit copy,
# not a reshape or a hidden alias. The kernel layer owns the one-pass
# contiguous path and complete-overlap snapshot for irregular strides.
def copy_view_into[rank: Int, src_origin: Origin, dst_origin: Origin[mut=True]](
    source: TensorView[rank, src_origin],
    mut destination: TensorMutView[rank, dst_origin],
) -> TensorStatus:
    return cpu_copy_strided(source, destination)


def fill_view[rank: Int, dst_origin: Origin[mut=True]](
    mut destination: TensorMutView[rank, dst_origin], value: Float32,
) -> TensorStatus:
    return cpu_fill_strided(destination, value)


# Explicit zero-fill of an already-owned writable view; no allocation.
def zero_view[rank: Int, dst_origin: Origin[mut=True]](
    mut destination: TensorMutView[rank, dst_origin],
) -> TensorStatus:
    return cpu_zero_strided(destination)

# Same-rank materialization into an existing FP32 tensor. The primary Tensor
# implementation handles shape, mutable uniqueness and ownership safety.
def copy[rank: Int](
    mut destination: Tensor[rank, HostStorage],
    ref source: Tensor[rank, HostStorage],
) -> TensorStatus:
    return destination.copy_from(source)


# This always materializes. It never returns a view or silently aliases the
# owning input; C/F order is explicit.
def contiguous[rank: Int](
    ref source: Tensor[rank, HostStorage], order: ContiguousOrder,
) -> TensorResult[Tensor[rank, HostStorage]]:
    return source.contiguous_copy_order(order)
