# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR FACTORY / EXTERNAL BUFFER CONSTRUCTION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Factories
# Module:        momijo.tensor_light.factory.from_buffer
# Source:        src/momijo/tensor_light/factory/from_buffer.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines external buffer construction in the tensor factories layer of the
#   FP32 Light tensor implementation.
#
#   Public declarations include `from_copy_span`, `from_copy_scalar`,
#   `unsafe_adopt_external_fp32_tensor`; preconditions and supported targets
#   are determined by their implementations.
#
# Responsibilities:
#   - Create tensors with checked layouts, allocation, and initialization.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `from_copy_span`,
#     `from_copy_scalar`, `unsafe_adopt_external_fp32_tensor`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate output shape, allocation size, and initialization state.
#   - Do not expose partially initialized storage as a completed tensor.
#   - Never promote unknown provenance to proven non-aliasing.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid redundant zeroing and intermediate allocations when safe.
#   - Use one checked allocation boundary per created result where possible.
#   - Do not recycle memory before lifetime and completion proofs hold.
#
# Maintainability:
#   - Keep factory names consistent with storage/ownership contracts.
#   - Test invalid shapes and allocation failures.
#   - Cover recycled blocks, ownership failures, and overlapping buffers.
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

# Explicit deep-copy construction from borrowed contiguous FP32 host memory.
# No implicit adoption and no lifetime escape into owning Tensor.
from std.collections import Span
from ..core.tensor import Tensor
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..storage.host_storage import HostStorage
from ..storage.external_buffer import ExternalFp32Deleter, ExternalOwnedStorage
from ..layout.layout import TensorLayout
from ..factory.creation import zeros, full
from ..internal.status import TensorResult, TensorStatus
from ..iterator.nd_iterator import NDIndexIterator


def from_copy_span[rank: Int, origin: Origin](source: Span[False, Float32, origin], shape: TensorShape[rank]) -> TensorResult[Tensor[rank, HostStorage]]:
    return Tensor[rank, HostStorage].from_copy_span_checked(source, shape)


def from_copy_scalar(value: Float32) -> TensorResult[Tensor[0, HostStorage]]:
    var shape = TensorShape[0].checked(StaticRankArray[0](0))
    if not shape.is_ok():
        return TensorResult[Tensor[0, HostStorage]](shape.status)
    return full[0](shape.payload.value(), value)


# Explicit ZERO-COPY ownership transfer from a foreign host allocation. This
# is deliberately unsafe because the foreign allocator and exclusivity cannot
# be inspected by Mojo. Rejection BEFORE successful adoption leaves ownership
# with the caller; rejection AFTER adoption destroys the newly owned storage
# using its original deleter, rather than leaking it or using Mojo free().
def unsafe_adopt_external_fp32_tensor[rank: Int, D: ExternalFp32Deleter](
    address: Int, allocation_elements: Int, alignment: Int,
    var deleter: D, layout: TensorLayout[rank],
) -> TensorResult[Tensor[rank, ExternalOwnedStorage[D]]]:
    # Tensor itself is the authoritative external ownership entry point.
    return Tensor[rank, ExternalOwnedStorage[D]].unsafe_adopt_external_checked[D](
        address, allocation_elements, alignment, deleter, layout
    )
