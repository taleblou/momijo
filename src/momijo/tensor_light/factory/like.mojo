# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR FACTORY / LIKE-SHAPED TENSOR FACTORIES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Factories
# Module:        momijo.tensor_light.factory.like
# Source:        src/momijo/tensor_light/factory/like.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines like-shaped tensor factories in the tensor factories layer of the
#   FP32 Light tensor implementation.
#
#   Public declarations include `empty_like`, `zeros_like`, `ones_like`,
#   `full_like`; preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Create tensors with checked layouts, allocation, and initialization.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `empty_like`, `zeros_like`,
#     `ones_like`, `full_like`.
#   - Handle scalar, empty, and rank-dependent edge cases explicitly.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate output shape, allocation size, and initialization state.
#   - Do not expose partially initialized storage as a completed tensor.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid redundant zeroing and intermediate allocations when safe.
#   - Use one checked allocation boundary per created result where possible.
#
# Maintainability:
#   - Keep factory names consistent with storage/ownership contracts.
#   - Test invalid shapes and allocation failures.
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

# Host-only FP32 *like factories. There is no implicit GPU-to-host fallback.
# Preserve genuinely F-contiguous layout via the owning Tensor methods.
# Check the canonical owner layout before allocating. Returned values remain
# the primary owning Tensor, never a separate facade or hidden borrowed view.
from ..core.tensor import Tensor, UninitializedHostTensor
from ..storage.host_storage import HostStorage
from ..internal.status import TensorResult


def empty_like[rank: Int](ref source: Tensor[rank, HostStorage]) -> TensorResult[UninitializedHostTensor[rank]]:
    return source.empty_like()


def zeros_like[rank: Int](ref source: Tensor[rank, HostStorage]) -> TensorResult[Tensor[rank, HostStorage]]:
    return source.zeros_like()


def ones_like[rank: Int](ref source: Tensor[rank, HostStorage]) -> TensorResult[Tensor[rank, HostStorage]]:
    return source.ones_like()


def full_like[rank: Int](ref source: Tensor[rank, HostStorage], value: Float32) -> TensorResult[Tensor[rank, HostStorage]]:
    return source.full_like(value)
