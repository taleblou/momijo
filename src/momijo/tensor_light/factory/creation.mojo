# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR FACTORY / CREATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Factories
# Module:        momijo.tensor_light.factory.creation
# Source:        src/momijo/tensor_light/factory/creation.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor creation in the tensor factories layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `empty`, `full`, `zeros`, `ones`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Create tensors with checked layouts, allocation, and initialization.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `empty`, `full`, `zeros`,
#     `ones`, `empty_with_options` and related symbols.
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

# Public convenience functions forward to the primary owning Tensor API.
# Allocation, checked shape, and canonical layout live in Tensor.full_checked.
from ..shape.shape import TensorShape
from ..storage.host_storage import HostStorage
from ..core.tensor import Tensor, UninitializedHostTensor
from ..internal.status import TensorResult
from ..core.tensor_options import TensorOptions


def empty[rank: Int](shape: TensorShape[rank]) -> TensorResult[UninitializedHostTensor[rank]]:
    # Allocation-only capability: no readable Tensor until initialize_fill()
    # or initialize_from_physical_span() completes.
    return Tensor[rank, HostStorage].empty_checked(shape)


def full[rank: Int](shape: TensorShape[rank], value: Float32) -> TensorResult[Tensor[rank, HostStorage]]:
    return Tensor[rank, HostStorage].full_checked(shape, value)


def _complete_overwrite_output[rank: Int](
    shape: TensorShape[rank]
) -> TensorResult[Tensor[rank, HostStorage]]:
    # Internal materialization only; never return the buffer with holes.
    return Tensor[rank, HostStorage]._empty_for_complete_overwrite_checked(shape)


def zeros[rank: Int](shape: TensorShape[rank]) -> TensorResult[Tensor[rank, HostStorage]]:
    return Tensor[rank, HostStorage].zeros_checked(shape)


def ones[rank: Int](shape: TensorShape[rank]) -> TensorResult[Tensor[rank, HostStorage]]:
    return Tensor[rank, HostStorage].ones_checked(shape)


def empty_with_options[rank: Int](
    shape: TensorShape[rank], options: TensorOptions
) -> TensorResult[UninitializedHostTensor[rank]]:
    return Tensor[rank, HostStorage].empty_with_options_checked(shape, options)


def full_with_options[rank: Int](
    shape: TensorShape[rank], value: Float32, options: TensorOptions
) -> TensorResult[Tensor[rank, HostStorage]]:
    return Tensor[rank, HostStorage].full_with_options_checked(shape, value, options)


def zeros_with_options[rank: Int](
    shape: TensorShape[rank], options: TensorOptions
) -> TensorResult[Tensor[rank, HostStorage]]:
    return Tensor[rank, HostStorage].zeros_with_options_checked(shape, options)


def ones_with_options[rank: Int](
    shape: TensorShape[rank], options: TensorOptions
) -> TensorResult[Tensor[rank, HostStorage]]:
    return Tensor[rank, HostStorage].ones_with_options_checked(shape, options)
