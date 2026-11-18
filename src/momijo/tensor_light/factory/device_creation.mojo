# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR FACTORY / DEVICE TENSOR CREATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Factories
# Module:        momijo.tensor_light.factory.device_creation
# Source:        src/momijo/tensor_light/factory/device_creation.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines device tensor creation in the tensor factories layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `empty_device`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Create tensors with checked layouts, allocation, and initialization.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `empty_device`.
#   - Reject unsupported target or unverified completion states.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate output shape, allocation size, and initialization state.
#   - Do not expose partially initialized storage as a completed tensor.
#   - Reject unsupported or unauthenticated execution paths explicitly.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid redundant zeroing and intermediate allocations when safe.
#   - Use one checked allocation boundary per created result where possible.
#   - Avoid implicit host/device migration or compilation at execution time.
#
# Maintainability:
#   - Keep factory names consistent with storage/ownership contracts.
#   - Test invalid shapes and allocation failures.
#   - Keep device-specific capability statements conservative.
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

# Explicit GPU owner factory is a thin alias; allocation belongs to Tensor.
from max.gpu.host import DeviceContext
from ..core.tensor import Tensor, UninitializedDeviceTensor
from ..storage.device_storage import DeviceStorage
from ..shape.shape import TensorShape
from ..internal.status import TensorResult


def empty_device[rank: Int](
    ref context: DeviceContext, shape: TensorShape[rank]
) -> TensorResult[UninitializedDeviceTensor[rank]]:
    return Tensor[rank, DeviceStorage].empty_device_checked(context, shape)
