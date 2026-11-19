# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / CREATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Creation
# Module:        momijo.tensor_light.ops.creation
# Source:        src/momijo/tensor_light/ops/creation.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Exposes owning tensor factory APIs through the FP32 Light operations
#   namespace.
#
#   The source re-exports names from `..factory.creation`, `..factory.like`,
#   `..factory.device_creation` without implementing tensor kernels or
#   allocating output storage here.
#
# Responsibilities:
#   - Preserve the imported public names and any intentional API aliases.
#   - Expose source-declared symbols including `empty`, `zeros`, `ones`,
#     `full`, `empty_with_options`.
#   - Keep implementation and validation logic in the canonical owner modules.
#   - Avoid circular dependencies or duplicate tensor and storage ownership.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Ensure every exported binding resolves to a source-defined canonical
#     symbol.
#   - Preserve alias names and the original implementation's type and error
#     contracts.
#   - Do not claim that this import-only file performs independent validation
#     or execution.
#
# Performance Objectives:
#   - Avoid import-time allocation, execution, and needless dependencies.
#   - Route calls directly to canonical implementations without adding
#     hot-path wrappers.
#
# Maintainability:
#   - Update exported bindings together with their canonical owner modules.
#   - Keep backward compatibility intentional and test import resolution.
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

# Public factory namespace mirrors Tensor's primary owning API.
from ..factory.creation import empty, zeros, ones, full
from ..factory.creation import empty_with_options, full_with_options, zeros_with_options, ones_with_options
from ..factory.like import empty_like, zeros_like, ones_like, full_like
from ..factory.device_creation import empty_device
