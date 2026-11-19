# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / LOGICAL OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Logical Operations
# Module:        momijo.tensor_light.ops.logical
# Source:        src/momijo/tensor_light/ops/logical.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Exposes Boolean mask operations under their logical API names in the FP32
#   Light package.
#
#   The source re-exports names from `.mask` without implementing tensor
#   kernels or allocating output storage here.
#
# Responsibilities:
#   - Preserve the imported public names and any intentional API aliases.
#   - Expose source-declared symbols including `logical_and`, `logical_or`,
#     `logical_xor`, `logical_not`, `any_true`.
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

# Boolean names over the compact fixed-rank TensorMask (not Float32 bitwise).
# The checked algorithms and storage stay in mask.mojo.
from .mask import mask_and as logical_and, mask_or as logical_or
from .mask import mask_xor as logical_xor, mask_not as logical_not
from .mask import mask_any as any_true, mask_all as all_true
from .mask import mask_count_true as count_true
