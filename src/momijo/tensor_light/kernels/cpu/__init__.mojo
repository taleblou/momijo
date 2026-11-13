# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | CPU KERNEL / CPU TENSOR KERNELS INTERFACE
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / CPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.cpu
# Source:        src/momijo/tensor_light/kernels/cpu/__init__.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines the CPU tensor kernels package boundary for the FP32 Light
#   implementation.
#
#   Re-exports explicitly imported interfaces without declaring an independent
#   implementation.
#
# Responsibilities:
#   - Define the CPU tensor kernels import and namespace boundary.
#   - Keep imported public names consistent with their owning modules.
#   - Avoid introducing duplicate implementations or initialization side
#     effects.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Expose only symbols actually defined by imported modules.
#   - Preserve the established package API and import graph.
#
# Performance Objectives:
#   - Avoid initialization side effects or eager heavy allocations.
#   - Keep executable hot-path implementation in dedicated modules.
#
# Maintainability:
#   - Update curated exports when public APIs move or change.
#   - Do not add import cycles or duplicate implementations.
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

# Curated package exports. No initialization side effects.
from .elementwise import cpu_add_contiguous, cpu_scale_contiguous
from .reduction import cpu_sum_contiguous
from .matrix import cpu_mm_contiguous
from .copy import cpu_copy_contiguous, cpu_copy_strided, cpu_fill, cpu_fill_strided, cpu_zero, cpu_zero_strided, cpu_prefetch_copy
