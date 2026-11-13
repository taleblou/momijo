# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR KERNEL / KERNELS INTERFACE
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Kernels
# Module:        momijo.tensor_light.kernels
# Source:        src/momijo/tensor_light/kernels/__init__.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines the tensor kernels package boundary for the FP32 Light
#   implementation.
#
#   Acts as a namespace marker; it contains no executable tensor algorithm.
#
# Responsibilities:
#   - Declare the package boundary without introducing executable behavior.
#   - Keep the namespace stable as related modules evolve.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Keep the package marker importable and free of executable initialization
#     side effects.
#   - Do not document a re-exported API until symbols are explicitly imported.
#
# Performance Objectives:
#   - Avoid initialization side effects or eager heavy allocations.
#   - Keep executable hot-path implementation in dedicated modules.
#
# Maintainability:
#   - Preserve the package layout and avoid import cycles.
#   - Add explicit exports only when the package API requires them.
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
# No public symbols are exported by this package yet.
