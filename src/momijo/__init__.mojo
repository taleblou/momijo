# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | PACKAGE INTERFACE
# ==============================================================================
#
# Project:       Momijo
# Component:     Package
# Subcomponent:  Package Boundary
# Module:        momijo
# Source:        src/momijo/__init__.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Root namespace for the full and FP32 Light tensor engines, plus
#   four reserved future application-domain package boundaries.
#
# Responsibilities:
#   - Define an explicit Mojo package boundary.
#   - Keep independent engine symbols in their respective namespaces.
#   - Reserve dataframe, learn, vision and visual without claiming an API.
#   - Introduce no runtime initialization side effects.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Respect the declared engine namespace and import graph.
#   - Keep package exports explicit; do not introduce name collisions.
#
# Performance Objectives:
#   - Avoid unnecessary eager initialization or execution at import time.
#   - Keep kernel and runtime hot paths in their dedicated modules.
#
# Maintainability:
#   - Update import paths, tests and documentation when this boundary moves.
#   - Avoid coupling the independent full and FP32 Light implementations.
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
