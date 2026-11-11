# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | LEARN / PACKAGE BOUNDARY
# ==============================================================================
#
# Project:       Momijo
# Component:     LEARN
# Subcomponent:  Package Boundary
# Module:        momijo.learn
# Source:        src/momijo/learn/__init__.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Modeling and learning APIs.
#   Reserved public namespace only; no runtime implementation exists yet.
#
# Responsibilities:
#   - Establish the future learn module boundary.
#   - Expose no unimplemented functions, symbols, or runtime behavior.
#   - Require a design, implementation, and tests before publishing APIs.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Preserve tensor engine independence and explicit data ownership.
#   - Define APIs only when behavior and failure cases have been tested.
#
# Performance Objectives:
#   - Do not add eager initialization or heavyweight imports.
#   - Keep compute kernels and specialization in the responsible modules.
#
# Maintainability:
#   - Define distinct backend adapters instead of importing the two tensor engines into each other.
#   - Synchronize future code, documentation, examples and tests.
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
#   of it must retain applicable notices under the MIT License.
#
# Attribution:
#   Project acknowledgment is appreciated but not a license requirement.
#
# ==============================================================================
