# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR INDEXING / INDEX DEFINITIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Indexing
# Module:        momijo.tensor_light.indexing.index
# Source:        src/momijo/tensor_light/indexing/index.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines index definitions in the tensor indexing layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `normalize_index`,
#   `normalize_index_unchecked`, `index_reduces_rank`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Normalize and validate index or slicing operations.
#   - Preserve documented behavior for bounds, strides, view provenance, and
#     write safety.
#   - Provide source-declared interfaces such as `normalize_index`,
#     `normalize_index_unchecked`, `index_reduces_rank`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Normalize and bounds-check indices before memory access.
#   - Preserve documented negative-index and slicing semantics.
#   - Check index bounds before converting indices into physical offsets.
#
# Performance Objectives:
#   - Perform checked offset computation outside repeated element access.
#   - Avoid unnecessary index-tensor copies.
#   - Reuse normalized index ranges when the operation permits.
#
# Maintainability:
#   - Centralize slicing and index normalization.
#   - Test empty, negative-step, out-of-range, and high-rank cases.
#   - Test invalid indices, zero extents, and boundary conditions.
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

# The valid negative range for an extent n is [-n, -1].
from ..internal.status import TensorStatus, TensorResult

def normalize_index(index: Int, extent: Int) -> TensorResult[Int]:
    if extent <= 0:
        return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
    if index < 0:
        if index < -extent:
            return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
        return TensorResult[Int](index + extent)
    if index >= extent:
        return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
    return TensorResult[Int](index)

def normalize_index_unchecked(index: Int, extent: Int) -> Int:
    # PRECONDITION: extent > 0 and -extent <= index < extent.
    # A checked normalizer or trusted iteration plan must establish this once.
    if index < 0:
        return index + extent
    return index


def index_reduces_rank() -> Bool:
    # A typed scalar integer index consumes exactly one source axis.
    return True
