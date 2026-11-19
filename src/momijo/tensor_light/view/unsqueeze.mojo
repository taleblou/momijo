# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR VIEW / DIMENSION INSERTION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Views
# Module:        momijo.tensor_light.view.unsqueeze
# Source:        src/momijo/tensor_light/view/unsqueeze.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines dimension insertion in the tensor views layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `unsqueeze`, `squeeze_axis`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Build safe tensor views and layout-only transformations.
#   - Preserve documented behavior for bounds, strides, view provenance, and
#     write safety.
#   - Provide source-declared interfaces such as `unsqueeze`, `squeeze_axis`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate shape/stride/span invariants for borrowed views.
#   - Enforce write restrictions for broadcast or overlapping mappings.
#
# Performance Objectives:
#   - Prefer zero-copy metadata transforms when valid.
#   - Avoid view materialization unless the API explicitly requires it.
#
# Maintainability:
#   - Keep view semantics consistent with owning-tensor operations.
#   - Test permutations, negative strides, and zero-length axes.
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

# Thin, zero-copy view metadata delegates; no second stride or rank algorithm.
from ..layout.layout import (
    TensorLayout, unsqueeze_layout, squeeze_axis_layout,
)
from ..internal.status import TensorResult


def unsqueeze[rank: Int, axis: Int](
    source: TensorLayout[rank], capacity: Int,
) -> TensorResult[TensorLayout[rank + 1]] where rank >= 0 and rank < 8 and axis >= 0 and axis <= rank:
    return unsqueeze_layout[rank, axis](source, capacity)


def squeeze_axis[rank: Int, axis: Int](
    source: TensorLayout[rank], capacity: Int,
) -> TensorResult[TensorLayout[rank - 1]] where rank >= 1 and rank <= 8 and axis >= 0 and axis < rank:
    return squeeze_axis_layout[rank, axis](source, capacity)
