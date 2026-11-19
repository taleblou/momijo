# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR INDEXING / SLICE OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Indexing
# Module:        momijo.tensor_light.indexing.slice
# Source:        src/momijo/tensor_light/indexing/slice.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines slice operations in the tensor indexing layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `SliceSpec`, `NormalizedSlice`,
#   `normalize_slice`, `slice_length`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Normalize and validate index or slicing operations.
#   - Preserve documented behavior for bounds, strides, view provenance, and
#     write safety.
#   - Provide source-declared interfaces such as `SliceSpec`,
#     `NormalizedSlice`, `normalize_slice`, `slice_length`, `is_full_slice`
#     and related symbols.
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

# Explicit slice spec supports negative step and Python-style omitted endpoints.
from std.collections import Optional
from ..internal.status import TensorStatus, TensorResult

@fieldwise_init
struct SliceSpec(Copyable):
    var start: Optional[Int]
    var stop: Optional[Int]
    var step: Int

@fieldwise_init
struct NormalizedSlice(Copyable):
    var start: Int
    var stop: Int
    var step: Int
    var length: Int

def normalize_slice(spec: SliceSpec, extent: Int) -> TensorResult[NormalizedSlice]:
    if extent < 0:
        return TensorResult[NormalizedSlice](TensorStatus.INVALID_SHAPE)
    if spec.step == 0 or spec.step == Int.MIN:
        return TensorResult[NormalizedSlice](TensorStatus.INVALID_LAYOUT)
    var positive = spec.step > 0
    var start = 0
    var stop = extent
    if positive:
        if spec.start:
            start = spec.start.value()
            if start < 0:
                if start < -extent:
                    start = 0
                else:
                    start += extent
            if start > extent:
                start = extent
        if spec.stop:
            stop = spec.stop.value()
            if stop < 0:
                if stop < -extent:
                    stop = 0
                else:
                    stop += extent
            if stop > extent:
                stop = extent
        if stop <= start:
            return TensorResult[NormalizedSlice](NormalizedSlice(start, stop, spec.step, 0))
        var length = 1 + (stop - 1 - start) // spec.step
        return TensorResult[NormalizedSlice](NormalizedSlice(start, stop, spec.step, length))
    start = extent - 1
    stop = -1
    if spec.start:
        start = spec.start.value()
        if start < 0:
            if start < -extent:
                start = -1
            else:
                start += extent
        if start >= extent:
            start = extent - 1
    if spec.stop:
        stop = spec.stop.value()
        if stop < 0:
            if stop < -extent:
                stop = -1
            else:
                stop += extent
        if stop >= extent:
            stop = extent - 1
    if start <= stop:
        return TensorResult[NormalizedSlice](NormalizedSlice(start, stop, spec.step, 0))
    var length = 1 + (start - 1 - stop) // (-spec.step)
    return TensorResult[NormalizedSlice](NormalizedSlice(start, stop, spec.step, length))


# A normalized slice length is authoritative; these helpers do no indexing
# and never allocate. Only a positive unit-step [0:extent] is the identity.
def slice_length(value: NormalizedSlice) -> Int:
    return value.length


def is_full_slice(value: NormalizedSlice, extent: Int) -> Bool:
    if extent < 0:
        return False
    return (value.step == 1 and value.start == 0 and
            value.stop == extent and value.length == extent)


def is_contiguous_slice(value: NormalizedSlice) -> Bool:
    # Both forward and reverse unit-step slices are physically consecutive.
    return value.step == 1 or value.step == -1
