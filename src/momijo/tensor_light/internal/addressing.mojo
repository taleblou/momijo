# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR INTERNAL / CHECKED ADDRESSING
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Internal Validation Utilities
# Module:        momijo.tensor_light.internal.addressing
# Source:        src/momijo/tensor_light/internal/addressing.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines checked addressing in the internal validation utilities layer of
#   the FP32 Light tensor implementation.
#
#   Public declarations include `byte_offset`, `range_fits`,
#   `PhysicalHalfOpenRange`, `element_offset_to_byte_offset`; preconditions
#   and supported targets are determined by their implementations.
#
# Responsibilities:
#   - Provide reusable checked primitives and status contracts.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `byte_offset`, `range_fits`,
#     `PhysicalHalfOpenRange`, `element_offset_to_byte_offset`,
#     `range_from_offset_count` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Detect integer overflow and invalid metadata before dereference.
#   - Return explicit statuses for unsupported or invalid states.
#
# Performance Objectives:
#   - Use simple inlinable checked primitives where appropriate.
#   - Keep validation work proportional to required metadata.
#
# Maintainability:
#   - Reuse one error vocabulary across dependent modules.
#   - Test boundary arithmetic and invalid-state propagation.
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

# Safe byte/element address conversion for Float32-only tensors.
from .checked_math import checked_mul_nonnegative, checked_add
from .status import TensorResult, TensorStatus

def byte_offset(element_offset: Int) -> TensorResult[Int]:
    if element_offset < 0:
        return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
    return checked_mul_nonnegative(element_offset, 4)

def range_fits(offset: Int, elements: Int, capacity: Int) -> Bool:
    if offset < 0 or elements < 0 or capacity < 0:
        return False
    if elements > capacity:
        return False
    return offset <= capacity - elements

@fieldwise_init
struct PhysicalHalfOpenRange(Copyable):
    var begin: Int
    var end: Int


def element_offset_to_byte_offset(element_offset: Int) -> TensorResult[Int]:
    return byte_offset(element_offset)


def range_from_offset_count(offset: Int, count: Int) -> TensorResult[PhysicalHalfOpenRange]:
    # Half-open offsets avoid inclusive-end overflow on empty allocations.
    if offset < 0 or count < 0:
        return TensorResult[PhysicalHalfOpenRange](TensorStatus.OUT_OF_BOUNDS)
    var end = checked_add(offset, count)
    if not end.is_ok():
        return TensorResult[PhysicalHalfOpenRange](end.status)
    return TensorResult[PhysicalHalfOpenRange](PhysicalHalfOpenRange(offset, end.payload.value()))


def ranges_disjoint(a: PhysicalHalfOpenRange, b: PhysicalHalfOpenRange) -> Bool:
    # Public range metadata is mutable. A forged reversed/negative interval
    # must NEVER establish a proof of non-overlap for an unsafe copy/write.
    # The intervals must also refer to the SAME verified allocation identity;
    # callers prove that separately at the owner boundary.
    if a.begin < 0 or b.begin < 0:
        return False
    if a.end < a.begin or b.end < b.begin:
        return False
    if a.begin == a.end or b.begin == b.end:
        return True
    return a.end <= b.begin or b.end <= a.begin
