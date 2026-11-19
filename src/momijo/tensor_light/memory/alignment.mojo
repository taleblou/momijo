# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR MEMORY / MEMORY ALIGNMENT
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Memory and Allocation
# Module:        momijo.tensor_light.memory.alignment
# Source:        src/momijo/tensor_light/memory/alignment.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines memory alignment in the memory and allocation layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `is_valid_alignment`,
#   `alignment_after_offset`, `is_power_of_two_alignment`,
#   `validate_alignment`; preconditions and supported targets are determined
#   by their implementations.
#
# Responsibilities:
#   - Validate power-of-two alignment and compute alignment after a byte or
#     element offset.
#   - Expose source-defined interfaces including `is_valid_alignment`,
#     `alignment_after_offset`, `is_power_of_two_alignment`,
#     `validate_alignment`.
#   - Keep alignment arithmetic consistent with the declared element size and
#     reject invalid offsets.
#   - Propagate failed validation or allocation results without silently
#     changing memory domains.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate sizes, alignment, pointer access, and memory-domain contracts
#     as applicable.
#   - Reject invalid inputs without asserting unrelated pool ownership or
#     completion semantics.
#
# Performance Objectives:
#   - Keep pointer, size, domain, and alignment checks inexpensive.
#   - Avoid introducing allocations or copies in metadata and validation
#     paths.
#
# Maintainability:
#   - Document the exact allocation, pointer, and address-space contracts
#     exposed here.
#   - Test overflow, invalid alignment, and allocation failures where
#     supported.
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

# Checked power-of-two allocation alignment; never assume noalias.
from ..internal.status import TensorStatus, TensorResult

def is_valid_alignment(alignment: Int) -> Bool:
    if alignment < 4:
        return False
    return (alignment & (alignment - 1)) == 0

def alignment_after_offset(base_alignment: Int, elements: Int) -> TensorResult[Int]:
    if not is_valid_alignment(base_alignment) or elements < 0:
        return TensorResult[Int](TensorStatus.INVALID_LAYOUT)
    # Four-byte payload: at worst every derived address remains 4-byte aligned.
    if elements == 0:
        return TensorResult[Int](base_alignment)
    var aligned = 4
    var remaining = elements
    while aligned < base_alignment and (remaining % 2) == 0:
        aligned *= 2
        remaining = remaining // 2
    return TensorResult[Int](aligned)

def is_power_of_two_alignment(alignment: Int) -> Bool:
    return is_valid_alignment(alignment)


def validate_alignment(alignment: Int) -> TensorStatus:
    if not is_valid_alignment(alignment):
        return TensorStatus.INVALID_ALIGNMENT
    return TensorStatus.OK


def natural_alignment() -> Int:
    return 4


def combine_alignment(base_alignment: Int, byte_offset: Int) -> TensorResult[Int]:
    if not is_valid_alignment(base_alignment) or byte_offset < 0 or byte_offset % 4 != 0:
        return TensorResult[Int](TensorStatus.INVALID_ALIGNMENT)
    return alignment_after_offset(base_alignment, byte_offset // 4)


def pointer_meets_alignment[origin: Origin](
    ptr: Pointer[Float32, origin], alignment: Int,
) -> Bool:
    # This proves ONLY numeric byte alignment; it does not prove that a
    # foreign address is allocated, mapped, owned, alive, or host-readable.
    if not is_valid_alignment(alignment):
        return False
    var address = Int(ptr)
    return address > 0 and address % alignment == 0
