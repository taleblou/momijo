# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR MEMORY / ALLOCATOR POLICY
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Memory and Allocation
# Module:        momijo.tensor_light.memory.allocator
# Source:        src/momijo/tensor_light/memory/allocator.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines allocator policy in the memory and allocation layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `HostAllocationPolicy`,
#   `default_host_allocation_policy`, `allocate_checked`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Provide checked typed host allocation and reject unsupported memory
#     domains.
#   - Expose source-defined interfaces including `HostAllocationPolicy`,
#     `default_host_allocation_policy`, `allocate_checked`.
#   - Enforce memory-space restrictions and delegate allocation ownership to
#     canonical storage.
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
#   - Limit allocation overhead while satisfying size, alignment, and
#     ownership rules.
#   - Avoid redundant initialization, allocation, and copying of storage.
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

# Host allocator configuration is per-context, not a mutable global registry.
from .allocation import HostAllocation, allocate_host_fp32
from ..internal.status import TensorResult, TensorStatus

@fieldwise_init
struct HostAllocationPolicy(Copyable):
    var maximum_bytes: Int
    var minimum_alignment: Int

def default_host_allocation_policy() -> HostAllocationPolicy:
    return HostAllocationPolicy(Int.MAX, 64)

def allocate_checked(count: Int, policy: HostAllocationPolicy) -> TensorResult[HostAllocation]:
    if count < 0:
        return TensorResult[HostAllocation](TensorStatus.INVALID_SHAPE)
    if count > Int.MAX // 4:
        return TensorResult[HostAllocation](TensorStatus.OVERFLOW)
    if policy.minimum_alignment != 64:
        # The reference allocation backend guarantees 64-byte base alignment.
        return TensorResult[HostAllocation](TensorStatus.UNSUPPORTED)
    if policy.maximum_bytes < 0 or count > policy.maximum_bytes // 4:
        return TensorResult[HostAllocation](TensorStatus.ALLOCATION_FAILED)
    return allocate_host_fp32(count)
