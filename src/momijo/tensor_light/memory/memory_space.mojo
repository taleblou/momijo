# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR MEMORY / MEMORY SPACES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Memory and Allocation
# Module:        momijo.tensor_light.memory.memory_space
# Source:        src/momijo/tensor_light/memory/memory_space.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines memory spaces in the memory and allocation layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `MemorySpace`, `AllocationDomain`,
#   `host_pageable_domain`, `device_local_domain`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Represent address-space and allocation-domain identities used in memory
#     checks.
#   - Expose source-defined interfaces including `MemorySpace`,
#     `AllocationDomain`, `same_pool_domain`, `is_valid`.
#   - Validate only declared domain compatibility; do not imply allocation or
#     transfer.
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

# Memory-space identities are explicit. No invisible CPU-GPU migration.
from ..internal.status import TensorResult, TensorStatus

@fieldwise_init
struct MemorySpace(Copyable, Equatable):
    var code: Int
    comptime HOST_PAGEABLE = Self(0)
    comptime HOST_PINNED = Self(1)
    comptime DEVICE_LOCAL = Self(2)
    comptime SHARED_MAPPED = Self(3)
    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

@fieldwise_init
struct AllocationDomain(Copyable):
    var space: MemorySpace
    var device_ordinal: Int
    var alignment: Int
    var size_class: Int

    def same_pool_domain(self, other: Self) -> Bool:
        # These public value fields may be modified after construction.
        # Never equate two malformed keys as proof for cross-owner reuse.
        if not self.is_valid() or not other.is_valid():
            return False
        return (self.space == other.space
                and self.device_ordinal == other.device_ordinal
                and self.alignment == other.alignment
                and self.size_class == other.size_class)

    def is_valid(self) -> Bool:
        if self.alignment < 4 or (self.alignment & (self.alignment - 1)) != 0:
            return False
        if self.size_class < 0:
            return False
        if self.space == MemorySpace.HOST_PAGEABLE:
            return self.device_ordinal == -1
        if (self.space == MemorySpace.HOST_PINNED
            or self.space == MemorySpace.DEVICE_LOCAL
            or self.space == MemorySpace.SHARED_MAPPED):
            return self.device_ordinal >= 0
        return False

# A pool domain is a (memory kind, device ordinal, alignment, size-class)
# tuple. Values are checked before a cached allocation is considered reusable.


def host_pageable_domain(alignment: Int, size_class: Int) -> TensorResult[AllocationDomain]:
    if alignment < 4 or (alignment & (alignment - 1)) != 0 or size_class < 0:
        return TensorResult[AllocationDomain](TensorStatus.INVALID_LAYOUT)
    return TensorResult[AllocationDomain](
        AllocationDomain(MemorySpace.HOST_PAGEABLE, -1, alignment, size_class)
    )


def device_local_domain(ordinal: Int, alignment: Int, size_class: Int) -> TensorResult[AllocationDomain]:
    if ordinal < 0 or alignment < 4 or (alignment & (alignment - 1)) != 0 or size_class < 0:
        return TensorResult[AllocationDomain](TensorStatus.INVALID_LAYOUT)
    return TensorResult[AllocationDomain](
        AllocationDomain(MemorySpace.DEVICE_LOCAL, ordinal, alignment, size_class)
    )


def host_pinned_domain(
    ordinal: Int, alignment: Int, size_class: Int,
) -> TensorResult[AllocationDomain]:
    # This is a declarative pool key, NOT a pin/registration operation.
    # A native DeviceContext HostBuffer owner and a completed transfer event
    # must be bound separately before DMA or cache reuse can occur.
    var domain = AllocationDomain(MemorySpace.HOST_PINNED, ordinal, alignment, size_class)
    if not domain.is_valid():
        return TensorResult[AllocationDomain](TensorStatus.INVALID_LAYOUT)
    return TensorResult[AllocationDomain](domain)


def shared_host_device_domain(
    ordinal: Int, alignment: Int, size_class: Int,
) -> TensorResult[AllocationDomain]:
    # A mapped/managed-memory key alone does not prove pointer mapping or
    # that an accelerator can dereference the host allocation. A checked
    # owner/backend capability gate remains mandatory before any access.
    var domain = AllocationDomain(MemorySpace.SHARED_MAPPED, ordinal, alignment, size_class)
    if not domain.is_valid():
        return TensorResult[AllocationDomain](TensorStatus.INVALID_LAYOUT)
    return TensorResult[AllocationDomain](domain)


def checked_same_pool_domain(
    a: AllocationDomain, b: AllocationDomain,
) -> TensorResult[Bool]:
    if not a.is_valid() or not b.is_valid():
        return TensorResult[Bool](TensorStatus.INVALID_LAYOUT)
    return TensorResult[Bool](a.same_pool_domain(b))


def is_host_visible(space: MemorySpace) -> Bool:
    return space == MemorySpace.HOST_PAGEABLE or space == MemorySpace.HOST_PINNED or space == MemorySpace.SHARED_MAPPED


def is_device_local(space: MemorySpace) -> Bool:
    return space == MemorySpace.DEVICE_LOCAL


def is_transfer_compatible(src: MemorySpace, dst: MemorySpace) -> Bool:
    if src.code < 0 or src.code > 3 or dst.code < 0 or dst.code > 3:
        return False
    # Logical eligibility only; runtime backend must separately confirm the
    # specific CPU/device pair and transfer mechanism before issuing a copy.
    return True
