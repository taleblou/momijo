# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR STORAGE / STORAGE PROVENANCE
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Storage and Ownership
# Module:        momijo.tensor_light.storage.ownership
# Source:        src/momijo/tensor_light/storage/ownership.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines storage provenance in the storage and ownership layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `StorageProvenance`, `StorageOwnership`,
#   `StorageIdentity`, `owned_provenance`; preconditions and supported targets
#   are determined by their implementations.
#
# Responsibilities:
#   - Enforce tensor storage lifetime and initialized-data contracts.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `StorageProvenance`,
#     `StorageOwnership`, `StorageIdentity`, `owned_provenance`,
#     `borrowed_provenance` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Preserve storage provenance and initialization state.
#   - Prevent use-after-release and unsafe mutable aliasing.
#   - Never promote unknown provenance to proven non-aliasing.
#
# Performance Objectives:
#   - Avoid storage duplication when a safe borrow is sufficient.
#   - Reclaim storage only after ownership/completion conditions hold.
#   - Do not recycle memory before lifetime and completion proofs hold.
#
# Maintainability:
#   - Document ownership transitions and destructor behavior.
#   - Test external-borrow lifetime and partially initialized buffers.
#   - Cover recycled blocks, ownership failures, and overlapping buffers.
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

# Provenance is required for alias/noalias claims, not merely address distance.
@fieldwise_init
struct StorageProvenance(Copyable):
    var allocation_id: Int
    var is_known: Bool
    var is_owned: Bool

    # All three fields are writable and callers can construct a forged value.
    # Without a sealed allocation-owner capability, this metadata is advisory,
    # never an authenticated right to declare pointer ranges disjoint.
    # Preserve the query methods for API compatibility but fail CLOSED until
    # the implementation supplies a proof tied to the concrete owned storage.
    def trusted_owned_identity(self) -> Bool:
        return False

    def definitely_distinct(self, other: Self) -> Bool:
        # Two different integer IDs may still refer to overlapping raw memory.
        # Never authorize unsafe_as_noalias or memcpy from this comparison.
        return False

    def same_allocation(self, other: Self) -> Bool:
        # Identical caller-controlled IDs also do not prove a common base;
        # offset intervals are meaningful only after a trusted owner proof.
        return False


# Stable storage-ownership vocabulary. These descriptors are informational:
# only a real live Storage borrow and checked physical pointer range may
# authorize a memcpy/noalias optimization. Integer IDs are never that proof.
@fieldwise_init
struct StorageOwnership(Copyable):
    var code: Int
    comptime OWNED = Self(0)
    comptime BORROWED = Self(1)
    comptime EXTERNAL_UNKNOWN = Self(2)


@fieldwise_init
struct StorageIdentity(Copyable):
    # Advisory identifier, not a security token. Callers can construct and
    # copy this value; it conveys no unforgeable allocation provenance.
    var allocation_id: Int


def owned_provenance(allocation_id: Int) -> StorageProvenance:
    # Known numeric identity for diagnostics; not an authenticated owner.
    if allocation_id < 0:
        return StorageProvenance(-1, False, False)
    return StorageProvenance(allocation_id, True, True)


def borrowed_provenance(identity: StorageIdentity) -> StorageProvenance:
    # Borrowed buffers remain borrowed; a matching ID does not prove they
    # refer to the same live allocation or are safely non-overlapping.
    if identity.allocation_id < 0:
        return StorageProvenance(-1, False, False)
    return StorageProvenance(identity.allocation_id, True, False)


def unknown_external_provenance() -> StorageProvenance:
    # A foreign pointer whose allocation identity and lifetime are not
    # tracked MUST be treated as potentially aliasing any other pointer.
    return StorageProvenance(-1, False, False)


def provenance_definitely_distinct(a: StorageProvenance, b: StorageProvenance) -> Bool:
    # Forgeable IDs cannot establish distinct physical owners. A true
    # disjointness proof belongs to owner-bound alias_info/writeability.
    return False


def provenance_maybe_same(a: StorageProvenance, b: StorageProvenance) -> Bool:
    # Includes owned/adopted/foreign descriptors: conservatively MAY alias.
    return True
