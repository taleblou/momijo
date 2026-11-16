# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR ALIAS / WRITABLE MAPPING VALIDATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Aliasing and Write Safety
# Module:        momijo.tensor_light.alias.writeability
# Source:        src/momijo/tensor_light/alias/writeability.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines writable mapping validation in the aliasing and write safety layer
#   of the FP32 Light tensor implementation.
#
#   Public declarations include `validate_unique_writes`,
#   `validate_noalias_preconditions`, `can_use_noalias`,
#   `broadcast_write_is_safe`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Classify alias relations and enforce safe write preconditions.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `validate_unique_writes`,
#     `validate_noalias_preconditions`, `can_use_noalias`,
#     `broadcast_write_is_safe`, `validate_writable` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Distinguish proven disjointness from unknown overlap.
#   - Reject or safely handle writes through overlapping mappings.
#   - Never promote unknown provenance to proven non-aliasing.
#
# Performance Objectives:
#   - Use constant-time provenance checks before expensive overlap analysis.
#   - Retain fast copy paths only when non-overlap is proven.
#   - Do not recycle memory before lifetime and completion proofs hold.
#
# Maintainability:
#   - Keep classification logic centralized for all consumers.
#   - Regression-test zero strides, negative strides, and shared owners.
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

# Mutable access is a type capability; structural uniqueness is an additional
# proof. Neither an external pointer nor a claimed offset is a noalias proof.
from ..layout.layout import TensorLayout
from ..storage.ownership import StorageProvenance
from ..internal.status import TensorStatus


def validate_unique_writes[rank: Int](
    layout: TensorLayout[rank], storage_capacity: Int,
) -> TensorStatus:
    # The public layout is a mutable value; its cached overlap flag can be
    # forged. Recalculate the span and uniqueness against owner capacity.
    var checked = layout.validated_snapshot(storage_capacity)
    if not checked.is_ok():
        return checked.status
    if not checked.payload.value().is_proven_unique():
        return TensorStatus.OVERLAPPING
    return TensorStatus.OK


def validate_noalias_preconditions[dst_rank: Int, src_rank: Int](
    dst: TensorLayout[dst_rank], dst_capacity: Int,
    dst_identity: StorageProvenance,
    src: TensorLayout[src_rank], src_capacity: Int,
    src_identity: StorageProvenance,
) -> TensorStatus:
    # Rebuild every cached span and overlap proof at the checked boundary.
    # This operation does not grant mutable access: the receiving pointer must
    # ALSO carry mutable type-level capability before any write is performed.
    var d_proof = dst.validated_snapshot(dst_capacity)
    if not d_proof.is_ok():
        return d_proof.status
    var s_proof = src.validated_snapshot(src_capacity)
    if not s_proof.is_ok():
        return s_proof.status
    var d = d_proof.payload.value()
    var s = s_proof.payload.value()
    if not d.is_proven_unique():
        return TensorStatus.OVERLAPPING
    # Only the freshly reconstructed, capacity-bound layouts can certify
    # emptiness. Never delegate that proof to a forged AliasInfo.empty flag.
    if d.shape.is_empty() or s.shape.is_empty():
        return TensorStatus.OK
    # Neither claimed ownership flags nor matching/different integer IDs can
    # authorize noalias: both are freely writable by the external caller.
    # An owner-bound capability-based noalias fast path must be a separate API.
    return TensorStatus.ALIAS_UNKNOWN


def can_use_noalias[dst_rank: Int, src_rank: Int](
    dst: TensorLayout[dst_rank], dst_capacity: Int,
    dst_identity: StorageProvenance,
    src: TensorLayout[src_rank], src_capacity: Int,
    src_identity: StorageProvenance,
) -> Bool:
    return validate_noalias_preconditions(
        dst, dst_capacity, dst_identity, src, src_capacity, src_identity
    ) == TensorStatus.OK


def broadcast_write_is_safe[rank: Int](layout: TensorLayout[rank], capacity: Int) -> Bool:
    # Arbitrary elementwise writes must not share physical output addresses.
    # A separate explicit fill kernel may define specialized overlap-safe
    # semantics, but generic scatter/parallel writes cannot assume them.
    var proof = layout.validated_snapshot(capacity)
    if not proof.is_ok():
        return False
    return proof.payload.value().is_proven_unique()


# Capability-safe public aliases for the specification's writeability queries.
# The compiler must still enforce a mutable TensorMutView or mutable owner at
# the call site; these predicates never manufacture a mutable pointer.
def validate_writable[rank: Int](
    layout: TensorLayout[rank], storage_capacity: Int,
) -> TensorStatus:
    return validate_unique_writes[rank](layout, storage_capacity)


def is_writable[rank: Int](
    layout: TensorLayout[rank], storage_capacity: Int,
) -> Bool:
    return validate_writable[rank](layout, storage_capacity) == TensorStatus.OK
