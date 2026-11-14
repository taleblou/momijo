# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR ALIAS / ALIAS CLASSIFICATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Aliasing and Write Safety
# Module:        momijo.tensor_light.alias.alias_info
# Source:        src/momijo/tensor_light/alias/alias_info.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines alias classification in the aliasing and write safety layer of the
#   FP32 Light tensor implementation.
#
#   Public declarations include `AliasInfo`, `definitely_disjoint`,
#   `make_alias_info`, `may_alias`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Classify alias relations and enforce safe write preconditions.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `AliasInfo`,
#     `definitely_disjoint`, `make_alias_info`, `may_alias`,
#     `same_base_storage`.
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

# Provenance-aware physical range relation; conservative on unknown ownership.
from ..storage.ownership import StorageProvenance
from ..layout.layout import TensorLayout
from ..storage.host_storage import HostStorage
from ..internal.status import TensorResult, TensorStatus

@fieldwise_init
struct AliasInfo(Copyable):
    var provenance: StorageProvenance
    var offset_begin: Int
    var offset_end: Int
    var empty: Bool

def definitely_disjoint(a: AliasInfo, b: AliasInfo) -> Bool:
    # AliasInfo fields and StorageProvenance are public mutable metadata.
    # In particular `empty=True` can be forged for a NONEMPTY buffer: this
    # predicate must not grant `noalias` based on that caller-controlled flag.
    # Even identical or different integer allocation IDs are not owner proofs.
    # The verified-layout case of genuinely empty tensors is handled at the
    # checked owner boundary in alias/writeability.mojo, not through here.
    return False

# Construct a provenance-aware summary from a freshly validated allocation
# interval. Layout offsets are ALLOCATION-RELATIVE; never compare intervals
# from distinct or unknown allocations as if they shared one address origin.
def make_alias_info[rank: Int](
    identity: StorageProvenance, layout: TensorLayout[rank], capacity: Int
) -> TensorResult[AliasInfo]:
    var proof = layout.validated_snapshot(capacity)
    if not proof.is_ok():
        return TensorResult[AliasInfo](proof.status)
    var checked = proof.payload.value()
    return TensorResult[AliasInfo](
        AliasInfo(identity, checked.span_lo, checked.span_hi, checked.shape.is_empty())
    )


def may_alias(a: AliasInfo, b: AliasInfo) -> Bool:
    # UNKNOWN provenance conservatively MAY alias. Only a proven separate
    # allocation, or disjoint spans within the same allocation, authorizes
    # the no-alias fast path. Empty views never access payload elements.
    return not definitely_disjoint(a, b)


def same_base_storage(a: AliasInfo, b: AliasInfo) -> Bool:
    # Never turn two unknown external pointers into a false identity proof.
    return a.provenance.same_allocation(b.provenance)


# Internal owner-bound physical interval proof. BOTH layouts are interpreted
# against the SAME live HostStorage owner. Their validated allocation-relative
# element spans can therefore be compared; arbitrary numeric provenance IDs
# cannot provide this guarantee. Only a caller retaining the owner borrow and
# both checked snapshots may consume the result. This function does NOT
# manufacture a noalias Pointer nor grant lasting authority after mutation.
def _owner_bound_mapping_disjoint[left_rank: Int, right_rank: Int](
    ref owner: HostStorage,
    left: TensorLayout[left_rank],
    right: TensorLayout[right_rank],
) -> TensorResult[Bool]:
    var capacity = owner.capacity()
    if capacity < 0:
        return TensorResult[Bool](TensorStatus.INVALID_LAYOUT)
    var checked_left = left.validated_snapshot(capacity)
    if not checked_left.is_ok():
        return TensorResult[Bool](checked_left.status)
    var checked_right = right.validated_snapshot(capacity)
    if not checked_right.is_ok():
        return TensorResult[Bool](checked_right.status)
    var lhs = checked_left.payload.value()
    var rhs = checked_right.payload.value()
    # Empty shapes cannot read/write any elements, regardless of the offset.
    if lhs.shape.is_empty() or rhs.shape.is_empty():
        return TensorResult[Bool](True)
    # Both endpoints are VALIDATED inclusive offsets into exactly this owner.
    # Avoid computing hi+1 because it could overflow a signed Int.
    return TensorResult[Bool](
        lhs.span_hi < rhs.span_lo or rhs.span_hi < lhs.span_lo
    )


# A second owner-backed proof for two independently borrowed HostStorage
# allocations. These are live Mojo owner borrows, not caller-constructed
# provenance IDs or cached base addresses. It is valid even if both borrows
# refer to the same owner. A positive result proves their used physical byte
# intervals do not overlap; a negative result is deliberately conservative.
def _two_owner_mappings_disjoint[left_rank: Int, right_rank: Int](
    ref left_owner: HostStorage,
    left_layout: TensorLayout[left_rank],
    ref right_owner: HostStorage,
    right_layout: TensorLayout[right_rank],
) -> TensorResult[Bool]:
    var left_capacity = left_owner.capacity()
    var right_capacity = right_owner.capacity()
    if left_capacity < 0 or right_capacity < 0:
        return TensorResult[Bool](TensorStatus.INVALID_LAYOUT)
    var left_proof = left_layout.validated_snapshot(left_capacity)
    if not left_proof.is_ok():
        return TensorResult[Bool](left_proof.status)
    var right_proof = right_layout.validated_snapshot(right_capacity)
    if not right_proof.is_ok():
        return TensorResult[Bool](right_proof.status)
    var lhs = left_proof.payload.value()
    var rhs = right_proof.payload.value()
    if lhs.shape.is_empty() or rhs.shape.is_empty():
        return TensorResult[Bool](True)
    # Capacities are FP32 elements and every referenced index has already
    # been checked against its owner. Bounds are inclusive element offsets.
    # Convert to half-open byte intervals only after overflow checking.
    if (lhs.span_lo < 0 or rhs.span_lo < 0
        or lhs.span_hi < lhs.span_lo or rhs.span_hi < rhs.span_lo):
        return TensorResult[Bool](TensorStatus.INVALID_LAYOUT)
    if lhs.span_hi >= left_capacity or rhs.span_hi >= right_capacity:
        return TensorResult[Bool](TensorStatus.OUT_OF_BOUNDS)
    var left_address = Int(left_owner.borrow_span().unsafe_ptr())
    var right_address = Int(right_owner.borrow_span().unsafe_ptr())
    if left_address <= 0 or right_address <= 0:
        return TensorResult[Bool](False)
    # span_hi + 1 is representable because the validated capacity is at
    # most Int.MAX // 4. Guard integer pointer arithmetic separately.
    var left_after = (lhs.span_hi + 1) * 4
    var right_after = (rhs.span_hi + 1) * 4
    if left_address > Int.MAX - left_after:
        return TensorResult[Bool](False)
    if right_address > Int.MAX - right_after:
        return TensorResult[Bool](False)
    var left_start = left_address + lhs.span_lo * 4
    var left_end = left_address + left_after
    var right_start = right_address + rhs.span_lo * 4
    var right_end = right_address + right_after
    return TensorResult[Bool](left_end <= right_start or right_end <= left_start)
