# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR ITERATION / STRIDED ITERATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Iteration
# Module:        momijo.tensor_light.iterator.strided_iterator
# Source:        src/momijo/tensor_light/iterator/strided_iterator.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines strided iteration in the tensor iteration layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `StridedCursor`, `StridedZipCursor`,
#   `make_strided_zip_cursor`, `source_offset`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Traverse logical elements with validated strides and offsets.
#   - Preserve documented behavior for bounds, strides, view provenance, and
#     write safety.
#   - Provide source-declared interfaces such as `StridedCursor`,
#     `StridedZipCursor`, `make_strided_zip_cursor`, `source_offset`,
#     `destination_offset` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Verify iteration bounds and storage offsets before traversal.
#   - Respect noncontiguous and broadcasted strides.
#
# Performance Objectives:
#   - Keep index advancement allocation-free where feasible.
#   - Specialize contiguous traversal without changing general semantics.
#
# Maintainability:
#   - Keep iterator behavior consistent with layout helpers.
#   - Cover rank-zero, empty, and high-rank traversal.
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

# No-allocation physical-offset cursors for validated FP32 layouts.
from ..layout.layout import TensorLayout
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus

struct StridedCursor[rank: Int]:
    var layout: TensorLayout[Self.rank]
    var coordinates: StaticRankArray[Self.rank]
    var visited: Int
    # Negative capacity means LOGICAL-ONLY: no public physical address can be
    # obtained from this cursor until it is bound to a real storage owner.
    var storage_capacity: Int

    def __init__(out self, layout: TensorLayout[Self.rank], storage_capacity: Int = -1):
        comptime assert Self.rank >= 0 and Self.rank <= 8, "StridedCursor rank must be 0..8"
        self.layout = layout
        self.coordinates = StaticRankArray[Self.rank](0)
        self.visited = 0
        self.storage_capacity = storage_capacity

    @staticmethod
    def checked_with_capacity(
        layout: TensorLayout[Self.rank], capacity: Int
    ) -> TensorResult[Self]:
        # Cross-module entry: rebuild the owner-relative bounds and all cached
        # overlap/shape proofs before creating an offset cursor. The cheap
        # direct constructor is for already-validated internal layouts only.
        var verified = layout.validated_snapshot(capacity)
        if not verified.is_ok():
            return TensorResult[Self](verified.status)
        return TensorResult[Self](Self(verified.payload.value(), capacity))

    def done_checked(self) -> TensorResult[Bool]:
        # A public Cursor exposes writable cached counts and coordinates.
        # Distinguish a *canonical* end sentinel from a forged one before a
        # caller decides whether to stop a logical or physical traversal.
        var count = self.layout.shape.checked_numel()
        if not count.is_ok():
            return TensorResult[Bool](count.status)
        if self.storage_capacity >= 0:
            var bound = self.layout.validated_snapshot(self.storage_capacity)
            if not bound.is_ok():
                return TensorResult[Bool](bound.status)
        var total = count.payload.value()
        if self.visited < 0 or self.visited > total:
            return TensorResult[Bool](TensorStatus.OUT_OF_BOUNDS)
        if self.visited == total:
            comptime for axis in range(Self.rank):
                if self.coordinates.get(axis) != 0:
                    return TensorResult[Bool](TensorStatus.INVALID_LAYOUT)
            return TensorResult[Bool](True)
        if not self._coordinates_match_visited():
            return TensorResult[Bool](TensorStatus.INVALID_LAYOUT)
        return TensorResult[Bool](False)

    def done(self) -> Bool:
        # Compatibility convenience: malformed public state fails closed.
        # Use done_checked() to distinguish an error from legitimate EOF.
        var state = self.done_checked()
        if not state.is_ok():
            return True
        return state.payload.value()

    def _coordinates_match_visited(self) -> Bool:
        # Public cursor fields can be edited independently. Never let a
        # forged coordinate vector authorize a longer physical run than the
        # remaining logical elements at `visited`. Numel <= MAX/4 prevents
        # overflow of this row-major linearization when indices are valid.
        if self.visited < 0 or self.visited >= self.layout.shape.numel():
            return False
        var linear = 0
        comptime for i in range(Self.rank):
            var extent = self.layout.shape.dims.get(i)
            var at = self.coordinates.get(i)
            if extent <= 0 or at < 0 or at >= extent:
                return False
            linear = linear * extent + at
        return linear == self.visited

    def offset(self) -> TensorResult[Int]:
        # A validated cursor cannot discard its physical storage-capacity
        # proof. The public metadata fields remain mutable, so revalidate the
        # live mapping before exposing a physical element address. Internal
        # hot kernels use their already-checked layout directly instead.
        if self.storage_capacity < 0:
            return TensorResult[Int](TensorStatus.NOT_REPRESENTABLE)
        var verified = self.layout.validated_snapshot(self.storage_capacity)
        if not verified.is_ok():
            return TensorResult[Int](verified.status)
        if not self._coordinates_match_visited():
            return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
        return verified.payload.value()._offset_from_validated(self.coordinates)

    def reset(mut self):
        self.coordinates = StaticRankArray[Self.rank](0)
        self.visited = 0

    def contiguous_run(self) -> Int:
        # This is an access/scheduling hint, not just a logical count. Refuse
        # to promise a physical run for an unbound or mutated cursor.
        if self.storage_capacity < 0:
            return 0
        var proof = self.layout.validated_snapshot(self.storage_capacity)
        if not proof.is_ok() or self.visited < 0:
            return 0
        var mapping = proof.payload.value()
        if not self._coordinates_match_visited():
            return 0
        var position = mapping._offset_from_validated(self.coordinates)
        if not position.is_ok():
            return 0
        if mapping.is_c_contiguous():
            return mapping.shape.numel() - self.visited
        comptime if Self.rank > 0:
            if mapping.strides.get(Self.rank - 1) == 1:
                return mapping.shape.dims.get(Self.rank - 1) - self.coordinates.get(Self.rank - 1)
        return 1

    def advance_checked(mut self) -> TensorStatus:
        # PUBLIC checked single step. Both coordinates and physical layout
        # fields are writable, so the Boolean done() query is not a proof of
        # a canonical cursor. Preserve the valid EOF as an idempotent step,
        # but reject a forged EOF or a corrupted live position unchanged.
        var count = self.layout.shape.checked_numel()
        if not count.is_ok():
            return count.status
        if self.visited == count.payload.value():
            return self.advance_run(0)
        return self.advance_run(1)

    def advance(mut self):
        # Compatibility entry point: never mutate malformed public metadata.
        # Call advance_checked() to receive the precise status. Optimized
        # kernels use their own sealed AOT index traversal rather than this
        # metadata-checking boundary.
        var status = self.advance_checked()

    def advance_run(mut self, count: Int) -> TensorStatus:
        # Validate position BEFORE subtraction. Publicly mutable cursor state
        # must not permit a negative/oversized visited count to underflow or
        # overflow the available-work calculation.
        var canonical = self.layout.shape.checked_numel()
        if not canonical.is_ok():
            return canonical.status
        var total = canonical.payload.value()
        if total < 0 or self.visited < 0 or self.visited > total:
            return TensorStatus.OUT_OF_BOUNDS
        if count < 0 or count > total - self.visited:
            return TensorStatus.OUT_OF_BOUNDS
        # A caller can alter public layout/coordinates independently of
        # visited. Refuse to silently turn a forged cursor into a valid
        # physical address merely by advancing it. This checked transition
        # is outside SIMD loops; hot kernels use their separately validated
        # rank-static traversal plan rather than a public cursor mutation.
        if self.storage_capacity >= 0:
            var bound = self.layout.validated_snapshot(self.storage_capacity)
            if not bound.is_ok():
                return bound.status
        if self.visited < total and not self._coordinates_match_visited():
            return TensorStatus.INVALID_LAYOUT
        if self.visited == total and count == 0:
            comptime for axis in range(Self.rank):
                if self.coordinates.get(axis) != 0:
                    return TensorStatus.INVALID_LAYOUT
            return TensorStatus.OK
        # O(rank), independent of count: derive the final C-order logical
        # coordinates directly, even for arbitrary physical strides.
        self.visited += count
        comptime if Self.rank > 0:
            var remaining = self.visited
            for reverse_axis in range(Self.rank):
                var axis = Self.rank - 1 - reverse_axis
                var extent = self.layout.shape.dims.get(axis)
                if extent == 0:
                    self.coordinates.set(axis, 0)
                else:
                    self.coordinates.set(axis, remaining % extent)
                    remaining //= extent
        return TensorStatus.OK


struct StridedZipCursor[src_rank: Int, dst_rank: Int]:
    var source: StridedCursor[Self.src_rank]
    var destination: StridedCursor[Self.dst_rank]

    def __init__(
        out self, src: TensorLayout[Self.src_rank], dst: TensorLayout[Self.dst_rank],
        src_capacity: Int = -1, dst_capacity: Int = -1,
    ):
        self.source = StridedCursor[Self.src_rank](src, src_capacity)
        self.destination = StridedCursor[Self.dst_rank](dst, dst_capacity)

    @staticmethod
    def checked(src: TensorLayout[Self.src_rank], dst: TensorLayout[Self.dst_rank]) -> TensorResult[Self]:
        # Without owner capacities this is a LOGICAL-ONLY cursor. Offset and
        # contiguous-run queries fail closed until checked_with_capacity().
        # A normal logical copy requires exactly matching shapes; an explicit
        # reshape/flatten must be requested separately, not inferred from a
        # coincidental equality of the products of unequal dimensions.
        comptime if Self.src_rank != Self.dst_rank:
            return TensorResult[Self](TensorStatus.SHAPE_MISMATCH)
        else:
            var src_shape = src.shape.checked_numel()
            if not src_shape.is_ok():
                return TensorResult[Self](src_shape.status)
            var dst_shape = dst.shape.checked_numel()
            if not dst_shape.is_ok():
                return TensorResult[Self](dst_shape.status)
            if not src.shape.same_extents(dst.shape):
                return TensorResult[Self](TensorStatus.SHAPE_MISMATCH)
            return TensorResult[Self](Self(src, dst))

    @staticmethod
    def checked_with_capacity(
        src: TensorLayout[Self.src_rank], src_capacity: Int,
        dst: TensorLayout[Self.dst_rank], dst_capacity: Int
    ) -> TensorResult[Self]:
        # Public/cross-module entry: independently re-establish both physical
        # layout bounds and cache invariants before creating an offset cursor.
        var source_proof = src.validated_snapshot(src_capacity)
        if not source_proof.is_ok():
            return TensorResult[Self](source_proof.status)
        var target_proof = dst.validated_snapshot(dst_capacity)
        if not target_proof.is_ok():
            return TensorResult[Self](target_proof.status)
        # Keep the independently verified capacities in each cursor. The
        # logical-only overload cannot authorize physical offset queries.
        var logical = Self.checked(source_proof.payload.value(), target_proof.payload.value())
        if not logical.is_ok():
            return logical
        return TensorResult[Self](Self(
            source_proof.payload.value(), target_proof.payload.value(),
            src_capacity, dst_capacity,
        ))

    def _pair_is_consistent(self) -> Bool:
        # Both cursor members are public mutable metadata. Pairwise offset
        # and fast-run queries are invalid after ANY mutation that changes
        # the logical shape or moves only one side of the zip.
        comptime if Self.src_rank != Self.dst_rank:
            return False
        else:
            if not self.source.layout.shape.same_extents(self.destination.layout.shape):
                return False
        # A public Shape can be modified after cursor construction. Compare
        # recomputed, overflow-checked counts rather than trusting cached
        # `elements` or accepting positions beyond the physical traversal.
        var src_count = self.source.layout.shape.checked_numel()
        if not src_count.is_ok():
            return False
        var dst_count = self.destination.layout.shape.checked_numel()
        if not dst_count.is_ok():
            return False
        if src_count.payload.value() != dst_count.payload.value():
            return False
        if self.source.visited != self.destination.visited:
            return False
        if self.source.visited < 0 or self.source.visited > src_count.payload.value():
            return False
        return True

    def done_checked(self) -> TensorResult[Bool]:
        # Source and destination are one logical traversal. A forged cursor
        # on EITHER side cannot be treated as an ordinary end-of-iteration.
        if not self._pair_is_consistent():
            return TensorResult[Bool](TensorStatus.SIZE_MISMATCH)
        var src = self.source.done_checked()
        if not src.is_ok():
            return TensorResult[Bool](src.status)
        var dst = self.destination.done_checked()
        if not dst.is_ok():
            return TensorResult[Bool](dst.status)
        if src.payload.value() != dst.payload.value():
            return TensorResult[Bool](TensorStatus.INVALID_LAYOUT)
        return TensorResult[Bool](src.payload.value())

    def done(self) -> Bool:
        var state = self.done_checked()
        if not state.is_ok():
            return True
        return state.payload.value()

    def source_offset(self) -> TensorResult[Int]:
        # A zip source address has meaning only at a mutually valid logical
        # position. A forged *destination* coordinate must not authorize
        # pairing this source address with a different destination element.
        var state = self.done_checked()
        if not state.is_ok():
            return TensorResult[Int](state.status)
        if state.payload.value():
            return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
        return self.source.offset()

    def destination_offset(self) -> TensorResult[Int]:
        var state = self.done_checked()
        if not state.is_ok():
            return TensorResult[Int](state.status)
        if state.payload.value():
            return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
        return self.destination.offset()

    def common_contiguous_run(self) -> Int:
        # Hints may never describe a nonzero physical run for an invalid or
        # consumed pair. Both sides must agree on their live coordinates.
        var state = self.done_checked()
        if not state.is_ok() or state.payload.value():
            return 0
        var first = self.source.contiguous_run()
        var second = self.destination.contiguous_run()
        if first < second:
            return first
        return second

    def advance_checked(mut self) -> TensorStatus:
        # A zip is one logical cursor. Neither side may advance unless BOTH
        # live coordinates, cached counts and storage-relative layouts have
        # passed the same preflight. A valid EOF is a no-op; malformed EOF
        # coordinates are an error, not silent completion.
        if not self._pair_is_consistent():
            return TensorStatus.SIZE_MISMATCH
        var count = self.source.layout.shape.checked_numel()
        if not count.is_ok():
            return count.status
        if self.source.visited == count.payload.value():
            return self.advance_run(0)
        return self.advance_run(1)

    def advance(mut self):
        # Safe legacy spelling, with no partial source/destination movement.
        # Call advance_checked() to observe an invalid metadata error.
        var status = self.advance_checked()

    def advance_run(mut self, count: Int) -> TensorStatus:
        # Transactional validation: neither cursor advances if either cursor
        # has invalid or externally modified position/shape metadata.
        # Independently validate both logical shapes before mutating EITHER
        # cursor. Otherwise the second advance could fail after the first has
        # already moved, violating the transactional zip contract.
        var source_shape = self.source.layout.shape.checked_numel()
        if not source_shape.is_ok():
            return source_shape.status
        var destination_shape = self.destination.layout.shape.checked_numel()
        if not destination_shape.is_ok():
            return destination_shape.status
        var src_count = source_shape.payload.value()
        var dst_count = destination_shape.payload.value()
        # Equal element counts are insufficient: a zip-copy of [2, 3] and
        # [3, 2] does not preserve logical indices. Recheck the shape (and
        # compile-time rank) at this mutable cursor boundary before either
        # position advances; offsets will separately prove storage capacity.
        comptime if Self.src_rank != Self.dst_rank:
            return TensorStatus.SHAPE_MISMATCH
        else:
            if not self.source.layout.shape.same_extents(self.destination.layout.shape):
                return TensorStatus.SHAPE_MISMATCH
        if src_count != dst_count or self.source.visited != self.destination.visited:
            return TensorStatus.SIZE_MISMATCH
        if self.source.visited < 0 or self.destination.visited < 0:
            return TensorStatus.OUT_OF_BOUNDS
        if self.source.visited > src_count or self.destination.visited > dst_count:
            return TensorStatus.OUT_OF_BOUNDS
        if count < 0 or count > src_count - self.source.visited:
            return TensorStatus.OUT_OF_BOUNDS
        # Both sides must pass the SAME physical-bounds and coordinate proof
        # before advancing either side. Without this preflight a valid first
        # cursor can move and a forged second layout can then fail, violating
        # the advertised all-or-nothing zipped cursor transition.
        if self.source.storage_capacity >= 0:
            var src_bound = self.source.layout.validated_snapshot(
                self.source.storage_capacity,
            )
            if not src_bound.is_ok():
                return src_bound.status
        if self.destination.storage_capacity >= 0:
            var dst_bound = self.destination.layout.validated_snapshot(
                self.destination.storage_capacity,
            )
            if not dst_bound.is_ok():
                return dst_bound.status
        if self.source.visited < src_count:
            if not self.source._coordinates_match_visited():
                return TensorStatus.INVALID_LAYOUT
            if not self.destination._coordinates_match_visited():
                return TensorStatus.INVALID_LAYOUT
        if count == 0 and self.source.visited == src_count:
            # Both end sentinels must be canonical before declaring a
            # successful no-op. Otherwise a mutated cursor is laundered.
            comptime for axis in range(Self.src_rank):
                if self.source.coordinates.get(axis) != 0:
                    return TensorStatus.INVALID_LAYOUT
            comptime for axis in range(Self.dst_rank):
                if self.destination.coordinates.get(axis) != 0:
                    return TensorStatus.INVALID_LAYOUT
            return TensorStatus.OK
        # After shared checks both independent advances have the same valid
        # count/position/bounds. Their checked implementations cannot reject
        # unless a different caller concurrently mutates this value, which
        # normal Mojo mutable-borrow rules prohibit.
        var first = self.source.advance_run(count)
        if first != TensorStatus.OK:
            return first
        return self.destination.advance_run(count)


# Documented functional cursor API. A two-layout construction is deliberately
# LOGICAL-ONLY: offsets cannot be observed until BOTH actual owner capacities
# are supplied. There is no unverified pointer or fabricated allocation ID.
def make_strided_zip_cursor[src_rank: Int, dst_rank: Int](
    source: TensorLayout[src_rank], destination: TensorLayout[dst_rank],
    source_capacity: Int = -1, destination_capacity: Int = -1,
) -> TensorResult[StridedZipCursor[src_rank, dst_rank]]:
    if source_capacity == -1 and destination_capacity == -1:
        return StridedZipCursor[src_rank, dst_rank].checked(source, destination)
    if source_capacity < 0 or destination_capacity < 0:
        return TensorResult[StridedZipCursor[src_rank, dst_rank]](
            TensorStatus.INVALID_LAYOUT
        )
    return StridedZipCursor[src_rank, dst_rank].checked_with_capacity(
        source, source_capacity, destination, destination_capacity,
    )


def source_offset[src_rank: Int, dst_rank: Int](
    cursor: StridedZipCursor[src_rank, dst_rank],
) -> TensorResult[Int]:
    return cursor.source_offset()


def destination_offset[src_rank: Int, dst_rank: Int](
    cursor: StridedZipCursor[src_rank, dst_rank],
) -> TensorResult[Int]:
    return cursor.destination_offset()


def common_contiguous_run[src_rank: Int, dst_rank: Int](
    cursor: StridedZipCursor[src_rank, dst_rank],
) -> Int:
    return cursor.common_contiguous_run()


def advance_zip[src_rank: Int, dst_rank: Int](
    mut cursor: StridedZipCursor[src_rank, dst_rank],
) -> TensorStatus:
    return cursor.advance_checked()


def advance_zip_run[src_rank: Int, dst_rank: Int](
    mut cursor: StridedZipCursor[src_rank, dst_rank], count: Int,
) -> TensorStatus:
    return cursor.advance_run(count)
