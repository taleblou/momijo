# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR ITERATION / ITERATORS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Iteration
# Module:        momijo.tensor_light.iterator.tensor_iterator
# Source:        src/momijo/tensor_light/iterator/tensor_iterator.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor iterators in the tensor iteration layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `TensorIterator2`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Traverse logical elements with validated strides and offsets.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `TensorIterator2`.
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

# A rank-static, exactly-two-input traversal facade for host views.
# Optimized AOT variants specialize contiguous runs in kernel modules.
from ..view.view import TensorView
from ..shape.shape_storage import StaticRankArray
from ..iterator.broadcast_iterator import BroadcastCursor
from ..internal.status import TensorResult, TensorStatus

struct TensorIterator2[rank: Int, left_origin: Origin, right_origin: Origin]:
    var lhs: TensorView[Self.rank, Self.left_origin]
    var rhs: TensorView[Self.rank, Self.right_origin]
    var cursor: BroadcastCursor[Self.rank]

    def __init__(out self, lhs: TensorView[Self.rank, Self.left_origin], rhs: TensorView[Self.rank, Self.right_origin], cursor: BroadcastCursor[Self.rank]):
        self.lhs = lhs.copy()
        self.rhs = rhs.copy()
        self.cursor = cursor.copy()

    @staticmethod
    def checked(lhs: TensorView[Self.rank, Self.left_origin], rhs: TensorView[Self.rank, Self.right_origin]) -> TensorResult[Self]:
        # The public TensorView can carry modified layout fields. Recover
        # canonical checked snapshots before constructing the iteration plan.
        var left_proof = lhs._checked_current_layout()
        if not left_proof.is_ok():
            return TensorResult[Self](left_proof.status)
        var right_proof = rhs._checked_current_layout()
        if not right_proof.is_ok():
            return TensorResult[Self](right_proof.status)
        # The two layouts were fully reconstructed against their live
        # borrowed allocations. Construct canonical views directly; a
        # second from_span_checked() would redo the same span proof.
        var canonical_left = TensorView[Self.rank, Self.left_origin](
            lhs.data, left_proof.payload.take(),
        )
        var canonical_right = TensorView[Self.rank, Self.right_origin](
            rhs.data, right_proof.payload.take(),
        )
        var traversal = BroadcastCursor[Self.rank].checked(
            canonical_left.layout.shape, canonical_right.layout.shape,
        )
        if not traversal.is_ok():
            return TensorResult[Self](traversal.status)
        return TensorResult[Self](Self(
            canonical_left, canonical_right, traversal.payload.take(),
        ))

    def _checked_cursor_mapping(self) -> TensorStatus:
        # TensorIterator2 and BroadcastCursor expose mutable value fields.
        # A caller can replace a valid broadcast shape/cursor after checked()
        # construction while keeping every raw index inside physical bounds.
        # That must not silently change WHICH logical element is returned by
        # a public checked read. Re-establish the two borrowed owner layouts
        # and the exact expected broadcast output shape at this boundary.
        # Internal AOT loops use their sealed traversal plan instead.
        var left_layout = self.lhs._checked_current_layout()
        if not left_layout.is_ok():
            return left_layout.status
        var right_layout = self.rhs._checked_current_layout()
        if not right_layout.is_ok():
            return right_layout.status
        var expected = BroadcastCursor[Self.rank].checked(
            left_layout.payload.value().shape,
            right_layout.payload.value().shape,
        )
        if not expected.is_ok():
            return expected.status
        if (not self.cursor.left.same_extents(left_layout.payload.value().shape)
            or not self.cursor.right.same_extents(right_layout.payload.value().shape)
            or not self.cursor.out.shape.same_extents(expected.payload.value().out.shape)):
            return TensorStatus.INVALID_LAYOUT
        return TensorStatus.OK

    def done_checked(self) -> TensorResult[Bool]:
        # A Bool alone cannot explain whether traversal ended naturally or
        # the public cursor/layout was changed after checked construction.
        var status = self._checked_cursor_mapping()
        if status != TensorStatus.OK:
            return TensorResult[Bool](status)
        # Delegate the live coordinate/EOF-sentinel proof to the one
        # canonical rank-static index cursor rather than trusting done().
        return self.cursor.out.done_checked()

    def done(self) -> Bool:
        # Backward-compatible fail-closed Boolean convenience query.
        # Call done_checked() to distinguish invalid metadata from EOF.
        var status = self.done_checked()
        if not status.is_ok():
            return True
        return status.payload.value()

    def left_value(self) -> TensorResult[Float32]:
        var mapping = self._checked_cursor_mapping()
        if mapping != TensorStatus.OK:
            return TensorResult[Float32](mapping)
        # A consumed (or manually corrupted) cursor must never read the first
        # element after the final iteration has wrapped its coordinates.
        var current = self.cursor.left_indices_checked()
        if not current.is_ok():
            return TensorResult[Float32](current.status)
        return self.lhs.get(current.payload.value())

    def right_value(self) -> TensorResult[Float32]:
        var mapping = self._checked_cursor_mapping()
        if mapping != TensorStatus.OK:
            return TensorResult[Float32](mapping)
        var current = self.cursor.right_indices_checked()
        if not current.is_ok():
            return TensorResult[Float32](current.status)
        return self.rhs.get(current.payload.value())

    def output_coordinates_checked(self) -> TensorResult[StaticRankArray[Self.rank]]:
        var mapping = self._checked_cursor_mapping()
        if mapping != TensorStatus.OK:
            return TensorResult[StaticRankArray[Self.rank]](mapping)
        return self.cursor.out.current_indices_checked()

    def output_coordinates(self) -> StaticRankArray[Self.rank]:
        # Unchecked internal accessor. Call output_coordinates_checked() at
        # untrusted API boundaries; callers in proven traversal loops may
        # avoid the extra rank-static verification.
        return self.cursor.out.coordinates.copy()

    def advance_checked(mut self) -> TensorStatus:
        # Cursor state is externally mutable. Verify the mapping, canonical
        # cached element count, linear position and coordinates before moving.
        # An already-consumed (valid) cursor remains consumed: EOF is an
        # idempotent transition, NOT a malformed-access error.
        var state = self.done_checked()
        if not state.is_ok():
            return state.status
        if state.payload.value():
            return TensorStatus.OK
        # The checked observation above proved every live coordinate. Do
        # not rebuild that proof inside an already-validated cursor step.
        self.cursor._advance_unchecked()
        return TensorStatus.OK

    def advance(mut self):
        # Safe convenience surface: failed validation does NOT move either
        # operand's cursor. Kernels with a sealed traversal plan use the
        # explicit internal _advance_unchecked() method below.
        var status = self.advance_checked()

    def _advance_unchecked(mut self):
        # INTERNAL ONLY: shape/cursor/owner must have been checked by caller.
        self.cursor._advance_unchecked()
