# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR ITERATION / BROADCAST ITERATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Iteration
# Module:        momijo.tensor_light.iterator.broadcast_iterator
# Source:        src/momijo/tensor_light/iterator/broadcast_iterator.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines broadcast iteration in the tensor iteration layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `BroadcastCursor`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Traverse logical elements with validated strides and offsets.
#   - Preserve documented behavior for shape compatibility, rank boundaries,
#     and empty dimensions.
#   - Provide source-declared interfaces such as `BroadcastCursor`.
#   - Handle scalar, empty, and rank-dependent edge cases explicitly.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Verify iteration bounds and storage offsets before traversal.
#   - Respect noncontiguous and broadcasted strides.
#   - Handle scalar, rank-zero, and zero-length dimensions without silent
#     mismatch.
#
# Performance Objectives:
#   - Keep index advancement allocation-free where feasible.
#   - Specialize contiguous traversal without changing general semantics.
#   - Reuse validated output-shape metadata before data traversal.
#
# Maintainability:
#   - Keep iterator behavior consistent with layout helpers.
#   - Cover rank-zero, empty, and high-rank traversal.
#   - Exercise incompatible dimensions and rank-limit regressions.
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

# Two-operand broadcast cursor. One immutable shape check at construction.
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..iterator.nd_iterator import NDIndexIterator
from ..internal.status import TensorStatus, TensorResult

struct BroadcastCursor[rank: Int](Copyable):
    var out: NDIndexIterator[Self.rank]
    var left: TensorShape[Self.rank]
    var right: TensorShape[Self.rank]

    def __init__(out self, output: TensorShape[Self.rank], left: TensorShape[Self.rank], right: TensorShape[Self.rank]):
        self.out = NDIndexIterator[Self.rank](output)
        self.left = left
        self.right = right

    @staticmethod
    def checked(left: TensorShape[Self.rank], right: TensorShape[Self.rank]) -> TensorResult[Self]:
        # TensorShape has mutable metadata; revalidate BOTH cached element
        # counts before any implicit stride-zero broadcast is computed.
        var checked_left = TensorShape[Self.rank].checked(left.dims)
        if not checked_left.is_ok():
            return TensorResult[Self](checked_left.status)
        var checked_right = TensorShape[Self.rank].checked(right.dims)
        if not checked_right.is_ok():
            return TensorResult[Self](checked_right.status)
        if checked_left.payload.value().numel() != left.numel() or checked_right.payload.value().numel() != right.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var dims = StaticRankArray[Self.rank](0)
        comptime for i in range(Self.rank):
            var a = checked_left.payload.value().dims.get(i)
            var b = checked_right.payload.value().dims.get(i)
            if a == b:
                dims.set(i, a)
            elif a == 1:
                dims.set(i, b)
            elif b == 1:
                dims.set(i, a)
            else:
                return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
        var shape = TensorShape[Self.rank].checked(dims)
        if not shape.is_ok():
            return TensorResult[Self](shape.status)
        return TensorResult[Self](Self(shape.payload.value(), checked_left.payload.value(), checked_right.payload.value()))

    def done_checked(self) -> TensorResult[Bool]:
        # A cursor's public fields can be replaced after checked().
        # Reconstruct the broadcast shape and reject altered cached counts,
        # incompatible operand extents, or forged C-order coordinates.
        var expected = Self.checked(self.left, self.right)
        if not expected.is_ok():
            return TensorResult[Bool](expected.status)
        if not self.out.shape.same_extents(expected.payload.value().out.shape):
            return TensorResult[Bool](TensorStatus.INVALID_SHAPE)
        # The checked ND cursor also verifies canonical EOF coordinates.
        # A forged end position must not silently skip pending tensor work.
        return self.out.done_checked()

    def done(self) -> Bool:
        # Fail closed rather than exposing an unchecked mutated cursor.
        var state = self.done_checked()
        if not state.is_ok():
            return True
        return state.payload.value()

    def left_indices_checked(self) -> TensorResult[StaticRankArray[Self.rank]]:
        # PUBLIC fallible observation. A caller may mutate `out`, `left` or
        # `right` after checked construction. Verify the broadcast relation,
        # canonical linear position and non-EOF coordinates BEFORE projecting
        # zero-stride axes. No payload is read by this metadata operation.
        var state = self.done_checked()
        if not state.is_ok():
            return TensorResult[StaticRankArray[Self.rank]](state.status)
        if state.payload.value():
            return TensorResult[StaticRankArray[Self.rank]](TensorStatus.OUT_OF_BOUNDS)
        # done_checked() already proved the exact current C-order coordinates.
        # Avoid reconstructing the same rank-static proof twice.
        var coords = self.out.coordinates.copy()
        comptime for i in range(Self.rank):
            if self.left.dims.get(i) == 1:
                coords.set(i, 0)
        return TensorResult[StaticRankArray[Self.rank]](coords)

    def right_indices_checked(self) -> TensorResult[StaticRankArray[Self.rank]]:
        var state = self.done_checked()
        if not state.is_ok():
            return TensorResult[StaticRankArray[Self.rank]](state.status)
        if state.payload.value():
            return TensorResult[StaticRankArray[Self.rank]](TensorStatus.OUT_OF_BOUNDS)
        # done_checked() already proved the exact current C-order coordinates.
        # Avoid reconstructing the same rank-static proof twice.
        var coords = self.out.coordinates.copy()
        comptime for i in range(Self.rank):
            if self.right.dims.get(i) == 1:
                coords.set(i, 0)
        return TensorResult[StaticRankArray[Self.rank]](coords)

    def left_indices(self) -> StaticRankArray[Self.rank]:
        # LEGACY unchecked metadata projection. Existing sealed AOT loops may
        # call this only after proving done_checked()==False. External users
        # should call left_indices_checked() to detect stale/mutated cursors.
        var coords = self.out.coordinates.copy()
        comptime for i in range(Self.rank):
            if self.left.dims.get(i) == 1:
                coords.set(i, 0)
        return coords

    def right_indices(self) -> StaticRankArray[Self.rank]:
        # Same internal-only precondition as left_indices().
        var coords = self.out.coordinates.copy()
        comptime for i in range(Self.rank):
            if self.right.dims.get(i) == 1:
                coords.set(i, 0)
        return coords

    def advance_checked(mut self) -> TensorStatus:
        var state = self.done_checked()
        if not state.is_ok():
            return state.status
        if state.payload.value():
            return TensorStatus.OK
        self._advance_unchecked()
        return TensorStatus.OK

    def advance(mut self):
        # Public path never advances a forged/mismatched broadcast cursor.
        var status = self.advance_checked()

    def _advance_unchecked(mut self):
        # INTERNAL: use only with a previously verified immutable traversal.
        self.out.advance()
