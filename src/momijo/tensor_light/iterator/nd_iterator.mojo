# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR ITERATION / N-D ITERATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Iteration
# Module:        momijo.tensor_light.iterator.nd_iterator
# Source:        src/momijo/tensor_light/iterator/nd_iterator.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines n-d iteration in the tensor iteration layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `NDIndexIterator`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Traverse logical elements with validated strides and offsets.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `NDIndexIterator`.
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

# Fixed rank scalar index cursor, used for correctness fallback only.
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus

struct NDIndexIterator[rank: Int](Copyable):
    var coordinates: StaticRankArray[Self.rank]
    var shape: TensorShape[Self.rank]
    var produced: Int

    def __init__(out self, shape: TensorShape[Self.rank]):
        self.coordinates = StaticRankArray[Self.rank](0)
        self.shape = shape
        self.produced = 0

    @staticmethod
    def checked(shape: TensorShape[Self.rank]) -> TensorResult[Self]:
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        if canonical.payload.value().numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        return TensorResult[Self](Self(canonical.payload.value()))

    def done(self) -> Bool:
        # Negative positions are invalid, not a valid first item.
        return self.produced < 0 or self.produced >= self.shape.numel()

    def current_indices_checked(self) -> TensorResult[StaticRankArray[Self.rank]]:
        # The iterator has publicly mutable fields. The checked public
        # observation boundary rejects a forged position or a coordinate
        # vector inconsistent with its C-order logical linear position.
        # Internal already-proven hot loops still use indices() directly.
        var verified = self.shape.checked_numel()
        if not verified.is_ok():
            return TensorResult[StaticRankArray[Self.rank]](verified.status)
        var total = verified.payload.value()
        if self.produced < 0 or self.produced >= total:
            return TensorResult[StaticRankArray[Self.rank]](TensorStatus.OUT_OF_BOUNDS)
        var linear = 0
        comptime for i in range(Self.rank):
            var extent = self.shape.dims.get(i)
            var coordinate = self.coordinates.get(i)
            if extent <= 0 or coordinate < 0 or coordinate >= extent:
                return TensorResult[StaticRankArray[Self.rank]](TensorStatus.OUT_OF_BOUNDS)
            # Each prefix product is at most total <= Int.MAX // 4.
            linear = linear * extent + coordinate
        if linear != self.produced:
            return TensorResult[StaticRankArray[Self.rank]](TensorStatus.INVALID_LAYOUT)
        return TensorResult[StaticRankArray[Self.rank]](self.coordinates.copy())

    def done_checked(self) -> TensorResult[Bool]:
        # Publicly mutable cursor metadata cannot claim EOF simply by
        # replacing the cached total or forging coordinates. The canonical
        # end sentinel has produced == numel and all logical coordinates 0.
        var count = self.shape.checked_numel()
        if not count.is_ok():
            return TensorResult[Bool](count.status)
        var total = count.payload.value()
        if self.produced < 0 or self.produced > total:
            return TensorResult[Bool](TensorStatus.OUT_OF_BOUNDS)
        if self.produced == total:
            comptime for i in range(Self.rank):
                if self.coordinates.get(i) != 0:
                    return TensorResult[Bool](TensorStatus.INVALID_LAYOUT)
            return TensorResult[Bool](True)
        var current = self.current_indices_checked()
        if not current.is_ok():
            return TensorResult[Bool](current.status)
        return TensorResult[Bool](False)

    def indices(self) -> StaticRankArray[Self.rank]:
        return self.coordinates.copy()

    def reset(mut self):
        self.coordinates = StaticRankArray[Self.rank](0)
        self.produced = 0

    def seek(mut self, linear_position: Int) -> TensorStatus:
        # Public checked random access to a logical C-order position. The
        # end sentinel is allowed; a zero-element shape has only that point.
        var canonical = self.shape.checked_numel()
        if not canonical.is_ok():
            return canonical.status
        var count = canonical.payload.value()
        if linear_position < 0 or linear_position > count:
            return TensorStatus.OUT_OF_BOUNDS
        self.seek_unchecked(linear_position)
        return TensorStatus.OK

    def seek_unchecked(mut self, linear_position: Int):
        # INTERNAL: the caller proved 0 <= linear_position <= numel and
        # canonical, nonnegative extents. O(rank), no walking from index 0;
        # used to begin CPU parallel shards at a disjoint output offset.
        self.produced = linear_position
        self.coordinates = StaticRankArray[Self.rank](0)
        if self.shape.numel() == 0:
            return
        var remaining = linear_position
        comptime for reverse_axis in range(Self.rank):
            var axis = Self.rank - 1 - reverse_axis
            var extent = self.shape.dims.get(axis)
            self.coordinates.set(axis, remaining % extent)
            remaining //= extent

    def advance(mut self):
        if self.done():
            return
        self.produced += 1
        comptime if Self.rank > 0:
            for i in range(Self.rank):
                var axis = Self.rank - 1 - i
                self.coordinates.set(axis, self.coordinates.get(axis) + 1)
                if self.coordinates.get(axis) < self.shape.dims.get(axis):
                    break
                self.coordinates.set(axis, 0)
