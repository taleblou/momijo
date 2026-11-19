# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR INDEXING / INDEX TENSOR HANDLING
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Indexing
# Module:        momijo.tensor_light.indexing.index_tensor
# Source:        src/momijo/tensor_light/indexing/index_tensor.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines index tensor handling in the tensor indexing layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `checked_index_tensor_nbytes`, `IndexTensor`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Normalize and validate index or slicing operations.
#   - Preserve documented behavior for bounds, strides, view provenance, and
#     write safety.
#   - Provide source-declared interfaces such as
#     `checked_index_tensor_nbytes`, `IndexTensor`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Normalize and bounds-check indices before memory access.
#   - Preserve documented negative-index and slicing semantics.
#   - Check index bounds before converting indices into physical offsets.
#
# Performance Objectives:
#   - Perform checked offset computation outside repeated element access.
#   - Avoid unnecessary index-tensor copies.
#   - Reuse normalized index ranges when the operation permits.
#
# Maintainability:
#   - Centralize slicing and index normalization.
#   - Test empty, negative-step, out-of-range, and high-rank cases.
#   - Test invalid indices, zero extents, and boundary conditions.
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

# Integer index output for Float32 extrema along a statically known rank.
# NOT a second generic numeric Tensor. It owns its compact rank-static shape
# and a flat Int index vector. A rank-0 output contains one scalar index.
from std.collections import List
from std.sys.info import size_of
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus


def checked_index_tensor_nbytes[rank: Int](
    shape: TensorShape[rank],
) -> TensorResult[Int]:
    # An IndexTensor stores MACHINE-SIZED Int, not the source Tensor's
    # four-byte Float32 payload. A perfectly valid FP32 shape can overflow
    # its index-output byte count on a 64-bit host. Qualify the allocation
    # BEFORE constructing a List with that (possibly enormous) capacity.
    # Shape checking is metadata-only and still supports ranks 0..8.
    var count = shape.checked_numel()
    if not count.is_ok():
        return TensorResult[Int](count.status)
    var index_bytes = size_of[Int]()
    if count.payload.value() > Int.MAX // index_bytes:
        return TensorResult[Int](TensorStatus.OVERFLOW)
    return TensorResult[Int](count.payload.value() * index_bytes)


struct IndexTensor[rank: Int](Movable):
    var _shape: TensorShape[Self.rank]
    var _values: List[Int]

    def __init__(out self, var shape: TensorShape[Self.rank], var values: List[Int]):
        self._shape = shape
        self._values = values

    @staticmethod
    def from_values_checked(
        shape: TensorShape[Self.rank], var values: List[Int]
    ) -> TensorResult[Self]:
        var bytes = checked_index_tensor_nbytes[Self.rank](shape)
        if not bytes.is_ok():
            return TensorResult[Self](bytes.status)
        if len(values) != shape.numel():
            return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
        return TensorResult[Self](Self(shape, values))

    def shape(self) -> TensorShape[Self.rank]:
        return self._shape.copy()

    def checked_numel(self) -> TensorResult[Int]:
        var n = self._shape.checked_numel()
        if not n.is_ok():
            return n
        if len(self._values) != n.payload.value():
            return TensorResult[Int](TensorStatus.INVALID_SHAPE)
        return n

    def get(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Int]:
        var count = self.checked_numel()
        if not count.is_ok():
            return count
        var linear = 0
        comptime for axis in range(Self.rank):
            var coordinate = indices.get(axis)
            var extent = self._shape.dims.get(axis)
            if coordinate < 0 or coordinate >= extent:
                return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
            linear = linear * extent + coordinate
        if linear >= count.payload.value():
            return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
        return TensorResult[Int](self._values[linear])
