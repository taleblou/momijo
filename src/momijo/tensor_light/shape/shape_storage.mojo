# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR SHAPE / SHAPE STORAGE
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Shapes
# Module:        momijo.tensor_light.shape.shape_storage
# Source:        src/momijo/tensor_light/shape/shape_storage.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines shape storage in the tensor shapes layer of the FP32 Light tensor
#   implementation.
#
#   Public declarations include `StaticRankArray`, `make_static_rank_array`,
#   `logical_get`, `logical_set_internal`; preconditions and supported targets
#   are determined by their implementations.
#
# Responsibilities:
#   - Construct, normalize, or validate tensor shape metadata.
#   - Preserve documented behavior for shape compatibility, rank boundaries,
#     and empty dimensions.
#   - Provide source-declared interfaces such as `StaticRankArray`,
#     `make_static_rank_array`, `logical_get`, `logical_set_internal`,
#     `copy_logical`.
#   - Handle scalar, empty, and rank-dependent edge cases explicitly.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate extents, axes, rank, and element-count overflow.
#   - Preserve rank-zero and zero-sized-shape semantics.
#   - Handle scalar, rank-zero, and zero-length dimensions without silent
#     mismatch.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Keep shape arithmetic in metadata-only code.
#   - Avoid allocating shape arrays unnecessarily in hot paths.
#   - Reuse validated output-shape metadata before data traversal.
#
# Maintainability:
#   - Share one broadcasting and axis normalization policy.
#   - Test invalid ranks, mismatched shapes, and inferred dimensions.
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

from ..internal.status import TensorResult, TensorStatus
# Rank-0 has one physical metadata slot but no logical axes.
struct StaticRankArray[rank: Int](ImplicitlyCopyable):
    var _entries: Array[Int, max(Self.rank, 1)]

    def __init__(out self, value: Int = 0):
        comptime assert Self.rank >= 0 and Self.rank <= 8, "Self.rank must be 0..8"
        comptime if Self.rank == 0:
            # Rank-zero metadata is always the canonical zero padding, even
            # if a caller passes a nonzero constructor fill value.
            self._entries = Array[Int, max(Self.rank, 1)](fill=0)
        else:
            self._entries = Array[Int, max(Self.rank, 1)](fill=value)

    def __init__(out self, *, copy: Self):
        comptime if Self.rank == 0:
            # Copying a scalar cannot propagate manually tampered padding.
            self._entries = Array[Int, max(Self.rank, 1)](fill=0)
        else:
            self._entries = copy._entries.copy()

    def get(self, axis: Int) -> Int:
        # UNCHECKED metadata accessor for already-proven 0 <= axis < rank.
        # Rank zero has NO logical axis, even though fixed Array storage
        # requires a physical padding element. Its value is never observable
        # as metadata through this accessor; checked callers use get_checked.
        comptime if Self.rank == 0:
            return 0
        else:
            return self._entries[axis]

    def set(mut self, axis: Int, value: Int):
        # Constructor-time unchecked setter. Rank-zero padding is immutable
        # through the logical metadata interface, preventing a forged scalar
        # descriptor from leaking an artificial axis value.
        comptime if Self.rank > 0:
            self._entries[axis] = value

    def logical_length(self) -> Int:
        return Self.rank

    def get_checked(self, axis: Int) -> TensorResult[Int]:
        if axis < 0 or axis >= Self.rank:
            return TensorResult[Int](TensorStatus.INVALID_AXIS)
        return TensorResult[Int](self._entries[axis])

    def set_checked(mut self, axis: Int, value: Int) -> TensorStatus:
        if axis < 0 or axis >= Self.rank:
            return TensorStatus.INVALID_AXIS
        self._entries[axis] = value
        return TensorStatus.OK

# Public-facing metadata constructors do not expose the padded rank-0 slot.
def make_static_rank_array[rank: Int](values: Array[Int, max(rank, 1)]) -> StaticRankArray[rank]:
    comptime assert rank >= 0 and rank <= 8, "rank must be 0..8"
    var result = StaticRankArray[rank](0)
    comptime for i in range(rank):
        result.set(i, values[i])
    return result


def logical_get[rank: Int](values: StaticRankArray[rank], axis: Int) -> TensorResult[Int]:
    return values.get_checked(axis)


def logical_set_internal[rank: Int](mut values: StaticRankArray[rank], axis: Int, value: Int) -> TensorStatus:
    return values.set_checked(axis, value)


def copy_logical[rank: Int](values: StaticRankArray[rank]) -> StaticRankArray[rank]:
    var result = StaticRankArray[rank](0)
    comptime for i in range(rank):
        result.set(i, values.get(i))
    return result
