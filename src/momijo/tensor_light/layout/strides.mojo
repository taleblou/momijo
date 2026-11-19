# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR LAYOUT / STRIDES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Layout and Strides
# Module:        momijo.tensor_light.layout.strides
# Source:        src/momijo/tensor_light/layout/strides.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor strides in the tensor layout and strides layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `c_strides`, `f_strides`,
#   `make_c_contiguous_strides`, `make_f_contiguous_strides`; preconditions
#   and supported targets are determined by their implementations.
#
# Responsibilities:
#   - Compute and validate shape-to-storage layout mappings.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `c_strides`, `f_strides`,
#     `make_c_contiguous_strides`, `make_f_contiguous_strides`,
#     `validate_strides` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Preserve correct offsets and spans for arbitrary legal strides.
#   - Detect overflow, invalid extents, and internal overlap.
#
# Performance Objectives:
#   - Prefer metadata-only layout analysis.
#   - Cache or reuse validated stride computations where safe.
#
# Maintainability:
#   - Use consistent C/F contiguity terminology.
#   - Test negative, zero, and unusually large strides.
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

# Canonical contiguous strides in Float32 elements; dimensions are runtime.
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus
from ..internal.checked_math import checked_mul_nonnegative
from .offset import checked_relative_span
from ..shape.broadcast import broadcast_shape

def c_strides[rank: Int](shape: TensorShape[rank]) -> TensorResult[StaticRankArray[rank]]:
    # Same C-order stride rule as TensorLayout.c_contiguous, including empty
    # shapes with an early zero extent. Never overflow empty-shape products.
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok():
        return TensorResult[StaticRankArray[rank]](canonical.status)
    if canonical.payload.value().numel() != shape.numel():
        return TensorResult[StaticRankArray[rank]](TensorStatus.INVALID_SHAPE)
    var strides = StaticRankArray[rank](0)
    var step = 1
    comptime for j in range(rank):
        var i = rank - 1 - j
        strides.set(i, step)
        if step == 0 or shape.dims.get(i) == 0:
            step = 0
        else:
            var product = checked_mul_nonnegative(step, shape.dims.get(i))
            if not product.is_ok():
                if not shape.is_empty():
                    return TensorResult[StaticRankArray[rank]](product.status)
                step = 0
            else:
                step = product.payload.value()
    return TensorResult[StaticRankArray[rank]](strides)


def f_strides[rank: Int](shape: TensorShape[rank]) -> TensorResult[StaticRankArray[rank]]:
    # Exact F-order counterpart. Rank 0 has zero logical entries.
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok():
        return TensorResult[StaticRankArray[rank]](canonical.status)
    if canonical.payload.value().numel() != shape.numel():
        return TensorResult[StaticRankArray[rank]](TensorStatus.INVALID_SHAPE)
    var strides = StaticRankArray[rank](0)
    var step = 1
    comptime for i in range(rank):
        strides.set(i, step)
        if step == 0 or shape.dims.get(i) == 0:
            step = 0
        else:
            var product = checked_mul_nonnegative(step, shape.dims.get(i))
            if not product.is_ok():
                if not shape.is_empty():
                    return TensorResult[StaticRankArray[rank]](product.status)
                step = 0
            else:
                step = product.payload.value()
    return TensorResult[StaticRankArray[rank]](strides)


# Exact documented compatibility names; they delegate instead of recomputing.
def make_c_contiguous_strides[rank: Int](shape: TensorShape[rank]) -> TensorResult[StaticRankArray[rank]]:
    return c_strides[rank](shape)


def make_f_contiguous_strides[rank: Int](shape: TensorShape[rank]) -> TensorResult[StaticRankArray[rank]]:
    return f_strides[rank](shape)

def validate_strides[rank: Int](shape: TensorShape[rank], strides: StaticRankArray[rank]) -> TensorStatus:
    # A stride vector is independent of the storage origin. A negative
    # relative displacement is legal when the owning view later supplies
    # an offset that keeps its physical span inside the allocation.
    return checked_relative_span[rank](shape, strides).status


def stride[rank: Int](strides: StaticRankArray[rank], axis: Int) -> TensorResult[Int]:
    var normalized = axis
    if normalized < 0:
        if normalized < -rank:
            return TensorResult[Int](TensorStatus.INVALID_AXIS)
        normalized += rank
    return strides.get_checked(normalized)


def stride_unchecked[rank: Int](strides: StaticRankArray[rank], axis: Int) -> Int:
    # PRECONDITION: axis is in [0, rank); source metadata is validated.
    return strides.get(axis)


def permute_strides[rank: Int](strides: StaticRankArray[rank], axes: StaticRankArray[rank]) -> TensorResult[StaticRankArray[rank]]:
    var used = Array[Bool, 8](fill=False)
    var result = StaticRankArray[rank](0)
    comptime for i in range(rank):
        var axis = axes.get(i)
        if axis < 0 or axis >= rank:
            return TensorResult[StaticRankArray[rank]](TensorStatus.INVALID_AXIS)
        if used[axis]:
            return TensorResult[StaticRankArray[rank]](TensorStatus.INVALID_AXIS)
        used[axis] = True
        result.set(i, strides.get(axis))
    return TensorResult[StaticRankArray[rank]](result)


def insert_stride_for_unsqueeze[rank: Int, axis: Int](strides: StaticRankArray[rank]) -> StaticRankArray[rank + 1]:
    comptime assert rank >= 0 and rank < 8, "unsqueeze rank must be 0..7"
    comptime assert axis >= 0 and axis <= rank, "invalid inserted axis"
    # Singleton-axis strides have no effect on address mapping. Zero avoids
    # overflowing a computed stride for enormous empty tensor shapes.
    var output = StaticRankArray[rank + 1](0)
    comptime for i in range(rank + 1):
        comptime if i < axis:
            output.set(i, strides.get(i))
        elif i > axis:
            output.set(i, strides.get(i - 1))
    return output


def make_broadcast_strides[src_rank: Int, dst_rank: Int](source: TensorShape[src_rank], strides: StaticRankArray[src_rank], target: TensorShape[dst_rank]) -> TensorResult[StaticRankArray[dst_rank]]:
    comptime assert src_rank >= 0 and src_rank <= dst_rank and dst_rank <= 8, "unsupported broadcast rank"
    var checked_target = TensorShape[dst_rank].checked(target.dims)
    if not checked_target.is_ok():
        return TensorResult[StaticRankArray[dst_rank]](checked_target.status)
    if target.numel() != checked_target.payload.value().numel():
        return TensorResult[StaticRankArray[dst_rank]](TensorStatus.INVALID_SHAPE)
    var compatible = broadcast_shape[src_rank, dst_rank, dst_rank](source, target)
    if not compatible.is_ok():
        return TensorResult[StaticRankArray[dst_rank]](compatible.status)
    if not compatible.payload.value().same_extents(target):
        return TensorResult[StaticRankArray[dst_rank]](TensorStatus.SIZE_MISMATCH)
    var result = StaticRankArray[dst_rank](0)
    comptime for i in range(dst_rank):
        # A compile-time conditional cannot depend on a runtime var.
        comptime source_axis = i - (dst_rank - src_rank)
        comptime if source_axis >= 0:
            if source.dims.get(source_axis) == target.dims.get(i):
                result.set(i, strides.get(source_axis))
    return TensorResult[StaticRankArray[dst_rank]](result)
