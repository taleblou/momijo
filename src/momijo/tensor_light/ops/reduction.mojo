# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / REDUCTIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Reductions
# Module:        momijo.tensor_light.ops.reduction
# Source:        src/momijo/tensor_light/ops/reduction.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines whole-tensor and axis-wise reductions, including supported extrema
#   and index results.
#
#   Public declarations include `sum_all`, `mean_all`, `prod_all`, `min_all`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `sum_all`, `mean_all`,
#     `prod_all`, `min_all`, `max_all` and related symbols.
#   - Normalize reduction axes and apply keepdims/result-shape rules.
#   - Compute extrema indices with the declared tie and empty-input behavior.
#   - Provide whole-tensor extrema index reductions.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate operand shape, dtype, and storage access.
#   - Preserve operation-specific broadcasting and result contracts.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid materializing broadcasts unless output storage needs it.
#   - Prefer specialized or fused paths only when semantics match.
#
# Maintainability:
#   - Keep public operations aligned with underlying kernel semantics.
#   - Cover edge cases and maintain relevant regression tests.
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

# Host baseline reduction. Empty sum == +0, empty mean is undefined.
from ..core.tensor import Tensor
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..shape.axis import normalize_axis
from ..storage.storage import TensorStorage
from ..storage.host_storage import HostStorage
from ..kernels.cpu.reduction import cpu_sum_contiguous
from std.sys.info import simd_width_of
from ..iterator.nd_iterator import NDIndexIterator
from ..internal.status import TensorStatus, TensorResult
from ..dtype.fp32 import fp32_minimum, fp32_maximum, fp32_argmin_prefers, fp32_argmax_prefers

def sum_all[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Float32]:
    if not input.impl.storage.is_host_accessible():
        return TensorResult[Float32](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if identical(S, HostStorage):
        return input.sum_all()
    # The contiguous SIMD shortcut relies on the FULL physical span proof.
    var verified = input._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[Float32](verified.status)
    var trusted = verified.payload.value()
    var total = Float32(0.0)
    var cursor = NDIndexIterator[rank](trusted.shape)
    while not cursor.done():
        var entry = input.impl.storage.read_host(trusted.physical_offset_unchecked(cursor.coordinates))
        if not entry.is_ok():
            return TensorResult[Float32](entry.status)
        total += entry.payload.value()
        cursor.advance()
    return TensorResult[Float32](total)

def mean_all[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Float32]:
    if not input.impl.storage.is_host_accessible():
        return TensorResult[Float32](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if identical(S, HostStorage):
        return input.mean_all()
    var count = input.checked_numel()
    if not count.is_ok():
        return TensorResult[Float32](count.status)
    if count.payload.value() == 0:
        return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
    var result = sum_all(input)
    if not result.is_ok():
        return result
    return TensorResult[Float32](result.payload.value() / Float32(count.payload.value()))


# HostStorage reductions are owned by the primary Tensor facade.
def prod_all[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Float32]:
    # The HostStorage facade performs its own single owner-bound proof and
    # selects a contiguous AOT FP32 leaf. Do not first build a second checked
    # layout/cursor here: it would be entirely discarded by this delegate.
    comptime if identical(S, HostStorage):
        return input.prod_all()
    if not input.impl.storage.is_host_accessible():
        return TensorResult[Float32](TensorStatus.BACKEND_UNAVAILABLE)
    var proof = input._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[Float32](proof.status)
    var layout = proof.payload.value()
    var product = Float32(1.0)
    var cursor = NDIndexIterator[rank](layout.shape)
    while not cursor.done():
        var entry = input.impl.storage.read_host(layout.physical_offset_unchecked(cursor.coordinates))
        if not entry.is_ok():
            return TensorResult[Float32](entry.status)
        product *= entry.payload.value()
        cursor.advance()
    return TensorResult[Float32](product)


def min_all[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Float32]:
    if not input.impl.storage.is_host_accessible():
        return TensorResult[Float32](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if identical(S, HostStorage):
        return input.min_all()
    var proof = input._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[Float32](proof.status)
    var trusted = proof.payload.value()
    if trusted.shape.numel() == 0:
        return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
    var cursor = NDIndexIterator[rank](trusted.shape)
    var first = input.impl.storage.read_host(trusted.physical_offset_unchecked(cursor.coordinates))
    if not first.is_ok():
        return first
    var best = first.payload.value()
    if best != best:
        return TensorResult[Float32](best)
    cursor.advance()
    while not cursor.done():
        var entry = input.impl.storage.read_host(trusted.physical_offset_unchecked(cursor.coordinates))
        if not entry.is_ok():
            return entry
        var value = entry.payload.value()
        if value != value:
            return TensorResult[Float32](value)
        best = fp32_minimum(best, value)
        cursor.advance()
    return TensorResult[Float32](best)


def max_all[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Float32]:
    if not input.impl.storage.is_host_accessible():
        return TensorResult[Float32](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if identical(S, HostStorage):
        return input.max_all()
    var proof = input._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[Float32](proof.status)
    var trusted = proof.payload.value()
    if trusted.shape.numel() == 0:
        return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
    var cursor = NDIndexIterator[rank](trusted.shape)
    var first = input.impl.storage.read_host(trusted.physical_offset_unchecked(cursor.coordinates))
    if not first.is_ok():
        return first
    var best = first.payload.value()
    if best != best:
        return TensorResult[Float32](best)
    cursor.advance()
    while not cursor.done():
        var entry = input.impl.storage.read_host(trusted.physical_offset_unchecked(cursor.coordinates))
        if not entry.is_ok():
            return entry
        var value = entry.payload.value()
        if value != value:
            return TensorResult[Float32](value)
        best = fp32_maximum(best, value)
        cursor.advance()
    return TensorResult[Float32](best)


def argmin_all[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Int]:
    if not input.impl.storage.is_host_accessible():
        return TensorResult[Int](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if identical(S, HostStorage):
        return input.argmin_all()
    var proof = input._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[Int](proof.status)
    var trusted = proof.payload.value()
    if trusted.shape.numel() == 0:
        return TensorResult[Int](TensorStatus.SIZE_MISMATCH)
    var cursor = NDIndexIterator[rank](trusted.shape)
    var first = input.impl.storage.read_host(trusted.physical_offset_unchecked(cursor.coordinates))
    if not first.is_ok():
        return TensorResult[Int](first.status)
    var best = first.payload.value()
    if best != best:
        return TensorResult[Int](0)
    var index = 1
    var best_index = 0
    cursor.advance()
    while not cursor.done():
        var entry = input.impl.storage.read_host(trusted.physical_offset_unchecked(cursor.coordinates))
        if not entry.is_ok():
            return TensorResult[Int](entry.status)
        var value = entry.payload.value()
        if value != value:
            return TensorResult[Int](index)
        if fp32_argmin_prefers(value, best):
            best = value
            best_index = index
        index += 1
        cursor.advance()
    return TensorResult[Int](best_index)


def argmax_all[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Int]:
    if not input.impl.storage.is_host_accessible():
        return TensorResult[Int](TensorStatus.BACKEND_UNAVAILABLE)
    comptime if identical(S, HostStorage):
        return input.argmax_all()
    var proof = input._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[Int](proof.status)
    var trusted = proof.payload.value()
    if trusted.shape.numel() == 0:
        return TensorResult[Int](TensorStatus.SIZE_MISMATCH)
    var cursor = NDIndexIterator[rank](trusted.shape)
    var first = input.impl.storage.read_host(trusted.physical_offset_unchecked(cursor.coordinates))
    if not first.is_ok():
        return TensorResult[Int](first.status)
    var best = first.payload.value()
    if best != best:
        return TensorResult[Int](0)
    var index = 1
    var best_index = 0
    cursor.advance()
    while not cursor.done():
        var entry = input.impl.storage.read_host(trusted.physical_offset_unchecked(cursor.coordinates))
        if not entry.is_ok():
            return TensorResult[Int](entry.status)
        var value = entry.payload.value()
        if value != value:
            return TensorResult[Int](index)
        if fp32_argmax_prefers(value, best):
            best = value
            best_index = index
        index += 1
        cursor.advance()
    return TensorResult[Int](best_index)


# Axis-reduction entry points keep the input rank statically known, with a
# singleton extent on the reduced axis. Callers wanting rank-1 use an explicit
# squeeze_axis_view after the materialization; no dynamic tensor rank exists.


def _reduce_axis_keepdims[operation: Int, rank: Int, S: TensorStorage](
    input: Tensor[rank, S], axis: Int
) -> TensorResult[Tensor[rank, HostStorage]]:
    comptime assert operation >= 0 and operation <= 4, "invalid FP32 reduction opcode"
    comptime if identical(S, HostStorage):
        return input._reduce_axis_keepdims_host[operation](axis)
    if not input.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    var input_layout = input._checked_current_layout()
    if not input_layout.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](input_layout.status)
    # Operation is a compile-time value already constrained by the assertion.
    var normalized = normalize_axis[rank](axis)
    if not normalized.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](normalized.status)
    var ax = normalized.payload.value()
    var reduce_extent = input_layout.payload.value().shape.dims.get(ax)
    var dims = input_layout.payload.value().shape.dims.copy()
    dims.set(ax, 1)
    var shaped = TensorShape[rank].checked(dims)
    if not shaped.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](shaped.status)
    # Empty extrema/mean fail only if an output value needs to be produced.
    # Other zero-sized axes may make the result empty even when reduce_extent
    # is zero; that is a valid zero-work reduction.
    if reduce_extent == 0 and operation >= 2 and shaped.payload.value().numel() > 0:
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var initial = Float32(0.0)
    if operation == 1:
        initial = Float32(1.0)
    var created = Tensor[rank, HostStorage]._empty_for_complete_overwrite_checked(shaped.payload.value())
    if not created.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](created.status)
    var result = created.payload.take()
    var source_layout = input_layout.payload.value()
    var output = result.impl.storage.borrow_mut_span()
    var output_index = 0
    var coordinates = NDIndexIterator[rank](shaped.payload.value())
    var reduced_pitch = source_layout.strides.get(ax)
    while not coordinates.done():
        var accumulator = initial
        var first = True
        var source_origin = source_layout.offset
        comptime for k in range(rank):
            source_origin += coordinates.coordinates.get(k) * source_layout.strides.get(k)
        for index in range(reduce_extent):
            var physical = source_origin + index * reduced_pitch
            var v = Float32(0.0)
            comptime if identical(S, HostStorage):
                # A borrowed Span is a lightweight view; no allocation and no
                # scalar TensorResult construction in the reduction hot loop.
                v = input.impl.storage.borrow_span()[physical]
            else:
                var loaded = input.impl.storage.read_host(physical)
                if not loaded.is_ok():
                    return TensorResult[Tensor[rank, HostStorage]](loaded.status)
                v = loaded.payload.value()
            comptime if operation == 0 or operation == 4:
                accumulator += v
            elif operation == 1:
                accumulator *= v
            else:
                if first:
                    accumulator = v
                    first = False
                else:
                    comptime if operation == 2:
                        accumulator = fp32_minimum(accumulator, v)
                    else:
                        accumulator = fp32_maximum(accumulator, v)
        comptime if operation == 4:
            accumulator /= Float32(reduce_extent)
        output[output_index] = accumulator
        output_index += 1
        coordinates.advance()
    return result._finish_complete_overwrite()


def sum_axis_keepdims[rank: Int, S: TensorStorage](input: Tensor[rank, S], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    return _reduce_axis_keepdims[0](input, axis)


def prod_axis_keepdims[rank: Int, S: TensorStorage](input: Tensor[rank, S], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    return _reduce_axis_keepdims[1](input, axis)


def min_axis_keepdims[rank: Int, S: TensorStorage](input: Tensor[rank, S], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    return _reduce_axis_keepdims[2](input, axis)


def max_axis_keepdims[rank: Int, S: TensorStorage](input: Tensor[rank, S], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    return _reduce_axis_keepdims[3](input, axis)


def mean_axis_keepdims[rank: Int, S: TensorStorage](input: Tensor[rank, S], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    return _reduce_axis_keepdims[4](input, axis)


# Public free-function compatibility. Real reduction and output ownership
# belong to struct Tensor; these wrappers introduce no parallel algorithm.
def sum_axis[rank: Int](input: Tensor[rank, HostStorage], axis: Int) -> TensorResult[Tensor[rank - 1, HostStorage]] where rank >= 1:
    return input.sum_axis(axis)


def prod_axis[rank: Int](input: Tensor[rank, HostStorage], axis: Int) -> TensorResult[Tensor[rank - 1, HostStorage]] where rank >= 1:
    return input.prod_axis(axis)


def min_axis[rank: Int](input: Tensor[rank, HostStorage], axis: Int) -> TensorResult[Tensor[rank - 1, HostStorage]] where rank >= 1:
    return input.min_axis(axis)


def max_axis[rank: Int](input: Tensor[rank, HostStorage], axis: Int) -> TensorResult[Tensor[rank - 1, HostStorage]] where rank >= 1:
    return input.max_axis(axis)


def mean_axis[rank: Int](input: Tensor[rank, HostStorage], axis: Int) -> TensorResult[Tensor[rank - 1, HostStorage]] where rank >= 1:
    return input.mean_axis(axis)


def sum_axes[rank: Int, axes_count: Int](input: Tensor[rank, HostStorage], axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[rank - axes_count, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.sum_axes[axes_count](axes)


def prod_axes[rank: Int, axes_count: Int](input: Tensor[rank, HostStorage], axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[rank - axes_count, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.prod_axes[axes_count](axes)


def min_axes[rank: Int, axes_count: Int](input: Tensor[rank, HostStorage], axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[rank - axes_count, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.min_axes[axes_count](axes)


def max_axes[rank: Int, axes_count: Int](input: Tensor[rank, HostStorage], axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[rank - axes_count, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.max_axes[axes_count](axes)


def mean_axes[rank: Int, axes_count: Int](input: Tensor[rank, HostStorage], axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[rank - axes_count, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.mean_axes[axes_count](axes)

# Static-count axis sets: output rank is unchanged because keepdims is true.
# Duplicate and out-of-range axes are rejected before any allocation.
def sum_axes_keepdims[rank: Int, axes_count: Int](input: Tensor[rank, HostStorage], axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[rank, HostStorage]]:
    return input.sum_axes_keepdims[axes_count](axes)


def prod_axes_keepdims[rank: Int, axes_count: Int](input: Tensor[rank, HostStorage], axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[rank, HostStorage]]:
    return input.prod_axes_keepdims[axes_count](axes)


def min_axes_keepdims[rank: Int, axes_count: Int](input: Tensor[rank, HostStorage], axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[rank, HostStorage]]:
    return input.min_axes_keepdims[axes_count](axes)


def max_axes_keepdims[rank: Int, axes_count: Int](input: Tensor[rank, HostStorage], axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[rank, HostStorage]]:
    return input.max_axes_keepdims[axes_count](axes)


def mean_axes_keepdims[rank: Int, axes_count: Int](input: Tensor[rank, HostStorage], axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[rank, HostStorage]]:
    return input.mean_axes_keepdims[axes_count](axes)


# Rank-sensitive reduction flags are types, not runtime Bool values.
# The output rank must be known before AOT specialization. A runtime
# keepdims Bool cannot change Tensor[rank] into Tensor[rank - axes_count].
@fieldwise_init
struct KeepDims:
    pass


@fieldwise_init
struct DropDims:
    pass


# Axes-aware overloads reuse the owner's already-validated implementations.
# The two flag types select a deterministic output rank without a runtime
# branch or an additional reduction engine.
def sum[rank: Int, axes_count: Int](
    input: Tensor[rank, HostStorage],
    axes: StaticRankArray[axes_count],
    keepdims: KeepDims,
) -> TensorResult[Tensor[rank, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.sum_axes_keepdims[axes_count](axes)


def sum[rank: Int, axes_count: Int](
    input: Tensor[rank, HostStorage],
    axes: StaticRankArray[axes_count],
    keepdims: DropDims,
) -> TensorResult[Tensor[rank - axes_count, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.sum_axes[axes_count](axes)


def prod[rank: Int, axes_count: Int](
    input: Tensor[rank, HostStorage],
    axes: StaticRankArray[axes_count],
    keepdims: KeepDims,
) -> TensorResult[Tensor[rank, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.prod_axes_keepdims[axes_count](axes)


def prod[rank: Int, axes_count: Int](
    input: Tensor[rank, HostStorage],
    axes: StaticRankArray[axes_count],
    keepdims: DropDims,
) -> TensorResult[Tensor[rank - axes_count, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.prod_axes[axes_count](axes)


def min[rank: Int, axes_count: Int](
    input: Tensor[rank, HostStorage],
    axes: StaticRankArray[axes_count],
    keepdims: KeepDims,
) -> TensorResult[Tensor[rank, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.min_axes_keepdims[axes_count](axes)


def min[rank: Int, axes_count: Int](
    input: Tensor[rank, HostStorage],
    axes: StaticRankArray[axes_count],
    keepdims: DropDims,
) -> TensorResult[Tensor[rank - axes_count, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.min_axes[axes_count](axes)


def max[rank: Int, axes_count: Int](
    input: Tensor[rank, HostStorage],
    axes: StaticRankArray[axes_count],
    keepdims: KeepDims,
) -> TensorResult[Tensor[rank, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.max_axes_keepdims[axes_count](axes)


def max[rank: Int, axes_count: Int](
    input: Tensor[rank, HostStorage],
    axes: StaticRankArray[axes_count],
    keepdims: DropDims,
) -> TensorResult[Tensor[rank - axes_count, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.max_axes[axes_count](axes)


def mean[rank: Int, axes_count: Int](
    input: Tensor[rank, HostStorage],
    axes: StaticRankArray[axes_count],
    keepdims: KeepDims,
) -> TensorResult[Tensor[rank, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.mean_axes_keepdims[axes_count](axes)


def mean[rank: Int, axes_count: Int](
    input: Tensor[rank, HostStorage],
    axes: StaticRankArray[axes_count],
    keepdims: DropDims,
) -> TensorResult[Tensor[rank - axes_count, HostStorage]] where axes_count >= 0 and axes_count <= rank:
    return input.mean_axes[axes_count](axes)


# Exact documented top-level reduction names. Keep struct Tensor as the
# primary owner and reuse the existing all-elements reduction contracts.
# These are one-call delegates: no duplicate traversal, allocation or AOT
# dispatch policy is introduced by the functional spelling.
def sum[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Float32]:
    return sum_all(input)


def prod[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Float32]:
    return prod_all(input)


def min[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Float32]:
    return min_all(input)


def max[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Float32]:
    return max_all(input)


def mean[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Float32]:
    return mean_all(input)


def argmin[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Int]:
    return argmin_all(input)


def argmax[rank: Int, S: TensorStorage](input: Tensor[rank, S]) -> TensorResult[Int]:
    return argmax_all(input)


# Rank-preserving and rank-reducing index-valued axis reductions.
# Static Float32 Tensor cannot store Int, so IndexTensor is the appropriate
# typed auxiliary output. NaN first-match and signed-zero rules agree with
# argmin_all/argmax_all; tie-breaks keep the lowest logical index.
from ..indexing.index_tensor import IndexTensor, checked_index_tensor_nbytes
from std.collections import List


def _extreme_axis_indices[minimum: Bool, rank: Int, S: TensorStorage](
    input: Tensor[rank, S], axis: Int
) -> TensorResult[IndexTensor[rank]]:
    if not input.impl.storage.is_host_accessible():
        return TensorResult[IndexTensor[rank]](TensorStatus.BACKEND_UNAVAILABLE)
    var normalized = normalize_axis[rank](axis)
    if not normalized.is_ok():
        return TensorResult[IndexTensor[rank]](normalized.status)
    var checked = input._checked_current_layout()
    if not checked.is_ok():
        return TensorResult[IndexTensor[rank]](checked.status)
    var source = checked.payload.value()
    var ax = normalized.payload.value()
    var reduction_length = source.shape.dims.get(ax)
    var dimensions = source.shape.dims.copy()
    dimensions.set(ax, 1)
    var output_shape = TensorShape[rank].checked(dimensions)
    if not output_shape.is_ok():
        return TensorResult[IndexTensor[rank]](output_shape.status)
    # Zero-size other axes => zero outputs, even with a zero reduced axis.
    if reduction_length == 0 and output_shape.payload.value().numel() > 0:
        return TensorResult[IndexTensor[rank]](TensorStatus.SIZE_MISMATCH)
    # Reject an Int-index output whose total storage bytes overflow *before*
    # allocating its List; validating the source Float32 tensor is not enough.
    var index_bytes = checked_index_tensor_nbytes[rank](
        output_shape.payload.value()
    )
    if not index_bytes.is_ok():
        return TensorResult[IndexTensor[rank]](index_bytes.status)
    var output = List[Int](capacity=output_shape.payload.value().numel())
    var cursor = NDIndexIterator[rank](output_shape.payload.value())
    var step = source.strides.get(ax)
    while not cursor.done():
        var base = source.offset
        comptime for j in range(rank):
            base += cursor.coordinates.get(j) * source.strides.get(j)
        var first = Float32(0.0)
        if reduction_length > 0:
            comptime if identical(S, HostStorage):
                first = input.impl.storage.borrow_span()[base]
            else:
                var initial = input.impl.storage.read_host(base)
                if not initial.is_ok():
                    return TensorResult[IndexTensor[rank]](initial.status)
                first = initial.payload.value()
        var best = first
        var best_at = 0
        if best == best:
            for k in range(1, reduction_length):
                var at = base + k * step
                var candidate = Float32(0.0)
                comptime if identical(S, HostStorage):
                    candidate = input.impl.storage.borrow_span()[at]
                else:
                    var item = input.impl.storage.read_host(at)
                    if not item.is_ok():
                        return TensorResult[IndexTensor[rank]](item.status)
                    candidate = item.payload.value()
                if candidate != candidate:
                    best_at = k
                    break
                comptime if minimum:
                    if fp32_argmin_prefers(candidate, best):
                        best_at = k
                        best = candidate
                else:
                    if fp32_argmax_prefers(candidate, best):
                        best_at = k
                        best = candidate
        output.append(best_at)
        cursor.advance()
    return IndexTensor[rank].from_values_checked(output_shape.payload.value(), output)


def argmin_axis_keepdims[rank: Int, S: TensorStorage](
    input: Tensor[rank, S], axis: Int
) -> TensorResult[IndexTensor[rank]]:
    return _extreme_axis_indices[True](input, axis)


def argmax_axis_keepdims[rank: Int, S: TensorStorage](
    input: Tensor[rank, S], axis: Int
) -> TensorResult[IndexTensor[rank]]:
    return _extreme_axis_indices[False](input, axis)


def _drop_index_axis[rank: Int](
    var indices: IndexTensor[rank], axis: Int
) -> TensorResult[IndexTensor[rank - 1]] where rank >= 1:
    var normalized = normalize_axis[rank](axis)
    if not normalized.is_ok():
        return TensorResult[IndexTensor[rank - 1]](normalized.status)
    var source_shape = indices.shape()
    var dims = StaticRankArray[rank - 1](0)
    var out_axis = 0
    comptime for j in range(rank):
        if j != normalized.payload.value():
            dims.set(out_axis, source_shape.dims.get(j))
            out_axis += 1
    var output_shape = TensorShape[rank - 1].checked(dims)
    if not output_shape.is_ok():
        return TensorResult[IndexTensor[rank - 1]](output_shape.status)
    # The keepdims owner remains valid for the single metadata copy. Move its
    # index vector without changing order or touching the Float32 source.
    return IndexTensor[rank - 1].from_values_checked(output_shape.payload.value(), indices._values)


def argmin_axis[rank: Int, S: TensorStorage](
    input: Tensor[rank, S], axis: Int
) -> TensorResult[IndexTensor[rank - 1]] where rank >= 1:
    var kept = argmin_axis_keepdims(input, axis)
    if not kept.is_ok():
        return TensorResult[IndexTensor[rank - 1]](kept.status)
    return _drop_index_axis[rank](kept.payload.take(), axis)


def argmax_axis[rank: Int, S: TensorStorage](
    input: Tensor[rank, S], axis: Int
) -> TensorResult[IndexTensor[rank - 1]] where rank >= 1:
    var kept = argmax_axis_keepdims(input, axis)
    if not kept.is_ok():
        return TensorResult[IndexTensor[rank - 1]](kept.status)
    return _drop_index_axis[rank](kept.payload.take(), axis)


# KeepDims/DropDims are overload tokens because output rank is a type.
def argmin[rank: Int, S: TensorStorage](
    input: Tensor[rank, S], axis: Int, mode: KeepDims
) -> TensorResult[IndexTensor[rank]] where rank >= 1:
    return argmin_axis_keepdims(input, axis)


def argmin[rank: Int, S: TensorStorage](
    input: Tensor[rank, S], axis: Int, mode: DropDims
) -> TensorResult[IndexTensor[rank - 1]] where rank >= 1:
    return argmin_axis(input, axis)


def argmax[rank: Int, S: TensorStorage](
    input: Tensor[rank, S], axis: Int, mode: KeepDims
) -> TensorResult[IndexTensor[rank]] where rank >= 1:
    return argmax_axis_keepdims(input, axis)


def argmax[rank: Int, S: TensorStorage](
    input: Tensor[rank, S], axis: Int, mode: DropDims
) -> TensorResult[IndexTensor[rank - 1]] where rank >= 1:
    return argmax_axis(input, axis)
