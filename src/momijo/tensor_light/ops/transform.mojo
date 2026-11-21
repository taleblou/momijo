# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / TRANSFORMATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Transformations
# Module:        momijo.tensor_light.ops.transform
# Source:        src/momijo/tensor_light/ops/transform.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines structural tensor transformations such as concatenation,
#   splitting, stacking, or tiling.
#
#   Public declarations include `concatenate2`, `repeat_axis`, `tile_axis`,
#   `stack2`; preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `concatenate2`,
#     `repeat_axis`, `tile_axis`, `stack2`, `tile` and related symbols.
#   - Validate concatenation, stacking, splitting, and repetition shapes.
#   - Provide an explicit transpose-copy path where declared.
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

# Explicit materializing transforms. They never claim to be views.
from ..core.tensor import Tensor
from ..storage.storage import TensorStorage
from ..storage.host_storage import HostStorage
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..shape.axis import normalize_axis
from ..factory.creation import _complete_overwrite_output
from ..iterator.nd_iterator import NDIndexIterator
from ..view.view import TensorView
from ..internal.checked_math import checked_add, checked_mul_nonnegative
from ..internal.status import TensorResult, TensorStatus
from std.memory import unsafe_memcpy


def concatenate2[rank: Int, SA: TensorStorage, SB: TensorStorage](a: Tensor[rank, SA], b: Tensor[rank, SB], axis: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    comptime if SA == HostStorage and SB == HostStorage:
        comptime if rank >= 1:
            return a.concatenate(rebind[Tensor[rank, HostStorage]](b), axis)
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    var checked_a = a._checked_current_layout()
    if not checked_a.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked_a.status)
    var checked_b = b._checked_current_layout()
    if not checked_b.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked_b.status)
    var selected = normalize_axis[rank](axis)
    if not selected.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](selected.status)
    var ax = selected.payload.value()
    var dims = checked_a.payload.value().shape.dims.copy()
    comptime for i in range(rank):
        if i != ax and checked_a.payload.value().shape.dims.get(i) != checked_b.payload.value().shape.dims.get(i):
            return TensorResult[Tensor[rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var joined = checked_add(checked_a.payload.value().shape.dims.get(ax), checked_b.payload.value().shape.dims.get(ax))
    if not joined.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](joined.status)
    dims.set(ax, joined.payload.value())
    var shape = TensorShape[rank].checked(dims)
    if not shape.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](shape.status)
    var result = _complete_overwrite_output[rank](shape.payload.value())
    if not result.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](result.status)
    var out = result.payload.take()
    # Fast path: concatenation along axis 0 of two dense C-order owners
    # consists of two disjoint FP32 runs in the newly allocated output.
    comptime if SA == HostStorage and SB == HostStorage:
        if ax == 0 and checked_a.payload.value().is_c_contiguous() and checked_b.payload.value().is_c_contiguous():
            if checked_a.payload.value().is_exhaustive() and checked_b.payload.value().is_exhaustive():
                var src_a = rebind[HostStorage](a.impl.storage).borrow_span()
                var src_b = rebind[HostStorage](b.impl.storage).borrow_span()
                var dst = out.impl.storage.borrow_mut_span()
                var start_a = checked_a.payload.value().span_lo
                var start_b = checked_b.payload.value().span_lo
                var count_a = a.numel()
                var count_b = b.numel()
                # Fresh destination has no storage identity shared with either input.
                # Using bulk copies avoids Python-style scalar dispatch and allows
                # the platform's tuned memcpy implementation to move long runs.
                if count_a > 0:
                    var left_run = src_a[start_a:start_a + count_a]
                    var left_dst = dst[0:count_a]
                    unsafe_memcpy(dest=left_dst.unsafe_ptr(), src=left_run.unsafe_ptr(), count=count_a)
                if count_b > 0:
                    var right_run = src_b[start_b:start_b + count_b]
                    var right_dst = dst[count_a:count_a + count_b]
                    unsafe_memcpy(dest=right_dst.unsafe_ptr(), src=right_run.unsafe_ptr(), count=count_b)
                return (out^)._finish_complete_overwrite()
    var la = checked_a.payload.value()
    var lb = checked_b.payload.value()
    var destination = out.impl.storage.borrow_mut_span()
    # The newly-created C-order destination is a unique, checked allocation.
    # Inputs are independently owned and both layouts were verified above.
    var result_layout = out.impl.layout
    var cursor_a = NDIndexIterator[rank](la.shape)
    while not cursor_a.done():
        var physical = la.physical_offset_unchecked(cursor_a.coordinates)
        var value = a.impl.storage.read_host(physical)
        if not value.is_ok():
            return TensorResult[Tensor[rank, HostStorage]](value.status)
        destination[result_layout.physical_offset_unchecked(cursor_a.coordinates)] = value.payload.value()
        cursor_a.advance()
    var cursor_b = NDIndexIterator[rank](lb.shape)
    while not cursor_b.done():
        var physical = lb.physical_offset_unchecked(cursor_b.coordinates)
        var value = b.impl.storage.read_host(physical)
        if not value.is_ok():
            return TensorResult[Tensor[rank, HostStorage]](value.status)
        var target = cursor_b.coordinates.copy()
        target.set(ax, target.get(ax) + la.shape.dims.get(ax))
        destination[result_layout.physical_offset_unchecked(target)] = value.payload.value()
        cursor_b.advance()
    return (out^)._finish_complete_overwrite()


def repeat_axis[rank: Int, S: TensorStorage](src: Tensor[rank, S], axis: Int, copies: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    comptime if S == HostStorage:
        return src.repeat_axis(axis, copies)
    if not src.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    if copies < 0:
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.INVALID_SHAPE)
    # Validate arbitrary external layout metadata and its physical bounds
    # BEFORE output allocation or unchecked access to the source shape.
    var checked_source = src._checked_current_layout()
    if not checked_source.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked_source.status)
    var selected = normalize_axis[rank](axis)
    if not selected.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](selected.status)
    var a = selected.payload.value()
    var dims = checked_source.payload.value().shape.dims.copy()
    var new_extent = checked_mul_nonnegative(dims.get(a), copies)
    if not new_extent.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](new_extent.status)
    dims.set(a, new_extent.payload.value())
    var shape = TensorShape[rank].checked(dims)
    if not shape.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](shape.status)
    var result = _complete_overwrite_output[rank](shape.payload.value())
    if not result.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](result.status)
    var output = result.payload.take()
    var safe = checked_source.payload.value()
    var dst = output.impl.storage.borrow_mut_span()
    # The output has exactly the checked shape and a fresh owning allocation.
    # All source coordinates are mapped into the checked source extent.
    # For an empty output the cursor never enters the division below.
    var cursor = NDIndexIterator[rank](shape.payload.value())
    var linear = 0
    while not cursor.done():
        var src_coord = cursor.coordinates.copy()
        src_coord.set(a, src_coord.get(a) // copies)
        var read = src.impl.storage.read_host(safe.physical_offset_unchecked(src_coord))
        if not read.is_ok():
            return TensorResult[Tensor[rank, HostStorage]](read.status)
        dst[linear] = read.payload.value()
        linear += 1
        cursor.advance()
    return (output^)._finish_complete_overwrite()


def tile_axis[rank: Int, S: TensorStorage](src: Tensor[rank, S], axis: Int, copies: Int) -> TensorResult[Tensor[rank, HostStorage]]:
    comptime if S == HostStorage:
        return src.tile_axis(axis, copies)
    if not src.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    if copies < 0:
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.INVALID_SHAPE)
    # Validate arbitrary external layout metadata and its physical bounds
    # BEFORE output allocation or unchecked access to the source shape.
    var checked_source = src._checked_current_layout()
    if not checked_source.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](checked_source.status)
    var selected = normalize_axis[rank](axis)
    if not selected.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](selected.status)
    var a = selected.payload.value()
    var dims = checked_source.payload.value().shape.dims.copy()
    var new_extent = checked_mul_nonnegative(dims.get(a), copies)
    if not new_extent.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](new_extent.status)
    dims.set(a, new_extent.payload.value())
    var shape = TensorShape[rank].checked(dims)
    if not shape.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](shape.status)
    var allocated = _complete_overwrite_output[rank](shape.payload.value())
    if not allocated.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](allocated.status)
    var out = allocated.payload.take()
    var safe = checked_source.payload.value()
    var dst = out.impl.storage.borrow_mut_span()
    # A nonempty output implies a positive extent on the selected source axis,
    # so modulo never sees a zero divisor. Empty outputs perform no reads.
    var cursor = NDIndexIterator[rank](shape.payload.value())
    var linear = 0
    while not cursor.done():
        var src_coord = cursor.coordinates.copy()
        src_coord.set(a, src_coord.get(a) % safe.shape.dims.get(a))
        var read = src.impl.storage.read_host(safe.physical_offset_unchecked(src_coord))
        if not read.is_ok():
            return TensorResult[Tensor[rank, HostStorage]](read.status)
        dst[linear] = read.payload.value()
        linear += 1
        cursor.advance()
    return (out^)._finish_complete_overwrite()


# Public, explicitly materializing two-input stack. The inserted axis has
# compile-time output rank rank+1 while its position is normalized at the
# checked boundary. Supports scalar (rank 0) and empty FP32 inputs.
def stack2[rank: Int, SA: TensorStorage, SB: TensorStorage](
    a: Tensor[rank, SA], b: Tensor[rank, SB], axis: Int
) -> TensorResult[Tensor[rank + 1, HostStorage]] where rank >= 0 and rank < 8:
    comptime if SA == HostStorage and SB == HostStorage:
        return a.stack(rebind[Tensor[rank, HostStorage]](b), axis)
    if not a.impl.storage.is_host_accessible() or not b.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank + 1, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    var a_layout = a._checked_current_layout()
    if not a_layout.is_ok():
        return TensorResult[Tensor[rank + 1, HostStorage]](a_layout.status)
    var b_layout = b._checked_current_layout()
    if not b_layout.is_ok():
        return TensorResult[Tensor[rank + 1, HostStorage]](b_layout.status)
    if not a_layout.payload.value().shape.same_extents(b_layout.payload.value().shape):
        return TensorResult[Tensor[rank + 1, HostStorage]](TensorStatus.SIZE_MISMATCH)
    var normalized = normalize_axis[rank + 1](axis)
    if not normalized.is_ok():
        return TensorResult[Tensor[rank + 1, HostStorage]](normalized.status)
    var chosen_axis = normalized.payload.value()
    var target_dims = StaticRankArray[rank + 1](1)
    comptime for i in range(rank + 1):
        if i < chosen_axis:
            target_dims.set(i, a_layout.payload.value().shape.dims.get(i))
        elif i == chosen_axis:
            target_dims.set(i, 2)
        else:
            target_dims.set(i, a_layout.payload.value().shape.dims.get(i - 1))
    var output_shape = TensorShape[rank + 1].checked(target_dims)
    if not output_shape.is_ok():
        return TensorResult[Tensor[rank + 1, HostStorage]](output_shape.status)
    var allocated = _complete_overwrite_output[rank + 1](output_shape.payload.value())
    if not allocated.is_ok():
        return TensorResult[Tensor[rank + 1, HostStorage]](allocated.status)
    var output = allocated.payload.take()
    # Axis-0 stacking of C-contiguous inputs is a pair of sequential
    # physical copies. This also handles rank-0 scalars and empty arrays.
    comptime if SA == HostStorage and SB == HostStorage:
        if chosen_axis == 0 and a_layout.payload.value().is_c_contiguous() and b_layout.payload.value().is_c_contiguous():
            if a_layout.payload.value().is_exhaustive() and b_layout.payload.value().is_exhaustive():
                var src_a = rebind[HostStorage](a.impl.storage).borrow_span()
                var src_b = rebind[HostStorage](b.impl.storage).borrow_span()
                var dst = output.impl.storage.borrow_mut_span()
                var base_a = a_layout.payload.value().span_lo
                var base_b = b_layout.payload.value().span_lo
                var elements = a_layout.payload.value().shape.numel()
                # Stack output is a fresh owning allocation: its two output
                # segments are disjoint from each other and from the inputs.
                if elements > 0:
                    var left_run = src_a[base_a:base_a + elements]
                    var right_run = src_b[base_b:base_b + elements]
                    var left_dst = dst[0:elements]
                    var right_dst = dst[elements:2 * elements]
                    unsafe_memcpy(dest=left_dst.unsafe_ptr(), src=left_run.unsafe_ptr(), count=elements)
                    unsafe_memcpy(dest=right_dst.unsafe_ptr(), src=right_run.unsafe_ptr(), count=elements)
                return (output^)._finish_complete_overwrite()
    var la = a_layout.payload.value()
    var lb = b_layout.payload.value()
    var dst_layout = output.impl.layout
    var dst = output.impl.storage.borrow_mut_span()
    var cursor = NDIndexIterator[rank](la.shape)
    var output_coord = StaticRankArray[rank + 1](0)
    while not cursor.done():
        comptime for i in range(rank):
            if i < chosen_axis:
                output_coord.set(i, cursor.coordinates.get(i))
            else:
                output_coord.set(i + 1, cursor.coordinates.get(i))
        var left = a.impl.storage.read_host(la.physical_offset_unchecked(cursor.coordinates))
        if not left.is_ok():
            return TensorResult[Tensor[rank + 1, HostStorage]](left.status)
        var right = b.impl.storage.read_host(lb.physical_offset_unchecked(cursor.coordinates))
        if not right.is_ok():
            return TensorResult[Tensor[rank + 1, HostStorage]](right.status)
        output_coord.set(chosen_axis, 0)
        dst[dst_layout.physical_offset_unchecked(output_coord)] = left.payload.value()
        output_coord.set(chosen_axis, 1)
        dst[dst_layout.physical_offset_unchecked(output_coord)] = right.payload.value()
        cursor.advance()
    return (output^)._finish_complete_overwrite()


# Rank-static N-D tile convenience. The owning Tensor carries the only
# specialized HostStorage copy loop; generic host-visible storage is read
# conservatively with a checked per-element backend capability.
def tile[rank: Int, S: TensorStorage](src: Tensor[rank, S], reps: StaticRankArray[rank]) -> TensorResult[Tensor[rank, HostStorage]]:
    comptime if S == HostStorage:
        return src.tile(reps)
    if not src.impl.storage.is_host_accessible():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
    var proof = src._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](proof.status)
    var safe = proof.payload.value()
    var dims = safe.shape.dims.copy()
    comptime for i in range(rank):
        if reps.get(i) < 0:
            return TensorResult[Tensor[rank, HostStorage]](TensorStatus.INVALID_SHAPE)
        var extent = checked_mul_nonnegative(dims.get(i), reps.get(i))
        if not extent.is_ok():
            return TensorResult[Tensor[rank, HostStorage]](extent.status)
        dims.set(i, extent.payload.value())
    var shaped = TensorShape[rank].checked(dims)
    if not shaped.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](shaped.status)
    var created = _complete_overwrite_output[rank](shaped.payload.value())
    if not created.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](created.status)
    var output = created.payload.take()
    var dst = output.impl.storage.borrow_mut_span()
    var cursor = NDIndexIterator[rank](shaped.payload.value())
    while not cursor.done():
        var at = cursor.coordinates.copy()
        comptime for i in range(rank):
            at.set(i, at.get(i) % safe.shape.dims.get(i))
        var read = src.impl.storage.read_host(safe.physical_offset_unchecked(at))
        if not read.is_ok():
            return TensorResult[Tensor[rank, HostStorage]](read.status)
        dst[cursor.produced] = read.payload.value()
        cursor.advance()
    return (output^)._finish_complete_overwrite()

# Rank-static borrowed split/chunk interfaces. User code normally calls the
# owning Tensor methods; free functions are exact thin delegates. Every
# returned view is lifetime-bound to the original HostStorage allocation.


def split[rank: Int, parts: Int](
    ref src: Tensor[rank, HostStorage], axis: Int, sections: Array[Int, parts],
) -> TensorResult[Array[TensorView[rank, origin_of(src.impl.storage)], parts]]:
    comptime assert parts > 0, "split requires one or more static sections"
    return src.split_sections[parts](axis, sections)


def chunk[rank: Int, parts: Int](
    ref src: Tensor[rank, HostStorage], axis: Int,
) -> TensorResult[Array[TensorView[rank, origin_of(src.impl.storage)], parts]]:
    comptime assert parts > 0, "chunk requires one or more static output views"
    return src.chunk_fixed[parts](axis)

def repeat[rank: Int, S: TensorStorage](
    src: Tensor[rank, S], repeats: Int, axis: Int,
) -> TensorResult[Tensor[rank, HostStorage]]:
    # Public Tensor.repeat(repeats, axis) and the free helper share exactly
    # the same checked materialization path, with no secondary copy loop.
    return repeat_axis(src, axis, repeats)



# Public, explicit materializing rank-2 transpose. Unlike Tensor.transpose(),
# this creates a separately-owned initialized HostStorage tensor. The AOT
# CPU kernel owns the checked fast path; no algorithm is duplicated here.
def transpose_copy2[S: TensorStorage](
    source: Tensor[2, S],
) -> TensorResult[Tensor[2, HostStorage]]:
    return source.transpose_copy2()
