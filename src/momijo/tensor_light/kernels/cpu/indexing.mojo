# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | CPU KERNEL / INDEXING OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / CPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.cpu.indexing
# Source:        src/momijo/tensor_light/kernels/cpu/indexing.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines indexing operations for the CPU tensor kernel layer.
#
#   Public declarations include `cpu_gather_kernel`, `cpu_scatter_kernel`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `cpu_gather_kernel`,
#     `cpu_scatter_kernel`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Respect the validated layout, bounds, and dtype contract.
#   - Protect scalar remainder and noncontiguous cases.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Use vectorized/parallel paths only when preconditions hold.
#   - Avoid repeated dispatch decisions in per-element loops.
#
# Maintainability:
#   - Separate launch/ABI declarations from verified execution.
#   - Benchmark against stable, correct reference paths.
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

# One rank-static FP32 CPU gather/scatter kernel substrate.
# The owning Tensor checks lifetime and initialization. Named kernels also
# revalidate their raw Span/layout/axis/index arguments before pointer math
# so direct calls cannot bypass address bounds or duplicate-scatter safety.
# Foreign host-pointer authenticity remains an explicit unsafe caller duty.
from std.collections import Span
from ...shape.shape import TensorShape
from ...layout.layout import TensorLayout
from ...iterator.nd_iterator import NDIndexIterator
from ...internal.status import TensorStatus
from ...parallel.parallel_for import CpuWorkPolicy, recommend_worker_count, execute_parallel_shards
from ...alias.overlap import host_output_disjoint_from_input, host_output_disjoint_from_indices


def cpu_gather_kernel[
    rank: Int, src_origin: Origin[mut=False], dst_origin: Origin[mut=True], index_origin: Origin[mut=False],
](
    source: Span[mut=False, Float32, src_origin],
    destination: Span[mut=True, Float32, dst_origin],
    mapping: TensorLayout[rank], output_shape: TensorShape[rank],
    selected_axis: Int, indices: Span[mut=False, Int, index_origin],
) -> TensorStatus:
    # A named CPU kernel is a callable boundary, not permission to trust a
    # caller's raw axis, mutable layout caches, lengths or index values.
    # The owning Tensor has already checked these on its normal path; these
    # guards also keep direct kernel use from reaching an unchecked pointer.
    if selected_axis < 0 or selected_axis >= rank:
        return TensorStatus.INVALID_AXIS
    var src_proof = mapping.validated_snapshot(len(source))
    if not src_proof.is_ok():
        return src_proof.status
    var shape_proof = output_shape.checked_numel()
    if not shape_proof.is_ok():
        return shape_proof.status
    var input_layout = src_proof.payload.value()
    var count = shape_proof.payload.value()
    if count != len(destination):
        return TensorStatus.SIZE_MISMATCH
    comptime for axis in range(rank):
        if axis == selected_axis:
            if output_shape.dims.get(axis) != len(indices):
                return TensorStatus.SIZE_MISMATCH
        elif output_shape.dims.get(axis) != input_layout.shape.dims.get(axis):
            return TensorStatus.SHAPE_MISMATCH
    var extent = input_layout.shape.dims.get(selected_axis)
    for i in range(len(indices)):
        var at = indices[i]
        if at < -extent or at >= extent:
            return TensorStatus.OUT_OF_BOUNDS
    if not host_output_disjoint_from_input(destination, source):
        return TensorStatus.ALIAS_UNKNOWN
    if not host_output_disjoint_from_indices(destination, indices):
        return TensorStatus.ALIAS_UNKNOWN
    if count == 0:
        return TensorStatus.OK
    var pitch = input_layout.strides.get(selected_axis)
    var steps = input_layout.strides.copy()
    steps.set(selected_axis, 0)
    var policy = CpuWorkPolicy(131072, 8)
    var workers = recommend_worker_count(count, policy)
    if workers > 1:
        var src_ptr = source.unsafe_ptr()
        var dst_ptr = destination.unsafe_ptr()
        var index_ptr = indices.unsafe_ptr()

        def gather_disjoint_shard(shard: Int) {imm}:
            var q = count // workers
            var rem = count % workers
            var first = shard * q + min(shard, rem)
            var last = first + q
            if shard < rem:
                last += 1
            var cursor = NDIndexIterator[rank](output_shape)
            cursor.seek_unchecked(first)
            for flat in range(first, last):
                var axis_position = cursor.coordinates.get(selected_axis)
                var at_axis = index_ptr.unsafe_load(axis_position)[0]
                if at_axis < 0:
                    at_axis += extent
                var physical = input_layout.offset + at_axis * pitch
                comptime for i in range(rank):
                    physical += cursor.coordinates.get(i) * steps.get(i)
                dst_ptr.unsafe_store[alignment=4](flat, src_ptr.unsafe_load[alignment=4](physical)[0])
                cursor.advance()

        return execute_parallel_shards(gather_disjoint_shard, count, policy)
    var cursor = NDIndexIterator[rank](output_shape)
    while not cursor.done():
        var at_axis = indices[cursor.coordinates.get(selected_axis)]
        if at_axis < 0:
            at_axis += extent
        var physical = input_layout.offset + at_axis * pitch
        comptime for i in range(rank):
            physical += cursor.coordinates.get(i) * steps.get(i)
        destination[cursor.produced] = source[physical]
        cursor.advance()
    return TensorStatus.OK


def cpu_scatter_kernel[
    rank: Int, src_origin: Origin[mut=False], dst_origin: Origin[mut=True], index_origin: Origin[mut=False],
](
    source: Span[mut=False, Float32, src_origin],
    destination: Span[mut=True, Float32, dst_origin],
    source_layout: TensorLayout[rank], destination_layout: TensorLayout[rank],
    selected_axis: Int, indices: Span[mut=False, Int, index_origin],
    strictly_monotonic_indices: Bool = False,
) -> TensorStatus:
    # A raw hint about monotonicity is not a write-disjointness proof.
    # Revalidate physical ownership, shape, axis and EVERY index before the
    # first store. This prevents a direct caller from forging the parallel
    # fast-path flag and racing duplicate destination indices.
    if selected_axis < 0 or selected_axis >= rank:
        return TensorStatus.INVALID_AXIS
    var src_proof = source_layout.validated_snapshot(len(source))
    if not src_proof.is_ok():
        return src_proof.status
    var dst_proof = destination_layout.validated_snapshot(len(destination))
    if not dst_proof.is_ok():
        return dst_proof.status
    var verified_src = src_proof.payload.value()
    var verified_dst = dst_proof.payload.value()
    if not verified_dst.is_proven_unique():
        return TensorStatus.OVERLAPPING
    comptime for axis in range(rank):
        if axis == selected_axis:
            if verified_src.shape.dims.get(axis) != len(indices):
                return TensorStatus.SIZE_MISMATCH
        elif verified_src.shape.dims.get(axis) != verified_dst.shape.dims.get(axis):
            return TensorStatus.SHAPE_MISMATCH
    if not host_output_disjoint_from_input(destination, source):
        return TensorStatus.ALIAS_UNKNOWN
    if not host_output_disjoint_from_indices(destination, indices):
        return TensorStatus.ALIAS_UNKNOWN
    var extent = verified_dst.shape.dims.get(selected_axis)
    var verified_monotonic = True
    var direction = 0
    var previous = 0
    for i in range(len(indices)):
        var at = indices[i]
        if at < -extent or at >= extent:
            return TensorStatus.OUT_OF_BOUNDS
        if at < 0:
            at += extent
        if i > 0:
            if at == previous:
                verified_monotonic = False
            elif direction == 0:
                if at > previous:
                    direction = 1
                else:
                    direction = -1
            elif (direction == 1 and at <= previous
                  or direction == -1 and at >= previous):
                verified_monotonic = False
        previous = at
    var count = verified_src.shape.numel()
    if count == 0:
        return TensorStatus.OK
    var pitch = verified_dst.strides.get(selected_axis)
    var steps = verified_dst.strides.copy()
    steps.set(selected_axis, 0)
    # The owning Tensor checked BOTH owner lifetime/non-alias and that the
    # normalized axis indices are strictly monotonic (thus distinct). If
    # duplicate or non-monotonic, stay sequential: last-write-wins semantics
    # must never be converted into a non-deterministic GPU/CPU race.
    var policy = CpuWorkPolicy(131072, 8)
    var workers = recommend_worker_count(count, policy)
    if strictly_monotonic_indices and verified_monotonic and workers > 1:
        var src_ptr = source.unsafe_ptr()
        var dst_ptr = destination.unsafe_ptr()
        var idx_ptr = indices.unsafe_ptr()

        def scatter_unique_shard(shard: Int) {imm}:
            var q = count // workers
            var rem = count % workers
            var first = shard * q + min(shard, rem)
            var last = first + q
            if shard < rem:
                last += 1
            var cursor = NDIndexIterator[rank](verified_src.shape)
            cursor.seek_unchecked(first)
            for flat in range(first, last):
                var selected = idx_ptr.unsafe_load(cursor.coordinates.get(selected_axis))[0]
                if selected < 0:
                    selected += extent
                var from_offset = verified_src.physical_offset_unchecked(cursor.coordinates)
                var to_offset = verified_dst.offset + selected * pitch
                comptime for i in range(rank):
                    to_offset += cursor.coordinates.get(i) * steps.get(i)
                dst_ptr.unsafe_store[alignment=4](to_offset, src_ptr.unsafe_load[alignment=4](from_offset)[0])
                cursor.advance()

        return execute_parallel_shards(scatter_unique_shard, count, policy)
    var cursor = NDIndexIterator[rank](verified_src.shape)
    while not cursor.done():
        var selected = indices[cursor.coordinates.get(selected_axis)]
        if selected < 0:
            selected += extent
        var from_offset = verified_src.physical_offset_unchecked(cursor.coordinates)
        var to_offset = verified_dst.offset + selected * pitch
        comptime for i in range(rank):
            to_offset += cursor.coordinates.get(i) * steps.get(i)
        destination[to_offset] = source[from_offset]
        cursor.advance()
    return TensorStatus.OK
