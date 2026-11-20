# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | CPU KERNEL / TRANSFORMATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / CPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.cpu.transform
# Source:        src/momijo/tensor_light/kernels/cpu/transform.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor transformations for the CPU tensor kernel layer.
#
#   Public declarations include `cpu_transpose_copy2_strided`; preconditions
#   and supported targets are determined by their implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as
#     `cpu_transpose_copy2_strided`.
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

# Rank-2 FP32 blocked materializing transpose LEAF kernel.
# This named entry point independently checks shape, signed physical span,
# destination capacity and host byte-aliasing before its unchecked tile loops.
# It remains allocation-free and does not own/publish any Tensor storage.
from std.collections import Span
from ...internal.status import TensorStatus
from ...shape.shape import TensorShape
from ...shape.shape_storage import StaticRankArray
from ...layout.layout import TensorLayout
from ...alias.overlap import host_output_disjoint_from_input
from ...parallel.parallel_for import CpuWorkPolicy, recommend_worker_count, execute_parallel_shards


def cpu_transpose_copy2_strided[
    source_origin: Origin[mut=False], destination_origin: Origin[mut=True],
](
    source: Span[mut=False, Float32, source_origin],
    destination: Span[mut=True, Float32, destination_origin],
    rows: Int, columns: Int, offset: Int, row_stride: Int, column_stride: Int,
) -> TensorStatus:
    # Raw Span callers are not allowed to bypass the Tensor facade's
    # shape/layout proof. Reconstruct an exact rank-two signed-stride mapping
    # against the actual source Span BEFORE obtaining unsafe pointers.
    if rows < 0 or columns < 0:
        return TensorStatus.INVALID_SHAPE
    var extents = StaticRankArray[2](0)
    extents.set(0, rows)
    extents.set(1, columns)
    var checked_shape = TensorShape[2].checked(extents)
    if not checked_shape.is_ok():
        return checked_shape.status
    var count = checked_shape.payload.value().numel()
    if len(destination) != count:
        return TensorStatus.SIZE_MISMATCH
    var strides = StaticRankArray[2](0)
    strides.set(0, row_stride)
    strides.set(1, column_stride)
    var checked_layout = TensorLayout[2].checked(
        checked_shape.payload.value(), strides, offset, len(source),
    )
    if not checked_layout.is_ok():
        return checked_layout.status
    if count == 0:
        return TensorStatus.OK
    # The source may use negative/zero strides, but the physical target is
    # one dense exclusive C-order span. A shifted or exact alias is unsafe
    # with the tiled transpose traversal, including under CPU parallelism.
    if not host_output_disjoint_from_input(destination, source):
        return TensorStatus.ALIAS_UNKNOWN
    var input_pointer = source.unsafe_ptr()
    var output_pointer = destination.unsafe_ptr()
    comptime tile = 32
    # Column-block owners write disjoint contiguous output row bands.
    # Partition only when the validated output area is large enough to
    # amortize MAX worker dispatch; smaller transposes retain local tiling.
    if rows > 0 and columns > 0 and rows > 262144 // columns:
        var column_tiles = columns // tile
        if columns % tile != 0:
            column_tiles += 1
        var policy = CpuWorkPolicy(1, 8)
        var workers = recommend_worker_count(column_tiles, policy)
        if workers > 1:
            def transpose_tile_shard(shard: Int) {imm}:
                var q = column_tiles // workers
                var rem = column_tiles % workers
                var first = shard * q + min(shard, rem)
                var last = first + q
                if shard < rem:
                    last += 1
                for column_tile in range(first, last):
                    var column_block = column_tile * tile
                    var column_limit = min(columns, column_block + tile)
                    for row_block in range(0, rows, tile):
                        var row_limit = min(rows, row_block + tile)
                        for j in range(column_block, column_limit):
                            var destination_row = j * rows
                            var source_column = offset + j * column_stride
                            for i in range(row_block, row_limit):
                                output_pointer.unsafe_store(
                                    destination_row + i,
                                    input_pointer.unsafe_load(source_column + i * row_stride)[0],
                                )

            # Propagate any failed worker dispatch before the owning Tensor
            # can publish uninitialized/partially written output storage.
            return execute_parallel_shards(transpose_tile_shard, column_tiles, policy)
    for row_block in range(0, rows, tile):
        var row_limit = min(rows, row_block + tile)
        for column_block in range(0, columns, tile):
            var column_limit = min(columns, column_block + tile)
            for j in range(column_block, column_limit):
                var destination_row = j * rows
                var source_column = offset + j * column_stride
                for i in range(row_block, row_limit):
                    output_pointer.unsafe_store(
                        destination_row + i,
                        input_pointer.unsafe_load(source_column + i * row_stride)[0],
                    )
    return TensorStatus.OK
