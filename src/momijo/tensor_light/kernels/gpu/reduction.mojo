# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | GPU KERNEL / REDUCTIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / GPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.gpu.reduction
# Source:        src/momijo/tensor_light/kernels/gpu/reduction.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines reductions for the GPU tensor kernel layer.
#
#   Public declarations include `gpu_reduce_rows_fp32`,
#   `gpu_reduce_rows_op_fp32`, `gpu_scan_rows_op_fp32`,
#   `gpu_reduce_axis_strided_fp32`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `gpu_reduce_rows_fp32`,
#     `gpu_reduce_rows_op_fp32`, `gpu_scan_rows_op_fp32`,
#     `gpu_reduce_axis_strided_fp32`, `gpu_scan_axis_strided_fp32` and related
#     symbols.
#   - Reject unsupported target or unverified completion states.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Respect the validated layout, bounds, and dtype contract.
#   - Protect scalar remainder and noncontiguous cases.
#   - Reject unsupported or unauthenticated execution paths explicitly.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Use vectorized/parallel paths only when preconditions hold.
#   - Avoid repeated dispatch decisions in per-element loops.
#   - Avoid implicit host/device migration or compilation at execution time.
#
# Maintainability:
#   - Separate launch/ABI declarations from verified execution.
#   - Benchmark against stable, correct reference paths.
#   - Keep device-specific capability statements conservative.
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

# Each output lane independently reduces one contiguous row, avoiding atomics.
# AOT warp/block collective variants may replace this conservative baseline.

# Input and output have distinct logical shapes: [rows * cols] -> [rows].
# Their static MAX layouts must NOT be constrained to one common type.
# Each lane writes a different row. All column reads stay in that row.
from max.gpu import global_idx
from max.gpu.primitives import warp
from layout import TileTensor
from ...dtype.fp32 import fp32_minimum, fp32_maximum
from std.memory import Pointer

def gpu_reduce_rows_fp32[LO: AnyType, LI: AnyType](
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    inp: TileTensor[DType.float32, LI, MutAnyOrigin],
    rows_abi: Int32, cols_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var rows = Int(rows_abi)
    var cols = Int(cols_abi)
    var row = global_idx.x
    if row < rows:
        var total = Float32(0.0)
        # One base-offset multiply per row rather than one per column.
        var row_begin = row * cols
        for col in range(cols):
            total += inp[row_begin + col]
        out[row] = total


# Native FP32 one-writer-per-row reduction baseline. This extends the
# existing SUM_ROWS kernel without changing the FP32 Tensor payload dtype.
# operation 0=sum, 1=prod, 2=min, 3=max, 4=mean. All inputs are flat,
# row-major and disjoint from output. The caller rejects empty reductions
# for min/max/mean and validates rows*cols before kernel enqueue.
# Fast-tree/cooperative versions remain independent AOT variants.

def gpu_reduce_rows_op_fp32[operation: Int, LO: AnyType, LI: AnyType](
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    inp: TileTensor[DType.float32, LI, MutAnyOrigin],
    rows_abi: Int32, cols_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var rows = Int(rows_abi)
    var cols = Int(cols_abi)
    comptime assert operation >= 0 and operation <= 4, "FP32 row reduce opcode"
    var row = global_idx.x
    if row < rows:
        var base = row * cols
        comptime if operation == 0 or operation == 4:
            var accumulator = Float32(0.0)
            for col in range(cols):
                accumulator += inp[base + col]
            comptime if operation == 4:
                accumulator /= Float32(cols)
            out[row] = accumulator
        elif operation == 1:
            var product = Float32(1.0)
            for col in range(cols):
                product *= inp[base + col]
            out[row] = product
        else:
            # PRECONDITION: cols > 0 for min/max and one distinct output.
            var extreme = inp[base]
            for col in range(1, cols):
                comptime if operation == 2:
                    extreme = fp32_minimum(extreme, inp[base + col])
                else:
                    extreme = fp32_maximum(extreme, inp[base + col])
            out[row] = extreme

# Four fixed-FP32 row-wise inclusive scans. One thread owns exactly one
# contiguous row: every output is written once, in column order. This simple
# baseline is correct for row-major input/output, independently parameterized
# MAX layouts and all positive column counts; an AOT warp scan can replace
# it without changing the public ABI. No implicit host copy or dtype cast.
# opcode: 0 CUMSUM, 1 CUMPROD, 2 CUMMIN, 3 CUMMAX.
def gpu_scan_rows_op_fp32[operation: Int, LO: AnyType, LI: AnyType](
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    inp: TileTensor[DType.float32, LI, MutAnyOrigin],
    rows_abi: Int32, cols_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var rows = Int(rows_abi)
    var cols = Int(cols_abi)
    comptime assert operation >= 0 and operation <= 3, "invalid FP32 row scan opcode"
    var row = global_idx.x
    if row < rows and cols > 0:
        var start = row * cols
        var prefix = inp[start]
        out[start] = prefix
        for column in range(1, cols):
            var next = inp[start + column]
            comptime if operation == 0:
                prefix += next
            elif operation == 1:
                prefix *= next
            elif operation == 2:
                prefix = fp32_minimum(prefix, next)
            else:
                prefix = fp32_maximum(prefix, next)
            out[start + column] = prefix


# Generic fixed-FP32 axis-reduction FALLBACK for strided rank 1..8 tensors.
# Each GPU lane owns one output coordinate, with a size-one reduced axis.
# Unlike a row-only reduction, the selected input axis may have any signed
# stride; all non-reduced axes may also have arbitrary signed/zero strides.
# The output has a proven unique mapping, and no GPU atomics are necessary.
# For long contiguous reductions the row/warp-specialized AOT kernel wins.

def gpu_reduce_axis_strided_fp32[operation: Int, rank: Int, axis: Int](
    out: Pointer[Float32, MutUnsafeAnyOrigin],
    source: Pointer[Float32, MutUnsafeAnyOrigin],
    output_dims: SIMD[DType.int64, 8],
    out_steps: SIMD[DType.int64, 8],
    source_steps: SIMD[DType.int64, 8],
    out_origin: Int64, source_origin: Int64,
    output_count: Int64, reduction_size: Int64,
):
    comptime assert rank >= 1 and rank <= 8, "invalid FP32 GPU reduction rank"
    comptime assert axis >= 0 and axis < rank, "invalid FP32 GPU reduction axis"
    comptime assert operation >= 0 and operation <= 4, "invalid FP32 GPU reduction op"
    var flat = global_idx.x
    if flat >= Int(output_count):
        return
    var remainder = flat
    var physical_output = Int(out_origin)
    var physical_input = Int(source_origin)
    comptime for j in range(rank):
        var dim = rank - 1 - j
        var extent = Int(output_dims[dim])
        var coordinate = remainder % extent
        remainder //= extent
        physical_output += coordinate * Int(out_steps[dim])
        physical_input += coordinate * Int(source_steps[dim])
    var axis_step = Int(source_steps[axis])
    comptime if operation == 0 or operation == 4:
        var accumulator = Float32(0.0)
        for index in range(Int(reduction_size)):
            accumulator += source.unsafe_load(physical_input + index * axis_step)[0]
        comptime if operation == 4:
            accumulator /= Float32(Int(reduction_size))
        out.unsafe_store(physical_output, accumulator)
    elif operation == 1:
        var product = Float32(1.0)
        for index in range(Int(reduction_size)):
            product *= source.unsafe_load(physical_input + index * axis_step)[0]
        out.unsafe_store(physical_output, product)
    else:
        # PRECONDITION: the checked host boundary rejects empty extrema.
        var extreme = source.unsafe_load(physical_input)[0]
        for index in range(1, Int(reduction_size)):
            var value = source.unsafe_load(physical_input + index * axis_step)[0]
            comptime if operation == 2:
                extreme = fp32_minimum(extreme, value)
            else:
                extreme = fp32_maximum(extreme, value)
        out.unsafe_store(physical_output, extreme)


# Axis-generic inclusive FP32 scan fallback. Each lane owns one independent
# scan line (all coordinates except axis); there are no inter-lane races.
# The direction is logical 0->extent-1, even with negative physical strides.
# Native warp/block scans remain preferred for long contiguous inner axes.
def gpu_scan_axis_strided_fp32[operation: Int, rank: Int, axis: Int](
    out: Pointer[Float32, MutUnsafeAnyOrigin],
    source: Pointer[Float32, MutUnsafeAnyOrigin],
    dimensions: SIMD[DType.int64, 8],
    out_steps: SIMD[DType.int64, 8],
    source_steps: SIMD[DType.int64, 8],
    out_origin: Int64, source_origin: Int64,
    rows: Int64, scan_length: Int64,
):
    comptime assert rank >= 1 and rank <= 8, "invalid FP32 GPU scan rank"
    comptime assert axis >= 0 and axis < rank, "invalid FP32 GPU scan axis"
    comptime assert operation >= 0 and operation <= 3, "invalid FP32 GPU scan op"
    var row = global_idx.x
    if row >= Int(rows) or scan_length <= 0:
        return
    var remainder = row
    var output_base = Int(out_origin)
    var source_base = Int(source_origin)
    comptime for step in range(rank):
        var current_axis = rank - 1 - step
        comptime if current_axis != axis:
            var extent = Int(dimensions[current_axis])
            var coordinate = remainder % extent
            remainder //= extent
            output_base += coordinate * Int(out_steps[current_axis])
            source_base += coordinate * Int(source_steps[current_axis])
    var output_step = Int(out_steps[axis])
    var source_step = Int(source_steps[axis])
    var prefix = source.unsafe_load(source_base)[0]
    out.unsafe_store(output_base, prefix)
    for index in range(1, Int(scan_length)):
        var next = source.unsafe_load(source_base + index * source_step)[0]
        comptime if operation == 0:
            prefix += next
        elif operation == 1:
            prefix *= next
        elif operation == 2:
            prefix = fp32_minimum(prefix, next)
        else:
            prefix = fp32_maximum(prefix, next)
        out.unsafe_store(output_base + index * output_step, prefix)


# Rank-1..8 MULTI-AXIS keepdims FP32 reduction fallback. The single-axis
# version above remains available as a narrower specialization. Each thread
# owns one destination element; its selected axes are traversed by a bounded
# mixed-radix scalar loop. Distinct validated output mappings remove atomic
# writes entirely. Output dimension=1 for every reduced axis.
#
# A compile-time rank/opcode selects ONE native family. A runtime axes_mask
# merely identifies the chosen subset inside that bounded family; it never
# authorizes memory access without the host's checked shape/span proof.
# The production UASDEA loader must bind a PRECOMPILED artifact separately:
# source enqueue is opt-in development-only and may compile at runtime.
def gpu_reduce_axes_strided_fp32[operation: Int, rank: Int](
    out: Pointer[Float32, MutUnsafeAnyOrigin],
    source: Pointer[Float32, MutUnsafeAnyOrigin],
    source_dims: SIMD[DType.int64, 8],
    output_dims: SIMD[DType.int64, 8],
    output_steps: SIMD[DType.int64, 8],
    source_steps: SIMD[DType.int64, 8],
    output_origin: Int64, source_origin: Int64,
    output_count: Int64, reduction_count: Int64, axes_mask: Int64,
):
    comptime assert rank >= 1 and rank <= 8, "FP32 reduction rank 1..8"
    comptime assert operation >= 0 and operation <= 4, "FP32 reduction opcode"
    var flat = global_idx.x
    if flat >= Int(output_count):
        return
    var remainder = flat
    var destination = Int(output_origin)
    var source_base = Int(source_origin)
    # Decode only the output coordinates. Reduced dimensions are singleton,
    # so this automatically ignores their source coordinate contributions.
    comptime for reverse_axis in range(rank):
        var axis = rank - 1 - reverse_axis
        var dim = Int(output_dims[axis])
        var coordinate = remainder % dim
        remainder //= dim
        destination += coordinate * Int(output_steps[axis])
        source_base += coordinate * Int(source_steps[axis])
    var accumulator = Float32(0.0)
    comptime if operation == 1:
        accumulator = Float32(1.0)
    var first = True
    for index in range(Int(reduction_count)):
        var reduced = index
        var address = source_base
        comptime for reverse_axis in range(rank):
            var axis = rank - 1 - reverse_axis
            if Int(axes_mask) & (Int(1) << axis) != 0:
                var extent = Int(source_dims[axis])
                var coordinate = reduced % extent
                reduced //= extent
                address += coordinate * Int(source_steps[axis])
        var value = source.unsafe_load(address)[0]
        comptime if operation == 0 or operation == 4:
            accumulator += value
        elif operation == 1:
            accumulator *= value
        else:
            if first:
                accumulator = value
                first = False
            else:
                comptime if operation == 2:
                    accumulator = fp32_minimum(accumulator, value)
                else:
                    accumulator = fp32_maximum(accumulator, value)
    comptime if operation == 4:
        accumulator /= Float32(Int(reduction_count))
    out.unsafe_store(destination, accumulator)


# Optional AOT warp variants for FAST_TREE mode only. Reassociation differs
# from the strict sequential FP32 reduction and scan, so these are never
# selected by the default DETERMINISTIC_PAIRWISE / STRICT_FP32 policy.
# For a CTA with 256 threads, exactly eight independent 32-lane warps each
# process one C-contiguous row. Final idle warps participate in collectives.
def gpu_reduce_rows_warp32_fast_fp32[
    operation: Int, LO: AnyType, LI: AnyType,
](
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    inp: TileTensor[DType.float32, LI, MutAnyOrigin],
    rows_abi: Int32, cols_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var rows = Int(rows_abi)
    var cols = Int(cols_abi)
    comptime assert operation == 0 or operation == 4, "warp sum or mean only"
    var index = global_idx.x
    var row = index // 32
    var lane = index % 32
    var partial = Float32(0.0)
    if row < rows:
        for column in range(lane, cols, 32):
            partial += inp[row * cols + column]
    var total = warp.sum(partial)
    if row < rows and lane == 0:
        comptime if operation == 4:
            # Host rejects mean(cols=0) before launch.
            total /= Float32(cols)
        out[row] = total


# Single-warp inclusive row scan for the common 1..32 column case. Every
# lane participates even for partial rows; inactive lanes contribute zero.
# This changes Float32 summation association and requires FAST_TREE opt-in.
def gpu_scan_rows_warp32_sum_fast_fp32[LO: AnyType, LI: AnyType](
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    inp: TileTensor[DType.float32, LI, MutAnyOrigin],
    rows_abi: Int32, cols_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var rows = Int(rows_abi)
    var cols = Int(cols_abi)
    # Host ABI: 1 <= cols <= 32, 256 threads/block, flat C-order FP32.
    var index = global_idx.x
    var row = index // 32
    var lane = index % 32
    var value = Float32(0.0)
    if row < rows and lane < cols:
        value = inp[row * cols + lane]
    var prefix = warp.prefix_sum(value)
    if row < rows and lane < cols:
        out[row * cols + lane] = prefix
