# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | CPU KERNEL / UNARY OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / CPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.cpu.unary
# Source:        src/momijo/tensor_light/kernels/cpu/unary.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines unary operations for the CPU tensor kernel layer.
#
#   Public declarations include `cpu_unary_scalar_fp32`,
#   `cpu_unary_contiguous_fp32`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `cpu_unary_scalar_fp32`,
#     `cpu_unary_contiguous_fp32`.
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

# AOT-specialized FP32 unary scalar operations. Never dispatch dtype at runtime.
from std.math import sqrt as math_sqrt, exp as math_exp, log as math_log
from std.math import sin as math_sin, cos as math_cos, tanh as math_tanh
from std.math import abs as math_abs
from std.collections import Span
from ...internal.status import TensorStatus
from ...parallel.parallel_for import CpuWorkPolicy, recommend_worker_count, execute_parallel_shards
from ...alias.overlap import (
    host_output_disjoint_from_input, host_output_exactly_matches_input,
)

@inline(.always)
def cpu_unary_scalar_fp32[operation: Int](value: Float32) -> Float32:
    comptime assert operation >= 0 and operation <= 8, "invalid FP32 unary opcode"
    comptime if operation == 0:
        return -value
    elif operation == 1:
        # One abs semantic for scalar and SIMD, including signed zero/NaNs.
        return math_abs(value)
    elif operation == 2:
        return math_sqrt(value)
    elif operation == 3:
        return math_exp(value)
    elif operation == 4:
        return math_log(value)
    elif operation == 5:
        return math_sin(value)
    elif operation == 6:
        return math_cos(value)
    elif operation == 7:
        return math_tanh(value)
    else:
        return Float32(1.0) / value


def cpu_unary_contiguous_fp32[
    operation: Int, width: Int, so: Origin, do: Origin[mut=True]
](input: Span[False, Float32, so], mut output: Span[True, Float32, do]) -> TensorStatus:
    comptime assert operation >= 0 and operation <= 8, "invalid FP32 unary opcode"
    comptime assert width > 0, "Float32 SIMD width must be positive"
    # Direct callers may not have the primary Tensor owner's preflight.
    # Validate count, physical byte extent and shifted alias before SIMD.
    var count = len(input)
    if count < 0 or len(output) < 0:
        return TensorStatus.INVALID_LAYOUT
    if count != len(output):
        return TensorStatus.SIZE_MISMATCH
    if count > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if count == 0:
        return TensorStatus.OK
    if (not host_output_disjoint_from_input(output, input)
        and not host_output_exactly_matches_input(output, input)):
        return TensorStatus.ALIAS_UNKNOWN
    # This parallel path uses only fully checked, disjoint output shards.
    # The unified capture is limited to this synchronous parallel worker; no runtime registry is used.
    var policy = CpuWorkPolicy(262144, 8)
    var workers = recommend_worker_count(count, policy)
    if workers > 1:
        var source_ptr = input.unsafe_ptr()
        var result_ptr = output.unsafe_ptr()

        def execute_unary_shard(shard_id: Int) {imm}:
            var per_shard = count // workers
            var extra = count % workers
            var first = shard_id * per_shard + min(shard_id, extra)
            var last = first + per_shard
            if shard_id < extra:
                last += 1
            var vec_end = first
            comptime if operation == 0 or operation == 1 or operation == 2 or operation == 8:
                var length = last - first
                vec_end = first + length - length % width
                for i in range(first, vec_end, width):
                    var vector = source_ptr.unsafe_load[width=width, alignment=4](i)
                    comptime if operation == 0:
                        result_ptr.unsafe_store[alignment=4](i, -vector)
                    elif operation == 1:
                        result_ptr.unsafe_store[alignment=4](i, math_abs(vector))
                    elif operation == 2:
                        result_ptr.unsafe_store[alignment=4](i, math_sqrt(vector))
                    else:
                        result_ptr.unsafe_store[alignment=4](i, SIMD[DType.float32, width](1.0) / vector)
            for i in range(vec_end, last):
                result_ptr.unsafe_store[alignment=4](
                    i, cpu_unary_scalar_fp32[operation](source_ptr.unsafe_load[alignment=4](i))
                )

        return execute_parallel_shards(execute_unary_shard, count, policy)
    var full = count - count % width
    comptime if operation == 0 or operation == 1 or operation == 2 or operation == 8:
        var sp = input.unsafe_ptr()
        var dp = output.unsafe_ptr()
        for i in range(0, full, width):
            var lanes = sp.unsafe_load[width=width, alignment=4](i)
            comptime if operation == 0:
                dp.unsafe_store[alignment=4](i, -lanes)
            elif operation == 1:
                dp.unsafe_store[alignment=4](i, math_abs(lanes))
            elif operation == 2:
                dp.unsafe_store[alignment=4](i, math_sqrt(lanes))
            else:
                dp.unsafe_store[alignment=4](i, SIMD[DType.float32, width](1.0) / lanes)
    else:
        full = 0
    for i in range(full, count):
        output[i] = cpu_unary_scalar_fp32[operation](input[i])
    return TensorStatus.OK
