# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | GPU KERNEL / ELEMENTWISE KERNELS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / GPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.gpu.elementwise
# Source:        src/momijo/tensor_light/kernels/gpu/elementwise.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines elementwise kernels for the GPU tensor kernel layer.
#
#   Public declarations include `gpu_add_fp32`, `gpu_mul_fp32`,
#   `gpu_sub_fp32`, `gpu_div_fp32`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `gpu_add_fp32`,
#     `gpu_mul_fp32`, `gpu_sub_fp32`, `gpu_div_fp32`, `gpu_fma_fp32` and
#     related symbols.
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

# Coalesced one-thread-per-element SIMT FP32 kernels. TileTensor is zero-copy.
# Layouts of output and inputs are separately specialized; all must be
# individually proved flat, contiguous, count-compatible before launching.
# Matching logical shapes do NOT imply identical native TileTensor layouts.

from max.gpu import global_idx
from layout import TileTensor
from ...dtype.fp32 import fp32_minimum, fp32_maximum
from std.math import fma as math_fma
from std.math import sqrt as math_sqrt, exp as math_exp, log as math_log
from std.math import sin as math_sin, cos as math_cos, tanh as math_tanh
from std.math import abs as math_abs
from std.memory import Pointer

def gpu_add_fp32[LO: AnyType, LA: AnyType, LB: AnyType](out: TileTensor[DType.float32, LO, MutAnyOrigin], a: TileTensor[DType.float32, LA, MutAnyOrigin], b: TileTensor[DType.float32, LB, MutAnyOrigin], count_abi: Int32):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var count = Int(count_abi)
    var idx = global_idx.x
    if idx < count:
        out[idx] = a[idx] + b[idx]

def gpu_mul_fp32[LO: AnyType, LA: AnyType, LB: AnyType](out: TileTensor[DType.float32, LO, MutAnyOrigin], a: TileTensor[DType.float32, LA, MutAnyOrigin], b: TileTensor[DType.float32, LB, MutAnyOrigin], count_abi: Int32):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var count = Int(count_abi)
    var idx = global_idx.x
    if idx < count:
        out[idx] = a[idx] * b[idx]

def gpu_sub_fp32[LO: AnyType, LA: AnyType, LB: AnyType](out: TileTensor[DType.float32, LO, MutAnyOrigin], a: TileTensor[DType.float32, LA, MutAnyOrigin], b: TileTensor[DType.float32, LB, MutAnyOrigin], count_abi: Int32):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var count = Int(count_abi)
    var idx = global_idx.x
    if idx < count:
        out[idx] = a[idx] - b[idx]


def gpu_div_fp32[LO: AnyType, LA: AnyType, LB: AnyType](out: TileTensor[DType.float32, LO, MutAnyOrigin], a: TileTensor[DType.float32, LA, MutAnyOrigin], b: TileTensor[DType.float32, LB, MutAnyOrigin], count_abi: Int32):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var count = Int(count_abi)
    var idx = global_idx.x
    if idx < count:
        out[idx] = a[idx] / b[idx]


# This is the fused FP32 ternary primitive, not a multiply followed by an
# independently rounded addition. The public FP32 math policy must select
# this AOT kernel only in a mode that explicitly permits fused arithmetic.
# STRICT_FP32 code requiring separate rounding uses a distinct compiled ABI.
def gpu_fma_fp32[LO: AnyType, LA: AnyType, LB: AnyType, LC: AnyType](out: TileTensor[DType.float32, LO, MutAnyOrigin], a: TileTensor[DType.float32, LA, MutAnyOrigin], b: TileTensor[DType.float32, LB, MutAnyOrigin], c: TileTensor[DType.float32, LC, MutAnyOrigin], count_abi: Int32):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var count = Int(count_abi)
    var idx = global_idx.x
    if idx < count:
        out[idx] = math_fma(a[idx], b[idx], c[idx])


# Single GPU unary family, compiled separately for every supported FP32 opcode.
# The LayoutTensor/TileTensor layout is provided by an AOT-exact zero-copy
# adapter at the boundary; no device/host transfers are performed here.


def gpu_unary_fp32[operation: Int, LO: AnyType, LI: AnyType](
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    source: TileTensor[DType.float32, LI, MutAnyOrigin],
    count_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var count = Int(count_abi)
    comptime assert operation >= 0 and operation <= 8, "invalid GPU unary FP32 opcode"
    var i = global_idx.x
    if i < count:
        var value = source[i]
        comptime if operation == 0:
            out[i] = -value
        elif operation == 1:
            out[i] = math_abs(value)
        elif operation == 2:
            out[i] = math_sqrt(value)
        elif operation == 3:
            out[i] = math_exp(value)
        elif operation == 4:
            out[i] = math_log(value)
        elif operation == 5:
            out[i] = math_sin(value)
        elif operation == 6:
            out[i] = math_cos(value)
        elif operation == 7:
            out[i] = math_tanh(value)
        else:
            out[i] = Float32(1.0) / value


# GPU SIMD lane semantics are identical to the CPU's FP32 extrema policy:
# propagate NaNs and prefer -0 for minimum, +0 for maximum.
# This flat kernel performs no dtype promotion and requires independent,
# representable, non-overlapping output/input storage bounds at the launcher.
def gpu_maximum_fp32[LO: AnyType, LA: AnyType, LB: AnyType](
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin], count_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var count = Int(count_abi)
    var idx = global_idx.x
    if idx < count:
        out[idx] = fp32_maximum(a[idx], b[idx])


def gpu_minimum_fp32[LO: AnyType, LA: AnyType, LB: AnyType](
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin], count_abi: Int32,
):
    # Fixed-width, DevicePassable source-kernel ABI; convert once outside hot loops.
    var count = Int(count_abi)
    var idx = global_idx.x
    if idx < count:
        out[idx] = fp32_minimum(a[idx], b[idx])


# Rank-static, arbitrary-stride GPU fallback for FP32 binary operations.
# The caller supplies validated physical spans and right-aligned broadcast
# strides. In particular, an expanded size-one input axis has stride zero.
# All offsets are ELEMENT offsets, relative to the respective DeviceBuffer.
# Negative strides are legal only when the checked base offset covers them.
# A flat C-order fast path must be preferred at the dispatch boundary.
# Metadata uses 8 fixed Int64 lanes, never a heap-allocated device array.

def gpu_binary_strided_fp32[operation: Int, rank: Int, dense: Bool](
    out: Pointer[Float32, MutUnsafeAnyOrigin],
    left: Pointer[Float32, MutUnsafeAnyOrigin],
    right: Pointer[Float32, MutUnsafeAnyOrigin],
    extents: SIMD[DType.int64, 8],
    out_strides: SIMD[DType.int64, 8],
    left_strides: SIMD[DType.int64, 8],
    right_strides: SIMD[DType.int64, 8],
    out_offset: Int64, left_offset: Int64, right_offset: Int64,
    count: Int64,
):
    comptime assert rank >= 0 and rank <= 8, "FP32 GPU rank must be 0..8"
    comptime assert operation >= 0 and operation <= 5, "invalid FP32 GPU binary op"
    var flat = global_idx.x
    if flat >= Int(count):
        return
    var remainder = flat
    var physical_out = Int(out_offset)
    var physical_left = Int(left_offset)
    var physical_right = Int(right_offset)
    comptime if dense:
        # C-contiguous, nonbroadcasting logical linear order; no per-element
        # div/mod or multi-axis address arithmetic in the fast kernel.
        physical_out += flat
        physical_left += flat
        physical_right += flat
    else:
        # Rank 0 has no logical axes and reads the scalar at base offset.
        comptime for step in range(rank):
            var axis = rank - 1 - step
            var extent = Int(extents[axis])
            # PRECONDITION: count>0 implies each output extent>0.
            var position = remainder % extent
            remainder //= extent
            physical_out += position * Int(out_strides[axis])
            physical_left += position * Int(left_strides[axis])
            physical_right += position * Int(right_strides[axis])
    var a = left.unsafe_load(physical_left)[0]
    var b = right.unsafe_load(physical_right)[0]
    comptime if operation == 0:
        out.unsafe_store(physical_out, a + b)
    elif operation == 1:
        out.unsafe_store(physical_out, a - b)
    elif operation == 2:
        out.unsafe_store(physical_out, a * b)
    elif operation == 3:
        out.unsafe_store(physical_out, a / b)
    elif operation == 4:
        out.unsafe_store(physical_out, fp32_maximum(a, b))
    else:
        out.unsafe_store(physical_out, fp32_minimum(a, b))


# Unary companion to the generic-stride binary fallback. Shapes are equal;
# physical traversal orders may differ and may use negative strides.
# Fused arithmetic/TF32 policy is not changed by this metadata-only lowering.
def gpu_unary_strided_fp32[operation: Int, rank: Int, dense: Bool](
    out: Pointer[Float32, MutUnsafeAnyOrigin],
    source: Pointer[Float32, MutUnsafeAnyOrigin],
    extents: SIMD[DType.int64, 8],
    out_strides: SIMD[DType.int64, 8],
    source_strides: SIMD[DType.int64, 8],
    out_offset: Int64, source_offset: Int64, count: Int64,
):
    comptime assert rank >= 0 and rank <= 8, "FP32 GPU rank must be 0..8"
    comptime assert operation >= 0 and operation <= 8, "invalid unary FP32 GPU op"
    var flat = global_idx.x
    if flat >= Int(count):
        return
    var remainder = flat
    var physical_out = Int(out_offset)
    var physical_source = Int(source_offset)
    comptime if dense:
        physical_out += flat
        physical_source += flat
    else:
        comptime for step in range(rank):
            var axis = rank - 1 - step
            var extent = Int(extents[axis])
            var position = remainder % extent
            remainder //= extent
            physical_out += position * Int(out_strides[axis])
            physical_source += position * Int(source_strides[axis])
    var value = source.unsafe_load(physical_source)[0]
    comptime if operation == 0:
        out.unsafe_store(physical_out, -value)
    elif operation == 1:
        out.unsafe_store(physical_out, math_abs(value))
    elif operation == 2:
        out.unsafe_store(physical_out, math_sqrt(value))
    elif operation == 3:
        out.unsafe_store(physical_out, math_exp(value))
    elif operation == 4:
        out.unsafe_store(physical_out, math_log(value))
    elif operation == 5:
        out.unsafe_store(physical_out, math_sin(value))
    elif operation == 6:
        out.unsafe_store(physical_out, math_cos(value))
    elif operation == 7:
        out.unsafe_store(physical_out, math_tanh(value))
    else:
        out.unsafe_store(physical_out, Float32(1.0) / value)
