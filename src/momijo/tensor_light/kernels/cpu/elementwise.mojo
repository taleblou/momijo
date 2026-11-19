# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | CPU KERNEL / ELEMENTWISE KERNELS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / CPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.cpu.elementwise
# Source:        src/momijo/tensor_light/kernels/cpu/elementwise.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines elementwise kernels for the CPU tensor kernel layer.
#
#   Public declarations include `cpu_fma_contiguous`, `cpu_scale_contiguous`,
#   `cpu_shift_contiguous`, `cpu_scalar_broadcast_binary_contiguous`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `cpu_fma_contiguous`,
#     `cpu_scale_contiguous`, `cpu_shift_contiguous`,
#     `cpu_scalar_broadcast_binary_contiguous`, `cpu_neg_contiguous` and
#     related symbols.
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

# Contiguous FP32 SIMD fast path. Only called after bounds/alias validation.
# Compiler specializes width and unroll before execution.
from std.collections import Span
from std.math import fma as std_fma
from ...dtype.fp32 import fp32_maximum, fp32_minimum
from ...internal.status import TensorStatus
from ...alias.overlap import host_output_disjoint_from_input, host_output_exactly_matches_input
from .unary import cpu_unary_contiguous_fp32
from ...parallel.parallel_for import CpuWorkPolicy, recommend_worker_count, execute_parallel_shards



# One-pass rank-static FP32 multiply-add: no two temporary Tensors, one SIMD
# pass and one scalar tail. Calling code validates all source lengths/layouts.
# Fused and strict routes branch ONCE before the vectorized inner loop.
@inline(.always)
def cpu_fma_contiguous[
    width: Int, o: Origin[mut=True], a: Origin, b: Origin, c: Origin,
](
    out: Span[True, Float32, o], lhs: Span[False, Float32, a],
    rhs: Span[False, Float32, b], addend: Span[False, Float32, c],
    fused: Bool,
) -> TensorStatus:
    comptime assert width > 0, "positive SIMD lane count required"
    # A caller-visible SIMD helper must validate every borrowed input before
    # the first unsafe vector load. Exact elementwise aliases are safe because
    # all three operands are read before the corresponding result is written;
    # shifted aliases can overwrite future unread inputs and are rejected.
    var n = len(out)
    if n < 0 or len(lhs) < 0 or len(rhs) < 0 or len(addend) < 0:
        return TensorStatus.INVALID_LAYOUT
    if n != len(lhs) or n != len(rhs) or n != len(addend):
        return TensorStatus.SIZE_MISMATCH
    if n > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if n == 0:
        return TensorStatus.OK
    var safe_lhs = (host_output_disjoint_from_input(out, lhs)
                    or host_output_exactly_matches_input(out, lhs))
    var safe_rhs = (host_output_disjoint_from_input(out, rhs)
                    or host_output_exactly_matches_input(out, rhs))
    var safe_addend = (host_output_disjoint_from_input(out, addend)
                       or host_output_exactly_matches_input(out, addend))
    if not safe_lhs or not safe_rhs or not safe_addend:
        return TensorStatus.ALIAS_UNKNOWN
    # The synchronous AOT parallel branch owns independent output spans.
    # Fused versus nonfused arithmetic is resolved once per shard, preserving
    # the public FP32 math-mode contract. Exact in-place aliases are legal.
    var policy = CpuWorkPolicy(262144, 8)
    var workers = recommend_worker_count(n, policy)
    if workers > 1:
        var dest_ptr = out.unsafe_ptr()
        var left_ptr = lhs.unsafe_ptr()
        var right_ptr = rhs.unsafe_ptr()
        var addend_ptr = addend.unsafe_ptr()

        def fma_shard(shard: Int) {imm}:
            var q = n // workers
            var rem = n % workers
            var first = shard * q + min(shard, rem)
            var last = first + q
            if shard < rem:
                last += 1
            var length = last - first
            var lanes = first + length - length % width
            if fused:
                for i in range(first, lanes, width):
                    dest_ptr.unsafe_store[alignment=4](i, std_fma(
                        left_ptr.unsafe_load[width=width, alignment=4](i),
                        right_ptr.unsafe_load[width=width, alignment=4](i),
                        addend_ptr.unsafe_load[width=width, alignment=4](i),
                    ))
                for i in range(lanes, last):
                    dest_ptr.unsafe_store[alignment=4](i, std_fma(
                        left_ptr.unsafe_load[alignment=4](i),
                        right_ptr.unsafe_load[alignment=4](i),
                        addend_ptr.unsafe_load[alignment=4](i),
                    ))
            else:
                for i in range(first, lanes, width):
                    dest_ptr.unsafe_store[alignment=4](i,
                        left_ptr.unsafe_load[width=width, alignment=4](i)
                        * right_ptr.unsafe_load[width=width, alignment=4](i)
                        + addend_ptr.unsafe_load[width=width, alignment=4](i))
                for i in range(lanes, last):
                    dest_ptr.unsafe_store[alignment=4](i,
                        left_ptr.unsafe_load[alignment=4](i)
                        * right_ptr.unsafe_load[alignment=4](i)
                        + addend_ptr.unsafe_load[alignment=4](i))

        return execute_parallel_shards(fma_shard, n, policy)
    var end = n - n % width
    var op = out.unsafe_ptr()
    var ap = lhs.unsafe_ptr()
    var bp = rhs.unsafe_ptr()
    var cp = addend.unsafe_ptr()
    if fused:
        for i in range(0, end, width):
            op.unsafe_store[alignment=4](i, std_fma(
                ap.unsafe_load[width=width, alignment=4](i),
                bp.unsafe_load[width=width, alignment=4](i),
                cp.unsafe_load[width=width, alignment=4](i),
            ))
        for i in range(end, n):
            out[i] = std_fma(lhs[i], rhs[i], addend[i])
    else:
        for i in range(0, end, width):
            op.unsafe_store[alignment=4](i,
                ap.unsafe_load[width=width, alignment=4](i) * bp.unsafe_load[width=width, alignment=4](i)
                + cp.unsafe_load[width=width, alignment=4](i))
        for i in range(end, n):
            out[i] = lhs[i] * rhs[i] + addend[i]
    return TensorStatus.OK

@inline(.always)
def _cpu_add_contiguous_unchecked[width: Int, o: Origin[mut=True], a: Origin, b: Origin](out: Span[True, Float32, o], lhs: Span[False, Float32, a], rhs: Span[False, Float32, b]):
    comptime assert width >= 1, "SIMD width must be positive"
    var n = len(out)
    var limit = n - n % width
    var op = out.unsafe_ptr()
    var ap = lhs.unsafe_ptr()
    var bp = rhs.unsafe_ptr()
    for i in range(0, limit, width):
        var left = ap.unsafe_load[width=width, alignment=4](i)
        var right = bp.unsafe_load[width=width, alignment=4](i)
        op.unsafe_store[alignment=4](i, left + right)
    for i in range(limit, n):
        out[i] = lhs[i] + rhs[i]

@inline(.always)
def cpu_scale_contiguous[width: Int, o: Origin[mut=True], a: Origin](out: Span[True, Float32, o], lhs: Span[False, Float32, a], scale: Float32) -> TensorStatus:
    comptime assert width >= 1, "SIMD width must be positive"
    return cpu_scalar_broadcast_binary_contiguous[2, False, width](
        out, lhs, scale,
    )

@inline(.always)
def cpu_shift_contiguous[width: Int, o: Origin[mut=True], a: Origin](out: Span[True, Float32, o], lhs: Span[False, Float32, a], amount: Float32) -> TensorStatus:
    comptime assert width >= 1, "SIMD width must be positive"
    return cpu_scalar_broadcast_binary_contiguous[0, False, width](
        out, lhs, amount,
    )

@inline(.always)
def _cpu_sub_contiguous_unchecked[width: Int, o: Origin[mut=True], a: Origin, b: Origin](out: Span[True, Float32, o], lhs: Span[False, Float32, a], rhs: Span[False, Float32, b]):
    comptime assert width >= 1, "SIMD width must be positive"
    var n = len(out)
    var end = n - n % width
    var op = out.unsafe_ptr()
    var ap = lhs.unsafe_ptr()
    var bp = rhs.unsafe_ptr()
    for i in range(0, end, width):
        op.unsafe_store[alignment=4](i, ap.unsafe_load[width=width, alignment=4](i) - bp.unsafe_load[width=width, alignment=4](i))
    for i in range(end, n):
        out[i] = lhs[i] - rhs[i]


@inline(.always)
def _cpu_mul_contiguous_unchecked[width: Int, o: Origin[mut=True], a: Origin, b: Origin](out: Span[True, Float32, o], lhs: Span[False, Float32, a], rhs: Span[False, Float32, b]):
    comptime assert width >= 1, "SIMD width must be positive"
    var n = len(out)
    var end = n - n % width
    var op = out.unsafe_ptr()
    var ap = lhs.unsafe_ptr()
    var bp = rhs.unsafe_ptr()
    for i in range(0, end, width):
        op.unsafe_store[alignment=4](i, ap.unsafe_load[width=width, alignment=4](i) * bp.unsafe_load[width=width, alignment=4](i))
    for i in range(end, n):
        out[i] = lhs[i] * rhs[i]


@inline(.always)
def _cpu_div_contiguous_unchecked[width: Int, o: Origin[mut=True], a: Origin, b: Origin](out: Span[True, Float32, o], lhs: Span[False, Float32, a], rhs: Span[False, Float32, b]):
    comptime assert width >= 1, "SIMD width must be positive"
    var n = len(out)
    var end = n - n % width
    var op = out.unsafe_ptr()
    var ap = lhs.unsafe_ptr()
    var bp = rhs.unsafe_ptr()
    for i in range(0, end, width):
        op.unsafe_store[alignment=4](i, ap.unsafe_load[width=width, alignment=4](i) / bp.unsafe_load[width=width, alignment=4](i))
    for i in range(end, n):
        out[i] = lhs[i] / rhs[i]


# AOT scalar-broadcast fast path for all ranks 0..8. The checked Tensor
# boundary proves that the non-scalar operand is one C-contiguous, exhaustive
# FP32 run and that the other operand is exactly one readable element.
# A scalar can appear on either side of subtraction/division (not commutative).
# The operation and operand order are compile-time specialization dimensions;
# no runtime op branch or N-D coordinate carry enters the vector loop.
@inline(.always)
def cpu_scalar_broadcast_binary_contiguous[
    operation: Int, scalar_on_left: Bool, width: Int,
    output_origin: Origin[mut=True], source_origin: Origin,
](
    out: Span[True, Float32, output_origin],
    source: Span[False, Float32, source_origin], scalar: Float32,
) -> TensorStatus:
    comptime assert operation >= 0 and operation <= 5, "invalid FP32 binary opcode"
    comptime assert width >= 1, "FP32 SIMD width must be positive"
    # The owner proves a dense input run and independently materializes the
    # scalar. Do not allow a direct helper caller to overread a short source,
    # or to overwrite later unread source values through a shifted alias.
    var count = len(out)
    if count < 0 or len(source) < 0:
        return TensorStatus.INVALID_LAYOUT
    if count != len(source):
        return TensorStatus.SIZE_MISMATCH
    if count > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if count == 0:
        return TensorStatus.OK
    if (not host_output_disjoint_from_input(out, source)
        and not host_output_exactly_matches_input(out, source)):
        return TensorStatus.ALIAS_UNKNOWN
    comptime if operation <= 3:
        var lanes = count - count % width
        var constant = SIMD[DType.float32, width](scalar)
        var sp = source.unsafe_ptr()
        var dp = out.unsafe_ptr()
        for i in range(0, lanes, width):
            # C-contiguous views may begin at a non-vector-aligned offset.
            # Natural Float32 alignment, not SIMD-wide alignment, is proven.
            var input_values = sp.unsafe_load[width=width, alignment=4](i)
            comptime if operation == 0:
                comptime if scalar_on_left:
                    dp.unsafe_store[alignment=4](i, constant + input_values)
                else:
                    dp.unsafe_store[alignment=4](i, input_values + constant)
            elif operation == 1:
                comptime if scalar_on_left:
                    dp.unsafe_store[alignment=4](i, constant - input_values)
                else:
                    dp.unsafe_store[alignment=4](i, input_values - constant)
            elif operation == 2:
                comptime if scalar_on_left:
                    dp.unsafe_store[alignment=4](i, constant * input_values)
                else:
                    dp.unsafe_store[alignment=4](i, input_values * constant)
            else:
                comptime if scalar_on_left:
                    dp.unsafe_store[alignment=4](i, constant / input_values)
                else:
                    dp.unsafe_store[alignment=4](i, input_values / constant)
        for i in range(lanes, count):
            comptime if operation == 0:
                comptime if scalar_on_left:
                    out[i] = scalar + source[i]
                else:
                    out[i] = source[i] + scalar
            elif operation == 1:
                comptime if scalar_on_left:
                    out[i] = scalar - source[i]
                else:
                    out[i] = source[i] - scalar
            elif operation == 2:
                comptime if scalar_on_left:
                    out[i] = scalar * source[i]
                else:
                    out[i] = source[i] * scalar
            else:
                comptime if scalar_on_left:
                    out[i] = scalar / source[i]
                else:
                    out[i] = source[i] / scalar
    else:
        # Preserve the canonical NaN and signed-zero policies rather than
        # using a target-dependent SIMD maximum/minimum instruction.
        for i in range(count):
            comptime if operation == 4:
                comptime if scalar_on_left:
                    out[i] = fp32_maximum(scalar, source[i])
                else:
                    out[i] = fp32_maximum(source[i], scalar)
            else:
                comptime if scalar_on_left:
                    out[i] = fp32_minimum(scalar, source[i])
                else:
                    out[i] = fp32_minimum(source[i], scalar)
    return TensorStatus.OK


# AOT FP32 unary negation for an already-validated contiguous host slice.
# The memory ranges must not overlap in a way that causes cross-lane hazards;
# the normal owning-operation caller guarantees fresh output storage.
# Standalone checked NEG adapter; native SIMD remains in the checked unary
# implementation, so it cannot read an undersized borrowed output.
@inline(.always)
def cpu_neg_contiguous[width: Int, o: Origin[mut=True], a: Origin](
    out: Span[True, Float32, o], source: Span[False, Float32, a],
) -> TensorStatus:
    return cpu_unary_contiguous_fp32[0, width](source, out)


# FP32 scalar-right subtraction. Keep native subtraction rather than rewriting
# as addition of a negated scalar (signed-zero and strict arithmetic semantics).
@inline(.always)
def cpu_sub_scalar_contiguous[width: Int, o: Origin[mut=True], a: Origin](
    out: Span[True, Float32, o], lhs: Span[False, Float32, a], rhs: Float32
) -> TensorStatus:
    comptime assert width >= 1, "SIMD width must be positive"
    return cpu_scalar_broadcast_binary_contiguous[1, False, width](
        out, lhs, rhs,
    )

@inline(.always)
def cpu_div_scalar_contiguous[width: Int, o: Origin[mut=True], a: Origin](
    out: Span[True, Float32, o], lhs: Span[False, Float32, a], rhs: Float32
) -> TensorStatus:
    comptime assert width >= 1, "SIMD width must be positive"
    return cpu_scalar_broadcast_binary_contiguous[3, False, width](
        out, lhs, rhs,
    )

@inline(.always)
def _cpu_maximum_contiguous_unchecked[
    width: Int, o: Origin[mut=True], a: Origin, b: Origin,
](
    out: Span[True, Float32, o], lhs: Span[False, Float32, a],
    rhs: Span[False, Float32, b],
):
    comptime assert width >= 1, "SIMD width must be positive"
    for i in range(len(out)):
        out[i] = fp32_maximum(lhs[i], rhs[i])


@inline(.always)
def _cpu_minimum_contiguous_unchecked[
    width: Int, o: Origin[mut=True], a: Origin, b: Origin,
](
    out: Span[True, Float32, o], lhs: Span[False, Float32, a],
    rhs: Span[False, Float32, b],
):
    comptime assert width >= 1, "SIMD width must be positive"
    for i in range(len(out)):
        out[i] = fp32_minimum(lhs[i], rhs[i])


# One checked binary SIMD entrance for every primary owning Tensor/CPU AOT
# caller. The six SIMD leaves above remain trusted internal leaf operations.
# Nothing can be written until all lengths, byte counts and address hazards
# have been checked. A shifted output/source alias is not SIMD-safe even
# when the spans have equal length; exact in-place elementwise alias is safe.
# Numeric opcodes: add/sub/mul/div/maximum/minimum = 0..5.
def cpu_binary_contiguous_checked[
    operation: Int, width: Int, output_origin: Origin[mut=True],
    left_origin: Origin, right_origin: Origin,
](
    output: Span[True, Float32, output_origin],
    left: Span[False, Float32, left_origin],
    right: Span[False, Float32, right_origin],
) -> TensorStatus:
    comptime assert operation >= 0 and operation <= 5, "invalid FP32 binary opcode"
    comptime assert width >= 1, "positive vector width required"
    var count = len(output)
    if count < 0 or len(left) < 0 or len(right) < 0:
        return TensorStatus.INVALID_LAYOUT
    if count != len(left) or count != len(right):
        return TensorStatus.SIZE_MISMATCH
    if count > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if count == 0:
        return TensorStatus.OK
    if (not host_output_disjoint_from_input(output, left)
        and not host_output_exactly_matches_input(output, left)):
        return TensorStatus.ALIAS_UNKNOWN
    if (not host_output_disjoint_from_input(output, right)
        and not host_output_exactly_matches_input(output, right)):
        return TensorStatus.ALIAS_UNKNOWN
    # The checked alias contract allows exact elementwise in-place aliases,
    # but rejects shifted writes that could race with a different shard.
    # Threshold/worker count are conservative AOT policy defaults. A target
    # release profile may choose a different statically compiled variant.
    # For tiny work, the existing single-thread SIMD path avoids task cost.
    var policy = CpuWorkPolicy(262144, 8)
    var workers = recommend_worker_count(count, policy)
    if workers > 1:
        var out_ptr = output.unsafe_ptr()
        var left_ptr = left.unsafe_ptr()
        var right_ptr = right.unsafe_ptr()

        def execute_binary_shard(shard_id: Int) {imm}:
            # Quotient/remainder partition cannot overflow; each shard owns
            # exactly [first,last) and all other shards have disjoint writes.
            var per_shard = count // workers
            var extra = count % workers
            var first = shard_id * per_shard + min(shard_id, extra)
            var last = first + per_shard
            if shard_id < extra:
                last += 1
            var vec_end = first
            comptime if operation <= 3:
                var length = last - first
                vec_end = first + length - length % width
                for i in range(first, vec_end, width):
                    var x = left_ptr.unsafe_load[width=width, alignment=4](i)
                    var y = right_ptr.unsafe_load[width=width, alignment=4](i)
                    comptime if operation == 0:
                        out_ptr.unsafe_store[alignment=4](i, x + y)
                    elif operation == 1:
                        out_ptr.unsafe_store[alignment=4](i, x - y)
                    elif operation == 2:
                        out_ptr.unsafe_store[alignment=4](i, x * y)
                    else:
                        out_ptr.unsafe_store[alignment=4](i, x / y)
            for i in range(vec_end, last):
                var x = left_ptr.unsafe_load[alignment=4](i)
                var y = right_ptr.unsafe_load[alignment=4](i)
                comptime if operation == 0:
                    out_ptr.unsafe_store[alignment=4](i, x + y)
                elif operation == 1:
                    out_ptr.unsafe_store[alignment=4](i, x - y)
                elif operation == 2:
                    out_ptr.unsafe_store[alignment=4](i, x * y)
                elif operation == 3:
                    out_ptr.unsafe_store[alignment=4](i, x / y)
                elif operation == 4:
                    out_ptr.unsafe_store[alignment=4](i, fp32_maximum(x, y))
                else:
                    out_ptr.unsafe_store[alignment=4](i, fp32_minimum(x, y))

        return execute_parallel_shards(execute_binary_shard, count, policy)
    comptime if operation == 0:
        _cpu_add_contiguous_unchecked[width](output, left, right)
    elif operation == 1:
        _cpu_sub_contiguous_unchecked[width](output, left, right)
    elif operation == 2:
        _cpu_mul_contiguous_unchecked[width](output, left, right)
    elif operation == 3:
        _cpu_div_contiguous_unchecked[width](output, left, right)
    elif operation == 4:
        _cpu_maximum_contiguous_unchecked[width](output, left, right)
    else:
        _cpu_minimum_contiguous_unchecked[width](output, left, right)
    return TensorStatus.OK


# Checked public adapter: unsafe SIMD loads are private to this module.
@inline(.always)
def cpu_add_contiguous[
    width: Int, out_origin: Origin[mut=True], a: Origin, b: Origin,
](
    out: Span[True, Float32, out_origin],
    lhs: Span[False, Float32, a], rhs: Span[False, Float32, b],
) -> TensorStatus:
    return cpu_binary_contiguous_checked[0, width](out, lhs, rhs)


# Checked public adapter: unsafe SIMD loads are private to this module.
@inline(.always)
def cpu_sub_contiguous[
    width: Int, out_origin: Origin[mut=True], a: Origin, b: Origin,
](
    out: Span[True, Float32, out_origin],
    lhs: Span[False, Float32, a], rhs: Span[False, Float32, b],
) -> TensorStatus:
    return cpu_binary_contiguous_checked[1, width](out, lhs, rhs)


# Checked public adapter: unsafe SIMD loads are private to this module.
@inline(.always)
def cpu_mul_contiguous[
    width: Int, out_origin: Origin[mut=True], a: Origin, b: Origin,
](
    out: Span[True, Float32, out_origin],
    lhs: Span[False, Float32, a], rhs: Span[False, Float32, b],
) -> TensorStatus:
    return cpu_binary_contiguous_checked[2, width](out, lhs, rhs)


# Checked public adapter: unsafe SIMD loads are private to this module.
@inline(.always)
def cpu_div_contiguous[
    width: Int, out_origin: Origin[mut=True], a: Origin, b: Origin,
](
    out: Span[True, Float32, out_origin],
    lhs: Span[False, Float32, a], rhs: Span[False, Float32, b],
) -> TensorStatus:
    return cpu_binary_contiguous_checked[3, width](out, lhs, rhs)


# Checked public adapter: unsafe SIMD loads are private to this module.
@inline(.always)
def cpu_maximum_contiguous[
    width: Int, out_origin: Origin[mut=True], a: Origin, b: Origin,
](
    out: Span[True, Float32, out_origin],
    lhs: Span[False, Float32, a], rhs: Span[False, Float32, b],
) -> TensorStatus:
    return cpu_binary_contiguous_checked[4, width](out, lhs, rhs)


# Checked public adapter: unsafe SIMD loads are private to this module.
@inline(.always)
def cpu_minimum_contiguous[
    width: Int, out_origin: Origin[mut=True], a: Origin, b: Origin,
](
    out: Span[True, Float32, out_origin],
    lhs: Span[False, Float32, a], rhs: Span[False, Float32, b],
) -> TensorStatus:
    return cpu_binary_contiguous_checked[5, width](out, lhs, rhs)
