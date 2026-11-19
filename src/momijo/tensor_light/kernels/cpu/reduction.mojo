# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | CPU KERNEL / REDUCTIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / CPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.cpu.reduction
# Source:        src/momijo/tensor_light/kernels/cpu/reduction.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines reductions for the CPU tensor kernel layer.
#
#   Public declarations include `cpu_sum_contiguous`,
#   `cpu_sum_contiguous_parallel_fast`, `cpu_sum_strided_parallel_fast`,
#   `cpu_sum_c_strict_fp32`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `cpu_sum_contiguous`,
#     `cpu_sum_contiguous_parallel_fast`, `cpu_sum_strided_parallel_fast`,
#     `cpu_sum_c_strict_fp32`, `cpu_prod_c_fp32` and related symbols.
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

# SIMD FP32 sum with scalar tail, keeping accumulator data in registers.
from std.collections import Span
from ...layout.layout import TensorLayout
from ...layout.strides import c_strides
from ...internal.status import TensorStatus, TensorResult
from ...internal.checked_math import checked_mul_nonnegative
from ...alias.overlap import host_output_disjoint_from_input
from ...iterator.nd_iterator import NDIndexIterator
from ...dtype.fp32 import fp32_minimum, fp32_maximum, fp32_argmin_prefers, fp32_argmax_prefers
from std.sys.info import simd_width_of
from ...parallel.parallel_for import CpuWorkPolicy, recommend_worker_count, execute_parallel_shards

def cpu_sum_contiguous[width: Int, origin: Origin](src: Span[False, Float32, origin]) -> Float32:
    comptime assert width >= 1, "SIMD width must be positive"
    var n = len(src)
    var bound = n - n % width
    var p = src.unsafe_ptr()
    var accum = SIMD[DType.float32, width](0.0)
    for i in range(0, bound, width):
        accum += p.unsafe_load[width=width, alignment=4](i)
    var total = accum.reduce_add()
    for i in range(bound, n):
        total += src[i]
    return total


# Deterministic sharded FP32 sum for the FAST_TREE numeric policy only.
# This is intentionally NOT used by strict reductions: combining shards
# changes the FP32 addition association and may change the last few bits.
# The eight partial slots are fixed AOT stack storage; no per-element heap,
# hidden atomics, runtime opcode selection, or persistent worker ownership.
# A direct caller supplies a Span with its own valid host-memory capability;
# no untrusted raw address or forged length is authenticated here.
def cpu_sum_contiguous_parallel_fast[
    width: Int, origin: Origin,
](src: Span[False, Float32, origin]) -> TensorResult[Float32]:
    # `width` is a positive AOT SIMD specialization supplied by the caller.
    comptime if width <= 0:
        return TensorResult[Float32](TensorStatus.INVALID_SHAPE)
    var count = len(src)
    if count < 0:
        return TensorResult[Float32](TensorStatus.INVALID_SHAPE)
    if count > Int.MAX // 4:
        return TensorResult[Float32](TensorStatus.OVERFLOW)
    # The normal SIMD leaf is faster for small inputs; keep its numerical
    # association and avoid the cost of initializing a parallel runtime.
    if count < 1048576:
        return TensorResult[Float32](cpu_sum_contiguous[width](src))
    var policy = CpuWorkPolicy(262144, 8)
    var workers = recommend_worker_count(count, policy)
    if workers <= 1:
        return TensorResult[Float32](cpu_sum_contiguous[width](src))
    # Maximum workers is 8, so every shard writes a distinct Array cell.
    # Do not move `partials` while its underlying pointer is captured by
    # synchronous parallel work; all workers are joined on return.
    var partials = Array[Float32, 8](fill=Float32(0.0))
    var source_ptr = src.unsafe_ptr()
    var partial_ptr = partials.unsafe_ptr()

    def sum_shard(shard: Int) {imm}:
        # Quotient-and-remainder partition covers [0, count) exactly once;
        # no potentially overflowing (count + workers - 1) arithmetic.
        var q = count // workers
        var rem = count % workers
        var first = shard * q + min(shard, rem)
        var last = first + q
        if shard < rem:
            last += 1
        var length = last - first
        var vector_end = first + length - length % width
        var lanes = SIMD[DType.float32, width](Float32(0.0))
        for i in range(first, vector_end, width):
            lanes += source_ptr.unsafe_load[width=width, alignment=4](i)
        var subtotal = lanes.reduce_add()
        for i in range(vector_end, last):
            subtotal += source_ptr.unsafe_load[alignment=4](i)[0]
        partial_ptr.unsafe_store[alignment=4](shard, subtotal)

    var completed = execute_parallel_shards(sum_shard, count, policy)
    if completed != TensorStatus.OK:
        # No output Tensor has been mutated: a caller can safely retry with
        # the existing scalar/SIMD policy if its runtime rejects dispatch.
        return TensorResult[Float32](completed)
    # Combine partials in a fixed increasing shard order, independent of
    # operating-system thread completion order. FAST_TREE allows this FP32
    # reassociation; STRICT uses cpu_sum_c_strict_fp32 instead.
    var total = Float32(0.0)
    for i in range(workers):
        total += partials[i]
    return TensorResult[Float32](total)


# FAST_TREE all-elements sum for arbitrary rank-static signed/zero-stride
# host mappings. A checked physical Span may be C/F, reversed, sliced or
# broadcast. The named entrypoint repeats its own layout/capacity proof,
# independently of its owning Tensor facade, before any raw pointer load.
# Shards split LOGICAL C indices, so each worker reads only validated input
# cells and writes one private stack partial; no input mutation or atomics.
def cpu_sum_strided_parallel_fast[
    rank: Int, origin: Origin,
](source: Span[False, Float32, origin], mapping: TensorLayout[rank]) -> TensorResult[Float32]:
    comptime if rank < 0 or rank > 8:
        return TensorResult[Float32](TensorStatus.INVALID_SHAPE)
    var checked = mapping.validated_snapshot(len(source))
    if not checked.is_ok():
        return TensorResult[Float32](checked.status)
    var trusted = checked.payload.value()
    var count = trusted.shape.numel()
    # For small/non-parallel work retain the existing C-index scalar fold.
    var workers = 1
    var policy = CpuWorkPolicy(262144, 8)
    if count >= 1048576:
        workers = recommend_worker_count(count, policy)
    if workers <= 1:
        var cursor = NDIndexIterator[rank](trusted.shape)
        var scalar = Float32(0.0)
        var src = source.unsafe_ptr()
        while not cursor.done():
            scalar += src.unsafe_load[alignment=4](
                trusted.physical_offset_unchecked(cursor.coordinates),
            )[0]
            cursor.advance()
        return TensorResult[Float32](scalar)
    var partials = Array[Float32, 8](fill=Float32(0.0))
    var src_ptr = source.unsafe_ptr()
    var part_ptr = partials.unsafe_ptr()

    def reduce_strided_shard(shard: Int) {imm}:
        var q = count // workers
        var rem = count % workers
        var first = shard * q + min(shard, rem)
        var last = first + q
        if shard < rem:
            last += 1
        var cursor = NDIndexIterator[rank](trusted.shape)
        cursor.seek_unchecked(first)
        var subtotal = Float32(0.0)
        for _ in range(first, last):
            subtotal += src_ptr.unsafe_load[alignment=4](
                trusted.physical_offset_unchecked(cursor.coordinates),
            )[0]
            cursor.advance()
        part_ptr.unsafe_store[alignment=4](shard, subtotal)

    var completed = execute_parallel_shards(reduce_strided_shard, count, policy)
    if completed != TensorStatus.OK:
        return TensorResult[Float32](completed)
    var total = Float32(0.0)
    for i in range(workers):
        total += partials[i]
    return TensorResult[Float32](total)


# One-pass C-contiguous all-elements reductions. The owner-bound Tensor
# entrypoint has already proved physical C-order and complete capacity. In
# particular, a rank-zero tensor simply passes count=1. These leaf kernels
# preserve C-logical traversal order, first-NaN behavior and signed-zero
# argmin/argmax tie-breaking without materializing N-D coordinates.
@inline(.always)
def cpu_sum_c_strict_fp32[origin: Origin](
    source: Span[False, Float32, origin], begin: Int, count: Int,
) -> Float32:
    var ptr = source.unsafe_ptr()
    var total = Float32(0.0)
    for i in range(count):
        total += ptr.unsafe_load(begin + i)[0]
    return total


@inline(.always)
def cpu_prod_c_fp32[origin: Origin](
    source: Span[False, Float32, origin], begin: Int, count: Int,
) -> Float32:
    var ptr = source.unsafe_ptr()
    var product = Float32(1.0)
    for i in range(count):
        product *= ptr.unsafe_load(begin + i)[0]
    return product


@inline(.always)
def cpu_extremum_c_fp32[find_min: Bool, origin: Origin](
    source: Span[False, Float32, origin], begin: Int, count: Int,
) -> Float32:
    # PRECONDITION: count >= 1 and every referenced FP32 is initialized.
    var ptr = source.unsafe_ptr()
    var best = ptr.unsafe_load(begin)[0]
    if best != best:
        return best
    for i in range(1, count):
        var value = ptr.unsafe_load(begin + i)[0]
        if value != value:
            return value
        comptime if find_min:
            best = fp32_minimum(best, value)
        else:
            best = fp32_maximum(best, value)
    return best


@inline(.always)
def cpu_argextremum_c_fp32[find_min: Bool, origin: Origin](
    source: Span[False, Float32, origin], begin: Int, count: Int,
) -> Int:
    # PRECONDITION: count >= 1; return the earliest C-logical index of
    # the selected extreme, including the first NaN if present.
    var ptr = source.unsafe_ptr()
    var best = ptr.unsafe_load(begin)[0]
    if best != best:
        return 0
    var selected_index = 0
    for i in range(1, count):
        var value = ptr.unsafe_load(begin + i)[0]
        if value != value:
            return i
        comptime if find_min:
            if fp32_argmin_prefers(value, best):
                best = value
                selected_index = i
        else:
            if fp32_argmax_prefers(value, best):
                best = value
                selected_index = i
    return selected_index


# Single shared FP32 one-axis reduction used by both keepdims and rank-dropping
# Tensor facades. Caller has already validated the complete physical span,
# output size, and nonempty min/max/mean. No allocation, dispatch, cast or
# repeated per-element bounds proof is introduced inside this hot kernel.
@inline(.always)
def cpu_reduce_strided_fp32_fast[operation: Int, origin: Origin](
    src: Span[False, Float32, origin],
    start: Int, step: Int, count: Int,
) -> Float32:
    comptime assert operation >= 0 and operation <= 4, "FP32 reduction opcode"
    comptime if operation == 0 or operation == 4:
        var accumulator = Float32(0.0)
        # A stride of exactly 1 gives a physical contiguous FP32 run even
        # when the surrounding N-D input layout is not globally C-order.
        # Negative/zero/irregular strides retain the scalar-safe path.
        comptime width = simd_width_of[DType.float32]()
        if step == 1 and count >= width:
            var lanes = count - count % width
            var sums = SIMD[DType.float32, width](0.0)
            var ptr = src.unsafe_ptr()
            for i in range(0, lanes, width):
                sums += ptr.unsafe_load[width=width, alignment=4](start + i)
            accumulator = sums.reduce_add()
            for i in range(lanes, count):
                accumulator += src[start + i]
        elif step == -1 and count >= width:
            # A reversed contiguous physical run can be loaded vector-wise.
            # Loading a group in ascending physical order changes only the
            # FAST_TREE summation association, which is already allowed here.
            var lanes = count - count % width
            var sums = SIMD[DType.float32, width](0.0)
            var ptr = src.unsafe_ptr()
            for i in range(0, lanes, width):
                var first_physical = start - i - width + 1
                sums += ptr.unsafe_load[width=width, alignment=4](first_physical)
            accumulator = sums.reduce_add()
            for i in range(lanes, count):
                accumulator += src[start - i]
        else:
            for i in range(count):
                accumulator += src[start + i * step]
        comptime if operation == 4:
            # PRECONDITION: nonempty mean, proven at checked Tensor boundary.
            accumulator /= Float32(count)
        return accumulator
    elif operation == 1:
        var product = Float32(1.0)
        for i in range(count):
            product *= src[start + i * step]
        return product
    else:
        # PRECONDITION: min/max of at least one element. Retain the same
        # NaN-propagation and signed-zero policy as the existing Tensor Core.
        var extreme = src[start]
        for i in range(1, count):
            var value = src[start + i * step]
            comptime if operation == 2:
                extreme = fp32_minimum(extreme, value)
            else:
                extreme = fp32_maximum(extreme, value)
        return extreme

# The canonical axis-wise SUM/MEAN entry point must not perform a horizontal
# SIMD reduction over the reduced dimension. An FP32 vector-tree fold can
# differ numerically from the ordered C-axis recurrence, even with
# --fp-mode contract=off. SIMD across *different output cells* remains legal
# in cpu_reduce_c_axis_block_fp32 because each lane follows increasing K.
# Call cpu_reduce_strided_fp32_fast explicitly to request reassociation.
@inline(.always)
def cpu_reduce_strided_fp32[operation: Int, origin: Origin](
    src: Span[False, Float32, origin],
    start: Int, step: Int, count: Int,
) -> Float32:
    comptime assert operation >= 0 and operation <= 4, "FP32 reduction opcode"
    comptime if operation == 0 or operation == 4:
        var accumulator = Float32(0.0)
        for i in range(count):
            accumulator += src[start + i * step]
        comptime if operation == 4:
            # A nonempty mean was checked by the owning Tensor boundary.
            accumulator /= Float32(count)
        return accumulator
    else:
        # Product/min/max already preserve order and their IEEE policies.
        return cpu_reduce_strided_fp32_fast[operation](src, start, step, count)


# Ordered one-row FP32 reduction for CPU/GPU split execution. The matching
# native GPU baseline accumulates in logical column order, not a SIMD tree;
# preserve that numeric association on CPU, including signed zeros and NaNs.
# No parallel tree or implicit fast-math policy is introduced here.
@inline(.always)
def cpu_reduce_row_ordered_fp32[operation: Int, origin: Origin](
    source: Span[False, Float32, origin], start: Int, columns: Int,
) -> Float32:
    comptime assert operation >= 0 and operation <= 4, "FP32 ordered row reduction"
    comptime if operation == 0 or operation == 4:
        var total = Float32(0.0)
        for j in range(columns):
            total += source[start + j]
        comptime if operation == 4:
            total /= Float32(columns)
        return total
    elif operation == 1:
        var product = Float32(1.0)
        for j in range(columns):
            product *= source[start + j]
        return product
    else:
        # PRECONDITION columns > 0, proven before any CPU/GPU split enqueue.
        var extreme = source[start]
        for j in range(1, columns):
            comptime if operation == 2:
                extreme = fp32_minimum(extreme, source[start + j])
            else:
                extreme = fp32_maximum(extreme, source[start + j])
        return extreme


# AOT-selected dense C-order trailing-axis reduction. Tensor validates the
# input owner/layout; this public named kernel additionally proves the exact
# Span geometry independently before any unchecked native store or worker.
# Each output row owns a distinct contiguous reduction interval, so there is
# no rank-N coordinate carry or intermediate Tensor in this hot path.
# Zero-length reductions produce their valid SUM/PROD identities; callers
# reject a nonempty MIN/MAX/MEAN output with an empty reduced domain.
# Shared fallible boundary for contiguous [outer, reduced, inner] geometry.
# All arithmetic is checked BEFORE reading a source element or publishing
# an output. Raw Span identity remains a caller-supplied host capability;
# this proves numeric bounds and disjointness, not OS address-space mapping.
def _validate_c_reduction_span_boundary[
    operation: Int, source_origin: Origin, target_origin: Origin[mut=True],
](
    source: Span[False, Float32, source_origin],
    target: Span[True, Float32, target_origin],
    source_base: Int, outer: Int, reduced_count: Int, inner: Int,
) -> TensorStatus:
    if outer < 0 or reduced_count < 0 or inner < 0:
        return TensorStatus.INVALID_SHAPE
    if source_base < 0 or source_base > len(source):
        return TensorStatus.OUT_OF_BOUNDS
    var output_count = checked_mul_nonnegative(outer, inner)
    if not output_count.is_ok():
        return output_count.status
    if output_count.payload.value() != len(target):
        return TensorStatus.SIZE_MISMATCH
    var input_count = checked_mul_nonnegative(
        output_count.payload.value(), reduced_count,
    )
    if not input_count.is_ok():
        return input_count.status
    if input_count.payload.value() > len(source) - source_base:
        return TensorStatus.OUT_OF_BOUNDS
    if output_count.payload.value() == 0:
        return TensorStatus.OK
    # An empty reduction reads NO source elements. SUM and PRODUCT only
    # write their identities, so even physically overlapping input/output
    # spans are harmless. Reject empty MIN/MAX/MEAN before any write.
    # This fast path saves address-range checking for every such call.
    if reduced_count == 0:
        comptime if operation == 0 or operation == 1:
            return TensorStatus.OK
        else:
            return TensorStatus.INVALID_SHAPE
    # The common case is two independent allocations. Reject/allow it in
    # one inexpensive whole-span address check without constructing a slice.
    # When the enclosing spans overlap, inspect the *actual* consumed window:
    # a destination may safely reuse unused storage outside that window.
    # The geometry above proves the end addition and slice are in bounds.
    if not host_output_disjoint_from_input(target, source):
        var consumed = input_count.payload.value()
        var input_window = source[source_base:source_base + consumed]
        if not host_output_disjoint_from_input(target, input_window):
            return TensorStatus.ALIAS_UNKNOWN
    return TensorStatus.OK


@inline(.always)
def cpu_reduce_c_trailing_axes_fp32[
    operation: Int, source_origin: Origin, target_origin: Origin[mut=True],
](
    source: Span[False, Float32, source_origin],
    target: Span[True, Float32, target_origin],
    source_base: Int, rows: Int, columns: Int,
) -> TensorStatus:
    comptime assert operation >= 0 and operation <= 4, "FP32 reduction opcode"
    # Recheck the direct kernel boundary, including the zero-length case.
    var verified = _validate_c_reduction_span_boundary[operation](
        source, target, source_base, rows, columns, 1,
    )
    if verified != TensorStatus.OK:
        return verified
    if rows == 0:
        return TensorStatus.OK
    if columns == 0:
        # For empty-K SUM/PROD, write only the identity. No source pointer,
        # MAX worker startup, reduction-loop entry or scratch allocation.
        comptime if operation == 0 or operation == 1:
            var identity = Float32(0.0)
            comptime if operation == 1:
                identity = Float32(1.0)
            for row in range(rows):
                target[row] = identity
            return TensorStatus.OK
        else:
            return TensorStatus.INVALID_SHAPE
    var destination = target.unsafe_ptr()
    # Independent output rows are safe CPU work units even for large
    # reductions. The Tensor boundary has validated the entire source span,
    # destination capacity, and disjointness. No CPU worker writes a shared
    # accumulator: every row is finalized by exactly one worker.
    if columns > 0 and rows > 1:
        var grain_rows = max(1, 262144 // columns)
        var policy = CpuWorkPolicy(grain_rows, 8)
        var workers = recommend_worker_count(rows, policy)
        if workers > 1:
            def reduce_row_shard(shard: Int) {imm}:
                var q = rows // workers
                var rem = rows % workers
                var first = shard * q + min(shard, rem)
                var last = first + q
                if shard < rem:
                    last += 1
                for row in range(first, last):
                    var start = source_base + row * columns
                    var value = cpu_reduce_strided_fp32[operation](
                        source, start, 1, columns,
                    )
                    destination.unsafe_store[alignment=4](row, value)

            # Reject a failed parallel dispatch instead of publishing an
            # output that may not have been fully written.
            return execute_parallel_shards(reduce_row_shard, rows, policy)
    for row in range(rows):
        var start = source_base + row * columns
        var value = cpu_reduce_strided_fp32[operation](
            source, start, 1, columns,
        )
        destination.unsafe_store[alignment=4](row, value)
    return TensorStatus.OK


# Contiguous reduction-axis BLOCK (not necessarily trailing). For a C-order
# tensor with dimensions [outer, reduced..., inner], each output element is a
# strided reduction of `reduced_count` values separated by `inner` FP32s.
# The owner has proved shape products, output independence and writable range
# before entering this kernel. This lowering handles single axes and ANY
# consecutive axis group within ranks 1..8; nonconsecutive/irregular inputs
# still use the fully general Tensor traversal.
def cpu_reduce_c_axis_block_fp32[
    operation: Int, source_origin: Origin, target_origin: Origin[mut=True],
](
    source: Span[False, Float32, source_origin],
    target: Span[True, Float32, target_origin],
    source_base: Int, outer: Int, reduced_count: Int, inner: Int,
) -> TensorStatus:
    comptime assert operation >= 0 and operation <= 4, "FP32 reduction opcode"
    if inner == 1:
        # The trailing entry point owns the common numeric/alias preflight.
        return cpu_reduce_c_trailing_axes_fp32[operation](
            source, target, source_base, outer, reduced_count,
        )
    var verified = _validate_c_reduction_span_boundary[operation](
        source, target, source_base, outer, reduced_count, inner,
    )
    if verified != TensorStatus.OK:
        return verified
    if outer == 0 or inner == 0:
        return TensorStatus.OK
    if reduced_count == 0:
        # Empty-K SUM/PROD is just an identity fill across independent
        # output cells. Do not borrow an empty source pointer or enter the
        # nested [outer, reduced, inner] traversal.
        comptime if operation == 0 or operation == 1:
            var identity = Float32(0.0)
            comptime if operation == 1:
                identity = Float32(1.0)
            for i in range(outer * inner):
                target[i] = identity
            return TensorStatus.OK
        else:
            return TensorStatus.INVALID_SHAPE
    var src = source.unsafe_ptr()
    var dst = target.unsafe_ptr()
    # Sum, product and mean can accumulate contiguous SIMD lanes across the
    # preserved inner axis. This reuses every scanned cache line rather than
    # walking strided columns separately. Min/max intentionally use the
    # scalar IEEE-754 helper to retain NaN / signed-zero rules.
    comptime if operation == 0 or operation == 1 or operation == 4:
        comptime width = simd_width_of[DType.float32]()
        var vector_end = inner - inner % width
        for row in range(outer):
            var base = source_base + row * reduced_count * inner
            var output_base = row * inner
            for col in range(0, vector_end, width):
                var acc = SIMD[DType.float32, width](0.0)
                comptime if operation == 1:
                    acc = SIMD[DType.float32, width](1.0)
                for k in range(reduced_count):
                    var value = src.unsafe_load[width=width, alignment=4](
                        base + k * inner + col,
                    )
                    comptime if operation == 1:
                        acc *= value
                    else:
                        acc += value
                comptime if operation == 4:
                    acc = acc / SIMD[DType.float32, width](Float32(reduced_count))
                dst.unsafe_store[alignment=4](output_base + col, acc)
            for col in range(vector_end, inner):
                var reduced_value = cpu_reduce_strided_fp32[operation](
                    source, base + col, inner, reduced_count,
                )
                dst.unsafe_store[alignment=4](output_base + col, reduced_value)
    else:
        for row in range(outer):
            var base = source_base + row * reduced_count * inner
            for col in range(inner):
                var reduced_value = cpu_reduce_strided_fp32[operation](
                    source, base + col, inner, reduced_count,
                )
                dst.unsafe_store[alignment=4](row * inner + col, reduced_value)
    return TensorStatus.OK


# The four FP32 prefix recurrences are compile-time-specialized. Keeping this
# local to the kernel layer avoids a runtime switch in the innermost loop.
@inline(.always)
def _scan_combine_fp32[operation: Int](previous: Float32, current: Float32) -> Float32:
    comptime assert operation >= 0 and operation <= 3, "FP32 scan opcode must be 0..3"
    comptime if operation == 0:
        return previous + current
    elif operation == 1:
        return previous * current
    elif operation == 2:
        return fp32_minimum(previous, current)
    else:
        return fp32_maximum(previous, current)


# Ordered FP32 prefix scan for validated rank-static strided inputs and a
# unique C-contiguous output. Tensor owns shape/axis/storage validation and
# first-readable publication. This kernel performs no allocation, metadata
# revalidation or per-element TensorResult construction in its hot loop.
def cpu_scan_strided_fp32[
    operation: Int, rank: Int, source_origin: Origin, target_origin: Origin[mut=True],
](
    source: Span[False, Float32, source_origin],
    destination: Span[True, Float32, target_origin],
    mapping: TensorLayout[rank], selected_axis: Int, output_pitch: Int,
) -> TensorStatus:
    # 0..3 inclusive sum/prod/min/max; 4..5 exclusive sum/prod.
    # Specialization prevents a runtime branch in the recurrence loop.
    comptime assert operation >= 0 and operation <= 5, "FP32 scan opcode must be 0..5"
    comptime assert rank >= 1 and rank <= 8, "scan requires an existing axis"
    # Validate even direct calls from outside Tensor: a forged axis/pitch
    # must not reach the `cursor.produced - output_pitch` raw load.
    if selected_axis < 0 or selected_axis >= rank:
        return TensorStatus.INVALID_AXIS
    var mapped = mapping.validated_snapshot(len(source))
    if not mapped.is_ok():
        return mapped.status
    var verified_mapping = mapped.payload.value()
    var count = verified_mapping.shape.numel()
    if len(destination) != count:
        return TensorStatus.SIZE_MISMATCH
    # An empty scan must still reject a forged C-order output pitch.  Do not
    # multiply trailing dimensions here: an empty rank-N shape may contain
    # enormous nonzero dimensions whose product overflows Int, while its
    # canonical C strides legitimately use the zero-extent fallback.
    if count == 0:
        var empty_strides = c_strides[rank](verified_mapping.shape)
        if not empty_strides.is_ok():
            return empty_strides.status
        if output_pitch != empty_strides.payload.value().get(selected_axis):
            return TensorStatus.INVALID_LAYOUT
        return TensorStatus.OK
    var expected_pitch = 1
    comptime for axis in range(rank):
        if axis > selected_axis:
            var stride_part = checked_mul_nonnegative(
                expected_pitch, verified_mapping.shape.dims.get(axis),
            )
            if not stride_part.is_ok():
                return stride_part.status
            expected_pitch = stride_part.payload.value()
    if output_pitch != expected_pitch:
        return TensorStatus.INVALID_LAYOUT
    # Most scans use independent owners: keep that preflight O(1). For a
    # source Span covering unused owner storage, the proven physical envelope
    # [span_lo, span_hi] contains every read for arbitrary signed/zero strides.
    # Recheck that narrower envelope only when the whole-span test overlaps;
    # this permits safe reuse of bytes outside the input view without a copy.
    if not host_output_disjoint_from_input(destination, source):
        var input_envelope = source[
            verified_mapping.span_lo:verified_mapping.span_hi + 1
        ]
        if not host_output_disjoint_from_input(destination, input_envelope):
            return TensorStatus.ALIAS_UNKNOWN
    # Both signed input strides and the output's physical C-order address
    # range are established once. Inner loops retain their prior SIMD policy.
    var input_pointer = source.unsafe_ptr()
    var output_pointer = destination.unsafe_ptr()
    comptime if rank == 1:
        # Rank-one input needs no N-D cursor or per-element carry. A signed
        # or zero source stride remains legal; destination is unique dense.
        var begin = verified_mapping.offset
        var step = verified_mapping.strides.get(0)
        comptime if operation >= 4:
            var prefix = Float32(0)
            comptime if operation == 5:
                prefix = Float32(1)
            for i in range(count):
                # Read before any write; source/destination are disjoint.
                var current = input_pointer.unsafe_load(begin + i * step)[0]
                output_pointer.unsafe_store[alignment=4](i, prefix)
                prefix = _scan_combine_fp32[operation - 4](prefix, current)
        else:
            var prefix = input_pointer.unsafe_load(begin)[0]
            output_pointer.unsafe_store[alignment=4](0, prefix)
            for i in range(1, count):
                prefix = _scan_combine_fp32[operation](
                    prefix, input_pointer.unsafe_load(begin + i * step)[0],
                )
                output_pointer.unsafe_store[alignment=4](i, prefix)
        return TensorStatus.OK
    else:
        # Common inner-axis C-order case: one sequential scan per dense row.
        # This eliminates eight-dimensional index carry and repeated affine
        # address arithmetic from the hot loop. Nonlast/F/signed-strided
        # inputs still use the fully general rank-static fallback below.
        if selected_axis == rank - 1 and verified_mapping.is_c_contiguous():
            var columns = verified_mapping.shape.dims.get(rank - 1)
            if columns > 0:
                var rows = count // columns
                var input_begin = verified_mapping.span_lo
                # Preserve EXACT ordered scan arithmetic within each row;
                # only independent rows are parallelized. Thus no
                # reassociation or global prefix synchronization is needed.
                if rows > 1:
                    var grain_rows = max(1, 262144 // columns)
                    var policy = CpuWorkPolicy(grain_rows, 8)
                    var workers = recommend_worker_count(rows, policy)
                    if workers > 1:
                        def scan_row_shard(shard: Int) {imm}:
                            var q = rows // workers
                            var rem = rows % workers
                            var first = shard * q + min(shard, rem)
                            var last = first + q
                            if shard < rem:
                                last += 1
                            for row in range(first, last):
                                var output_begin = row * columns
                                var source_begin = input_begin + output_begin
                                comptime if operation >= 4:
                                    var prefix = Float32(0)
                                    comptime if operation == 5:
                                        prefix = Float32(1)
                                    for j in range(columns):
                                        var current = input_pointer.unsafe_load(source_begin + j)[0]
                                        output_pointer.unsafe_store[alignment=4](output_begin + j, prefix)
                                        prefix = _scan_combine_fp32[operation - 4](prefix, current)
                                else:
                                    var prefix = input_pointer.unsafe_load(source_begin)[0]
                                    output_pointer.unsafe_store[alignment=4](output_begin, prefix)
                                    for j in range(1, columns):
                                        prefix = _scan_combine_fp32[operation](
                                            prefix, input_pointer.unsafe_load(source_begin + j)[0],
                                        )
                                        output_pointer.unsafe_store[alignment=4](output_begin + j, prefix)

                        # A rejected dispatch must never publish the fresh
                        # output as completely initialized.
                        return execute_parallel_shards(scan_row_shard, rows, policy)
                for row in range(rows):
                    var output_begin = row * columns
                    var source_begin = input_begin + output_begin
                    comptime if operation >= 4:
                        var prefix = Float32(0)
                        comptime if operation == 5:
                            prefix = Float32(1)
                        for j in range(columns):
                            var current = input_pointer.unsafe_load(source_begin + j)[0]
                            output_pointer.unsafe_store[alignment=4](output_begin + j, prefix)
                            prefix = _scan_combine_fp32[operation - 4](prefix, current)
                    else:
                        var prefix = input_pointer.unsafe_load(source_begin)[0]
                        output_pointer.unsafe_store[alignment=4](output_begin, prefix)
                        for j in range(1, columns):
                            prefix = _scan_combine_fp32[operation](
                                prefix, input_pointer.unsafe_load(source_begin + j)[0],
                            )
                            output_pointer.unsafe_store[alignment=4](output_begin + j, prefix)
                return TensorStatus.OK
        # C-contiguous non-trailing-axis fast path. Both source and output
        # are linear runs; the selected axis has a compile-time-ranked but
        # runtime-sized inner span. Process [outer, axis, inner] directly,
        # avoiding an N-D cursor and affine recomputation for every element.
        # The per-line recurrence and IEEE extrema order are unchanged.
        if selected_axis < rank - 1 and verified_mapping.is_c_contiguous():
            var extent = verified_mapping.shape.dims.get(selected_axis)
            var inner_span = output_pitch
            var outer_span = extent * inner_span
            var input_begin = verified_mapping.span_lo
            var outer_begin = 0
            while outer_begin < count:
                var axis_index = 0
                while axis_index < extent:
                    var row_begin = outer_begin + axis_index * inner_span
                    var inner_index = 0
                    while inner_index < inner_span:
                        var logical_index = row_begin + inner_index
                        var next_value = Float32(0)
                        comptime if operation >= 4:
                            comptime if operation == 5:
                                next_value = Float32(1)
                            if axis_index != 0:
                                next_value = _scan_combine_fp32[operation - 4](
                                    output_pointer.unsafe_load(logical_index - inner_span)[0],
                                    input_pointer.unsafe_load(
                                        input_begin + logical_index - inner_span
                                    )[0],
                                )
                        else:
                            next_value = input_pointer.unsafe_load(
                                input_begin + logical_index
                            )[0]
                            if axis_index != 0:
                                next_value = _scan_combine_fp32[operation](
                                    output_pointer.unsafe_load(logical_index - inner_span)[0],
                                    next_value,
                                )
                        output_pointer.unsafe_store[alignment=4](
                            logical_index, next_value,
                        )
                        inner_index += 1
                    axis_index += 1
                outer_begin += outer_span
            return TensorStatus.OK
        # Generic signed/zero/F-strided fallback for rank 2..8 and any axis.
        var cursor = NDIndexIterator[rank](verified_mapping.shape)
        while not cursor.done():
            var position = verified_mapping.physical_offset_unchecked(cursor.coordinates)
            var next_value = Float32(0)
            comptime if operation >= 4:
                comptime if operation == 5:
                    next_value = Float32(1)
                if cursor.coordinates.get(selected_axis) != 0:
                    next_value = _scan_combine_fp32[operation - 4](
                        output_pointer.unsafe_load(cursor.produced - output_pitch)[0],
                        input_pointer.unsafe_load(
                            position - verified_mapping.strides.get(selected_axis)
                        )[0],
                    )
            else:
                next_value = input_pointer.unsafe_load(position)[0]
                if cursor.coordinates.get(selected_axis) != 0:
                    next_value = _scan_combine_fp32[operation](
                        output_pointer.unsafe_load(cursor.produced - output_pitch)[0],
                        next_value,
                    )
            output_pointer.unsafe_store[alignment=4](cursor.produced, next_value)
            cursor.advance()
    return TensorStatus.OK
