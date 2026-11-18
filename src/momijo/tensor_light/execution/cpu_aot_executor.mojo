# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR EXECUTION / CPU AOT EXECUTOR
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Execution Planning and AOT
# Module:        momijo.tensor_light.execution.cpu_aot_executor
# Source:        src/momijo/tensor_light/execution/cpu_aot_executor.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines CPU AOT executor within checked AOT execution planning for the
#   FP32 Light implementation.
#
#   Public declarations include `CpuDataflowRunOutcome`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Prepare or execute source-defined tensor plans with validated bindings,
#     resource lifetimes, and explicit completion requirements.
#   - Preserve documented behavior for immutable plan metadata and trusted
#     dispatch preconditions.
#   - Provide source-declared interfaces such as `CpuDataflowRunOutcome`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Bind plans to validated owners, layouts, and execution contexts.
#   - Do not treat an AOT plan or event handle as proof of native completion.
#
# Performance Objectives:
#   - Hoist preparation and repeated metadata checks out of hot paths.
#   - Retain fused or borrowed execution paths only with validated
#     preconditions.
#
# Maintainability:
#   - Keep CPU, GPU, and hybrid routing contracts explicit.
#   - Test stale binding, failed joins, and unsupported targets.
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

# Bounded, owner-backed CPU lowering for the SAME sealed FP32 static dataflow IR.
# No JIT, runtime callback registry, implicit device migration or allocation.
# The caller passes one already-initialized, C-contiguous owning Tensor arena.
# Every declared slot is an independently addressable, non-overlapping subrange
# of that same owner. The checked boundary proves all byte ranges before any
# scalar or SIMD store. This is a real CPU execution path, not a GPU/Hybrid
# artifact loader or a claim of universal native AOT coverage.
from std.collections import Span
from std.math import fma as std_fma, abs as std_abs, sqrt as std_sqrt
from std.sys.info import simd_width_of
from ..dtype.fp32 import fp32_maximum, fp32_minimum
from ..kernels.cpu.unary import cpu_unary_scalar_fp32
from ..internal.status import TensorStatus
from .static_dataflow import SealedStaticDataflowPlan, DataflowNode, DataflowRegion


@inline(.always)
def _cpu_binary_fp32[operation: Int](left: Float32, right: Float32) -> Float32:
    comptime if operation == 0:
        return left + right
    elif operation == 1:
        return left - right
    elif operation == 2:
        return left * right
    elif operation == 3:
        return left / right
    elif operation == 4:
        return fp32_maximum(left, right)
    else:
        return fp32_minimum(left, right)


@inline(.always)
def _cpu_binary_simd_fp32[operation: Int, width: Int](
    left: SIMD[DType.float32, width], right: SIMD[DType.float32, width],
) -> SIMD[DType.float32, width]:
    comptime if operation == 0:
        return left + right
    elif operation == 1:
        return left - right
    elif operation == 2:
        return left * right
    else:
        return left / right


@inline(.always)
def _cpu_dataflow_binary[operation: Int, origin: Origin[mut=True]](
    mut arena: Span[True, Float32, origin], source_a: Int, source_b: Int,
    dest: Int, count: Int, scalar_a: Bool, scalar_b: Bool,
):
    # AOT-specialized arithmetic opcode. The per-node scalar broadcast flags
    # are hoisted OUTSIDE vector loops, including left/right division order.
    comptime width = simd_width_of[DType.float32]()
    comptime if operation <= 3:
        var p = arena.unsafe_ptr()
        var vector_end = count - count % width
        if scalar_a:
            var a_lanes = SIMD[DType.float32, width](arena[source_a])
            if scalar_b:
                var b_lanes = SIMD[DType.float32, width](arena[source_b])
                var result = _cpu_binary_simd_fp32[operation, width](a_lanes, b_lanes)
                for i in range(0, vector_end, width):
                    p.unsafe_store[alignment=4](i + dest, result)
            else:
                for i in range(0, vector_end, width):
                    var b_lanes = p.unsafe_load[width=width, alignment=4](source_b + i)
                    p.unsafe_store[alignment=4](i + dest, _cpu_binary_simd_fp32[operation, width](a_lanes, b_lanes))
        else:
            if scalar_b:
                var b_lanes = SIMD[DType.float32, width](arena[source_b])
                for i in range(0, vector_end, width):
                    var a_lanes = p.unsafe_load[width=width, alignment=4](source_a + i)
                    p.unsafe_store[alignment=4](i + dest, _cpu_binary_simd_fp32[operation, width](a_lanes, b_lanes))
            else:
                for i in range(0, vector_end, width):
                    var a_lanes = p.unsafe_load[width=width, alignment=4](source_a + i)
                    var b_lanes = p.unsafe_load[width=width, alignment=4](source_b + i)
                    p.unsafe_store[alignment=4](i + dest, _cpu_binary_simd_fp32[operation, width](a_lanes, b_lanes))
        for i in range(vector_end, count):
            var ai = source_a
            var bi = source_b
            if not scalar_a:
                ai += i
            if not scalar_b:
                bi += i
            arena[dest + i] = _cpu_binary_fp32[operation](arena[ai], arena[bi])
    else:
        # SIMD maximum/minimum instructions may have target-dependent NaN
        # and signed-zero semantics. Preserve the canonical FP32 policy.
        for i in range(count):
            var ai = source_a
            var bi = source_b
            if not scalar_a:
                ai += i
            if not scalar_b:
                bi += i
            arena[dest + i] = _cpu_binary_fp32[operation](arena[ai], arena[bi])


@inline(.always)
def _cpu_dataflow_unary[operation: Int, origin: Origin[mut=True]](
    mut arena: Span[True, Float32, origin], source: Int, dest: Int, count: Int,
):
    # The sealed graph and owner-backed arena prove disjoint input/output
    # regions. Vectorize only operations with an established FP32 SIMD
    # implementation. Keep transcendental functions in their scalar path;
    # never substitute an unverified approximate vector intrinsic.
    comptime if operation == 0 or operation == 1 or operation == 2 or operation == 8:
        comptime width = simd_width_of[DType.float32]()
        var pointer = arena.unsafe_ptr()
        var vector_end = count - count % width
        for i in range(0, vector_end, width):
            var value = pointer.unsafe_load[width=width, alignment=4](source + i)
            comptime if operation == 0:
                pointer.unsafe_store[alignment=4](dest + i, -value)
            elif operation == 1:
                pointer.unsafe_store[alignment=4](dest + i, std_abs(value))
            elif operation == 2:
                pointer.unsafe_store[alignment=4](dest + i, std_sqrt(value))
            else:
                pointer.unsafe_store[alignment=4](
                    dest + i, SIMD[DType.float32, width](1.0) / value,
                )
        for i in range(vector_end, count):
            arena[dest + i] = cpu_unary_scalar_fp32[operation](arena[source + i])
    else:
        for i in range(count):
            arena[dest + i] = cpu_unary_scalar_fp32[operation](arena[source + i])


@inline(.always)
def _cpu_dataflow_reduce[operation: Int, origin: Origin[mut=True]](
    mut arena: Span[True, Float32, origin], source: Int, dest: Int,
    output_count: Int, extent: Int,
):
    # The sealed DAG proves extent > 0 and input_count=output_count*extent.
    for row in range(output_count):
        var base = source + row * extent
        # Match the canonical FP32 reduction identities used by Tensor and
        # kernels/cpu/reduction.mojo. A zero-start SUM/MEAN and one-start PROD
        # matter for signed-zero semantics; the previous first-element seed
        # silently returned -0 for a single -0 SUM instead of canonical +0.
        var accumulator = Float32(0.0)
        var first_column = 0
        comptime if operation == 1:
            accumulator = Float32(1.0)
        elif operation == 2 or operation == 3:
            # MIN/MAX require a real first element and preserve the common
            # NaN and signed-zero extrema rules.
            accumulator = arena[base]
            first_column = 1
        for column in range(first_column, extent):
            var value = arena[base + column]
            comptime if operation == 0 or operation == 4:
                accumulator += value
            elif operation == 1:
                accumulator *= value
            elif operation == 2:
                accumulator = fp32_minimum(accumulator, value)
            else:
                accumulator = fp32_maximum(accumulator, value)
        comptime if operation == 4:
            accumulator /= Float32(extent)
        arena[dest + row] = accumulator


@inline(.always)
def _cpu_dataflow_scan[operation: Int, origin: Origin[mut=True]](
    mut arena: Span[True, Float32, origin], source: Int, dest: Int,
    count: Int, row_length: Int,
):
    for row in range(count // row_length):
        var begin = row * row_length
        var accumulator = arena[source + begin]
        arena[dest + begin] = accumulator
        for j in range(1, row_length):
            var value = arena[source + begin + j]
            comptime if operation == 0:
                accumulator += value
            elif operation == 1:
                accumulator *= value
            elif operation == 2:
                accumulator = fp32_minimum(accumulator, value)
            else:
                accumulator = fp32_maximum(accumulator, value)
            arena[dest + begin + j] = accumulator



@inline(.always)
def _cpu_dataflow_gemm[use_fma: Bool, origin: Origin[mut=True]](
    mut arena: Span[True, Float32, origin], left_begin: Int,
    right_begin: Int, dest: Int, m: Int, k: Int, n: Int,
):
    # Bounded dense row-major lowering for rank-2 FP32 GEMM. Four adjacent
    # output rows reuse each RHS SIMD vector instead of reloading B separately
    # for all four scalar rows. This is an allocation-free register microkernel
    # and a source-level speed candidate, not a benchmark-derived optimum.
    # Contract: owner-checked, disjoint A/B/C spans, with m*k, k*n, m*n in
    # bounds, is already established before entering this unchecked function.
    comptime width = simd_width_of[DType.float32]()
    var ptr = arena.unsafe_ptr()
    var vector_n = n - n % width
    var full_rows = m - m % 4

    for base_row in range(0, full_rows, 4):
        for column in range(0, vector_n, width):
            var c0 = SIMD[DType.float32, width](0.0)
            var c1 = SIMD[DType.float32, width](0.0)
            var c2 = SIMD[DType.float32, width](0.0)
            var c3 = SIMD[DType.float32, width](0.0)
            for inner in range(k):
                var rhs = ptr.unsafe_load[width=width, alignment=4](
                    right_begin + inner * n + column
                )
                var a0 = SIMD[DType.float32, width](
                    arena[left_begin + base_row * k + inner]
                )
                var a1 = SIMD[DType.float32, width](
                    arena[left_begin + (base_row + 1) * k + inner]
                )
                var a2 = SIMD[DType.float32, width](
                    arena[left_begin + (base_row + 2) * k + inner]
                )
                var a3 = SIMD[DType.float32, width](
                    arena[left_begin + (base_row + 3) * k + inner]
                )
                comptime if use_fma:
                    c0 = std_fma(a0, rhs, c0)
                    c1 = std_fma(a1, rhs, c1)
                    c2 = std_fma(a2, rhs, c2)
                    c3 = std_fma(a3, rhs, c3)
                else:
                    c0 += a0 * rhs
                    c1 += a1 * rhs
                    c2 += a2 * rhs
                    c3 += a3 * rhs
            ptr.unsafe_store[alignment=4](dest + base_row * n + column, c0)
            ptr.unsafe_store[alignment=4](dest + (base_row + 1) * n + column, c1)
            ptr.unsafe_store[alignment=4](dest + (base_row + 2) * n + column, c2)
            ptr.unsafe_store[alignment=4](dest + (base_row + 3) * n + column, c3)

        # SIMD tails are scalar and stay inside the validated row extent.
        for column in range(vector_n, n):
            var s0 = Float32(0.0)
            var s1 = Float32(0.0)
            var s2 = Float32(0.0)
            var s3 = Float32(0.0)
            for inner in range(k):
                var rhs = arena[right_begin + inner * n + column]
                comptime if use_fma:
                    s0 = std_fma(arena[left_begin + base_row * k + inner], rhs, s0)
                    s1 = std_fma(arena[left_begin + (base_row + 1) * k + inner], rhs, s1)
                    s2 = std_fma(arena[left_begin + (base_row + 2) * k + inner], rhs, s2)
                    s3 = std_fma(arena[left_begin + (base_row + 3) * k + inner], rhs, s3)
                else:
                    s0 += arena[left_begin + base_row * k + inner] * rhs
                    s1 += arena[left_begin + (base_row + 1) * k + inner] * rhs
                    s2 += arena[left_begin + (base_row + 2) * k + inner] * rhs
                    s3 += arena[left_begin + (base_row + 3) * k + inner] * rhs
            arena[dest + base_row * n + column] = s0
            arena[dest + (base_row + 1) * n + column] = s1
            arena[dest + (base_row + 2) * n + column] = s2
            arena[dest + (base_row + 3) * n + column] = s3

    for row in range(full_rows, m):
        for column in range(0, vector_n, width):
            var accumulator = SIMD[DType.float32, width](0.0)
            for inner in range(k):
                var scalar = SIMD[DType.float32, width](
                    arena[left_begin + row * k + inner]
                )
                var values = ptr.unsafe_load[width=width, alignment=4](
                    right_begin + inner * n + column
                )
                comptime if use_fma:
                    accumulator = std_fma(scalar, values, accumulator)
                else:
                    accumulator += scalar * values
            ptr.unsafe_store[alignment=4](dest + row * n + column, accumulator)
        for column in range(vector_n, n):
            var accumulator = Float32(0.0)
            for inner in range(k):
                var av = arena[left_begin + row * k + inner]
                var bv = arena[right_begin + inner * n + column]
                comptime if use_fma:
                    accumulator = std_fma(av, bv, accumulator)
                else:
                    accumulator += av * bv
            arena[dest + row * n + column] = accumulator


def _cpu_dataflow_single_node[origin: Origin[mut=True], slots: Int](
    mut arena: Span[True, Float32, origin], node: DataflowNode,
    offsets: Array[Int, slots],
) -> TensorStatus:
    # No fallible operations or metadata validation are placed in these
    # element loops. The executor boundary already sealed the DAG and arena.
    var dst = offsets[node.write.slot] + node.write.begin
    var output_count = node.write.end - node.write.begin
    if node.primitive == 8:
        for i in range(output_count):
            arena[dst + i] = node.fill_value
        return TensorStatus.OK
    var a = offsets[node.read_a.slot] + node.read_a.begin
    if node.primitive == 0:
        # Vectorize only after the source/destination ranges have been proven
        # disjoint inside one real owner-backed host arena.
        comptime width = simd_width_of[DType.float32]()
        var p = arena.unsafe_ptr()
        var end = output_count - output_count % width
        for i in range(0, end, width):
            var lane = p.unsafe_load[width=width, alignment=4](a + i)
            p.unsafe_store[alignment=4](dst + i, lane)
        for i in range(end, output_count):
            arena[dst + i] = arena[a + i]
        return TensorStatus.OK
    if node.primitive == 1:
        if node.opcode == 0:
            _cpu_dataflow_unary[0](arena, a, dst, output_count)
        elif node.opcode == 1:
            _cpu_dataflow_unary[1](arena, a, dst, output_count)
        elif node.opcode == 2:
            _cpu_dataflow_unary[2](arena, a, dst, output_count)
        elif node.opcode == 3:
            _cpu_dataflow_unary[3](arena, a, dst, output_count)
        elif node.opcode == 4:
            _cpu_dataflow_unary[4](arena, a, dst, output_count)
        elif node.opcode == 5:
            _cpu_dataflow_unary[5](arena, a, dst, output_count)
        elif node.opcode == 6:
            _cpu_dataflow_unary[6](arena, a, dst, output_count)
        elif node.opcode == 7:
            _cpu_dataflow_unary[7](arena, a, dst, output_count)
        else:
            _cpu_dataflow_unary[8](arena, a, dst, output_count)
        return TensorStatus.OK
    if node.primitive == 2:
        var b = offsets[node.read_b.slot] + node.read_b.begin
        var scalar_a = node.read_a.end - node.read_a.begin == 1
        var scalar_b = node.read_b.end - node.read_b.begin == 1
        if node.opcode == 0:
            _cpu_dataflow_binary[0](arena, a, b, dst, output_count, scalar_a, scalar_b)
        elif node.opcode == 1:
            _cpu_dataflow_binary[1](arena, a, b, dst, output_count, scalar_a, scalar_b)
        elif node.opcode == 2:
            _cpu_dataflow_binary[2](arena, a, b, dst, output_count, scalar_a, scalar_b)
        elif node.opcode == 3:
            _cpu_dataflow_binary[3](arena, a, b, dst, output_count, scalar_a, scalar_b)
        elif node.opcode == 4:
            _cpu_dataflow_binary[4](arena, a, b, dst, output_count, scalar_a, scalar_b)
        else:
            _cpu_dataflow_binary[5](arena, a, b, dst, output_count, scalar_a, scalar_b)
        return TensorStatus.OK
    if node.primitive == 3:
        var b = offsets[node.read_b.slot] + node.read_b.begin
        var c = offsets[node.read_c.slot] + node.read_c.begin
        var scalar_a = node.read_a.end - node.read_a.begin == 1
        var scalar_b = node.read_b.end - node.read_b.begin == 1
        var scalar_c = node.read_c.end - node.read_c.begin == 1
        comptime width = simd_width_of[DType.float32]()
        var end = output_count - output_count % width
        var ptr = arena.unsafe_ptr()
        for i in range(0, end, width):
            var ai = a
            var bi = b
            var ci = c
            if not scalar_a:
                ai += i
            if not scalar_b:
                bi += i
            if not scalar_c:
                ci += i
            var av = SIMD[DType.float32, width](arena[ai])
            var bv = SIMD[DType.float32, width](arena[bi])
            var cv = SIMD[DType.float32, width](arena[ci])
            if not scalar_a:
                av = ptr.unsafe_load[width=width, alignment=4](ai)
            if not scalar_b:
                bv = ptr.unsafe_load[width=width, alignment=4](bi)
            if not scalar_c:
                cv = ptr.unsafe_load[width=width, alignment=4](ci)
            ptr.unsafe_store[alignment=4](dst + i, std_fma(av, bv, cv))
        for i in range(end, output_count):
            var ai = a
            var bi = b
            var ci = c
            if not scalar_a:
                ai += i
            if not scalar_b:
                bi += i
            if not scalar_c:
                ci += i
            arena[dst + i] = std_fma(arena[ai], arena[bi], arena[ci])
        return TensorStatus.OK
    if node.primitive == 4:
        if node.opcode == 0:
            _cpu_dataflow_reduce[0](arena, a, dst, output_count, node.reduction_extent)
        elif node.opcode == 1:
            _cpu_dataflow_reduce[1](arena, a, dst, output_count, node.reduction_extent)
        elif node.opcode == 2:
            _cpu_dataflow_reduce[2](arena, a, dst, output_count, node.reduction_extent)
        elif node.opcode == 3:
            _cpu_dataflow_reduce[3](arena, a, dst, output_count, node.reduction_extent)
        else:
            _cpu_dataflow_reduce[4](arena, a, dst, output_count, node.reduction_extent)
        return TensorStatus.OK
    if node.primitive == 5:
        if node.opcode == 0:
            _cpu_dataflow_scan[0](arena, a, dst, output_count, node.scan_row_length)
        elif node.opcode == 1:
            _cpu_dataflow_scan[1](arena, a, dst, output_count, node.scan_row_length)
        elif node.opcode == 2:
            _cpu_dataflow_scan[2](arena, a, dst, output_count, node.scan_row_length)
        else:
            _cpu_dataflow_scan[3](arena, a, dst, output_count, node.scan_row_length)
        return TensorStatus.OK
    if node.primitive == 6:
        var b = offsets[node.read_b.slot] + node.read_b.begin
        if node.math_mode == 0:
            _cpu_dataflow_gemm[False](
                arena, a, b, dst, node.matrix_m, node.matrix_k, node.matrix_n,
            )
        else:
            _cpu_dataflow_gemm[True](
                arena, a, b, dst, node.matrix_m, node.matrix_k, node.matrix_n,
            )
        return TensorStatus.OK
    if node.primitive == 7:
        var written = 0
        for i in range(node.read_a.end - node.read_a.begin):
            arena[dst + written] = arena[a + i]
            written += 1
        if node.read_b.active():
            var b = offsets[node.read_b.slot] + node.read_b.begin
            for i in range(node.read_b.end - node.read_b.begin):
                arena[dst + written] = arena[b + i]
                written += 1
        if node.read_c.active():
            var c = offsets[node.read_c.slot] + node.read_c.begin
            for i in range(node.read_c.end - node.read_c.begin):
                arena[dst + written] = arena[c + i]
                written += 1
        return TensorStatus.OK
    return TensorStatus.UNSUPPORTED


@fieldwise_init
struct CpuDataflowRunOutcome(Copyable):
    # False means ALL rejections occurred before the first arena write.
    # True means any later failure may have partially overwritten the owner.
    var status: TensorStatus
    var writes_may_have_started: Bool


def _run_cpu_static_dataflow_in_arena[
    nodes: Int, slots: Int, origin: Origin[mut=True],
](
    plan: SealedStaticDataflowPlan[nodes, slots],
    mut arena: Span[True, Float32, origin],
    offsets: Array[Int, slots],
) -> CpuDataflowRunOutcome:
    # Recheck BEFORE the first write; the mutable source-level plan is not
    # treated as immutable until a private checked snapshot has been copied.
    var validated = plan.checked_snapshot()
    if not validated.is_ok():
        return CpuDataflowRunOutcome(validated.status, False)
    var sealed = validated.payload.take()
    # Freeze the fixed-capacity placement table once: checked addresses and
    # later stores must use the SAME offsets, never a mutable caller alias.
    var physical_offsets = offsets.copy()
    var size = len(arena)
    for i in range(sealed._cpu_lowering_used_slots()):
        var slot = sealed._cpu_lowering_slot_unchecked(i)
        # This lowering has no device-buffer, DMA or pinned-stream ABI.
        if slot.domain != 0:
            return CpuDataflowRunOutcome(TensorStatus.DEVICE_MISMATCH, False)
        var begin = physical_offsets[i]
        if begin < 0 or begin > size:
            return CpuDataflowRunOutcome(TensorStatus.OUT_OF_BOUNDS, False)
        if slot.capacity_elements < 0 or slot.capacity_elements > size - begin:
            return CpuDataflowRunOutcome(TensorStatus.OUT_OF_BOUNDS, False)
        for j in range(i):
            var earlier = sealed._cpu_lowering_slot_unchecked(j)
            var other = physical_offsets[j]
            # A single owned arena supplies the real allocation provenance;
            # distinct slot IDs become disjoint only after this span proof.
            if (slot.capacity_elements > 0 and earlier.capacity_elements > 0
                and begin < other + earlier.capacity_elements
                and other < begin + slot.capacity_elements):
                return CpuDataflowRunOutcome(TensorStatus.OVERLAPPING, False)
    for i in range(sealed.node_count()):
        var stage = sealed._cpu_lowering_node_unchecked(i)
        if stage.kind != 0 and stage.kind != 5 and stage.kind != 6:
            return CpuDataflowRunOutcome(TensorStatus.UNSUPPORTED, False)
        # No scratch implementation has been authenticated by this CPU ABI.
        if stage.scratch_bytes != 0:
            return CpuDataflowRunOutcome(TensorStatus.UNSUPPORTED, False)
        # Variant zero has an exact, precompiled CPU callable in this module.
        # Reject other variant identities: silently executing a different
        # kernel would break the immutable AOT specialization contract.
        if stage.aot_variant != 0:
            return CpuDataflowRunOutcome(TensorStatus.UNSUPPORTED, False)
    # ALL preconditions now hold. Topological ID order and validated
    # predecessor masks yield an immutable deterministic sequential lowering.
    for i in range(sealed.node_count()):
        var node = sealed._cpu_lowering_node_unchecked(i)
        var status = _cpu_dataflow_single_node(arena, node, physical_offsets)
        if status != TensorStatus.OK:
            return CpuDataflowRunOutcome(status, True)
    # A sealed graph with no nodes performed no writes. Preserve precise
    # execution-phase provenance even on a successful empty plan.
    return CpuDataflowRunOutcome(TensorStatus.OK, sealed.node_count() > 0)
