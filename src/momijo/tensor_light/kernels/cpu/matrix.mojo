# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | CPU KERNEL / MATRIX OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / CPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.cpu.matrix
# Source:        src/momijo/tensor_light/kernels/cpu/matrix.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines matrix operations for the CPU tensor kernel layer.
#
#   Public declarations include `validate_fp32_gemm_buffers`,
#   `cpu_mm_contiguous`, `cpu_dot_contiguous`, `cpu_gemv_contiguous`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for matrix dimensional contracts,
#     reductions, and edge sizes.
#   - Provide source-declared interfaces such as `validate_fp32_gemm_buffers`,
#     `cpu_mm_contiguous`, `cpu_dot_contiguous`, `cpu_gemv_contiguous`,
#     `cpu_vecmat_contiguous` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Respect the validated layout, bounds, and dtype contract.
#   - Protect scalar remainder and noncontiguous cases.
#   - Handle degenerate matrix dimensions and accumulation behavior
#     explicitly.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Use vectorized/parallel paths only when preconditions hold.
#   - Avoid repeated dispatch decisions in per-element loops.
#   - Keep tiled/contiguous optimizations behind verified layout
#     preconditions.
#
# Maintainability:
#   - Separate launch/ABI declarations from verified execution.
#   - Benchmark against stable, correct reference paths.
#   - Test tails, transposes, zero-K, and nonuniform operands.
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

# Fixed-FP32 rank-2 GEMM kernel: 4-row register block and N-lane SIMD tile.
# Each loaded B SIMD vector is reused by four independent accumulators.
# Preconditions: C-order contiguous FP32 inputs/outputs, correct extents,
# proven-disjoint output, initialized inputs, full output overwrite.
# Matrix packing/cache-level macro tiling and TF32 are separate AOT candidates.
from std.collections import Span
from ...storage.host_storage import HostStorage
from ...dtype.math_mode import Fp32MathMode
from ...dtype.fp32 import fp32_maximum
from ...internal.status import TensorStatus, TensorResult
from ...alias.overlap import host_output_disjoint_from_input, host_mutable_fp32_spans_disjoint
from ...parallel.parallel_for import CpuWorkPolicy, recommend_worker_count, execute_parallel_shards

# The checked FP32 GEMM boundary validates geometry, physical extents and
# output/input independence ONCE before selecting the fast microkernel.
# Kernels with an `_unchecked` execution contract never repeat these checks.
def validate_fp32_gemm_buffers[co: Origin[mut=True], ao: Origin, bo: Origin](
    c: Span[True, Float32, co], a: Span[False, Float32, ao],
    b: Span[False, Float32, bo], m: Int, k: Int, n: Int,
) -> TensorStatus:
    if m < 0 or k < 0 or n < 0:
        return TensorStatus.INVALID_SHAPE
    if (n > 0 and m > (Int.MAX // 4) // n
        or k > 0 and m > (Int.MAX // 4) // k
        or n > 0 and k > (Int.MAX // 4) // n):
        return TensorStatus.OVERFLOW
    var out_count = m * n
    var a_count = m * k
    var b_count = k * n
    if out_count > len(c) or a_count > len(a) or b_count > len(b):
        return TensorStatus.SIZE_MISMATCH
    if out_count == 0 or k == 0:
        # Empty output performs no work. K=0 only writes zeros; it does not
        # read either input, so an output/input alias cannot corrupt a read.
        return TensorStatus.OK
    # Compare precisely the logical slices, not padded host allocations.
    # A/B may overlap EACH OTHER; the result must be disjoint from BOTH.
    var output_run = c[0:out_count]
    var left_run = a[0:a_count]
    var right_run = b[0:b_count]
    if not host_output_disjoint_from_input(output_run, left_run):
        return TensorStatus.ALIAS_UNKNOWN
    if not host_output_disjoint_from_input(output_run, right_run):
        return TensorStatus.ALIAS_UNKNOWN
    return TensorStatus.OK


@inline(.always)
def cpu_mm_contiguous[width: Int, co: Origin[mut=True], ao: Origin, bo: Origin](
    c: Span[True, Float32, co],
    a: Span[False, Float32, ao],
    b: Span[False, Float32, bo],
    m: Int, k: Int, n: Int,
):
    comptime assert width > 0, "invalid FP32 SIMD width"
    var cp = c.unsafe_ptr()
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var row_block_end = m - m % 4
    var vector_col_end = n - n % width

    # Four output rows share the same B loads in each K step.
    for i in range(0, row_block_end, 4):
        for j in range(0, vector_col_end, width):
            var c0 = SIMD[DType.float32, width](0.0)
            var c1 = SIMD[DType.float32, width](0.0)
            var c2 = SIMD[DType.float32, width](0.0)
            var c3 = SIMD[DType.float32, width](0.0)
            for p in range(k):
                var bvec = bp.unsafe_load[width=width, alignment=4](p * n + j)
                var a0 = ap.unsafe_load(i * k + p)[0]
                var a1 = ap.unsafe_load((i + 1) * k + p)[0]
                var a2 = ap.unsafe_load((i + 2) * k + p)[0]
                var a3 = ap.unsafe_load((i + 3) * k + p)[0]
                c0 += SIMD[DType.float32, width](a0) * bvec
                c1 += SIMD[DType.float32, width](a1) * bvec
                c2 += SIMD[DType.float32, width](a2) * bvec
                c3 += SIMD[DType.float32, width](a3) * bvec
            cp.unsafe_store[alignment=4](i * n + j, c0)
            cp.unsafe_store[alignment=4]((i + 1) * n + j, c1)
            cp.unsafe_store[alignment=4]((i + 2) * n + j, c2)
            cp.unsafe_store[alignment=4]((i + 3) * n + j, c3)

    # Remaining rows retain the SIMD-K accumulation path.
    for i in range(row_block_end, m):
        for j in range(0, vector_col_end, width):
            var acc = SIMD[DType.float32, width](0.0)
            for p in range(k):
                var av = ap.unsafe_load(i * k + p)[0]
                var bv = bp.unsafe_load[width=width, alignment=4](p * n + j)
                acc += SIMD[DType.float32, width](av) * bv
            cp.unsafe_store[alignment=4](i * n + j, acc)

    # Tail columns are scalar to ensure no vector load crosses a physical span.
    for i in range(m):
        for j in range(vector_col_end, n):
            var acc = Float32(0.0)
            for p in range(k):
                acc += ap.unsafe_load(i * k + p)[0] * bp.unsafe_load(p * n + j)[0]
            cp.unsafe_store[alignment=4](i * n + j, acc)


# SIMD reduction for an exact, disjoint contiguous FP32 vector pair. The
# reduction tree permits FASTEST-mode reassociation relative to scalar order.
@inline(.always)
def cpu_dot_contiguous[width: Int, ao: Origin, bo: Origin](
    a: Span[False, Float32, ao], b: Span[False, Float32, bo]
) -> Float32:
    comptime assert width > 0, "SIMD width must be positive"
    var n = len(a)
    var limit = n - n % width
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var partial = SIMD[DType.float32, width](0.0)
    for i in range(0, limit, width):
        partial += ap.unsafe_load[width=width, alignment=4](i) * bp.unsafe_load[width=width, alignment=4](i)
    var result = partial.reduce_add()
    for i in range(limit, n):
        result += a[i] * b[i]
    return result

# Matrix-vector kernel: contiguous A and x, each output row consumes the
# same x without any GEMM packing or heap temporaries.
@inline(.always)
def cpu_gemv_contiguous[width: Int, co: Origin[mut=True], ao: Origin, xo: Origin](
    out: Span[True, Float32, co], a: Span[False, Float32, ao],
    x: Span[False, Float32, xo], m: Int, k: Int,
):
    comptime assert width > 0, "SIMD width must be positive"
    var limit = k - k % width
    var ap = a.unsafe_ptr()
    var xp = x.unsafe_ptr()
    var dst = out.unsafe_ptr()
    for row in range(m):
        var start = row * k
        var partial = SIMD[DType.float32, width](0.0)
        for j in range(0, limit, width):
            partial += ap.unsafe_load[width=width, alignment=4](start + j) * xp.unsafe_load[width=width, alignment=4](j)
        var value = partial.reduce_add()
        for j in range(limit, k):
            value += ap.unsafe_load(start + j)[0] * xp.unsafe_load(j)[0]
        dst.unsafe_store[alignment=4](row, value)


# Contiguous vector @ matrix: traverse B rows with coalesced N-wide SIMD
# loads instead of walking B one strided column at a time.
@inline(.always)
def cpu_vecmat_contiguous[width: Int, co: Origin[mut=True], xo: Origin, bo: Origin](
    out: Span[True, Float32, co], x: Span[False, Float32, xo],
    b: Span[False, Float32, bo], k: Int, n: Int,
):
    comptime assert width > 0, "SIMD width must be positive"
    var dst = out.unsafe_ptr()
    var xp = x.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var bound = n - n % width
    for j in range(0, bound, width):
        var acc = SIMD[DType.float32, width](0.0)
        for p in range(k):
            var left = xp.unsafe_load(p)[0]
            var right = bp.unsafe_load[width=width, alignment=4](p * n + j)
            acc += SIMD[DType.float32, width](left) * right
        dst.unsafe_store[alignment=4](j, acc)
    for j in range(bound, n):
        var acc = Float32(0.0)
        for p in range(k):
            acc += xp.unsafe_load(p)[0] * bp.unsafe_load(p * n + j)[0]
        dst.unsafe_store[alignment=4](j, acc)


# Outer product: each x[i] is reused for a vectorized contiguous y run.
@inline(.always)
def cpu_outer_contiguous[width: Int, co: Origin[mut=True], ao: Origin, bo: Origin](
    out: Span[True, Float32, co], a: Span[False, Float32, ao],
    b: Span[False, Float32, bo], m: Int, n: Int,
):
    comptime assert width > 0, "SIMD width must be positive"
    var dst = out.unsafe_ptr()
    var ap = a.unsafe_ptr()
    var bp = b.unsafe_ptr()
    var bound = n - n % width
    for i in range(m):
        var av = ap.unsafe_load(i)[0]
        var lane = SIMD[DType.float32, width](av)
        for j in range(0, bound, width):
            dst.unsafe_store[alignment=4](i * n + j, lane * bp.unsafe_load[width=width, alignment=4](j))
        for j in range(bound, n):
            dst.unsafe_store[alignment=4](i * n + j, av * bp.unsafe_load(j)[0])


# Tall-and-narrow C-order GEMM uses disjoint four-row MAX shards without
# packing or heap temporaries. This makes multi-core execution worthwhile
# when N is too narrow for packed 64-column macro-panels. All buffers were
# checked by cpu_mm_autoselect_fp32 before entering this leaf.
def cpu_mm_parallel_narrow_unchecked[width: Int, co: Origin[mut=True], ao: Origin, bo: Origin](
    c: Span[True, Float32, co], a: Span[False, Float32, ao],
    b: Span[False, Float32, bo], m: Int, k: Int, n: Int,
) -> TensorStatus:
    comptime assert width > 0, "positive FP32 SIMD width required"
    var groups = m // 4
    var policy = CpuWorkPolicy(16, 8)
    var workers = recommend_worker_count(groups, policy)
    if workers <= 1:
        cpu_mm_contiguous[width](c, a, b, m, k, n)
        return TensorStatus.OK
    var dst = c.unsafe_ptr()
    var lhs = a.unsafe_ptr()
    var rhs = b.unsafe_ptr()
    var lanes_end = n - n % width

    def matrix_narrow_shard(shard: Int) {imm}:
        var q = groups // workers
        var extra = groups % workers
        var first = shard * q + min(shard, extra)
        var last = first + q
        if shard < extra:
            last += 1
        for block in range(first, last):
            var row = block * 4
            for j in range(0, lanes_end, width):
                var acc0 = SIMD[DType.float32, width](0.0)
                var acc1 = SIMD[DType.float32, width](0.0)
                var acc2 = SIMD[DType.float32, width](0.0)
                var acc3 = SIMD[DType.float32, width](0.0)
                for p in range(k):
                    var v = rhs.unsafe_load[width=width, alignment=4](p * n + j)
                    acc0 += SIMD[DType.float32, width](lhs.unsafe_load(row * k + p)[0]) * v
                    acc1 += SIMD[DType.float32, width](lhs.unsafe_load((row + 1) * k + p)[0]) * v
                    acc2 += SIMD[DType.float32, width](lhs.unsafe_load((row + 2) * k + p)[0]) * v
                    acc3 += SIMD[DType.float32, width](lhs.unsafe_load((row + 3) * k + p)[0]) * v
                dst.unsafe_store[alignment=4](row * n + j, acc0)
                dst.unsafe_store[alignment=4]((row + 1) * n + j, acc1)
                dst.unsafe_store[alignment=4]((row + 2) * n + j, acc2)
                dst.unsafe_store[alignment=4]((row + 3) * n + j, acc3)
            for j in range(lanes_end, n):
                for r in range(row, row + 4):
                    var scalar = Float32(0.0)
                    for p in range(k):
                        scalar += lhs.unsafe_load(r * k + p)[0] * rhs.unsafe_load(p * n + j)[0]
                    dst.unsafe_store[alignment=4](r * n + j, scalar)

    var done = execute_parallel_shards(matrix_narrow_shard, groups, policy)
    if done != TensorStatus.OK:
        return done
    # Only the final 1..3 rows are owned by this serial tail.
    for row in range(groups * 4, m):
        for j in range(0, lanes_end, width):
            var accum = SIMD[DType.float32, width](0.0)
            for p in range(k):
                accum += SIMD[DType.float32, width](lhs.unsafe_load(row * k + p)[0]) * rhs.unsafe_load[width=width, alignment=4](p * n + j)
            dst.unsafe_store[alignment=4](row * n + j, accum)
        for j in range(lanes_end, n):
            var scalar = Float32(0.0)
            for p in range(k):
                scalar += lhs.unsafe_load(row * k + p)[0] * rhs.unsafe_load(p * n + j)[0]
            dst.unsafe_store[alignment=4](row * n + j, scalar)
    return TensorStatus.OK


# Packed L1/L2-aware FP32 GEMM candidate. B is copied into a reusable k-by-n
# micro-panel ONCE per (K,N) macro-tile and reused by every M macro-tile.
# This is an AOT tile-geometry candidate, not a claim of universally fastest
# performance. No alloc/free, shape checks, or dispatch in the microkernel.
# Preconditions: C-order contiguous, disjoint initialized A/B, fresh output;
# dimensions and FP32 byte counts proved by the checked owning caller.
# No allocations, geometry checks or profiling in the packed panel leaf.
# Preconditions: checked disjoint spans, workspace >= kc*nc and M,N,K > 0.
def _cpu_mm_packed_with_workspace_unchecked[
    width: Int, kc: Int, nc: Int, mc: Int,
    co: Origin[mut=True], ao: Origin, bo: Origin, so: Origin[mut=True],
](
    c: Span[True, Float32, co],
    a: Span[False, Float32, ao],
    b: Span[False, Float32, bo],
    scratch: Span[True, Float32, so],
    m: Int, k: Int, n: Int,
) -> TensorStatus:
    comptime assert width > 0 and kc > 0 and nc > 0 and mc > 0, "invalid GEMM tile"
    comptime assert nc % width == 0, "panel SIMD alignment"
    var packed = scratch.unsafe_ptr()
    var dest = c.unsafe_ptr()
    var lhs = a.unsafe_ptr()
    var rhs = b.unsafe_ptr()

    for col_begin in range(0, n, nc):
        var width_n = min(nc, n - col_begin)
        var aligned_n = width_n - width_n % width
        for depth_begin in range(0, k, kc):
            var width_k = min(kc, k - depth_begin)
            # Pack right-hand matrix into a compact row-major panel; the panel
            # pitch is nc to give every row an aligned SIMD-wide start.
            for p in range(width_k):
                var source_row = (depth_begin + p) * n + col_begin
                var target_row = p * nc
                # The macro-panel has SIMD-aligned pitch `nc`. Packing
                # contiguous FP32 lanes in one vector transaction avoids a
                # scalar load/store pair per B element. Both addresses are
                # already proved in bounds by the checked GEMM boundary.
                # Never load past width_n: a partial rightmost panel uses
                # exactly the same scalar tail as before this optimization.
                for j in range(0, aligned_n, width):
                    var lanes = rhs.unsafe_load[width=width, alignment=4](source_row + j)
                    packed.unsafe_store[alignment=4](target_row + j, lanes)
                for j in range(aligned_n, width_n):
                    packed.unsafe_store[alignment=4](target_row + j, rhs.unsafe_load(source_row + j)[0])

            # Native MAX workers are entered only for large, proven-disjoint
            # M row groups. B is packed ONCE per K/N panel and shared read-only
            # by all workers. No per-worker allocation or repeated validation.
            # Each work item owns exactly four C rows; the residual rows are
            # handled synchronously after the worker join.
            var full_groups = m // 4
            var work_policy = CpuWorkPolicy(max(1, mc // 4), 8)
            var workers = 1
            if m >= 384 and n >= 256 and k >= 256:
                workers = recommend_worker_count(full_groups, work_policy)

            def compute_packed_groups(shard: Int) {imm}:
                var q = full_groups // workers
                var extra = full_groups % workers
                var first_group = shard * q + min(shard, extra)
                var last_group = first_group + q
                if shard < extra:
                    last_group += 1
                for group in range(first_group, last_group):
                    var row = group * 4
                    for j in range(0, aligned_n, width):
                        var c0 = SIMD[DType.float32, width](0.0)
                        var c1 = SIMD[DType.float32, width](0.0)
                        var c2 = SIMD[DType.float32, width](0.0)
                        var c3 = SIMD[DType.float32, width](0.0)
                        var dst0 = row * n + col_begin + j
                        var dst1 = (row + 1) * n + col_begin + j
                        var dst2 = (row + 2) * n + col_begin + j
                        var dst3 = (row + 3) * n + col_begin + j
                        if depth_begin != 0:
                            c0 = dest.unsafe_load[width=width, alignment=4](dst0)
                            c1 = dest.unsafe_load[width=width, alignment=4](dst1)
                            c2 = dest.unsafe_load[width=width, alignment=4](dst2)
                            c3 = dest.unsafe_load[width=width, alignment=4](dst3)
                        for p in range(width_k):
                            var bv = packed.unsafe_load[width=width, alignment=4](p * nc + j)
                            c0 += SIMD[DType.float32, width](lhs.unsafe_load(row * k + depth_begin + p)[0]) * bv
                            c1 += SIMD[DType.float32, width](lhs.unsafe_load((row + 1) * k + depth_begin + p)[0]) * bv
                            c2 += SIMD[DType.float32, width](lhs.unsafe_load((row + 2) * k + depth_begin + p)[0]) * bv
                            c3 += SIMD[DType.float32, width](lhs.unsafe_load((row + 3) * k + depth_begin + p)[0]) * bv
                        dest.unsafe_store[alignment=4](dst0, c0)
                        dest.unsafe_store[alignment=4](dst1, c1)
                        dest.unsafe_store[alignment=4](dst2, c2)
                        dest.unsafe_store[alignment=4](dst3, c3)
                    for j in range(aligned_n, width_n):
                        for r in range(row, row + 4):
                            var offset = r * n + col_begin + j
                            var scalar = Float32(0.0)
                            if depth_begin != 0:
                                scalar = dest.unsafe_load(offset)[0]
                            for p in range(width_k):
                                scalar += lhs.unsafe_load(r * k + depth_begin + p)[0] * packed.unsafe_load(p * nc + j)[0]
                            dest.unsafe_store[alignment=4](offset, scalar)

            if workers > 1:
                var done = execute_parallel_shards(
                    compute_packed_groups, full_groups, work_policy,
                )
                if done != TensorStatus.OK:
                    # A partially written owner is NOT published on failure.
                    return done
            elif full_groups > 0:
                compute_packed_groups(0)

            # Independent tail rows cannot overlap a four-row worker group.
            for row in range(full_groups * 4, m):
                for j in range(0, aligned_n, width):
                    var offset = row * n + col_begin + j
                    var accum = SIMD[DType.float32, width](0.0)
                    if depth_begin != 0:
                        accum = dest.unsafe_load[width=width, alignment=4](offset)
                    for p in range(width_k):
                        var av = lhs.unsafe_load(row * k + depth_begin + p)[0]
                        accum += SIMD[DType.float32, width](av) * packed.unsafe_load[width=width, alignment=4](p * nc + j)
                    dest.unsafe_store[alignment=4](offset, accum)
                for j in range(aligned_n, width_n):
                    var offset = row * n + col_begin + j
                    var accum_scalar = Float32(0.0)
                    if depth_begin != 0:
                        accum_scalar = dest.unsafe_load(offset)[0]
                    for p in range(width_k):
                        accum_scalar += lhs.unsafe_load(row * k + depth_begin + p)[0] * packed.unsafe_load(p * nc + j)[0]
                    dest.unsafe_store[alignment=4](offset, accum_scalar)
    return TensorStatus.OK


# The public auto-selector and public packed entry validate disjoint memory
# BEFORE calling this trusted internal leaf. A second validation at this
# boundary would add redundant metadata and alias checks to every GEMM.
# Preconditions: m>0, n>0, k>0; all three logical regions validated.
def _cpu_mm_packed_alloc_after_preflight_unchecked[
    width: Int, kc: Int, nc: Int, mc: Int,
    co: Origin[mut=True], ao: Origin, bo: Origin,
](
    c: Span[True, Float32, co],
    a: Span[False, Float32, ao],
    b: Span[False, Float32, bo],
    m: Int, k: Int, n: Int,
) -> TensorStatus:
    var allocation = HostStorage._allocate_for_complete_overwrite(kc * nc)
    if not allocation.is_ok():
        return allocation.status
    var owner = allocation.payload.take()
    var panel = owner.borrow_mut_span()
    return _cpu_mm_packed_with_workspace_unchecked[width, kc, nc, mc](
        c, a, b, panel, m, k, n,
    )


# Public packed path. Allocates exactly ONE scratch panel per GEMM call.
# Repeated-call users can pass their own fixed buffer to the workspace API.
def cpu_mm_packed_fp32[
    width: Int, kc: Int, nc: Int, mc: Int,
    co: Origin[mut=True], ao: Origin, bo: Origin,
](
    c: Span[True, Float32, co],
    a: Span[False, Float32, ao],
    b: Span[False, Float32, bo],
    m: Int, k: Int, n: Int,
) -> TensorStatus:
    comptime assert width > 0 and kc > 0 and nc > 0 and mc > 0, "invalid GEMM tile"
    comptime assert nc % width == 0, "N panel must be SIMD lane-aligned"
    comptime assert kc <= (Int.MAX // 4) // nc, "oversized packed panel"
    var boundary = validate_fp32_gemm_buffers(c, a, b, m, k, n)
    if boundary != TensorStatus.OK:
        return boundary
    if m == 0 or n == 0:
        return TensorStatus.OK
    if k == 0:
        var dst = c.unsafe_ptr()
        for i in range(m * n):
            dst.unsafe_store[alignment=4](i, Float32(0.0))
        return TensorStatus.OK
    return _cpu_mm_packed_alloc_after_preflight_unchecked[width, kc, nc, mc](
        c, a, b, m, k, n,
    )


# Allocation-free repeated-call GEMM entry for a caller-owned workspace.
# The scratch region is never published as a Tensor: stale bytes are
# overwritten by packing before use. Caller retains exclusive ownership
# through the synchronous computation and may safely reuse it next time.
def cpu_mm_packed_workspace_fp32[
    width: Int, kc: Int, nc: Int, mc: Int,
    co: Origin[mut=True], ao: Origin, bo: Origin, so: Origin[mut=True],
](
    c: Span[True, Float32, co],
    a: Span[False, Float32, ao],
    b: Span[False, Float32, bo],
    scratch: Span[True, Float32, so],
    m: Int, k: Int, n: Int,
) -> TensorStatus:
    comptime assert width > 0 and kc > 0 and nc > 0 and mc > 0, "invalid GEMM tile"
    comptime assert nc % width == 0, "N panel must be SIMD lane-aligned"
    comptime assert kc <= (Int.MAX // 4) // nc, "oversized packed panel"
    var boundary = validate_fp32_gemm_buffers(c, a, b, m, k, n)
    if boundary != TensorStatus.OK:
        return boundary
    if m == 0 or n == 0:
        return TensorStatus.OK
    if k == 0:
        # The scratch region is intentionally irrelevant for an empty K axis.
        for i in range(m * n):
            c[i] = Float32(0.0)
        return TensorStatus.OK
    var required = kc * nc
    if len(scratch) < required:
        return TensorStatus.SIZE_MISMATCH
    # Three proven address-range checks prevent scratch from corrupting
    # either read input or a partly written output. No global workspace.
    var panel = scratch[0:required]
    if not host_output_disjoint_from_input(panel, a[0:m * k]):
        return TensorStatus.ALIAS_UNKNOWN
    if not host_output_disjoint_from_input(panel, b[0:k * n]):
        return TensorStatus.ALIAS_UNKNOWN
    if not host_mutable_fp32_spans_disjoint(panel, c[0:m * n]):
        return TensorStatus.ALIAS_UNKNOWN
    return _cpu_mm_packed_with_workspace_unchecked[width, kc, nc, mc](
        c, a, b, panel, m, k, n,
    )


# With K=1 a GEMM reduces exactly ONE product starting from +0.0.
# Calling a plain outer-product kernel loses that initial addition on -0.0,
# potentially changing the strict signed-zero contract. Retain a single
# contiguous SIMD pass, but do exactly (0.0 + (a * b)) for each lane.
@inline(.always)
def cpu_mm_k1_strict_contiguous[
    width: Int, co: Origin[mut=True], ao: Origin, bo: Origin,
](
    c: Span[True, Float32, co], a: Span[False, Float32, ao],
    b: Span[False, Float32, bo], m: Int, n: Int,
):
    comptime assert width > 0, "SIMD width must be positive"
    var dst = c.unsafe_ptr()
    var lhs = a.unsafe_ptr()
    var rhs = b.unsafe_ptr()
    var bound = n - n % width
    var zeros = SIMD[DType.float32, width](0.0)
    for row in range(m):
        var av = SIMD[DType.float32, width](lhs.unsafe_load(row)[0])
        for col in range(0, bound, width):
            var products = av * rhs.unsafe_load[width=width, alignment=4](col)
            dst.unsafe_store[alignment=4](row * n + col, zeros + products)
        for col in range(bound, n):
            var product = lhs.unsafe_load(row)[0] * rhs.unsafe_load(col)[0]
            dst.unsafe_store[alignment=4](row * n + col, Float32(0.0) + product)


# The sole C-order FP32 GEMM strategy entry for OWNED CPU tensors. The
# geometry cutoff selects between two precompiled kernels; it is not JIT,
# allocation-based autotuning or a per-launch performance measurement. This
# decision belongs outside both hot loops and uses the same threshold for
# rank-2, batch and broadcasted rank-2 matrix slices.
@inline(.always)
def cpu_mm_autoselect_fp32[
    width: Int, co: Origin[mut=True], ao: Origin, bo: Origin,
](
    c: Span[True, Float32, co],
    a: Span[False, Float32, ao],
    b: Span[False, Float32, bo],
    m: Int, k: Int, n: Int,
    math_mode: Fp32MathMode = Fp32MathMode.STRICT_FP32,
) -> TensorStatus:
    # PRECONDITION: these are checked, independent C-order physical slices.
    # This path never transfers to GPU and never guesses a different dtype.
    var boundary = validate_fp32_gemm_buffers(c, a, b, m, k, n)
    if boundary != TensorStatus.OK:
        return boundary
    if math_mode.code < 0 or math_mode.code > 3:
        return TensorStatus.MATH_MODE_UNSUPPORTED
    if m == 0 or n == 0:
        return TensorStatus.OK
    if k == 0:
        # The empty reduction axis produces an initialized zero matrix.
        # Never obtain a raw pointer from either empty operand in the SIMT
        # microkernel when no operand elements exist.
        for i in range(m * n):
            c[i] = Float32(0.0)
        return TensorStatus.OK
    # Stable-shape GEMM degeneracies need neither B packing nor a general
    # matrix tile. These exact-shape paths keep Float32 and the same geometry.
    # K=1 MUST still retain GEMM's initial 0.0 addition for signed-zero.
    if k == 1:
        cpu_mm_k1_strict_contiguous[width](c, a, b, m, n)
        return TensorStatus.OK
    if m == 1:
        # One row is a vector-by-matrix multiplication; SIMD spans columns
        # while each column retains the same sequential K association.
        cpu_vecmat_contiguous[width](c, a, b, k, n)
        return TensorStatus.OK
    if n == 1 and math_mode == Fp32MathMode.FASTEST:
        # SIMD GEMV horizontally reduces K lanes in a different order.
        # NEVER select this under default strict arithmetic semantics.
        cpu_gemv_contiguous[width](c, a, b, m, k)
        return TensorStatus.OK
    if m >= 64 and n >= 64 and k >= 128:
        # Packed B is shared read-only by all disjoint M-row worker shards.
        # No benchmark, JIT, per-call hardware probe or dynamic tuning.
        return _cpu_mm_packed_alloc_after_preflight_unchecked[width, 128, 64, 64](
            c, a, b, m, k, n,
        )
    if m >= 256 and n >= 8 and k >= 64:
        return cpu_mm_parallel_narrow_unchecked[width](c, a, b, m, k, n)
    cpu_mm_contiguous[width](c, a, b, m, k, n)
    return TensorStatus.OK


# Fused FP32 rank-2 GEMM+bias+ReLU expert: write each output element only
# ONCE and never materialize an intermediate GEMM tensor. K is reduced in
# forward order (no implied TF32 or reassociation). Float32 input/output and
# shape [M,N] are identical to GEMM followed by row bias and IEEE max(+0).
# The bias is READ for every output row; it must never alias the destination.
# For nontrivial GEMM, prefer a measured packed epilogue for large workloads;
# this correctness-preserving SIMD-N candidate is not assumed fastest.
def cpu_mm_bias_relu_into_fp32[
    width: Int, co: Origin[mut=True], ao: Origin, bo: Origin, bias_origin: Origin,
](
    c: Span[True, Float32, co], a: Span[False, Float32, ao],
    b: Span[False, Float32, bo], bias: Span[False, Float32, bias_origin],
    m: Int, k: Int, n: Int,
) -> TensorStatus:
    comptime assert width > 0, "positive Float32 lane width required"
    var valid = validate_fp32_gemm_buffers(c, a, b, m, k, n)
    if valid != TensorStatus.OK:
        return valid
    if len(bias) != n:
        return TensorStatus.SIZE_MISMATCH
    if m == 0 or n == 0:
        return TensorStatus.OK
    if not host_output_disjoint_from_input(c[0:m * n], bias):
        return TensorStatus.ALIAS_UNKNOWN
    var dst = c.unsafe_ptr()
    var bvalues = bias.unsafe_ptr()
    var bound = n - n % width
    if k == 0:
        # Empty K: the correct GEMM identity is zero, then bias and ReLU.
        # Do NOT form a pointer to either empty input operand.
        for row in range(m):
            for column in range(n):
                dst.unsafe_store[alignment=4](
                    row * n + column,
                    fp32_maximum(bvalues.unsafe_load(column)[0], Float32(0.0)),
                )
        return TensorStatus.OK
    var lhs = a.unsafe_ptr()
    var rhs = b.unsafe_ptr()
    for row in range(m):
        for col in range(0, bound, width):
            var acc = SIMD[DType.float32, width](0.0)
            for depth in range(k):
                var av = lhs.unsafe_load(row * k + depth)[0]
                var bv = rhs.unsafe_load[width=width, alignment=4](depth * n + col)
                acc += SIMD[DType.float32, width](av) * bv
            var biased = acc + bvalues.unsafe_load[width=width, alignment=4](col)
            for lane in range(width):
                dst.unsafe_store[alignment=4](
                    row * n + col + lane,
                    fp32_maximum(biased[lane], Float32(0.0)),
                )
        for col in range(bound, n):
            var acc = Float32(0.0)
            for depth in range(k):
                acc += lhs.unsafe_load(row * k + depth)[0] * rhs.unsafe_load(depth * n + col)[0]
            var biased = acc + bvalues.unsafe_load(col)[0]
            dst.unsafe_store[alignment=4](
                row * n + col, fp32_maximum(biased, Float32(0.0)),
            )
    return TensorStatus.OK


# Public checked FP32 matrix/vector entry points. They validate geometry and
# ownership BEFORE invoking the allocation-free, SIMD-specialized kernels.
# The preflight is O(1) and never runs in a reduction or GEMM inner loop.
# Raw cpu_*_contiguous microkernels above retain their internal unchecked ABI.
def cpu_dot_checked_fp32[width: Int, ao: Origin, bo: Origin](
    a: Span[False, Float32, ao], b: Span[False, Float32, bo],
) -> TensorResult[Float32]:
    var n = len(a)
    if n < 0 or n != len(b):
        return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
    if n > Int.MAX // 4:
        return TensorResult[Float32](TensorStatus.OVERFLOW)
    if n == 0:
        return TensorResult[Float32](Float32(0.0))
    # Read-only operands may overlap; no output is written until completion.
    return TensorResult[Float32](cpu_dot_contiguous[width](a, b))


def cpu_gemv_checked_fp32[width: Int, co: Origin[mut=True], ao: Origin, xo: Origin](
    out: Span[True, Float32, co], a: Span[False, Float32, ao],
    x: Span[False, Float32, xo], m: Int, k: Int,
) -> TensorStatus:
    if m < 0 or k < 0:
        return TensorStatus.INVALID_SHAPE
    if m > Int.MAX // 4 or k > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if k > 0 and m > (Int.MAX // 4) // k:
        return TensorStatus.OVERFLOW
    var a_count = m * k
    if m > len(out) or a_count > len(a) or k > len(x):
        return TensorStatus.SIZE_MISMATCH
    if m == 0:
        return TensorStatus.OK
    if k == 0:
        # Empty inner dimension is a well-defined zero vector. Never form a
        # pointer into an empty input just to produce its zero-valued result.
        for row in range(m):
            out[row] = Float32(0.0)
        return TensorStatus.OK
    var target = out[0:m]
    if not host_output_disjoint_from_input(target, a[0:a_count]):
        return TensorStatus.ALIAS_UNKNOWN
    if not host_output_disjoint_from_input(target, x[0:k]):
        return TensorStatus.ALIAS_UNKNOWN
    # Each output row has exactly one writer. Input matrices and vectors are
    # immutable throughout this call; physical output/input ranges have just
    # been proven disjoint, so synchronous CPU workers can share the owners.
    # Tune the grain outside the library for each target before release.
    var grain_rows = max(1, 262144 // k)
    var policy = CpuWorkPolicy(grain_rows, 8)
    var workers = recommend_worker_count(m, policy)
    if workers > 1:
        var a_ptr = a.unsafe_ptr()
        var x_ptr = x.unsafe_ptr()
        var dest_ptr = out.unsafe_ptr()
        var vector_end = k - k % width

        def gemv_row_shard(shard: Int) {imm}:
            var base_rows = m // workers
            var extra_rows = m % workers
            var first = shard * base_rows + min(shard, extra_rows)
            var last = first + base_rows
            if shard < extra_rows:
                last += 1
            for row in range(first, last):
                var accumulator = SIMD[DType.float32, width](0.0)
                var base = row * k
                for j in range(0, vector_end, width):
                    accumulator += a_ptr.unsafe_load[width=width, alignment=4](base + j) * x_ptr.unsafe_load[width=width, alignment=4](j)
                var total = accumulator.reduce_add()
                for j in range(vector_end, k):
                    total += a_ptr.unsafe_load[alignment=4](base + j) * x_ptr.unsafe_load[alignment=4](j)
                dest_ptr.unsafe_store[alignment=4](row, total)

        return execute_parallel_shards(gemv_row_shard, m, policy)
    cpu_gemv_contiguous[width](out, a, x, m, k)
    return TensorStatus.OK


def cpu_vecmat_checked_fp32[width: Int, co: Origin[mut=True], xo: Origin, bo: Origin](
    out: Span[True, Float32, co], x: Span[False, Float32, xo],
    b: Span[False, Float32, bo], k: Int, n: Int,
) -> TensorStatus:
    if k < 0 or n < 0:
        return TensorStatus.INVALID_SHAPE
    if k > Int.MAX // 4 or n > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if n > 0 and k > (Int.MAX // 4) // n:
        return TensorStatus.OVERFLOW
    var b_count = k * n
    if n > len(out) or k > len(x) or b_count > len(b):
        return TensorStatus.SIZE_MISMATCH
    if n == 0:
        return TensorStatus.OK
    if k == 0:
        for col in range(n):
            out[col] = Float32(0.0)
        return TensorStatus.OK
    var target = out[0:n]
    if not host_output_disjoint_from_input(target, x[0:k]):
        return TensorStatus.ALIAS_UNKNOWN
    if not host_output_disjoint_from_input(target, b[0:b_count]):
        return TensorStatus.ALIAS_UNKNOWN
    # A vector @ wide matrix writes independent output columns. Divide
    # whole SIMD vectors between MAX workers so no lane is written twice.
    # Fixed size threshold avoids excessive worker overhead on small GEMV.
    if n >= 1024 and k >= 256:
        var vector_groups = n // width
        if n % width != 0:
            vector_groups += 1
        var policy = CpuWorkPolicy(32, 8)
        var workers = recommend_worker_count(vector_groups, policy)
        if workers > 1:
            var dst_ptr = out.unsafe_ptr()
            var vec_ptr = x.unsafe_ptr()
            var mat_ptr = b.unsafe_ptr()

            def vecmat_column_shard(shard: Int) {imm}:
                var q = vector_groups // workers
                var rem = vector_groups % workers
                var first = shard * q + min(shard, rem)
                var last = first + q
                if shard < rem:
                    last += 1
                for group in range(first, last):
                    var col = group * width
                    if col + width <= n:
                        var acc = SIMD[DType.float32, width](0.0)
                        for p in range(k):
                            var coeff = vec_ptr.unsafe_load(p)[0]
                            var values = mat_ptr.unsafe_load[width=width, alignment=4](p * n + col)
                            acc += SIMD[DType.float32, width](coeff) * values
                        dst_ptr.unsafe_store[alignment=4](col, acc)
                    else:
                        # Only the rightmost incomplete SIMD group is scalar.
                        for tail_col in range(col, n):
                            var scalar = Float32(0.0)
                            for p in range(k):
                                scalar += vec_ptr.unsafe_load(p)[0] * mat_ptr.unsafe_load(p * n + tail_col)[0]
                            dst_ptr.unsafe_store[alignment=4](tail_col, scalar)

            return execute_parallel_shards(vecmat_column_shard, vector_groups, policy)
    cpu_vecmat_contiguous[width](out, x, b, k, n)
    return TensorStatus.OK


def cpu_outer_checked_fp32[width: Int, co: Origin[mut=True], ao: Origin, bo: Origin](
    out: Span[True, Float32, co], a: Span[False, Float32, ao],
    b: Span[False, Float32, bo], m: Int, n: Int,
) -> TensorStatus:
    if m < 0 or n < 0:
        return TensorStatus.INVALID_SHAPE
    if m > Int.MAX // 4 or n > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if n > 0 and m > (Int.MAX // 4) // n:
        return TensorStatus.OVERFLOW
    var total = m * n
    if total > len(out) or m > len(a) or n > len(b):
        return TensorStatus.SIZE_MISMATCH
    if total == 0:
        return TensorStatus.OK
    var target = out[0:total]
    if not host_output_disjoint_from_input(target, a[0:m]):
        return TensorStatus.ALIAS_UNKNOWN
    if not host_output_disjoint_from_input(target, b[0:n]):
        return TensorStatus.ALIAS_UNKNOWN
    var grain_rows = max(1, 262144 // n)
    var policy = CpuWorkPolicy(grain_rows, 8)
    var workers = recommend_worker_count(m, policy)
    if workers > 1:
        var a_ptr = a.unsafe_ptr()
        var b_ptr = b.unsafe_ptr()
        var dst_ptr = out.unsafe_ptr()
        var vector_end = n - n % width

        def outer_row_shard(shard: Int) {imm}:
            var base_rows = m // workers
            var extra_rows = m % workers
            var first = shard * base_rows + min(shard, extra_rows)
            var last = first + base_rows
            if shard < extra_rows:
                last += 1
            for row in range(first, last):
                var scalar = a_ptr.unsafe_load[alignment=4](row)
                var lane = SIMD[DType.float32, width](scalar)
                for col in range(0, vector_end, width):
                    dst_ptr.unsafe_store[alignment=4](row * n + col,
                        lane * b_ptr.unsafe_load[width=width, alignment=4](col))
                for col in range(vector_end, n):
                    dst_ptr.unsafe_store[alignment=4](row * n + col,
                        scalar * b_ptr.unsafe_load[alignment=4](col))

        return execute_parallel_shards(outer_row_shard, m, policy)
    cpu_outer_contiguous[width](out, a, b, m, n)
    return TensorStatus.OK
