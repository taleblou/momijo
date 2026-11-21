# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | CPU KERNEL / COPYING
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / CPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.cpu.copy
# Source:        src/momijo/tensor_light/kernels/cpu/copy.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor copying for the CPU tensor kernel layer.
#
#   Public declarations include `cpu_copy_noalias`, `cpu_copy_contiguous`,
#   `cpu_copy_strided`, `cpu_fill`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `cpu_copy_noalias`,
#     `cpu_copy_contiguous`, `cpu_copy_strided`, `cpu_fill`,
#     `cpu_fill_strided` and related symbols.
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

# Verified FP32 CPU copy/fill boundaries. The contiguous fast path uses one
# owner-independent pointer range proof; arbitrary strided overlap snapshots
# the full logical source BEFORE the first destination write.
from std.collections import Span
from std.memory import unsafe_memcpy, unsafe_memmove
from std.sys.info import simd_width_of
from std.sys import prefetch
from std.sys.intrinsics import PrefetchOptions
from ...internal.status import TensorStatus
from ...alias.overlap import host_output_disjoint_from_input
from ...view.view import TensorView, TensorMutView
from ...view.adapters import as_c_span, as_f_span, mut_as_c_span, mut_as_f_span
from ...iterator.nd_iterator import NDIndexIterator
from ...layout.layout_analysis import is_reverse_c_contiguous, is_reverse_f_contiguous
from ...storage.host_storage import HostStorage, _fill_fp32_contiguous
from ...parallel.parallel_for import CpuWorkPolicy, recommend_worker_count, execute_parallel_shards


def _cpu_copy_noalias_unchecked[o: Origin[mut=True], s: Origin[mut=False]](
    dest: Span[mut=True, Float32, o], src: Span[mut=False, Float32, s],
):
    # INTERNAL: caller has already proved equal length, representable byte
    # count and disjoint, live host regions. One native bulk copy only.
    if len(dest) > 0:
        unsafe_memcpy(dest=dest.unsafe_ptr(), src=src.unsafe_ptr(), count=len(dest))


def _cpu_copy_noalias_selected_after_proof[
    o: Origin[mut=True], s: Origin[mut=False],
](dest: Span[mut=True, Float32, o], src: Span[mut=False, Float32, s]) -> TensorStatus:
    # The checked caller establishes valid equal-size, live, DISJOINT FP32
    # ranges before entering this function. A single tuned native memcpy
    # remains the default; only a very large AOT candidate uses CPU shards.
    # Shards cannot reproduce memmove semantics and must NEVER run for alias.
    var count = len(dest)
    var policy = CpuWorkPolicy(4194304, 8)
    var workers = recommend_worker_count(count, policy)
    if workers == 1:
        _cpu_copy_noalias_unchecked(dest, src)
        return TensorStatus.OK
    var dst_ptr = dest.unsafe_ptr()
    var src_ptr = src.unsafe_ptr()

    def copy_disjoint_shard(shard: Int) {imm}:
        var q = count // workers
        var rem = count % workers
        var first = shard * q + min(shard, rem)
        var length = q
        if shard < rem:
            length += 1
        if length > 0:
            unsafe_memcpy(
                dest=dst_ptr.unsafe_offset(first),
                src=src_ptr.unsafe_offset(first),
                count=length,
            )

    return execute_parallel_shards(copy_disjoint_shard, count, policy)


def cpu_copy_noalias[o: Origin[mut=True], s: Origin[mut=False]](
    dest: Span[mut=True, Float32, o], src: Span[mut=False, Float32, s],
) -> TensorStatus:
    # A public name containing `noalias` is not an aliasing proof. Reject a
    # mismatched length, an unrepresentable byte range, or a shifted/exact
    # alias BEFORE calling the unchecked memcpy kernel. Users that want
    # overlapping copy semantics should call cpu_copy_contiguous (memmove).
    var count = len(dest)
    if count < 0 or len(src) < 0:
        return TensorStatus.INVALID_LAYOUT
    if count != len(src):
        return TensorStatus.SIZE_MISMATCH
    if count > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if count == 0:
        return TensorStatus.OK
    if not host_output_disjoint_from_input(dest, src):
        return TensorStatus.ALIAS_UNKNOWN
    return _cpu_copy_noalias_selected_after_proof(dest, src)


def cpu_copy_contiguous[o: Origin[mut=True], s: Origin[mut=False]](
    mut dest: Span[mut=True, Float32, o], src: Span[mut=False, Float32, s]
) -> TensorStatus:
    # A memcpy-like primitive must never corrupt shifted aliases. A real
    # address disjointness proof selects memcpy, otherwise memmove preserves
    # the original source values. Both paths use one bulk native operation.
    var count = len(dest)
    if count < 0 or len(src) < 0:
        return TensorStatus.INVALID_LAYOUT
    if count != len(src):
        return TensorStatus.SIZE_MISMATCH
    if count > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if count == 0:
        return TensorStatus.OK
    var dst_address = Int(dest.unsafe_ptr())
    var src_address = Int(src.unsafe_ptr())
    var bytes = count * 4
    if dst_address <= 0 or src_address <= 0:
        return TensorStatus.INVALID_LAYOUT
    if dst_address > Int.MAX - bytes or src_address > Int.MAX - bytes:
        return TensorStatus.OVERFLOW
    if host_output_disjoint_from_input(dest, src):
        return _cpu_copy_noalias_selected_after_proof(dest, src)
    # A shifted or identical alias uses ONE overlap-safe native memmove;
    # parallel noalias shards are forbidden here regardless of length.
    unsafe_memmove(dest=dest.unsafe_ptr(), src=src.unsafe_ptr(), count=count)
    return TensorStatus.OK


def cpu_copy_strided[
    rank: Int, s: Origin, o: Origin[mut=True],
](
    src: TensorView[rank, s], mut dst: TensorMutView[rank, o]
) -> TensorStatus:
    comptime assert rank >= 0 and rank <= 8, "FP32 tensor rank must be 0..8"
    # Unlike a flat memcpy, this respects the logical C-order of arbitrary
    # positive, negative or zero source strides. A writeable destination must
    # have a unique physical mapping; overlapping destination strides cannot
    # implement one-write-per-element semantics.
    var src_proof = src._checked_current_layout()
    if not src_proof.is_ok():
        return src_proof.status
    var dst_proof = dst._checked_current_layout()
    if not dst_proof.is_ok():
        return dst_proof.status
    var input_layout = src_proof.payload.value()
    var output_layout = dst_proof.payload.value()
    if not input_layout.shape.same_extents(output_layout.shape):
        return TensorStatus.SHAPE_MISMATCH
    if not output_layout.is_proven_unique():
        return TensorStatus.OVERLAPPING
    var count = input_layout.shape.numel()
    if count == 0:
        return TensorStatus.OK
    # Identical physical traversal orders permit one bulk operation, even
    # for exact or shifted host aliases. C/F disagreement requires iteration.
    if input_layout.is_c_contiguous() and output_layout.is_c_contiguous():
        var input_run = as_c_span(src)
        if not input_run.is_ok():
            return input_run.status
        var output_run = mut_as_c_span(dst)
        if not output_run.is_ok():
            return output_run.status
        var output_span = output_run.payload.value()
        var input_span = input_run.payload.value()
        return cpu_copy_contiguous(output_span, input_span)
    if input_layout.is_f_contiguous() and output_layout.is_f_contiguous():
        var input_run = as_f_span(src)
        if not input_run.is_ok():
            return input_run.status
        var output_run = mut_as_f_span(dst)
        if not output_run.is_ok():
            return output_run.status
        var output_span = output_run.payload.value()
        var input_span = input_run.payload.value()
        return cpu_copy_contiguous(output_span, input_span)
    if input_layout.is_exhaustive() and output_layout.is_exhaustive():
        # Even when neither tensor is forward C/F-contiguous, identical
        # per-axis address steps mean every logical index has the SAME
        # offset relative to the two physical span bases. When both
        # mappings are proven unique and exhaustive, the entire operation
        # is a single physical memmove/memcpy. This includes identically
        # transposed, flipped, and mixed-sign dense views, and guarantees
        # correct shifted-overlap behavior without a full staging buffer.
        var equal_steps = True
        comptime for axis in range(rank):
            if (input_layout.shape.dims.get(axis) > 1
                and input_layout.strides.get(axis) != output_layout.strides.get(axis)):
                equal_steps = False
        if equal_steps:
            var source_base = input_layout.span_lo
            var target_base = output_layout.span_lo
            var source_run = src.data[source_base:source_base + count]
            var target_run = dst.data[target_base:target_base + count]
            return cpu_copy_contiguous(target_run, source_run)
        var both_reverse_c = (
            is_reverse_c_contiguous(input_layout)
            and is_reverse_c_contiguous(output_layout)
        )
        var both_reverse_f = (
            is_reverse_f_contiguous(input_layout)
            and is_reverse_f_contiguous(output_layout)
        )
        if both_reverse_c or both_reverse_f:
            # Both logical traversals enumerate exactly the same REVERSED
            # physical order. A physical memmove preserves the logical copy,
            # including shifted aliases, without an intermediate allocation.
            var src_begin = input_layout.span_lo
            var dst_begin = output_layout.span_lo
            var src_run = src.data[src_begin:src_begin + count]
            var dst_run = dst.data[dst_begin:dst_begin + count]
            return cpu_copy_contiguous(dst_run, src_run)
    # The envelope is a conservative superset of physically selected cells;
    # false positives pay for a snapshot but cannot authorize an unsafe copy.
    var src_envelope = src.data[input_layout.span_lo:input_layout.span_hi + 1]
    var dst_envelope = dst.data[output_layout.span_lo:output_layout.span_hi + 1]
    var disjoint_envelopes = host_output_disjoint_from_input(dst_envelope, src_envelope)
    if input_layout.is_exhaustive() and output_layout.is_exhaustive() and disjoint_envelopes:
        var opposite_c = (
            (input_layout.is_c_contiguous() and is_reverse_c_contiguous(output_layout))
            or (is_reverse_c_contiguous(input_layout) and output_layout.is_c_contiguous())
        )
        var opposite_f = (
            (input_layout.is_f_contiguous() and is_reverse_f_contiguous(output_layout))
            or (is_reverse_f_contiguous(input_layout) and output_layout.is_f_contiguous())
        )
        if opposite_c or opposite_f:
            # Either mapping traverses the same full physical run in reverse
            # relative to the other. Distinct address intervals guarantee
            # writes cannot corrupt unread input, so no snapshot is needed.
            # Rank-zero and single-element views also satisfy this reversal.
            var source_pointer = src.data.unsafe_ptr()
            var output_pointer = dst.data.unsafe_ptr()
            var source_last = input_layout.span_hi
            var target_first = output_layout.span_lo
            for physical in range(count):
                output_pointer.unsafe_store(
                    target_first + physical,
                    source_pointer.unsafe_load(source_last - physical)[0],
                )
            return TensorStatus.OK
    if not disjoint_envelopes:
        var pending = HostStorage._allocate_for_complete_overwrite(count)
        if not pending.is_ok():
            return pending.status
        var scratch_owner = pending.payload.take()
        var scratch = scratch_owner.borrow_mut_span()
        var source_cursor = NDIndexIterator[rank](input_layout.shape)
        for i in range(count):
            scratch[i] = src.data[input_layout.physical_offset_unchecked(source_cursor.coordinates)]
            source_cursor.advance()
        var destination_cursor = NDIndexIterator[rank](output_layout.shape)
        for i in range(count):
            dst.data[output_layout.physical_offset_unchecked(destination_cursor.coordinates)] = scratch[i]
            destination_cursor.advance()
        return TensorStatus.OK
    var source_cursor = NDIndexIterator[rank](input_layout.shape)
    var destination_cursor = NDIndexIterator[rank](output_layout.shape)
    for i in range(count):
        var source_pos = input_layout.physical_offset_unchecked(source_cursor.coordinates)
        var output_pos = output_layout.physical_offset_unchecked(destination_cursor.coordinates)
        dst.data[output_pos] = src.data[source_pos]
        source_cursor.advance()
        destination_cursor.advance()
    return TensorStatus.OK


def cpu_fill[o: Origin[mut=True]](
    mut dst: Span[mut=True, Float32, o], value: Float32,
) -> TensorStatus:
    if len(dst) < 0:
        return TensorStatus.INVALID_LAYOUT
    if len(dst) > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if len(dst) == 0:
        return TensorStatus.OK
    # Independent output segments can be filled concurrently. This is an
    # AOT-selected large-buffer path, not an unsupported performance claim.
    # Every shard writes disjoint FP32 cells; no reads of uninitialized dst.
    var policy = CpuWorkPolicy(1048576, 8)
    var workers = recommend_worker_count(len(dst), policy)
    if workers > 1:
        var ptr = dst.unsafe_ptr()
        comptime width = simd_width_of[DType.float32]()
        var count = len(dst)

        def fill_disjoint_shard(shard: Int) {imm}:
            var q = count // workers
            var rem = count % workers
            var first = shard * q + min(shard, rem)
            var length = q
            if shard < rem:
                length += 1
            var last = first + length
            var vector_end = first + length - length % width
            var values = SIMD[DType.float32, width](value)
            for i in range(first, vector_end, width):
                ptr.unsafe_store[alignment=4](i, values)
            for i in range(vector_end, last):
                ptr.unsafe_store[alignment=4](i, value)

        return execute_parallel_shards(fill_disjoint_shard, count, policy)
    _fill_fp32_contiguous[simd_width_of[DType.float32]()](dst, value)
    return TensorStatus.OK


def cpu_fill_strided[rank: Int, o: Origin[mut=True]](
    mut dst: TensorMutView[rank, o], value: Float32,
) -> TensorStatus:
    comptime assert rank >= 0 and rank <= 8, "FP32 tensor rank must be 0..8"
    var proof = dst._checked_current_layout()
    if not proof.is_ok():
        return proof.status
    var mapping = proof.payload.value()
    if not mapping.is_proven_unique():
        return TensorStatus.OVERLAPPING
    if mapping.shape.numel() == 0:
        return TensorStatus.OK
    if mapping.is_c_contiguous():
        var contiguous = mut_as_c_span(dst)
        if not contiguous.is_ok():
            return contiguous.status
        var writable = contiguous.payload.value()
        return cpu_fill(writable, value)
    if mapping.is_f_contiguous():
        var contiguous = mut_as_f_span(dst)
        if not contiguous.is_ok():
            return contiguous.status
        var writable = contiguous.payload.value()
        return cpu_fill(writable, value)
    if mapping.is_exhaustive():
        # Logical traversal order is irrelevant to fill. In particular,
        # flipped C/F views and dense transposes are one unique physical run.
        # The validated span bounds and uniqueness proof authorize SIMD fill
        # of that run without N-D carry, scatter or a temporary allocation.
        var start = mapping.span_lo
        var length = mapping.shape.numel()
        var contiguous = dst.data[start:start + length]
        return cpu_fill(contiguous, value)
    var cursor = NDIndexIterator[rank](mapping.shape)
    while not cursor.done():
        dst.data[mapping.physical_offset_unchecked(cursor.coordinates)] = value
        cursor.advance()
    return TensorStatus.OK


# Zero is a specialized FP32 SIMD fill. Reusing the canonical contiguous
# writer avoids a separate scalar loop or an unchecked memset byte policy.
def cpu_zero[o: Origin[mut=True]](
    mut dst: Span[mut=True, Float32, o],
) -> TensorStatus:
    return cpu_fill(dst, Float32(0.0))


def cpu_zero_strided[rank: Int, o: Origin[mut=True]](
    mut dst: TensorMutView[rank, o],
) -> TensorStatus:
    return cpu_fill_strided(dst, Float32(0.0))


# Optional hardware-guided prefetch policy for proven-disjoint large FP32
# copies. The caller must pass a device/CPU-profile-chosen threshold and
# prefetch budget; this routine never silently enables prefetch on a normal
# copy. The final native memcpy remains ONE bulk operation, preserving the
# platform's tuned copy implementation instead of creating tiny memcpy chunks.
# Locality is a compile-time specialization, so no per-line runtime branch.
def cpu_prefetch_copy[
    locality: Int, dst_origin: Origin[mut=True], src_origin: Origin[mut=False],
](
    mut destination: Span[mut=True, Float32, dst_origin],
    source: Span[mut=False, Float32, src_origin],
    min_prefetch_items: Int,
    max_prefetch_lines: Int,
) -> TensorStatus:
    comptime assert locality >= 0 and locality <= 3, "prefetch locality must be 0..3"
    if min_prefetch_items < 0 or max_prefetch_lines < 0:
        return TensorStatus.INVALID_LAYOUT
    if len(destination) != len(source):
        return TensorStatus.SIZE_MISMATCH
    var count = len(source)
    if count < 0:
        return TensorStatus.INVALID_LAYOUT
    if count > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if count == 0:
        return TensorStatus.OK
    # Unsafe alias metadata or a shifted overlap must NEVER select memcpy.
    # cpu_copy_contiguous will conservatively choose memmove on that path.
    if (count >= min_prefetch_items and max_prefetch_lines > 0
        and host_output_disjoint_from_input(destination, source)):
        # 16 Float32 values = 64 bytes. Only prefetch lines entirely inside
        # the validated source Span. This is a non-binding cache hint, not a
        # read and not a memory-lifetime fence. Caller budgets the hint count.
        var lines = min(max_prefetch_lines, count // 16)
        var source_ptr = source.unsafe_ptr()
        for i in range(lines):
            var at = i * 16
            comptime if locality == 0:
                prefetch[params=PrefetchOptions().no_locality()](
                    source_ptr.unsafe_offset(at)
                )
            elif locality == 1:
                prefetch[params=PrefetchOptions().low_locality()](
                    source_ptr.unsafe_offset(at)
                )
            elif locality == 2:
                prefetch[params=PrefetchOptions().medium_locality()](
                    source_ptr.unsafe_offset(at)
                )
            else:
                prefetch[params=PrefetchOptions().high_locality()](
                    source_ptr.unsafe_offset(at)
                )
    return cpu_copy_contiguous(destination, source)
