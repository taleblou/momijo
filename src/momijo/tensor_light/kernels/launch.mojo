# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR KERNEL / KERNEL LAUNCH GEOMETRY
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Kernels
# Module:        momijo.tensor_light.kernels.launch
# Source:        src/momijo/tensor_light/kernels/launch.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines kernel launch geometry for the checked CPU/GPU tensor kernel
#   layer.
#
#   Public declarations include `GpuLaunchShape`, `gpu_grid_1d`,
#   `run_owned_bound_gpu_binary_fp32`, `run_owned_bound_gpu_gemm_fp32`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement or declare tensor kernels behind checked launch boundaries.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `GpuLaunchShape`,
#     `gpu_grid_1d`, `run_owned_bound_gpu_binary_fp32`,
#     `run_owned_bound_gpu_gemm_fp32`, `run_owned_bound_gpu_unary_fp32` and
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

# Explicit launch-geometry policy shared by all precompiled GPU variants.

from ..internal.status import TensorResult, TensorStatus
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.host.device_context import DeviceExternalFunction
from ..execution.execution_plan import ExecutionPlan, ExecutionPlacement
from ..dispatch.specialization import BinaryAOTArtifactIdentity, GemmAOTArtifactIdentity, UnaryAOTArtifactIdentity, ReduceRowsAOTArtifactIdentity, ScanRowsAOTArtifactIdentity
from ..transfer.device_transfer import device_fp32_regions_disjoint
from ..layout.layout import TensorLayout
from ..kernels.gpu.elementwise import gpu_binary_strided_fp32
from ..kernels.gpu.elementwise import gpu_unary_strided_fp32
from ..kernels.gpu.reduction import gpu_reduce_axis_strided_fp32
from ..kernels.gpu.reduction import gpu_scan_axis_strided_fp32
from ..kernels.gpu.reduction import gpu_reduce_axes_strided_fp32

@fieldwise_init
struct GpuLaunchShape(Copyable):
    var grid_blocks: Int
    var block_threads: Int
    var shared_bytes: Int

    def validated_snapshot(
        self, maximum_threads: Int, maximum_blocks: Int,
        maximum_shared_bytes: Int,
    ) -> TensorResult[Self]:
        # For external AOT launch shapes, verify every hardware-facing field
        # before native enqueue. The zero-grid case is legal only as metadata;
        # the caller must SKIP native enqueue when grid_blocks == 0.
        if maximum_threads <= 0 or maximum_blocks <= 0 or maximum_shared_bytes < 0:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if self.grid_blocks < 0 or self.grid_blocks > maximum_blocks:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if self.block_threads <= 0 or self.block_threads > maximum_threads:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if self.shared_bytes < 0 or self.shared_bytes > maximum_shared_bytes:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        return TensorResult[Self](
            Self(self.grid_blocks, self.block_threads, self.shared_bytes)
        )

def gpu_grid_1d(
    items: Int, threads: Int, maximum_threads: Int = 1024,
    maximum_blocks: Int = 2147483647,
) -> TensorResult[GpuLaunchShape]:
    # The one-thread-per-output-element ABI requires a representable 1-D
    # launch. Do not silently truncate the grid dimension for huge tensors.
    # A backend with a narrower limit supplies its own maximum_blocks.
    if items < 0 or threads <= 0 or maximum_threads <= 0 or threads > maximum_threads or maximum_blocks <= 0:
        return TensorResult[GpuLaunchShape](TensorStatus.INVALID_LAYOUT)
    # This FP32 launch maps one logical work item to one stored Float32.
    # Reject byte-unrepresentable work counts at the public geometry boundary,
    # rather than assuming every caller arrived through a checked Tensor.
    if items > Int.MAX // 4:
        return TensorResult[GpuLaunchShape](TensorStatus.OVERFLOW)
    var blocks = items // threads
    if items % threads != 0:
        blocks += 1
    if blocks > maximum_blocks:
        return TensorResult[GpuLaunchShape](TensorStatus.UNSUPPORTED)
    return TensorResult[GpuLaunchShape](GpuLaunchShape(blocks, threads, 0))


# Owner-scoped, synchronous, FP32-only AOT GPU primitive.
# Unlike enqueue helpers this never returns an event without its resources;
# caller-owned DeviceBuffers remain borrowed through the completion wait.
# This is a LOWERING adapter, not a second tensor architecture or JIT path.

def run_owned_bound_gpu_binary_fp32[operation: Int](
    ctx: DeviceContext,
    artifact: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity,
    target_arch: Int,
    selected: ExecutionPlan,
    output: DeviceBuffer[DType.float32],
    a: DeviceBuffer[DType.float32],
    b: DeviceBuffer[DType.float32],
) -> TensorStatus:
    comptime assert operation >= 0 and operation <= 5, "FP32 binary opcode must be 0..5"
    if target_arch <= 0:
        return TensorStatus.DEVICE_MISMATCH
    var native = selected.checked_native_device_snapshot()
    if not native.is_ok():
        return native.status
    var plan = native.payload.value().copy()
    var count = plan.total_items
    if count < 0 or count > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if len(output) < count or len(a) < count or len(b) < count:
        return TensorStatus.SIZE_MISMATCH
    # The flat native AOT output must cover the complete physical owner.
    # An arbitrary subregion instead requires the checked strided writer.
    if count > 0 and len(output) != count:
        return TensorStatus.NOT_REPRESENTABLE
    if count == 0:
        # Empty tensors have no native launch, so no executable binding is needed.
        return TensorStatus.OK
    var binding = identity.validate_for(operation, plan.aot_variant, target_arch)
    if binding != TensorStatus.OK:
        return binding
    # The fixed block size and ABI are checked only for real native work.
    if identity.threads_per_block != 256:
        return TensorStatus.UNSUPPORTED
    # This AOT binary ABI is out-of-place. Distinct DeviceBuffer wrappers do
    # not imply distinct allocations, including for non-owning sub-buffers.
    if not device_fp32_regions_disjoint(output, count, a, count):
        return TensorStatus.ALIAS_UNKNOWN
    if not device_fp32_regions_disjoint(output, count, b, count):
        return TensorStatus.ALIAS_UNKNOWN
    try:
        var geometry = gpu_grid_1d(count, 256)
        if not geometry.is_ok():
            return geometry.status
        var blocks = geometry.payload.value().grid_blocks
        ctx.enqueue_function(
            artifact, output, a, b, count, grid_dim=blocks, block_dim=256,
        )
        var completed = ctx.create_event()
        ctx.stream().record_event(completed)
        completed.synchronize()
        return TensorStatus.OK
    except:
        # Never return a borrowed owner while a native enqueue may still be
        # pending. The adapter is fail-closed; the execution context must
        # remain alive and quiesce its stream after a backend exception.
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Trusted-ARTIFACT GEMM ABI adapter for fixed FP32 storage, not an MMA
# implementation. The source-only package cannot certify arbitrary native
# PTX or ensure that a caller's external function truly executes TF32.
# Public Tensor methods must validate C-order/zero-offset logical descriptors
# before supplying these physical buffers. All three owners stay borrowed and
# in scope until the stream's completion event is synchronized.
def run_owned_bound_gpu_gemm_fp32(
    ctx: DeviceContext,
    artifact: DeviceExternalFunction,
    identity: GemmAOTArtifactIdentity,
    target_arch: Int,
    math_mode: Int,
    selected: ExecutionPlan,
    output: DeviceBuffer[DType.float32],
    a: DeviceBuffer[DType.float32],
    b: DeviceBuffer[DType.float32],
    m: Int, k: Int, n: Int,
) -> TensorStatus:
    var native = selected.checked_native_device_snapshot()
    if not native.is_ok():
        return native.status
    var plan = native.payload.value().copy()
    # Shape-zero and K=0 require no GEMM artifact, but they must not turn
    # invalid hardware/math metadata into an apparently valid invocation.
    if target_arch <= 0:
        return TensorStatus.DEVICE_MISMATCH
    if math_mode < 0 or math_mode > 3:
        return TensorStatus.MATH_MODE_UNSUPPORTED
    if m < 0 or k < 0 or n < 0:
        return TensorStatus.INVALID_SHAPE
    if n > 0 and (m > Int.MAX // n or k > Int.MAX // n):
        return TensorStatus.OVERFLOW
    if k > 0 and m > Int.MAX // k:
        return TensorStatus.OVERFLOW
    var c_count = m * n
    var a_count = m * k
    var b_count = k * n
    if c_count > Int.MAX // 4 or a_count > Int.MAX // 4 or b_count > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if plan.total_items != c_count:
        return TensorStatus.SIZE_MISMATCH
    if len(output) < c_count or len(a) < a_count or len(b) < b_count:
        return TensorStatus.SIZE_MISMATCH
    if c_count > 0 and len(output) != c_count:
        return TensorStatus.NOT_REPRESENTABLE
    # GEMM with K=0 has a mathematically defined result: C = zeros(M,N).
    # Never call an AOT GEMM artifact whose K=0 ABI is not authenticated.
    # MAX's whole-buffer fill is only legal if physical C has exactly the
    # validated logical extent (including no trailing unrelated data).
    if c_count > 0 and k == 0:
        if len(output) != c_count:
            return TensorStatus.NOT_REPRESENTABLE
        try:
            output.enqueue_fill(Float32(0.0))
            var filled_event = ctx.create_event()
            ctx.stream().record_event(filled_event)
            filled_event.synchronize()
            return TensorStatus.OK
        except:
            try:
                ctx.stream().synchronize()
            except:
                return TensorStatus.GPU_QUIESCENCE_UNPROVEN
            return TensorStatus.TRANSFER_UNAVAILABLE
    if c_count == 0:
        # No output element is writable: no AOT symbol, enqueue or event.
        return TensorStatus.OK
    # Strict native ABI is required only when the GEMM artifact actually runs;
    # K=0 was handled by a separately initialized zero fill above.
    var binding = identity.validate_for(plan.aot_variant, target_arch, math_mode)
    if binding != TensorStatus.OK:
        return binding
    if (identity.flavor != GemmAOTArtifactIdentity.STRICT_SIMT
        and identity.flavor != GemmAOTArtifactIdentity.TF32_MMA):
        return TensorStatus.UNSUPPORTED
    if not device_fp32_regions_disjoint(output, c_count, a, a_count):
        return TensorStatus.ALIAS_UNKNOWN
    if not device_fp32_regions_disjoint(output, c_count, b, b_count):
        return TensorStatus.ALIAS_UNKNOWN
    try:
        if c_count > 0:
            var blocks = 0
            if identity.flavor == GemmAOTArtifactIdentity.TF32_MMA:
                # Each native CTA computes exactly one 16x16 C tile. This
                # MUST match the independently reviewed TF32 ABI v2 symbol.
                var tm = m // 16
                if m % 16 != 0:
                    tm += 1
                var tn = n // 16
                if n % 16 != 0:
                    tn += 1
                if tm <= 0 or tn <= 0 or tm > 2147483647 // tn:
                    return TensorStatus.OVERFLOW
                blocks = tm * tn
            else:
                var geometry = gpu_grid_1d(c_count, 256)
                if not geometry.is_ok():
                    return geometry.status
                blocks = geometry.payload.value().grid_blocks
            ctx.enqueue_function(
                artifact, output, a, b, m, k, n, grid_dim=blocks, block_dim=256,
            )
        var event = ctx.create_event()
        ctx.stream().record_event(event)
        event.synchronize()
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Synchronous owner-borrowed GPU unary AOT adapter; a completion event must
# precede release/reuse of either allocation. No implicit host fallback.
def run_owned_bound_gpu_unary_fp32[operation: Int](
    ctx: DeviceContext, artifact: DeviceExternalFunction,
    identity: UnaryAOTArtifactIdentity, target_arch: Int,
    selected: ExecutionPlan,
    output: DeviceBuffer[DType.float32], source: DeviceBuffer[DType.float32],
) -> TensorStatus:
    comptime assert operation >= 0 and operation <= 8, "invalid unary opcode"
    if target_arch <= 0:
        return TensorStatus.DEVICE_MISMATCH
    var native = selected.checked_native_device_snapshot()
    if not native.is_ok():
        return native.status
    var plan = native.payload.value().copy()
    var count = plan.total_items
    if count < 0 or count > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if len(output) < count or len(source) < count:
        return TensorStatus.SIZE_MISMATCH
    if count > 0 and len(output) != count:
        return TensorStatus.NOT_REPRESENTABLE
    if count == 0:
        # No kernel or artifact needed for an empty GPU-native operation.
        return TensorStatus.OK
    var bound = identity.validate_for(operation, plan.aot_variant, target_arch)
    if bound != TensorStatus.OK:
        return bound
    if not device_fp32_regions_disjoint(output, count, source, count):
        return TensorStatus.ALIAS_UNKNOWN
    try:
        var geometry = gpu_grid_1d(count, 256)
        if not geometry.is_ok():
            return geometry.status
        var blocks = geometry.payload.value().grid_blocks
        ctx.enqueue_function(
            artifact, output, source, count, grid_dim=blocks, block_dim=256,
        )
        var event = ctx.create_event()
        ctx.stream().record_event(event)
        event.synchronize()
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Checked borrowed-owner AOT adapter for five distinct row reductions.
# Op 0..4 maps to SUM/PROD/MIN/MAX/MEAN, with a distinct offline variant
# identity for each symbol. All kernels use (out, input, rows, cols).
# The lifetime of the two device owners covers completion synchronization.
def run_owned_bound_gpu_reduce_rows_fp32[operation: Int](
    ctx: DeviceContext,
    artifact: DeviceExternalFunction,
    identity: ReduceRowsAOTArtifactIdentity,
    target_arch: Int,
    selected: ExecutionPlan,
    output: DeviceBuffer[DType.float32],
    source: DeviceBuffer[DType.float32],
    rows: Int,
    cols: Int,
) -> TensorStatus:
    var native = selected.checked_native_device_snapshot()
    if not native.is_ok():
        return native.status
    var plan = native.payload.value().copy()
    comptime if operation < 0 or operation > 4:
        return TensorStatus.UNSUPPORTED
    if target_arch <= 0:
        return TensorStatus.DEVICE_MISMATCH
    if rows < 0 or cols < 0:
        return TensorStatus.INVALID_SHAPE
    # For non-empty outputs, MIN/MAX/MEAN have no identity value for an
    # empty reduction axis. Reject before AOT launch (no out-of-range read).
    comptime if operation >= 2:
        if rows > 0 and cols == 0:
            return TensorStatus.SIZE_MISMATCH
    if rows > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if rows > 0 and cols > Int.MAX // rows:
        return TensorStatus.OVERFLOW
    var source_count = rows * cols
    if source_count > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    # For a reduction, logical plan work is OUTPUT rows, not input elements.
    if plan.total_items != rows:
        return TensorStatus.SIZE_MISMATCH
    if len(output) < rows or len(source) < source_count:
        return TensorStatus.SIZE_MISMATCH
    # Positive outputs must occupy the entire device allocation. An empty
    # Tensor may legitimately have a physical one-element allocation pad.
    if rows > 0 and len(output) != rows:
        return TensorStatus.NOT_REPRESENTABLE
    if rows == 0:
        # No output elements or native work; bypass unused AOT artifact.
        return TensorStatus.OK
    # For empty reduction axes, SUM and PROD have exact identities. This
    # output still contains 'rows' elements and MUST be fully initialized;
    # an AOT reduction symbol is neither needed nor safe to assume for K=0.
    # The event is joined before the caller publishes a complete owner write.
    if cols == 0:
        try:
            comptime if operation == 0:
                output.enqueue_fill(Float32(0.0))
            else:
                output.enqueue_fill(Float32(1.0))
            var identity_fill_done = ctx.create_event()
            ctx.stream().record_event(identity_fill_done)
            identity_fill_done.synchronize()
            return TensorStatus.OK
        except:
            try:
                ctx.stream().synchronize()
            except:
                return TensorStatus.GPU_QUIESCENCE_UNPROVEN
            return TensorStatus.TRANSFER_UNAVAILABLE
    var binding = identity.validate_for_op(operation, plan.aot_variant, target_arch)
    if binding != TensorStatus.OK:
        return binding
    # A row reduction is out-of-place. Even a zero-column reduction writes
    # rows output values; a nonzero input must not share those addresses.
    if not device_fp32_regions_disjoint(output, rows, source, source_count):
        return TensorStatus.ALIAS_UNKNOWN
    try:
        var geometry = gpu_grid_1d(rows, 256)
        if not geometry.is_ok():
            return geometry.status
        var blocks = geometry.payload.value().grid_blocks
        ctx.enqueue_function(
            artifact, output, source, rows, cols,
            grid_dim=blocks, block_dim=256,
        )
        var completion = ctx.create_event()
        ctx.stream().record_event(completion)
        completion.synchronize()
        return TensorStatus.OK
    except:
        # If the event fails, quiesce the stream before returning borrowed
        # owners. Stream failure remains a context-lifetime recovery case.
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Retain the established public SUM entry point and ABI. Its identity is
# additionally pinned to SUM_ROWS by validate_for_op inside the shared path.
def run_owned_bound_gpu_sum_rows_fp32(
    ctx: DeviceContext,
    artifact: DeviceExternalFunction,
    identity: ReduceRowsAOTArtifactIdentity,
    target_arch: Int,
    selected: ExecutionPlan,
    output: DeviceBuffer[DType.float32],
    source: DeviceBuffer[DType.float32],
    rows: Int,
    cols: Int,
) -> TensorStatus:
    return run_owned_bound_gpu_reduce_rows_fp32[0](
        ctx, artifact, identity, target_arch, selected,
        output, source, rows, cols,
    )

# Bound synchronous native AOT row-wise scan, each output Float32 written
# once. The complete contiguous input/output owners are borrowed until the
# GPU completion event (or a successful stream quiescence) is observed.
# This adapter cannot attest that an arbitrary external function is trusted;
# artifact loading and target-native validation are separate release duties.
def run_owned_bound_gpu_scan_rows_fp32[operation: Int](
    ctx: DeviceContext,
    artifact: DeviceExternalFunction,
    identity: ScanRowsAOTArtifactIdentity,
    target_arch: Int,
    selected: ExecutionPlan,
    output: DeviceBuffer[DType.float32],
    source: DeviceBuffer[DType.float32],
    rows: Int,
    cols: Int,
) -> TensorStatus:
    comptime if operation < 0 or operation > 3:
        return TensorStatus.UNSUPPORTED
    if target_arch <= 0:
        return TensorStatus.DEVICE_MISMATCH
    var checked = selected.checked_native_device_snapshot()
    if not checked.is_ok():
        return checked.status
    var plan = checked.payload.value().copy()
    if rows < 0 or cols < 0:
        return TensorStatus.INVALID_SHAPE
    if rows > 0 and cols > Int.MAX // rows:
        return TensorStatus.OVERFLOW
    var count = rows * cols
    if count > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    # A scan writes the entire input-sized output, unlike a row reduction.
    if plan.total_items != count:
        return TensorStatus.SIZE_MISMATCH
    if len(output) < count or len(source) < count:
        return TensorStatus.SIZE_MISMATCH
    if count > 0 and len(output) != count:
        return TensorStatus.NOT_REPRESENTABLE
    if count == 0:
        # An empty scan does not depend on any prelinked GPU artifact.
        return TensorStatus.OK
    var bound = identity.validate_for_op(operation, plan.aot_variant, target_arch)
    if bound != TensorStatus.OK:
        return bound
    if not device_fp32_regions_disjoint(output, count, source, count):
        return TensorStatus.ALIAS_UNKNOWN
    var shape = gpu_grid_1d(rows, 256)
    if not shape.is_ok():
        return shape.status
    try:
        var blocks = shape.payload.value().grid_blocks
        ctx.enqueue_function(
            artifact, output, source, rows, cols,
            grid_dim=blocks, block_dim=256,
        )
        var completion = ctx.create_event()
        ctx.stream().record_event(completion)
        completion.synchronize()
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            # The native context may still reference the borrowed owners;
            # caller must quarantine/reset context on unrecoverable failure.
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Fully checked owner-scoped fallback for arbitrary-layout FP32 binary GPU
# execution. It lowers the SAME ExecutionPlan as the contiguous AOT path.
# All three DeviceBuffer owners and DeviceContext remain borrowed until the
# stream event has joined; no temporary host buffer or implicit migration.
# This is a compiled source-kernel fallback, NOT an authenticated external
# binary artifact. Its source identity cannot satisfy UASDEA's AOT manifest
# requirement by itself.

def run_owned_gpu_binary_strided_fp32[
    output_rank: Int, left_rank: Int, right_rank: Int, operation: Int,
](
    ctx: DeviceContext,
    selected: ExecutionPlan,
    development_allow_runtime_compilation: Bool,
    output: DeviceBuffer[DType.float32],
    left: DeviceBuffer[DType.float32],
    right: DeviceBuffer[DType.float32],
    out_layout: TensorLayout[output_rank],
    left_layout: TensorLayout[left_rank],
    right_layout: TensorLayout[right_rank],
) -> TensorStatus:
    comptime assert output_rank >= 0 and output_rank <= 8
    comptime assert left_rank >= 0 and left_rank <= 8
    comptime assert right_rank >= 0 and right_rank <= 8
    comptime assert operation >= 0 and operation <= 5
    # NumPy-like binary broadcast rank: no extra unrequested leading axes.
    comptime assert output_rank == max(left_rank, right_rank), "output rank must equal maximum input rank"
    var proof = selected.checked_native_device_snapshot()
    if not proof.is_ok():
        return proof.status
    var plan = proof.payload.value().copy()
    if plan.aot_variant != 0:
        return TensorStatus.UNSUPPORTED
    # UASDEA RELEASE SAFETY: ctx.enqueue_function[kernel] may compile on
    # demand. Production must use a prelinked immutable AOT artifact, not
    # this source fallback. Disable implicit JIT unless explicitly opted in.
    if not development_allow_runtime_compilation:
        return TensorStatus.UNSUPPORTED
    var out_proof = out_layout.validated_snapshot(len(output))
    if not out_proof.is_ok():
        return out_proof.status
    var left_proof = left_layout.validated_snapshot(len(left))
    if not left_proof.is_ok():
        return left_proof.status
    var right_proof = right_layout.validated_snapshot(len(right))
    if not right_proof.is_ok():
        return right_proof.status
    var dest = out_proof.payload.value()
    var a = left_proof.payload.value()
    var b = right_proof.payload.value()
    if not dest.is_proven_unique():
        return TensorStatus.OVERLAPPING
    var count = dest.shape.numel()
    if plan.total_items != count:
        return TensorStatus.SIZE_MISMATCH
    # All arrays are fixed register metadata with a padded unused rank-0
    # slot. A compiler can unroll the per-axis conversion up to 8 axes.
    var extents = SIMD[DType.int64, 8](1)
    var dest_steps = SIMD[DType.int64, 8](0)
    var a_steps = SIMD[DType.int64, 8](0)
    var b_steps = SIMD[DType.int64, 8](0)
    comptime for axis in range(output_rank):
        var target_dim = dest.shape.dims.get(axis)
        var a_dim = 1
        var b_dim = 1
        var a_step = 0
        var b_step = 0
        comptime if axis >= output_rank - left_rank:
            var ai = axis - (output_rank - left_rank)
            a_dim = a.shape.dims.get(ai)
            a_step = a.strides.get(ai)
        comptime if axis >= output_rank - right_rank:
            var bi = axis - (output_rank - right_rank)
            b_dim = b.shape.dims.get(bi)
            b_step = b.strides.get(bi)
        if a_dim != b_dim and a_dim != 1 and b_dim != 1:
            return TensorStatus.SHAPE_MISMATCH
        var expected = a_dim
        if a_dim == 1:
            expected = b_dim
        if expected != target_dim:
            return TensorStatus.SHAPE_MISMATCH
        extents[axis] = Int64(target_dim)
        dest_steps[axis] = Int64(dest.strides.get(axis))
        if a_dim == 1 and target_dim != 1:
            a_step = 0
        if b_dim == 1 and target_dim != 1:
            b_step = 0
        a_steps[axis] = Int64(a_step)
        b_steps[axis] = Int64(b_step)
    if count == 0:
        return TensorStatus.OK
    # DeviceBuffer views can overlap even when the Mojo objects differ.
    # Prove disjointness conservatively against the WHOLE owner allocations,
    # not merely equal logical element counts, before launching in parallel.
    if not device_fp32_regions_disjoint(output, len(output), left, len(left)):
        return TensorStatus.ALIAS_UNKNOWN
    if not device_fp32_regions_disjoint(output, len(output), right, len(right)):
        return TensorStatus.ALIAS_UNKNOWN
    var grid = gpu_grid_1d(count, 256)
    if not grid.is_ok():
        return grid.status
    try:
        var use_dense = (
            dest.is_c_contiguous() and a.is_c_contiguous()
            and b.is_c_contiguous() and a.shape.numel() == count
            and b.shape.numel() == count
        )
        if use_dense:
            ctx.enqueue_function[
                gpu_binary_strided_fp32[operation, output_rank, True]
            ](
                output, left, right, extents, dest_steps, a_steps, b_steps,
                Int64(dest.offset), Int64(a.offset), Int64(b.offset), Int64(count),
                grid_dim=grid.payload.value().grid_blocks, block_dim=256,
            )
        else:
            ctx.enqueue_function[
                gpu_binary_strided_fp32[operation, output_rank, False]
            ](
                output, left, right, extents, dest_steps, a_steps, b_steps,
                Int64(dest.offset), Int64(a.offset), Int64(b.offset), Int64(count),
                grid_dim=grid.payload.value().grid_blocks, block_dim=256,
            )
        var completion = ctx.create_event()
        ctx.stream().record_event(completion)
        completion.synchronize()
        return TensorStatus.OK
    except:
        # The fallback wait proves quiescence, NOT successful device writes.
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED



def run_owned_gpu_unary_strided_fp32[rank: Int, operation: Int](
    ctx: DeviceContext,
    selected: ExecutionPlan,
    development_allow_runtime_compilation: Bool,
    output: DeviceBuffer[DType.float32],
    source: DeviceBuffer[DType.float32],
    out_layout: TensorLayout[rank],
    source_layout: TensorLayout[rank],
) -> TensorStatus:
    comptime assert rank >= 0 and rank <= 8
    comptime assert operation >= 0 and operation <= 8
    var proof = selected.checked_native_device_snapshot()
    if not proof.is_ok():
        return proof.status
    var plan = proof.payload.value().copy()
    if plan.aot_variant != 0:
        return TensorStatus.UNSUPPORTED
    # UASDEA RELEASE SAFETY: ctx.enqueue_function[kernel] may compile on
    # demand. Production must use a prelinked immutable AOT artifact, not
    # this source fallback. Disable implicit JIT unless explicitly opted in.
    if not development_allow_runtime_compilation:
        return TensorStatus.UNSUPPORTED
    var out_proof = out_layout.validated_snapshot(len(output))
    if not out_proof.is_ok():
        return out_proof.status
    var src_proof = source_layout.validated_snapshot(len(source))
    if not src_proof.is_ok():
        return src_proof.status
    var dest = out_proof.payload.value()
    var src = src_proof.payload.value()
    if not dest.shape.same_extents(src.shape):
        return TensorStatus.SHAPE_MISMATCH
    if not dest.is_proven_unique():
        return TensorStatus.OVERLAPPING
    var count = dest.shape.numel()
    if plan.total_items != count:
        return TensorStatus.SIZE_MISMATCH
    var extents = SIMD[DType.int64, 8](1)
    var dest_steps = SIMD[DType.int64, 8](0)
    var src_steps = SIMD[DType.int64, 8](0)
    comptime for axis in range(rank):
        extents[axis] = Int64(dest.shape.dims.get(axis))
        dest_steps[axis] = Int64(dest.strides.get(axis))
        src_steps[axis] = Int64(src.strides.get(axis))
    if count == 0:
        return TensorStatus.OK
    if not device_fp32_regions_disjoint(output, len(output), source, len(source)):
        return TensorStatus.ALIAS_UNKNOWN
    var grid = gpu_grid_1d(count, 256)
    if not grid.is_ok():
        return grid.status
    try:
        if dest.is_c_contiguous() and src.is_c_contiguous():
            ctx.enqueue_function[
                gpu_unary_strided_fp32[operation, rank, True]
            ](
                output, source, extents, dest_steps, src_steps,
                Int64(dest.offset), Int64(src.offset), Int64(count),
                grid_dim=grid.payload.value().grid_blocks, block_dim=256,
            )
        else:
            ctx.enqueue_function[
                gpu_unary_strided_fp32[operation, rank, False]
            ](
                output, source, extents, dest_steps, src_steps,
                Int64(dest.offset), Int64(src.offset), Int64(count),
                grid_dim=grid.payload.value().grid_blocks, block_dim=256,
            )
        var completion = ctx.create_event()
        ctx.stream().record_event(completion)
        completion.synchronize()
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Native FP32 keepdims reduction for ANY checked, strided input axis.
# AOT rank/axis/opcode specialize the device kernel, while source bounds,
# allocation identity and output mapping are checked once at the host edge.

def run_owned_gpu_reduce_axis_strided_fp32[rank: Int, axis: Int, operation: Int](
    ctx: DeviceContext,
    selected: ExecutionPlan,
    development_allow_runtime_compilation: Bool,
    output: DeviceBuffer[DType.float32],
    source: DeviceBuffer[DType.float32],
    out_layout: TensorLayout[rank],
    source_layout: TensorLayout[rank],
) -> TensorStatus:
    comptime assert rank >= 1 and rank <= 8
    comptime assert axis >= 0 and axis < rank
    comptime assert operation >= 0 and operation <= 4
    var proof = selected.checked_native_device_snapshot()
    if not proof.is_ok():
        return proof.status
    var plan = proof.payload.value().copy()
    if plan.aot_variant != 0:
        return TensorStatus.UNSUPPORTED
    # UASDEA RELEASE SAFETY: ctx.enqueue_function[kernel] may compile on
    # demand. Production must use a prelinked immutable AOT artifact, not
    # this source fallback. Disable implicit JIT unless explicitly opted in.
    if not development_allow_runtime_compilation:
        return TensorStatus.UNSUPPORTED
    var out_proof = out_layout.validated_snapshot(len(output))
    if not out_proof.is_ok():
        return out_proof.status
    var source_proof = source_layout.validated_snapshot(len(source))
    if not source_proof.is_ok():
        return source_proof.status
    var dest = out_proof.payload.value()
    var src = source_proof.payload.value()
    if not dest.is_proven_unique():
        return TensorStatus.OVERLAPPING
    var reduction_size = src.shape.dims.get(axis)
    comptime for i in range(rank):
        var wanted = src.shape.dims.get(i)
        if i == axis:
            wanted = 1
        if dest.shape.dims.get(i) != wanted:
            return TensorStatus.SHAPE_MISMATCH
    var count = dest.shape.numel()
    if plan.total_items != count:
        return TensorStatus.SIZE_MISMATCH
    if count > 0 and reduction_size == 0 and operation >= 2:
        return TensorStatus.SIZE_MISMATCH
    var dimensions = SIMD[DType.int64, 8](1)
    var output_steps = SIMD[DType.int64, 8](0)
    var source_steps = SIMD[DType.int64, 8](0)
    comptime for i in range(rank):
        dimensions[i] = Int64(dest.shape.dims.get(i))
        output_steps[i] = Int64(dest.strides.get(i))
        source_steps[i] = Int64(src.strides.get(i))
    if count == 0:
        return TensorStatus.OK
    if not device_fp32_regions_disjoint(output, len(output), source, len(source)):
        return TensorStatus.ALIAS_UNKNOWN
    var grid = gpu_grid_1d(count, 256)
    if not grid.is_ok():
        return grid.status
    try:
        ctx.enqueue_function[
            gpu_reduce_axis_strided_fp32[operation, rank, axis]
        ](
            output, source, dimensions, output_steps, source_steps,
            Int64(dest.offset), Int64(src.offset), Int64(count), Int64(reduction_size),
            grid_dim=grid.payload.value().grid_blocks, block_dim=256,
        )
        var completion = ctx.create_event()
        ctx.stream().record_event(completion)
        completion.synchronize()
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED



def run_owned_gpu_scan_axis_strided_fp32[rank: Int, axis: Int, operation: Int](
    ctx: DeviceContext,
    selected: ExecutionPlan,
    development_allow_runtime_compilation: Bool,
    output: DeviceBuffer[DType.float32],
    source: DeviceBuffer[DType.float32],
    out_layout: TensorLayout[rank],
    source_layout: TensorLayout[rank],
) -> TensorStatus:
    comptime assert rank >= 1 and rank <= 8
    comptime assert axis >= 0 and axis < rank
    comptime assert operation >= 0 and operation <= 3
    var proof = selected.checked_native_device_snapshot()
    if not proof.is_ok():
        return proof.status
    var plan = proof.payload.value().copy()
    if plan.aot_variant != 0:
        return TensorStatus.UNSUPPORTED
    # UASDEA RELEASE SAFETY: ctx.enqueue_function[kernel] may compile on
    # demand. Production must use a prelinked immutable AOT artifact, not
    # this source fallback. Disable implicit JIT unless explicitly opted in.
    if not development_allow_runtime_compilation:
        return TensorStatus.UNSUPPORTED
    var out_proof = out_layout.validated_snapshot(len(output))
    if not out_proof.is_ok():
        return out_proof.status
    var src_proof = source_layout.validated_snapshot(len(source))
    if not src_proof.is_ok():
        return src_proof.status
    var dest = out_proof.payload.value()
    var src = src_proof.payload.value()
    if not dest.shape.same_extents(src.shape):
        return TensorStatus.SHAPE_MISMATCH
    if not dest.is_proven_unique():
        return TensorStatus.OVERLAPPING
    var total = dest.shape.numel()
    if plan.total_items != total:
        return TensorStatus.SIZE_MISMATCH
    var extent = dest.shape.dims.get(axis)
    if extent == 0:
        return TensorStatus.OK
    var rows = total // extent
    if rows == 0:
        return TensorStatus.OK
    var dimensions = SIMD[DType.int64, 8](1)
    var output_steps = SIMD[DType.int64, 8](0)
    var source_steps = SIMD[DType.int64, 8](0)
    comptime for i in range(rank):
        dimensions[i] = Int64(dest.shape.dims.get(i))
        output_steps[i] = Int64(dest.strides.get(i))
        source_steps[i] = Int64(src.strides.get(i))
    if not device_fp32_regions_disjoint(output, len(output), source, len(source)):
        return TensorStatus.ALIAS_UNKNOWN
    var grid = gpu_grid_1d(rows, 256)
    if not grid.is_ok():
        return grid.status
    try:
        ctx.enqueue_function[
            gpu_scan_axis_strided_fp32[operation, rank, axis]
        ](
            output, source, dimensions, output_steps, source_steps,
            Int64(dest.offset), Int64(src.offset), Int64(rows), Int64(extent),
            grid_dim=grid.payload.value().grid_blocks, block_dim=256,
        )
        var completion = ctx.create_event()
        ctx.stream().record_event(completion)
        completion.synchronize()
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Any NONEMPTY subset of axes can now be reduced in ONE checked device pass,
# without a chain of intermediate device buffers or an accidental host copy.
# This is the development SOURCE-kernel fallback, NOT an authenticated native
# AOT artifact. The default release path rejects runtime compilation.

def run_owned_gpu_reduce_axes_strided_fp32[rank: Int, operation: Int](
    ctx: DeviceContext,
    selected: ExecutionPlan,
    development_allow_runtime_compilation: Bool,
    output: DeviceBuffer[DType.float32],
    source: DeviceBuffer[DType.float32],
    out_layout: TensorLayout[rank],
    source_layout: TensorLayout[rank],
    axes_mask: Int,
) -> TensorStatus:
    comptime assert rank >= 1 and rank <= 8, "FP32 multi-axis reduction rank"
    comptime assert operation >= 0 and operation <= 4, "FP32 reduction opcode"
    var allowed_mask = (Int(1) << rank) - 1
    if axes_mask <= 0 or axes_mask & ~allowed_mask != 0:
        return TensorStatus.INVALID_AXIS
    var proof = selected.checked_native_device_snapshot()
    if not proof.is_ok():
        return proof.status
    var plan = proof.payload.value().copy()
    if plan.aot_variant != 0:
        return TensorStatus.UNSUPPORTED
    var out_proof = out_layout.validated_snapshot(len(output))
    if not out_proof.is_ok():
        return out_proof.status
    var source_proof = source_layout.validated_snapshot(len(source))
    if not source_proof.is_ok():
        return source_proof.status
    var dest = out_proof.payload.value()
    var src = source_proof.payload.value()
    if not dest.is_proven_unique():
        return TensorStatus.OVERLAPPING
    comptime for axis in range(rank):
        var wanted = src.shape.dims.get(axis)
        if axes_mask & (Int(1) << axis) != 0:
            wanted = 1
        if dest.shape.dims.get(axis) != wanted:
            return TensorStatus.SHAPE_MISMATCH
    var output_count = dest.shape.numel()
    if plan.total_items != output_count:
        return TensorStatus.SIZE_MISMATCH
    if output_count == 0:
        # No output lane exists: no need to multiply possibly enormous
        # reduced extents or create a GPU launch with zero grid blocks.
        return TensorStatus.OK
    # A zero reduction extent dominates every other reduced extent.
    # Detect emptiness first so a giant extent that precedes a zero cannot
    # cause a spurious arithmetic overflow for an empty logical source.
    var empty_reduction = False
    comptime for axis in range(rank):
        if axes_mask & (Int(1) << axis) != 0:
            if src.shape.dims.get(axis) == 0:
                empty_reduction = True
    var reduction_count = 1
    if empty_reduction:
        reduction_count = 0
    else:
        comptime for axis in range(rank):
            if axes_mask & (Int(1) << axis) != 0:
                var extent = src.shape.dims.get(axis)
                if reduction_count > (Int.MAX // 4) // extent:
                    return TensorStatus.OVERFLOW
                reduction_count *= extent
    if reduction_count == 0:
        # Empty SUM/PRODUCT have identity results. Empty MEAN/EXTREMA
        # have no valid initialized Float32 output under this contract.
        comptime if operation >= 2:
            return TensorStatus.SIZE_MISMATCH
    if reduction_count > 0:
        if src.shape.numel() // output_count != reduction_count:
            return TensorStatus.SHAPE_MISMATCH
    elif src.shape.numel() != 0:
        return TensorStatus.SHAPE_MISMATCH
    if not device_fp32_regions_disjoint(output, len(output), source, len(source)):
        return TensorStatus.ALIAS_UNKNOWN
    var source_dims = SIMD[DType.int64, 8](1)
    var output_dims = SIMD[DType.int64, 8](1)
    var output_steps = SIMD[DType.int64, 8](0)
    var source_steps = SIMD[DType.int64, 8](0)
    comptime for axis in range(rank):
        source_dims[axis] = Int64(src.shape.dims.get(axis))
        output_dims[axis] = Int64(dest.shape.dims.get(axis))
        output_steps[axis] = Int64(dest.strides.get(axis))
        source_steps[axis] = Int64(src.strides.get(axis))
    var grid = gpu_grid_1d(output_count, 256)
    if not grid.is_ok():
        return grid.status
    # This is the last preflight gate. A zero-size output was returned
    # above without a launch; only real GPU work may need runtime compile.
    if not development_allow_runtime_compilation:
        return TensorStatus.UNSUPPORTED
    try:
        ctx.enqueue_function[gpu_reduce_axes_strided_fp32[operation, rank]](
            output, source, source_dims, output_dims,
            output_steps, source_steps,
            Int64(dest.offset), Int64(src.offset),
            Int64(output_count), Int64(reduction_count), Int64(axes_mask),
            grid_dim=grid.payload.value().grid_blocks, block_dim=256,
        )
        var completion = ctx.create_event()
        ctx.stream().record_event(completion)
        completion.synchronize()
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED
