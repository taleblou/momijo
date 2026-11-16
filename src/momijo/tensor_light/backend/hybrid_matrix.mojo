# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR BACKEND / HYBRID MATRIX EXECUTION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Backend Availability
# Module:        momijo.tensor_light.backend.hybrid_matrix
# Source:        src/momijo/tensor_light/backend/hybrid_matrix.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines hybrid matrix execution in the backend availability layer of the
#   FP32 Light tensor implementation.
#
#   Public declarations include `unsafe_execute_hybrid_gemm_rows_fp32`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Select or report supported execution backends explicitly.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as
#     `unsafe_execute_hybrid_gemm_rows_fp32`.
#   - Reject unsupported target or unverified completion states.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Reject unsupported targets rather than silently falling back.
#   - Keep backend selection consistent with actual runtime capability.
#   - Reject unsupported or unauthenticated execution paths explicitly.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid device initialization during pure capability queries.
#   - Preserve direct CPU paths when validated and applicable.
#   - Avoid implicit host/device migration or compilation at execution time.
#
# Maintainability:
#   - Separate availability checks from native launch authorization.
#   - Make missing GPU/hybrid implementations explicit.
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

# One bounded CPU/GPU row partition of strict-FP32 rank-two GEMM.
# The public data owners are Tensor; this module borrows their checked spans.
# No source-kernel enqueue, per-call compilation, implicit staging, or JIT.
# The caller MUST authenticate the externally supplied native AOT symbol.
# This unsafe expert path is NOT the general UASDEA GPU DAG executor.
from std.collections import Span, Optional
from std.memory import forget_deinit
from std.sys.info import simd_width_of
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer, DeviceExternalFunction
from ..internal.status import TensorStatus
from ..execution.execution_plan import ExecutionPlan, ExecutionPlacement
from ..dispatch.specialization import GemmAOTArtifactIdentity
from ..kernels.launch import gpu_grid_1d
from ..kernels.cpu.matrix import cpu_mm_contiguous
from ..alias.overlap import host_output_disjoint_from_input


struct _HybridGemmRowOwners(Movable):
    # Six separately owned allocations. Acquisition is sequential, so any
    # partial failure still has a well-defined destructor and context owner.
    var host_a: Optional[HostBuffer[DType.float32]]
    var host_b: Optional[HostBuffer[DType.float32]]
    var host_c: Optional[HostBuffer[DType.float32]]
    var device_a: Optional[DeviceBuffer[DType.float32]]
    var device_b: Optional[DeviceBuffer[DType.float32]]
    var device_c: Optional[DeviceBuffer[DType.float32]]
    var context_owner: Optional[DeviceContext]
    var quiescence_unproven: Bool

    def __init__(out self):
        self.host_a = Optional[HostBuffer[DType.float32]](None)
        self.host_b = Optional[HostBuffer[DType.float32]](None)
        self.host_c = Optional[HostBuffer[DType.float32]](None)
        self.device_a = Optional[DeviceBuffer[DType.float32]](None)
        self.device_b = Optional[DeviceBuffer[DType.float32]](None)
        self.device_c = Optional[DeviceBuffer[DType.float32]](None)
        self.context_owner = Optional[DeviceContext](None)
        self.quiescence_unproven = False

    def acquire(
        mut self, ctx: DeviceContext, left_count: Int,
        right_count: Int, result_count: Int,
    ) raises:
        # Preserve context BEFORE the first asynchronous device allocation.
        self.context_owner = Optional[DeviceContext](ctx.copy())
        self.host_a = Optional[HostBuffer[DType.float32]](
            ctx.enqueue_create_host_buffer[DType.float32](left_count)
        )
        self.host_b = Optional[HostBuffer[DType.float32]](
            ctx.enqueue_create_host_buffer[DType.float32](right_count)
        )
        self.host_c = Optional[HostBuffer[DType.float32]](
            ctx.enqueue_create_host_buffer[DType.float32](result_count)
        )
        self.device_a = Optional[DeviceBuffer[DType.float32]](
            ctx.enqueue_create_buffer[DType.float32](left_count)
        )
        self.device_b = Optional[DeviceBuffer[DType.float32]](
            ctx.enqueue_create_buffer[DType.float32](right_count)
        )
        self.device_c = Optional[DeviceBuffer[DType.float32]](
            ctx.enqueue_create_buffer[DType.float32](result_count)
        )

    def quarantine(mut self):
        self.quiescence_unproven = True

    def __deinit__(deinit self):
        if self.quiescence_unproven:
            # Unproven native completion: keep EVERY acquired allocation and
            # the issuing context. Never free a staging allocation early.
            if self.host_a:
                var owner = self.host_a.take()
                forget_deinit(owner)
            if self.host_b:
                var owner = self.host_b.take()
                forget_deinit(owner)
            if self.host_c:
                var owner = self.host_c.take()
                forget_deinit(owner)
            if self.device_a:
                var owner = self.device_a.take()
                forget_deinit(owner)
            if self.device_b:
                var owner = self.device_b.take()
                forget_deinit(owner)
            if self.device_c:
                var owner = self.device_c.take()
                forget_deinit(owner)
            if self.context_owner:
                var owner = self.context_owner.take()
                forget_deinit(owner)
        else:
            # The caller joined either its final event or the entire stream.
            # Destroy native allocations BEFORE their final context owner.
            self.host_a = Optional[HostBuffer[DType.float32]](None)
            self.host_b = Optional[HostBuffer[DType.float32]](None)
            self.host_c = Optional[HostBuffer[DType.float32]](None)
            self.device_a = Optional[DeviceBuffer[DType.float32]](None)
            self.device_b = Optional[DeviceBuffer[DType.float32]](None)
            self.device_c = Optional[DeviceBuffer[DType.float32]](None)
            # Native buffers have been destructed while the originating
            # context still existed; only now release that context owner.
            # MAX buffer destructors enqueue asynchronous frees on the
            # originating context. An earlier compute/D2H completion
            # event does NOT fence these later free commands. Drain them
            # while the native context is still alive, once at final
            # teardown (never between ordinary reused windows).
            # On a failed drain, retain the context instead of allowing
            # premature native-context destruction.
            var cleanup_completed = True
            if self.context_owner:
                try:
                    self.context_owner.value().synchronize()
                except:
                    cleanup_completed = False
            if cleanup_completed:
                self.context_owner = Optional[DeviceContext](None)
            else:
                var unrecovered_context = self.context_owner.take()
                forget_deinit(unrecovered_context)


def unsafe_execute_hybrid_gemm_rows_fp32[
    static_gpu_row_capacity: Int,
    left_origin: Origin, right_origin: Origin,
    result_origin: Origin[mut=True],
](
    ctx: DeviceContext,
    precompiled_gemm: DeviceExternalFunction,
    identity: GemmAOTArtifactIdentity,
    target_arch: Int,
    plan: ExecutionPlan,
    m: Int, k: Int, n: Int,
    left: Span[False, Float32, left_origin],
    right: Span[False, Float32, right_origin],
    mut result: Span[True, Float32, result_origin],
) -> TensorStatus:
    # Work-item units are OUTPUT ROWS, not M*N FP32 output elements.
    # GPU receives A[gpu rows,K], the ENTIRE shared B[K,N], and produces
    # C[gpu rows,N]. CPU computes the complementary, disjoint output rows.
    comptime assert static_gpu_row_capacity > 0, "positive GPU staging rows"
    var verified = plan.checked_host_gemm_rows_split_snapshot(k, n)
    if not verified.is_ok():
        return verified.status
    var selected = verified.payload.value()
    if selected.total_items != m:
        return TensorStatus.SIZE_MISMATCH
    if m <= 1 or k <= 0 or n <= 0:
        return TensorStatus.INVALID_SHAPE
    var maximum_elements = Int.MAX // 4
    if m > maximum_elements // k or k > maximum_elements // n:
        return TensorStatus.OVERFLOW
    if m > maximum_elements // n:
        return TensorStatus.OVERFLOW
    if len(left) != m * k or len(right) != k * n or len(result) != m * n:
        return TensorStatus.SIZE_MISMATCH
    # A CPU write may otherwise clobber B (which every row reuses), or an A
    # row that is still being copied to pinned staging. Reject all aliases.
    if not host_output_disjoint_from_input(result, left):
        return TensorStatus.ALIAS_UNKNOWN
    if not host_output_disjoint_from_input(result, right):
        return TensorStatus.ALIAS_UNKNOWN
    var gpu_rows = selected.gpu_end - selected.gpu_begin
    var cpu_rows = selected.cpu_end - selected.cpu_begin
    if gpu_rows <= 0 or cpu_rows <= 0 or gpu_rows > static_gpu_row_capacity:
        return TensorStatus.HYBRID_PARTITION_INVALID
    if target_arch <= 0:
        return TensorStatus.DEVICE_MISMATCH
    # The fixed flat native ABI is exactly (C,A,B,m,k,n), strict FP32 SIMT
    # with variant 14 and 256 threads. TF32 MMA is NOT silently substituted.
    var status = identity.validate_for(selected.aot_variant, target_arch, 0)
    if status != TensorStatus.OK:
        return status
    if identity.flavor != GemmAOTArtifactIdentity.STRICT_SIMT:
        return TensorStatus.UNSUPPORTED
    var left_gpu_count = gpu_rows * k
    var right_gpu_count = k * n
    var output_gpu_count = gpu_rows * n
    var geometry = gpu_grid_1d(output_gpu_count, 256)
    if not geometry.is_ok():
        return geometry.status
    var owners = _HybridGemmRowOwners()
    try:
        owners.acquire(ctx, left_gpu_count, right_gpu_count, output_gpu_count)
        var buffers_ready = ctx.create_event()
        ctx.stream().record_event(buffers_ready)
        buffers_ready.synchronize()
        for i in range(left_gpu_count):
            owners.host_a.value()[i] = left[selected.gpu_begin * k + i]
        for i in range(right_gpu_count):
            owners.host_b.value()[i] = right[i]
        ctx.enqueue_copy(
            dst_buf=owners.device_a.value(), src_buf=owners.host_a.value(),
        )
        ctx.enqueue_copy(
            dst_buf=owners.device_b.value(), src_buf=owners.host_b.value(),
        )
        ctx.enqueue_function(
            precompiled_gemm,
            owners.device_c.value(), owners.device_a.value(),
            owners.device_b.value(), gpu_rows, k, n,
            grid_dim=geometry.payload.value().grid_blocks, block_dim=256,
        )
        ctx.enqueue_copy(
            dst_buf=owners.host_c.value(), src_buf=owners.device_c.value(),
        )
        var completed = ctx.create_event()
        ctx.stream().record_event(completed)
        # CPU and GPU write disjoint C rows. CPU work is independent of the
        # submitted GPU event; no host readback occurs until event completion.
        comptime width = simd_width_of[DType.float32]()
        var cpu_a_begin = selected.cpu_begin * k
        var cpu_c_begin = selected.cpu_begin * n
        cpu_mm_contiguous[width](
            result[cpu_c_begin:cpu_c_begin + cpu_rows * n],
            left[cpu_a_begin:cpu_a_begin + cpu_rows * k],
            right, cpu_rows, k, n,
        )
        completed.synchronize()
        for i in range(output_gpu_count):
            result[selected.gpu_begin * n + i] = owners.host_c.value()[i]
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            owners.quarantine()
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        # The host output may be partially written. On error the caller must
        # discard the output and must not infer all cells were overwritten.
        return TensorStatus.NATIVE_LAUNCH_FAILED
