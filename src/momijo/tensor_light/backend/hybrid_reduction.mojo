# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR BACKEND / HYBRID REDUCTIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Backend Availability
# Module:        momijo.tensor_light.backend.hybrid_reduction
# Source:        src/momijo/tensor_light/backend/hybrid_reduction.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines hybrid reductions in the backend availability layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `unsafe_execute_hybrid_reduce_rows_fp32`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Select or report supported execution backends explicitly.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as
#     `unsafe_execute_hybrid_reduce_rows_fp32`.
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

# A bounded, genuinely concurrent CPU/GPU mapping of one checked ExecutionPlan.
# Each backend owns disjoint output ROWS of an FP32 row reduction. This is NOT
# a second scheduler or owner type; the public input/output owners remain Tensor.
# The GPU kernel must be an EXTERNALLY TRUSTED, PRECOMPILED row-reduction
# DeviceExternalFunction. Identity metadata alone cannot authenticate its code.
# The expert path is deliberately unsafe until an offline artifact loader is
# implemented. No MAX source-kernel enqueue or JIT occurs in this module.
from std.collections import Span, Optional
from std.memory import forget_deinit
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from max.gpu.host.device_context import DeviceExternalFunction
from ..internal.status import TensorStatus
from ..execution.execution_plan import ExecutionPlan, ExecutionPlacement
from ..dispatch.specialization import ReduceRowsAOTArtifactIdentity
from ..kernels.launch import gpu_grid_1d
from ..kernels.cpu.reduction import cpu_reduce_row_ordered_fp32
from ..alias.overlap import host_output_disjoint_from_input


struct _HybridReduceRowOwners(Movable):
    # Every successful asynchronous allocation is immediately placed in one
    # move-only owner. This covers failures at any of the four create calls.
    var host_input: Optional[HostBuffer[DType.float32]]
    var host_output: Optional[HostBuffer[DType.float32]]
    var device_input: Optional[DeviceBuffer[DType.float32]]
    var device_output: Optional[DeviceBuffer[DType.float32]]
    var context_owner: Optional[DeviceContext]
    var quiescence_unproven: Bool

    def __init__(out self):
        self.host_input = Optional[HostBuffer[DType.float32]](None)
        self.host_output = Optional[HostBuffer[DType.float32]](None)
        self.device_input = Optional[DeviceBuffer[DType.float32]](None)
        self.device_output = Optional[DeviceBuffer[DType.float32]](None)
        self.context_owner = Optional[DeviceContext](None)
        self.quiescence_unproven = False

    def acquire(mut self, ctx: DeviceContext, input_count: Int, output_count: Int) raises:
        # Keep the issuing context alive BEFORE even the first async create.
        self.context_owner = Optional[DeviceContext](ctx.copy())
        self.host_input = Optional[HostBuffer[DType.float32]](
            ctx.enqueue_create_host_buffer[DType.float32](input_count)
        )
        self.host_output = Optional[HostBuffer[DType.float32]](
            ctx.enqueue_create_host_buffer[DType.float32](output_count)
        )
        self.device_input = Optional[DeviceBuffer[DType.float32]](
            ctx.enqueue_create_buffer[DType.float32](input_count)
        )
        self.device_output = Optional[DeviceBuffer[DType.float32]](
            ctx.enqueue_create_buffer[DType.float32](output_count)
        )

    def quarantine(mut self):
        self.quiescence_unproven = True

    def __deinit__(deinit self):
        if self.quiescence_unproven:
            if self.host_input:
                var input_owner = self.host_input.take()
                forget_deinit(input_owner)
            if self.host_output:
                var output_owner = self.host_output.take()
                forget_deinit(output_owner)
            if self.device_input:
                var device_input_owner = self.device_input.take()
                forget_deinit(device_input_owner)
            if self.device_output:
                var device_output_owner = self.device_output.take()
                forget_deinit(device_output_owner)
            if self.context_owner:
                var context = self.context_owner.take()
                forget_deinit(context)
        else:
            # The successful caller has joined the final D2H event; on a
            # recoverable error it has joined the issuing stream. Release
            # all native allocations BEFORE dropping their context owner.
            self.host_input = Optional[HostBuffer[DType.float32]](None)
            self.host_output = Optional[HostBuffer[DType.float32]](None)
            self.device_input = Optional[DeviceBuffer[DType.float32]](None)
            self.device_output = Optional[DeviceBuffer[DType.float32]](None)
            # Make the context release order explicit after native owners.
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


def unsafe_execute_hybrid_reduce_rows_fp32[
    operation: Int, static_gpu_row_capacity: Int,
    input_origin: Origin[mut=False], output_origin: Origin[mut=True],
](
    ctx: DeviceContext,
    precompiled_reduce_rows: DeviceExternalFunction,
    identity: ReduceRowsAOTArtifactIdentity,
    target_arch: Int,
    plan: ExecutionPlan,
    columns: Int,
    source: Span[mut=False, Float32, input_origin],
    mut output: Span[mut=True, Float32, output_origin],
) -> TensorStatus:
    # Narrow CPU/GPU SPLIT row reduction: C-order input [rows,columns],
    # C-order output [rows]. CPU/GPU partitions are DISJOINT output rows.
    # The untrusted native artifact is the sole unsafe part of this mapping;
    # the caller must authenticate its symbol, ABI and target externally.
    comptime assert operation >= 0 and operation <= 4, "FP32 row reduction opcode"
    comptime assert static_gpu_row_capacity > 0, "positive AOT staging rows"
    var checked = plan.checked_host_reduce_rows_split_snapshot(columns)
    if not checked.is_ok():
        return checked.status
    var selected = checked.payload.value().copy()
    if (selected.placement != materialize[ExecutionPlacement.HYBRID_SPLIT]()
        and selected.placement != materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]()):
        return TensorStatus.DEVICE_MISMATCH
    if selected.output_device != 0:
        return TensorStatus.DEVICE_MISMATCH
    var rows = selected.total_items
    if rows <= 1 or columns <= 0:
        return TensorStatus.INVALID_SHAPE
    if rows > Int.MAX // 4 or columns > (Int.MAX // 4) // rows:
        return TensorStatus.OVERFLOW
    if len(source) != rows * columns or len(output) != rows:
        return TensorStatus.SIZE_MISMATCH
    # In-place reduction into a source row could overwrite data pending H2D.
    # Reject even exact-address aliases until a joined snapshot ABI exists.
    if not host_output_disjoint_from_input(output, source):
        return TensorStatus.ALIAS_UNKNOWN
    var gpu_rows = selected.gpu_end - selected.gpu_begin
    var cpu_rows = selected.cpu_end - selected.cpu_begin
    if gpu_rows <= 0 or cpu_rows <= 0:
        return TensorStatus.HYBRID_PARTITION_INVALID
    if gpu_rows > static_gpu_row_capacity:
        return TensorStatus.SIZE_MISMATCH
    var input_gpu_count = gpu_rows * columns
    # The canonical checked_host_reduce_rows_split_snapshot() has already
    # proved the full input/output transport byte budget without overflow.
    if target_arch <= 0:
        return TensorStatus.DEVICE_MISMATCH
    var bound = identity.validate_for_op(
        operation, selected.aot_variant, target_arch,
    )
    if bound != TensorStatus.OK:
        return bound
    var launch = gpu_grid_1d(gpu_rows, 256)
    if not launch.is_ok():
        return launch.status
    var owners = _HybridReduceRowOwners()
    try:
        owners.acquire(ctx, input_gpu_count, gpu_rows)
        # MAX staging allocations themselves are asynchronous. Join their
        # creation before writing pinned host input for the first H2D copy.
        var buffers_ready = ctx.create_event()
        ctx.stream().record_event(buffers_ready)
        buffers_ready.synchronize()
        for i in range(input_gpu_count):
            owners.host_input.value()[i] = source[
                selected.gpu_begin * columns + i
            ]
        ctx.enqueue_copy(
            dst_buf=owners.device_input.value(),
            src_buf=owners.host_input.value(),
        )
        ctx.enqueue_function(
            precompiled_reduce_rows,
            owners.device_output.value(), owners.device_input.value(),
            gpu_rows, columns,
            grid_dim=launch.payload.value().grid_blocks, block_dim=256,
        )
        ctx.enqueue_copy(
            dst_buf=owners.host_output.value(),
            src_buf=owners.device_output.value(),
        )
        var ready = ctx.create_event()
        ctx.stream().record_event(ready)
        # GPU has been enqueued, so the CPU can reduce its distinct rows
        # concurrently. Output ranges are disjoint by the checked plan.
        for row in range(selected.cpu_begin, selected.cpu_end):
            output[row] = cpu_reduce_row_ordered_fp32[operation](
                source, row * columns, columns,
            )
        ready.synchronize()
        # No native allocation can be destroyed before this successful join.
        for i in range(gpu_rows):
            output[selected.gpu_begin + i] = owners.host_output.value()[i]
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            owners.quarantine()
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        # GPU was drained, but an intermediate output may have been written;
        # the caller MUST inspect status and discard this result on error.
        return TensorStatus.NATIVE_LAUNCH_FAILED
