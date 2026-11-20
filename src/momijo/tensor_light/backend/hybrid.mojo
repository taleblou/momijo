# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR BACKEND / HYBRID BACKEND
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Backend Availability
# Module:        momijo.tensor_light.backend.hybrid
# Source:        src/momijo/tensor_light/backend/hybrid.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines hybrid backend in the backend availability layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `execute_cpu_binary_host_fp32`,
#   `execute_cpu_add_host_fp32`, `execute_host_binary_fp32`,
#   `unsafe_execute_host_binary_fp32`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Select or report supported execution backends explicitly.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as
#     `execute_cpu_binary_host_fp32`, `execute_cpu_add_host_fp32`,
#     `execute_host_binary_fp32`, `unsafe_execute_host_binary_fp32`,
#     `hybrid_add_host_fp32` and related symbols.
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

# Real CPU+GPU overlap for HOST-RESIDENT 1-D FP32
# elementwise binary mapping, with a static AOT staging capacity and a validated split.
# This is one target lowering of ExecutionPlan, not a second tensor engine.
# GPU is enqueued BEFORE the CPU partition; the final synchronize is a join.
from std.collections import Span, Optional
from std.collections.list import List
from max.gpu.host import DeviceContext, HostBuffer, DeviceBuffer
from max.gpu.host.device_context import DeviceExternalFunction
from std.memory import forget_deinit
from ..execution.execution_plan import ExecutionPlan, ExecutionPlacement
from ..execution.launch_policy import build_host_binary_split_schedule, build_host_binary_pipeline_window_schedule
from ..execution.static_dataflow import validate_host_binary_native_schedule_agreement, validate_host_binary_pipeline_window_agreement, checked_host_binary_pipeline_window
from ..internal.status import TensorStatus
from ..dispatch.specialization import BinaryAOTArtifactIdentity, UnaryAOTArtifactIdentity
from ..kernels.launch import gpu_grid_1d
from ..alias.overlap import host_output_disjoint_from_input, host_output_exactly_matches_input
from ..transfer.device_transfer import validate_binary_fp32_device_owners, validate_binary_fp32_pipeline_device_owners
from ..kernels.cpu.unary import cpu_unary_contiguous_fp32
from ..kernels.cpu.elementwise import cpu_binary_contiguous_checked
from ..dtype.fp32 import fp32_maximum, fp32_minimum
from std.sys.info import simd_width_of

# All six real allocation owners of ONE binary GPU window are tracked as
# move-only optional resources. A native creation/launch may fail after SOME
# allocations have been enqueued, so declaring six locals inside a try block
# does not safely handle partially acquired owners. The optional fields keep
# each successfully returned owner live across recovery and unwind.
#
# This is a *lifetime* fence, not proof that the kernel ran successfully.
# On a fatal failure of both event and stream quiescence, intentionally retain
# every acquired native owner until explicit device-context recovery. Never
# put these allocations into a pool or release a pending DMA host buffer.
struct _HybridBinaryWindowOwners(Movable):
    var host_a: Optional[HostBuffer[DType.float32]]
    var host_b: Optional[HostBuffer[DType.float32]]
    var host_out: Optional[HostBuffer[DType.float32]]
    var device_a: Optional[DeviceBuffer[DType.float32]]
    var device_b: Optional[DeviceBuffer[DType.float32]]
    var device_out: Optional[DeviceBuffer[DType.float32]]
    # Copy of the exact native context that issued this window's work.
    # Borrowing the caller's ctx cannot protect queued DMA after return.
    var context_owner: Optional[DeviceContext]
    var quiescence_unproven: Bool

    def __init__(out self):
        self.host_a = Optional[HostBuffer[DType.float32]](None)
        self.host_b = Optional[HostBuffer[DType.float32]](None)
        self.host_out = Optional[HostBuffer[DType.float32]](None)
        self.device_a = Optional[DeviceBuffer[DType.float32]](None)
        self.device_b = Optional[DeviceBuffer[DType.float32]](None)
        self.device_out = Optional[DeviceBuffer[DType.float32]](None)
        self.context_owner = Optional[DeviceContext](None)
        self.quiescence_unproven = False

    def acquire(mut self, ctx: DeviceContext, count: Int) raises:
        # Caller proves 0 < count <= Int.MAX//4 before any allocation.
        # Store the real context reference BEFORE enqueueing the first native
        # allocation: an allocation failure can still leave pending GPU work.
        # The enclosing handler MUST synchronize or quarantine every owner.
        if not self.context_owner:
            self.context_owner = Optional[DeviceContext](ctx.copy())
        self.host_a = Optional[HostBuffer[DType.float32]](
            ctx.enqueue_create_host_buffer[DType.float32](count)
        )
        self.host_b = Optional[HostBuffer[DType.float32]](
            ctx.enqueue_create_host_buffer[DType.float32](count)
        )
        self.host_out = Optional[HostBuffer[DType.float32]](
            ctx.enqueue_create_host_buffer[DType.float32](count)
        )
        self.device_a = Optional[DeviceBuffer[DType.float32]](
            ctx.enqueue_create_buffer[DType.float32](count)
        )
        self.device_b = Optional[DeviceBuffer[DType.float32]](
            ctx.enqueue_create_buffer[DType.float32](count)
        )
        self.device_out = Optional[DeviceBuffer[DType.float32]](
            ctx.enqueue_create_buffer[DType.float32](count)
        )

    def quarantine(mut self):
        self.quiescence_unproven = True

    def release_after_join(mut self) -> TensorStatus:
        # Drop old native allocations BEFORE creating an exact-size tail.
        # Assigning a new Optional during acquire() may allocate the new
        # buffer before dropping the previous one and exceed a one-slot cap.
        # The caller must have joined the previous D2H completion event.
        if self.quiescence_unproven:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.context_owner:
            # Without the issuing context we cannot prove that asynchronous
            # deallocations have been drained. Keep any surviving owners.
            self.quiescence_unproven = True
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        self.host_a = Optional[HostBuffer[DType.float32]](None)
        self.host_b = Optional[HostBuffer[DType.float32]](None)
        self.host_out = Optional[HostBuffer[DType.float32]](None)
        self.device_a = Optional[DeviceBuffer[DType.float32]](None)
        self.device_b = Optional[DeviceBuffer[DType.float32]](None)
        self.device_out = Optional[DeviceBuffer[DType.float32]](None)
        # MAX HostBuffer/DeviceBuffer destructors may enqueue an ASYNC free.
        # Joining the prior D2H event alone does not join those later frees.
        # Drain the issuing context before trying a smaller exact-size tail
        # allocation, retaining its owner while the release is in flight.
        # This is a one-time tail transition, not a steady-state window sync.
        try:
            self.context_owner.value().synchronize()
        except:
            self.quiescence_unproven = True
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        self.context_owner = Optional[DeviceContext](None)
        return TensorStatus.OK

    def __deinit__(deinit self):
        # Unsuccessful stream synchronization leaves even an *allocation*
        # possibly in flight. Optional ownership prevents reading or moving an
        # uninitialized local and covers failure at every one of six creates.
        if self.quiescence_unproven:
            if self.host_a:
                var owned = self.host_a.take()
                forget_deinit(owned)
            if self.host_b:
                var owned = self.host_b.take()
                forget_deinit(owned)
            if self.host_out:
                var owned = self.host_out.take()
                forget_deinit(owned)
            if self.device_a:
                var owned = self.device_a.take()
                forget_deinit(owned)
            if self.device_b:
                var owned = self.device_b.take()
                forget_deinit(owned)
            if self.device_out:
                var owned = self.device_out.take()
                forget_deinit(owned)
            # All native resources AND their DeviceContext must outlive an
            # operation whose event/stream join was not established.
            if self.context_owner:
                var pending_context = self.context_owner.take()
                forget_deinit(pending_context)
        else:
            # A successful event/stream join has made the staged allocations
            # safe to release. Explicitly destroy every host/device buffer
            # while the originating MAX context is STILL retained. In Mojo,
            # default field teardown uses ASAP lifetime semantics; merely
            # declaring context_owner after the buffers is not a lifetime
            # ordering proof.
            self.host_a = Optional[HostBuffer[DType.float32]](None)
            self.host_b = Optional[HostBuffer[DType.float32]](None)
            self.host_out = Optional[HostBuffer[DType.float32]](None)
            self.device_a = Optional[DeviceBuffer[DType.float32]](None)
            self.device_b = Optional[DeviceBuffer[DType.float32]](None)
            self.device_out = Optional[DeviceBuffer[DType.float32]](None)
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


def _hybrid_binary_host_fp32[
    static_gpu_capacity: Int,
    operation: Int,
    a: Origin[mut=False],
    b: Origin[mut=False],
    out_origin: Origin[mut=True],
](
    ctx: DeviceContext,
    precompiled_binary: DeviceExternalFunction,
    plan: ExecutionPlan,
    left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) raises -> TensorStatus:
    comptime assert static_gpu_capacity > 0, "GPU staging capacity must be positive"
    comptime assert operation >= 0 and operation <= 5, "FP32 opcode must be 0..5"
    var plan_proof = plan.checked_snapshot()
    if not plan_proof.is_ok():
        return plan_proof.status
    var checked_plan = plan_proof.payload.value().copy()
    if checked_plan.placement != materialize[ExecutionPlacement.HYBRID_SPLIT]() and checked_plan.placement != materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]():
        return TensorStatus.DEVICE_MISMATCH
    # This lowering materializes one host output, never a device tensor.
    if checked_plan.output_device != 0:
        return TensorStatus.DEVICE_MISMATCH
    if checked_plan.total_items != len(left) or len(left) != len(right) or len(left) != len(result):
        return TensorStatus.SIZE_MISMATCH
    # ExecutionPlan.checked proves both partitions are disjoint, contiguous
    # and cover the complete output. Split permits both partition orders;
    # CPU assist uses GPU-first only. Every access uses explicit offsets.
    var cpu_first = checked_plan.cpu_begin == 0 and checked_plan.cpu_end == checked_plan.gpu_begin and checked_plan.gpu_end == checked_plan.total_items
    var gpu_first = checked_plan.gpu_begin == 0 and checked_plan.gpu_end == checked_plan.cpu_begin and checked_plan.cpu_end == checked_plan.total_items
    if checked_plan.placement == materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]():
        if not gpu_first:
            return TensorStatus.INVALID_LAYOUT
    elif not cpu_first and not gpu_first:
        return TensorStatus.INVALID_LAYOUT
    # External spans may overlap. Only an input whose alias relation
    # is unsafe is snapshotted before writes can modify its unread elements.
    # Staging buffers and device buffers are independently owned allocations.
    var gpu_count = checked_plan.gpu_end - checked_plan.gpu_begin
    if gpu_count <= 0:
        return TensorStatus.SIZE_MISMATCH
    # A host-resident FP32 binary operation stages TWO input slices to device and
    # returns ONE FP32 output slice: at least 3 * 4 bytes per GPU element.
    if gpu_count > Int.MAX // 12:
        return TensorStatus.OVERFLOW
    if checked_plan.transfer_bytes < gpu_count * 12:
        return TensorStatus.SIZE_MISMATCH
    # Every GPU window is at most the fixed staging capacity. Prove even the
    # largest window fits this target's 1-D grid BEFORE creating async owners.
    var grid_proof = gpu_grid_1d(min(gpu_count, static_gpu_capacity), 256)
    if not grid_proof.is_ok():
        return grid_proof.status
    # The AOT plan carries an allocation-aware dependency graph. Reject
    # inconsistent hazards before any asynchronous GPU work is enqueued.
    var schedule = build_host_binary_split_schedule(checked_plan)
    if not schedule.is_ok():
        return schedule.status
    # Stage DAG and declared physical bounds are verified together before
    # launch. Native allocation ownership must be separately established.
    var bound_check = schedule.payload.value().validate_host_result_launch(
        checked_plan.total_items, 0, checked_plan.total_items,
    )
    if bound_check != TensorStatus.OK:
        return bound_check
    # The native stage DAG and the canonical UASDEA graph now share one
    # checked snapshot. A well-formed native schedule alone cannot authorize
    # a hybrid mapping whose output/partitions/transfers disagree with the IR.
    var graph_agreement = validate_host_binary_native_schedule_agreement(
        schedule.payload.value(), checked_plan, operation,
    )
    if graph_agreement != TensorStatus.OK:
        return graph_agreement

    # Exact host address proofs allow a zero-snapshot path for independent
    # spans and exact element-wise in-place aliases. Shifted/unknown aliases
    # still snapshot ALL chunks before the first output write; skipping a
    # cross-chunk snapshot for those aliases would corrupt unread elements.
    var direct_left = host_output_disjoint_from_input(result, left)
    if not direct_left:
        direct_left = host_output_exactly_matches_input(result, left)
    var direct_right = host_output_disjoint_from_input(result, right)
    if not direct_right:
        direct_right = host_output_exactly_matches_input(result, right)
    var direct_inputs = direct_left and direct_right
    if gpu_count > static_gpu_capacity:
        var cpu_count = checked_plan.cpu_end - checked_plan.cpu_begin
        # Snapshot only the operand that could be clobbered by later output
        # writes; a disjoint or precisely in-place operand needs no full copy.
        var saved_left_count_gpu = 0
        var saved_right_count_gpu = 0
        var saved_left_count_cpu = 0
        var saved_right_count_cpu = 0
        if not direct_left:
            saved_left_count_gpu = gpu_count
            saved_left_count_cpu = cpu_count
        if not direct_right:
            saved_right_count_gpu = gpu_count
            saved_right_count_cpu = cpu_count
        var gpu_left = List[Float32](length=saved_left_count_gpu, fill=0.0)
        var gpu_right = List[Float32](length=saved_right_count_gpu, fill=0.0)
        var cpu_left = List[Float32](length=saved_left_count_cpu, fill=0.0)
        var cpu_right = List[Float32](length=saved_right_count_cpu, fill=0.0)
        if not direct_left:
            for i in range(gpu_count):
                gpu_left[i] = left[checked_plan.gpu_begin + i]
            for i in range(cpu_count):
                cpu_left[i] = left[checked_plan.cpu_begin + i]
        if not direct_right:
            for i in range(gpu_count):
                gpu_right[i] = right[checked_plan.gpu_begin + i]
            for i in range(cpu_count):
                cpu_right[i] = right[checked_plan.cpu_begin + i]

        # Split CPU work across all GPU chunks so CPU compute can overlap
        # EVERY H2D/kernel/D2H group rather than only the first group.
        var chunk_count = gpu_count // static_gpu_capacity
        if gpu_count % static_gpu_capacity != 0:
            chunk_count += 1
        var cpu_per_chunk = cpu_count // chunk_count
        var cpu_extra_chunks = cpu_count % chunk_count
        var cpu_at = 0
        var gpu_at = 0
        var chunk_index = 0
        # Reuse the SIX owned native buffers for every full-capacity window.
        # This default-stream loop joins each D2H completion event before
        # re-filling the pinned inputs. Only the final short window requires
        # a different flat MAX copy capacity and therefore a new allocation.
        # Keep owners OUTSIDE the loop so repeated full windows incur zero
        # native alloc/registration calls after the first initialization.
        var window_owners = _HybridBinaryWindowOwners()
        while gpu_at < gpu_count:
            var chunk = min(static_gpu_capacity, gpu_count - gpu_at)
            try:
                if chunk_index == 0 or chunk != static_gpu_capacity:
                    # Reuse full windows. Before an exact-sized short tail,
                    # discard completed full owners to honor the planner's
                    # one-slot memory reservation even transiently.
                    if chunk_index > 0:
                        var released = window_owners.release_after_join()
                        if released != TensorStatus.OK:
                            return released
                    window_owners.acquire(ctx, chunk)
                    var allocation_ready = ctx.create_event()
                    ctx.stream().record_event(allocation_ready)
                    allocation_ready.synchronize()
                var native_owners = validate_binary_fp32_device_owners(
                    window_owners.device_out.value(), window_owners.device_a.value(), window_owners.device_b.value(), chunk,
                )
                if native_owners != TensorStatus.OK:
                    return native_owners
                for j in range(chunk):
                    if direct_left:
                        window_owners.host_a.value()[j] = left[checked_plan.gpu_begin + gpu_at + j]
                    else:
                        window_owners.host_a.value()[j] = gpu_left[gpu_at + j]
                    if direct_right:
                        window_owners.host_b.value()[j] = right[checked_plan.gpu_begin + gpu_at + j]
                    else:
                        window_owners.host_b.value()[j] = gpu_right[gpu_at + j]
                ctx.enqueue_copy(dst_buf=window_owners.device_a.value(), src_buf=window_owners.host_a.value())
                ctx.enqueue_copy(dst_buf=window_owners.device_b.value(), src_buf=window_owners.host_b.value())
                var blocks = chunk // 256
                if chunk % 256 != 0:
                    blocks += 1
                ctx.enqueue_function(
                    precompiled_binary, window_owners.device_out.value(), window_owners.device_a.value(), window_owners.device_b.value(), chunk,
                    grid_dim=blocks, block_dim=256
                )
                ctx.enqueue_copy(dst_buf=window_owners.host_out.value(), src_buf=window_owners.device_out.value())
                var ready = ctx.create_event()
                ctx.stream().record_event(ready)
                # Balance CPU work across GPU chunks. Any alias-unsafe input
                # was snapshotted before output modifications. The quotient/
                # remainder distribution avoids overflow-prone index products.
                var cpu_this_chunk = cpu_per_chunk
                if chunk_index < cpu_extra_chunks:
                    cpu_this_chunk += 1
                if direct_inputs:
                    var start = checked_plan.cpu_begin + cpu_at
                    var finish = start + cpu_this_chunk
                    var cpu_out = result[start:finish]
                    var cpu_a = left[start:finish]
                    var cpu_b = right[start:finish]
                    comptime width = simd_width_of[DType.float32]()
                    var cpu_status = cpu_binary_contiguous_checked[
                        operation, width,
                    ](cpu_out, cpu_a, cpu_b)
                    if cpu_status != TensorStatus.OK:
                        # GPU work has already been queued. Keep all native owners
                        # alive until this issuing stream is proven quiescent.
                        try:
                            ctx.stream().synchronize()
                        except:
                            window_owners.quarantine()
                            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
                        return cpu_status
                else:
                    for j in range(cpu_this_chunk):
                        var cpu_index = checked_plan.cpu_begin + cpu_at + j
                        var x = Float32(0)
                        var y = Float32(0)
                        if direct_left:
                            x = left[cpu_index]
                        else:
                            x = cpu_left[cpu_at + j]
                        if direct_right:
                            y = right[cpu_index]
                        else:
                            y = cpu_right[cpu_at + j]
                        comptime if operation == 0:
                            result[checked_plan.cpu_begin + cpu_at + j] = x + y
                        elif operation == 1:
                            result[checked_plan.cpu_begin + cpu_at + j] = x - y
                        elif operation == 2:
                            result[checked_plan.cpu_begin + cpu_at + j] = x * y
                        elif operation == 3:
                            result[checked_plan.cpu_begin + cpu_at + j] = x / y
                        elif operation == 4:
                            result[checked_plan.cpu_begin + cpu_at + j] = fp32_maximum(x, y)
                        else:
                            result[checked_plan.cpu_begin + cpu_at + j] = fp32_minimum(x, y)
                cpu_at += cpu_this_chunk
                ready.synchronize()
                # The event is complete before any local device or staging owner
                # can go out of scope or be reused for a subsequent chunk.
                for j in range(chunk):
                    result[checked_plan.gpu_begin + gpu_at + j] = window_owners.host_out.value()[j]
                gpu_at += chunk
                chunk_index += 1
            except:
                # An error after a native enqueue must not let borrowed
                # owners be recycled before the queued work is drained.
                try:
                    ctx.stream().synchronize()
                except:
                    window_owners.quarantine()
                    return TensorStatus.GPU_QUIESCENCE_UNPROVEN
                return TensorStatus.NATIVE_LAUNCH_FAILED
        return TensorStatus.OK

    # Use exact logical transfer size. The compile-time capacity is only an
    # upper bound; transferring its full padded size wastes PCIe bandwidth.
    var window_owners = _HybridBinaryWindowOwners()
    try:
        window_owners.acquire(ctx, gpu_count)
        # MAX host allocations are asynchronous. This barrier occurs only during
        # staging-buffer creation; pooled steady-state lowering can remove it.
        var allocation_ready = ctx.create_event()
        ctx.stream().record_event(allocation_ready)
        allocation_ready.synchronize()
        for i in range(gpu_count):
            window_owners.host_a.value()[i] = left[checked_plan.gpu_begin + i]
            window_owners.host_b.value()[i] = right[checked_plan.gpu_begin + i]
        ctx.enqueue_copy(dst_buf=window_owners.device_a.value(), src_buf=window_owners.host_a.value())
        ctx.enqueue_copy(dst_buf=window_owners.device_b.value(), src_buf=window_owners.host_b.value())
        var blocks = gpu_count // 256
        if gpu_count % 256 != 0:
            blocks += 1
        # The offline artifact must implement THIS operation's opcode, FP32 ABI
        # (dst_ptr, left_ptr, right_ptr, count), and selected GPU architecture.
        # The caller must bind checked_plan.aot_variant to the matching artifact.
        # Loading the artifact is an explicit setup step before this function.
        ctx.enqueue_function(
            precompiled_binary, window_owners.device_out.value(), window_owners.device_a.value(), window_owners.device_b.value(), gpu_count,
            grid_dim=blocks, block_dim=256
        )
        ctx.enqueue_copy(dst_buf=window_owners.host_out.value(), src_buf=window_owners.device_out.value())
        # A narrow event is sufficient for the D2H result dependency. Do not
        # synchronize unrelated work on the device during the hybrid join.
        var gpu_result_ready = ctx.create_event()
        ctx.stream().record_event(gpu_result_ready)

        # GPU input staging above is already complete on the host. When the
        # output is disjoint/exactly aligned with both inputs, CPU lanes may
        # write directly through the checked SIMD fast path. All other alias
        # patterns retain a complete CPU-input snapshot before any CPU write.
        var cpu_count = checked_plan.cpu_end - checked_plan.cpu_begin
        if direct_inputs:
            var c0 = checked_plan.cpu_begin
            var c1 = checked_plan.cpu_end
            var cpu_dst = result[c0:c1]
            var cpu_a = left[c0:c1]
            var cpu_b = right[c0:c1]
            comptime width = simd_width_of[DType.float32]()
            var cpu_status = cpu_binary_contiguous_checked[
                operation, width,
            ](cpu_dst, cpu_a, cpu_b)
            if cpu_status != TensorStatus.OK:
                # GPU work has already been queued. Keep all native owners
                # alive until this issuing stream is proven quiescent.
                try:
                    ctx.stream().synchronize()
                except:
                    window_owners.quarantine()
                    return TensorStatus.GPU_QUIESCENCE_UNPROVEN
                return cpu_status
        else:
            var left_count = 0
            var right_count = 0
            if not direct_left:
                left_count = cpu_count
            if not direct_right:
                right_count = cpu_count
            var cpu_left = List[Float32](length=left_count, fill=0.0)
            var cpu_right = List[Float32](length=right_count, fill=0.0)
            if not direct_left:
                for i in range(cpu_count):
                    cpu_left[i] = left[checked_plan.cpu_begin + i]
            if not direct_right:
                for i in range(cpu_count):
                    cpu_right[i] = right[checked_plan.cpu_begin + i]
            for i in range(cpu_count):
                var current = checked_plan.cpu_begin + i
                var x = Float32(0)
                var y = Float32(0)
                if direct_left:
                    x = left[current]
                else:
                    x = cpu_left[i]
                if direct_right:
                    y = right[current]
                else:
                    y = cpu_right[i]
                comptime if operation == 0:
                    result[checked_plan.cpu_begin + i] = x + y
                elif operation == 1:
                    result[checked_plan.cpu_begin + i] = x - y
                elif operation == 2:
                    result[checked_plan.cpu_begin + i] = x * y
                elif operation == 3:
                    result[checked_plan.cpu_begin + i] = x / y
                elif operation == 4:
                    result[checked_plan.cpu_begin + i] = fp32_maximum(x, y)
                else:
                    result[checked_plan.cpu_begin + i] = fp32_minimum(x, y)
        # The CPU partition completed synchronously; only the result-copy event
        # must complete before the staged GPU output is accessed or released.
        gpu_result_ready.synchronize()
        for i in range(gpu_count):
            result[checked_plan.gpu_begin + i] = window_owners.host_out.value()[i]
        return TensorStatus.OK
    except:
        # An error after a native enqueue must not let borrowed
        # owners be recycled before the queued work is drained.
        try:
            ctx.stream().synchronize()
        except:
            window_owners.quarantine()
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Two-slot, host-result, offline-AOT FP32 binary pipeline. This is a
# deliberately narrow REAL execution lowering for HYBRID_PIPELINE. The CPU
# prepares the next pinned input slot while GPU work for the current slot
# remains enqueued. Two independent staging/device owners remain live until
# their event-fenced D2H results are joined to the output. No heap allocation,
# materialization, benchmark, or dynamic opcode dispatch occurs IN the kernel.
# Shifted/unknown aliases are snapshotted before writes; proven disjoint or
# exact in-place ranges stream from inputs without full-size host temporaries.
def _pipeline_binary_host_fp32[
    capacity: Int, operation: Int,
    a: Origin[mut=False], b: Origin[mut=False], out_origin: Origin[mut=True],
](
    ctx: DeviceContext,
    precompiled_binary: DeviceExternalFunction,
    plan: ExecutionPlan,
    left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) raises -> TensorStatus:
    comptime assert capacity > 0, "pipeline capacity must be positive"
    comptime assert capacity <= Int.MAX // 2, "two-window capacity overflows Int"
    comptime assert operation >= 0 and operation <= 5, "invalid FP32 opcode"
    var plan_proof = plan.checked_snapshot()
    if not plan_proof.is_ok():
        return plan_proof.status
    var checked_plan = plan_proof.payload.value().copy()
    if checked_plan.placement != materialize[ExecutionPlacement.HYBRID_PIPELINE]():
        return TensorStatus.DEVICE_MISMATCH
    if checked_plan.total_items != len(left) or len(left) != len(right) or len(result) != len(left):
        return TensorStatus.SIZE_MISMATCH
    var schedule = build_host_binary_pipeline_window_schedule(checked_plan, capacity)
    if not schedule.is_ok():
        return schedule.status
    # Stage DAG and declared physical bounds are verified together before
    # launch. Native allocation ownership must be separately established.
    var bound_check = schedule.payload.value().validate_host_result_launch(
        checked_plan.total_items, 0, 2 * capacity,
    )
    if bound_check != TensorStatus.OK:
        return bound_check
    # The two-window native template must be a valid prefix of the SAME
    # canonical UASDEA pipeline model. This does not replace stream/event
    # fences when the template is replayed for subsequent full/tail chunks.
    var graph_agreement = validate_host_binary_pipeline_window_agreement(
        schedule.payload.value(), checked_plan, 2 * capacity, operation,
    )
    if graph_agreement != TensorStatus.OK:
        return graph_agreement
    # Both full slots and the smaller tail share one bounded launch ABI.
    var grid_proof = gpu_grid_1d(capacity, 256)
    if not grid_proof.is_ok():
        return grid_proof.status
    var count = checked_plan.total_items
    var batches = count // capacity
    var tail_count = count % capacity
    # The checked two-slot schedule MUST contain two full windows. This is
    # redundant with the schedule builder by design: do not ever read from a
    # slot whose completion event was never recorded.
    if batches < 2:
        return TensorStatus.NOT_REPRESENTABLE
    # The two full slots are never used for a partial window. A final short
    # window owns exactly-sized host/device buffers, avoiding padded copies.
    # Real and disjoint storage identities for the first two AOT windows.
    var direct_left = host_output_disjoint_from_input(result, left)
    if not direct_left:
        direct_left = host_output_exactly_matches_input(result, left)
    var direct_right = host_output_disjoint_from_input(result, right)
    if not direct_right:
        direct_right = host_output_exactly_matches_input(result, right)
    var direct_inputs = direct_left and direct_right
    var saved_left_count = 0
    var saved_right_count = 0
    if not direct_left:
        saved_left_count = count
    if not direct_right:
        saved_right_count = count
    var saved_left = List[Float32](length=saved_left_count, fill=0.0)
    var saved_right = List[Float32](length=saved_right_count, fill=0.0)
    if not direct_left:
        for i in range(count):
            saved_left[i] = left[i]
    if not direct_right:
        for i in range(count):
            saved_right[i] = right[i]

    # Real three-lane pipeline. MAX DeviceContext.select_stream() returns a
    # view on the SAME device/context and memory pool, with its own enqueue
    # stream. Two copy streams allow H2D(N+1) to overlap compute(N) while
    # D2H(N-1) is pending. The host retains every owner until its D2H fence.
    # A backend without independent streams transparently uses its original
    # single stream. All event dependencies below remain correct in both cases.
    var upload_ctx = ctx.copy()
    var download_ctx = ctx.copy()
    try:
        upload_ctx = ctx.select_stream(1)
    except:
        pass
    try:
        download_ctx = ctx.select_stream(2)
    except:
        pass

    # Exactly-sized fixed chunks (including the final one) avoid padded
    # H2D/D2H transfers and any uninitialized input tail exposure.
    var slot0 = _HybridBinaryWindowOwners()
    var slot1 = _HybridBinaryWindowOwners()
    var tail_slot = _HybridBinaryWindowOwners()
    try:
        slot0.acquire(ctx, capacity)
        slot1.acquire(ctx, capacity)
        var allocation_ready = ctx.create_event()
        ctx.stream().record_event(allocation_ready)
        allocation_ready.synchronize()
        # The two windows can be in flight at the same time. Therefore the
        # six device allocations must be mutually disjoint BETWEEN windows,
        # not merely result-versus-input disjoint within each window.
        var owners = validate_binary_fp32_pipeline_device_owners(
            slot0.device_out.value(), slot0.device_a.value(), slot0.device_b.value(), slot1.device_out.value(), slot1.device_a.value(), slot1.device_b.value(), capacity,
        )
        if owners != TensorStatus.OK:
            return owners
        var ready0 = ctx.create_event()
        var ready1 = ctx.create_event()
        var uploaded0 = ctx.create_event()
        var uploaded1 = ctx.create_event()
        var computed0 = ctx.create_event()
        var computed1 = ctx.create_event()
        var blocks = capacity // 256
        if capacity % 256 != 0:
            blocks += 1
        for batch in range(batches):
            # Prove each replayed full window against the same immutable
            # plan, not just the first two windows of its AOT prefix DAG.
            var window = checked_host_binary_pipeline_window(
                checked_plan, capacity, batch,
            )
            if not window.is_ok():
                # Prior windows may already have enqueued asynchronous work.
                # Drain the stream before destructing any of their owners.
                upload_ctx.stream().synchronize()
                ctx.stream().synchronize()
                download_ctx.stream().synchronize()
                return window.status
            var region = window.payload.value().copy()
            if region.elements != capacity or region.tail:
                upload_ctx.stream().synchronize()
                ctx.stream().synchronize()
                download_ctx.stream().synchronize()
                return TensorStatus.INVALID_LAYOUT
            var offset = region.begin
            if region.slot == 0:
                if region.reuses_slot:
                    # All H2D, device compute and D2H for this slot must finish
                    # before host reads its result or overwrites the input slot.
                    ready0.synchronize()
                    var previous = (batch - 2) * capacity
                    for j in range(capacity):
                        result[previous + j] = slot0.host_out.value()[j]
                for j in range(capacity):
                    if direct_left:
                        slot0.host_a.value()[j] = left[offset + j]
                    else:
                        slot0.host_a.value()[j] = saved_left[offset + j]
                    if direct_right:
                        slot0.host_b.value()[j] = right[offset + j]
                    else:
                        slot0.host_b.value()[j] = saved_right[offset + j]
                upload_ctx.enqueue_copy(dst_buf=slot0.device_a.value(), src_buf=slot0.host_a.value())
                upload_ctx.enqueue_copy(dst_buf=slot0.device_b.value(), src_buf=slot0.host_b.value())
                upload_ctx.stream().record_event(uploaded0)
                ctx.stream().enqueue_wait_for(uploaded0)
                ctx.enqueue_function(
                    precompiled_binary, slot0.device_out.value(), slot0.device_a.value(), slot0.device_b.value(), capacity,
                    grid_dim=blocks, block_dim=256,
                )
                ctx.stream().record_event(computed0)
                download_ctx.stream().enqueue_wait_for(computed0)
                download_ctx.enqueue_copy(dst_buf=slot0.host_out.value(), src_buf=slot0.device_out.value())
                download_ctx.stream().record_event(ready0)
            else:
                if region.reuses_slot:
                    ready1.synchronize()
                    var previous = (batch - 2) * capacity
                    for j in range(capacity):
                        result[previous + j] = slot1.host_out.value()[j]
                for j in range(capacity):
                    if direct_left:
                        slot1.host_a.value()[j] = left[offset + j]
                    else:
                        slot1.host_a.value()[j] = saved_left[offset + j]
                    if direct_right:
                        slot1.host_b.value()[j] = right[offset + j]
                    else:
                        slot1.host_b.value()[j] = saved_right[offset + j]
                upload_ctx.enqueue_copy(dst_buf=slot1.device_a.value(), src_buf=slot1.host_a.value())
                upload_ctx.enqueue_copy(dst_buf=slot1.device_b.value(), src_buf=slot1.host_b.value())
                upload_ctx.stream().record_event(uploaded1)
                ctx.stream().enqueue_wait_for(uploaded1)
                ctx.enqueue_function(
                    precompiled_binary, slot1.device_out.value(), slot1.device_a.value(), slot1.device_b.value(), capacity,
                    grid_dim=blocks, block_dim=256,
                )
                ctx.stream().record_event(computed1)
                download_ctx.stream().enqueue_wait_for(computed1)
                download_ctx.enqueue_copy(dst_buf=slot1.host_out.value(), src_buf=slot1.device_out.value())
                download_ctx.stream().record_event(ready1)
        # The two most recently enqueued chunk owners remain live through their
        # narrow completion events. The host output is not exposed until both are
        # joined; no global device sync or untracked async owner escape is used.
        var last_even = batches - 1
        if last_even % 2 != 0:
            last_even -= 1
        var last_odd = batches - 1
        if last_odd % 2 == 0:
            last_odd -= 1
        ready0.synchronize()
        for j in range(capacity):
            result[last_even * capacity + j] = slot0.host_out.value()[j]
        ready1.synchronize()
        for j in range(capacity):
            result[last_odd * capacity + j] = slot1.host_out.value()[j]
        if tail_count > 0:
            # Complete all prior full windows before allocating the distinct tail
            # slot. Count/shape are bounded by the checked full-data plan.
            # No copy can include an uninitialized capacity-sized suffix.
            var tail_window = checked_host_binary_pipeline_window(
                checked_plan, capacity, batches,
            )
            if not tail_window.is_ok():
                return tail_window.status
            var tail_region = tail_window.payload.value().copy()
            if (not tail_region.tail or tail_region.slot != 2
                or tail_region.elements != tail_count):
                return TensorStatus.INVALID_LAYOUT
            var tail_offset = tail_region.begin
            tail_slot.acquire(ctx, tail_count)
            var tail_allocated = ctx.create_event()
            ctx.stream().record_event(tail_allocated)
            tail_allocated.synchronize()
            var tail_owners = validate_binary_fp32_device_owners(
                tail_slot.device_out.value(), tail_slot.device_a.value(), tail_slot.device_b.value(), tail_count,
            )
            if tail_owners != TensorStatus.OK:
                return tail_owners
            for j in range(tail_count):
                if direct_left:
                    tail_slot.host_a.value()[j] = left[tail_offset + j]
                else:
                    tail_slot.host_a.value()[j] = saved_left[tail_offset + j]
                if direct_right:
                    tail_slot.host_b.value()[j] = right[tail_offset + j]
                else:
                    tail_slot.host_b.value()[j] = saved_right[tail_offset + j]
            upload_ctx.enqueue_copy(dst_buf=tail_slot.device_a.value(), src_buf=tail_slot.host_a.value())
            upload_ctx.enqueue_copy(dst_buf=tail_slot.device_b.value(), src_buf=tail_slot.host_b.value())
            var tail_upload_ready = ctx.create_event()
            upload_ctx.stream().record_event(tail_upload_ready)
            ctx.stream().enqueue_wait_for(tail_upload_ready)
            var tail_blocks = tail_count // 256
            if tail_count % 256 != 0:
                tail_blocks += 1
            ctx.enqueue_function(
                precompiled_binary, tail_slot.device_out.value(), tail_slot.device_a.value(), tail_slot.device_b.value(),
                tail_count, grid_dim=tail_blocks, block_dim=256,
            )
            var tail_compute_ready = ctx.create_event()
            ctx.stream().record_event(tail_compute_ready)
            download_ctx.stream().enqueue_wait_for(tail_compute_ready)
            download_ctx.enqueue_copy(dst_buf=tail_slot.host_out.value(), src_buf=tail_slot.device_out.value())
            var tail_ready = ctx.create_event()
            download_ctx.stream().record_event(tail_ready)
            tail_ready.synchronize()
            for j in range(tail_count):
                result[tail_offset + j] = tail_slot.host_out.value()[j]
        return TensorStatus.OK
    except:
        # An error after a native enqueue must not let borrowed
        # owners be recycled before the queued work is drained.
        try:
            # Do not free or recycle pinned/device memory unless EVERY lane
            # has quiesced. DeviceContext.synchronize() is NOT a device-wide
            # fence and cannot cover the selected independent streams.
            upload_ctx.stream().synchronize()
            ctx.stream().synchronize()
            download_ctx.stream().synchronize()
        except:
            slot0.quarantine()
            slot1.quarantine()
            tail_slot.quarantine()
            # The three issuing stream contexts are independent lifetime
            # capabilities. The window owners retain the BASE context, but
            # that does not establish lifetime of select_stream(1/2) views.
            # A failed quiescence must retain those views as well: dropping
            # either one could destroy a live foreign queue while its pinned
            # staging or device allocation remains in flight.
            forget_deinit(upload_ctx)
            forget_deinit(download_ctx)
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# CPU-only lowering has no accelerator context, artifact or GPU requirement.
# This is usable on devices with no MAX GPU device/driver at all.
def execute_cpu_binary_host_fp32[
    operation: Int,
    a: Origin[mut=False],
    b: Origin[mut=False],
    out_origin: Origin[mut=True],
](
    plan: ExecutionPlan,
    left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    comptime assert operation >= 0 and operation <= 5, "FP32 opcode must be 0..5"
    var plan_proof = plan.checked_snapshot()
    if not plan_proof.is_ok():
        return plan_proof.status
    var checked_plan = plan_proof.payload.value().copy()
    if checked_plan.placement != materialize[ExecutionPlacement.CPU_ONLY]() or checked_plan.output_device != 0:
        return TensorStatus.DEVICE_MISMATCH
    var count = checked_plan.total_items
    if count != len(left) or count != len(right) or count != len(result):
        return TensorStatus.SIZE_MISMATCH
    # An exact address-range proof enables a no-staging CPU fast path for
    # disjoint borrowed buffers. A conservative failure (including signed
    # address overflow) preserves the old snapshot-before-write behavior.
    if count == 0:
        return TensorStatus.OK
    var left_safe = host_output_disjoint_from_input(result, left)
    if not left_safe:
        left_safe = host_output_exactly_matches_input(result, left)
    var right_safe = host_output_disjoint_from_input(result, right)
    if not right_safe:
        right_safe = host_output_exactly_matches_input(result, right)
    if left_safe and right_safe:
        comptime width = simd_width_of[DType.float32]()
        return cpu_binary_contiguous_checked[
            operation, width,
        ](result, left, right)
    # Each unknown/shifted alias requires a pre-write snapshot.
    # An independently safe second input is read directly without a copy.
    var saved_left_count = 0
    var saved_right_count = 0
    if not left_safe:
        saved_left_count = count
    if not right_safe:
        saved_right_count = count
    var saved_left = List[Float32](length=saved_left_count, fill=0.0)
    var saved_right = List[Float32](length=saved_right_count, fill=0.0)
    if not left_safe:
        for i in range(count):
            saved_left[i] = left[i]
    if not right_safe:
        for i in range(count):
            saved_right[i] = right[i]
    for i in range(count):
        var x = Float32(0)
        var y = Float32(0)
        if left_safe:
            x = left[i]
        else:
            x = saved_left[i]
        if right_safe:
            y = right[i]
        else:
            y = saved_right[i]
        comptime if operation == 0:
            result[i] = x + y
        elif operation == 1:
            result[i] = x - y
        elif operation == 2:
            result[i] = x * y
        elif operation == 3:
            result[i] = x / y
        elif operation == 4:
            result[i] = fp32_maximum(x, y)
        else:
            result[i] = fp32_minimum(x, y)
    return TensorStatus.OK


# INTERNAL UNBOUND host-result lowering. This is an unsafe adapter: a raw
# DeviceExternalFunction cannot prove that its opcode/ABI matches the plan.
# Public host-result execution must pass an explicit AOT manifest identity.
# Operation: 0=add, 1=sub, 2=mul, 3=div, 4=maximum, 5=minimum.
# No implicit GPU result residency or untracked asynchronous owner escape. Unknown overlap between caller spans is resolved by snapshots.
# GPU_ONLY is chunked by a compile-time per-launch staging capacity; the
# logical tensor size is NOT capped by the staging allocation capacity.
def _unsafe_execute_host_binary_fp32[
    static_gpu_capacity: Int,
    operation: Int,
    a: Origin[mut=False],
    b: Origin[mut=False],
    out_origin: Origin[mut=True],
](
    ctx: DeviceContext,
    precompiled_binary: DeviceExternalFunction,
    plan: ExecutionPlan,
    left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) raises -> TensorStatus:
    comptime assert static_gpu_capacity > 0, "GPU staging capacity must be positive"
    comptime assert operation >= 0 and operation <= 5, "FP32 opcode must be 0..5"
    var plan_proof = plan.checked_host_binary_snapshot()
    if not plan_proof.is_ok():
        return plan_proof.status
    var checked_plan = plan_proof.payload.value().copy()
    var count = checked_plan.total_items
    if count != len(left) or count != len(right) or count != len(result):
        return TensorStatus.SIZE_MISMATCH
    if checked_plan.output_device != 0:
        return TensorStatus.DEVICE_MISMATCH
    if checked_plan.placement != materialize[ExecutionPlacement.CPU_ONLY]() and checked_plan.placement != materialize[ExecutionPlacement.GPU_ONLY]() and checked_plan.placement != materialize[ExecutionPlacement.HYBRID_SPLIT]() and checked_plan.placement != materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]() and checked_plan.placement != materialize[ExecutionPlacement.HYBRID_PIPELINE]():
        return TensorStatus.UNSUPPORTED
    if count == 0:
        return TensorStatus.OK
    if checked_plan.placement == materialize[ExecutionPlacement.CPU_ONLY]():
        return execute_cpu_binary_host_fp32[operation](checked_plan, left, right, result)
    if checked_plan.placement == materialize[ExecutionPlacement.HYBRID_PIPELINE]():
        return _pipeline_binary_host_fp32[static_gpu_capacity, operation](
            ctx, precompiled_binary, checked_plan, left, right, result
        )
    if checked_plan.placement == materialize[ExecutionPlacement.HYBRID_SPLIT]() or checked_plan.placement == materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]():
        # Both mappings use a checked host-result stage DAG and an event join.
        return _hybrid_binary_host_fp32[static_gpu_capacity, operation](
            ctx, precompiled_binary, checked_plan, left, right, result
        )
    if checked_plan.placement != materialize[ExecutionPlacement.GPU_ONLY]():
        return TensorStatus.UNSUPPORTED
    if count > Int.MAX // 12:
        return TensorStatus.OVERFLOW
    if checked_plan.transfer_bytes < count * 12:
        return TensorStatus.SIZE_MISMATCH
    # All GPU_ONLY chunks use this finite AOT workgroup ABI. Fail before
    # taking full-host input snapshots or allocating any stage buffers.
    var grid_proof = gpu_grid_1d(min(count, static_gpu_capacity), 256)
    if not grid_proof.is_ok():
        return grid_proof.status
    # Snapshot each alias-unsafe source BEFORE the first output write.
    # Shifted aliases must never corrupt a later GPU chunk.
    var direct_left = host_output_disjoint_from_input(result, left)
    if not direct_left:
        direct_left = host_output_exactly_matches_input(result, left)
    var direct_right = host_output_disjoint_from_input(result, right)
    if not direct_right:
        direct_right = host_output_exactly_matches_input(result, right)
    var direct_inputs = direct_left and direct_right
    var saved_left_count = 0
    var saved_right_count = 0
    if not direct_left:
        saved_left_count = count
    if not direct_right:
        saved_right_count = count
    var saved_left = List[Float32](length=saved_left_count, fill=0.0)
    var saved_right = List[Float32](length=saved_right_count, fill=0.0)
    if not direct_left:
        for i in range(count):
            saved_left[i] = left[i]
    if not direct_right:
        for i in range(count):
            saved_right[i] = right[i]
    # Kernel and transfer buffer sizes match on every chunk: never copy a
    # capacity-sized padded tail nor read a stale uninitialized staging slot.
    var at = 0
    # One event-owned set of buffers survives all same-capacity GPU_ONLY
    # windows. The preceding D2H event is joined before host inputs are
    # overwritten, so there is no in-flight DMA when a slot is reused.
    # A final short window uses exact capacities rather than copying padding.
    var window_owners = _HybridBinaryWindowOwners()
    while at < count:
        var chunk = min(static_gpu_capacity, count - at)
        try:
            if at == 0 or chunk != static_gpu_capacity:
                if at > 0:
                    var released = window_owners.release_after_join()
                    if released != TensorStatus.OK:
                        return released
                window_owners.acquire(ctx, chunk)
            # Async allocation completion is required before CPU writes to host
            # staging. A pooled specialization can eliminate this setup barrier.
            if at == 0 or chunk != static_gpu_capacity:
                var allocation_ready = ctx.create_event()
                ctx.stream().record_event(allocation_ready)
                allocation_ready.synchronize()
            var native_owners = validate_binary_fp32_device_owners(
                window_owners.device_out.value(), window_owners.device_a.value(), window_owners.device_b.value(), chunk,
            )
            if native_owners != TensorStatus.OK:
                return native_owners
            for j in range(chunk):
                if direct_left:
                    window_owners.host_a.value()[j] = left[at + j]
                else:
                    window_owners.host_a.value()[j] = saved_left[at + j]
                if direct_right:
                    window_owners.host_b.value()[j] = right[at + j]
                else:
                    window_owners.host_b.value()[j] = saved_right[at + j]
            ctx.enqueue_copy(dst_buf=window_owners.device_a.value(), src_buf=window_owners.host_a.value())
            ctx.enqueue_copy(dst_buf=window_owners.device_b.value(), src_buf=window_owners.host_b.value())
            var blocks = chunk // 256
            if chunk % 256 != 0:
                blocks += 1
            ctx.enqueue_function(
                precompiled_binary, window_owners.device_out.value(), window_owners.device_a.value(), window_owners.device_b.value(), chunk,
                grid_dim=blocks, block_dim=256
            )
            ctx.enqueue_copy(dst_buf=window_owners.host_out.value(), src_buf=window_owners.device_out.value())
            var completion = ctx.create_event()
            ctx.stream().record_event(completion)
            completion.synchronize()
            for j in range(chunk):
                result[at + j] = window_owners.host_out.value()[j]
            at += chunk
        except:
            # An error after a native enqueue must not let borrowed
            # owners be recycled before the queued work is drained.
            try:
                ctx.stream().synchronize()
            except:
                window_owners.quarantine()
                return TensorStatus.GPU_QUIESCENCE_UNPROVEN
            return TensorStatus.NATIVE_LAUNCH_FAILED
    return TensorStatus.OK

# Backward-compatible host-add entry points. CPU-only requires no device.
def execute_cpu_add_host_fp32[
    a: Origin[mut=False], b: Origin[mut=False], out_origin: Origin[mut=True],
](
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_cpu_binary_host_fp32[0](plan, left, right, result)


# Expert borrowed-Span AOT host-result binary entry point. It requires a
# declared native artifact identity for accelerator mappings and validates
# the transfer contract. A non-OK return can follow completed writes in an
# earlier CPU/GPU window; callers must regard the ENTIRE destination as
# uninitialized after ANY failed invocation, not just native launch failures.
# Prefer Tensor.unsafe_hybrid_binary_into for owner-scoped invalidation.
# This is metadata validation, not cryptographic native-symbol authentication.
def execute_host_binary_fp32[
    static_gpu_capacity: Int, operation: Int,
    a: Origin[mut=False], b: Origin[mut=False], out_origin: Origin[mut=True],
](
    ctx: DeviceContext, precompiled_binary: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity, target_arch: Int,
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    comptime assert static_gpu_capacity > 0, "GPU staging capacity must be positive"
    comptime assert operation >= 0 and operation <= 5, "FP32 opcode must be 0..5"
    var proof = plan.checked_host_binary_snapshot()
    if not proof.is_ok():
        return proof.status
    var selected = proof.payload.value().copy()
    if selected.total_items != len(left) or selected.total_items != len(right) or selected.total_items != len(result):
        return TensorStatus.SIZE_MISMATCH
    if selected.placement == materialize[ExecutionPlacement.CPU_ONLY]():
        # No GPU, stream, or device artifact is accessed by this path.
        return execute_cpu_binary_host_fp32[operation](selected, left, right, result)
    var binding = identity.validate_for(operation, selected.aot_variant, target_arch)
    if binding != TensorStatus.OK:
        return binding
    # These AOT host binary kernels expose a fixed 256-thread flat ABI.
    if identity.threads_per_block != 256:
        return TensorStatus.UNSUPPORTED
    try:
        return _unsafe_execute_host_binary_fp32[static_gpu_capacity, operation](
            ctx, precompiled_binary, selected, left, right, result,
        )
    except:
        # An exception may happen after enqueue; a failed event recording
        # must not silently release buffers while work is still pending.
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Explicitly unsafe advanced adapter for an externally verified native ABI.
# The normal public call is execute_host_binary_fp32 with an artifact identity.
def unsafe_execute_host_binary_fp32[
    static_gpu_capacity: Int, operation: Int,
    a: Origin[mut=False], b: Origin[mut=False], out_origin: Origin[mut=True],
](
    ctx: DeviceContext, precompiled_binary: DeviceExternalFunction,
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) raises -> TensorStatus:
    return _unsafe_execute_host_binary_fp32[static_gpu_capacity, operation](
        ctx, precompiled_binary, plan, left, right, result,
    )


# Source-level named spellings all use the checked public entry point.
# Different FP32 operations share ONE AOT dispatch/lowering implementation.
def hybrid_add_host_fp32[
    static_gpu_capacity: Int, a: Origin[mut=False], b: Origin[mut=False],
    out_origin: Origin[mut=True],
](
    ctx: DeviceContext, precompiled_add: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity, target_arch: Int,
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    var proof = plan.checked_host_binary_snapshot()
    if not proof.is_ok():
        return proof.status
    var selected = proof.payload.value().copy()
    if selected.placement != materialize[ExecutionPlacement.HYBRID_SPLIT]() and selected.placement != materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]():
        return TensorStatus.DEVICE_MISMATCH
    return execute_host_binary_fp32[static_gpu_capacity, 0](
        ctx, precompiled_add, identity, target_arch, selected, left, right, result,
    )


def execute_host_add_fp32[
    static_gpu_capacity: Int, a: Origin[mut=False], b: Origin[mut=False],
    out_origin: Origin[mut=True],
](
    ctx: DeviceContext, precompiled_add: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity, target_arch: Int,
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_host_binary_fp32[static_gpu_capacity, 0](
        ctx, precompiled_add, identity, target_arch, plan, left, right, result,
    )


def execute_cpu_sub_host_fp32[
    a: Origin[mut=False], b: Origin[mut=False], out_origin: Origin[mut=True],
](
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_cpu_binary_host_fp32[1](plan, left, right, result)


def execute_host_sub_fp32[
    static_gpu_capacity: Int, a: Origin[mut=False], b: Origin[mut=False],
    out_origin: Origin[mut=True],
](
    ctx: DeviceContext, precompiled_sub: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity, target_arch: Int,
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_host_binary_fp32[static_gpu_capacity, 1](
        ctx, precompiled_sub, identity, target_arch, plan, left, right, result,
    )


def execute_cpu_mul_host_fp32[
    a: Origin[mut=False], b: Origin[mut=False], out_origin: Origin[mut=True],
](
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_cpu_binary_host_fp32[2](plan, left, right, result)


def execute_host_mul_fp32[
    static_gpu_capacity: Int, a: Origin[mut=False], b: Origin[mut=False],
    out_origin: Origin[mut=True],
](
    ctx: DeviceContext, precompiled_mul: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity, target_arch: Int,
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_host_binary_fp32[static_gpu_capacity, 2](
        ctx, precompiled_mul, identity, target_arch, plan, left, right, result,
    )


def execute_cpu_div_host_fp32[
    a: Origin[mut=False], b: Origin[mut=False], out_origin: Origin[mut=True],
](
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_cpu_binary_host_fp32[3](plan, left, right, result)


def execute_host_div_fp32[
    static_gpu_capacity: Int, a: Origin[mut=False], b: Origin[mut=False],
    out_origin: Origin[mut=True],
](
    ctx: DeviceContext, precompiled_div: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity, target_arch: Int,
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_host_binary_fp32[static_gpu_capacity, 3](
        ctx, precompiled_div, identity, target_arch, plan, left, right, result,
    )


def execute_cpu_maximum_host_fp32[
    a: Origin[mut=False], b: Origin[mut=False], out_origin: Origin[mut=True],
](
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_cpu_binary_host_fp32[4](plan, left, right, result)


def execute_host_maximum_fp32[
    static_gpu_capacity: Int, a: Origin[mut=False], b: Origin[mut=False],
    out_origin: Origin[mut=True],
](
    ctx: DeviceContext, precompiled_maximum: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity, target_arch: Int,
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_host_binary_fp32[static_gpu_capacity, 4](
        ctx, precompiled_maximum, identity, target_arch, plan, left, right, result,
    )


def execute_cpu_minimum_host_fp32[
    a: Origin[mut=False], b: Origin[mut=False], out_origin: Origin[mut=True],
](
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_cpu_binary_host_fp32[5](plan, left, right, result)


def execute_host_minimum_fp32[
    static_gpu_capacity: Int, a: Origin[mut=False], b: Origin[mut=False],
    out_origin: Origin[mut=True],
](
    ctx: DeviceContext, precompiled_minimum: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity, target_arch: Int,
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_host_binary_fp32[static_gpu_capacity, 5](
        ctx, precompiled_minimum, identity, target_arch, plan, left, right, result,
    )


# Backward-compatible checked spelling. The canonical implementation is the
# same execute_host_binary_fp32() status-only boundary above.
def execute_bound_host_binary_fp32[
    static_gpu_capacity: Int, operation: Int,
    a: Origin[mut=False], b: Origin[mut=False], out_origin: Origin[mut=True],
](
    ctx: DeviceContext, precompiled_binary: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity, target_arch: Int,
    plan: ExecutionPlan, left: Span[mut=False, Float32, a],
    right: Span[mut=False, Float32, b],
    mut result: Span[mut=True, Float32, out_origin],
) -> TensorStatus:
    return execute_host_binary_fp32[static_gpu_capacity, operation](
        ctx, precompiled_binary, identity, target_arch, plan,
        left, right, result,
    )


# A native-AOT unary lowering of the SAME fixed-FP32 ExecutionPlan. Covers
# each of the nine primitive CPU unary opcodes with host CPU_ONLY, GPU_ONLY,
# and cost-selected disjoint HYBRID_SPLIT / CPU_ASSIST placement. The source
# tensor and destination remain the public Tensor owners. The supplied
# DeviceExternalFunction is a trusted-expert ABI prerequisite; identity
# metadata alone cannot authenticate its compiled instructions.
struct _HybridUnaryWindowOwners(Movable):
    var host_input: Optional[HostBuffer[DType.float32]]
    var host_output: Optional[HostBuffer[DType.float32]]
    var device_input: Optional[DeviceBuffer[DType.float32]]
    var device_output: Optional[DeviceBuffer[DType.float32]]
    var lifetime_context: Optional[DeviceContext]
    var quiescence_unproven: Bool

    def __init__(out self):
        self.host_input = Optional[HostBuffer[DType.float32]](None)
        self.host_output = Optional[HostBuffer[DType.float32]](None)
        self.device_input = Optional[DeviceBuffer[DType.float32]](None)
        self.device_output = Optional[DeviceBuffer[DType.float32]](None)
        self.lifetime_context = Optional[DeviceContext](None)
        self.quiescence_unproven = False

    def acquire(mut self, ctx: DeviceContext, count: Int) raises:
        # Keep the original allocating context alive even if the first
        # asynchronous buffer creation raises before returning an owner.
        self.lifetime_context = Optional[DeviceContext](ctx.copy())
        self.host_input = Optional[HostBuffer[DType.float32]](
            ctx.enqueue_create_host_buffer[DType.float32](count)
        )
        self.host_output = Optional[HostBuffer[DType.float32]](
            ctx.enqueue_create_host_buffer[DType.float32](count)
        )
        self.device_input = Optional[DeviceBuffer[DType.float32]](
            ctx.enqueue_create_buffer[DType.float32](count)
        )
        self.device_output = Optional[DeviceBuffer[DType.float32]](
            ctx.enqueue_create_buffer[DType.float32](count)
        )

    def quarantine(mut self):
        self.quiescence_unproven = True

    def __deinit__(deinit self):
        if self.quiescence_unproven:
            if self.host_input:
                var owner = self.host_input.take()
                forget_deinit(owner)
            if self.host_output:
                var owner = self.host_output.take()
                forget_deinit(owner)
            if self.device_input:
                var owner = self.device_input.take()
                forget_deinit(owner)
            if self.device_output:
                var owner = self.device_output.take()
                forget_deinit(owner)
            if self.lifetime_context:
                var owner = self.lifetime_context.take()
                forget_deinit(owner)
        else:
            # Explicitly drop buffers before the context that allocated them.
            # Native work has already been joined by the caller.
            self.host_input = Optional[HostBuffer[DType.float32]](None)
            self.host_output = Optional[HostBuffer[DType.float32]](None)
            self.device_input = Optional[DeviceBuffer[DType.float32]](None)
            self.device_output = Optional[DeviceBuffer[DType.float32]](None)
            # MAX buffer destructors enqueue asynchronous frees on the
            # originating context. An earlier compute/D2H completion
            # event does NOT fence these later free commands. Drain them
            # while the native context is still alive, once at final
            # teardown (never between ordinary reused windows).
            # On a failed drain, retain the context instead of allowing
            # premature native-context destruction.
            var cleanup_completed = True
            if self.lifetime_context:
                try:
                    self.lifetime_context.value().synchronize()
                except:
                    cleanup_completed = False
            if cleanup_completed:
                self.lifetime_context = Optional[DeviceContext](None)
            else:
                var unrecovered_context = self.lifetime_context.take()
                forget_deinit(unrecovered_context)


def unsafe_execute_host_unary_fp32[
    operation: Int, static_gpu_capacity: Int,
    source_origin: Origin[mut=False], result_origin: Origin[mut=True],
](
    ctx: DeviceContext, native_unary: DeviceExternalFunction,
    identity: UnaryAOTArtifactIdentity, target_arch: Int,
    plan: ExecutionPlan, source: Span[mut=False, Float32, source_origin],
    mut result: Span[mut=True, Float32, result_origin],
) -> TensorStatus:
    # The checked boundary owns all shape/transfer/partition proofs, and
    # intentionally refuses non-disjoint source/result aliases for native
    # DMA. This avoids both shifted-alias corruption and hidden snapshots.
    if operation < 0 or operation > 8:
        return TensorStatus.UNSUPPORTED
    var plan_proof = plan.checked_host_unary_snapshot()
    if not plan_proof.is_ok():
        return plan_proof.status
    var selected = plan_proof.payload.value().copy()
    # Bind ALL placements to the same AOT unary opcode. In particular, a
    # forged CPU_ONLY plan with variant 27 must not be executed as NEG (20).
    if selected.aot_variant != 20 + operation:
        return TensorStatus.DEVICE_MISMATCH
    var count = selected.total_items
    if count != len(source) or count != len(result):
        return TensorStatus.SIZE_MISMATCH
    if count <= 0:
        return TensorStatus.OK
    if selected.placement == materialize[ExecutionPlacement.CPU_ONLY]():
        # The scalar/SIMD CPU unary kernel loads each lane before storing
        # that lane, so exact element-for-element in-place operation is safe.
        # Shifted/unknown overlaps are not safe without an input snapshot.
        if (not host_output_disjoint_from_input(result, source)
            and not host_output_exactly_matches_input(result, source)):
            return TensorStatus.ALIAS_UNKNOWN
        return cpu_unary_contiguous_fp32[
            operation, simd_width_of[DType.float32](),
        ](source, result)
    if static_gpu_capacity <= 0:
        return TensorStatus.UNSUPPORTED
    if not host_output_disjoint_from_input(result, source):
        return TensorStatus.ALIAS_UNKNOWN
    if (selected.placement != materialize[ExecutionPlacement.GPU_ONLY]()
        and selected.placement != materialize[ExecutionPlacement.HYBRID_SPLIT]()
        and selected.placement != materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]()):
        return TensorStatus.UNSUPPORTED
    if target_arch <= 0 or identity.threads_per_block != 256:
        return TensorStatus.DEVICE_MISMATCH
    var binding = identity.validate_for(
        operation, selected.aot_variant, target_arch,
    )
    if binding != TensorStatus.OK:
        return binding
    # The CPU and GPU output regions are disjoint by ExecutionPlan.checked().
    # A GPU-only plan has zero CPU elements. The first GPU chunk is queued
    # BEFORE synchronous CPU work, yielding actual host/device concurrency.
    var cpu_pending = selected.placement != materialize[ExecutionPlacement.GPU_ONLY]()
    var at = selected.gpu_begin
    while at < selected.gpu_end:
        var chunk = min(static_gpu_capacity, selected.gpu_end - at)
        var geometry = gpu_grid_1d(chunk, 256)
        if not geometry.is_ok():
            return geometry.status
        var owners = _HybridUnaryWindowOwners()
        try:
            owners.acquire(ctx, chunk)
            var allocation_ready = ctx.create_event()
            ctx.stream().record_event(allocation_ready)
            allocation_ready.synchronize()
            for i in range(chunk):
                owners.host_input.value()[i] = source[at + i]
            ctx.enqueue_copy(
                dst_buf=owners.device_input.value(),
                src_buf=owners.host_input.value(),
            )
            ctx.enqueue_function(
                native_unary, owners.device_output.value(),
                owners.device_input.value(), chunk,
                grid_dim=geometry.payload.value().grid_blocks, block_dim=256,
            )
            ctx.enqueue_copy(
                dst_buf=owners.host_output.value(),
                src_buf=owners.device_output.value(),
            )
            var ready = ctx.create_event()
            ctx.stream().record_event(ready)
            if cpu_pending:
                var cpu_source = source[selected.cpu_begin:selected.cpu_end]
                var cpu_output = result[selected.cpu_begin:selected.cpu_end]
                var cpu_status = cpu_unary_contiguous_fp32[
                    operation, simd_width_of[DType.float32](),
                ](cpu_source, cpu_output)
                if cpu_status != TensorStatus.OK:
                    # GPU D2H is outstanding. Join the exact recorded event
                    # before destroying any of its device or staging owners.
                    ready.synchronize()
                    return cpu_status
                cpu_pending = False
            ready.synchronize()
            # Publish host output only AFTER the final D2H dependency joins.
            for i in range(chunk):
                result[at + i] = owners.host_output.value()[i]
            at += chunk
        except:
            try:
                ctx.stream().synchronize()
            except:
                owners.quarantine()
                return TensorStatus.GPU_QUIESCENCE_UNPROVEN
            # Some CPU or earlier GPU chunks may already have modified result;
            # the owning Tensor facade poisons it before returning this error.
            return TensorStatus.NATIVE_LAUNCH_FAILED
    return TensorStatus.OK
