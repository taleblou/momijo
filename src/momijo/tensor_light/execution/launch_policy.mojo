# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR EXECUTION / LAUNCH AND PLACEMENT POLICY
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Execution Planning and AOT
# Module:        momijo.tensor_light.execution.launch_policy
# Source:        src/momijo/tensor_light/execution/launch_policy.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines launch and placement policy within checked AOT execution planning
#   for the FP32 Light implementation.
#
#   Public declarations include `HybridTuningProfile`,
#   `validate_hybrid_tuning_profile`, `choose_elementwise_plan`,
#   `choose_validated_split`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Select or prepare validated launch, placement, and execution-plan
#     policies; do not claim that policy selection performs device work.
#   - Expose module-level declarations including `HybridTuningProfile`,
#     `validate_hybrid_tuning_profile`, `choose_elementwise_plan`,
#     `choose_validated_split`.
#   - Preserve shape, dtype, target, provenance, and capacity constraints
#     where required by the declarations.
#   - Keep planning and validation separate from proof of successful kernel
#     execution.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate the declared policy, dependency, shape, capacity, and target
#     constraints before dispatch.
#   - Do not treat a selected, prepared, or sealed plan as evidence of a
#     completed native execution.
#
# Performance Objectives:
#   - Keep plan construction, policy checks, and metadata queries inexpensive.
#   - Avoid repeated target selection and unnecessary allocations in
#     dispatch-critical paths.
#
# Maintainability:
#   - Keep metadata schemas synchronized with the binding and execution
#     consumers.
#   - Test invalid plans, stale bindings, unsupported targets, and policy
#     boundary cases.
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

# Pure cost model consumes offline calibrated coefficients; never benchmarks at call time.
from ..internal.status import TensorStatus, TensorResult
from .dependency import ExecutionStage, ExecutionStageKind, ExecutionAccess, ExecutionBufferBounds, CheckedStageSchedule, BoundStageSchedule, bind_schedule_buffers_checked
from .execution_plan import ExecutionPlan, ExecutionPlacement
from std.utils.numerics import isfinite
from ..memory.fp32_budget import Fp32MemoryLimits, plan_fp32_binary_windows

@fieldwise_init
struct HybridTuningProfile(Copyable):
    var cpu_elems_per_us: Float64
    var gpu_elems_per_us: Float64
    var h2d_bytes_per_us: Float64
    var gpu_launch_us: Float64
    var join_us: Float64


# Validate immutable offline coefficients once at the checked dispatch boundary.
# NaN compares false to ordinary bounds, so finite checks are not optional.
def validate_hybrid_tuning_profile(profile: HybridTuningProfile) -> TensorStatus:
    if not Bool(isfinite(profile.cpu_elems_per_us)) or profile.cpu_elems_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.gpu_elems_per_us)) or profile.gpu_elems_per_us < 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.h2d_bytes_per_us)) or profile.h2d_bytes_per_us < 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.gpu_launch_us)) or profile.gpu_launch_us < 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.join_us)) or profile.join_us < 0.0:
        return TensorStatus.INVALID_LAYOUT
    return TensorStatus.OK


def choose_elementwise_plan(
    count: Int, gpu_partition_transfer_bytes: Int,
    profile: HybridTuningProfile, variant: Int, gpu_enabled: Bool,
    device_residency_verified: Bool = False,
) -> TensorResult[ExecutionPlan]:
    # This selector receives no Tensor owners or transfer graph. Therefore it
    # may NOT choose a CPU mapping for device-only operands, or an implicit
    # GPU migration for host operands. Device residency is established by the
    # caller's checked ownership boundary, then rechecked against real native
    # buffers by the lowering. This Bool is NOT a pointer/ABI authentication.
    # Host-result binary workloads use choose_host_add_plan_for_capacity(),
    # which explicitly budgets two H2D copies, one D2H, and their events.
    var profile_status = validate_hybrid_tuning_profile(profile)
    if profile_status != TensorStatus.OK:
        return TensorResult[ExecutionPlan](profile_status)
    if count < 0 or gpu_partition_transfer_bytes < 0 or variant < 0 or gpu_partition_transfer_bytes % 4 != 0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    if count > Int.MAX // 4:
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    if device_residency_verified:
        # A CPU-only plan here would silently read GPU-local storage on the
        # host. If the GPU target is unavailable, fail rather than migrate.
        if not gpu_enabled or profile.gpu_elems_per_us <= 0.0:
            return TensorResult[ExecutionPlan](TensorStatus.BACKEND_UNAVAILABLE)
        if gpu_partition_transfer_bytes != 0:
            # Native pre-resident ABI has no explicit transfer steps.
            return TensorResult[ExecutionPlan](TensorStatus.NOT_REPRESENTABLE)
        var gpu_dependencies = 1
        if count == 0:
            gpu_dependencies = 0
        return ExecutionPlan.checked(
            ExecutionPlacement.GPU_ONLY, count, 0, 0, 0, count,
            0, variant, gpu_dependencies, 1,
        )
    # With only the host operands proven resident, no GPU lowering or legal
    # transfer schedule is supplied to this generic selector. Do not choose a
    # seemingly faster but unexecutable GPU_ONLY plan from a cost estimate.
    return ExecutionPlan.checked(
        ExecutionPlacement.CPU_ONLY, count, 0, count, 0, 0,
        0, variant, 0, 0,
    )


def choose_validated_split(count: Int, cpu_items: Int, gpu_split_bytes: Int, gpu_full_bytes: Int, profile: HybridTuningProfile, variant: Int, gpu_enabled: Bool) -> TensorResult[ExecutionPlan]:
    var profile_status = validate_hybrid_tuning_profile(profile)
    if profile_status != TensorStatus.OK:
        return TensorResult[ExecutionPlan](profile_status)
    if not gpu_enabled or count <= 0 or cpu_items <= 0 or cpu_items >= count or gpu_split_bytes < 0 or gpu_full_bytes < 0 or variant < 0 or gpu_split_bytes % 4 != 0 or gpu_full_bytes % 4 != 0:
        return TensorResult[ExecutionPlan](TensorStatus.NOT_REPRESENTABLE)
    if count > Int.MAX // 4:
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    # This selector's only implemented hybrid lowering is host-result FP32
    # binary: TWO input copies and ONE result copy for each GPU-owned item.
    # Reject an optimistic transfer estimate BEFORE it can win the model.
    var gpu_items = count - cpu_items
    if gpu_items > Int.MAX // 12 or gpu_split_bytes < gpu_items * 12:
        return TensorResult[ExecutionPlan](TensorStatus.SIZE_MISMATCH)
    # GPU_ONLY is a host-result mapping in this selector. The full transfer
    # estimate must cover the same three FP32 buffers, not just one direction.
    if count > Int.MAX // 12 or gpu_full_bytes < count * 12:
        return TensorResult[ExecutionPlan](TensorStatus.SIZE_MISMATCH)
    if profile.cpu_elems_per_us <= 0.0 or profile.gpu_elems_per_us <= 0.0 or profile.h2d_bytes_per_us <= 0.0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var cpu_time = Float64(cpu_items) / profile.cpu_elems_per_us
    var gpu_time = profile.gpu_launch_us + Float64(gpu_split_bytes) / profile.h2d_bytes_per_us + Float64(count - cpu_items) / profile.gpu_elems_per_us
    if not Bool(isfinite(cpu_time)) or not Bool(isfinite(gpu_time)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var hybrid_time = gpu_time
    if cpu_time > hybrid_time:
        hybrid_time = cpu_time
    hybrid_time += profile.join_us
    if not Bool(isfinite(hybrid_time)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var cpu_only = Float64(count) / profile.cpu_elems_per_us
    var gpu_only = profile.gpu_launch_us + Float64(gpu_full_bytes) / profile.h2d_bytes_per_us + Float64(count) / profile.gpu_elems_per_us
    if not Bool(isfinite(cpu_only)) or not Bool(isfinite(gpu_only)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    if hybrid_time >= cpu_only or hybrid_time >= gpu_only:
        return TensorResult[ExecutionPlan](TensorStatus.NOT_REPRESENTABLE)
    return ExecutionPlan.checked(ExecutionPlacement.HYBRID_SPLIT, count, 0, cpu_items, cpu_items, count, gpu_split_bytes, variant, 2, 0)


# A Float64 representation cannot distinguish every Int above 2^53.
# Clamping only BEFORE Int conversion can round a legal upper bound UP to
# count (a zero-element GPU partition), or above an allowed row capacity.
# Apply the integer-domain bounds AFTER converting a FINITE candidate.
# Callers check isfinite and use this only for representable nonnegative
# FP32 counts; this is PREPARE-only and never in a Tensor arithmetic loop.
def _bounded_integer_hint(value: Float64, lower: Int, upper: Int) -> Int:
    var floating_bound = max(Float64(lower), min(Float64(upper), value))
    var integral = Int(floating_bound)
    return min(upper, max(lower, integral))


# A host-resident FP32 add requires 2 H2D inputs and 1 D2H output.
# The analytical, constant-time split uses immutable offline timing coefficients,
# not runtime kernel measurements or fixed partition percentages.
def choose_host_add_plan(
    count: Int, profile: HybridTuningProfile, variant: Int, gpu_enabled: Bool
) -> TensorResult[ExecutionPlan]:
    var profile_status = validate_hybrid_tuning_profile(profile)
    if profile_status != TensorStatus.OK:
        return TensorResult[ExecutionPlan](profile_status)
    if count < 0 or variant < 0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    if profile.cpu_elems_per_us <= 0.0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    if profile.gpu_launch_us < 0.0 or profile.join_us < 0.0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var cpu_time = Float64(count) / profile.cpu_elems_per_us
    if not Bool(isfinite(cpu_time)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    if not gpu_enabled or profile.gpu_elems_per_us <= 0.0 or profile.h2d_bytes_per_us <= 0.0:
        return ExecutionPlan.checked(ExecutionPlacement.CPU_ONLY, count, 0, count, 0, 0, 0, variant, 0, 0)
    # The transfer-byte field is representable only below this boundary.
    # Do not reject a valid CPU plan for an unavailable GPU transfer size.
    if count > Int.MAX // 12:
        return ExecutionPlan.checked(ExecutionPlacement.CPU_ONLY, count, 0, count, 0, 0, 0, variant, 0, 0)
    var full_bytes = count * 12
    var gpu_time = (
        profile.gpu_launch_us + Float64(full_bytes) / profile.h2d_bytes_per_us
        + Float64(count) / profile.gpu_elems_per_us
    )
    if not Bool(isfinite(gpu_time)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var best_cost = cpu_time
    var best_kind = 0
    if gpu_time < best_cost:
        best_cost = gpu_time
        best_kind = 1
    var split_at = 0
    # The calibrated model is the maximum of two affine functions:
    #   CPU(x) = x / cpu_rate
    #   GPU(n-x) = launch + (n-x) * (12 / transfer_rate + 1 / gpu_rate)
    # The minimizer over the integer domain 1..n-1 is at one of the two
    # integers adjacent to their intersection (or at the clamped boundary).
    # This is O(1) and avoids arbitrary fixed-percentage partitions.
    if count > 1:
        var cpu_per_element = 1.0 / profile.cpu_elems_per_us
        var gpu_per_element = (
            12.0 / profile.h2d_bytes_per_us + 1.0 / profile.gpu_elems_per_us
        )
        if not Bool(isfinite(cpu_per_element)) or not Bool(isfinite(gpu_per_element)):
            return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
        var crossing = (
            profile.gpu_launch_us + Float64(count) * gpu_per_element
        ) / (cpu_per_element + gpu_per_element)
        if not Bool(isfinite(crossing)):
            return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
        # Apply floating and integer-domain clamps: above 2^53, Float64
        # can round count - 1 UP to count, creating an illegal empty GPU lane.
        crossing = max(1.0, min(Float64(count - 1), crossing))
        var floor_candidate = _bounded_integer_hint(crossing, 1, count - 1)
        for neighbor in range(2):
            var cpu_items = min(count - 1, max(1, floor_candidate + neighbor))
            var gpu_items = count - cpu_items
            var cpu_part_us = Float64(cpu_items) * cpu_per_element
            var gpu_part_us = (
                profile.gpu_launch_us + Float64(gpu_items) * gpu_per_element
            )
            var split_us = max(cpu_part_us, gpu_part_us) + profile.join_us
            if split_us < best_cost:
                best_cost = split_us
                best_kind = 2
                split_at = cpu_items
    if best_kind == 2:
        return ExecutionPlan.checked(
            ExecutionPlacement.HYBRID_SPLIT, count, 0, split_at,
            split_at, count, (count - split_at) * 12, variant, 2, 0,
        )
    if best_kind == 1:
        return ExecutionPlan.checked(
            ExecutionPlacement.GPU_ONLY, count, 0, 0, 0, count, full_bytes, variant, 1, 0,
        )
    return ExecutionPlan.checked(ExecutionPlacement.CPU_ONLY, count, 0, count, 0, 0, 0, variant, 0, 0)

# The executor accepts arbitrarily many GPU chunks, while each staging block
# is limited by the target's calibrated/selected capacity. Account for those
# extra AOT launches when comparing CPU, GPU and CPU+GPU split placements.
# This is a constant-time dispatch policy, not a benchmark or runtime autotuner.
def _host_add_split_time(
    count: Int, cpu_items: Int, capacity: Int,
    cpu_per_item: Float64, gpu_per_item: Float64,
    launch_us: Float64, join_us: Float64,
) -> Float64:
    var gpu_items = count - cpu_items
    var gpu_chunks = gpu_items // capacity
    if gpu_items % capacity != 0:
        gpu_chunks += 1
    # The executor assigns CPU work in quotient/remainder chunks. Each GPU
    # chunk and its CPU slice run concurrently, then join on that chunk's
    # completion event before either allocation is released. There are K-1
    # full-capacity chunks and one last (possibly partial) chunk.
    var cpu_each = cpu_items // gpu_chunks
    var cpu_extra = cpu_items % gpu_chunks
    var full_chunks = gpu_chunks - 1
    var full_with_extra = min(cpu_extra, full_chunks)
    var full_regular = full_chunks - full_with_extra
    var full_gpu_time = launch_us + Float64(capacity) * gpu_per_item
    var last_gpu_items = gpu_items - full_chunks * capacity
    var last_gpu_time = launch_us + Float64(last_gpu_items) * gpu_per_item
    var full_extra_cost = max(
        Float64(cpu_each + 1) * cpu_per_item, full_gpu_time
    )
    var full_regular_cost = max(
        Float64(cpu_each) * cpu_per_item, full_gpu_time
    )
    # The remainder is assigned to the FIRST cpu_extra chunks, so the final
    # chunk always owns cpu_each CPU elements.
    var last_cost = max(Float64(cpu_each) * cpu_per_item, last_gpu_time)
    return (
        Float64(full_with_extra) * full_extra_cost
        + Float64(full_regular) * full_regular_cost
        + last_cost + join_us
    )


def choose_host_add_plan_for_capacity(
    count: Int, gpu_chunk_capacity: Int, profile: HybridTuningProfile,
    variant: Int, gpu_enabled: Bool,
) -> TensorResult[ExecutionPlan]:
    # Unlike the unbounded analytic policy, this matches the chunked host
    # lowering and never picks a split that cannot be staged by the executor.
    var valid = validate_hybrid_tuning_profile(profile)
    if valid != TensorStatus.OK:
        return TensorResult[ExecutionPlan](valid)
    if count < 0 or variant < 0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    # Logical CPU work is counted in elements. No GPU byte-size bound
    # applies until a GPU mapping is actually considered.
    var cpu_cost = Float64(count) / profile.cpu_elems_per_us
    if not Bool(isfinite(cpu_cost)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    if not gpu_enabled or count == 0 or profile.gpu_elems_per_us <= 0.0 or profile.h2d_bytes_per_us <= 0.0:
        return ExecutionPlan.checked(
            ExecutionPlacement.CPU_ONLY, count, 0, count, 0, 0,
            0, variant, 0, 0,
        )
    # A host-result GPU add transfers three FP32 buffers. If the aggregate
    # transfer byte count cannot be represented, the CPU plan still can.
    if count > Int.MAX // 12:
        return ExecutionPlan.checked(
            ExecutionPlacement.CPU_ONLY, count, 0, count, 0, 0,
            0, variant, 0, 0,
        )
    # CPU-only hardware never needs a GPU staging allocation.
    if gpu_chunk_capacity <= 0:
        # No positive staging capacity means this target cannot lower a GPU
        # partition. Prefer the valid CPU_ONLY mapping instead of failing a
        # perfectly valid host workload merely because GPU staging is absent.
        return ExecutionPlan.checked(
            ExecutionPlacement.CPU_ONLY, count, 0, count, 0, 0,
            0, variant, 0, 0,
        )
    var cap = min(count, gpu_chunk_capacity)
    var cpu_per = 1.0 / profile.cpu_elems_per_us
    var gpu_per = (
        12.0 / profile.h2d_bytes_per_us + 1.0 / profile.gpu_elems_per_us
    )
    if not Bool(isfinite(cpu_per)) or not Bool(isfinite(gpu_per)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var all_gpu_chunks = count // cap
    if count % cap != 0:
        all_gpu_chunks += 1
    var all_gpu_time = (
        Float64(all_gpu_chunks) * profile.gpu_launch_us
        + Float64(count) * gpu_per
    )
    if not Bool(isfinite(all_gpu_time)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var best_kind = 0
    var best_time = cpu_cost
    var best_cpu_items = 0
    if all_gpu_time < best_time:
        best_kind = 1
        best_time = all_gpu_time
    if count > 1:
        # A few analytically motivated candidates capture the one-chunk
        # intersection, the amortized multi-chunk intersection, and the
        # boundary at which staging becomes exactly one chunk. No fixed
        # request/step limit and no exhaustive latency-heavy search.
        var denominator = cpu_per + gpu_per
        if denominator <= 0.0 or not Bool(isfinite(denominator)):
            return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
        var continuous = (
            profile.gpu_launch_us + Float64(count) * gpu_per
        ) / denominator
        var amortized = (
            profile.gpu_launch_us + Float64(count) * (
                gpu_per + profile.gpu_launch_us / Float64(cap)
            )
        ) / (denominator + profile.gpu_launch_us / Float64(cap))
        if not Bool(isfinite(continuous)) or not Bool(isfinite(amortized)):
            return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
        var candidates = Array[Int, 8](fill=1)
        candidates[0] = 1
        candidates[1] = count - 1
        candidates[2] = max(1, count - cap)
        candidates[3] = min(count - 1, max(1, count - cap + 1))
        candidates[4] = _bounded_integer_hint(continuous, 1, count - 1)
        candidates[5] = min(count - 1, candidates[4] + 1)
        candidates[6] = _bounded_integer_hint(amortized, 1, count - 1)
        candidates[7] = min(count - 1, candidates[6] + 1)
        for i in range(8):
            var cpu_items = candidates[i]
            var predicted = _host_add_split_time(
                count, cpu_items, cap, cpu_per, gpu_per,
                profile.gpu_launch_us, profile.join_us,
            )
            if Bool(isfinite(predicted)) and predicted < best_time:
                best_kind = 2
                best_time = predicted
                best_cpu_items = cpu_items
        # The last GPU window can be MUCH shorter than `cap`, whereas the
        # preceding windows are full. The continuous whole-GPU estimate above
        # balances the CPU cost against *all* GPU items and can miss the
        # piecewise optimum after each per-window event join. For example,
        # count=218, cap=130 may prefer 21 CPU elements, not ~30. Evaluate
        # the last-window CPU/GPU intersection under three already identified
        # chunk counts. This is a BOUNDED six-candidate refinement during
        # PREPARE, never a per-tile exhaustive search or hot-path autotune.
        # Use actual _host_add_split_time for the final candidate decision:
        # the intersection is only a hint, not a new performance guarantee.
        for seed_index in range(3):
            var seed_cpu = candidates[4]
            if seed_index == 1:
                seed_cpu = candidates[6]
            elif seed_index == 2:
                seed_cpu = candidates[0]
            var seed_gpu = count - seed_cpu
            var chunk_count = seed_gpu // cap
            if seed_gpu % cap != 0:
                chunk_count += 1
            # This chunk count is positive because 1 <= seed_cpu < count.
            # (chunk_count - 1) * cap <= seed_gpu, so there is no Int overflow.
            var last_window_base = count - (chunk_count - 1) * cap
            var tail_denominator = gpu_per + cpu_per / Float64(chunk_count)
            if not Bool(isfinite(tail_denominator)) or tail_denominator <= 0.0:
                continue
            var tail_crossing = (
                profile.gpu_launch_us + Float64(last_window_base) * gpu_per
            ) / tail_denominator
            if not Bool(isfinite(tail_crossing)):
                continue
            tail_crossing = max(1.0, min(Float64(count - 1), tail_crossing))
            var first_tail_candidate = _bounded_integer_hint(tail_crossing, 1, count - 1)
            for neighbor in range(2):
                var cpu_items = min(count - 1, first_tail_candidate + neighbor)
                var remaining_gpu = count - cpu_items
                var actual_chunks = remaining_gpu // cap
                if remaining_gpu % cap != 0:
                    actual_chunks += 1
                if actual_chunks != chunk_count:
                    continue
                var predicted = _host_add_split_time(
                    count, cpu_items, cap, cpu_per, gpu_per,
                    profile.gpu_launch_us, profile.join_us,
                )
                if Bool(isfinite(predicted)) and predicted < best_time:
                    best_kind = 2
                    best_time = predicted
                    best_cpu_items = cpu_items

        # Piecewise correction for discontinuous GPU window counts. The
        # eight analytic candidates plus last-window intersections can still
        # miss the exact first CPU item that drops one whole GPU chunk.
        # Evaluate both legal ends and the final-window intersection for
        # nearby chunk counts. This is a FIXED upper bound of 60 estimates
        # during plan selection, never an exhaustive per-item kernel search.
        var max_gpu_chunks = (count - 1) // cap
        if (count - 1) % cap != 0:
            max_gpu_chunks += 1
        for seed_index in range(3):
            var seed_cpu = candidates[4]
            if seed_index == 1:
                seed_cpu = candidates[6]
            elif seed_index == 2:
                seed_cpu = 1
            var seed_remaining = count - seed_cpu
            var seed_chunks = seed_remaining // cap
            if seed_remaining % cap != 0:
                seed_chunks += 1
            for chunk_offset in range(-2, 3):
                var chunks = seed_chunks + chunk_offset
                if chunks < 1 or chunks > max_gpu_chunks:
                    continue
                # cap and count are positive and count <= Int.MAX // 12;
                # the two products below therefore cannot overflow Int.
                var cpu_low = max(1, count - chunks * cap)
                var cpu_high = min(count - 1, count - (chunks - 1) * cap - 1)
                if cpu_low > cpu_high:
                    continue
                var tail_base = count - (chunks - 1) * cap
                var tail_denom = gpu_per + cpu_per / Float64(chunks)
                if not Bool(isfinite(tail_denom)) or tail_denom <= 0.0:
                    continue
                var tail_point = (
                    profile.gpu_launch_us + Float64(tail_base) * gpu_per
                ) / tail_denom
                if not Bool(isfinite(tail_point)):
                    continue
                var tail_cpu = _bounded_integer_hint(tail_point, cpu_low, cpu_high)
                for pick in range(4):
                    var cpu_items = cpu_low
                    if pick == 1:
                        cpu_items = cpu_high
                    elif pick == 2:
                        cpu_items = tail_cpu
                    elif pick == 3:
                        cpu_items = min(cpu_high, tail_cpu + 1)
                    var predicted = _host_add_split_time(
                        count, cpu_items, cap, cpu_per, gpu_per,
                        profile.gpu_launch_us, profile.join_us,
                    )
                    if Bool(isfinite(predicted)) and predicted < best_time:
                        best_kind = 2
                        best_time = predicted
                        best_cpu_items = cpu_items

        # Quotient/remainder distribution can jump when one more CPU item
        # is assigned to EVERY GPU window. Check a few boundaries around
        # full-window CPU/GPU cost parity; e.g. cap=1, count=195 has a
        # minimum at 130 CPU items, far from the affine intersection.
        # Finite guards avoid unsafe Float64-to-Int conversions.
        if cpu_per > 0.0:
            var full_gpu_us = profile.gpu_launch_us + Float64(cap) * gpu_per
            var cpu_per_full_window = full_gpu_us / cpu_per
            if Bool(isfinite(cpu_per_full_window)):
                var quotient_hint = _bounded_integer_hint(cpu_per_full_window, 0, count)
                for neighbor in range(-1, 4):
                    var quotient = quotient_hint + neighbor
                    if quotient <= 0:
                        continue
                    var real_quotient = Float64(quotient)
                    var estimate = (
                        Float64(count) * real_quotient
                        / (Float64(cap) + real_quotient)
                    )
                    if not Bool(isfinite(estimate)):
                        continue
                    var at = _bounded_integer_hint(estimate, 1, count - 1)
                    for shift in range(-1, 2):
                        var cpu_items = min(count - 1, max(1, at + shift))
                        var predicted = _host_add_split_time(
                            count, cpu_items, cap, cpu_per, gpu_per,
                            profile.gpu_launch_us, profile.join_us,
                        )
                        if Bool(isfinite(predicted)) and predicted < best_time:
                            best_kind = 2
                            best_time = predicted
                            best_cpu_items = cpu_items

        # Integer quotient changes can leave the best analytic candidate a
        # few items away from a local discrete minimum. This bounded final
        # polish is applied only to a predicted HYBRID winner in PREPARE.
        # It never runs inside a Tensor kernel or allocates a Tensor buffer.
        if best_kind == 2:
            var neighborhood_anchor = best_cpu_items
            for local_offset in range(-4, 5):
                var cpu_items = neighborhood_anchor + local_offset
                if cpu_items <= 0 or cpu_items >= count:
                    continue
                var predicted = _host_add_split_time(
                    count, cpu_items, cap, cpu_per, gpu_per,
                    profile.gpu_launch_us, profile.join_us,
                )
                if Bool(isfinite(predicted)) and predicted < best_time:
                    best_time = predicted
                    best_cpu_items = cpu_items
    if best_kind == 2:
        return ExecutionPlan.checked(
            ExecutionPlacement.HYBRID_SPLIT, count, 0, best_cpu_items,
            best_cpu_items, count, (count - best_cpu_items) * 12,
            variant, 2, 0,
        )
    if best_kind == 1:
        return ExecutionPlan.checked(
            ExecutionPlacement.GPU_ONLY, count, 0, 0, 0, count,
            count * 12, variant, 1, 0,
        )
    return ExecutionPlan.checked(
        ExecutionPlacement.CPU_ONLY, count, 0, count, 0, 0,
        0, variant, 0, 0,
    )


# Explicit, offline-selected split of strict-FP32 dense rank-two GEMM.
# The work item is one C row; the full B matrix is uploaded ONCE, not once
# per row. The profile-based automatic choice remains a separate obligation.
def build_host_gemm_rows_split_plan(
    m: Int, k: Int, n: Int, cpu_rows: Int, variant: Int,
) -> TensorResult[ExecutionPlan]:
    if m <= 1 or k <= 0 or n <= 0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    if cpu_rows <= 0 or cpu_rows >= m or variant != 14:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var maximum_elements = Int.MAX // 4
    if m > maximum_elements // k or k > maximum_elements // n:
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    if m > maximum_elements // n:
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    var gpu_rows = m - cpu_rows
    var upload_a = gpu_rows * k
    var upload_b = k * n
    var download_c = gpu_rows * n
    if upload_a > maximum_elements - upload_b:
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    var total = upload_a + upload_b
    if download_c > maximum_elements - total:
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    var actual_bytes = (total + download_c) * 4
    # The GEMM lowerer transfers exactly A[gpu_rows,K], shared B[K,N]
    # and C[gpu_rows,N]. It does not consume two inputs per output row.
    var built = ExecutionPlan.checked(
        ExecutionPlacement.HYBRID_SPLIT, m,
        0, cpu_rows, cpu_rows, m,
        actual_bytes, variant, 2, 0,
    )
    if not built.is_ok():
        return built
    return built.payload.value().checked_host_gemm_rows_split_snapshot(k, n)


@fieldwise_init
struct HybridGemmTuningProfile(Copyable):
    # Offline coefficients for THIS shape bucket, target and math policy.
    # Never estimated or benchmarked inside an operator invocation.
    var cpu_fp32_flops_per_us: Float64
    var gpu_fp32_flops_per_us: Float64
    var h2d_bytes_per_us: Float64
    var d2h_bytes_per_us: Float64
    var host_stage_bytes_per_us: Float64
    var host_result_bytes_per_us: Float64
    var gpu_launch_us: Float64
    var completion_us: Float64


def validate_hybrid_gemm_profile(profile: HybridGemmTuningProfile) -> TensorStatus:
    if not Bool(isfinite(profile.cpu_fp32_flops_per_us)) or profile.cpu_fp32_flops_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.gpu_fp32_flops_per_us)) or profile.gpu_fp32_flops_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.h2d_bytes_per_us)) or profile.h2d_bytes_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.d2h_bytes_per_us)) or profile.d2h_bytes_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.host_stage_bytes_per_us)) or profile.host_stage_bytes_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.host_result_bytes_per_us)) or profile.host_result_bytes_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.gpu_launch_us)) or profile.gpu_launch_us < 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.completion_us)) or profile.completion_us < 0.0:
        return TensorStatus.INVALID_LAYOUT
    return TensorStatus.OK


def choose_host_gemm_rows_split_plan_for_capacity(
    m: Int, k: Int, n: Int, gpu_row_capacity: Int,
    profile: HybridGemmTuningProfile, gpu_enabled: Bool,
) -> TensorResult[ExecutionPlan]:
    # A conservative CPU_ONLY vs GPU/CPU split choice. The GPU keeps ONE
    # copy of the entire B[K,N] matrix, not one copy per output row.
    # AOT variant 14 is the only authenticated-by-contract flat FP32 SIMT
    # ABI currently described in this source release. The actual binary
    # still requires provenance verification by a trusted offline loader.
    var valid = validate_hybrid_gemm_profile(profile)
    if valid != TensorStatus.OK:
        return TensorResult[ExecutionPlan](valid)
    if m < 0 or k < 0 or n < 0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    var max_elems = Int.MAX // 4
    if (k > 0 and m > max_elems // k
        or n > 0 and k > max_elems // n
        or n > 0 and m > max_elems // n):
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    if not gpu_enabled or m < 2 or k == 0 or n == 0 or gpu_row_capacity <= 0:
        return ExecutionPlan.checked(
            ExecutionPlacement.CPU_ONLY, m, 0, m, 0, 0, 0, 0, 0, 0,
        )
    var shared = k * n
    if k > Int.MAX - n:
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    var per_row_elements = k + n
    var remaining = max_elems - shared
    var maximal_gpu_rows = min(
        min(gpu_row_capacity, m - 1),
        remaining // per_row_elements,
    )
    if maximal_gpu_rows <= 0:
        return ExecutionPlan.checked(
            ExecutionPlacement.CPU_ONLY, m, 0, m, 0, 0, 0, 0, 0, 0,
        )
    var flops_per_row = 2.0 * Float64(k) * Float64(n)
    var cpu_us_per_row = flops_per_row / profile.cpu_fp32_flops_per_us
    var gpu_us_per_row = (
        flops_per_row / profile.gpu_fp32_flops_per_us
        + 4.0 * Float64(k) / profile.h2d_bytes_per_us
        + 4.0 * Float64(n) / profile.d2h_bytes_per_us
    )
    var gpu_setup = (
        profile.gpu_launch_us + profile.completion_us
        + 4.0 * Float64(shared) / profile.h2d_bytes_per_us
    )
    # Host copies into pinned A/B staging happen BEFORE CPU/GPU overlap;
    # writing back host C happens AFTER the completion event. Neither cost
    # may be disguised as H2D/D2H transfer or silently omitted.
    var serial_setup = 4.0 * Float64(shared) / profile.host_stage_bytes_per_us
    var serial_us_per_gpu_row = (
        4.0 * Float64(k) / profile.host_stage_bytes_per_us
        + 4.0 * Float64(n) / profile.host_result_bytes_per_us
    )
    var cpu_only_time = Float64(m) * cpu_us_per_row
    if (not Bool(isfinite(cpu_us_per_row))
        or not Bool(isfinite(gpu_us_per_row))
        or not Bool(isfinite(gpu_setup))
        or not Bool(isfinite(serial_setup))
        or not Bool(isfinite(serial_us_per_gpu_row))
        or not Bool(isfinite(cpu_only_time))):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var denominator = cpu_us_per_row + gpu_us_per_row
    if denominator <= 0.0 or not Bool(isfinite(denominator)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    # Minimize max(CPU work, GPU DMA+work), a convex piecewise-linear cost.
    # Its optimum is at a capacity boundary or immediately around the
    # intersection; only four candidates are needed, even for huge M.
    var intersection = (cpu_only_time - gpu_setup) / denominator
    if not Bool(isfinite(intersection)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    intersection = max(1.0, min(Float64(maximal_gpu_rows), intersection))
    var candidates = Array[Int, 4](fill=1)
    candidates[0] = 1
    candidates[1] = maximal_gpu_rows
    candidates[2] = _bounded_integer_hint(intersection, 1, maximal_gpu_rows)
    candidates[3] = min(maximal_gpu_rows, candidates[2] + 1)
    var best_gpu_rows = 0
    var best_us = cpu_only_time
    for j in range(4):
        var gpu_rows = candidates[j]
        var cpu_time = Float64(m - gpu_rows) * cpu_us_per_row
        var gpu_time = gpu_setup + Float64(gpu_rows) * gpu_us_per_row
        var split_time = (
            serial_setup + Float64(gpu_rows) * serial_us_per_gpu_row
            + max(cpu_time, gpu_time)
        )
        if Bool(isfinite(split_time)) and split_time < best_us:
            best_gpu_rows = gpu_rows
            best_us = split_time
    if best_gpu_rows == 0:
        return ExecutionPlan.checked(
            ExecutionPlacement.CPU_ONLY, m, 0, m, 0, 0, 0, 0, 0, 0,
        )
    return build_host_gemm_rows_split_plan(
        m, k, n, m - best_gpu_rows, 14,
    )


@fieldwise_init
struct HybridRowReductionTuningProfile(Copyable):
    var cpu_rows_per_us: Float64
    var gpu_rows_per_us: Float64
    var h2d_bytes_per_us: Float64
    var d2h_bytes_per_us: Float64
    var host_stage_bytes_per_us: Float64
    var host_result_bytes_per_us: Float64
    var gpu_launch_us: Float64
    var completion_us: Float64


def validate_hybrid_row_reduction_profile(
    profile: HybridRowReductionTuningProfile,
) -> TensorStatus:
    if not Bool(isfinite(profile.cpu_rows_per_us)) or profile.cpu_rows_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.gpu_rows_per_us)) or profile.gpu_rows_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.h2d_bytes_per_us)) or profile.h2d_bytes_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.d2h_bytes_per_us)) or profile.d2h_bytes_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.host_stage_bytes_per_us)) or profile.host_stage_bytes_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.host_result_bytes_per_us)) or profile.host_result_bytes_per_us <= 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.gpu_launch_us)) or profile.gpu_launch_us < 0.0:
        return TensorStatus.INVALID_LAYOUT
    if not Bool(isfinite(profile.completion_us)) or profile.completion_us < 0.0:
        return TensorStatus.INVALID_LAYOUT
    return TensorStatus.OK


def choose_host_reduce_rows_split_plan_for_capacity(
    rows: Int, columns: Int, capacity: Int,
    operation: Int, profile: HybridRowReductionTuningProfile,
    gpu_enabled: Bool,
) -> TensorResult[ExecutionPlan]:
    # Throughput units are output ROWS/us at the exact operation/shape bucket.
    # H2D, D2H, host staging and final host-result copies are independent.
    var valid = validate_hybrid_row_reduction_profile(profile)
    if valid != TensorStatus.OK:
        return TensorResult[ExecutionPlan](valid)
    if rows < 0 or columns < 0 or operation < 0 or operation > 4:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    # MIN/MAX/MEAN are undefined on nonempty collections of empty rows.
    # A CPU_ONLY fallback must not turn an invalid reduction into a valid plan.
    if rows > 0 and columns == 0 and operation >= 2:
        return TensorResult[ExecutionPlan](TensorStatus.SIZE_MISMATCH)
    if rows > Int.MAX // 4:
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    if columns > 0 and rows > (Int.MAX // 4) // columns:
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    var variant = 30 + operation
    if (not gpu_enabled or rows < 2 or columns == 0
        or capacity <= 0):
        return ExecutionPlan.checked(
            ExecutionPlacement.CPU_ONLY, rows, 0, rows, 0, 0,
            0, 0, 0, 0,
        )
    var row_transport = columns + 1
    var bound = min(min(rows - 1, capacity), (Int.MAX // 4) // row_transport)
    if bound <= 0:
        return ExecutionPlan.checked(
            ExecutionPlacement.CPU_ONLY, rows, 0, rows, 0, 0,
            0, 0, 0, 0,
        )
    var cpu_time_per_row = 1.0 / profile.cpu_rows_per_us
    var gpu_time_per_row = (
        1.0 / profile.gpu_rows_per_us
        + 4.0 * Float64(columns) / profile.h2d_bytes_per_us
        + 4.0 / profile.d2h_bytes_per_us
    )
    var serial_time_per_gpu_row = (
        4.0 * Float64(columns) / profile.host_stage_bytes_per_us
        + 4.0 / profile.host_result_bytes_per_us
    )
    var setup_time = profile.gpu_launch_us + profile.completion_us
    var cpu_only_time = Float64(rows) * cpu_time_per_row
    if (not Bool(isfinite(cpu_only_time))
        or not Bool(isfinite(gpu_time_per_row))
        or not Bool(isfinite(serial_time_per_gpu_row))
        or not Bool(isfinite(setup_time))):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var intersect = (
        cpu_only_time - setup_time
    ) / (cpu_time_per_row + gpu_time_per_row)
    if not Bool(isfinite(intersect)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    intersect = max(1.0, min(Float64(bound), intersect))
    var options = Array[Int, 4](fill=1)
    options[0] = 1
    options[1] = bound
    options[2] = _bounded_integer_hint(intersect, 1, bound)
    options[3] = min(bound, options[2] + 1)
    var chosen_gpu_rows = 0
    var best = cpu_only_time
    for j in range(4):
        var gpu_rows = options[j]
        var duration = (
            Float64(gpu_rows) * serial_time_per_gpu_row
            + max(
                Float64(rows - gpu_rows) * cpu_time_per_row,
                setup_time + Float64(gpu_rows) * gpu_time_per_row,
            )
        )
        if Bool(isfinite(duration)) and duration < best:
            best = duration
            chosen_gpu_rows = gpu_rows
    if chosen_gpu_rows == 0:
        return ExecutionPlan.checked(
            ExecutionPlacement.CPU_ONLY, rows, 0, rows, 0, 0,
            0, 0, 0, 0,
        )
    return build_host_reduce_rows_split_plan(
        rows, columns, chosen_gpu_rows, operation, False,
    )


# AOT lowering for host-resident FP32 elementwise binary kernels. Every output write must
# follow the snapshots of ALL affected input regions. This remains correct
# even if caller-provided host spans have overlapping underlying ranges.
# Resource IDs describe logical buffers. Device and staging owners are
# distinct allocations; original host inputs and output may alias, hence
# their snapshot phases are ordered before either output writer.
# This plan is a validated schedule, not yet a general graph executor.
def build_host_binary_split_schedule(plan: ExecutionPlan) -> TensorResult[BoundStageSchedule[10, 11]]:
    var verified = plan.checked_snapshot()
    if not verified.is_ok():
        return TensorResult[BoundStageSchedule[10, 11]](verified.status)
    var selected = verified.payload.take()
    if selected.output_device != 0:
        return TensorResult[BoundStageSchedule[10, 11]](TensorStatus.NOT_REPRESENTABLE)
    if selected.placement != ExecutionPlacement.HYBRID_SPLIT and selected.placement != ExecutionPlacement.HYBRID_CPU_ASSIST:
        return TensorResult[BoundStageSchedule[10, 11]](TensorStatus.NOT_REPRESENTABLE)
    # Both contiguous partition orders are legal; physical access ranges
    # derive from the explicit plan boundaries rather than an ordering guess.
    var cpu_first = selected.cpu_begin == 0 and selected.cpu_end == selected.gpu_begin and selected.gpu_end == selected.total_items
    var gpu_first = selected.gpu_begin == 0 and selected.gpu_end == selected.cpu_begin and selected.cpu_end == selected.total_items
    if selected.placement == ExecutionPlacement.HYBRID_CPU_ASSIST:
        if not gpu_first:
            return TensorResult[BoundStageSchedule[10, 11]](TensorStatus.INVALID_LAYOUT)
    elif not cpu_first and not gpu_first:
        return TensorResult[BoundStageSchedule[10, 11]](TensorStatus.INVALID_LAYOUT)
    var gpu_items = selected.gpu_end - selected.gpu_begin
    if gpu_items > Int.MAX // 12:
        return TensorResult[BoundStageSchedule[10, 11]](TensorStatus.OVERFLOW)
    if selected.transfer_bytes < gpu_items * 12:
        return TensorResult[BoundStageSchedule[10, 11]](TensorStatus.SIZE_MISMATCH)

    # Host inputs A/B and output: 0/1/2. GPU-slice staging: 3/4.
    # Device buffers: input A/B=5/6, output=7. Device-result staging=8.
    # CPU-slice snapshots: 9/10. Space 2 denotes staging, pinned where DMA
    # is used and ordinary host allocations for CPU-local snapshots.
    var disabled = ExecutionAccess.none()
    var unused = ExecutionStage(0, ExecutionStageKind.CPU_PREPARE, 0, disabled, disabled, disabled, 0)
    var stages = Array[ExecutionStage, 10](fill=unused)
    var g0 = selected.gpu_begin
    var g1 = selected.gpu_end
    var c0 = selected.cpu_begin
    var c1 = selected.cpu_end
    var cpu_items = c1 - c0
    # Host spans use global logical coordinates. All independent GPU and
    # staging allocations use their own zero-based physical offsets.

    # Copy GPU-owned inputs to independent staging before output writes.
    stages[0] = ExecutionStage(
        0, ExecutionStageKind.CPU_PREPARE, 0,
        ExecutionAccess(0, 0, g0, g1), disabled, ExecutionAccess(2, 3, 0, gpu_items), 0,
    )
    stages[1] = ExecutionStage(
        1, ExecutionStageKind.CPU_PREPARE, 0,
        ExecutionAccess(0, 1, g0, g1), disabled, ExecutionAccess(2, 4, 0, gpu_items), 0,
    )
    # CPU-owned inputs are also snapshotted so shifted/overlapping host
    # views cannot be corrupted by a write earlier in the CPU loop.
    stages[2] = ExecutionStage(
        2, ExecutionStageKind.CPU_PREPARE, 0,
        ExecutionAccess(0, 0, c0, c1), disabled, ExecutionAccess(2, 9, 0, cpu_items), 0,
    )
    stages[3] = ExecutionStage(
        3, ExecutionStageKind.CPU_PREPARE, 0,
        ExecutionAccess(0, 1, c0, c1), disabled, ExecutionAccess(2, 10, 0, cpu_items), 0,
    )
    # The H2D uploads depend on their corresponding preparation stages.
    stages[4] = ExecutionStage(
        4, ExecutionStageKind.H2D, 1,
        ExecutionAccess(2, 3, 0, gpu_items), disabled, ExecutionAccess(1, 5, 0, gpu_items), 2,
    )
    stages[5] = ExecutionStage(
        5, ExecutionStageKind.H2D, 2,
        ExecutionAccess(2, 4, 0, gpu_items), disabled, ExecutionAccess(1, 6, 0, gpu_items), 2,
    )
    stages[6] = ExecutionStage(
        6, ExecutionStageKind.GPU_KERNEL, 48,
        ExecutionAccess(1, 5, 0, gpu_items), ExecutionAccess(1, 6, 0, gpu_items),
        ExecutionAccess(1, 7, 0, gpu_items), 1,
    )
    # Both GPU and CPU source snapshots must precede the first CPU write.
    stages[7] = ExecutionStage(
        7, ExecutionStageKind.CPU_COMPUTE, 15,
        ExecutionAccess(2, 9, 0, cpu_items), ExecutionAccess(2, 10, 0, cpu_items),
        ExecutionAccess(0, 2, c0, c1), 0,
    )
    stages[8] = ExecutionStage(
        8, ExecutionStageKind.D2H, 64,
        ExecutionAccess(1, 7, 0, gpu_items), disabled, ExecutionAccess(2, 8, 0, gpu_items), 2,
    )
    # The final join copies only GPU-owned output and waits for CPU work.
    stages[9] = ExecutionStage(
        9, ExecutionStageKind.HOST_JOIN, 384,
        ExecutionAccess(2, 8, 0, gpu_items), disabled, ExecutionAccess(0, 2, g0, g1), 3,
    )
    var built = CheckedStageSchedule[10].checked(stages, 10, selected.total_items)
    if not built.is_ok():
        return TensorResult[BoundStageSchedule[10, 11]](built.status)
    var coverage = built.payload.value().checked_host_output_coverage(0, selected.total_items)
    if coverage != TensorStatus.OK:
        return TensorResult[BoundStageSchedule[10, 11]](coverage)
    var absent = ExecutionBufferBounds(-1, -1, 0)
    var buffers = Array[ExecutionBufferBounds, 11](fill=absent)
    buffers[0] = ExecutionBufferBounds(0, 0, selected.total_items)
    buffers[1] = ExecutionBufferBounds(0, 1, selected.total_items)
    buffers[2] = ExecutionBufferBounds(0, 2, selected.total_items)
    buffers[3] = ExecutionBufferBounds(2, 3, gpu_items)
    buffers[4] = ExecutionBufferBounds(2, 4, gpu_items)
    buffers[5] = ExecutionBufferBounds(1, 5, gpu_items)
    buffers[6] = ExecutionBufferBounds(1, 6, gpu_items)
    buffers[7] = ExecutionBufferBounds(1, 7, gpu_items)
    buffers[8] = ExecutionBufferBounds(2, 8, gpu_items)
    buffers[9] = ExecutionBufferBounds(2, 9, cpu_items)
    buffers[10] = ExecutionBufferBounds(2, 10, cpu_items)
    # Return exactly the schedule snapshot that passed both the dependency
    # DAG proof and the physical allocation-bound proof.
    return bind_schedule_buffers_checked(built.payload.value(), buffers)


# Source compatibility for callers of the pre-generalized add-only schedule.
def build_host_add_split_schedule(
    plan: ExecutionPlan,
) -> TensorResult[BoundStageSchedule[10, 11]]:
    return build_host_binary_split_schedule(plan)


# The profile MUST be calibrated for the specified FP32 operation and target.
# Add, subtract, multiply, divide, maximum and minimum can have different
# throughput. All six use two H2D transfers and one D2H host-result transfer.
# The caller must provide a per-operation calibrated immutable cost profile.
def choose_host_binary_plan_for_capacity[operation: Int](
    count: Int, gpu_chunk_capacity: Int, profile: HybridTuningProfile,
    variant: Int, gpu_enabled: Bool,
) -> TensorResult[ExecutionPlan]:
    comptime assert operation >= 0 and operation <= 5, "FP32 opcode must be 0..5"
    return choose_host_add_plan_for_capacity(
        count, gpu_chunk_capacity, profile, variant, gpu_enabled
    )


# A narrowly scoped CPU-assist lowering: GPU processes [0, n-tail) while
# CPU processes the small contiguous tail [n-tail, n). The SAME FP32 binary
# schedule and host join as split are used. It is never selected merely to
# activate both processors: a calibrated end-to-end cost must beat CPU_ONLY
# and GPU_ONLY. No per-call benchmarking or JIT specialization occurs.
def choose_host_cpu_tail_assist_plan_for_capacity(
    count: Int, cpu_tail_items: Int, gpu_chunk_capacity: Int,
    profile: HybridTuningProfile, variant: Int, gpu_enabled: Bool,
) -> TensorResult[ExecutionPlan]:
    var valid = validate_hybrid_tuning_profile(profile)
    if valid != TensorStatus.OK:
        return TensorResult[ExecutionPlan](valid)
    if count < 0 or variant < 0 or cpu_tail_items < 0 or cpu_tail_items > count:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    var cpu_cost = Float64(count) / profile.cpu_elems_per_us
    if not Bool(isfinite(cpu_cost)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    if not gpu_enabled or count == 0 or count > Int.MAX // 12:
        return ExecutionPlan.checked(
            ExecutionPlacement.CPU_ONLY, count, 0, count, 0, 0,
            0, variant, 0, 0,
        )
    if gpu_chunk_capacity <= 0 or profile.gpu_elems_per_us <= 0.0 or profile.h2d_bytes_per_us <= 0.0:
        return ExecutionPlan.checked(
            ExecutionPlacement.CPU_ONLY, count, 0, count, 0, 0,
            0, variant, 0, 0,
        )
    var cap = min(count, gpu_chunk_capacity)
    var cpu_per = 1.0 / profile.cpu_elems_per_us
    var gpu_per = 12.0 / profile.h2d_bytes_per_us + 1.0 / profile.gpu_elems_per_us
    if not Bool(isfinite(cpu_per)) or not Bool(isfinite(gpu_per)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var gpu_chunks = count // cap
    if count % cap != 0:
        gpu_chunks += 1
    var gpu_only_cost = Float64(gpu_chunks) * profile.gpu_launch_us + Float64(count) * gpu_per
    if not Bool(isfinite(gpu_only_cost)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var best_single_gpu = gpu_only_cost < cpu_cost
    var best_single_cost = cpu_cost
    if best_single_gpu:
        best_single_cost = gpu_only_cost
    # A CPU-dominant partition is not a CPU assist mapping. Such a layout
    # requires a separately selected HYBRID_SPLIT plan instead.
    if cpu_tail_items > 0 and cpu_tail_items < count and cpu_tail_items <= count - cpu_tail_items:
        var gpu_prefix = count - cpu_tail_items
        var tail_cost = _host_add_split_time(
            count, cpu_tail_items, cap, cpu_per, gpu_per,
            profile.gpu_launch_us, profile.join_us,
        )
        if Bool(isfinite(tail_cost)) and tail_cost < best_single_cost:
            return ExecutionPlan.checked(
                ExecutionPlacement.HYBRID_CPU_ASSIST, count,
                gpu_prefix, count, 0, gpu_prefix, gpu_prefix * 12,
                variant, 2, 0,
            )
    if best_single_gpu:
        return ExecutionPlan.checked(
            ExecutionPlacement.GPU_ONLY, count, 0, 0, 0, count,
            count * 12, variant, 1, 0,
        )
    return ExecutionPlan.checked(
        ExecutionPlacement.CPU_ONLY, count, 0, count, 0, 0,
        0, variant, 0, 0,
    )


# A single immutable two-slot AOT dataflow template. The two independent
# chunks use distinct pinned/device allocations. Later chunks reuse exactly
# those slots only AFTER their GPU+D2H completion events are synchronized.
# This DAG is intentionally specific to an out-of-place FP32 binary pipeline;
# it must not be reused for matmul, reductions, or in-place graph operations.
def build_host_binary_pipeline_window_schedule(
    plan: ExecutionPlan, capacity: Int,
) -> TensorResult[BoundStageSchedule[14, 15]]:
    var verified = plan.checked_snapshot()
    if not verified.is_ok():
        return TensorResult[BoundStageSchedule[14, 15]](verified.status)
    var selected = verified.payload.take()
    if selected.placement != ExecutionPlacement.HYBRID_PIPELINE:
        return TensorResult[BoundStageSchedule[14, 15]](TensorStatus.DEVICE_MISMATCH)
    if capacity <= 0 or capacity > Int.MAX // 2:
        return TensorResult[BoundStageSchedule[14, 15]](TensorStatus.OVERFLOW)
    if capacity > selected.total_items // 2:
        return TensorResult[BoundStageSchedule[14, 15]](TensorStatus.NOT_REPRESENTABLE)
    # Only the first two windows are in this fixed 14-stage DAG. Later full
    # windows reuse its two slots after their completion fences; a short final
    # window has a separate exactly-sized allocation and completion fence.
    # The staging/transfer ABI never reads capacity-sized uninitialized tails.

    var disabled = ExecutionAccess.none()
    var placeholder = ExecutionStage(0, ExecutionStageKind.CPU_PREPARE, 0, disabled, disabled, disabled, 0)
    var stages = Array[ExecutionStage, 14](fill=placeholder)
    # Snapshot sources 0,1 and disjoint final host output 2. Each slot has
    # unique stage buffers 3+6*q .. 8+6*q, disjoint by ownership.
    for q in range(2):
        var base = q * 7
        var off = q * capacity
        var end = off + capacity
        var sid = 3 + q * 6
        stages[base] = ExecutionStage(
            base, ExecutionStageKind.CPU_PREPARE, 0,
            ExecutionAccess(0, 0, off, end), disabled,
            ExecutionAccess(2, sid, 0, capacity), 0,
        )
        stages[base+1] = ExecutionStage(
            base+1, ExecutionStageKind.CPU_PREPARE, 0,
            ExecutionAccess(0, 1, off, end), disabled,
            ExecutionAccess(2, sid+1, 0, capacity), 0,
        )
        stages[base+2] = ExecutionStage(
            base+2, ExecutionStageKind.H2D, Int(1) << base,
            ExecutionAccess(2, sid, 0, capacity), disabled,
            ExecutionAccess(1, sid+2, 0, capacity), 2,
        )
        stages[base+3] = ExecutionStage(
            base+3, ExecutionStageKind.H2D, Int(1) << (base+1),
            ExecutionAccess(2, sid+1, 0, capacity), disabled,
            ExecutionAccess(1, sid+3, 0, capacity), 2,
        )
        stages[base+4] = ExecutionStage(
            base+4, ExecutionStageKind.GPU_KERNEL,
            (Int(1) << (base+2)) | (Int(1) << (base+3)),
            ExecutionAccess(1, sid+2, 0, capacity),
            ExecutionAccess(1, sid+3, 0, capacity),
            ExecutionAccess(1, sid+4, 0, capacity), 1,
        )
        stages[base+5] = ExecutionStage(
            base+5, ExecutionStageKind.D2H, Int(1) << (base+4),
            ExecutionAccess(1, sid+4, 0, capacity), disabled,
            ExecutionAccess(2, sid+5, 0, capacity), 2,
        )
        stages[base+6] = ExecutionStage(
            base+6, ExecutionStageKind.HOST_JOIN, Int(1) << (base+5),
            ExecutionAccess(2, sid+5, 0, capacity), disabled,
            ExecutionAccess(0, 2, off, end), 3,
        )
    var checked = CheckedStageSchedule[14].checked(stages, 14, selected.total_items)
    if not checked.is_ok():
        return TensorResult[BoundStageSchedule[14, 15]](checked.status)
    # This reusable AOT template covers exactly the first two staging slots.
    # Later windows are event-fenced by the executor, not represented as
    # additional writers in this 14-stage prefix.
    var coverage = checked.payload.value().checked_host_output_coverage(0, 2 * capacity)
    if coverage != TensorStatus.OK:
        return TensorResult[BoundStageSchedule[14, 15]](coverage)
    var absent = ExecutionBufferBounds(-1, -1, 0)
    var buffers = Array[ExecutionBufferBounds, 15](fill=absent)
    buffers[0] = ExecutionBufferBounds(0, 0, selected.total_items)
    buffers[1] = ExecutionBufferBounds(0, 1, selected.total_items)
    buffers[2] = ExecutionBufferBounds(0, 2, selected.total_items)
    for q in range(2):
        var sid = 3 + q * 6
        buffers[sid] = ExecutionBufferBounds(2, sid, capacity)
        buffers[sid+1] = ExecutionBufferBounds(2, sid+1, capacity)
        buffers[sid+2] = ExecutionBufferBounds(1, sid+2, capacity)
        buffers[sid+3] = ExecutionBufferBounds(1, sid+3, capacity)
        buffers[sid+4] = ExecutionBufferBounds(1, sid+4, capacity)
        buffers[sid+5] = ExecutionBufferBounds(2, sid+5, capacity)
    # A physical-bound proof cannot remain detached from the schedule value.
    return bind_schedule_buffers_checked(checked.payload.value(), buffers)


# Optional offline-profile selection, not an unconditional GPU preference.
# The CPU preparation throughput explicitly models two staged inputs/element.
# A pipeline is chosen only if its cost beats both CPU-only and the standard
# sequential host->device->host GPU placement under the target's profile.
def choose_host_binary_pipeline_plan_for_capacity[operation: Int](
    count: Int, capacity: Int, profile: HybridTuningProfile,
    cpu_staging_elems_per_us: Float64, variant: Int, gpu_enabled: Bool,
) -> TensorResult[ExecutionPlan]:
    comptime assert operation >= 0 and operation <= 5, "invalid FP32 opcode"
    var valid = validate_hybrid_tuning_profile(profile)
    if valid != TensorStatus.OK:
        return TensorResult[ExecutionPlan](valid)
    if count < 0 or variant < 0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    if not Bool(isfinite(cpu_staging_elems_per_us)) or cpu_staging_elems_per_us <= 0.0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    if not gpu_enabled or capacity <= 0 or count < 2 or count > Int.MAX // 12:
        return choose_host_binary_plan_for_capacity[operation](count, capacity, profile, variant, gpu_enabled)
    if capacity > count // 2:
        return choose_host_binary_plan_for_capacity[operation](count, capacity, profile, variant, gpu_enabled)
    if profile.gpu_elems_per_us <= 0.0 or profile.h2d_bytes_per_us <= 0.0:
        return choose_host_binary_plan_for_capacity[operation](count, capacity, profile, variant, gpu_enabled)
    var batches = count // capacity
    if count % capacity != 0:
        batches += 1
    var cpu_time = Float64(count) / profile.cpu_elems_per_us
    var gpu_time = (
        Float64(batches) * profile.gpu_launch_us
        + Float64(count) * (
            12.0 / profile.h2d_bytes_per_us
            + 1.0 / profile.gpu_elems_per_us
        )
    )
    var prepare_time = Float64(count) * 2.0 / cpu_staging_elems_per_us
    var first_prepare = Float64(capacity) * 2.0 / cpu_staging_elems_per_us
    # A deliberately conservative estimate of achievable overlap. Per-chunk
    # allocations and full-input snapshots must be included in offline rates.
    var pipeline_time = first_prepare + max(gpu_time, prepare_time) + profile.join_us
    if Bool(isfinite(pipeline_time)) and pipeline_time < cpu_time and pipeline_time < gpu_time + prepare_time + profile.join_us:
        return ExecutionPlan.checked(
            ExecutionPlacement.HYBRID_PIPELINE, count, 0, 0, 0, count,
            count * 12, variant, 2, 0,
        )
    return choose_host_binary_plan_for_capacity[operation](count, capacity, profile, variant, gpu_enabled)

# One canonical ExecutionPlan constructor for the bounded CPU/GPU row-reduce
# mapping. Unlike an elementwise split, each GPU work item consumes `columns`
# input FP32 values and produces one FP32 output. Its transfer budget MUST
# include BOTH the row-major H2D input and the row-result D2H output.
# This is an explicit AOT plan constructor, NOT a runtime cost-model guess.
def build_host_reduce_rows_split_plan(
    rows: Int, columns: Int, gpu_rows: Int, operation: Int,
    gpu_first: Bool,
) -> TensorResult[ExecutionPlan]:
    if operation < 0 or operation > 4:
        return TensorResult[ExecutionPlan](TensorStatus.UNSUPPORTED)
    if rows <= 1 or columns <= 0 or gpu_rows <= 0 or gpu_rows >= rows:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    if rows > Int.MAX // 4 or columns > (Int.MAX // 4) // rows:
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    # Each GPU row consumes `columns` input values and produces one output;
    # a single-column reduction requires exactly two FP32 transfers.
    var transfer_elements_per_row = columns + 1
    if gpu_rows > (Int.MAX // 4) // transfer_elements_per_row:
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    var transfer_bytes = gpu_rows * transfer_elements_per_row * 4
    var cpu_rows = rows - gpu_rows
    var cpu_begin = 0
    var cpu_end = cpu_rows
    var gpu_begin = cpu_rows
    var gpu_end = rows
    if gpu_first:
        gpu_begin = 0
        gpu_end = gpu_rows
        cpu_begin = gpu_rows
        cpu_end = rows
    return ExecutionPlan.checked(
        ExecutionPlacement.HYBRID_SPLIT, rows,
        cpu_begin, cpu_end, gpu_begin, gpu_end,
        transfer_bytes, 30 + operation, 2, 0,
    )

# Dedicated host-resident FP32 unary placement. The same finite AOT artifact
# identity (20+opcode) is carried by the canonical ExecutionPlan. It requires
# one H2D input and one D2H output, unlike the 12-byte binary contract.
# No runtime tuning or GPU work is performed by this offline-profile selector.
def choose_host_unary_plan_for_capacity(
    count: Int, gpu_chunk_capacity: Int, operation: Int,
    profile: HybridTuningProfile, gpu_enabled: Bool,
) -> TensorResult[ExecutionPlan]:
    if operation < 0 or operation > 8 or count < 0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    if count > Int.MAX // 4:
        return TensorResult[ExecutionPlan](TensorStatus.OVERFLOW)
    var valid = validate_hybrid_tuning_profile(profile)
    if valid != TensorStatus.OK:
        return TensorResult[ExecutionPlan](valid)
    var variant = 20 + operation
    var cpu_plan = ExecutionPlan.checked(
        ExecutionPlacement.CPU_ONLY, count, 0, count, 0, 0,
        0, variant, 0, 0,
    )
    if not gpu_enabled or count <= 1 or gpu_chunk_capacity <= 0:
        return cpu_plan
    if profile.gpu_elems_per_us <= 0.0 or profile.h2d_bytes_per_us <= 0.0:
        return cpu_plan
    if count > Int.MAX // 8:
        # CPU-only is legal even when the aggregate GPU transfer is not.
        return cpu_plan
    var cap = min(count, gpu_chunk_capacity)
    var cpu_per = 1.0 / profile.cpu_elems_per_us
    var gpu_per = 8.0 / profile.h2d_bytes_per_us + 1.0 / profile.gpu_elems_per_us
    if not Bool(isfinite(cpu_per)) or not Bool(isfinite(gpu_per)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var full_chunks = count // cap
    if count % cap != 0:
        full_chunks += 1
    var cpu_cost = Float64(count) * cpu_per
    var gpu_cost = Float64(full_chunks) * profile.gpu_launch_us + Float64(count) * gpu_per
    if not Bool(isfinite(cpu_cost)) or not Bool(isfinite(gpu_cost)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var best_cost = cpu_cost
    var best_placement = 0
    var best_cpu_items = 0
    if gpu_cost < best_cost:
        best_cost = gpu_cost
        best_placement = 1
    # Compute the analytic CPU/GPU crossing and a bounded AOT-compatible
    # candidate set, allowing a staged GPU tail shorter than one buffer.
    var crossing = (
        profile.gpu_launch_us + Float64(count) * gpu_per
    ) / (cpu_per + gpu_per)
    if not Bool(isfinite(crossing)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    crossing = max(1.0, min(Float64(count - 1), crossing))
    # With a short first/last GPU window, the cost is piecewise affine and
    # NOT globally convex: reducing gpu_items by one may also remove an
    # entire launch. The simple one-window crossing alone can miss this
    # discontinuity (e.g. 64 items, capacity 25, costly GPU launch).
    # For q >= 2 windows the first GPU chunk is full. At a fixed q the
    # maximum crosses at cpu_items = (launch + cap*gpu_per) / cpu_per.
    # Evaluate that kink and neighboring launch-count boundaries, as well
    # as the q=1 crossing. This is a FIXED 20-candidate PREPARE-time search;
    # no dynamic kernel dispatch or extra work enters the Tensor hot loop.
    var multi_crossing = (
        profile.gpu_launch_us + Float64(cap) * gpu_per
    ) / cpu_per
    if not Bool(isfinite(multi_crossing)):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
    var multi_cpu = _bounded_integer_hint(multi_crossing, 1, count - 1)
    var gpu_at_kink = count - multi_cpu
    var full_blocks_at_kink = gpu_at_kink // cap
    # full_blocks_at_kink * cap never exceeds count; stay in the integer
    # domain instead of converting very large tensor sizes through Float64.
    var aligned_cpu = count - full_blocks_at_kink * cap
    var next_blocks = min(count // cap, full_blocks_at_kink + 1)
    var next_aligned_cpu = count - next_blocks * cap
    var candidates = Array[Int, 20](fill=1)
    candidates[0] = 1
    candidates[1] = count - 1
    candidates[2] = _bounded_integer_hint(crossing, 1, count - 1)
    candidates[3] = min(count - 1, candidates[2] + 1)
    candidates[4] = max(1, count - cap)
    candidates[5] = min(count - 1, max(1, count - cap + 1))
    candidates[6] = max(1, candidates[2] - 1)
    candidates[7] = min(count - 1, candidates[2] + 2)
    candidates[8] = multi_cpu
    candidates[9] = max(1, multi_cpu - 1)
    candidates[10] = min(count - 1, multi_cpu + 1)
    candidates[11] = min(count - 1, max(1, aligned_cpu))
    candidates[12] = min(count - 1, max(1, aligned_cpu - 1))
    candidates[13] = min(count - 1, max(1, aligned_cpu + 1))
    candidates[14] = min(count - 1, max(1, next_aligned_cpu))
    candidates[15] = min(count - 1, max(1, next_aligned_cpu + 1))
    candidates[16] = max(1, count - cap - 1)
    candidates[17] = max(1, count - 2 * cap)
    candidates[18] = min(count - 1, aligned_cpu + cap)
    candidates[19] = min(count - 1, aligned_cpu + cap + 1)
    for candidate_index in range(20):
        var candidate = candidates[candidate_index]
        var gpu_items = count - candidate
        if gpu_items <= 0:
            continue
        var gpu_chunks = gpu_items // cap
        if gpu_items % cap != 0:
            gpu_chunks += 1
        var first_gpu_items = min(cap, gpu_items)
        var first_gpu_us = profile.gpu_launch_us + Float64(first_gpu_items) * gpu_per
        var remaining_gpu_us = (
            Float64(gpu_chunks - 1) * profile.gpu_launch_us
            + Float64(gpu_items - first_gpu_items) * gpu_per
        )
        # This matches the actual lowering: enqueue first GPU chunk, execute
        # the disjoint CPU partition, join and then enqueue remaining chunks.
        var overlapped_us = (
            max(Float64(candidate) * cpu_per, first_gpu_us)
            + remaining_gpu_us + profile.join_us
        )
        if not Bool(isfinite(overlapped_us)):
            return TensorResult[ExecutionPlan](TensorStatus.INVALID_LAYOUT)
        if overlapped_us < best_cost:
            best_cost = overlapped_us
            best_placement = 2
            best_cpu_items = candidate
    if best_placement == 2:
        return ExecutionPlan.checked(
            ExecutionPlacement.HYBRID_SPLIT, count, 0, best_cpu_items,
            best_cpu_items, count, (count - best_cpu_items) * 8,
            variant, 2, 0,
        )
    if best_placement == 1:
        return ExecutionPlan.checked(
            ExecutionPlacement.GPU_ONLY, count, 0, 0, 0, count,
            count * 8, variant, 1, 0,
        )
    return cpu_plan


# Resource admission is deliberately distinct from latency-based placement.
# A HYBRID_SPLIT plan owns staging only for its GPU partition, not the
# disjoint CPU subset. Validate the complete binary transfer contract before
# touching native resources. Structural caller errors are never disguised as
# harmless CPU-only fallbacks; only actual budget insufficiency falls back.
def admit_host_binary_plan_with_budget(
    plan: ExecutionPlan, capacity: Int, limits: Fp32MemoryLimits,
    fixed_device_bytes: Int, fixed_pinned_host_bytes: Int,
) -> TensorResult[ExecutionPlan]:
    var checked = plan.checked_host_binary_snapshot()
    if not checked.is_ok():
        return checked
    # Malformed budget metadata is invalid even if the latency policy had
    # already selected the CPU path; do not report it as successful admission.
    var budget_valid = limits.validated()
    if budget_valid != TensorStatus.OK:
        return TensorResult[ExecutionPlan](budget_valid)
    if fixed_device_bytes < 0 or fixed_pinned_host_bytes < 0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    var selected = checked.payload.value()
    if selected.placement == ExecutionPlacement.CPU_ONLY:
        return checked
    if capacity <= 0:
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    # One complete resident set for GPU_ONLY/SPLIT/ASSIST; exactly two
    # independently event-owned full sets for the three-stream PIPELINE.
    var wanted = 1
    if selected.placement == ExecutionPlacement.HYBRID_PIPELINE:
        wanted = 2
    var gpu_items = selected.gpu_end - selected.gpu_begin
    if gpu_items <= 0:
        return TensorResult[ExecutionPlan](TensorStatus.HYBRID_PARTITION_INVALID)
    var memory = plan_fp32_binary_windows(
        limits, gpu_items, min(gpu_items, capacity), wanted,
        fixed_device_bytes, fixed_pinned_host_bytes,
        wanted == 1,  # serial event-joined tail replaces a full owner
    )
    if memory.is_ok():
        if memory.payload.value().full_slots >= wanted:
            return checked
        # A small serial window has only an exactly-sized tail slot.
        if wanted == 1 and memory.payload.value().full_slots == 0:
            return checked
    elif memory.status != TensorStatus.NOT_REPRESENTABLE:
        # An invalid budget or arithmetic overflow is NOT a memory shortage.
        return TensorResult[ExecutionPlan](memory.status)
    # A resource-constrained host output remains identical in dtype and
    # shape on CPU, but this is NOT proof of a faster CPU placement.
    return ExecutionPlan.checked(
        ExecutionPlacement.CPU_ONLY, selected.total_items,
        0, selected.total_items, 0, 0, 0,
        selected.aot_variant, 0, 0,
    )


# Optional BUDGET-AWARE offline pipeline selection. The normal optimizer
# remains unchanged; the wrapper refuses a plan exceeding the measured
# GPU VRAM or pinned-host cap and may return CPU_ONLY on insufficiency.
def choose_host_binary_pipeline_with_budget[operation: Int](
    count: Int, capacity: Int, profile: HybridTuningProfile,
    cpu_staging_elems_per_us: Float64, variant: Int, gpu_enabled: Bool,
    limits: Fp32MemoryLimits,
    fixed_device_bytes: Int, fixed_pinned_host_bytes: Int,
) -> TensorResult[ExecutionPlan]:
    comptime if operation < 0 or operation > 5:
        return TensorResult[ExecutionPlan](TensorStatus.UNSUPPORTED)
    var choice = choose_host_binary_pipeline_plan_for_capacity[operation](
        count, capacity, profile, cpu_staging_elems_per_us,
        variant, gpu_enabled,
    )
    if not choice.is_ok():
        return choice
    return admit_host_binary_plan_with_budget(
        choice.payload.value(), capacity, limits,
        fixed_device_bytes, fixed_pinned_host_bytes,
    )


# Opt-in total-RAM-aware PREPARE admission. Pinned host memory is part of
# overall host RAM, not an independent pool. The original public selector
# has no total-RAM telemetry and cannot safely infer it; callers with a
# measured total-RAM envelope must use this overload. No replay-time work.
def admit_host_binary_plan_with_total_ram_budget(
    plan: ExecutionPlan, capacity: Int, limits: Fp32MemoryLimits,
    fixed_device_bytes: Int, fixed_pinned_host_bytes: Int,
    max_total_ram_bytes: Int, fixed_total_ram_bytes: Int,
) -> TensorResult[ExecutionPlan]:
    var checked = plan.checked_host_binary_snapshot()
    if not checked.is_ok():
        return checked
    var valid = limits.validated()
    if valid != TensorStatus.OK:
        return TensorResult[ExecutionPlan](valid)
    if (max_total_ram_bytes <= 0 or fixed_total_ram_bytes < 0
        or fixed_device_bytes < 0 or fixed_pinned_host_bytes < 0
        or fixed_total_ram_bytes < fixed_pinned_host_bytes):
        return TensorResult[ExecutionPlan](TensorStatus.INVALID_SHAPE)
    if (fixed_total_ram_bytes > max_total_ram_bytes
        or fixed_pinned_host_bytes > limits.pinned_host_budget_bytes):
        return TensorResult[ExecutionPlan](TensorStatus.NOT_REPRESENTABLE)
    # Reserve ALL existing pageable/pinned owners in fixed_total_ram_bytes.
    # Additional pinned staging bytes must fit BOTH independent ceilings.
    # These guarded subtractions and addition cannot overflow signed Int.
    var ram_headroom = max_total_ram_bytes - fixed_total_ram_bytes
    var pinned_headroom = limits.pinned_host_budget_bytes - fixed_pinned_host_bytes
    var staging_headroom = min(ram_headroom, pinned_headroom)
    var effective_pinned_budget = fixed_pinned_host_bytes + staging_headroom
    var effective_limits = Fp32MemoryLimits(
        limits.device_budget_bytes, limits.device_reserve_bytes,
        effective_pinned_budget,
    )
    # Reuse the ONE validated GPU-only/split/pipeline admission path. If the
    # new RAM ceiling excludes device staging, return the existing CPU_ONLY
    # fallback with unchanged logical item count and Float32 contract.
    return admit_host_binary_plan_with_budget(
        checked.payload.value(), capacity, effective_limits,
        fixed_device_bytes, fixed_pinned_host_bytes,
    )


def choose_host_binary_pipeline_with_total_ram_budget[operation: Int](
    count: Int, capacity: Int, profile: HybridTuningProfile,
    cpu_staging_elems_per_us: Float64, variant: Int, gpu_enabled: Bool,
    limits: Fp32MemoryLimits,
    fixed_device_bytes: Int, fixed_pinned_host_bytes: Int,
    max_total_ram_bytes: Int, fixed_total_ram_bytes: Int,
) -> TensorResult[ExecutionPlan]:
    # Match the original AOT candidate search and apply the independent RAM
    # hard gate before any native allocation or launch. The old entrypoint
    # retains its ABI and behavior for consumers without total-RAM evidence.
    comptime if operation < 0 or operation > 5:
        return TensorResult[ExecutionPlan](TensorStatus.UNSUPPORTED)
    var choice = choose_host_binary_pipeline_plan_for_capacity[operation](
        count, capacity, profile, cpu_staging_elems_per_us,
        variant, gpu_enabled,
    )
    if not choice.is_ok():
        return choice
    return admit_host_binary_plan_with_total_ram_budget(
        choice.payload.value(), capacity, limits,
        fixed_device_bytes, fixed_pinned_host_bytes,
        max_total_ram_bytes, fixed_total_ram_bytes,
    )
