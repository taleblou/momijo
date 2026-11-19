# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR MEMORY / FP32 MEMORY BUDGETS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Memory and Allocation
# Module:        momijo.tensor_light.memory.fp32_budget
# Source:        src/momijo/tensor_light/memory/fp32_budget.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines FP32 memory budgets in the memory and allocation layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `Fp32MemoryLimits`, `Fp32BinaryWindowBudget`,
#   `plan_fp32_binary_windows`, `plan_fp32_binary_windows_with_total_ram`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Calculate or validate bounded FP32 host/device memory budgets and window
#     sizes.
#   - Expose source-defined interfaces including `Fp32MemoryLimits`,
#     `checked`, `validated`, `Fp32BinaryWindowBudget`.
#   - Reject capacity arithmetic overflow and incompatible RAM/VRAM limits.
#   - Propagate failed validation or allocation results without silently
#     changing memory domains.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate sizes, alignment, pointer access, and memory-domain contracts
#     as applicable.
#   - Reject invalid inputs without asserting unrelated pool ownership or
#     completion semantics.
#
# Performance Objectives:
#   - Keep pointer, size, domain, and alignment checks inexpensive.
#   - Avoid introducing allocations or copies in metadata and validation
#     paths.
#
# Maintainability:
#   - Document the exact allocation, pointer, and address-space contracts
#     exposed here.
#   - Test overflow, invalid alignment, and allocation failures where
#     supported.
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

# Conservative OFFLINE resource admission for fixed-Float32 host/device binary
# windows. This module allocates nothing, never changes a Tensor shape/dtype,
# and never treats Unified Memory oversubscription as available capacity.
# Each window owns two FP32 input buffers plus one FP32 result, on BOTH host
# and device: 12 bytes/element in each memory domain. MAX flat enqueue_copy
# requires exact matching capacities; a final partial window has an exact,
# independently sized owner (not necessarily concurrently resident).
# All cache, graph, workspace, live Tensor and driver reserves must be passed
# as fixed_bytes by the calling binder. No hardware is probed on hot replay.
from ..internal.status import TensorStatus, TensorResult


@fieldwise_init
struct Fp32MemoryLimits(Copyable):
    var device_budget_bytes: Int
    var device_reserve_bytes: Int
    var pinned_host_budget_bytes: Int

    @staticmethod
    def checked(
        device_budget_bytes: Int, device_reserve_bytes: Int,
        pinned_host_budget_bytes: Int,
    ) -> TensorResult[Self]:
        # Zero native-device or pinned-host capacity is a LEGAL CPU-only
        # deployment, not malformed metadata. A GPU request then fails
        # capacity admission and deterministically chooses CPU fallback.
        if (device_budget_bytes < 0 or device_reserve_bytes < 0
            or pinned_host_budget_bytes < 0):
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        if device_reserve_bytes > device_budget_bytes:
            return TensorResult[Self](TensorStatus.NOT_REPRESENTABLE)
        return TensorResult[Self](Self(
            device_budget_bytes, device_reserve_bytes,
            pinned_host_budget_bytes,
        ))

    def validated(self) -> TensorStatus:
        if (self.device_budget_bytes < 0 or self.device_reserve_bytes < 0
            or self.pinned_host_budget_bytes < 0):
            return TensorStatus.INVALID_SHAPE
        if self.device_reserve_bytes > self.device_budget_bytes:
            return TensorStatus.NOT_REPRESENTABLE
        return TensorStatus.OK


@fieldwise_init
struct Fp32BinaryWindowBudget(Copyable):
    # Values are metadata, not an event-ownership or GPU execution proof.
    var full_slots: Int
    # True only when the tail requires a SEPARATE concurrently allocated
    # owner. tail_elements can be nonzero with has_tail_slot=False when the
    # already synchronized full slot is replaced rather than co-resident.
    var has_tail_slot: Bool
    var full_capacity_elements: Int
    var tail_elements: Int
    var reserved_device_bytes: Int
    var reserved_pinned_host_bytes: Int


def plan_fp32_binary_windows(
    limits: Fp32MemoryLimits,
    count: Int, capacity: Int, preferred_full_slots: Int,
    fixed_device_bytes: Int, fixed_pinned_host_bytes: Int,
    reuse_full_slot_for_tail: Bool = False,
) -> TensorResult[Fp32BinaryWindowBudget]:
    # Staging slots count only the additional bounded allocations for this
    # plan. The caller must include EVERY other still-live byte in fixed_*.
    var valid = limits.validated()
    if valid != TensorStatus.OK:
        return TensorResult[Fp32BinaryWindowBudget](valid)
    if (count <= 0 or capacity <= 0 or preferred_full_slots < 1
        or preferred_full_slots > 3 or fixed_device_bytes < 0
        or fixed_pinned_host_bytes < 0):
        return TensorResult[Fp32BinaryWindowBudget](TensorStatus.INVALID_SHAPE)
    # Each *input Tensor* must fit the fixed-FP32 addressable byte range:
    # count * 4 is the representability check, not count * 12. The latter
    # describes the aggregate H2D/H2D/D2H traffic over ALL windows and can
    # overflow Int even though no single buffer is that large. This pure
    # staging planner counts only simultaneously resident WINDOW buffers.
    # ExecutionPlan separately validates its transfer_bytes field and may
    # reject large total traffic that its Int ABI cannot represent.
    if count > Int.MAX // 4:
        return TensorResult[Fp32BinaryWindowBudget](TensorStatus.OVERFLOW)
    # For a short first/only window, allocate exactly count elements, not
    # the possibly enormous nominal maximum capacity. For a full window,
    # capacity <= count. Bound each native three-buffer domain (3*FP32).
    var resident_window_elements = min(count, capacity)
    if resident_window_elements > Int.MAX // 12:
        return TensorResult[Fp32BinaryWindowBudget](TensorStatus.OVERFLOW)
    var available_device = limits.device_budget_bytes - limits.device_reserve_bytes
    if fixed_device_bytes > available_device:
        return TensorResult[Fp32BinaryWindowBudget](TensorStatus.NOT_REPRESENTABLE)
    available_device -= fixed_device_bytes
    if fixed_pinned_host_bytes > limits.pinned_host_budget_bytes:
        return TensorResult[Fp32BinaryWindowBudget](TensorStatus.NOT_REPRESENTABLE)
    var available_pinned = limits.pinned_host_budget_bytes - fixed_pinned_host_bytes
    var full_windows = count // capacity
    var tail = count % capacity
    # A single short window always gets exact-sized native buffers.
    # Three-stream pipelines retain their two full slots while they allocate
    # a separate exact-sized tail owner, so the tail IS additional memory.
    # Serial GPU_ONLY and HYBRID_SPLIT reallocate the SAME owner only AFTER
    # joining the previous D2H event. Such a tail does not coexist with the
    # previous full owner and must NOT consume a second memory reservation.
    var dedicated_tail_bytes = tail * 12
    if reuse_full_slot_for_tail and full_windows > 0:
        dedicated_tail_bytes = 0
    if full_windows == 0:
        dedicated_tail_bytes = count * 12
    if (dedicated_tail_bytes > available_device
        or dedicated_tail_bytes > available_pinned):
        return TensorResult[Fp32BinaryWindowBudget](TensorStatus.NOT_REPRESENTABLE)
    available_device -= dedicated_tail_bytes
    available_pinned -= dedicated_tail_bytes
    if full_windows == 0:
        return TensorResult[Fp32BinaryWindowBudget](Fp32BinaryWindowBudget(
            0, True, count, count,
            dedicated_tail_bytes, dedicated_tail_bytes,
        ))
    var full_bytes = capacity * 12
    var max_slots = min(preferred_full_slots, full_windows)
    max_slots = min(max_slots, available_device // full_bytes)
    max_slots = min(max_slots, available_pinned // full_bytes)
    if max_slots < 1:
        return TensorResult[Fp32BinaryWindowBudget](TensorStatus.NOT_REPRESENTABLE)
    var staging_bytes = max_slots * full_bytes + dedicated_tail_bytes
    return TensorResult[Fp32BinaryWindowBudget](Fp32BinaryWindowBudget(
        max_slots, tail != 0 and not reuse_full_slot_for_tail,
        capacity, tail, staging_bytes, staging_bytes,
    ))


def plan_fp32_binary_windows_with_total_ram(
    limits: Fp32MemoryLimits,
    count: Int, capacity: Int, preferred_full_slots: Int,
    fixed_device_bytes: Int, fixed_pinned_host_bytes: Int,
    max_total_ram_bytes: Int, fixed_total_ram_bytes: Int,
    reuse_full_slot_for_tail: Bool = False,
) -> TensorResult[Fp32BinaryWindowBudget]:
    # Optional PREPARE-time total RAM envelope. All existing owners, paged
    # input buffers, caches, workspace and non-pinned overhead must be counted
    # in fixed_total_ram_bytes. The NEW pinned staging bytes also consume RAM.
    # Unlike a post-hoc veto, this uses the existing window planner's slot
    # downshift to preserve a legal one/two-slot plan where one fits.
    # The Tensor dtype remains Float32 and the logical dimensions are intact.
    var valid = limits.validated()
    if valid != TensorStatus.OK:
        return TensorResult[Fp32BinaryWindowBudget](valid)
    if (max_total_ram_bytes <= 0 or fixed_total_ram_bytes < 0
        or fixed_pinned_host_bytes < 0
        or fixed_total_ram_bytes < fixed_pinned_host_bytes):
        return TensorResult[Fp32BinaryWindowBudget](TensorStatus.INVALID_SHAPE)
    if (fixed_total_ram_bytes > max_total_ram_bytes
        or fixed_pinned_host_bytes > limits.pinned_host_budget_bytes):
        return TensorResult[Fp32BinaryWindowBudget](TensorStatus.NOT_REPRESENTABLE)
    var ram_headroom = max_total_ram_bytes - fixed_total_ram_bytes
    var pinned_headroom = limits.pinned_host_budget_bytes - fixed_pinned_host_bytes
    var new_staging_cap = min(ram_headroom, pinned_headroom)
    # No overflow: cap <= pinned_headroom by construction.
    var capped_pinned_budget = fixed_pinned_host_bytes + new_staging_cap
    var capped = Fp32MemoryLimits(
        limits.device_budget_bytes, limits.device_reserve_bytes,
        capped_pinned_budget,
    )
    return plan_fp32_binary_windows(
        capped, count, capacity, preferred_full_slots,
        fixed_device_bytes, fixed_pinned_host_bytes,
        reuse_full_slot_for_tail,
    )
