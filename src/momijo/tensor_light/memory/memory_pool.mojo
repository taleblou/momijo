# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR MEMORY / MEMORY POOLS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Memory and Allocation
# Module:        momijo.tensor_light.memory.memory_pool
# Source:        src/momijo/tensor_light/memory/memory_pool.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines memory pools in the memory and allocation layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `HostPool`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Provide reusable storage pooling with checked borrow, recycle, and
#     resource-lifetime behavior.
#   - Expose source-defined interfaces including `HostPool`, `acquire`,
#     `release`, `cached_bytes`.
#   - Prevent recycling in-use storage and validate pool capacity/ownership.
#   - Propagate failed validation or allocation results without silently
#     changing memory domains.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Enforce capacity, allocation-domain, and ownership invariants for pooled
#     buffers.
#   - Do not recycle pending device/async resources without their required
#     completion proof.
#
# Performance Objectives:
#   - Reuse compatible buffers only after they are safe to recycle.
#   - Bound pool growth and unnecessary allocation or transfer costs.
#
# Maintainability:
#   - Document pool ownership, reclamation, invalidation, and quarantine
#     rules.
#   - Test allocation failures, capacity limits, and incomplete asynchronous
#     work.
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

# Bounded move-only FP32 host allocation cache. Up to three idle blocks are
# retained, so repeated independent CPU tiles and double/triple-buffered host
# staging do not immediately churn the allocator. No dynamic registry, hash
# table, heap metadata or per-element pool lookup is introduced.
#
# Only completely quiescent HostAllocation owners may be released here. This
# pageable HOST pool is NOT a GPU event-aware or pinned DMA staging pool; an
# asynchronous user must establish completion before relinquishing an owner.
from std.collections import Optional
from std.sys.info import simd_width_of
from .allocation import HostAllocation, _allocate_host_fp32_uninitialized
from ..internal.status import TensorResult, TensorStatus

struct HostPool(Movable):
    # Keep the original first-slot field for existing callers. Additional
    # slots permit distinct shape buckets to coexist without fake ownership.
    var spare: Optional[HostAllocation]
    var spare_second: Optional[HostAllocation]
    var spare_third: Optional[HostAllocation]
    var cached_elements: Int
    var cache_limit_elements: Int

    def __init__(out self, cache_limit_elements: Int):
        self.spare = Optional[HostAllocation](None)
        self.spare_second = Optional[HostAllocation](None)
        self.spare_third = Optional[HostAllocation](None)
        self.cached_elements = 0
        # Every retained FP32 byte sum must remain representable in Int.
        self.cache_limit_elements = min(max(0, cache_limit_elements), Int.MAX // 4)

    def _validate_cache(self) -> TensorStatus:
        # The pool fields are mutable Mojo values: cached_elements alone is
        # not a physical-size or unique-ownership proof. Verify the actual
        # ManagedAllocation layout of each stored block before reusing one.
        if (self.cache_limit_elements < 0
            or self.cache_limit_elements > Int.MAX // 4
            or self.cached_elements < 0):
            return TensorStatus.INVALID_LAYOUT
        var actual = 0
        if self.spare:
            var status = self.spare.value().validate_metadata()
            if status != TensorStatus.OK:
                return status
            var n = self.spare.value().physical_count
            if n > self.cache_limit_elements - actual:
                return TensorStatus.INVALID_LAYOUT
            actual += n
        if self.spare_second:
            var status = self.spare_second.value().validate_metadata()
            if status != TensorStatus.OK:
                return status
            var n = self.spare_second.value().physical_count
            if n > self.cache_limit_elements - actual:
                return TensorStatus.INVALID_LAYOUT
            actual += n
        if self.spare_third:
            var status = self.spare_third.value().validate_metadata()
            if status != TensorStatus.OK:
                return status
            var n = self.spare_third.value().physical_count
            if n > self.cache_limit_elements - actual:
                return TensorStatus.INVALID_LAYOUT
            actual += n
        if actual != self.cached_elements:
            return TensorStatus.INVALID_LAYOUT
        return TensorStatus.OK

    def _smallest_slot(self) -> Int:
        # INTERNAL: _validate_cache() has been established at the boundary.
        var selected = -1
        var smallest = Int.MAX
        if self.spare:
            var count = self.spare.value().physical_count
            if count < smallest:
                selected = 0
                smallest = count
        if self.spare_second:
            var count = self.spare_second.value().physical_count
            if count < smallest:
                selected = 1
                smallest = count
        if self.spare_third:
            var count = self.spare_third.value().physical_count
            if count < smallest:
                selected = 2
        return selected

    def _discard_slot(mut self, slot: Int) -> Bool:
        # Commit exactly the pre-approved eviction set. The admission policy
        # must never silently replace a different cached owner because a
        # fixed slot happened to be empty or cache order changed.
        if slot == 0 and self.spare:
            self.cached_elements -= self.spare.value().physical_count
            self.spare = Optional[HostAllocation](None)
            return True
        if slot == 1 and self.spare_second:
            self.cached_elements -= self.spare_second.value().physical_count
            self.spare_second = Optional[HostAllocation](None)
            return True
        if slot == 2 and self.spare_third:
            self.cached_elements -= self.spare_third.value().physical_count
            self.spare_third = Optional[HostAllocation](None)
            return True
        return False

    def _discard_smallest(mut self) -> Bool:
        # Convenience operation, not used for precomputed admission plans.
        return self._discard_slot(self._smallest_slot())

    def _acquire_uninitialized(mut self, count: Int) -> TensorResult[HostAllocation]:
        # The only uninitialized lease is internal. A readable Tensor is
        # exposed only after its caller has completed every logical write.
        if count < 0:
            return TensorResult[HostAllocation](TensorStatus.INVALID_SHAPE)
        if count > Int.MAX // 4:
            return TensorResult[HostAllocation](TensorStatus.OVERFLOW)
        var safe = self._validate_cache()
        if safe != TensorStatus.OK:
            return TensorResult[HostAllocation](safe)
        var needed = max(count, 1)
        # Best-fit, not first-fit: preserve larger blocks for larger work.
        var selected = -1
        var smallest_fit = Int.MAX
        if self.spare:
            var size = self.spare.value().physical_count
            if size >= needed and size < smallest_fit:
                selected = 0
                smallest_fit = size
        if self.spare_second:
            var size = self.spare_second.value().physical_count
            if size >= needed and size < smallest_fit:
                selected = 1
                smallest_fit = size
        if self.spare_third:
            var size = self.spare_third.value().physical_count
            if size >= needed and size < smallest_fit:
                selected = 2
        if selected == 0:
            var acquired = self.spare.take()
            self.cached_elements -= acquired.physical_count
            acquired.logical_count = count
            acquired.initialized = count == 0
            acquired.initialized_elements = -1
            if count == 0:
                acquired.initialized_elements = 0
            return TensorResult[HostAllocation](acquired^)
        if selected == 1:
            var acquired = self.spare_second.take()
            self.cached_elements -= acquired.physical_count
            acquired.logical_count = count
            acquired.initialized = count == 0
            acquired.initialized_elements = -1
            if count == 0:
                acquired.initialized_elements = 0
            return TensorResult[HostAllocation](acquired^)
        if selected == 2:
            var acquired = self.spare_third.take()
            self.cached_elements -= acquired.physical_count
            acquired.logical_count = count
            acquired.initialized = count == 0
            acquired.initialized_elements = -1
            if count == 0:
                acquired.initialized_elements = 0
            return TensorResult[HostAllocation](acquired^)
        return _allocate_host_fp32_uninitialized(count)

    def acquire(mut self, count: Int) -> TensorResult[HostAllocation]:
        # A cached block may contain another owner's old payload. Never
        # return it as an initialized HostAllocation without overwriting it.
        return self.acquire_filled(count, Float32(0.0))

    def release(mut self, var owned: HostAllocation) -> TensorStatus:
        var valid = owned.validate_metadata()
        if valid != TensorStatus.OK:
            return valid
        var safe = self._validate_cache()
        if safe != TensorStatus.OK:
            return safe
        var size = owned.physical_count
        if size > self.cache_limit_elements:
            # Out-of-budget owners deallocate normally upon return.
            return TensorStatus.OK
        # Select cache victims against a local fixed-size metadata snapshot.
        # Do NOT evict anything before checking that the new block can fit:
        # partial eviction followed by rejection would gratuitously destroy
        # existing useful allocations and cause allocator churn.
        var sizes = Array[Int, 3](fill=0)
        var present = Array[Bool, 3](fill=False)
        var occupied = 0
        if self.spare:
            sizes[0] = self.spare.value().physical_count
            present[0] = True
            occupied += 1
        if self.spare_second:
            sizes[1] = self.spare_second.value().physical_count
            present[1] = True
            occupied += 1
        if self.spare_third:
            sizes[2] = self.spare_third.value().physical_count
            present[2] = True
            occupied += 1
        var required_victims = 0
        if occupied == 3:
            required_victims = 1
        var missing_budget = 0
        var available = self.cache_limit_elements - self.cached_elements
        if size > available:
            missing_budget = size - available
        # Exhaust all eight fixed eviction sets OFF the acquisition hot path.
        # The previous greedy-smallest policy could evict three cached blocks
        # when evicting one sufficiently large (but still smaller than the
        # incoming owner) would have met both the slot and byte budgets.
        # Minimize allocator churn: first choose FEWEST evictions, then the
        # LEAST total evicted capacity. All size sums fit the verified cache
        # budget; no dynamic metadata allocation or per-element work occurs.
        var best_mask = -1
        var best_count = 4
        var best_freed = Int.MAX
        for mask in range(8):
            var evictions = 0
            var bytes_freed = 0
            var legal = True
            for slot in range(3):
                if (mask & (1 << slot)) != 0:
                    if not present[slot] or sizes[slot] >= size:
                        legal = False
                    else:
                        evictions += 1
                        bytes_freed += sizes[slot]
            if legal and evictions >= required_victims and bytes_freed >= missing_budget:
                if evictions < best_count or (evictions == best_count and bytes_freed < best_freed):
                    best_mask = mask
                    best_count = evictions
                    best_freed = bytes_freed
        if best_mask < 0:
            # Admission would evict an equal/larger reusable block or still
            # fail the capacity budget. Leave the existing cache untouched.
            return TensorStatus.OK
        var planned = Array[Bool, 3](fill=False)
        for slot in range(3):
            planned[slot] = (best_mask & (1 << slot)) != 0
        # Commit precisely the pre-approved eviction set. The resulting
        # cached count is updated only after all selected owners are dropped.
        for i in range(3):
            if planned[i]:
                if not self._discard_slot(i):
                    return TensorStatus.INVALID_LAYOUT
        self.cached_elements += size
        if not self.spare:
            self.spare = Optional[HostAllocation](owned^)
        elif not self.spare_second:
            self.spare_second = Optional[HostAllocation](owned^)
        else:
            self.spare_third = Optional[HostAllocation](owned^)
        return TensorStatus.OK

    def cached_bytes(self) -> Int:
        # _validate_cache() can be used by expert callers when pool fields
        # may have been mutated through unsafe APIs.
        if self._validate_cache() != TensorStatus.OK:
            return -1
        return self.cached_elements * 4

    def largest_cached_block(self) -> Int:
        if self._validate_cache() != TensorStatus.OK:
            return -1
        var best = 0
        if self.spare:
            best = max(best, self.spare.value().physical_count)
        if self.spare_second:
            best = max(best, self.spare_second.value().physical_count)
        if self.spare_third:
            best = max(best, self.spare_third.value().physical_count)
        return best

    def clear(mut self):
        self.spare = Optional[HostAllocation](None)
        self.spare_second = Optional[HostAllocation](None)
        self.spare_third = Optional[HostAllocation](None)
        self.cached_elements = 0

    def trim(mut self):
        self.clear()

    def acquire_filled(mut self, count: Int, value: Float32) -> TensorResult[HostAllocation]:
        # Exactly one vectorized logical-region initialization. Physical
        # padding in a reused larger block is not readable by the new owner.
        var acquired = self._acquire_uninitialized(count)
        if not acquired.is_ok():
            return TensorResult[HostAllocation](acquired.status)
        var owned = acquired.payload.take()
        var data = owned.memory.unsafe_ptr()
        comptime width = simd_width_of[DType.float32]()
        var logical_items = owned.logical_count
        var limit = logical_items - logical_items % width
        var lanes = SIMD[DType.float32, width](value)
        for i in range(0, limit, width):
            data.unsafe_store(i, lanes)
        for i in range(limit, logical_items):
            data.unsafe_store(i, value)
        var done = owned._mark_complete_overwrite()
        if done != TensorStatus.OK:
            return TensorResult[HostAllocation](done)
        return TensorResult[HostAllocation](owned^)
