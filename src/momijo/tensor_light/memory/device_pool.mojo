# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR MEMORY / DEVICE FLOAT32 BUFFER POOL
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Memory and Allocation
# Module:        momijo.tensor_light.memory.device_pool
# Source:        src/momijo/tensor_light/memory/device_pool.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines device float32 buffer pool in the memory and allocation layer of
#   the FP32 Light tensor implementation.
#
#   Public declarations include `DeviceFp32Lease`, `DeviceFp32Pool`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Manage explicitly leased FP32 device allocations with checked ownership
#     and reuse.
#   - Expose source-defined interfaces including `DeviceFp32Lease`,
#     `physical_capacity`, `quarantine`, `DeviceFp32Pool`.
#   - Respect pool capacity, lease lifetime, and device completion conditions.
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

# Bounded device-local FP32 cache for one explicitly owned MAX DeviceContext.
# The pool retains BOTH the physical DeviceBuffer owner and the completion
# event which fences its last use. A cached allocation is never given out
# until its event establishes quiescence; failure to prove quiescence poisons
# the pool and quarantines live native owners rather than risking use-after-free.
#
# This is an expert SINGLE-DEFAULT-STREAM pool. It does not infer stream
# dependencies from a DeviceBuffer or silently synchronize other streams.
# All work that touches a lease MUST be ordered on pool.context.stream(),
# including any external stream waits, BEFORE unsafe_release_after_stream().
# Other streams require an explicit joined dependency on that default stream.
# Leases carry their own DeviceContext owner. Dropping a lease outside its
# originating pool is fail-closed for ordered default-stream work: its own
# destructor joins before freeing, or quarantines buffer plus context on
# an unprovable join. A refcounted origin token now REJECTS returns to a
# different pool, including another pool with the same DeviceContext.
# The owner type of the public application tensor remains Tensor[rank, Storage].
from std.collections import Optional
from std.memory import forget_deinit, ArcPointer
from max.gpu.host import DeviceContext, DeviceBuffer, DeviceEvent
from ..internal.status import TensorResult, TensorStatus


struct DeviceFp32Lease(Movable):
    # Move-only allocation handoff. A lease is NOT a readable Tensor: it
    # contains no layout and does not certify initialized FP32 payload data.
    # Keep a separate native context owner: the caller can destroy its pool
    # without invalidating THIS lease's default-stream destruction barrier.
    # The caller must NOT use the lease on an unordered foreign stream or
    # return it to a different pool. Those uses need an explicit event edge.
    var buffer: Optional[DeviceBuffer[DType.float32]]
    var context: DeviceContext
    # This token is the SAME refcounted allocation as the one in the
    # originating pool. Comparing its value would NOT prove identity: use
    # ArcPointer.__is__ to compare the allocation itself. The copy keeps the
    # identity valid even if the pool is moved or destroyed first.
    var _origin: ArcPointer[Int]
    var logical_count: Int
    var _unproven_completion: Bool

    def __init__(
        out self, var owned: DeviceBuffer[DType.float32],
        count: Int, ctx: DeviceContext, origin: ArcPointer[Int],
    ):
        self.buffer = Optional[DeviceBuffer[DType.float32]](owned)
        self.context = ctx.copy()
        self._origin = origin.copy()
        self.logical_count = count
        self._unproven_completion = False

    def _quarantine_unproven(mut self):
        # After a fatal event/stream join, the original context may be the
        # ONLY owner capable of keeping this allocation's runtime alive.
        # Retain the lease context at its destructor independently from the
        # pool context; a wrong-pool return must never strand a buffer.
        self._unproven_completion = True
        if self.buffer:
            var pending = self.buffer.take()
            forget_deinit(pending)

    def __deinit__(deinit self):
        # A lease can be dropped without being returned to its pool. A bare
        # DeviceBuffer destructor is not an asynchronous GPU completion proof.
        # If it still owns an allocation, join the ordered default stream
        # before ordinary teardown. On an unprovable join, deliberately keep
        # BOTH the device allocation and the context owner alive.
        if self._unproven_completion:
            if self.buffer:
                var pending = self.buffer.take()
                forget_deinit(pending)
            forget_deinit(self.context)
        elif self.buffer:
            try:
                # First join the final device user. Then explicitly schedule
                # buffer destruction while this lease still owns its context.
                self.context.stream().synchronize()
                self.buffer = Optional[DeviceBuffer[DType.float32]](None)
                # MAX DeviceBuffer.__deinit__ schedules an ASYNC free, AFTER
                # the compute join above. Complete that free before dropping
                # the lease's final owning context reference.
                self.context.stream().synchronize()
            except:
                if self.buffer:
                    var pending = self.buffer.take()
                    forget_deinit(pending)
                # Even if the buffer's async free was already submitted, a
                # failed drain must not destroy the allocating context.
                forget_deinit(self.context)

    def physical_capacity(self) -> Int:
        if not self.buffer:
            return -1
        return len(self.buffer.value())


struct _CachedDeviceFp32Block(Movable):
    var buffer: Optional[DeviceBuffer[DType.float32]]
    var completion: Optional[DeviceEvent]
    var capacity_elements: Int

    def __init__(
        out self, var owned: DeviceBuffer[DType.float32],
        var last_use_event: DeviceEvent, capacity: Int,
    ):
        self.buffer = Optional[DeviceBuffer[DType.float32]](owned)
        self.completion = Optional[DeviceEvent](last_use_event)
        self.capacity_elements = capacity

    def quarantine(mut self):
        # The paired event and buffer must survive failed GPU quiescence.
        # They are intentionally leaked until out-of-band device reset.
        if self.completion:
            var event = self.completion.take()
            forget_deinit(event)
        if self.buffer:
            var native_owner = self.buffer.take()
            forget_deinit(native_owner)


struct DeviceFp32Pool(Movable):
    # Bounded cache: no runtime registry, no dynamic slot allocation, no JIT.
    # Native owners from one device can never be mixed with another context.
    var context: DeviceContext
    # The token allocation is unique per pool and reference-counted. Unlike
    # a stack self-pointer, its identity survives moves and outstanding
    # leases; no global counter, shared registry or address reuse is needed.
    var _origin: ArcPointer[Int]
    var slot0: Optional[_CachedDeviceFp32Block]
    var slot1: Optional[_CachedDeviceFp32Block]
    var slot2: Optional[_CachedDeviceFp32Block]
    var cached_elements: Int
    var cache_limit_elements: Int
    var poisoned: Bool
    # Poisoning does not imply all still-cached owners must be leaked.
    # A later successful DEFAULT-STREAM join may allow cache reclamation.
    # Unjoined foreign-stream work and detached owner leases do not have
    # the same local recovery proof.
    var _unproven_orphan_lease: Bool
    var _unjoined_external_producer: Bool

    def __init__(
        out self, var context: DeviceContext, cache_limit_elements: Int,
    ):
        self.context = context
        self._origin = ArcPointer[Int](0)
        self.slot0 = Optional[_CachedDeviceFp32Block](None)
        self.slot1 = Optional[_CachedDeviceFp32Block](None)
        self.slot2 = Optional[_CachedDeviceFp32Block](None)
        self.cached_elements = 0
        self.cache_limit_elements = min(max(0, cache_limit_elements), Int.MAX // 4)
        self.poisoned = False
        self._unproven_orphan_lease = False
        self._unjoined_external_producer = False

    def __deinit__(deinit self):
        # Do not let the implicit destructors of cached DeviceBuffers run
        # while default-stream work might still reference their addresses.
        # An empty cache has no owned native allocations to release: avoid
        # blocking on unrelated work in its DeviceContext during teardown.
        var had_cached_blocks = Bool(self.slot0) or Bool(self.slot1) or Bool(self.slot2)
        if not self.poisoned and had_cached_blocks:
            try:
                self.context.stream().synchronize()
            except:
                self.poisoned = True
        if not self.poisoned:
            # Join established native completion. Release every cached event
            # and DeviceBuffer EXPLICITLY while this pool still owns its
            # live DeviceContext, rather than relying on unspecified member
            # destructor ordering between context and cached blocks.
            self.slot0 = Optional[_CachedDeviceFp32Block](None)
            self.slot1 = Optional[_CachedDeviceFp32Block](None)
            self.slot2 = Optional[_CachedDeviceFp32Block](None)
            self.cached_elements = 0
            # Cache release destructors may themselves enqueue native frees.
            # Their commands are later than the first join above; drain once
            # at final pool teardown before losing the allocator context.
            # No cached blocks means no deferred native-free commands from
            # this pool, so keep empty-pool teardown free of forced syncs.
            if had_cached_blocks:
                try:
                    self.context.stream().synchronize()
                except:
                    self.poisoned = True
        if self.poisoned:
            if self.slot0:
                var block0 = self.slot0.take()
                block0.quarantine()
            if self.slot1:
                var block1 = self.slot1.take()
                block1.quarantine()
            if self.slot2:
                var block2 = self.slot2.take()
                block2.quarantine()
            # The context is a native runtime owner as well: an unresolved
            # device operation must not outlive an automatically freed ctx.
            forget_deinit(self.context)

    def cached_bytes(self) -> Int:
        if self.poisoned or self.cached_elements < 0:
            return -1
        return self.cached_elements * 4

    def _validate_cache(self) -> TensorStatus:
        if self.poisoned:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if self.cached_elements < 0 or self.cached_elements > self.cache_limit_elements:
            return TensorStatus.INVALID_LAYOUT
        var actual = 0
        if self.slot0:
            if not self.slot0.value().buffer or not self.slot0.value().completion:
                return TensorStatus.INVALID_LAYOUT
            var physical = self.slot0.value().capacity_elements
            if physical < 1 or physical != len(self.slot0.value().buffer.value()):
                return TensorStatus.INVALID_LAYOUT
            if physical > self.cache_limit_elements - actual:
                return TensorStatus.INVALID_LAYOUT
            actual += physical
        if self.slot1:
            if not self.slot1.value().buffer or not self.slot1.value().completion:
                return TensorStatus.INVALID_LAYOUT
            var physical = self.slot1.value().capacity_elements
            if physical < 1 or physical != len(self.slot1.value().buffer.value()):
                return TensorStatus.INVALID_LAYOUT
            if physical > self.cache_limit_elements - actual:
                return TensorStatus.INVALID_LAYOUT
            actual += physical
        if self.slot2:
            if not self.slot2.value().buffer or not self.slot2.value().completion:
                return TensorStatus.INVALID_LAYOUT
            var physical = self.slot2.value().capacity_elements
            if physical < 1 or physical != len(self.slot2.value().buffer.value()):
                return TensorStatus.INVALID_LAYOUT
            if physical > self.cache_limit_elements - actual:
                return TensorStatus.INVALID_LAYOUT
            actual += physical
        if actual != self.cached_elements:
            return TensorStatus.INVALID_LAYOUT
        return TensorStatus.OK

    def unsafe_acquire(
        mut self, count: Int,
    ) -> TensorResult[DeviceFp32Lease]:
        # This API is unsafe only because the caller must exclusively use
        # THIS pool's live DeviceContext and its ordered default stream.
        # A returned buffer is ALWAYS uninitialized, including cache hits.
        if count < 0:
            return TensorResult[DeviceFp32Lease](TensorStatus.INVALID_SHAPE)
        if count > Int.MAX // 4:
            return TensorResult[DeviceFp32Lease](TensorStatus.OVERFLOW)
        var valid = self._validate_cache()
        if valid != TensorStatus.OK:
            return TensorResult[DeviceFp32Lease](valid)
        var needed = max(count, 1)
        # Exact-capacity reuse preserves the flat MAX copy ABI: callers
        # never silently receive a padded allocation of a different size.
        var chosen = -1
        if self.slot0 and self.slot0.value().capacity_elements == needed:
            chosen = 0
        elif self.slot1 and self.slot1.value().capacity_elements == needed:
            chosen = 1
        elif self.slot2 and self.slot2.value().capacity_elements == needed:
            chosen = 2
        if chosen >= 0:
            var saved = Optional[_CachedDeviceFp32Block](None)
            if chosen == 0:
                saved = Optional[_CachedDeviceFp32Block](self.slot0.take())
            elif chosen == 1:
                saved = Optional[_CachedDeviceFp32Block](self.slot1.take())
            else:
                saved = Optional[_CachedDeviceFp32Block](self.slot2.take())
            var block = saved.take()
            self.cached_elements -= needed
            # An event may report a GPU execution failure as well as a
            # synchronization failure. Never turn failed work into a clean
            # initialized lease merely because a fallback stream did join.
            try:
                block.completion.value().synchronize()
            except:
                try:
                    self.context.stream().synchronize()
                except:
                    # Keep the actual cache allocation/event owned by the
                    # live pool; preserve the exact slot and capacity.
                    # __deinit__ still quarantines it if no later safe
                    # default-stream recovery establishes completion.
                    self.poisoned = True
                    self.cached_elements += needed
                    if chosen == 0:
                        self.slot0 = Optional[_CachedDeviceFp32Block](block)
                    elif chosen == 1:
                        self.slot1 = Optional[_CachedDeviceFp32Block](block)
                    else:
                        self.slot2 = Optional[_CachedDeviceFp32Block](block)
                    return TensorResult[DeviceFp32Lease](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
                # The event reported a failure even though the fallback stream
                # join succeeded. Retire this buffer using the owning lease's
                # pre-free and post-free fences; dropping block directly would
                # enqueue an async free with no guaranteed context drain.
                var retired = DeviceFp32Lease(
                    block.buffer.take(), count, self.context, self._origin
                )
                return TensorResult[DeviceFp32Lease](TensorStatus.TRANSFER_UNAVAILABLE)
            return TensorResult[DeviceFp32Lease](
                DeviceFp32Lease(block.buffer.take(), count, self.context, self._origin)
            )
        try:
            var native_owner = self.context.enqueue_create_buffer[DType.float32](needed)
            # Allocation is asynchronous. All subsequent users must enqueue
            # on this very same ordered stream or explicitly join it first.
            return TensorResult[DeviceFp32Lease](DeviceFp32Lease(native_owner, count, self.context, self._origin))
        except:
            try:
                self.context.stream().synchronize()
            except:
                self.poisoned = True
                return TensorResult[DeviceFp32Lease](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
            return TensorResult[DeviceFp32Lease](TensorStatus.ALLOCATION_FAILED)

    def unsafe_release_after_stream(
        mut self, var lease: DeviceFp32Lease,
    ) -> TensorStatus:
        # No borrowed copies may remain; the caller has enqueued EVERY use
        # before entry on pool.context.stream(). Other streams MUST have a
        # stream-wait/event edge to the default stream before this release.
        # Ownership transfers only once; a lease is never simultaneously
        # present in the cache and usable by the caller.
        # Verify pool identity BEFORE consuming buffer metadata or touching
        # this pool's stream. Same-device and even same-context pools are
        # NOT interchangeable native allocation/stream owners. A rejected
        # lease falls through its OWN safe destructor, which joins its
        # originating ordered stream (or quarantines on failure).
        if not self._origin.__is__(lease._origin):
            return TensorStatus.DEVICE_MISMATCH
        if self.context != lease.context:
            return TensorStatus.DEVICE_MISMATCH
        if not lease.buffer:
            return TensorStatus.INVALID_LAYOUT
        var physical_count = len(lease.buffer.value())
        var metadata_ok = (
            lease.logical_count >= 0 and lease.logical_count <= Int.MAX // 4
            and physical_count == max(lease.logical_count, 1)
        )
        # Even an invalid lease may be in flight. Establish quiescence or
        # quarantine it BEFORE allowing Mojo to destroy the native buffer.
        if self.poisoned:
            self._unproven_orphan_lease = True
            lease._quarantine_unproven()
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        var valid = self._validate_cache()
        if valid != TensorStatus.OK and valid != TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            # Do not drop potentially pending owners on a metadata error.
            try:
                self.context.stream().synchronize()
            except:
                self.poisoned = True
                self._unproven_orphan_lease = True
                lease._quarantine_unproven()
                return TensorStatus.GPU_QUIESCENCE_UNPROVEN
            # Leave the native owner in lease: its destructor joins queued
            # work, schedules the async free, then joins that free before
            # relinquishing its context. Taking it into a bare local would
            # bypass the lease's required post-free fence.
            return valid
        var recorded = Optional[DeviceEvent](None)
        try:
            recorded = Optional[DeviceEvent](self.context.create_event())
            self.context.stream().record_event(recorded.value())
        except:
            try:
                self.context.stream().synchronize()
            except:
                self.poisoned = True
                if recorded:
                    var failed_event = recorded.take()
                    forget_deinit(failed_event)
                self._unproven_orphan_lease = True
                lease._quarantine_unproven()
                return TensorStatus.GPU_QUIESCENCE_UNPROVEN
            # Retain lease ownership through return for its post-free fence.
            return TensorStatus.TRANSFER_UNAVAILABLE
        if not metadata_ok:
            try:
                recorded.value().synchronize()
            except:
                try:
                    self.context.stream().synchronize()
                except:
                    self.poisoned = True
                    var leaked_event = recorded.take()
                    forget_deinit(leaked_event)
                    self._unproven_orphan_lease = True
                    lease._quarantine_unproven()
                    return TensorStatus.GPU_QUIESCENCE_UNPROVEN
            # Even an invalid lease must run its joined native destructor.
            return TensorStatus.INVALID_LAYOUT
        # An uncached owner must not be deallocated while an earlier GPU
        # operation is still using it. Fenced release joins ONLY this event.
        var free_immediately = physical_count > self.cache_limit_elements
        if not self.slot0 and not self.slot1 and not self.slot2:
            free_immediately = False
        elif self.slot0 and self.slot1 and self.slot2:
            free_immediately = True
        if physical_count > self.cache_limit_elements - self.cached_elements:
            free_immediately = True
        if free_immediately:
            try:
                recorded.value().synchronize()
            except:
                try:
                    self.context.stream().synchronize()
                except:
                    self.poisoned = True
                    var leaked_event = recorded.take()
                    forget_deinit(leaked_event)
                    self._unproven_orphan_lease = True
                    lease._quarantine_unproven()
                    return TensorStatus.GPU_QUIESCENCE_UNPROVEN
                # A failed event does not authorize bypassing the lease's
                # native free completion fence after fallback stream join.
                return TensorStatus.TRANSFER_UNAVAILABLE
            # The last GPU use has completed. Keep the DeviceBuffer owned by
            # this lease through function return so its destructor also
            # joins the ASYNCHRONOUS free before releasing its context. This
            # is the uncached/overflow path, not steady-state cache reuse.
            return TensorStatus.OK
        var kept = _CachedDeviceFp32Block(
            lease.buffer.take(), recorded.take(), physical_count
        )
        self.cached_elements += physical_count
        if not self.slot0:
            self.slot0 = Optional[_CachedDeviceFp32Block](kept)
        elif not self.slot1:
            self.slot1 = Optional[_CachedDeviceFp32Block](kept)
        else:
            self.slot2 = Optional[_CachedDeviceFp32Block](kept)
        return TensorStatus.OK

    def unsafe_release_after_device_event(
        mut self, var lease: DeviceFp32Lease, producer_done: DeviceEvent,
    ) -> TensorStatus:
        # Opt-in cross-stream producer dependency. The caller MUST prove
        # producer_done was recorded AFTER every last use of this exact
        # DeviceFp32Lease on a stream belonging to this SAME context. The
        # stream-wait transfers the use-after ordering to the pool's default
        # stream without blocking the host. The normal release below then
        # records a NEW completion event after this wait; the buffer cannot
        # be reissued until that event completes successfully.
        # Reject a foreign pool/context BEFORE enqueuing a device-side wait;
        # the event from another context cannot certify our allocation.
        if not self._origin.__is__(lease._origin):
            return TensorStatus.DEVICE_MISMATCH
        if self.context != lease.context:
            return TensorStatus.DEVICE_MISMATCH
        if not lease.buffer:
            return TensorStatus.INVALID_LAYOUT
        if self.poisoned:
            self._unproven_orphan_lease = True
            lease._quarantine_unproven()
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        try:
            self.context.stream().enqueue_wait_for(producer_done)
        except:
            # An event from another context, a failed wait enqueue, or an
            # invalid native event DOES NOT prove the producer stream idle.
            # Synchronizing the destination stream would not fix that.
            self.poisoned = True
            self._unjoined_external_producer = True
            self._unproven_orphan_lease = True
            lease._quarantine_unproven()
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return self.unsafe_release_after_stream(lease)

    def clear(mut self) -> TensorStatus:
        if self.poisoned:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        var had_cached_blocks = Bool(self.slot0) or Bool(self.slot1) or Bool(self.slot2)
        # One explicit stream join proves all three cached release events
        # completed. An error means we MUST retain context plus owners.
        try:
            self.context.stream().synchronize()
        except:
            # Do NOT discard the cached owners before an actual quiescence
            # proof. They remain unreachable for reuse while poisoned, and
            # are quarantined on destruction if recovery is never proven.
            self.poisoned = True
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        self.slot0 = Optional[_CachedDeviceFp32Block](None)
        self.slot1 = Optional[_CachedDeviceFp32Block](None)
        self.slot2 = Optional[_CachedDeviceFp32Block](None)
        self.cached_elements = 0
        # MAX DeviceBuffer destructors enqueue native frees AFTER the join
        # above. Drain those frees while the pool still owns the context;
        # otherwise clear()+immediate pool destruction can drop the last
        # context reference before actual deallocation finishes. Skip this
        # extra join if there was no cached allocation to free.
        if had_cached_blocks:
            try:
                self.context.stream().synchronize()
            except:
                self.poisoned = True
                # __deinit__ retains the context on an unprovable cleanup drain.
                return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.OK

    def unsafe_try_reclaim_quiesced_cache(mut self) -> TensorStatus:
        # Narrow cache-only recovery; NOT a GPU driver reset or a general
        # recovery of detached leases. A later synchronization may succeed
        # after an earlier transient default-stream failure. Only THEN may
        # the still-owned cached DeviceBuffers/Events be destroyed safely.
        # No implicit retry is performed on hot acquire/release paths.
        if not self.poisoned:
            return TensorStatus.OK
        if self._unjoined_external_producer:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        try:
            self.context.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        # Quiescence has been established for every cached default-stream
        # resource; release them while this pool still owns its context.
        var had_cached_blocks = Bool(self.slot0) or Bool(self.slot1) or Bool(self.slot2)
        self.slot0 = Optional[_CachedDeviceFp32Block](None)
        self.slot1 = Optional[_CachedDeviceFp32Block](None)
        self.slot2 = Optional[_CachedDeviceFp32Block](None)
        self.cached_elements = 0
        # The first join proved kernel completion, not completion of frees
        # submitted by the cached blocks' destructors. Retain the native
        # context until their release commands are also confirmed complete.
        # A poisoned-but-empty cache has no new frees to drain.
        if had_cached_blocks:
            try:
                self.context.stream().synchronize()
            except:
                self.poisoned = True
                return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if self._unproven_orphan_lease:
            # Such an owner was detached with forget_deinit() and cannot
            # be reconstituted as a tracked cache allocation. Its resource
            # reclamation still requires an external/native recovery path.
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        self.poisoned = False
        return TensorStatus.OK
