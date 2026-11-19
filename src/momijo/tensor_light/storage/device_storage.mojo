# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR STORAGE / GPU DEVICE STORAGE
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Storage and Ownership
# Module:        momijo.tensor_light.storage.device_storage
# Source:        src/momijo/tensor_light/storage/device_storage.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines GPU device storage in the storage and ownership layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `NativeWriteTicket`, `DeviceStorage`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Enforce tensor storage lifetime and initialized-data contracts.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `NativeWriteTicket`,
#     `DeviceStorage`.
#   - Reject unsupported target or unverified completion states.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Preserve storage provenance and initialization state.
#   - Prevent use-after-release and unsafe mutable aliasing.
#   - Reject unsupported or unauthenticated execution paths explicitly.
#
# Performance Objectives:
#   - Avoid storage duplication when a safe borrow is sufficient.
#   - Reclaim storage only after ownership/completion conditions hold.
#   - Avoid implicit host/device migration or compilation at execution time.
#
# Maintainability:
#   - Document ownership transitions and destructor behavior.
#   - Test external-borrow lifetime and partially initialized buffers.
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

# Device-local Float32 allocation, with no implicit CPU fallback/readback.
from max.gpu.host import DeviceBuffer, DeviceContext
from std.collections import Optional
from std.memory import forget_deinit
from ..internal.status import TensorStatus, TensorResult, device_write_failure_may_have_enqueued
from .storage import TensorStorage


# A fatal native join can outlive a borrowed DeviceContext argument. If no
# DeviceStorage object remains to retain the context, keep a separate native
# context owner alive alongside every deliberately quarantined raw buffer.
# This is emergency resource retention, NOT a device reset or reclamation.
@inline(.always)
def _retain_context_on_unproven_gpu_completion(ctx: DeviceContext):
    var retained = ctx.copy()
    forget_deinit(retained)

# An epoch-tagged native write ticket is issued only by the owning storage.
# This prevents completion of an earlier write from publishing a later write.
@fieldwise_init
struct NativeWriteTicket(Copyable):
    var epoch: Int
    var previously_readable: Bool


struct DeviceStorage(Movable, TensorStorage):
    # Keep the actual MAX allocation context alive for the ENTIRE ordinary
    # device-buffer lifetime, not only on exceptional/quarantined paths.
    # Destructor explicitly extends the context lifetime until AFTER the
    # native buffer's last use. Mojo uses ASAP field destruction, not
    # C++ reverse-declaration-order destruction; the context is a retained
    # concrete capability, never a global current-device registry.
    var lifetime_context: Optional[DeviceContext]
    var buffer: DeviceBuffer[DType.float32]
    var count: Int
    # Source-level completion state. The trusted adapter must supply a real
    # DeviceEvent/stream proof BEFORE transitioning after a complete fill/copy.
    var initialized: Bool
    var initialized_elements: Int
    # True only after native completion AND fallback stream sync both failed.
    # Do not destroy or recycle the DeviceBuffer in that state.
    var quiescence_unproven: Bool
    # A checked native out= writer must transition through begin -> complete.
    # An abandoned/raising writer is never allowed to make the old payload
    # readable or allow the buffer to be destroyed before a native join.
    var _native_update_pending: Bool
    # The readable state is captured by the owner, not trusted from a
    # subsequent caller-provided bool passed to the completion entry point.
    var _native_update_was_readable: Bool
    var _native_update_epoch: Int
    # A pending native owner must keep the associated runtime context alive.
    # A borrowed ctx argument on the call stack is NOT a lifetime proof after
    # an unsuccessful join. The quarantined context remains a separate owner.
    var quarantined_context: Optional[DeviceContext]

    def __init__(
        out self, var buffer: DeviceBuffer[DType.float32],
        count: Int, ctx: DeviceContext,
    ):
        # A DeviceBuffer alone is not proof that its borrowed DeviceContext
        # survives after the creating Tensor factory returns. Retain the
        # issuing context before admitting the buffer to the owning Tensor.
        self.lifetime_context = Optional[DeviceContext](ctx.copy())
        self.buffer = buffer
        self.count = count
        # Even a logically empty device buffer has a physical allocation.
        # MAX may enqueue that allocation asynchronously; declaring it ready
        # in this constructor would allow an unchecked/raw DeviceBuffer to be
        # released before its allocation-completion event. Only a trusted
        # joined allocation/fill/copy path may promote DeviceStorage, even
        # when logical_count == 0 and no FP32 payload bytes are addressed.
        self.initialized = False
        self.initialized_elements = -1
        self.quiescence_unproven = False
        self._native_update_pending = False
        self._native_update_was_readable = False
        self._native_update_epoch = 0
        self.quarantined_context = Optional[DeviceContext](None)

    def _shares_issuing_context(self, other: Self) -> Bool:
        # A DeviceBuffer address is NOT a device/queue identity proof.
        # Even if two device addresses appear disjoint, a kernel or DMA
        # submitted through a foreign context may not order the producer.
        # Borrowed native Tensor paths permit only the SAME retained MAX
        # DeviceContext; cross-context copies require an explicit trusted
        # peer/event transfer adapter instead of pretending default streams
        # have a happens-before relation. Equality is only a necessary native
        # context identity gate: it does NOT prove ordering for an external
        # producer using an independent stream of that same context.
        if (not self.lifetime_context or not other.lifetime_context
            or self.quiescence_unproven or other.quiescence_unproven
            or self._native_update_pending or other._native_update_pending
            or self.quarantined_context or other.quarantined_context
            or self._native_update_epoch < 0 or other._native_update_epoch < 0
            or self.capacity() < 0 or other.capacity() < 0):
            return False
        return self.lifetime_context.value() == other.lifetime_context.value()

    def is_initialized(self) -> Bool:
        return (not self.quiescence_unproven
                and not self._native_update_pending
                and Bool(self.lifetime_context)
                and not Bool(self.quarantined_context)
                and self.capacity() >= 0
                and self.initialized
                and self.initialized_elements == self.count)

    def _invalidate_after_failed_write(
        mut self, ctx: DeviceContext, unproven: Bool,
    ):
        # Any unsuccessful native write invalidates the prior payload,
        # including a partial overwrite of an already-readable out= owner.
        # On an unproven join, keep BOTH the buffer and a concrete context
        # owner; retaining only one native resource can leave the pending GPU
        # operation accessing a context already destroyed by its caller.
        self.initialized = False
        self.initialized_elements = -1
        self._native_update_pending = False
        self._native_update_was_readable = False
        if unproven:
            self.quiescence_unproven = True
            if not self.quarantined_context:
                self.quarantined_context = Optional[DeviceContext](ctx.copy())

    def __deinit__(deinit self):
        # A failed GPU stream must never outlive a freed output allocation.
        # As in OwnedAsyncFp32Transfer, retain BOTH the native allocation
        # and the context copy on irrecoverable failure. This is quarantine,
        # not device reset or proof of eventual native recovery.
        if (self.quiescence_unproven or self.quarantined_context
            or self._native_update_pending or not self.lifetime_context):
            # An abandoned out= launch may still be reading/writing this
            # allocation. Conservatively quarantine its physical owner and
            # native context even when no error callback was reached.
            # If a caller damaged the metadata by removing the context
            # capability, leaking a native buffer is preferable to release
            # via an unproven/dead MAX allocation context.
            forget_deinit(self.buffer)
            if self.quarantined_context:
                var pending_ctx = self.quarantined_context.take()
                forget_deinit(pending_ctx)
            # A native buffer intentionally leaked for unknown completion
            # must also retain its ORDINARY context owner. The separate
            # quarantined handle documents the failed-stream recovery path.
            if self.lifetime_context:
                var pending_lifetime = self.lifetime_context.take()
                forget_deinit(pending_lifetime)
        else:
            # A native DeviceBuffer destructor queues its free on the owning
            # MAX stream. Before freeing, join the last default-stream user;
            # AFTER freeing, join the newly queued free while this storage
            # still retains its allocating DeviceContext.
            try:
                self.lifetime_context.value().stream().synchronize()
            except:
                # No proven final-use completion: quarantine BOTH resources.
                forget_deinit(self.buffer)
                var unrecovered = self.lifetime_context.take()
                forget_deinit(unrecovered)
                return
            _ = self.buffer
            try:
                self.lifetime_context.value().stream().synchronize()
            except:
                # The buffer was already destroyed, but its async free may
                # still be pending. Never release the last context owner.
                var unrecovered = self.lifetime_context.take()
                forget_deinit(unrecovered)
                return
            _ = self.lifetime_context

    def unsafe_recover_quarantined_default_stream(mut self) -> TensorStatus:
        # Expert-only recovery boundary. The caller MUST establish that all
        # native work touching this allocation was submitted to the retained
        # DeviceContext's stream, that no other stream/external consumer can
        # access it, and that no new producer can race this synchronization.
        # A synchronized stream is NOT a global device reset. This method
        # never marks untrusted FP32 payload data initialized after failure.
        if not self.lifetime_context:
            # A forged or dropped context owner cannot establish safe native
            # recovery, even if no poisoned-bit was set in metadata.
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        # A stream fence does not rehabilitate a malformed owner capacity or
        # epoch. Reject corrupted metadata before reporting it recovered.
        if self.capacity() < 0 or self._native_update_epoch < 0:
            return TensorStatus.INVALID_LAYOUT
        if self._native_update_pending:
            # The caller abandoned the begin/complete transaction. For this
            # explicitly unsafe default-stream recovery entry point, join
            # its retained issuing context before allowing any next writer.
            try:
                self.lifetime_context.value().synchronize()
            except:
                self.quiescence_unproven = True
                if not self.quarantined_context:
                    self.quarantined_context = Optional[DeviceContext](
                        self.lifetime_context.value().copy()
                    )
                return TensorStatus.GPU_QUIESCENCE_UNPROVEN
            self._native_update_pending = False
            self._native_update_was_readable = False
            self.initialized = False
            self.initialized_elements = -1
        if not self.quiescence_unproven and not self.quarantined_context:
            return TensorStatus.OK
        if not self.quarantined_context:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        try:
            self.quarantined_context.value().synchronize()
        except:
            # Preserve both native owner and context on any failed join.
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        self.quiescence_unproven = False
        self._native_update_pending = False
        self._native_update_was_readable = False
        self.initialized = False
        self.initialized_elements = -1
        # Release the retained context only after proven stream quiescence;
        # the buffer stays live and unreadable until an explicit full fill.
        var released_context = self.quarantined_context.take()
        return TensorStatus.OK

    def _begin_native_output_update(mut self) -> TensorResult[NativeWriteTicket]:
        # The returned epoch belongs to this storage's single outstanding
        # writer. Replaying a completion from an earlier write is rejected.
        if (not self.lifetime_context or self.quiescence_unproven
            or self.quarantined_context or self._native_update_pending):
            return TensorResult[NativeWriteTicket](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        # An owner can be reconstituted from mutable metadata. Reject a
        # corrupted physical capacity or negative epoch BEFORE issuing a
        # token; negative epochs could otherwise be reused across lifetimes.
        if self.capacity() < 0 or self._native_update_epoch < 0:
            return TensorResult[NativeWriteTicket](TensorStatus.INVALID_LAYOUT)
        # Refuse wraparound instead of recycling a ticket value: even a
        # pathological owner cannot mistake a stale transaction for a new one.
        if self._native_update_epoch == Int.MAX:
            return TensorResult[NativeWriteTicket](TensorStatus.OVERFLOW)
        var readable_before = self.is_initialized()
        self.initialized = False
        self.initialized_elements = -1
        self._native_update_was_readable = readable_before
        self._native_update_epoch += 1
        self._native_update_pending = True
        return TensorResult[NativeWriteTicket](NativeWriteTicket(
            self._native_update_epoch, readable_before,
        ))

    def _complete_native_output_update(
        mut self, issuing: DeviceContext, result: TensorStatus,
        ticket: NativeWriteTicket, complete_owner_write: Bool,
    ) -> TensorStatus:
        # This is an internal, post-event/post-join boundary; calling this
        # with OK after only an enqueue is forbidden. A source-level ticket
        # cannot stand in for the physical completion event.
        if not self._native_update_pending:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.lifetime_context or issuing != self.lifetime_context.value():
            # Preserve the pending owner so destruction quarantines its
            # allocation. A foreign context cannot authorize stream writes.
            return TensorStatus.DEVICE_MISMATCH
        if (ticket.epoch != self._native_update_epoch
            or ticket.previously_readable != self._native_update_was_readable):
            # Keep pending=true: reclaiming/reusing the allocation after a
            # stale completion would permit an in-flight GPU use-after-free.
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        self._native_update_pending = False
        self._native_update_was_readable = False
        if result == TensorStatus.OK:
            if ticket.previously_readable or complete_owner_write:
                return self._mark_complete_overwrite()
            return TensorStatus.UNINITIALIZED
        if device_write_failure_may_have_enqueued(result):
            self._invalidate_after_failed_write(
                issuing, result == TensorStatus.GPU_QUIESCENCE_UNPROVEN,
            )
            return result
        # A preflight-only failure cannot be used to restore an earlier
        # incomplete payload. The old owner was readable only if its *own*
        # begin ticket proved that fact before this write transaction.
        if ticket.previously_readable:
            var restored = self._mark_complete_overwrite()
            if restored != TensorStatus.OK:
                return restored
        return result

    def _mark_complete_overwrite(mut self) -> TensorStatus:
        # INTERNAL: caller already joined every asynchronous write that
        # initializes the complete logical FP32 region. This metadata flag
        # alone is not an event, native allocation identity or AOT proof.
        if (self.quiescence_unproven or self.quarantined_context
            or self._native_update_pending or not self.lifetime_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if self.capacity() < 0:
            return TensorStatus.INVALID_LAYOUT
        self.initialized = True
        self.initialized_elements = self.count
        return TensorStatus.OK

    @staticmethod
    def _unsafe_from_completed_buffer_checked(
        var buffer: DeviceBuffer[DType.float32], count: Int,
        ctx: DeviceContext,
    ) -> TensorResult[Self]:
        # INTERNAL UNSAFE ADOPTION BOUNDARY. The caller MUST already have
        # joined the buffer allocation and all preceding native writes using
        # a real DeviceEvent or a successfully synchronized device stream.
        # Capacity checks alone are NOT a GPU quiescence/ownership proof:
        # permitting an unjoined raw buffer here could let the normal owner
        # destructor free in-flight native memory. Never expose this helper
        # as a public checked factory without a context/event parameter.
        # A GPU buffer may be physically padded (notably for an empty tensor),
        # but its logical FP32 capacity must never exceed the allocation.
        if count < 0:
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        if count > Int.MAX // 4:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        if count > len(buffer):
            return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
        return TensorResult[Self](Self(buffer, count, ctx))

    @staticmethod
    def _allocate_owned_checked(
        ctx: DeviceContext, count: Int,
        initialize: Bool, value: Float32,
    ) -> TensorResult[Self]:
        # Canonical owner-aware allocation AND optional full initialization.
        # No extra synchronization step is added when initialize=True: MAX
        # enqueues allocation and fill in one ordered stream, then one Event
        # proves both completed before readable storage is returned.
        if count < 0:
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        if count > Int.MAX // 4:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        try:
            var buffer = ctx.enqueue_create_buffer[DType.float32](max(count, 1))
            try:
                if initialize and count > 0:
                    buffer.enqueue_fill(value)
                var completion = ctx.create_event()
                ctx.stream().record_event(completion)
                completion.synchronize()
            except:
                try:
                    ctx.stream().synchronize()
                except:
                    # Neither Event nor stream could establish quiescence.
                    # This buffer might still be used by DMA/GPU: never free
                    # or recycle it until the device runtime is recovered.
                    forget_deinit(buffer)
                    # The borrowed context must not be the only remaining
                    # owner while an asynchronous allocation/fill can run.
                    _retain_context_on_unproven_gpu_completion(ctx)
                    return TensorResult[Self](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
                # After a failed event/fill the stream join proves the final
                # use, NOT completion of DeviceBuffer's later queued free.
                # Destroy it before dropping the caller's context capability.
                _ = buffer
                try:
                    ctx.stream().synchronize()
                except:
                    _retain_context_on_unproven_gpu_completion(ctx)
                    return TensorResult[Self](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
                return TensorResult[Self](TensorStatus.TRANSFER_UNAVAILABLE)
            var validated = Self._unsafe_from_completed_buffer_checked(buffer, count, ctx)
            if not validated.is_ok():
                return validated
            var owner = validated.payload.take()
            if initialize or count == 0:
                var status = owner._mark_complete_overwrite()
                if status != TensorStatus.OK:
                    return TensorResult[Self](status)
            return TensorResult[Self](owner)
        except:
            # If allocation itself failed, no completed buffer owner was
            # returned. The context handles its internal pending resources.
            try:
                ctx.stream().synchronize()
            except:
                # The allocation call failed before returning a DeviceBuffer;
                # its internal queue may still be in flight. Retain a context
                # owner when even the fallback stream join cannot be proved.
                _retain_context_on_unproven_gpu_completion(ctx)
                return TensorResult[Self](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
            return TensorResult[Self](TensorStatus.ALLOCATION_FAILED)

    def capacity(self) -> Int:
        # DeviceStorage.count can be changed after checked construction.
        # A forged count must not become the capacity proof for a TensorLayout.
        if self.count < 0 or self.count > Int.MAX // 4:
            return -1
        if self.count > len(self.buffer):
            return -1
        return self.count

    def is_host_accessible(self) -> Bool:
        return False

    def read_host(self, index: Int) -> TensorResult[Float32]:
        if self.capacity() < 0:
            return TensorResult[Float32](TensorStatus.INVALID_LAYOUT)
        if index < 0 or index >= self.count:
            return TensorResult[Float32](TensorStatus.OUT_OF_BOUNDS)
        return TensorResult[Float32](TensorStatus.BACKEND_UNAVAILABLE)
