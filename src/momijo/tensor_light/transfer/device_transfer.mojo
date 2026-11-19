# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR TRANSFER / HOST AND DEVICE TRANSFERS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Transfers
# Module:        momijo.tensor_light.transfer.device_transfer
# Source:        src/momijo/tensor_light/transfer/device_transfer.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines host and device transfers in the tensor transfers layer of the
#   FP32 Light tensor implementation.
#
#   Public declarations include `device_fp32_regions_disjoint`,
#   `validate_binary_fp32_device_owners`,
#   `validate_binary_fp32_pipeline_device_owners`, `unsafe_enqueue_h2d_fp32`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Plan or perform data copies and device transfers safely.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as
#     `device_fp32_regions_disjoint`, `validate_binary_fp32_device_owners`,
#     `validate_binary_fp32_pipeline_device_owners`,
#     `unsafe_enqueue_h2d_fp32`, `unsafe_enqueue_d2h_fp32` and related
#     symbols.
#   - Reject unsupported target or unverified completion states.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate host/device accessibility, bounds, and transfer direction.
#   - Preserve required completion and ownership lifetimes.
#   - Reject unsupported or unauthenticated execution paths explicitly.
#
# Performance Objectives:
#   - Use contiguous transfer paths when layouts permit.
#   - Minimize redundant staging or host/device round trips.
#   - Avoid implicit host/device migration or compilation at execution time.
#
# Maintainability:
#   - Keep checked transfer and unsafe enqueue APIs distinct.
#   - Test overlapping regions, failures, and asynchronous completion.
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

# Explicit FP32 host/device transfer adapters for MAX 26.6.
# The internal _enqueue_*_untracked helpers never establish ownership or
# synchronization contracts and must not be exported as user-facing APIs.
# A completion event is part of the asynchronous operation contract.
# A returned DeviceEvent does NOT own source/destination allocations. Expert
# enqueue APIs are therefore explicitly named unsafe_enqueue_*; normal users
# should call the synchronizing copy_* functions or a scoped Tensor method.
# On errors, unsafe enqueue synchronizes the stream before returning failure,
# because a failure to record its event may occur AFTER copy submission.
# This adapter copies WHOLE equal-length buffers; subrange transfers need a
# separately validated slice-aware path and must not silently copy extra data.
# Both buffer owners MUST outlive the event; do not return the allocation to a
# memory pool before the event has completed. No implicit host fallback.
from std.memory import forget_deinit
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer, DeviceEvent
from ..internal.status import TensorResult, TensorStatus


def _enqueue_h2d_untracked(ctx: DeviceContext, dst: DeviceBuffer[DType.float32], src: HostBuffer[DType.float32], count: Int) raises -> TensorStatus:
    if count < 0 or count > Int.MAX // 4:
        return TensorStatus.SIZE_MISMATCH
    # Empty Tensor owners may have one physically allocated padding element.
    # A zero-byte transfer never examines or enqueues that allocation.
    if count == 0:
        return TensorStatus.OK
    if len(dst) != count or len(src) != count:
        return TensorStatus.SIZE_MISMATCH
    ctx.enqueue_copy(dst_buf=dst, src_buf=src)
    return TensorStatus.OK


def _enqueue_d2h_untracked(ctx: DeviceContext, dst: HostBuffer[DType.float32], src: DeviceBuffer[DType.float32], count: Int) raises -> TensorStatus:
    if count < 0 or count > Int.MAX // 4:
        return TensorStatus.SIZE_MISMATCH
    # Empty Tensor owners may have one physically allocated padding element.
    # A zero-byte transfer never examines or enqueues that allocation.
    if count == 0:
        return TensorStatus.OK
    if len(dst) != count or len(src) != count:
        return TensorStatus.SIZE_MISMATCH
    ctx.enqueue_copy(dst_buf=dst, src_buf=src)
    return TensorStatus.OK


def device_fp32_regions_disjoint(
    dst: DeviceBuffer[DType.float32], dst_count: Int,
    src: DeviceBuffer[DType.float32], src_count: Int,
) -> Bool:
    # The region sizes can differ (GEMM inputs and outputs, row reduction).
    # Both buffers are required to belong to the SAME DeviceContext/address
    # space by the calling adapter; an unknown context must not use this proof.
    if dst_count < 0 or src_count < 0:
        return False
    if dst_count > len(dst) or src_count > len(src):
        return False
    # Never use an empty-region shortcut to bless malformed huge FP32
    # lengths; all arithmetic must first be representable in bytes.
    if dst_count > Int.MAX // 4 or src_count > Int.MAX // 4:
        return False
    if dst_count == 0 or src_count == 0:
        return True
    var dst_address = Int(dst.unsafe_ptr())
    var src_address = Int(src.unsafe_ptr())
    if dst_address <= 0 or src_address <= 0:
        return False
    var dst_bytes = dst_count * 4
    var src_bytes = src_count * 4
    if dst_address > Int.MAX - dst_bytes or src_address > Int.MAX - src_bytes:
        return False
    if dst_address < src_address:
        return src_address - dst_address >= dst_bytes
    if src_address < dst_address:
        return dst_address - src_address >= src_bytes
    return False


def validate_binary_fp32_device_owners(
    out: DeviceBuffer[DType.float32],
    left: DeviceBuffer[DType.float32],
    right: DeviceBuffer[DType.float32],
    count: Int,
) -> TensorStatus:
    # Ownership-aware HOST->GPU binary lowering: the WRITABLE result buffer
    # must be disjoint from both readable FP32 input regions. The two READ-
    # ONLY inputs may legally alias one another (x + x, reused weights).
    # This is stronger than checking synthetic DAG buffer identifiers. The
    # caller is responsible for having allocated all three in ONE live
    # DeviceContext and for keeping them alive until the completion event.
    # Do not use this as a cross-device pointer-identity or AOT-code proof.
    if count < 0 or count > Int.MAX // 4:
        return TensorStatus.INVALID_SHAPE
    if count > len(out) or count > len(left) or count > len(right):
        return TensorStatus.SIZE_MISMATCH
    if count == 0:
        return TensorStatus.OK
    if not device_fp32_regions_disjoint(out, count, left, count):
        return TensorStatus.ALIAS_UNKNOWN
    if not device_fp32_regions_disjoint(out, count, right, count):
        return TensorStatus.ALIAS_UNKNOWN
    return TensorStatus.OK


def validate_binary_fp32_pipeline_device_owners(
    out0: DeviceBuffer[DType.float32],
    left0: DeviceBuffer[DType.float32],
    right0: DeviceBuffer[DType.float32],
    out1: DeviceBuffer[DType.float32],
    left1: DeviceBuffer[DType.float32],
    right1: DeviceBuffer[DType.float32],
    count: Int,
) -> TensorStatus:
    # A double-buffered pipeline keeps BOTH GPU slots live before either
    # slot's completion fence. Pairwise validation within each slot does not
    # prove that its output or inputs are distinct from the other slot's
    # buffers. Validate all nine cross-slot pairs with the canonical checked
    # device-address-range predicate, before the FIRST GPU enqueue.
    #
    # This is a native device-pointer range proof for ONE live DeviceContext,
    # not proof of a compiled kernel's ABI, a host-staging address, or a
    # completed asynchronous event. Read-only inputs within one slot may
    # still alias, as allowed by validate_binary_fp32_device_owners().
    var first = validate_binary_fp32_device_owners(
        out0, left0, right0, count,
    )
    if first != TensorStatus.OK:
        return first
    var second = validate_binary_fp32_device_owners(
        out1, left1, right1, count,
    )
    if second != TensorStatus.OK:
        return second
    if count == 0:
        return TensorStatus.OK
    if not device_fp32_regions_disjoint(out0, count, out1, count):
        return TensorStatus.ALIAS_UNKNOWN
    if not device_fp32_regions_disjoint(out0, count, left1, count):
        return TensorStatus.ALIAS_UNKNOWN
    if not device_fp32_regions_disjoint(out0, count, right1, count):
        return TensorStatus.ALIAS_UNKNOWN
    if not device_fp32_regions_disjoint(left0, count, out1, count):
        return TensorStatus.ALIAS_UNKNOWN
    if not device_fp32_regions_disjoint(left0, count, left1, count):
        return TensorStatus.ALIAS_UNKNOWN
    if not device_fp32_regions_disjoint(left0, count, right1, count):
        return TensorStatus.ALIAS_UNKNOWN
    if not device_fp32_regions_disjoint(right0, count, out1, count):
        return TensorStatus.ALIAS_UNKNOWN
    if not device_fp32_regions_disjoint(right0, count, left1, count):
        return TensorStatus.ALIAS_UNKNOWN
    if not device_fp32_regions_disjoint(right0, count, right1, count):
        return TensorStatus.ALIAS_UNKNOWN
    return TensorStatus.OK


def _d2d_ranges_proven_disjoint(
    dst: DeviceBuffer[DType.float32], src: DeviceBuffer[DType.float32], count: Int
) -> Bool:
    # One canonical GPU region predicate for transfer and kernel launch.
    return device_fp32_regions_disjoint(dst, count, src, count)


def _enqueue_d2d_untracked(ctx: DeviceContext, dst: DeviceBuffer[DType.float32], src: DeviceBuffer[DType.float32], count: Int) raises -> TensorStatus:
    if count < 0 or count > Int.MAX // 4:
        return TensorStatus.SIZE_MISMATCH
    # Empty Tensor owners may have one physically allocated padding element.
    # A zero-byte transfer never examines or enqueues that allocation.
    if count == 0:
        return TensorStatus.OK
    if len(dst) != count or len(src) != count:
        return TensorStatus.SIZE_MISMATCH
    # An ordinary async device-to-device copy has memcpy-like overlap
    # semantics. Reject a potentially aliased sub-buffer before enqueue.
    if not _d2d_ranges_proven_disjoint(dst, src, count):
        return TensorStatus.ALIAS_UNKNOWN
    ctx.enqueue_copy(dst_buf=dst, src_buf=src)
    return TensorStatus.OK


# Preconditions for all unsafe_enqueue_* functions: the source and
# destination owners remain alive, unchanged, and unavailable for pool reuse
# until the returned DeviceEvent has completed. This is an UNSAFE expert API.
# On failure AFTER a copy may have been enqueued, attempt to quiesce the
# stream before returning an error (including failed event creation). If
# synchronization itself fails with GPU_QUIESCENCE_UNPROVEN, the caller must
# quarantine both owners until device reset or recovery establishes safety.
def unsafe_enqueue_h2d_fp32(
    ctx: DeviceContext, dst: DeviceBuffer[DType.float32],
    src: HostBuffer[DType.float32], count: Int
) -> TensorResult[DeviceEvent]:
    if count < 0 or count > Int.MAX // 4 or (count > 0 and (len(dst) != count or len(src) != count)):
        return TensorResult[DeviceEvent](TensorStatus.SIZE_MISMATCH)
    try:
        var accepted = _enqueue_h2d_untracked(ctx, dst, src, count)
        if accepted != TensorStatus.OK:
            return TensorResult[DeviceEvent](accepted)
        var completion = ctx.create_event()
        ctx.stream().record_event(completion)
        return TensorResult[DeviceEvent](completion)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorResult[DeviceEvent](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
        return TensorResult[DeviceEvent](TensorStatus.TRANSFER_UNAVAILABLE)


def unsafe_enqueue_d2h_fp32(
    ctx: DeviceContext, dst: HostBuffer[DType.float32],
    src: DeviceBuffer[DType.float32], count: Int
) -> TensorResult[DeviceEvent]:
    if count < 0 or count > Int.MAX // 4 or (count > 0 and (len(dst) != count or len(src) != count)):
        return TensorResult[DeviceEvent](TensorStatus.SIZE_MISMATCH)
    try:
        var accepted = _enqueue_d2h_untracked(ctx, dst, src, count)
        if accepted != TensorStatus.OK:
            return TensorResult[DeviceEvent](accepted)
        var completion = ctx.create_event()
        ctx.stream().record_event(completion)
        return TensorResult[DeviceEvent](completion)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorResult[DeviceEvent](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
        return TensorResult[DeviceEvent](TensorStatus.TRANSFER_UNAVAILABLE)


def unsafe_enqueue_d2d_fp32(
    ctx: DeviceContext, dst: DeviceBuffer[DType.float32],
    src: DeviceBuffer[DType.float32], count: Int
) -> TensorResult[DeviceEvent]:
    if count < 0 or count > Int.MAX // 4 or (count > 0 and (len(dst) != count or len(src) != count)):
        return TensorResult[DeviceEvent](TensorStatus.SIZE_MISMATCH)
    try:
        var accepted = _enqueue_d2d_untracked(ctx, dst, src, count)
        if accepted != TensorStatus.OK:
            return TensorResult[DeviceEvent](accepted)
        var completion = ctx.create_event()
        ctx.stream().record_event(completion)
        return TensorResult[DeviceEvent](completion)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorResult[DeviceEvent](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
        return TensorResult[DeviceEvent](TensorStatus.TRANSFER_UNAVAILABLE)


# Synchronous copies have a completion guarantee on return. Normal AOT
# pipelines may call unsafe_enqueue_* only while owners are kept alive and
# event completion controls reuse. Synchronous adapters are the safe choice
# when caller scope cannot preserve owners through asynchronous completion.
def copy_h2d_fp32(
    ctx: DeviceContext, dst: DeviceBuffer[DType.float32],
    src: HostBuffer[DType.float32], count: Int
) -> TensorStatus:
    # The owners are borrowed until this function returns: even an error
    # after enqueue cannot release or reuse their memory before quiescence.
    try:
        var status = _enqueue_h2d_untracked(ctx, dst, src, count)
        if status != TensorStatus.OK:
            return status
        if count > 0:
            var completion = ctx.create_event()
            ctx.stream().record_event(completion)
            completion.synchronize()
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.TRANSFER_UNAVAILABLE


def copy_d2h_fp32(
    ctx: DeviceContext, dst: HostBuffer[DType.float32],
    src: DeviceBuffer[DType.float32], count: Int
) -> TensorStatus:
    # The owners are borrowed until this function returns: even an error
    # after enqueue cannot release or reuse their memory before quiescence.
    try:
        var status = _enqueue_d2h_untracked(ctx, dst, src, count)
        if status != TensorStatus.OK:
            return status
        if count > 0:
            var completion = ctx.create_event()
            ctx.stream().record_event(completion)
            completion.synchronize()
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.TRANSFER_UNAVAILABLE


def copy_d2d_fp32(
    ctx: DeviceContext, dst: DeviceBuffer[DType.float32],
    src: DeviceBuffer[DType.float32], count: Int
) -> TensorStatus:
    # The owners are borrowed until this function returns: even an error
    # after enqueue cannot release or reuse their memory before quiescence.
    try:
        var status = _enqueue_d2d_untracked(ctx, dst, src, count)
        if status != TensorStatus.OK:
            return status
        if count > 0:
            var completion = ctx.create_event()
            ctx.stream().record_event(completion)
            completion.synchronize()
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.TRANSFER_UNAVAILABLE


# An asynchronously-created HostBuffer/DeviceBuffer must not escape a
# completed transfer as a BARE value after its issuing DeviceContext dies.
# This move-only joined owner preserves the context for the entire returned
# buffer lifetime. Borrowing cannot outlive the completed owner; deliberately
# moving out the private buffer alone has no checked/supported public API.
struct CompletedFp32Transfer[Buffer: Movable](Movable):
    var _buffer: Self.Buffer
    var _context: DeviceContext

    def __init__(out self, var buffer: Self.Buffer, var context: DeviceContext):
        self._buffer = buffer
        self._context = context

    def borrow_buffer(ref self) -> ref[origin_of(self._buffer)] Self.Buffer:
        return self._buffer

    def __deinit__(deinit self):
        # A joined copy is not necessarily the last user of the borrowed
        # native destination. Join the owning stream before releasing it.
        # MAX buffer destructors enqueue ANOTHER asynchronous command, so
        # also drain AFTER buffer destruction and BEFORE context destruction.
        # These waits belong to final owner cleanup, never hot replay.
        try:
            self._context.stream().synchronize()
        except:
            forget_deinit(self._buffer)
            forget_deinit(self._context)
            return
        _ = self._buffer
        try:
            self._context.stream().synchronize()
        except:
            # A scheduled free has uncertain completion: keep the underlying
            # context alive even though the buffer owner was already dropped.
            forget_deinit(self._context)
            return
        _ = self._context


# Owner-retaining asynchronous transfer. Unlike unsafe_enqueue_* returning
# an event alone, this move-only handle owns BOTH allocations and the context
# until a completion wait. The destructor joins pending work before freeing
# its buffers, so dropping the handle does not silently cause use-after-free.
# A fatal backend failure preventing synchronization cannot establish a safe
# completion proof; recovery/quarantine remains the responsibility of the
# device runtime. No implicit fallback or cross-device pointer conversion.
struct OwnedAsyncFp32Transfer[Destination: Movable, Source: Movable](Movable):
    # There is deliberately NO caller-writable `joined` switch: even a
    # forged mutable flag must not permit premature buffer deallocation.
    # The leading underscores mark an expert-only internal ownership state.
    var _context: DeviceContext
    var _destination: Self.Destination
    var _source: Self.Source
    var _completion: DeviceEvent

    def __init__(
        out self, var context: DeviceContext,
        var destination: Self.Destination, var source: Self.Source,
        var completion: DeviceEvent,
    ):
        self._context = context
        self._destination = destination
        self._source = source
        self._completion = completion

    def wait(self) -> TensorStatus:
        try:
            self._completion.synchronize()
            return TensorStatus.OK
        except:
            # A stream fence may prove that the copy is no longer in flight,
            # but CANNOT prove that a failed event/copy wrote correct bytes.
            # In particular, do not let finish() publish an incomplete output.
            try:
                self._context.stream().synchronize()
                return TensorStatus.TRANSFER_UNAVAILABLE
            except:
                return TensorStatus.GPU_QUIESCENCE_UNPROVEN

    def finish(deinit self) -> TensorResult[CompletedFp32Transfer[Self.Destination]]:
        # A bare DeviceBuffer result could outlive the DeviceContext that
        # owns its allocator. Return a joined owner retaining BOTH resources
        # and expose only a lifetime-bound borrowed native buffer.
        comptime Joined = CompletedFp32Transfer[Self.Destination]
        var completed = self.wait()
        if completed != TensorStatus.OK:
            if completed == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
                # Double failure: deliberately quarantine ALL native owners.
                forget_deinit(self._source)
                forget_deinit(self._destination)
                forget_deinit(self._completion)
                forget_deinit(self._context)
            else:
                # Stream recovery proves copy completion, not completion of
                # the frees queued by HostBuffer/DeviceBuffer destruction.
                # Drain those frees before dropping the final native context.
                _ = self._source
                _ = self._destination
                _ = self._completion
                try:
                    self._context.stream().synchronize()
                except:
                    forget_deinit(self._context)
                    return TensorResult[Joined](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
                _ = self._context
            return TensorResult[Joined](completed)
        # On success the source/event may die only after the exact event
        # join. Transfer destination AND context as one move-only owner.
        _ = self._source
        _ = self._completion
        return TensorResult[Joined](Joined(self._destination, self._context))

    def __deinit__(deinit self):
        # An event wait, followed by a stream recovery wait, normally proves
        # the source and destination are no longer in use by the device.
        # On a DOUBLE failure there is NO completion proof. Do NOT drop the
        # owners or return their blocks to an allocator/pool: deliberately
        # retain their physical resources by forgetting their destructors.
        # This is a bounded-to-the-failed-operation leak, not a safe retry;
        # callers must reset/recreate the failed device context.
        try:
            self._completion.synchronize()
        except:
            try:
                self._context.stream().synchronize()
            except:
                forget_deinit(self._source)
                forget_deinit(self._destination)
                forget_deinit(self._completion)
                forget_deinit(self._context)
                return
        # The event join proves completion of the transfer, not completion
        # of the later native frees queued by buffer destructors. Keep the
        # context owning those frees until the second stream barrier succeeds.
        _ = self._source
        _ = self._destination
        _ = self._completion
        try:
            self._context.stream().synchronize()
        except:
            forget_deinit(self._context)
            return
        _ = self._context


# Consuming an invalid owned-transfer request still destroys its native
# source/destination owners. MAX buffer destructors may enqueue asynchronous
# frees: a rejected request must drain those frees BEFORE its final owning
# DeviceContext drops. This slow path never runs on a valid transfer.
def _retire_rejected_owned_fp32_transfer[Destination: Movable, Source: Movable](
    var ctx: DeviceContext, var destination: Destination,
    var source: Source, failure: TensorStatus,
) -> TensorStatus:
    # Owned buffers may already have work ordered on their issuing stream,
    # despite this request being rejected before its own enqueue.
    try:
        ctx.stream().synchronize()
    except:
        forget_deinit(source)
        forget_deinit(destination)
        forget_deinit(ctx)
        return TensorStatus.GPU_QUIESCENCE_UNPROVEN
    _ = source
    _ = destination
    # Their destructors schedule frees AFTER the first stream join.
    try:
        ctx.stream().synchronize()
    except:
        forget_deinit(ctx)
        return TensorStatus.GPU_QUIESCENCE_UNPROVEN
    return failure


def enqueue_owned_h2d_fp32(
    var ctx: DeviceContext,
    var destination: DeviceBuffer[DType.float32],
    var source: HostBuffer[DType.float32],
    count: Int,
) -> TensorResult[OwnedAsyncFp32Transfer[DeviceBuffer[DType.float32], HostBuffer[DType.float32]]]:
    comptime Pending = OwnedAsyncFp32Transfer[DeviceBuffer[DType.float32], HostBuffer[DType.float32]]
    if count < 0 or count > Int.MAX // 4 or (count > 0 and (len(destination) != count or len(source) != count)):
        return TensorResult[Pending](_retire_rejected_owned_fp32_transfer(
            ctx, destination, source, TensorStatus.SIZE_MISMATCH,
        ))
    # The ownership transfer takes place only AFTER the potentially
    # raising enqueue/event setup has completed; error recovery still owns
    # all buffers and the context and can synchronize safely.
    var event: DeviceEvent
    try:
        var accepted = _enqueue_h2d_untracked(ctx, destination, source, count)
        if accepted != TensorStatus.OK:
            # Preflight rejection (including aliased D2D buffers) does not
            # skip the owned buffers' post-destruction free barrier.
            return TensorResult[Pending](_retire_rejected_owned_fp32_transfer(
                ctx, destination, source, accepted,
            ))
        event = ctx.create_event()
        ctx.stream().record_event(event)
    except:
        try:
            ctx.stream().synchronize()
        except:
            # Copy submission may have succeeded before event recording
            # failed. A fatal stream failure provides no quiescence proof.
            # The only safe local fallback is to forfeit these owners rather
            # than deallocate memory which a device might still access.
            forget_deinit(source)
            forget_deinit(destination)
            forget_deinit(ctx)
            return TensorResult[Pending](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
        # A failed enqueue/event setup can still have enqueued a copy. The
        # first synchronize proved quiescence; the frees scheduled by owner
        # destruction need a SECOND join before releasing the context.
        _ = source
        _ = destination
        try:
            ctx.stream().synchronize()
        except:
            forget_deinit(ctx)
            return TensorResult[Pending](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
        return TensorResult[Pending](TensorStatus.TRANSFER_UNAVAILABLE)
    return TensorResult[Pending](Pending(ctx, destination, source, event))


def enqueue_owned_d2h_fp32(
    var ctx: DeviceContext,
    var destination: HostBuffer[DType.float32],
    var source: DeviceBuffer[DType.float32],
    count: Int,
) -> TensorResult[OwnedAsyncFp32Transfer[HostBuffer[DType.float32], DeviceBuffer[DType.float32]]]:
    comptime Pending = OwnedAsyncFp32Transfer[HostBuffer[DType.float32], DeviceBuffer[DType.float32]]
    if count < 0 or count > Int.MAX // 4 or (count > 0 and (len(destination) != count or len(source) != count)):
        return TensorResult[Pending](_retire_rejected_owned_fp32_transfer(
            ctx, destination, source, TensorStatus.SIZE_MISMATCH,
        ))
    # The ownership transfer takes place only AFTER the potentially
    # raising enqueue/event setup has completed; error recovery still owns
    # all buffers and the context and can synchronize safely.
    var event: DeviceEvent
    try:
        var accepted = _enqueue_d2h_untracked(ctx, destination, source, count)
        if accepted != TensorStatus.OK:
            # Preflight rejection (including aliased D2D buffers) does not
            # skip the owned buffers' post-destruction free barrier.
            return TensorResult[Pending](_retire_rejected_owned_fp32_transfer(
                ctx, destination, source, accepted,
            ))
        event = ctx.create_event()
        ctx.stream().record_event(event)
    except:
        try:
            ctx.stream().synchronize()
        except:
            # Copy submission may have succeeded before event recording
            # failed. A fatal stream failure provides no quiescence proof.
            # The only safe local fallback is to forfeit these owners rather
            # than deallocate memory which a device might still access.
            forget_deinit(source)
            forget_deinit(destination)
            forget_deinit(ctx)
            return TensorResult[Pending](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
        # A failed enqueue/event setup can still have enqueued a copy. The
        # first synchronize proved quiescence; the frees scheduled by owner
        # destruction need a SECOND join before releasing the context.
        _ = source
        _ = destination
        try:
            ctx.stream().synchronize()
        except:
            forget_deinit(ctx)
            return TensorResult[Pending](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
        return TensorResult[Pending](TensorStatus.TRANSFER_UNAVAILABLE)
    return TensorResult[Pending](Pending(ctx, destination, source, event))


def enqueue_owned_d2d_fp32(
    var ctx: DeviceContext,
    var destination: DeviceBuffer[DType.float32],
    var source: DeviceBuffer[DType.float32],
    count: Int,
) -> TensorResult[OwnedAsyncFp32Transfer[DeviceBuffer[DType.float32], DeviceBuffer[DType.float32]]]:
    comptime Pending = OwnedAsyncFp32Transfer[DeviceBuffer[DType.float32], DeviceBuffer[DType.float32]]
    if count < 0 or count > Int.MAX // 4 or (count > 0 and (len(destination) != count or len(source) != count)):
        return TensorResult[Pending](_retire_rejected_owned_fp32_transfer(
            ctx, destination, source, TensorStatus.SIZE_MISMATCH,
        ))
    # The ownership transfer takes place only AFTER the potentially
    # raising enqueue/event setup has completed; error recovery still owns
    # all buffers and the context and can synchronize safely.
    var event: DeviceEvent
    try:
        var accepted = _enqueue_d2d_untracked(ctx, destination, source, count)
        if accepted != TensorStatus.OK:
            # Preflight rejection (including aliased D2D buffers) does not
            # skip the owned buffers' post-destruction free barrier.
            return TensorResult[Pending](_retire_rejected_owned_fp32_transfer(
                ctx, destination, source, accepted,
            ))
        event = ctx.create_event()
        ctx.stream().record_event(event)
    except:
        try:
            ctx.stream().synchronize()
        except:
            # Copy submission may have succeeded before event recording
            # failed. A fatal stream failure provides no quiescence proof.
            # The only safe local fallback is to forfeit these owners rather
            # than deallocate memory which a device might still access.
            forget_deinit(source)
            forget_deinit(destination)
            forget_deinit(ctx)
            return TensorResult[Pending](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
        # A failed enqueue/event setup can still have enqueued a copy. The
        # first synchronize proved quiescence; the frees scheduled by owner
        # destruction need a SECOND join before releasing the context.
        _ = source
        _ = destination
        try:
            ctx.stream().synchronize()
        except:
            forget_deinit(ctx)
            return TensorResult[Pending](TensorStatus.GPU_QUIESCENCE_UNPROVEN)
        return TensorResult[Pending](TensorStatus.TRANSFER_UNAVAILABLE)
    return TensorResult[Pending](Pending(ctx, destination, source, event))
