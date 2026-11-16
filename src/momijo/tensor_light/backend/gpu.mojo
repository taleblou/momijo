# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR BACKEND / GPU BACKEND
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Backend Availability
# Module:        momijo.tensor_light.backend.gpu
# Source:        src/momijo/tensor_light/backend/gpu.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines GPU backend in the backend availability layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `unsafe_enqueue_add_fp32`,
#   `unsafe_enqueue_mul_fp32`, `unsafe_enqueue_sub_fp32`,
#   `unsafe_enqueue_div_fp32`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Select or report supported execution backends explicitly.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `unsafe_enqueue_add_fp32`,
#     `unsafe_enqueue_mul_fp32`, `unsafe_enqueue_sub_fp32`,
#     `unsafe_enqueue_div_fp32`, `unsafe_enqueue_maximum_fp32` and related
#     symbols.
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

# Target lowering of the SAME immutable ExecutionPlan. Precompiled TileTensor
# specializations are enqueued through the public MAX 26.6 host API.
# Every unsafe_enqueue_* function returns a completion event but retains NO
# owners. Its caller must keep its buffers and context alive until completion.
# Prefer run_* methods, which join the event before returning a borrowed owner.
# An unbound DeviceExternalFunction is UNSAFE even on a synchronous path:
# its code, operation, architecture, and ABI are not authenticated here.
from max.gpu.host import DeviceContext, DeviceExternalFunction, DeviceBuffer, DeviceEvent
from layout import TileTensor
from ..internal.status import TensorStatus, TensorResult
from ..execution.execution_plan import ExecutionPlan, ExecutionPlacement
from ..kernels.gpu.elementwise import gpu_add_fp32, gpu_mul_fp32, gpu_sub_fp32, gpu_div_fp32
from ..kernels.gpu.elementwise import gpu_maximum_fp32, gpu_minimum_fp32
from ..kernels.gpu.elementwise import gpu_unary_fp32
from ..kernels.gpu.reduction import gpu_reduce_rows_op_fp32
from ..kernels.gpu.copy import gpu_copy_fp32, gpu_fill_fp32
from ..kernels.gpu.matrix import gpu_mm_strict_fp32, gpu_mm_simt_col4_fp32, gpu_gemv_row_fp32
from ..dispatch.specialization import BinaryAOTArtifactIdentity
from ..core.tensor import Tensor
from ..storage.device_storage import DeviceStorage
from ..kernels.launch import gpu_grid_1d
from ..transfer.device_transfer import device_fp32_regions_disjoint


# RELEASE JIT CONTRACT (UASDEA): source-instantiated GPU entry points are
# development-only and deny real work unless the caller explicitly opts in.
# A native release must call the identity-bound DeviceExternalFunction paths
# with a separately authenticated offline image. No silent recompilation or
# source fallback is allowed when the native artifact is unavailable.
# Zero-element operations are exempt: they submit no source kernel.

# Exactly one checked mapping boundary for every native, device-resident
# launcher below. There is no hidden transfer stage in these kernels.
# The distinct copy is consumed for all subsequent lowering decisions.
def _checked_native_device_plan(plan: ExecutionPlan) -> TensorResult[ExecutionPlan]:
    # The same transfer-free native ABI gate is used by source kernels and
    # external AOT adapters. This also avoids accepting a plan checked under
    # weaker GPU_ONLY semantics in one backend but not the other.
    return plan.checked_native_device_snapshot()


def _checked_source_binary_variant(plan: ExecutionPlan) -> TensorStatus:
    # Source-instantiated binary SIMT kernels use generic variant 0 only.
    # External kernels have separately versioned identity metadata, and a
    # GEMM/reduction variant must never silently execute a binary primitive.
    if plan.aot_variant != 0:
        return TensorStatus.UNSUPPORTED
    return TensorStatus.OK

# A direct device-buffer lowering produces a device-resident tensor only.
# A host-resident result requires a separately lowered, event-ordered D2H.
# DeviceEvent proves completion, NOT the lifetime of independently-owned buffers.
def unsafe_enqueue_add_fp32[LO: AnyType, LA: AnyType, LB: AnyType](ctx: DeviceContext, plan: ExecutionPlan, out: TileTensor[DType.float32, LO, MutAnyOrigin], a: TileTensor[DType.float32, LA, MutAnyOrigin], b: TileTensor[DType.float32, LB, MutAnyOrigin], development_allow_runtime_compilation: Bool = False) raises -> TensorResult[DeviceEvent]:
    var plan_proof = _checked_native_device_plan(plan)
    if not plan_proof.is_ok():
        return TensorResult[DeviceEvent](plan_proof.status)
    var checked_plan = plan_proof.payload.value()
    if _checked_source_binary_variant(checked_plan) != TensorStatus.OK:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var count = checked_plan.total_items
    # Source kernels use 32-bit device indices; refuse truncation before enqueue.
    if count > 2147483647:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if count < 0:
        return TensorResult[DeviceEvent](TensorStatus.INVALID_SHAPE)
    if count == 0:
        # Even zero-work launch returns a valid stream-ordered completion.
        var completed = ctx.create_event()
        ctx.stream().record_event(completed)
        return TensorResult[DeviceEvent](completed)
    var threads = 256
    # Source enqueue may trigger MAX compilation. A release needs a pinned native AOT artifact.
    if not development_allow_runtime_compilation:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var geometry = gpu_grid_1d(count, threads)
    if not geometry.is_ok():
        return TensorResult[DeviceEvent](geometry.status)
    var blocks = geometry.payload.value().grid_blocks
    ctx.enqueue_function[gpu_add_fp32[LO, LA, LB]](out, a, b, Int32(count), grid_dim=blocks, block_dim=threads)
    var completion = ctx.create_event()
    ctx.stream().record_event(completion)
    return TensorResult[DeviceEvent](completion)

def unsafe_enqueue_mul_fp32[LO: AnyType, LA: AnyType, LB: AnyType](ctx: DeviceContext, plan: ExecutionPlan, out: TileTensor[DType.float32, LO, MutAnyOrigin], a: TileTensor[DType.float32, LA, MutAnyOrigin], b: TileTensor[DType.float32, LB, MutAnyOrigin], development_allow_runtime_compilation: Bool = False) raises -> TensorResult[DeviceEvent]:
    var plan_proof = _checked_native_device_plan(plan)
    if not plan_proof.is_ok():
        return TensorResult[DeviceEvent](plan_proof.status)
    var checked_plan = plan_proof.payload.value()
    if _checked_source_binary_variant(checked_plan) != TensorStatus.OK:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var count = checked_plan.total_items
    # Source kernels use 32-bit device indices; refuse truncation before enqueue.
    if count > 2147483647:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if count < 0:
        return TensorResult[DeviceEvent](TensorStatus.INVALID_SHAPE)
    if count == 0:
        # Even zero-work launch returns a valid stream-ordered completion.
        var completed = ctx.create_event()
        ctx.stream().record_event(completed)
        return TensorResult[DeviceEvent](completed)
    # Source enqueue may trigger MAX compilation. A release needs a pinned native AOT artifact.
    if not development_allow_runtime_compilation:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var geometry = gpu_grid_1d(count, 256)
    if not geometry.is_ok():
        return TensorResult[DeviceEvent](geometry.status)
    var blocks = geometry.payload.value().grid_blocks
    ctx.enqueue_function[gpu_mul_fp32[LO, LA, LB]](out, a, b, Int32(count), grid_dim=blocks, block_dim=256)
    var completion = ctx.create_event()
    ctx.stream().record_event(completion)
    return TensorResult[DeviceEvent](completion)

# These intrinsic GPU kernels bind the FP32 opcode in the source/AOT symbol.
# They do not accept an unchecked external DeviceExternalFunction pointer.
def unsafe_enqueue_sub_fp32[LO: AnyType, LA: AnyType, LB: AnyType](ctx: DeviceContext, plan: ExecutionPlan, out: TileTensor[DType.float32, LO, MutAnyOrigin], a: TileTensor[DType.float32, LA, MutAnyOrigin], b: TileTensor[DType.float32, LB, MutAnyOrigin], development_allow_runtime_compilation: Bool = False) raises -> TensorResult[DeviceEvent]:
    var plan_proof = _checked_native_device_plan(plan)
    if not plan_proof.is_ok():
        return TensorResult[DeviceEvent](plan_proof.status)
    var checked_plan = plan_proof.payload.value()
    if _checked_source_binary_variant(checked_plan) != TensorStatus.OK:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var count = checked_plan.total_items
    # Source kernels use 32-bit device indices; refuse truncation before enqueue.
    if count > 2147483647:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if count == 0:
        var completed = ctx.create_event()
        ctx.stream().record_event(completed)
        return TensorResult[DeviceEvent](completed)
    # Source enqueue may trigger MAX compilation. A release needs a pinned native AOT artifact.
    if not development_allow_runtime_compilation:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var geometry = gpu_grid_1d(count, 256)
    if not geometry.is_ok():
        return TensorResult[DeviceEvent](geometry.status)
    var blocks = geometry.payload.value().grid_blocks
    ctx.enqueue_function[gpu_sub_fp32[LO, LA, LB]](out, a, b, Int32(count), grid_dim=blocks, block_dim=256)
    var completion = ctx.create_event()
    ctx.stream().record_event(completion)
    return TensorResult[DeviceEvent](completion)


def unsafe_enqueue_div_fp32[LO: AnyType, LA: AnyType, LB: AnyType](ctx: DeviceContext, plan: ExecutionPlan, out: TileTensor[DType.float32, LO, MutAnyOrigin], a: TileTensor[DType.float32, LA, MutAnyOrigin], b: TileTensor[DType.float32, LB, MutAnyOrigin], development_allow_runtime_compilation: Bool = False) raises -> TensorResult[DeviceEvent]:
    var plan_proof = _checked_native_device_plan(plan)
    if not plan_proof.is_ok():
        return TensorResult[DeviceEvent](plan_proof.status)
    var checked_plan = plan_proof.payload.value()
    if _checked_source_binary_variant(checked_plan) != TensorStatus.OK:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var count = checked_plan.total_items
    # Source kernels use 32-bit device indices; refuse truncation before enqueue.
    if count > 2147483647:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if count == 0:
        var completed = ctx.create_event()
        ctx.stream().record_event(completed)
        return TensorResult[DeviceEvent](completed)
    # Source enqueue may trigger MAX compilation. A release needs a pinned native AOT artifact.
    if not development_allow_runtime_compilation:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var geometry = gpu_grid_1d(count, 256)
    if not geometry.is_ok():
        return TensorResult[DeviceEvent](geometry.status)
    var blocks = geometry.payload.value().grid_blocks
    ctx.enqueue_function[gpu_div_fp32[LO, LA, LB]](out, a, b, Int32(count), grid_dim=blocks, block_dim=256)
    var completion = ctx.create_event()
    ctx.stream().record_event(completion)
    return TensorResult[DeviceEvent](completion)


# Separate statically instantiated FP32 extrema kernels; no runtime
# operation switch or Float32-to-boolean tensor dtype conversion.
def unsafe_enqueue_maximum_fp32[LO: AnyType, LA: AnyType, LB: AnyType](ctx: DeviceContext, plan: ExecutionPlan, out: TileTensor[DType.float32, LO, MutAnyOrigin], a: TileTensor[DType.float32, LA, MutAnyOrigin], b: TileTensor[DType.float32, LB, MutAnyOrigin], development_allow_runtime_compilation: Bool = False) raises -> TensorResult[DeviceEvent]:
    var plan_proof = _checked_native_device_plan(plan)
    if not plan_proof.is_ok():
        return TensorResult[DeviceEvent](plan_proof.status)
    var checked_plan = plan_proof.payload.value()
    if _checked_source_binary_variant(checked_plan) != TensorStatus.OK:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var count = checked_plan.total_items
    # Source kernels use 32-bit device indices; refuse truncation before enqueue.
    if count > 2147483647:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if count == 0:
        var completed = ctx.create_event()
        ctx.stream().record_event(completed)
        return TensorResult[DeviceEvent](completed)
    # Source enqueue may trigger MAX compilation. A release needs a pinned native AOT artifact.
    if not development_allow_runtime_compilation:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var geometry = gpu_grid_1d(count, 256)
    if not geometry.is_ok():
        return TensorResult[DeviceEvent](geometry.status)
    var blocks = geometry.payload.value().grid_blocks
    ctx.enqueue_function[gpu_maximum_fp32[LO, LA, LB]](out, a, b, Int32(count), grid_dim=blocks, block_dim=256)
    var completion = ctx.create_event()
    ctx.stream().record_event(completion)
    return TensorResult[DeviceEvent](completion)


def unsafe_enqueue_minimum_fp32[LO: AnyType, LA: AnyType, LB: AnyType](ctx: DeviceContext, plan: ExecutionPlan, out: TileTensor[DType.float32, LO, MutAnyOrigin], a: TileTensor[DType.float32, LA, MutAnyOrigin], b: TileTensor[DType.float32, LB, MutAnyOrigin], development_allow_runtime_compilation: Bool = False) raises -> TensorResult[DeviceEvent]:
    var plan_proof = _checked_native_device_plan(plan)
    if not plan_proof.is_ok():
        return TensorResult[DeviceEvent](plan_proof.status)
    var checked_plan = plan_proof.payload.value()
    if _checked_source_binary_variant(checked_plan) != TensorStatus.OK:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var count = checked_plan.total_items
    # Source kernels use 32-bit device indices; refuse truncation before enqueue.
    if count > 2147483647:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if count == 0:
        var completed = ctx.create_event()
        ctx.stream().record_event(completed)
        return TensorResult[DeviceEvent](completed)
    # Source enqueue may trigger MAX compilation. A release needs a pinned native AOT artifact.
    if not development_allow_runtime_compilation:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var geometry = gpu_grid_1d(count, 256)
    if not geometry.is_ok():
        return TensorResult[DeviceEvent](geometry.status)
    var blocks = geometry.payload.value().grid_blocks
    ctx.enqueue_function[gpu_minimum_fp32[LO, LA, LB]](out, a, b, Int32(count), grid_dim=blocks, block_dim=256)
    var completion = ctx.create_event()
    ctx.stream().record_event(completion)
    return TensorResult[DeviceEvent](completion)


def unsafe_enqueue_fill_fp32[L: AnyType](ctx: DeviceContext, plan: ExecutionPlan, out: TileTensor[DType.float32, L, MutAnyOrigin], value: Float32, development_allow_runtime_compilation: Bool = False) raises -> TensorResult[DeviceEvent]:
    var plan_proof = _checked_native_device_plan(plan)
    if not plan_proof.is_ok():
        return TensorResult[DeviceEvent](plan_proof.status)
    var checked_plan = plan_proof.payload.value()
    var count = checked_plan.total_items
    # Source kernels use 32-bit device indices; refuse truncation before enqueue.
    if count > 2147483647:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if count < 0:
        return TensorResult[DeviceEvent](TensorStatus.INVALID_SHAPE)
    if count == 0:
        # Even zero-work launch returns a valid stream-ordered completion.
        var completed = ctx.create_event()
        ctx.stream().record_event(completed)
        return TensorResult[DeviceEvent](completed)
    # Source enqueue may trigger MAX compilation. A release needs a pinned native AOT artifact.
    if not development_allow_runtime_compilation:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var geometry = gpu_grid_1d(count, 256)
    if not geometry.is_ok():
        return TensorResult[DeviceEvent](geometry.status)
    var blocks = geometry.payload.value().grid_blocks
    ctx.enqueue_function[gpu_fill_fp32[L]](out, value, Int32(count), grid_dim=blocks, block_dim=256)
    var completion = ctx.create_event()
    ctx.stream().record_event(completion)
    return TensorResult[DeviceEvent](completion)

# Single AOT/SIMT device-resident FP32 lowering. Opcode is part of the
# compile-time specialization, so the caller cannot accidentally select a
# differently named external binary kernel through this entry point.
# This is NOT a replacement for the separately loaded offline AOT artifact
# registry used by host-result GPU/HYBRID lowerings.
def unsafe_enqueue_binary_fp32[operation: Int, LO: AnyType, LA: AnyType, LB: AnyType](
    ctx: DeviceContext, plan: ExecutionPlan,
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    development_allow_runtime_compilation: Bool = False,
) raises -> TensorResult[DeviceEvent]:
    comptime assert operation >= 0 and operation <= 5, "FP32 binary opcode must be 0..5"
    comptime if operation == 0:
        return unsafe_enqueue_add_fp32[LO, LA, LB](ctx, plan, out, a, b, development_allow_runtime_compilation)
    elif operation == 1:
        return unsafe_enqueue_sub_fp32[LO, LA, LB](ctx, plan, out, a, b, development_allow_runtime_compilation)
    elif operation == 2:
        return unsafe_enqueue_mul_fp32[LO, LA, LB](ctx, plan, out, a, b, development_allow_runtime_compilation)
    elif operation == 3:
        return unsafe_enqueue_div_fp32[LO, LA, LB](ctx, plan, out, a, b, development_allow_runtime_compilation)
    elif operation == 4:
        return unsafe_enqueue_maximum_fp32[LO, LA, LB](ctx, plan, out, a, b, development_allow_runtime_compilation)
    else:
        return unsafe_enqueue_minimum_fp32[LO, LA, LB](ctx, plan, out, a, b, development_allow_runtime_compilation)


# A borrowed-owner public call must join the completion token before it
# returns. Backend exceptions are reported through the common status API.
# Context-level recovery remains necessary if the stream itself cannot sync.
def _join_checked_device_event(
    ctx: DeviceContext, var queued: TensorResult[DeviceEvent]
) -> TensorStatus:
    if not queued.is_ok():
        return queued.status
    var completion = queued.payload.take()
    try:
        completion.synchronize()
        return TensorStatus.OK
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


def run_binary_fp32[operation: Int, LO: AnyType, LA: AnyType, LB: AnyType](
    ctx: DeviceContext, plan: ExecutionPlan,
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    development_allow_runtime_compilation: Bool = False,
) -> TensorStatus:
    # The validated empty GPU_ONLY plan has no native work, dependency, or
    # transfer. The synchronous owner-scoped API must not create a redundant
    # event or synchronize a stream for this call. Raw enqueue_* APIs still
    # return stream-ordered events as promised to their expert callers.
    var proven = _checked_native_device_plan(plan)
    if not proven.is_ok():
        return proven.status
    var safe_plan = proven.payload.value()
    if safe_plan.total_items == 0:
        return TensorStatus.OK
    try:
        var queued = unsafe_enqueue_binary_fp32[operation, LO, LA, LB](ctx, safe_plan, out, a, b, development_allow_runtime_compilation)
        return _join_checked_device_event(ctx, queued)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# UNSAFE ADVANCED UNBOUND AOT path. The caller MUST validate the loaded binary,
# operation, target architecture, variant, and FP32 ABI separately. Prefer
# unsafe_enqueue_bound_aot_binary_fp32() for a metadata-checked boundary. Neither
# path can authenticate untrusted native code by inspecting the function handle.
# Signature: (out_ptr, a_ptr, b_ptr, FP32 element_count).
def unsafe_enqueue_unbound_aot_binary_fp32(
    ctx: DeviceContext,
    artifact: DeviceExternalFunction,
    plan: ExecutionPlan,
    out: DeviceBuffer[DType.float32],
    a: DeviceBuffer[DType.float32],
    b: DeviceBuffer[DType.float32],
) raises -> TensorResult[DeviceEvent]:
    # A successful enqueue is NOT proof of completed device writes. Return
    # a stream-ordered completion event even for a zero-element operation.
    # The caller must retain all three buffer owners until that event completes.
    var plan_proof = _checked_native_device_plan(plan)
    if not plan_proof.is_ok():
        return TensorResult[DeviceEvent](plan_proof.status)
    var checked_plan = plan_proof.payload.value()
    var count = checked_plan.total_items
    if count < 0 or len(out) < count or len(a) < count or len(b) < count:
        return TensorResult[DeviceEvent](TensorStatus.SIZE_MISMATCH)
    if count > 0:
        # An out-of-place AOT binary kernel cannot use overlapping output
        # and input ranges. Validate physical spans before enqueue; the
        # presence of distinct DeviceBuffer handles is not an alias proof.
        if not device_fp32_regions_disjoint(out, count, a, count):
            return TensorResult[DeviceEvent](TensorStatus.ALIAS_UNKNOWN)
        if not device_fp32_regions_disjoint(out, count, b, count):
            return TensorResult[DeviceEvent](TensorStatus.ALIAS_UNKNOWN)
        var geometry = gpu_grid_1d(count, 256)
        if not geometry.is_ok():
            return TensorResult[DeviceEvent](geometry.status)
        var blocks = geometry.payload.value().grid_blocks
        ctx.enqueue_function(artifact, out, a, b, count, grid_dim=blocks, block_dim=256)
    var completion = ctx.create_event()
    ctx.stream().record_event(completion)
    return TensorResult[DeviceEvent](completion)


# Safe borrowed-owner entry points. Their synchronous completion guarantee
# prevents a caller from dropping or recycling a borrowed allocation while
# this operation is still in flight. Async enqueue_* remains advanced/untracked:
# its caller MUST externally retain the buffers until the DeviceEvent completes.
def unsafe_run_unbound_aot_binary_fp32(
    ctx: DeviceContext,
    artifact: DeviceExternalFunction,
    plan: ExecutionPlan,
    out: DeviceBuffer[DType.float32],
    a: DeviceBuffer[DType.float32],
    b: DeviceBuffer[DType.float32],
) -> TensorStatus:
    try:
        var queued = unsafe_enqueue_unbound_aot_binary_fp32(ctx, artifact, plan, out, a, b)
        return _join_checked_device_event(ctx, queued)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


def run_add_fp32[LO: AnyType, LA: AnyType, LB: AnyType](
    ctx: DeviceContext, plan: ExecutionPlan,
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    development_allow_runtime_compilation: Bool = False,
) -> TensorStatus:
    # The validated empty GPU_ONLY plan has no native work, dependency, or
    # transfer. The synchronous owner-scoped API must not create a redundant
    # event or synchronize a stream for this call. Raw enqueue_* APIs still
    # return stream-ordered events as promised to their expert callers.
    var proven = _checked_native_device_plan(plan)
    if not proven.is_ok():
        return proven.status
    var safe_plan = proven.payload.value()
    if safe_plan.total_items == 0:
        return TensorStatus.OK
    try:
        var queued = unsafe_enqueue_add_fp32(ctx, safe_plan, out, a, b, development_allow_runtime_compilation)
        return _join_checked_device_event(ctx, queued)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


def run_mul_fp32[LO: AnyType, LA: AnyType, LB: AnyType](
    ctx: DeviceContext, plan: ExecutionPlan,
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    development_allow_runtime_compilation: Bool = False,
) -> TensorStatus:
    # The validated empty GPU_ONLY plan has no native work, dependency, or
    # transfer. The synchronous owner-scoped API must not create a redundant
    # event or synchronize a stream for this call. Raw enqueue_* APIs still
    # return stream-ordered events as promised to their expert callers.
    var proven = _checked_native_device_plan(plan)
    if not proven.is_ok():
        return proven.status
    var safe_plan = proven.payload.value()
    if safe_plan.total_items == 0:
        return TensorStatus.OK
    try:
        var queued = unsafe_enqueue_mul_fp32(ctx, safe_plan, out, a, b, development_allow_runtime_compilation)
        return _join_checked_device_event(ctx, queued)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


def run_sub_fp32[LO: AnyType, LA: AnyType, LB: AnyType](
    ctx: DeviceContext, plan: ExecutionPlan,
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    development_allow_runtime_compilation: Bool = False,
) -> TensorStatus:
    # The validated empty GPU_ONLY plan has no native work, dependency, or
    # transfer. The synchronous owner-scoped API must not create a redundant
    # event or synchronize a stream for this call. Raw enqueue_* APIs still
    # return stream-ordered events as promised to their expert callers.
    var proven = _checked_native_device_plan(plan)
    if not proven.is_ok():
        return proven.status
    var safe_plan = proven.payload.value()
    if safe_plan.total_items == 0:
        return TensorStatus.OK
    try:
        var queued = unsafe_enqueue_sub_fp32(ctx, safe_plan, out, a, b, development_allow_runtime_compilation)
        return _join_checked_device_event(ctx, queued)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


def run_div_fp32[LO: AnyType, LA: AnyType, LB: AnyType](
    ctx: DeviceContext, plan: ExecutionPlan,
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    development_allow_runtime_compilation: Bool = False,
) -> TensorStatus:
    # The validated empty GPU_ONLY plan has no native work, dependency, or
    # transfer. The synchronous owner-scoped API must not create a redundant
    # event or synchronize a stream for this call. Raw enqueue_* APIs still
    # return stream-ordered events as promised to their expert callers.
    var proven = _checked_native_device_plan(plan)
    if not proven.is_ok():
        return proven.status
    var safe_plan = proven.payload.value()
    if safe_plan.total_items == 0:
        return TensorStatus.OK
    try:
        var queued = unsafe_enqueue_div_fp32(ctx, safe_plan, out, a, b, development_allow_runtime_compilation)
        return _join_checked_device_event(ctx, queued)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


def run_fill_fp32[L: AnyType](
    ctx: DeviceContext, plan: ExecutionPlan,
    out: TileTensor[DType.float32, L, MutAnyOrigin], value: Float32,
    development_allow_runtime_compilation: Bool = False,
) -> TensorStatus:
    # The validated empty GPU_ONLY plan has no native work, dependency, or
    # transfer. The synchronous owner-scoped API must not create a redundant
    # event or synchronize a stream for this call. Raw enqueue_* APIs still
    # return stream-ordered events as promised to their expert callers.
    var proven = _checked_native_device_plan(plan)
    if not proven.is_ok():
        return proven.status
    var safe_plan = proven.payload.value()
    if safe_plan.total_items == 0:
        return TensorStatus.OK
    try:
        var queued = unsafe_enqueue_fill_fp32(ctx, safe_plan, out, value, development_allow_runtime_compilation)
        return _join_checked_device_event(ctx, queued)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Checked, operation-bound adapter for externally compiled FP32 binary code.
# A loaded DeviceExternalFunction has no introspectable opcode or architecture:
# bind its loader-supplied identity at SETUP and recheck it at EVERY launch.
# Callers must not manufacture an identity unrelated to the loaded binary.
# A successful return means enqueue only; owners must outlive completion.
def unsafe_enqueue_bound_aot_binary_fp32[operation: Int](
    ctx: DeviceContext, artifact: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity, target_arch: Int,
    plan: ExecutionPlan, out: DeviceBuffer[DType.float32],
    a: DeviceBuffer[DType.float32], b: DeviceBuffer[DType.float32],
) raises -> TensorResult[DeviceEvent]:
    comptime assert operation >= 0 and operation <= 5, "FP32 opcode must be 0..5"
    # The flat ABI consumes already-resident device inputs and output.
    # Enforce the same no-implicit-transfer proof as every native GPU path.
    var plan_proof = _checked_native_device_plan(plan)
    if not plan_proof.is_ok():
        return TensorResult[DeviceEvent](plan_proof.status)
    var checked_plan = plan_proof.payload.value()
    var binding = identity.validate_for(operation, checked_plan.aot_variant, target_arch)
    if binding != TensorStatus.OK:
        return TensorResult[DeviceEvent](binding)
    if checked_plan.placement != ExecutionPlacement.GPU_ONLY or checked_plan.output_device != 1:
        return TensorResult[DeviceEvent](TensorStatus.DEVICE_MISMATCH)
    # This native FP32 ABI was compiled for exactly 256 threads. The generic
    # metadata validator accepts other legal block sizes for future ABIs,
    # but those are NOT compatible with this particular lowering.
    if identity.threads_per_block != 256:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var count = checked_plan.total_items
    if count < 0 or len(out) < count or len(a) < count or len(b) < count:
        return TensorResult[DeviceEvent](TensorStatus.SIZE_MISMATCH)
    if count > 0:
        # An out-of-place AOT binary kernel cannot use overlapping output
        # and input ranges. Validate physical spans before enqueue; the
        # presence of distinct DeviceBuffer handles is not an alias proof.
        if not device_fp32_regions_disjoint(out, count, a, count):
            return TensorResult[DeviceEvent](TensorStatus.ALIAS_UNKNOWN)
        if not device_fp32_regions_disjoint(out, count, b, count):
            return TensorResult[DeviceEvent](TensorStatus.ALIAS_UNKNOWN)
        var geometry = gpu_grid_1d(count, identity.threads_per_block)
        if not geometry.is_ok():
            return TensorResult[DeviceEvent](geometry.status)
        var blocks = geometry.payload.value().grid_blocks
        ctx.enqueue_function(
            artifact, out, a, b, count,
            grid_dim=blocks, block_dim=identity.threads_per_block,
        )
    var completion = ctx.create_event()
    ctx.stream().record_event(completion)
    return TensorResult[DeviceEvent](completion)


def run_bound_aot_binary_fp32[operation: Int](
    ctx: DeviceContext, artifact: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity, target_arch: Int,
    plan: ExecutionPlan, out: DeviceBuffer[DType.float32],
    a: DeviceBuffer[DType.float32], b: DeviceBuffer[DType.float32],
) -> TensorStatus:
    try:
        var queued = unsafe_enqueue_bound_aot_binary_fp32[operation](
            ctx, artifact, identity, target_arch, plan, out, a, b,
        )
        return _join_checked_device_event(ctx, queued)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Strict FP32 SIMT reference GEMM for device-resident C-order matrices.
# Matrices must be row-major with exact logical shapes (m,k), (k,n), (m,n).
# This is not a Tensor Core or TF32 kernel. It cannot silently use a
# different floating-point precision mode or generic strides.
def unsafe_enqueue_mm_strict_fp32[LC: AnyType, LA: AnyType, LB: AnyType](
    ctx: DeviceContext, plan: ExecutionPlan,
    out: TileTensor[DType.float32, LC, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    m: Int, k: Int, n: Int,
    development_allow_runtime_compilation: Bool = False,
) raises -> TensorResult[DeviceEvent]:
    var plan_proof = _checked_native_device_plan(plan)
    if not plan_proof.is_ok():
        return TensorResult[DeviceEvent](plan_proof.status)
    var checked_plan = plan_proof.payload.value()
    # The intrinsic strict-SIMT launcher implements variants 10 and 11 only.
    # Reject external (14) and future TF32 (12) plans rather than silently
    # executing the wrong AOT strategy under their identity.
    if checked_plan.aot_variant != 10 and checked_plan.aot_variant != 11:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if m < 0 or k < 0 or n < 0:
        return TensorResult[DeviceEvent](TensorStatus.INVALID_SHAPE)
    # Prove every product used to address the FP32 matrices representable
    # before the unchecked SIMT kernel is permitted to run.
    if n > 0 and m > Int.MAX // n:
        return TensorResult[DeviceEvent](TensorStatus.OVERFLOW)
    if k > 0 and m > Int.MAX // k:
        return TensorResult[DeviceEvent](TensorStatus.OVERFLOW)
    if n > 0 and k > Int.MAX // n:
        return TensorResult[DeviceEvent](TensorStatus.OVERFLOW)
    var output_elements = m * n
    if output_elements > Int.MAX // 4 or m * k > Int.MAX // 4 or k * n > Int.MAX // 4:
        return TensorResult[DeviceEvent](TensorStatus.OVERFLOW)
    # The source SIMT kernel computes FP32 element offsets using native Int.
    # Do not truncate host geometry to the fixed-width Int32 device ABI.
    if (output_elements > 0 and
        (m > 2147483647 or k > 2147483647 or n > 2147483647
        or output_elements > 2147483647 or m * k > 2147483647
        or k * n > 2147483647)):
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if checked_plan.total_items != output_elements:
        return TensorResult[DeviceEvent](TensorStatus.SIZE_MISMATCH)
    if output_elements > 0:
        if not development_allow_runtime_compilation:
            return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
        # AOT lowering choice is completed before enqueue. Variant 10 is a
        # four-column-per-lane strict SIMT kernel, not a tensor-core kernel.
        # Variant 11 (and older generic plans) retain the scalar reference.
        if checked_plan.aot_variant == 10 and n >= 8 and k >= 8:
            var tiles_per_row = n // 4
            if n % 4 != 0:
                tiles_per_row += 1
            # Each row owns disjoint columns, including the last partial tile.
            var tiles = m * tiles_per_row
            var geometry = gpu_grid_1d(tiles, 256)
            if not geometry.is_ok():
                return TensorResult[DeviceEvent](geometry.status)
            var blocks = geometry.payload.value().grid_blocks
            ctx.enqueue_function[gpu_mm_simt_col4_fp32[LC, LA, LB]](
                out, a, b, Int32(m), Int32(k), Int32(n), Int32(tiles_per_row),
                grid_dim=blocks, block_dim=256,
            )
        else:
            var geometry = gpu_grid_1d(output_elements, 256)
            if not geometry.is_ok():
                return TensorResult[DeviceEvent](geometry.status)
            var blocks = geometry.payload.value().grid_blocks
            ctx.enqueue_function[gpu_mm_strict_fp32[LC, LA, LB]](
                out, a, b, Int32(m), Int32(k), Int32(n), grid_dim=blocks, block_dim=256,
            )
    # For k == 0, the kernel writes zeros to every nonempty output element.
    # For m == 0 or n == 0 there is no output access at all.
    var completion = ctx.create_event()
    ctx.stream().record_event(completion)
    return TensorResult[DeviceEvent](completion)


def run_mm_strict_fp32[LC: AnyType, LA: AnyType, LB: AnyType](
    ctx: DeviceContext, plan: ExecutionPlan,
    out: TileTensor[DType.float32, LC, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    b: TileTensor[DType.float32, LB, MutAnyOrigin],
    m: Int, k: Int, n: Int,
    development_allow_runtime_compilation: Bool = False,
) -> TensorStatus:
    try:
        var queued = unsafe_enqueue_mm_strict_fp32[LC, LA, LB](ctx, plan, out, a, b, m, k, n, development_allow_runtime_compilation)
        return _join_checked_device_event(ctx, queued)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Safe synchronous AOT entry point with Tensor as the public data object,
# rather than naked DeviceBuffer handles. The three owners are borrowed for
# the full operation, and the function waits for the D2H/device completion
# event before the caller may drop or mutate any of them. NO implicit host
# migration, dtype promotion, or runtime kernel compilation is permitted.
# This requires C-order same-shape tensors: broadcasting and arbitrary
# strided device views need different explicitly specialized kernels.
def run_device_tensor_binary_fp32[rank: Int, operation: Int](
    ctx: DeviceContext,
    artifact: DeviceExternalFunction,
    identity: BinaryAOTArtifactIdentity,
    target_arch: Int,
    plan: ExecutionPlan,
    mut out: Tensor[rank, DeviceStorage],
    left: Tensor[rank, DeviceStorage],
    right: Tensor[rank, DeviceStorage],
) -> TensorStatus:
    comptime assert rank >= 0 and rank <= 8, "Tensor rank must be 0..8"
    comptime assert operation >= 0 and operation <= 5, "FP32 opcode must be 0..5"
    # Single canonical public facade; no duplicated descriptor verification or
    # launch loops between backend and owning Tensor methods.
    return left.device_binary_into[operation](
        ctx, artifact, identity, target_arch, plan, right, out,
    )


# AOT-specialized GPU unary lowering. The caller must supply the exact
# zero-copy layout, validated element count, and owner-lifetime contract.
# This is deliberately not an implicit Tensor.to_device() operation.
def unsafe_enqueue_unary_fp32[operation: Int, LO: AnyType, LI: AnyType](
    ctx: DeviceContext,
    plan: ExecutionPlan,
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    source: TileTensor[DType.float32, LI, MutAnyOrigin],
    development_allow_runtime_compilation: Bool = False,
) raises -> TensorResult[DeviceEvent]:
    comptime assert operation >= 0 and operation <= 8, "invalid GPU unary FP32 opcode"
    var plan_proof = _checked_native_device_plan(plan)
    if not plan_proof.is_ok():
        return TensorResult[DeviceEvent](plan_proof.status)
    var checked_plan = plan_proof.payload.value()
    # The source kernel has a fixed operation specialization. External AOT
    # unary opcodes 20..28 have a different ABI binding and must use their
    # identity-validated launcher rather than this source entry point.
    if checked_plan.aot_variant != 0:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    var count = checked_plan.total_items
    # Source kernels use 32-bit device indices; refuse truncation before enqueue.
    if count > 2147483647:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if count > 0:
        if not development_allow_runtime_compilation:
            return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
        var geometry = gpu_grid_1d(count, 256)
        if not geometry.is_ok():
            return TensorResult[DeviceEvent](geometry.status)
        var blocks = geometry.payload.value().grid_blocks
        ctx.enqueue_function[gpu_unary_fp32[operation, LO, LI]](
            out, source, Int32(count), grid_dim=blocks, block_dim=256
        )
    var completion = ctx.create_event()
    ctx.stream().record_event(completion)
    return TensorResult[DeviceEvent](completion)


def run_unary_fp32[operation: Int, LO: AnyType, LI: AnyType](
    ctx: DeviceContext,
    plan: ExecutionPlan,
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    source: TileTensor[DType.float32, LI, MutAnyOrigin],
    development_allow_runtime_compilation: Bool = False,
) -> TensorStatus:
    # The validated empty GPU_ONLY plan has no native work, dependency, or
    # transfer. The synchronous owner-scoped API must not create a redundant
    # event or synchronize a stream for this call. Raw enqueue_* APIs still
    # return stream-ordered events as promised to their expert callers.
    var proven = _checked_native_device_plan(plan)
    if not proven.is_ok():
        return proven.status
    var safe_plan = proven.payload.value()
    if safe_plan.total_items == 0:
        return TensorStatus.OK
    try:
        var queued = unsafe_enqueue_unary_fp32[operation, LO, LI](ctx, safe_plan, out, source, development_allow_runtime_compilation)
        return _join_checked_device_event(ctx, queued)
    except:
        try:
            ctx.stream().synchronize()
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        return TensorStatus.NATIVE_LAUNCH_FAILED


# Exact SOURCE-native GEMV lowering. This is a reference SIMT kernel, not a
# claim of maximum GEMV throughput. Output has `rows` FP32 elements; the A
# matrix has `rows * cols`, x has `cols`. The caller is responsible for exact
# flat C-order layout representability, nonaliasing output, owner lifetime and
# device-context identity. Nothing moves automatically between host/device.
def unsafe_enqueue_gemv_row_fp32[LY: AnyType, LA: AnyType, LX: AnyType](
    ctx: DeviceContext, plan: ExecutionPlan,
    out: TileTensor[DType.float32, LY, MutAnyOrigin],
    a: TileTensor[DType.float32, LA, MutAnyOrigin],
    x: TileTensor[DType.float32, LX, MutAnyOrigin],
    rows: Int, cols: Int,
    development_allow_runtime_compilation: Bool = False,
) raises -> TensorResult[DeviceEvent]:
    var verified = _checked_native_device_plan(plan)
    if not verified.is_ok():
        return TensorResult[DeviceEvent](verified.status)
    var selected = verified.payload.value()
    if selected.aot_variant != 0:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if rows < 0 or cols < 0 or selected.total_items != rows:
        return TensorResult[DeviceEvent](TensorStatus.SIZE_MISMATCH)
    if rows > Int.MAX // 4 or cols > Int.MAX // 4:
        return TensorResult[DeviceEvent](TensorStatus.OVERFLOW)
    if cols > 0 and rows > (Int.MAX // 4) // cols:
        return TensorResult[DeviceEvent](TensorStatus.OVERFLOW)
    # All source-device address products use a 32-bit indexable range.
    if (rows > 0 and
        (rows > 2147483647 or cols > 2147483647
        or (cols > 0 and rows > 2147483647 // cols))):
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if rows > 0:
        if not development_allow_runtime_compilation:
            return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
        var geometry = gpu_grid_1d(rows, 256)
        if not geometry.is_ok():
            return TensorResult[DeviceEvent](geometry.status)
        ctx.enqueue_function[gpu_gemv_row_fp32[LY, LA, LX]](
            out, a, x, Int32(rows), Int32(cols),
            grid_dim=geometry.payload.value().grid_blocks, block_dim=256,
        )
    var event = ctx.create_event()
    ctx.stream().record_event(event)
    return TensorResult[DeviceEvent](event)


# Output rank is one below the original rank; only the flattened last-axis
# native reduction is implemented. Ops outside 0..4 or without a validated
# physical out/in buffer remain unsupported at this low-level entry point.
def unsafe_enqueue_reduce_rows_op_fp32[operation: Int, LO: AnyType, LI: AnyType](
    ctx: DeviceContext, plan: ExecutionPlan,
    out: TileTensor[DType.float32, LO, MutAnyOrigin],
    source: TileTensor[DType.float32, LI, MutAnyOrigin],
    rows: Int, cols: Int,
    development_allow_runtime_compilation: Bool = False,
) raises -> TensorResult[DeviceEvent]:
    comptime assert operation >= 0 and operation <= 4, "FP32 reduction operation"
    var verified = _checked_native_device_plan(plan)
    if not verified.is_ok():
        return TensorResult[DeviceEvent](verified.status)
    var selected = verified.payload.value()
    if selected.aot_variant != 0:
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if rows < 0 or cols < 0 or selected.total_items != rows:
        return TensorResult[DeviceEvent](TensorStatus.SIZE_MISMATCH)
    if rows > Int.MAX // 4 or cols > Int.MAX // 4:
        return TensorResult[DeviceEvent](TensorStatus.OVERFLOW)
    if cols > 0 and rows > (Int.MAX // 4) // cols:
        return TensorResult[DeviceEvent](TensorStatus.OVERFLOW)
    comptime if operation >= 2:
        if rows > 0 and cols == 0:
            return TensorResult[DeviceEvent](TensorStatus.SIZE_MISMATCH)
    # All source-device address products use a 32-bit indexable range.
    if (rows > 0 and
        (rows > 2147483647 or cols > 2147483647
        or (cols > 0 and rows > 2147483647 // cols))):
        return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
    if rows > 0:
        if not development_allow_runtime_compilation:
            return TensorResult[DeviceEvent](TensorStatus.UNSUPPORTED)
        var geometry = gpu_grid_1d(rows, 256)
        if not geometry.is_ok():
            return TensorResult[DeviceEvent](geometry.status)
        ctx.enqueue_function[gpu_reduce_rows_op_fp32[operation, LO, LI]](
            out, source, Int32(rows), Int32(cols),
            grid_dim=geometry.payload.value().grid_blocks, block_dim=256,
        )
    var event = ctx.create_event()
    ctx.stream().record_event(event)
    return TensorResult[DeviceEvent](event)
