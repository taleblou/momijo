# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR EXECUTION / EXECUTION CONTEXT
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Execution Planning and AOT
# Module:        momijo.tensor_light.execution.execution_context
# Source:        src/momijo/tensor_light/execution/execution_context.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines execution context within checked AOT execution planning for the
#   FP32 Light implementation.
#
#   Public declarations include `ExecutionPreference`, `CpuExecutionLane`,
#   `GpuExecutionLane`, `ExecutionContext`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Provide device and execution-context metadata for CPU, GPU, or hybrid
#     routing as defined by this module.
#   - Expose module-level declarations including `ExecutionPreference`,
#     `CpuExecutionLane`, `GpuExecutionLane`, `ExecutionContext`.
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

# Context owns MAX device queue; plans reference lane policy but never own
# tensor buffers. GPU operations remain asynchronous until explicit join.
from max.gpu.host import DeviceContext, DeviceStream, DeviceEvent
from std.collections import Optional
from ..internal.status import TensorResult, TensorStatus
from ..dtype.math_mode import Fp32MathMode, ReductionAccuracyMode

@fieldwise_init
struct ExecutionPreference(Copyable):
    var code: Int
    comptime CPU = Self(0)
    comptime GPU = Self(1)
    comptime AUTO = Self(2)

struct CpuExecutionLane(Copyable):
    var minimum_parallel_items: Int
    def __init__(out self, minimum_parallel_items: Int = 65536):
        self.minimum_parallel_items = max(1, minimum_parallel_items)

struct GpuExecutionLane(Movable):
    var context: DeviceContext
    def __init__(out self, var context: DeviceContext):
        self.context = context

    def compute_stream(ref self) raises -> DeviceStream:
        return self.context.stream()

    def create_transfer_stream(ref self, priority: Int = 0) raises -> DeviceStream:
        return self.context.create_stream(priority=priority)

    def create_completion_event(ref self) raises -> DeviceEvent:
        return self.context.create_event()

struct ExecutionContext(Movable):
    var cpu_lane: CpuExecutionLane
    var gpu_lane: Optional[GpuExecutionLane]
    var preference: ExecutionPreference
    # FP32 storage does not imply TF32 arithmetic permission. Keep per-
    # operation compute/reduction policies in the explicit execution context,
    # never in a global mutable dtype registry or Tensor ownership identity.
    var fp32_mode: Fp32MathMode
    var reduction_accuracy: ReductionAccuracyMode

    def __init__(
        out self, cpu_lane: CpuExecutionLane,
        gpu_lane: Optional[GpuExecutionLane], preference: ExecutionPreference,
        math_mode: Fp32MathMode = Fp32MathMode.STRICT_FP32,
        accuracy_mode: ReductionAccuracyMode = ReductionAccuracyMode.DETERMINISTIC_PAIRWISE,
    ):
        self.cpu_lane = cpu_lane.copy()
        if gpu_lane:
            self.gpu_lane = GpuExecutionLane(gpu_lane.value().context)
        else:
            self.gpu_lane = None
        self.preference = preference.copy()
        self.fp32_mode = math_mode
        self.reduction_accuracy = accuracy_mode.copy()

    @staticmethod
    def cpu_only() -> Self:
        return Self(CpuExecutionLane(), None, materialize[ExecutionPreference.CPU]())

    def has_accelerator(self) -> Bool:
        return Bool(self.gpu_lane)

    def math_mode(self) -> Fp32MathMode:
        return self.fp32_mode

    def reduction_mode(self) -> ReductionAccuracyMode:
        return self.reduction_accuracy.copy()

    def supports_hybrid(self) -> Bool:
        # Indicates only that both execution lanes were supplied and AUTO is
        # requested. Workload-specific transfer, alias, memory residency and
        # profitability MUST still pass the immutable ExecutionPlan gates.
        return Bool(self.gpu_lane) and self.preference.code == ExecutionPreference.AUTO.code

    def synchronize_gpu(ref self) -> TensorStatus:
        # The CPU lane uses synchronous MAX parallel primitives. Join only
        # the GPU default stream; do not unconditionally wait the device.
        # If the wait fails, owners with pending GPU writes stay poisoned.
        if not Bool(self.gpu_lane):
            return TensorStatus.OK
        try:
            self.gpu_lane.value().context.stream().synchronize()
            return TensorStatus.OK
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN

    def unsafe_synchronize_declared_streams(
        ref self, transfer: Optional[DeviceStream], auxiliary: Optional[DeviceStream],
    ) -> TensorStatus:
        # Expert-only: the caller proves that these are exactly ALL nondefault
        # streams which accessed the owners in THIS closed-world workload,
        # and that no external producer/consumer is outstanding. Never use
        # this as a substitute for source/target allocation provenance.
        # Join each explicit stream, then the default queue, before returning
        # a successful narrow quiescence status. A failed join poisons the
        # corresponding allocations at the caller's owner boundary.
        if not Bool(self.gpu_lane):
            if transfer or auxiliary:
                return TensorStatus.DEVICE_MISMATCH
            return TensorStatus.OK
        if self.validate() != TensorStatus.OK:
            return TensorStatus.INVALID_LAYOUT
        try:
            if transfer:
                transfer.value().synchronize()
            if auxiliary:
                auxiliary.value().synchronize()
            self.gpu_lane.value().context.stream().synchronize()
            return TensorStatus.OK
        except:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN

    def synchronize_all(ref self) -> TensorStatus:
        # Fail closed: this context does not own/track ALL auxiliary streams
        # created by expert callers, so a default-stream wait cannot prove a
        # full host/device join. CPU-only execution is already synchronous.
        # A future full stream/event owner registry may implement this safely.
        if Bool(self.gpu_lane):
            return TensorStatus.UNSUPPORTED
        return TensorStatus.OK

    def validate(self) -> TensorStatus:
        # External code can mutate these value fields after construction.
        # Recheck every policy boundary before selecting an AOT device lane.
        if self.cpu_lane.minimum_parallel_items < 1:
            return TensorStatus.INVALID_LAYOUT
        if self.preference.code < 0 or self.preference.code > 2:
            return TensorStatus.INVALID_LAYOUT
        if self.preference.code == ExecutionPreference.GPU.code and not Bool(self.gpu_lane):
            return TensorStatus.BACKEND_UNAVAILABLE
        if self.fp32_mode.code < 0 or self.fp32_mode.code > 3:
            return TensorStatus.MATH_MODE_UNSUPPORTED
        if self.reduction_accuracy.code < 0 or self.reduction_accuracy.code > 1:
            return TensorStatus.MATH_MODE_UNSUPPORTED
        return TensorStatus.OK

    @staticmethod
    def gpu_only(
        var gpu_context: DeviceContext,
        math_mode: Fp32MathMode = Fp32MathMode.STRICT_FP32,
        accuracy_mode: ReductionAccuracyMode = ReductionAccuracyMode.DETERMINISTIC_PAIRWISE,
    ) -> Self:
        return Self(
            CpuExecutionLane(),
            Optional[GpuExecutionLane](GpuExecutionLane(gpu_context)),
            materialize[ExecutionPreference.GPU](), math_mode, accuracy_mode,
        )

    @staticmethod
    def auto_with_gpu(
        var gpu_context: DeviceContext, cpu_grain: Int = 65536,
        math_mode: Fp32MathMode = Fp32MathMode.STRICT_FP32,
        accuracy_mode: ReductionAccuracyMode = ReductionAccuracyMode.DETERMINISTIC_PAIRWISE,
    ) -> Self:
        # A single context can supply both target mappings; placement is
        # decided by an immutable workload-specific plan before the launch.
        return Self(
            CpuExecutionLane(cpu_grain),
            Optional[GpuExecutionLane](GpuExecutionLane(gpu_context)),
            materialize[ExecutionPreference.AUTO](), math_mode, accuracy_mode,
        )


def make_cpu_context(
    cpu_grain: Int = 65536,
    math_mode: Fp32MathMode = Fp32MathMode.STRICT_FP32,
    accuracy_mode: ReductionAccuracyMode = ReductionAccuracyMode.DETERMINISTIC_PAIRWISE,
) -> ExecutionContext:
    return ExecutionContext(
        CpuExecutionLane(cpu_grain), None, materialize[ExecutionPreference.CPU](),
        math_mode, accuracy_mode,
    )


def make_gpu_context(
    var gpu_context: DeviceContext,
    math_mode: Fp32MathMode = Fp32MathMode.STRICT_FP32,
    accuracy_mode: ReductionAccuracyMode = ReductionAccuracyMode.DETERMINISTIC_PAIRWISE,
) -> ExecutionContext:
    return ExecutionContext.gpu_only(gpu_context, math_mode, accuracy_mode)


def make_hybrid_context(
    var gpu_context: DeviceContext, cpu_grain: Int = 65536,
    math_mode: Fp32MathMode = Fp32MathMode.STRICT_FP32,
    accuracy_mode: ReductionAccuracyMode = ReductionAccuracyMode.DETERMINISTIC_PAIRWISE,
) -> ExecutionContext:
    return ExecutionContext.auto_with_gpu(
        gpu_context, cpu_grain, math_mode, accuracy_mode,
    )
