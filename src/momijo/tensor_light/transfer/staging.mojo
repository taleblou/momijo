# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR TRANSFER / TRANSFER STAGING
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Transfers
# Module:        momijo.tensor_light.transfer.staging
# Source:        src/momijo/tensor_light/transfer/staging.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines transfer staging in the tensor transfers layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `StagingPlan`, `plan_staging`,
#   `validate_staging_capacity`, `validate_staging_source_layout`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Plan or perform data copies and device transfers safely.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `StagingPlan`,
#     `plan_staging`, `validate_staging_capacity`,
#     `validate_staging_source_layout`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate host/device accessibility, bounds, and transfer direction.
#   - Preserve required completion and ownership lifetimes.
#
# Performance Objectives:
#   - Use contiguous transfer paths when layouts permit.
#   - Minimize redundant staging or host/device round trips.
#
# Maintainability:
#   - Keep checked transfer and unsafe enqueue APIs distinct.
#   - Test overlapping regions, failures, and asynchronous completion.
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

# Explicit FP32 staging for one target mapping of the UASDEA dataflow plan.
# The public value is a checked, ownable snapshot: never lower a caller-mutated
# plan, forge a zero-copy transfer, or transfer outside the source allocation.
from ..internal.status import TensorStatus, TensorResult
from ..execution.execution_plan import ExecutionPlan, ExecutionPlacement
from ..layout.layout import TensorLayout

@fieldwise_init
struct StagingPlan(Copyable):
    var source_begin: Int
    var source_end: Int
    var required_bytes: Int
    var double_buffered: Bool
    var transfer_to_gpu: Bool

    @staticmethod
    def checked(
        source_begin: Int, source_end: Int, required_bytes: Int,
        double_buffered: Bool, transfer_to_gpu: Bool,
    ) -> TensorResult[Self]:
        # All ranges are HALF-OPEN element offsets. Count*4 must be exact;
        # a transfer estimate is not a legal allocation/byte-count proof.
        if not transfer_to_gpu:
            if source_begin != 0 or source_end != 0 or required_bytes != 0 or double_buffered:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            return TensorResult[Self](Self(0, 0, 0, False, False))
        if source_begin < 0 or source_end <= source_begin:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if source_end > Int.MAX // 4:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        var items = source_end - source_begin
        if items > Int.MAX // 4 or required_bytes != items * 4:
            return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
        return TensorResult[Self](
            Self(source_begin, source_end, required_bytes, double_buffered, True)
        )

    def checked_snapshot(self) -> TensorResult[Self]:
        # Mojo 1.1 struct fields are declared 'var'. This validated copy,
        # rather than a claimed field-level immutable type, is the boundary.
        return Self.checked(
            self.source_begin, self.source_end, self.required_bytes,
            self.double_buffered, self.transfer_to_gpu,
        )


def plan_staging(
    plan: ExecutionPlan, source_on_gpu: Bool, reuse_count: Int,
    fenced_buffers_available: Bool = False,
) -> TensorResult[StagingPlan]:
    if reuse_count < 0:
        return TensorResult[StagingPlan](TensorStatus.INVALID_SHAPE)
    var proof = plan.checked_snapshot()
    if not proof.is_ok():
        return TensorResult[StagingPlan](proof.status)
    var selected = proof.payload.value().copy()
    if selected.placement == materialize[ExecutionPlacement.CPU_ONLY]() or source_on_gpu:
        return StagingPlan.checked(0, 0, 0, False, False)
    var count = selected.gpu_end - selected.gpu_begin
    if count == 0:
        return StagingPlan.checked(0, 0, 0, False, False)
    if count > Int.MAX // 4:
        return TensorResult[StagingPlan](TensorStatus.OVERFLOW)
    # A host source cannot be staged under a GPU plan that declared zero
    # transfer bytes. This is only a lower bound for ONE FP32 input; multi-
    # operand lowerings must additionally prove their complete transfer DAG.
    if selected.transfer_bytes < count * 4:
        return TensorResult[StagingPlan](TensorStatus.SIZE_MISMATCH)
    # The caller MUST bind real event-fenced staging slots before using this
    # optional double-buffer hint. This Boolean is a policy declaration, not
    # an authenticated ownership or completed-event proof; lowerings that
    # cannot maintain the lifetime contract must pass False.
    return StagingPlan.checked(
        selected.gpu_begin, selected.gpu_end, count * 4,
        reuse_count > 1 and fenced_buffers_available, True,
    )


def validate_staging_capacity(plan: StagingPlan, physical_elements: Int) -> TensorStatus:
    if physical_elements < 0 or physical_elements > Int.MAX // 4:
        return TensorStatus.INVALID_LAYOUT
    var proof = plan.checked_snapshot()
    if not proof.is_ok():
        return proof.status
    var selected = proof.payload.value().copy()
    if not selected.transfer_to_gpu:
        return TensorStatus.OK
    # Checking only required_bytes <= physical_bytes is insufficient for a
    # sliced source: begin=1000, end=1001 cannot be read from a 10-item owner.
    if selected.source_end > physical_elements:
        return TensorStatus.OUT_OF_BOUNDS
    return TensorStatus.OK


def validate_staging_source_layout[rank: Int](
    stage: StagingPlan, source: TensorLayout[rank], physical_elements: Int,
) -> TensorStatus:
    # Checked bulk staging uses physical FP32 offsets, whereas an arbitrary
    # tensor view has logical coordinates. Require an exact base-zero C-order
    # view before reusing stage.source_begin as a physical input offset.
    # Transposed, reversed, broadcasted, and sliced-offset inputs need an
    # explicitly validated pack/slice lowering, NEVER silent raw DMA.
    var stage_check = stage.checked_snapshot()
    if not stage_check.is_ok():
        return stage_check.status
    var checked_source = source.validated_snapshot(physical_elements)
    if not checked_source.is_ok():
        return checked_source.status
    var trusted = checked_source.payload.value()
    var plan = stage_check.payload.value().copy()
    if not plan.transfer_to_gpu:
        return TensorStatus.OK
    if trusted.offset != 0 or not trusted.is_c_contiguous():
        return TensorStatus.NOT_REPRESENTABLE
    if not trusted.is_proven_unique() or not trusted.is_exhaustive():
        return TensorStatus.NOT_REPRESENTABLE
    if trusted.span_lo != 0 or trusted.span_hi != trusted.shape.numel() - 1:
        return TensorStatus.NOT_REPRESENTABLE
    if plan.source_end > trusted.shape.numel():
        return TensorStatus.OUT_OF_BOUNDS
    return validate_staging_capacity(plan, physical_elements)
