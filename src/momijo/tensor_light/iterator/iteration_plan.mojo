# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR ITERATION / ITERATION PLANS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Iteration
# Module:        momijo.tensor_light.iterator.iteration_plan
# Source:        src/momijo/tensor_light/iterator/iteration_plan.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines iteration plans in the tensor iteration layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `IterationPlan`, `make_iteration_plan`,
#   `make_iteration_plan_checked`, `make_iteration_plan_for_layout_checked`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Traverse logical elements with validated strides and offsets.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `IterationPlan`,
#     `make_iteration_plan`, `make_iteration_plan_checked`,
#     `make_iteration_plan_for_layout_checked`, `validate_iteration_plan` and
#     related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Verify iteration bounds and storage offsets before traversal.
#   - Respect noncontiguous and broadcasted strides.
#
# Performance Objectives:
#   - Keep index advancement allocation-free where feasible.
#   - Specialize contiguous traversal without changing general semantics.
#
# Maintainability:
#   - Keep iterator behavior consistent with layout helpers.
#   - Cover rank-zero, empty, and high-rank traversal.
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

# Immutable traversal selection prepared before executing a kernel.
from ..layout.layout import TensorLayout
from ..layout.layout_analysis import LayoutAnalysis, analyze_layout
from ..internal.status import TensorResult, TensorStatus

@fieldwise_init
struct IterationPlan(Copyable):
    var total_items: Int
    var vector_axis: Int
    var contiguous_run: Int
    var vector_width: Int
    var parallel_grain: Int

def make_iteration_plan(analysis: LayoutAnalysis, total_items: Int, simd_width: Int, grain: Int) -> IterationPlan:
    var width = 1
    if analysis.contiguous_run >= simd_width and simd_width > 1:
        width = simd_width
    return IterationPlan(total_items, analysis.contiguous_inner_axis, analysis.contiguous_run, width, grain)


# Checked boundary for foreign/offline profile metadata. Hot loops consume
# the returned plain IterationPlan after structural verification, not an
# extra wrapper or per-element status check. LayoutAnalysis still must have
# been derived from a checked owner/layout; these numbers alone do not prove
# a physical pointer is aligned, in bounds or unique.
def make_iteration_plan_checked(
    analysis: LayoutAnalysis, total_items: Int, simd_width: Int, grain: Int,
) -> TensorResult[IterationPlan]:
    if total_items < 0 or total_items > Int.MAX // 4:
        return TensorResult[IterationPlan](TensorStatus.INVALID_SHAPE)
    if simd_width <= 0 or grain <= 0 or analysis.contiguous_run < 0:
        return TensorResult[IterationPlan](TensorStatus.INVALID_LAYOUT)
    # Only represent SIMD lane counts that can map to an AOT vector.
    if simd_width & (simd_width - 1) != 0:
        return TensorResult[IterationPlan](TensorStatus.INVALID_LAYOUT)
    if analysis.contiguous_inner_axis < -1 or analysis.contiguous_inner_axis > 7:
        return TensorResult[IterationPlan](TensorStatus.INVALID_AXIS)
    if total_items == 0:
        if analysis.contiguous_run != 0:
            return TensorResult[IterationPlan](TensorStatus.INVALID_LAYOUT)
    elif analysis.contiguous_run <= 0 or analysis.contiguous_run > total_items:
        return TensorResult[IterationPlan](TensorStatus.INVALID_LAYOUT)
    if analysis.contiguous_inner_axis == -1 and analysis.contiguous_run > 1:
        # The canonical C/F flat path may provide a full contiguous run even
        # when its rank has no unit-stride axis; a scalar has one element.
        if total_items != analysis.contiguous_run:
            return TensorResult[IterationPlan](TensorStatus.INVALID_LAYOUT)
    var built = make_iteration_plan(analysis, total_items, simd_width, grain)
    return TensorResult[IterationPlan](built)

def make_iteration_plan_for_layout_checked[rank: Int](
    layout: TensorLayout[rank], storage_capacity: Int,
    simd_width: Int, grain: Int,
) -> TensorResult[IterationPlan]:
    # Checked owner-facing entry. Reconstruct a storage-bound layout snapshot
    # before consuming any shape, span or contiguous-run claim. No heap
    # metadata allocation and no per-element validation in the resulting plan.
    var canonical = layout.validated_snapshot(storage_capacity)
    if not canonical.is_ok():
        return TensorResult[IterationPlan](canonical.status)
    var safe = canonical.payload.value()
    # analyze_layout() only inspects canonical metadata; the physical storage
    # proof was completed exactly once by validated_snapshot().
    var analysis = analyze_layout[rank](safe)
    return make_iteration_plan_checked(
        analysis, safe.shape.numel(), simd_width, grain,
    )


# Public planner queries are checked because IterationPlan currently carries
# mutable value fields. These checks run once at the launch boundary and
# never inside a Float32 SIMD/element loop.
def validate_iteration_plan(plan: IterationPlan) -> TensorStatus:
    if plan.total_items < 0 or plan.total_items > Int.MAX // 4:
        return TensorStatus.INVALID_SHAPE
    if plan.parallel_grain <= 0:
        return TensorStatus.INVALID_LAYOUT
    if plan.vector_axis < -1 or plan.vector_axis > 7:
        return TensorStatus.INVALID_AXIS
    if plan.vector_width <= 0 or (plan.vector_width & (plan.vector_width - 1)) != 0:
        return TensorStatus.INVALID_LAYOUT
    if plan.total_items == 0:
        if plan.contiguous_run != 0 or plan.vector_width != 1:
            return TensorStatus.INVALID_LAYOUT
    else:
        if plan.contiguous_run <= 0 or plan.contiguous_run > plan.total_items:
            return TensorStatus.INVALID_LAYOUT
        if plan.vector_width > plan.contiguous_run:
            return TensorStatus.INVALID_LAYOUT
        if plan.vector_axis == -1 and plan.contiguous_run > 1:
            if plan.contiguous_run != plan.total_items:
                return TensorStatus.INVALID_LAYOUT
    return TensorStatus.OK


def inner_size(plan: IterationPlan) -> TensorResult[Int]:
    var status = validate_iteration_plan(plan)
    if status != TensorStatus.OK:
        return TensorResult[Int](status)
    return TensorResult[Int](plan.contiguous_run)


def vector_width(plan: IterationPlan) -> TensorResult[Int]:
    var status = validate_iteration_plan(plan)
    if status != TensorStatus.OK:
        return TensorResult[Int](status)
    return TensorResult[Int](plan.vector_width)


def is_contiguous_fast_path(plan: IterationPlan) -> TensorResult[Bool]:
    var status = validate_iteration_plan(plan)
    if status != TensorStatus.OK:
        return TensorResult[Bool](status)
    # Forward canonical C-order only. A layout's F-contiguous flag alone
    # never certifies this predicate; analyze_layout() proves the run.
    return TensorResult[Bool](
        plan.total_items > 0 and plan.contiguous_run == plan.total_items,
    )


def build_iteration_plan_for_layout[rank: Int](
    layout: TensorLayout[rank], storage_capacity: Int,
    simd_width: Int, grain: Int,
) -> TensorResult[IterationPlan]:
    # One public owner-capacity checked entry with a stable naming surface.
    return make_iteration_plan_for_layout_checked[rank](
        layout, storage_capacity, simd_width, grain,
    )


def validate_iteration_plan_against_layout[rank: Int](
    plan: IterationPlan, layout: TensorLayout[rank], storage_capacity: Int,
    simd_width: Int, grain: Int,
) -> TensorStatus:
    # A caller may fabricate an internally consistent IterationPlan that is
    # WRONG for its actual tensor (e.g. claim one forward vector run for a
    # multi-axis F-contiguous layout). Recover the expected plan from the
    # owner-capacity checked layout before selecting an unchecked kernel.
    var current = validate_iteration_plan(plan)
    if current != TensorStatus.OK:
        return current
    var canonical = make_iteration_plan_for_layout_checked[rank](
        layout, storage_capacity, simd_width, grain,
    )
    if not canonical.is_ok():
        return canonical.status
    var expected = canonical.payload.value()
    if (plan.total_items != expected.total_items
        or plan.vector_axis != expected.vector_axis
        or plan.contiguous_run != expected.contiguous_run
        or plan.vector_width != expected.vector_width
        or plan.parallel_grain != expected.parallel_grain):
        return TensorStatus.INVALID_LAYOUT
    return TensorStatus.OK


def is_contiguous_fast_path_checked[rank: Int](
    plan: IterationPlan, layout: TensorLayout[rank], storage_capacity: Int,
    simd_width: Int, grain: Int,
) -> TensorResult[Bool]:
    var status = validate_iteration_plan_against_layout[rank](
        plan, layout, storage_capacity, simd_width, grain,
    )
    if status != TensorStatus.OK:
        return TensorResult[Bool](status)
    # Proof is about canonical logical traversal, not native alignment or
    # physical alias identity. The kernel adapter must check those too.
    return TensorResult[Bool](
        plan.total_items > 0 and plan.contiguous_run == plan.total_items,
    )
