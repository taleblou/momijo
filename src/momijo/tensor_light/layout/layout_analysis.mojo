# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR LAYOUT / LAYOUT ANALYSIS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Layout and Strides
# Module:        momijo.tensor_light.layout.layout_analysis
# Source:        src/momijo/tensor_light/layout/layout_analysis.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines layout analysis in the tensor layout and strides layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `LayoutAnalysis`, `CoalescedLayout`,
#   `find_unit_stride_axis`, `is_reverse_c_contiguous`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Compute and validate shape-to-storage layout mappings.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `LayoutAnalysis`,
#     `CoalescedLayout`, `find_unit_stride_axis`, `is_reverse_c_contiguous`,
#     `is_reverse_f_contiguous` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Preserve correct offsets and spans for arbitrary legal strides.
#   - Detect overflow, invalid extents, and internal overlap.
#
# Performance Objectives:
#   - Prefer metadata-only layout analysis.
#   - Cache or reuse validated stride computations where safe.
#
# Maintainability:
#   - Use consistent C/F contiguity terminology.
#   - Test negative, zero, and unusually large strides.
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

# Optimizer-only analysis of validated fixed-rank FP32 layouts.
# This layer never changes tensor indexing, ownership or iteration semantics.
from .layout import TensorLayout
from .contiguity import internal_overlap_kind
from ..shape.shape_storage import StaticRankArray
from ..shape.shape import TensorShape
from .offset import checked_physical_span
from ..internal.checked_math import checked_mul, checked_mul_nonnegative
from ..internal.status import TensorResult, TensorStatus

@fieldwise_init
struct LayoutAnalysis(Copyable):
    var contiguous_inner_axis: Int
    var contiguous_run: Int
    var c_contiguous: Bool
    var f_contiguous: Bool
    var proven_unique: Bool


struct CoalescedLayout[rank: Int](Copyable):
    # Only the first active_rank entries are logical. A rank-0 tensor has
    # active_rank=0, while an empty positive-rank tensor uses a zero extent.
    var dims: StaticRankArray[Self.rank]
    var strides: StaticRankArray[Self.rank]
    var active_rank: Int
    var offset: Int

    def __init__(out self, dims: StaticRankArray[Self.rank], strides: StaticRankArray[Self.rank], active_rank: Int, offset: Int):
        self.dims = dims
        self.strides = strides
        self.active_rank = active_rank
        self.offset = offset

    def __init__(out self, *, copy: Self):
        self.dims = copy.dims.copy()
        self.strides = copy.strides.copy()
        self.active_rank = copy.active_rank
        self.offset = copy.offset


def _metadata_span_consistent[rank: Int](layout: TensorLayout[rank]) -> Bool:
    # A public optimizer hint cannot certify raw SIMD/parallel access merely
    # from matching extents and strides when cached span metadata is forged.
    # This is a metadata check, NOT a physical owner/capacity proof. An
    # actual memory access still needs TensorLayout.validated_snapshot(owner).
    var canonical = TensorShape[rank].checked(layout.shape.dims)
    if not canonical.is_ok():
        return False
    if canonical.payload.value().numel() != layout.shape.numel():
        return False
    var proof = checked_physical_span(
        canonical.payload.value(), layout.strides, layout.offset,
    )
    if not proof.is_ok():
        return False
    var physical = proof.payload.value()
    if physical.lo < 0:
        return False
    if physical.lo != layout.span_lo or physical.hi != layout.span_hi:
        return False
    # Cached uniqueness is a mutable optimization hint, not a proof.
    # Rebuild overlap from canonical extents/strides; otherwise a caller can
    # forge overlap=NONE while retaining matching span metadata and trick
    # AOT scheduling into unsafe concurrent writes or noalias vectorization.
    if internal_overlap_kind(canonical.payload.value(), layout.strides) != layout.overlap:
        return False
    return True


def find_unit_stride_axis[rank: Int](layout: TensorLayout[rank]) -> Int:
    # A positive axis is a fast-path hint and must not be inferred from a
    # forged cached element count. The owning view additionally validates
    # this layout against its physical storage before any dereference.
    if not _metadata_span_consistent(layout):
        return -1
    if layout.shape.is_empty():
        return -1
    var best_axis = -1
    var best_extent = 1
    comptime for axis in range(rank):
        var extent = layout.shape.dims.get(axis)
        if extent > best_extent and layout.strides.get(axis) == 1:
            best_axis = axis
            best_extent = extent
    return best_axis



# Dense backwards mappings are still one physical run. They are NOT normal
# C/F-contiguous layouts: traversing logical indices decreases addresses.
# Returning a separate predicate prevents treating them as forward SIMD runs.
def is_reverse_c_contiguous[rank: Int](layout: TensorLayout[rank]) -> Bool:
    # These public metadata predicates cannot trust caller-constructed cached
    # numel. Overflow in the expected reverse stride must never certify a
    # raw reverse/SIMD copy as a single physical run.
    if not _metadata_span_consistent(layout):
        return False
    if layout.shape.is_empty():
        return True
    var expected = 1
    comptime for reversed_index in range(rank):
        var axis = rank - 1 - reversed_index
        var extent = layout.shape.dims.get(axis)
        if extent > 1:
            if layout.strides.get(axis) != -expected:
                return False
            if expected > (Int.MAX // 4) // extent:
                return False
            expected *= extent
    return True


def is_reverse_f_contiguous[rank: Int](layout: TensorLayout[rank]) -> Bool:
    if not _metadata_span_consistent(layout):
        return False
    if layout.shape.is_empty():
        return True
    var expected = 1
    comptime for axis in range(rank):
        var extent = layout.shape.dims.get(axis)
        if extent > 1:
            if layout.strides.get(axis) != -expected:
                return False
            if expected > (Int.MAX // 4) // extent:
                return False
            expected *= extent
    return True


def analyze_layout_checked[rank: Int](
    layout: TensorLayout[rank], storage_capacity: Int,
) -> TensorResult[LayoutAnalysis]:
    # This checked entry belongs at the boundary between a storage owner
    # and AOT iteration planning. The internal analyze_layout() helper only
    # sees its freshly verified layout snapshot; no stale cached span is
    # allowed to authorize a vector/parallel schedule.
    var proof = layout.validated_snapshot(storage_capacity)
    if not proof.is_ok():
        return TensorResult[LayoutAnalysis](proof.status)
    return TensorResult[LayoutAnalysis](analyze_layout[rank](proof.payload.value()))


def analyze_layout[rank: Int](layout: TensorLayout[rank]) -> LayoutAnalysis:
    if not _metadata_span_consistent(layout):
        # Do not export an optimizer hint based on altered address metadata.
        return LayoutAnalysis(-1, 0, False, False, False)
    # Keep optimizer proof flags consistent with the public checked metadata
    # predicate. Recomputing overlap independently would ignore a stale or
    # forged cached overlap flag and could certify a different answer.
    var unique = layout.is_proven_unique()
    if layout.shape.is_empty():
        return LayoutAnalysis(-1, 0, unique, unique, unique)
    var c = layout.is_c_contiguous()
    var f = layout.is_f_contiguous()
    # IterationPlan represents the canonical C-order logical iteration.
    # An F-contiguous multi-axis array is physically dense but its C-order
    # logical neighbours are NOT successive physical addresses. Marking all
    # its elements as one forward-SIMD run would silently misread/reorder
    # source values. A separate F-order lowering may exploit its physical
    # contiguity only after ALL operands share an explicit F traversal plan.
    var axis = -1
    var last_nontrivial_found = False
    comptime for reversed_axis in range(rank):
        var logical_axis = rank - 1 - reversed_axis
        if not last_nontrivial_found and layout.shape.dims.get(logical_axis) > 1:
            last_nontrivial_found = True
            if layout.strides.get(logical_axis) == 1:
                axis = logical_axis
    var run = 1
    if c:
        run = layout.shape.numel()
    elif axis >= 0:
        # Only the last non-singleton axis can form a forward run in
        # C-order. Another unit-stride axis is a *potential alternative*
        # traversal, not a proven contiguous logical inner run.
        run = layout.shape.dims.get(axis)
    return LayoutAnalysis(axis, run, c, f, unique)


def collapse_singleton_axes[rank: Int](layout: TensorLayout[rank]) -> CoalescedLayout[rank]:
    var dims = StaticRankArray[rank](0)
    var strides = StaticRankArray[rank](0)
    var used = 0
    comptime for i in range(rank):
        var extent = layout.shape.dims.get(i)
        if extent != 1:
            dims.set(used, extent)
            strides.set(used, layout.strides.get(i))
            used += 1
    return CoalescedLayout[rank](dims, strides, used, layout.offset)


def coalesce_contiguous_axes[rank: Int](layout: TensorLayout[rank]) -> CoalescedLayout[rank]:
    # Combine adjacent right-to-left axes only where an exact, overflow-safe
    # stride equality proves that logical progression is physically linear.
    # A zero-stride broadcast is NOT interpreted as unique contiguous data.
    var collapsed = collapse_singleton_axes[rank](layout)
    var dims = StaticRankArray[rank](0)
    var strides = StaticRankArray[rank](0)
    var used = 0
    for reverse_axis in range(collapsed.active_rank):
        var axis = collapsed.active_rank - 1 - reverse_axis
        var extent = collapsed.dims.get(axis)
        var stride = collapsed.strides.get(axis)
        if used > 0:
            var prev_extent = dims.get(used - 1)
            var previous_stride = strides.get(used - 1)
            if previous_stride != 0 and stride != 0:
                # The exact affine identity works for BOTH positive and
                # negative strides. For example, a reversed C-order view
                # with [-N, -1] is one reverse-contiguous run. Use signed
                # checked multiplication rather than dropping this fast path.
                # Do not coalesce zero-stride broadcast axes: they carry
                # overlapping logical elements rather than unique memory.
                var expected = checked_mul(previous_stride, prev_extent)
                if expected.is_ok() and expected.payload.value() == stride:
                    var combined = checked_mul_nonnegative(extent, prev_extent)
                    if combined.is_ok():
                        dims.set(used - 1, combined.payload.value())
                        continue
        dims.set(used, extent)
        strides.set(used, stride)
        used += 1
    # Restore outer-to-inner ordering of the coalesced physical dimensions.
    var ordered_dims = StaticRankArray[rank](0)
    var ordered_strides = StaticRankArray[rank](0)
    for i in range(used):
        ordered_dims.set(i, dims.get(used - 1 - i))
        ordered_strides.set(i, strides.get(used - 1 - i))
    return CoalescedLayout[rank](ordered_dims, ordered_strides, used, layout.offset)


def iteration_order[rank: Int](layout: TensorLayout[rank]) -> StaticRankArray[rank]:
    # Largest absolute stride outermost; the smallest is the fast inner axis.
    # No payload operation may change its documented floating-point reduction
    # order without an explicit accuracy policy.
    var order = StaticRankArray[rank](0)
    comptime for i in range(rank):
        order.set(i, i)
    for i in range(rank):
        for j in range(i + 1, rank):
            var a = order.get(i)
            var b = order.get(j)
            var sa = layout.strides.get(a)
            var sb = layout.strides.get(b)
            # Int.MIN has no representable positive absolute value: it is
            # always the largest possible magnitude and belongs outermost.
            if sa == Int.MIN:
                continue
            var replace_outer = sb == Int.MIN
            if sb != Int.MIN:
                var abs_a = sa
                var abs_b = sb
                if abs_a < 0:
                    abs_a = -abs_a
                if abs_b < 0:
                    abs_b = -abs_b
                if abs_b > abs_a:
                    replace_outer = True
            if replace_outer:
                order.set(i, b)
                order.set(j, a)
    return order


def largest_contiguous_run[rank: Int](
    layout: TensorLayout[rank], position: StaticRankArray[rank]
) -> TensorResult[Int]:
    # This contract uses C-order LOGICAL traversal, not an optimizer-selected
    # reordering. For non-C layouts, only a trailing unit-stride axis can
    # safely become a bulk-copy run without changing logical element order.
    var validated_position = layout.physical_offset(position)
    if not validated_position.is_ok():
        return TensorResult[Int](validated_position.status)
    # physical_offset() accepts negative *logical* indices. Reusing their
    # original negative values to compute a contiguous-run length can grant
    # more elements than remain (for example [-1] in a rank-1 C layout).
    # Normalize coordinates after the checked offset proof, before giving a
    # vectorized kernel an unchecked run length.
    var normalized = StaticRankArray[rank](0)
    comptime for i in range(rank):
        var at = position.get(i)
        if at < 0:
            at += layout.shape.dims.get(i)
        normalized.set(i, at)
    if layout.is_c_contiguous():
        var linear = 0
        comptime for i in range(rank):
            # The checked numel bound ensures every prefix stays in range.
            linear = linear * layout.shape.dims.get(i) + normalized.get(i)
        return TensorResult[Int](layout.shape.numel() - linear)
    comptime if rank > 0:
        if layout.strides.get(rank - 1) == 1:
            return TensorResult[Int](layout.shape.dims.get(rank - 1) - normalized.get(rank - 1))
    return TensorResult[Int](1)


# Canonical affine reshape feasibility and resulting validated strides.
# All zero-copy view APIs delegate here, so there is ONE implementation of
# the F/C/order chunk-matching proof and no layout -> view import cycle.
def _reshape_view_layout_checked[
    src_rank: Int, dst_rank: Int, c_order: Bool
](
    src: TensorLayout[src_rank], desired: TensorShape[dst_rank], capacity: Int
) -> TensorResult[TensorLayout[dst_rank]]:
    comptime assert src_rank >= 0 and src_rank <= 8, "source rank must be 0..8"
    comptime assert dst_rank >= 0 and dst_rank <= 8, "target rank must be 0..8"
    var validated = src.validated_snapshot(capacity)
    if not validated.is_ok():
        return TensorResult[TensorLayout[dst_rank]](validated.status)
    var target = TensorShape[dst_rank].checked(desired.dims)
    if not target.is_ok():
        return TensorResult[TensorLayout[dst_rank]](target.status)
    if target.payload.value().numel() != desired.numel():
        return TensorResult[TensorLayout[dst_rank]](TensorStatus.INVALID_SHAPE)
    var original = validated.payload.value()
    if original.shape.numel() != desired.numel():
        return TensorResult[TensorLayout[dst_rank]](TensorStatus.SIZE_MISMATCH)
    comptime if src_rank == dst_rank:
        if original.shape.same_extents(desired):
            return TensorLayout[dst_rank].checked(
                target.payload.value(), original.strides, original.offset, capacity
            )

    # With zero/one logical element, the single reachable address (if any)
    # is preserved regardless of the original stride pattern.
    if original.shape.numel() <= 1:
        comptime if c_order:
            var canonical = TensorLayout[dst_rank].c_contiguous(target.payload.value(), capacity)
            if not canonical.is_ok():
                return TensorResult[TensorLayout[dst_rank]](canonical.status)
            return TensorLayout[dst_rank].checked(
                target.payload.value(), canonical.payload.value().strides, original.offset, capacity
            )
        else:
            var canonical = TensorLayout[dst_rank].f_contiguous(target.payload.value(), capacity)
            if not canonical.is_ok():
                return TensorResult[TensorLayout[dst_rank]](canonical.status)
            return TensorLayout[dst_rank].checked(
                target.payload.value(), canonical.payload.value().strides, original.offset, capacity
            )

    # Transform F-order into the same fast-to-slow block matching used for
    # C-order simply by reversing its axes. Singleton dimensions contribute
    # no address changes and are excluded from the matching calculation.
    var old_dims = Array[Int, 8](fill=1)
    var old_strides = Array[Int, 8](fill=0)
    var new_dims = Array[Int, 8](fill=1)
    var matched_strides = Array[Int, 8](fill=0)
    var old_count = 0
    var new_count = 0
    comptime for i in range(src_rank):
        var axis = i
        comptime if not c_order:
            axis = src_rank - 1 - i
        var size = original.shape.dims.get(axis)
        if size > 1:
            old_dims[old_count] = size
            old_strides[old_count] = original.strides.get(axis)
            old_count += 1
    comptime for i in range(dst_rank):
        var axis = i
        comptime if not c_order:
            axis = dst_rank - 1 - i
        var size = target.payload.value().dims.get(axis)
        if size > 1:
            new_dims[new_count] = size
            new_count += 1

    var old_begin = 0
    var new_begin = 0
    while old_begin < old_count and new_begin < new_count:
        var old_end = old_begin
        var new_end = new_begin
        var old_elements = old_dims[old_begin]
        var new_elements = new_dims[new_begin]
        # All partial products are bounded by the already-validated numel.
        while old_elements != new_elements:
            if old_elements < new_elements:
                old_end += 1
                if old_end >= old_count:
                    return TensorResult[TensorLayout[dst_rank]](TensorStatus.NOT_REPRESENTABLE)
                old_elements *= old_dims[old_end]
            else:
                new_end += 1
                if new_end >= new_count:
                    return TensorResult[TensorLayout[dst_rank]](TensorStatus.NOT_REPRESENTABLE)
                new_elements *= new_dims[new_end]
        # A matching element-count block is viewable only if its old axes
        # are adjacent in their requested logical flattening order. This
        # also handles negative strides and broadcast strides correctly.
        for i in range(old_begin, old_end):
            var expected = checked_mul(old_strides[i + 1], old_dims[i + 1])
            if not expected.is_ok():
                return TensorResult[TensorLayout[dst_rank]](TensorStatus.NOT_REPRESENTABLE)
            if old_strides[i] != expected.payload.value():
                return TensorResult[TensorLayout[dst_rank]](TensorStatus.NOT_REPRESENTABLE)
        var running_stride = old_strides[old_end]
        var out_axis = new_end
        while out_axis >= new_begin:
            matched_strides[out_axis] = running_stride
            if out_axis > new_begin:
                var increment = checked_mul(running_stride, new_dims[out_axis])
                if not increment.is_ok():
                    return TensorResult[TensorLayout[dst_rank]](TensorStatus.NOT_REPRESENTABLE)
                running_stride = increment.payload.value()
            out_axis -= 1
        old_begin = old_end + 1
        new_begin = new_end + 1

    if old_begin != old_count or new_begin != new_count:
        return TensorResult[TensorLayout[dst_rank]](TensorStatus.NOT_REPRESENTABLE)

    var result_strides = StaticRankArray[dst_rank](0)
    var compact_axis = 0
    comptime for i in range(dst_rank):
        var axis = i
        comptime if not c_order:
            axis = dst_rank - 1 - i
        if target.payload.value().dims.get(axis) > 1:
            result_strides.set(axis, matched_strides[compact_axis])
            compact_axis += 1
        # Any stride represents a singleton axis. Zero is chosen to avoid
        # overflow and keep metadata arithmetic minimal in hot loops.
    return TensorLayout[dst_rank].checked(
        target.payload.value(), result_strides, original.offset, capacity
    )



# Explicit optimizer-facing functions. The allocation capacity is required
# because a metadata-only match without a bound physical span is not a valid
# zero-copy view of an owned FP32 buffer.
def reshape_view_strides[src_rank: Int, dst_rank: Int, c_order: Bool](
    source: TensorLayout[src_rank], target: TensorShape[dst_rank], capacity: Int,
) -> TensorResult[StaticRankArray[dst_rank]]:
    var result = _reshape_view_layout_checked[src_rank, dst_rank, c_order](
        source, target, capacity,
    )
    if not result.is_ok():
        return TensorResult[StaticRankArray[dst_rank]](result.status)
    return TensorResult[StaticRankArray[dst_rank]](result.payload.value().strides.copy())


def can_view_reshape[src_rank: Int, dst_rank: Int, c_order: Bool](
    source: TensorLayout[src_rank], target: TensorShape[dst_rank], capacity: Int,
) -> Bool:
    return _reshape_view_layout_checked[src_rank, dst_rank, c_order](
        source, target, capacity,
    ).is_ok()
