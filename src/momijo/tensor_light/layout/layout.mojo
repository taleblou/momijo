# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR LAYOUT / LAYOUT
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Layout and Strides
# Module:        momijo.tensor_light.layout.layout
# Source:        src/momijo/tensor_light/layout/layout.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor layout in the tensor layout and strides layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `TensorLayout`, `make_c_contiguous_layout`,
#   `make_f_contiguous_layout`, `make_strided_layout`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Compute and validate shape-to-storage layout mappings.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `TensorLayout`,
#     `make_c_contiguous_layout`, `make_f_contiguous_layout`,
#     `make_strided_layout`, `make_scalar_layout` and related symbols.
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

# Fixed-rank strided mapping, checked before any address is dereferenced.
from ..shape.shape import TensorShape
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus
from ..internal.checked_math import checked_mul, checked_add, checked_sub, checked_mul_nonnegative
from .offset import checked_physical_span
from .strides import make_broadcast_strides
from .contiguity import InternalOverlapKind, internal_overlap_kind, is_c_contiguous, is_f_contiguous

struct TensorLayout[rank: Int](ImplicitlyCopyable):
    var shape: TensorShape[Self.rank]
    var strides: StaticRankArray[Self.rank]
    var offset: Int
    var span_lo: Int
    var span_hi: Int
    var overlap: InternalOverlapKind

    def __init__(out self, shape: TensorShape[Self.rank], strides: StaticRankArray[Self.rank], offset: Int, lo: Int, hi: Int, overlap: InternalOverlapKind):
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.span_lo = lo
        self.span_hi = hi
        self.overlap = overlap

    def __init__(out self, *, copy: Self):
        self.shape = copy.shape.copy()
        self.strides = copy.strides.copy()
        self.offset = copy.offset
        self.span_lo = copy.span_lo
        self.span_hi = copy.span_hi
        self.overlap = copy.overlap

    @staticmethod
    def checked(shape: TensorShape[Self.rank], strides: StaticRankArray[Self.rank], offset: Int, capacity: Int) -> TensorResult[Self]:
        if capacity < 0:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        # Every physical element is four bytes. A forged or external capacity
        # larger than the representable FP32 byte span cannot serve as a
        # valid safety proof even when a small view happens to fit in it.
        if capacity > Int.MAX // 4:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        # A raw TensorShape constructor can carry an inconsistent cached numel.
        # Do not allow an unchecked metadata object to bypass the shape boundary.
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        if canonical.payload.value().numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var trusted_shape = canonical.payload.value()
        var span = checked_physical_span(trusted_shape, strides, offset)
        if not span.is_ok():
            return TensorResult[Self](span.status)
        var physical = span.payload.value()
        if physical.empty:
            # An empty mapping still has a logical origin. It may be one past
            # the end, but never before allocation or beyond that position.
            if physical.lo < 0 or physical.lo > capacity:
                return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
        else:
            if physical.lo < 0 or physical.hi >= capacity:
                return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
        var alias = internal_overlap_kind(trusted_shape, strides)
        return TensorResult[Self](Self(trusted_shape, strides, offset, physical.lo, physical.hi, alias))

    def validated_snapshot(self, capacity: Int) -> TensorResult[Self]:
        # Rebuild the complete structural proof from extents, strides and
        # allocation capacity. Cached span and overlap flags are not authority.
        var rebuilt = Self.checked(self.shape, self.strides, self.offset, capacity)
        if not rebuilt.is_ok():
            return rebuilt^
        var canonical = rebuilt.payload.value()
        if canonical.span_lo != self.span_lo or canonical.span_hi != self.span_hi:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if canonical.overlap != self.overlap:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        return rebuilt^

    @staticmethod
    def c_contiguous(shape: TensorShape[Self.rank], capacity: Int) -> TensorResult[Self]:
        # Canonicalize before any stride arithmetic. An unchecked TensorShape
        # may have inconsistent cached element count or negative dimensions.
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        if canonical.payload.value().numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var safe_shape = canonical.payload.value()
        var strides = StaticRankArray[Self.rank](0)
        var step = 1
        comptime for j in range(Self.rank):
            var i = Self.rank - 1 - j
            strides.set(i, step)
            # Preserve canonical row-major strides around zero dimensions.
            # Once a zero extent is encountered all outer strides are zero.
            # For very large *empty* shapes, outer products may not fit Int;
            # the zeroed fallback is safe because no element is addressable.
            if step == 0 or safe_shape.dims.get(i) == 0:
                step = 0
            else:
                var product = checked_mul_nonnegative(step, safe_shape.dims.get(i))
                if not product.is_ok():
                    if not safe_shape.is_empty():
                        return TensorResult[Self](product.status)
                    step = 0
                else:
                    step = product.payload.value()
        return Self.checked(safe_shape, strides, 0, capacity)

    @staticmethod
    def f_contiguous(shape: TensorShape[Self.rank], capacity: Int) -> TensorResult[Self]:
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        if canonical.payload.value().numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var safe_shape = canonical.payload.value()
        var strides = StaticRankArray[Self.rank](0)
        var step = 1
        comptime for i in range(Self.rank):
            strides.set(i, step)
            # Column-major strides remain canonical through empty axes.
            if step == 0 or safe_shape.dims.get(i) == 0:
                step = 0
            else:
                var product = checked_mul_nonnegative(step, safe_shape.dims.get(i))
                if not product.is_ok():
                    if not safe_shape.is_empty():
                        return TensorResult[Self](product.status)
                    step = 0
                else:
                    step = product.payload.value()
        return Self.checked(safe_shape, strides, 0, capacity)

    def physical_offset_unchecked(self, indices: StaticRankArray[Self.rank]) -> Int:
        # PRECONDITION: the layout was checked against this allocation, each
        # coordinate is in range, and the caller holds a live host borrow.
        # The checked physical span also proves bounded signed offset sums.
        var position = self.offset
        comptime for i in range(Self.rank):
            position += indices.get(i) * self.strides.get(i)
        return position

    def _offset_from_validated(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Int]:
        # INTERNAL checked-index fast path. PRECONDITION: this value is the
        # newly returned TensorLayout.checked()/validated_snapshot() result.
        # Its shape/strides and physical span are already proved to fit the
        # owning allocation. Normalize only coordinates here; repeating the
        # complete descriptor/span proof for each scalar access is redundant.
        var normalized = StaticRankArray[Self.rank](0)
        comptime for i in range(Self.rank):
            var axis_extent = self.shape.dims.get(i)
            var at = indices.get(i)
            if at < 0:
                if at < -axis_extent:
                    return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
                at += axis_extent
            if at >= axis_extent:
                return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
            normalized.set(i, at)
        # The checked span proof bounds every in-range coordinate combination.
        # In particular rank 0 has zero iterations and returns self.offset.
        return TensorResult[Int](self.physical_offset_unchecked(normalized))

    def physical_offset(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Int]:
        # Checked scalar lookup: this public layout value can be assembled
        # with stale shape, offset and cached span metadata. Only a verified
        # descriptor may enter the separate `physical_offset_unchecked` hot
        # path. The storage owner independently proves allocation capacity.
        var canonical = TensorShape[Self.rank].checked(self.shape.dims)
        if not canonical.is_ok():
            return TensorResult[Int](canonical.status)
        if canonical.payload.value().numel() != self.shape.numel():
            return TensorResult[Int](TensorStatus.INVALID_SHAPE)
        var physical = checked_physical_span(self.shape, self.strides, self.offset)
        if not physical.is_ok():
            return TensorResult[Int](physical.status)
        var bounds = physical.payload.value()
        if bounds.lo != self.span_lo or bounds.hi != self.span_hi:
            return TensorResult[Int](TensorStatus.INVALID_LAYOUT)
        if bounds.empty or bounds.lo < 0:
            return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
        var result = self.offset
        comptime for i in range(Self.rank):
            var at = indices.get(i)
            var dim = self.shape.dims.get(i)
            if at < 0:
                if at < -dim:
                    return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
                at += dim
            if at >= dim:
                return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
            var mul = checked_mul(at, self.strides.get(i))
            if not mul.is_ok():
                return TensorResult[Int](mul.status)
            var added = checked_add(result, mul.payload.value())
            if not added.is_ok():
                return TensorResult[Int](added.status)
            result = added.payload.value()
        if result < bounds.lo or result > bounds.hi:
            return TensorResult[Int](TensorStatus.OUT_OF_BOUNDS)
        return TensorResult[Int](result)

    def is_c_contiguous(self) -> Bool:
        # Public optimizer/packed-copy hints must not trust a mutable cached
        # span or caller-supplied no-alias flag. The owner is still checked
        # separately by Tensor/View before any raw memory access.
        if not self.is_proven_unique():
            return False
        return is_c_contiguous(self.shape, self.strides)

    def is_f_contiguous(self) -> Bool:
        if not self.is_proven_unique():
            return False
        return is_f_contiguous(self.shape, self.strides)

    def is_proven_unique(self) -> Bool:
        # A PUBLIC no-overlap result may authorize in-place/parallel writes.
        # Do not trust a caller-mutated NONE flag, stale shape/numel, or the
        # cached address envelope. This is only a metadata proof: the owning
        # Tensor/View must STILL bind the mapping to its real allocation.
        if self.overlap != InternalOverlapKind.NONE:
            return False
        var canonical = TensorShape[Self.rank].checked(self.shape.dims)
        if not canonical.is_ok():
            return False
        if canonical.payload.value().numel() != self.shape.numel():
            return False
        var checked_span = checked_physical_span(
            canonical.payload.value(), self.strides, self.offset,
        )
        if not checked_span.is_ok():
            return False
        var physical = checked_span.payload.value()
        if physical.lo < 0:
            return False
        if physical.lo != self.span_lo or physical.hi != self.span_hi:
            return False
        return internal_overlap_kind(
            canonical.payload.value(), self.strides,
        ) == InternalOverlapKind.NONE

    def is_exhaustive(self) -> Bool:
        # Trust neither cached numel nor cached physical span when evaluating
        # externally reachable mutable metadata. Verified Tensor/View entry
        # points still construct one snapshot outside their hot loops.
        var canonical = TensorShape[Self.rank].checked(self.shape.dims)
        if not canonical.is_ok() or canonical.payload.value().numel() != self.shape.numel():
            return False
        # Even an empty layout must have a legal (nonnegative) origin and
        # consistent cached span. A fabricated negative-origin mapping is
        # not exhaustive merely because no element will be dereferenced.
        if not self.is_proven_unique():
            return False
        if canonical.payload.value().is_empty():
            return self.span_lo == self.offset and self.span_hi == self.offset
        var physical = checked_physical_span(self.shape, self.strides, self.offset)
        if not physical.is_ok() or physical.payload.value().empty:
            return False
        var proven = physical.payload.value()
        if proven.lo < 0 or proven.hi < proven.lo:
            return False
        if self.span_lo != proven.lo or self.span_hi != proven.hi:
            return False
        var width = checked_sub(proven.hi, proven.lo)
        if not width.is_ok():
            return False
        var span_count = checked_add(width.payload.value(), 1)
        if not span_count.is_ok():
            return False
        return span_count.payload.value() == canonical.payload.value().numel()

    def span_size(self) -> Int:
        # Public TensorLayout metadata is mutable. Never turn a corrupt
        # cached span (especially Int.MIN..MAX) into a wrapped positive
        # allocation-size hint. -1 means invalid/unrepresentable.
        var canonical = TensorShape[Self.rank].checked(self.shape.dims)
        if not canonical.is_ok() or canonical.payload.value().numel() != self.shape.numel():
            return -1
        var range_proof = checked_physical_span(self.shape, self.strides, self.offset)
        if not range_proof.is_ok():
            return -1
        var physical = range_proof.payload.value()
        if physical.lo != self.span_lo or physical.hi != self.span_hi:
            return -1
        if physical.lo < 0:
            return -1
        if physical.empty:
            return 0
        if physical.hi < physical.lo:
            return -1
        var difference = checked_sub(physical.hi, physical.lo)
        if not difference.is_ok():
            return -1
        var length = checked_add(difference.payload.value(), 1)
        if not length.is_ok():
            return -1
        return length.payload.value()

    def transpose(self, a: Int, b: Int, capacity: Int) -> TensorResult[Self]:
        var source_proof = self.validated_snapshot(capacity)
        if not source_proof.is_ok():
            return TensorResult[Self](source_proof.status)
        var trusted = source_proof.payload.value()
        var ia = a
        var ib = b
        if ia < 0:
            if ia < -Self.rank:
                return TensorResult[Self](TensorStatus.INVALID_AXIS)
            ia += Self.rank
        if ib < 0:
            if ib < -Self.rank:
                return TensorResult[Self](TensorStatus.INVALID_AXIS)
            ib += Self.rank
        if ia >= Self.rank or ib >= Self.rank:
            return TensorResult[Self](TensorStatus.INVALID_AXIS)
        var dims = trusted.shape.dims.copy()
        var new_strides = trusted.strides.copy()
        var tmp_dim = dims.get(ia)
        dims.set(ia, dims.get(ib))
        dims.set(ib, tmp_dim)
        var tmp_stride = new_strides.get(ia)
        new_strides.set(ia, new_strides.get(ib))
        new_strides.set(ib, tmp_stride)
        var new_shape = TensorShape[Self.rank](dims, trusted.shape.numel())
        return Self.checked(new_shape, new_strides, trusted.offset, capacity)

    def permute(self, axes: StaticRankArray[Self.rank], capacity: Int) -> TensorResult[Self]:
        var source_proof = self.validated_snapshot(capacity)
        if not source_proof.is_ok():
            return TensorResult[Self](source_proof.status)
        var trusted = source_proof.payload.value()
        var seen = Array[Bool, 8](fill=False)
        var dims = StaticRankArray[Self.rank](0)
        var strides = StaticRankArray[Self.rank](0)
        comptime for i in range(Self.rank):
            var axis = axes.get(i)
            if axis < 0 or axis >= Self.rank:
                return TensorResult[Self](TensorStatus.INVALID_AXIS)
            if seen[axis]:
                return TensorResult[Self](TensorStatus.INVALID_AXIS)
            seen[axis] = True
            dims.set(i, trusted.shape.dims.get(axis))
            strides.set(i, trusted.strides.get(axis))
        return Self.checked(TensorShape[Self.rank](dims, trusted.shape.numel()), strides, trusted.offset, capacity)

    def flip(self, axis: Int, capacity: Int) -> TensorResult[Self]:
        var source_proof = self.validated_snapshot(capacity)
        if not source_proof.is_ok():
            return TensorResult[Self](source_proof.status)
        var trusted = source_proof.payload.value()
        var normalized = axis
        if normalized < 0:
            if normalized < -Self.rank:
                return TensorResult[Self](TensorStatus.INVALID_AXIS)
            normalized += Self.rank
        if normalized >= Self.rank:
            return TensorResult[Self](TensorStatus.INVALID_AXIS)
        var strides = trusted.strides.copy()
        var old = strides.get(normalized)
        if old == Int.MIN:
            if not trusted.shape.is_empty() and trusted.shape.dims.get(normalized) > 1:
                return TensorResult[Self](TensorStatus.OVERFLOW)
            # Empty and singleton axes have no observable direction; flipping
            # them cannot require the negation of an unrepresentable Int.MIN.
            strides.set(normalized, 0)
        else:
            strides.set(normalized, -old)
        var origin = trusted.offset
        # An empty tensor has no addressable elements, even when the flipped
        # axis itself has nonzero extent. Preserve its allocation-valid origin
        # rather than computing an unnecessary potentially out-of-range shift.
        if not trusted.shape.is_empty() and trusted.shape.dims.get(normalized) > 0:
            var delta = checked_mul(trusted.shape.dims.get(normalized) - 1, old)
            if not delta.is_ok():
                return TensorResult[Self](delta.status)
            var next = checked_add(origin, delta.payload.value())
            if not next.is_ok():
                return TensorResult[Self](next.status)
            origin = next.payload.value()
        return Self.checked(trusted.shape, strides, origin, capacity)

# Documented functional vocabulary delegates to the canonical checked
# TensorLayout methods. No secondary layout representation is introduced.
def make_c_contiguous_layout[rank: Int](shape: TensorShape[rank], capacity: Int) -> TensorResult[TensorLayout[rank]]:
    return TensorLayout[rank].c_contiguous(shape, capacity)


def make_f_contiguous_layout[rank: Int](shape: TensorShape[rank], capacity: Int) -> TensorResult[TensorLayout[rank]]:
    return TensorLayout[rank].f_contiguous(shape, capacity)


def make_strided_layout[rank: Int](
    shape: TensorShape[rank], strides: StaticRankArray[rank],
    offset: Int, capacity: Int,
) -> TensorResult[TensorLayout[rank]]:
    return TensorLayout[rank].checked(shape, strides, offset, capacity)


def make_scalar_layout(capacity: Int) -> TensorResult[TensorLayout[0]]:
    var scalar = TensorShape[0].checked(StaticRankArray[0](0))
    if not scalar.is_ok():
        return TensorResult[TensorLayout[0]](scalar.status)
    return TensorLayout[0].checked(scalar.payload.value(), StaticRankArray[0](0), 0, capacity)


def shape[rank: Int](mapping: TensorLayout[rank]) -> TensorShape[rank]:
    return mapping.shape.copy()


def stride[rank: Int](mapping: TensorLayout[rank], axis: Int) -> TensorResult[Int]:
    # Structural-only metadata boundary: reject mutable/fabricated cached
    # shape/span/overlap before reporting a stride. This query does NOT prove
    # allocation ownership; checked Tensor/View entry points still bind the
    # entire mapping to the real capacity before dereference or SIMD dispatch.
    var canonical = TensorShape[rank].checked(mapping.shape.dims)
    if not canonical.is_ok():
        return TensorResult[Int](canonical.status)
    if canonical.payload.value().numel() != mapping.shape.numel():
        return TensorResult[Int](TensorStatus.INVALID_SHAPE)
    var span = checked_physical_span(
        canonical.payload.value(), mapping.strides, mapping.offset,
    )
    if not span.is_ok():
        return TensorResult[Int](span.status)
    # A metadata-only stride query cannot establish allocation ownership,
    # but it must at least reject impossible negative physical origins.
    if span.payload.value().lo < 0:
        return TensorResult[Int](TensorStatus.INVALID_LAYOUT)
    if (span.payload.value().lo != mapping.span_lo
        or span.payload.value().hi != mapping.span_hi):
        return TensorResult[Int](TensorStatus.INVALID_LAYOUT)
    if internal_overlap_kind(canonical.payload.value(), mapping.strides) != mapping.overlap:
        return TensorResult[Int](TensorStatus.INVALID_LAYOUT)
    var normalized = axis
    if normalized < 0:
        if normalized < -rank:
            return TensorResult[Int](TensorStatus.INVALID_AXIS)
        normalized += rank
    return mapping.strides.get_checked(normalized)


def offset[rank: Int](mapping: TensorLayout[rank]) -> Int:
    return mapping.offset


def required_span_size[rank: Int](mapping: TensorLayout[rank]) -> Int:
    return mapping.span_size()


def transpose_layout[rank: Int](mapping: TensorLayout[rank], first: Int, second: Int, capacity: Int) -> TensorResult[TensorLayout[rank]]:
    return mapping.transpose(first, second, capacity)


def permute_layout[rank: Int](mapping: TensorLayout[rank], axes: StaticRankArray[rank], capacity: Int) -> TensorResult[TensorLayout[rank]]:
    return mapping.permute(axes, capacity)


def flip_layout[rank: Int](mapping: TensorLayout[rank], axis: Int, capacity: Int) -> TensorResult[TensorLayout[rank]]:
    return mapping.flip(axis, capacity)


# Canonical rank-changing layout operations belong to the layout owner, not
# an independently duplicated transform in view/. The view layer only creates
# a lifetime-bound borrow of the original FP32 storage from these results.
def unsqueeze_layout[rank: Int, axis: Int](
    mapping: TensorLayout[rank], capacity: Int,
) -> TensorResult[TensorLayout[rank + 1]] where rank >= 0 and rank < 8 and axis >= 0 and axis <= rank:
    var verified = mapping.validated_snapshot(capacity)
    if not verified.is_ok():
        return TensorResult[TensorLayout[rank + 1]](verified.status)
    var trusted = verified.payload.value()
    var dims = StaticRankArray[rank + 1](1)
    var strides = StaticRankArray[rank + 1](0)
    comptime for i in range(rank + 1):
        comptime if i < axis:
            dims.set(i, trusted.shape.dims.get(i))
            strides.set(i, trusted.strides.get(i))
        elif i > axis:
            dims.set(i, trusted.shape.dims.get(i - 1))
            strides.set(i, trusted.strides.get(i - 1))
    var output_shape = TensorShape[rank + 1].checked(dims)
    if not output_shape.is_ok():
        return TensorResult[TensorLayout[rank + 1]](output_shape.status)
    return TensorLayout[rank + 1].checked(
        output_shape.payload.value(), strides, trusted.offset, capacity,
    )


def squeeze_axis_layout[rank: Int, axis: Int](
    mapping: TensorLayout[rank], capacity: Int,
) -> TensorResult[TensorLayout[rank - 1]] where rank >= 1 and rank <= 8 and axis >= 0 and axis < rank:
    var verified = mapping.validated_snapshot(capacity)
    if not verified.is_ok():
        return TensorResult[TensorLayout[rank - 1]](verified.status)
    var trusted = verified.payload.value()
    if trusted.shape.dims.get(axis) != 1:
        return TensorResult[TensorLayout[rank - 1]](TensorStatus.SIZE_MISMATCH)
    var dims = StaticRankArray[rank - 1](0)
    var strides = StaticRankArray[rank - 1](0)
    comptime for i in range(rank - 1):
        comptime if i < axis:
            dims.set(i, trusted.shape.dims.get(i))
            strides.set(i, trusted.strides.get(i))
        else:
            dims.set(i, trusted.shape.dims.get(i + 1))
            strides.set(i, trusted.strides.get(i + 1))
    var output_shape = TensorShape[rank - 1].checked(dims)
    if not output_shape.is_ok():
        return TensorResult[TensorLayout[rank - 1]](output_shape.status)
    return TensorLayout[rank - 1].checked(
        output_shape.payload.value(), strides, trusted.offset, capacity,
    )


def expand_layout[src_rank: Int, dst_rank: Int](
    mapping: TensorLayout[src_rank], target: TensorShape[dst_rank], capacity: Int,
) -> TensorResult[TensorLayout[dst_rank]] where src_rank >= 0 and dst_rank >= src_rank and dst_rank <= 8:
    var verified = mapping.validated_snapshot(capacity)
    if not verified.is_ok():
        return TensorResult[TensorLayout[dst_rank]](verified.status)
    var checked_target = TensorShape[dst_rank].checked(target.dims)
    if not checked_target.is_ok():
        return TensorResult[TensorLayout[dst_rank]](checked_target.status)
    if checked_target.payload.value().numel() != target.numel():
        return TensorResult[TensorLayout[dst_rank]](TensorStatus.INVALID_SHAPE)
    var trusted = verified.payload.value()
    var output_strides = make_broadcast_strides[src_rank, dst_rank](
        trusted.shape, trusted.strides, checked_target.payload.value(),
    )
    if not output_strides.is_ok():
        return TensorResult[TensorLayout[dst_rank]](output_strides.status)
    # Zero-stride expansion is an immutable metadata view. Writable callers
    # must separately prove uniqueness; no element is moved or allocated.
    return TensorLayout[dst_rank].checked(
        checked_target.payload.value(), output_strides.payload.value(),
        trusted.offset, capacity,
    )
