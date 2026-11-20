# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR VIEW / VIEW OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Views
# Module:        momijo.tensor_light.view.view
# Source:        src/momijo/tensor_light/view/view.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines view operations in the tensor views layer of the FP32 Light tensor
#   implementation.
#
#   Public declarations include `TensorView`, `TensorMutView`; preconditions
#   and supported targets are determined by their implementations.
#
# Responsibilities:
#   - Build safe tensor views and layout-only transformations.
#   - Preserve documented behavior for bounds, strides, view provenance, and
#     write safety.
#   - Provide source-declared interfaces such as `TensorView`,
#     `TensorMutView`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate shape/stride/span invariants for borrowed views.
#   - Enforce write restrictions for broadcast or overlapping mappings.
#
# Performance Objectives:
#   - Prefer zero-copy metadata transforms when valid.
#   - Avoid view materialization unless the API explicitly requires it.
#
# Maintainability:
#   - Keep view semantics consistent with owning-tensor operations.
#   - Test permutations, negative strides, and zero-length axes.
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

# Borrowed zero-copy views have no storage ownership or implicit materialization.
from std.collections import Span
from std.sys.info import simd_width_of
from ..shape.shape_storage import StaticRankArray
from ..shape.shape import TensorShape, flatten_shape
from ..layout.layout import TensorLayout
from ..layout.memory_format import ContiguousOrder
from ..layout.contiguity import InternalOverlapKind
from ..internal.status import TensorResult, TensorStatus
from ..view.reshape import reshape_c_view, reshape_f_view as reshape_f_layout
from ..view.as_strided import as_strided, slice_axis as slice_layout_axis
from ..view.expand import broadcast_layout, squeeze_axes as squeeze_axes_layout
from ..view.unsqueeze import unsqueeze, squeeze_axis
from ..view.transpose import movedim as movedim_layout
from ..indexing.slice import SliceSpec
from ..indexing.slice_spec import BasicIndex, NewAxis, EllipsisIndex, as_basic_index, inferred_basic_index_rank
from ..indexing.index_math import basic_index_layout

# Resolve the concrete type before calling the overload. A heterogeneous
# pack element still has a generic type when Mojo checks this function.
def _pack_basic_index[T: AnyType](ref value: T) -> BasicIndex:
    comptime if T == Int:
        return as_basic_index(rebind[Int](value))
    elif T == SliceSpec:
        return as_basic_index(rebind[SliceSpec](value))
    elif T == NewAxis:
        return as_basic_index(rebind[NewAxis](value))
    elif T == EllipsisIndex:
        return as_basic_index(rebind[EllipsisIndex](value))
    else:
        comptime assert T == BasicIndex, "unsupported basic index type"
        return as_basic_index(rebind[BasicIndex](value))


struct TensorView[rank: Int, origin: Origin](Copyable):
    var data: Span[mut=False, Float32, Self.origin]
    var layout: TensorLayout[Self.rank]

    def __init__(out self, data: Span[mut=False, Float32, Self.origin], layout: TensorLayout[Self.rank]):
        self.data = data
        self.layout = layout

    @staticmethod
    def from_span_checked(data: Span[mut=False, Float32, Self.origin], layout: TensorLayout[Self.rank]) -> TensorResult[Self]:
        var true_shape = TensorShape[Self.rank].checked(layout.shape.dims)
        if not true_shape.is_ok():
            return TensorResult[Self](true_shape.status)
        if true_shape.payload.value().numel() != layout.shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var checked_layout = TensorLayout[Self.rank].checked(
            true_shape.payload.value(), layout.strides, layout.offset, len(data)
        )
        if not checked_layout.is_ok():
            return TensorResult[Self](checked_layout.status)
        return TensorResult[Self](Self(data, checked_layout.payload.value()))

    def __init__(out self, *, copy: Self):
        self.data = copy.data
        self.layout = copy.layout.copy()

    def numel(self) -> Int:
        return self.layout.shape.numel()

    def checked_numel(self) -> TensorResult[Int]:
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Int](proof.status)
        return TensorResult[Int](proof.payload.value().shape.numel())

    def checked_nbytes(self) -> TensorResult[Int]:
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Int](proof.status)
        return TensorResult[Int](proof.payload.value().shape.nbytes())

    def _checked_current_layout(self) -> TensorResult[TensorLayout[Self.rank]]:
        return self.layout.validated_snapshot(len(self.data))

    def checked_layout(self) -> TensorResult[TensorLayout[Self.rank]]:
        return self._checked_current_layout()

    def checked_shape(self) -> TensorResult[TensorShape[Self.rank]]:
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[TensorShape[Self.rank]](proof.status)
        return TensorResult[TensorShape[Self.rank]](proof.payload.value().shape.copy())

    def get(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Float32]:
        var checked_layout = self._checked_current_layout()
        if not checked_layout.is_ok():
            return TensorResult[Float32](checked_layout.status)
        var pos = checked_layout.payload.value()._offset_from_validated(indices)
        if not pos.is_ok():
            return TensorResult[Float32](pos.status)
        var at = pos.payload.value()
        if at < 0 or at >= len(self.data):
            return TensorResult[Float32](TensorStatus.OUT_OF_BOUNDS)
        return TensorResult[Float32](self.data[at])

    def __getitem__(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Float32]:
        return self.get(indices)

    def __getitem__(self, index: Int) -> TensorResult[Float32] where Self.rank == 1:
        var coords = StaticRankArray[Self.rank](0)
        coords.set(0, index)
        return self.get(coords)

    def __getitem__(
        self, index: Int,
    ) -> TensorResult[TensorView[Self.rank - 1, Self.origin]] where Self.rank >= 2:
        return self.index_basic_as[Self.rank - 1](index)

    def is_empty(self) -> Bool:
        # Cached element counts can be changed on externally constructed
        # descriptors; extents remain the source of logical emptiness.
        return self.layout.shape.is_empty()

    def stride(self, axis: Int) -> TensorResult[Int]:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Int](checked.status)
        var a = axis
        if a < 0:
            if a < -Self.rank:
                return TensorResult[Int](TensorStatus.INVALID_AXIS)
            a += Self.rank
        if a >= Self.rank:
            return TensorResult[Int](TensorStatus.INVALID_AXIS)
        return TensorResult[Int](checked.payload.value().strides.get(a))

    def transpose(self, a: Int, b: Int) -> TensorResult[Self]:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[Self](checked_source.status)
        var next = checked_source.payload.value().transpose(a, b, len(self.data))
        if not next.is_ok():
            return TensorResult[Self](next.status)
        # The transform returns a validated, storage-bound layout.
        return TensorResult[Self](Self(self.data, next.payload.take()))

    def permute(self, axes: StaticRankArray[Self.rank]) -> TensorResult[Self]:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[Self](checked_source.status)
        var next = checked_source.payload.value().permute(axes, len(self.data))
        if not next.is_ok():
            return TensorResult[Self](next.status)
        # The transform returns a validated, storage-bound layout.
        return TensorResult[Self](Self(self.data, next.payload.take()))

    def swapaxes(self, first: Int, second: Int) -> TensorResult[Self]:
        return self.transpose(first, second)

    def movedim(self, source: Int, destination: Int) -> TensorResult[Self]:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[Self](checked_source.status)
        var mapped = movedim_layout(checked_source.payload.value(), source, destination, len(self.data))
        if not mapped.is_ok():
            return TensorResult[Self](mapped.status)
        # The transform returns a validated, storage-bound layout.
        return TensorResult[Self](Self(self.data, mapped.payload.take()))

    def flip(self, axis: Int) -> TensorResult[Self]:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[Self](checked_source.status)
        var next = checked_source.payload.value().flip(axis, len(self.data))
        if not next.is_ok():
            return TensorResult[Self](next.status)
        # The transform returns a validated, storage-bound layout.
        return TensorResult[Self](Self(self.data, next.payload.take()))

    def shape(self) -> TensorShape[Self.rank]:
        return self.layout.shape.copy()

    def ndim(self) -> Int:
        return Self.rank

    def is_scalar(self) -> Bool:
        return Self.rank == 0

    def storage_offset(self) -> Int:
        return self.layout.offset

    def dim(self, axis: Int) -> TensorResult[Int]:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Int](checked.status)
        return checked.payload.value().shape.dim(axis)

    def size(self, axis: Int) -> TensorResult[Int]:
        return self.dim(axis)

    def internal_overlap_kind(self) -> InternalOverlapKind:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return InternalOverlapKind.UNKNOWN
        return checked.payload.value().overlap

    def itemsize(self) -> Int:
        return 4

    def nbytes(self) -> Int:
        return self.layout.shape.nbytes()

    def is_c_contiguous(self) -> Bool:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return False
        return checked.payload.value().is_c_contiguous()

    def is_f_contiguous(self) -> Bool:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return False
        return checked.payload.value().is_f_contiguous()

    def is_contiguous(self, order: ContiguousOrder) -> Bool:
        # C/F contiguity remains independently defined. Unsupported orders
        # are not silently treated as either memory format.
        if order == ContiguousOrder.C:
            return self.is_c_contiguous()
        if order == ContiguousOrder.F:
            return self.is_f_contiguous()
        return False

    def is_unique(self) -> Bool:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return False
        return checked.payload.value().is_proven_unique()

    def is_exhaustive(self) -> Bool:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return False
        return checked.payload.value().is_exhaustive()

    def item(self) -> TensorResult[Float32]:
        # A one-element strided view has a single physical address even for
        # negative or non-unit strides. One structural proof is sufficient;
        # re-entering get() would revalidate the entire layout unnecessarily.
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Float32](checked.status)
        var safe = checked.payload.value()
        if safe.shape.numel() != 1:
            return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
        var at = safe.offset
        if at < 0 or at >= len(self.data):
            return TensorResult[Float32](TensorStatus.OUT_OF_BOUNDS)
        return TensorResult[Float32](self.data[at])

    def index_view[out_rank: Int, count: Int](
        self, terms: Array[BasicIndex, count]
    ) -> TensorResult[TensorView[out_rank, Self.origin]]:
        comptime assert out_rank >= 0 and out_rank <= 8, "view index result rank must be 0..8"
        comptime assert count >= 1 and count <= 17, "a fixed Array index pack must contain 1..17 terms"
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[out_rank, Self.origin]](checked_source.status)
        var mapped = basic_index_layout[Self.rank, out_rank, count](
            checked_source.payload.value(), terms, len(self.data)
        )
        if not mapped.is_ok():
            return TensorResult[TensorView[out_rank, Self.origin]](mapped.status)
        return TensorResult[TensorView[out_rank, Self.origin]](
            TensorView[out_rank, Self.origin](self.data, mapped.payload.take())
        )

    def index_basic_as[out_rank: Int, *IndexTypes: AnyType](
        self, *terms: *IndexTypes
    ) -> TensorResult[TensorView[out_rank, Self.origin]]:
        # Heterogeneous runtime index terms are normalized by a compile-time
        # selected conversion overload. There is no heap/virtual dispatch.
        comptime count = terms.__len__()
        comptime assert count <= 17, "basic index pack must have at most 17 terms"
        comptime assert out_rank >= 0 and out_rank <= 8, "output rank must be 0..8"
        comptime if count == 0:
            comptime assert out_rank == Self.rank, "identity index must preserve rank"
            return TensorView[out_rank, Self.origin].from_span_checked(
                self.data, rebind[TensorLayout[out_rank]](self.layout)
            )
        else:
            var packed = Array[BasicIndex, count](fill=BasicIndex.full_slice())
            comptime for i in range(count):
                packed[i] = _pack_basic_index(terms[i])
            return self.index_view[out_rank, count](packed)

    # Auto rank inference for statically typed heterogeneous indexes.
    # A BasicIndex runtime-tagged list still requires index_view[out_rank].
    def index_basic[*IndexTypes: AnyType](
        self, *terms: *IndexTypes
    ) -> TensorResult[TensorView[inferred_basic_index_rank[Self.rank, *IndexTypes](), Self.origin]]:
        comptime out_rank = inferred_basic_index_rank[Self.rank, *IndexTypes]()
        comptime assert out_rank >= 0 and out_rank <= 8, "invalid heterogeneous index signature"
        return self.index_basic_as[out_rank](*terms)

    # Slice/ellipsis/newaxis sugar shares the same basic_index_layout proof
    # as the owning Tensor facade. No copy or fallback materialization occurs.
    def __getitem__(
        self, spec: SliceSpec,
    ) -> TensorResult[TensorView[Self.rank, Self.origin]] where Self.rank >= 1:
        return self.index_basic_as[Self.rank](spec)

    def __getitem__(
        self, spec: EllipsisIndex,
    ) -> TensorResult[TensorView[Self.rank, Self.origin]]:
        return self.index_basic_as[Self.rank](spec)

    def __getitem__(
        self, spec: NewAxis,
    ) -> TensorResult[TensorView[Self.rank + 1, Self.origin]] where Self.rank < 8:
        return self.index_basic_as[Self.rank + 1](spec)

    def slice_axis(self, axis: Int, slice_spec: SliceSpec) -> TensorResult[Self]:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[Self](checked_source.status)
        var mapped = slice_layout_axis(checked_source.payload.value(), axis, slice_spec, len(self.data))
        if not mapped.is_ok():
            return TensorResult[Self](mapped.status)
        # The transform returns a validated, storage-bound layout.
        return TensorResult[Self](Self(self.data, mapped.payload.take()))

    def narrow(self, axis: Int, start: Int, length: Int) -> TensorResult[Self]:
        # A borrowed view exposes mutable metadata through its value fields.
        # Rebuild the current storage-relative proof BEFORE consulting shape
        # extents or computing the slice endpoint. Use this same snapshot for
        # the transformation to avoid a second checked traversal.
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[Self](source.status)
        var a = axis
        if a < 0:
            if a < -Self.rank:
                return TensorResult[Self](TensorStatus.INVALID_AXIS)
            a += Self.rank
        if a >= Self.rank:
            return TensorResult[Self](TensorStatus.INVALID_AXIS)
        var extent = source.payload.value().shape.dims.get(a)
        if start < 0 or length < 0 or start > extent or length > extent - start:
            return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
        var mapped = slice_layout_axis(
            source.payload.value(), a, SliceSpec(start, start + length, 1),
            len(self.data),
        )
        if not mapped.is_ok():
            return TensorResult[Self](mapped.status)
        # The transform returns a validated, storage-bound layout.
        return TensorResult[Self](Self(self.data, mapped.payload.take()))

    def flatten_scalar_view(self) -> TensorResult[TensorView[1, Self.origin]] where Self.rank == 0:
        var one = TensorShape[1].checked(StaticRankArray[1](1))
        if not one.is_ok():
            return TensorResult[TensorView[1, Self.origin]](one.status)
        return self.reshape_view[1](one.payload.value())

    def reshape_view[new_rank: Int](self, target: TensorShape[new_rank]) -> TensorResult[TensorView[new_rank, Self.origin]]:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[new_rank, Self.origin]](checked_source.status)
        var mapped = reshape_c_view[Self.rank, new_rank](checked_source.payload.value(), target, len(self.data))
        if not mapped.is_ok():
            return TensorResult[TensorView[new_rank, Self.origin]](mapped.status)
        return TensorResult[TensorView[new_rank, Self.origin]](
            TensorView[new_rank, Self.origin](self.data, mapped.payload.take())
        )

    def reshape_f_view[new_rank: Int](self, target: TensorShape[new_rank]) -> TensorResult[TensorView[new_rank, Self.origin]]:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[new_rank, Self.origin]](checked_source.status)
        var mapped = reshape_f_layout[Self.rank, new_rank](checked_source.payload.value(), target, len(self.data))
        if not mapped.is_ok():
            return TensorResult[TensorView[new_rank, Self.origin]](mapped.status)
        return TensorResult[TensorView[new_rank, Self.origin]](
            TensorView[new_rank, Self.origin](self.data, mapped.payload.take())
        )

    def ravel_view(self) -> TensorResult[TensorView[1, Self.origin]]:
        # C-order, borrow-only flatten for rank 0..8. A transposed or
        # broadcast layout that cannot be flattened without a copy must
        # return NOT_REPRESENTABLE rather than silently materialize.
        var count = self.layout.shape.checked_numel()
        if not count.is_ok():
            return TensorResult[TensorView[1, Self.origin]](count.status)
        var target = TensorShape[1].checked(
            StaticRankArray[1](count.payload.value())
        )
        if not target.is_ok():
            return TensorResult[TensorView[1, Self.origin]](target.status)
        var mapped = reshape_c_view[Self.rank, 1](
            self.layout, target.payload.value(), len(self.data),
        )
        if not mapped.is_ok():
            return TensorResult[TensorView[1, Self.origin]](mapped.status)
        return TensorResult[TensorView[1, Self.origin]](
            TensorView[1, Self.origin](self.data, mapped.payload.take())
        )

    def ravel_f_view(self) -> TensorResult[TensorView[1, Self.origin]]:
        # Explicit column-major view-only linearization; never copies data.
        var count = self.layout.shape.checked_numel()
        if not count.is_ok():
            return TensorResult[TensorView[1, Self.origin]](count.status)
        var target = TensorShape[1].checked(
            StaticRankArray[1](count.payload.value())
        )
        if not target.is_ok():
            return TensorResult[TensorView[1, Self.origin]](target.status)
        var mapped = reshape_f_layout[Self.rank, 1](
            self.layout, target.payload.value(), len(self.data),
        )
        if not mapped.is_ok():
            return TensorResult[TensorView[1, Self.origin]](mapped.status)
        return TensorResult[TensorView[1, Self.origin]](
            TensorView[1, Self.origin](self.data, mapped.payload.take())
        )

    def flatten_view[start_dim: Int, end_dim: Int](
        self
    ) -> TensorResult[TensorView[Self.rank - (end_dim - start_dim), Self.origin]] where Self.rank >= 1:
        comptime assert start_dim >= 0 and end_dim >= start_dim and end_dim < Self.rank, "invalid flatten axis interval"
        comptime out_rank = Self.rank - (end_dim - start_dim)
        # Validate shape locally and perform the storage-bound span proof
        # once inside reshape_c_view, not twice at the view boundary.
        var target = flatten_shape[Self.rank, start_dim, end_dim](
            self.layout.shape
        )
        if not target.is_ok():
            return TensorResult[TensorView[out_rank, Self.origin]](target.status)
        var mapped = reshape_c_view[Self.rank, out_rank](
            self.layout, target.payload.value(), len(self.data),
        )
        if not mapped.is_ok():
            return TensorResult[TensorView[out_rank, Self.origin]](mapped.status)
        return TensorResult[TensorView[out_rank, Self.origin]](
            TensorView[out_rank, Self.origin](self.data, mapped.payload.take())
        )

    def expand_view[new_rank: Int](self, target: TensorShape[new_rank]) -> TensorResult[TensorView[new_rank, Self.origin]] where new_rank >= Self.rank and new_rank <= 8:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[new_rank, Self.origin]](checked_source.status)
        comptime assert Self.rank >= 0
        var mapped = broadcast_layout[Self.rank, new_rank](checked_source.payload.value(), target, len(self.data))
        if not mapped.is_ok():
            return TensorResult[TensorView[new_rank, Self.origin]](mapped.status)
        return TensorResult[TensorView[new_rank, Self.origin]](
            TensorView[new_rank, Self.origin](self.data, mapped.payload.take())
        )

    def unsqueeze_view[axis: Int](self) -> TensorResult[TensorView[Self.rank + 1, Self.origin]] where Self.rank < 8:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[Self.rank + 1, Self.origin]](checked_source.status)
        comptime assert Self.rank >= 0 and Self.rank < 8 and axis >= 0 and axis <= Self.rank
        var mapped = unsqueeze[Self.rank, axis](checked_source.payload.value(), len(self.data))
        if not mapped.is_ok():
            return TensorResult[TensorView[Self.rank + 1, Self.origin]](mapped.status)
        return TensorResult[TensorView[Self.rank + 1, Self.origin]](
            TensorView[Self.rank + 1, Self.origin](self.data, mapped.payload.take())
        )

    def squeeze_axis_view[axis: Int](self) -> TensorResult[TensorView[Self.rank - 1, Self.origin]] where Self.rank > 0:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[Self.rank - 1, Self.origin]](checked_source.status)
        comptime assert Self.rank >= 1 and Self.rank <= 8 and axis >= 0 and axis < Self.rank
        var mapped = squeeze_axis[Self.rank, axis](checked_source.payload.value(), len(self.data))
        if not mapped.is_ok():
            return TensorResult[TensorView[Self.rank - 1, Self.origin]](mapped.status)
        return TensorResult[TensorView[Self.rank - 1, Self.origin]](
            TensorView[Self.rank - 1, Self.origin](self.data, mapped.payload.take())
        )

    def squeeze_axes_view[count: Int](
        self, axes: StaticRankArray[count],
    ) -> TensorResult[TensorView[Self.rank - count, Self.origin]] where count >= 0 and count <= Self.rank:
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[TensorView[Self.rank - count, Self.origin]](proof.status)
        comptime assert Self.rank >= 0 and Self.rank <= 8 and count >= 0 and count <= Self.rank
        var reduced = squeeze_axes_layout[Self.rank, count](
            proof.payload.value(), axes, len(self.data),
        )
        if not reduced.is_ok():
            return TensorResult[TensorView[Self.rank - count, Self.origin]](reduced.status)
        return TensorResult[TensorView[Self.rank - count, Self.origin]](
            TensorView[Self.rank - count, Self.origin](self.data, reduced.payload.take())
        )

    def as_strided_view[new_rank: Int](
        self, shape: TensorShape[new_rank], strides: StaticRankArray[new_rank], offset: Int
    ) -> TensorResult[TensorView[new_rank, Self.origin]]:
        # Explicitly addresses the underlying storage, not only this view's
        # logical subset; the entire new span is checked against storage size.
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[new_rank, Self.origin]](checked_source.status)
        var mapped = as_strided[Self.rank, new_rank](checked_source.payload.value(), shape, strides, offset, len(self.data))
        if not mapped.is_ok():
            return TensorResult[TensorView[new_rank, Self.origin]](mapped.status)
        return TensorResult[TensorView[new_rank, Self.origin]](
            TensorView[new_rank, Self.origin](self.data, mapped.payload.take())
        )

struct TensorMutView[rank: Int, origin: Origin[mut=True]](Movable):
    var data: Span[mut=True, Float32, Self.origin]
    var layout: TensorLayout[Self.rank]

    def __init__(out self, data: Span[mut=True, Float32, Self.origin], layout: TensorLayout[Self.rank]):
        self.data = data
        self.layout = layout

    @staticmethod
    def from_span_checked(data: Span[mut=True, Float32, Self.origin], layout: TensorLayout[Self.rank]) -> TensorResult[Self]:
        var true_shape = TensorShape[Self.rank].checked(layout.shape.dims)
        if not true_shape.is_ok():
            return TensorResult[Self](true_shape.status)
        if true_shape.payload.value().numel() != layout.shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var checked_layout = TensorLayout[Self.rank].checked(
            true_shape.payload.value(), layout.strides, layout.offset, len(data)
        )
        if not checked_layout.is_ok():
            return TensorResult[Self](checked_layout.status)
        if not checked_layout.payload.value().is_proven_unique():
            return TensorResult[Self](TensorStatus.OVERLAPPING)
        return TensorResult[Self](Self(data, checked_layout.payload.value()))

    def _checked_current_layout(self) -> TensorResult[TensorLayout[Self.rank]]:
        return self.layout.validated_snapshot(len(self.data))

    def checked_layout(self) -> TensorResult[TensorLayout[Self.rank]]:
        return self._checked_current_layout()

    def checked_shape(self) -> TensorResult[TensorShape[Self.rank]]:
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[TensorShape[Self.rank]](proof.status)
        return TensorResult[TensorShape[Self.rank]](proof.payload.value().shape.copy())

    def get(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Float32]:
        var checked_layout = self._checked_current_layout()
        if not checked_layout.is_ok():
            return TensorResult[Float32](checked_layout.status)
        var pos = checked_layout.payload.value()._offset_from_validated(indices)
        if not pos.is_ok():
            return TensorResult[Float32](pos.status)
        var at = pos.payload.value()
        if at < 0 or at >= len(self.data):
            return TensorResult[Float32](TensorStatus.OUT_OF_BOUNDS)
        return TensorResult[Float32](self.data[at])

    def set(mut self, indices: StaticRankArray[Self.rank], value: Float32) -> TensorStatus:
        # A mutable borrowed view exposes its layout metadata to callers. Do
        # not trust cached overlap or span flags as a writeability proof.
        var rebuilt = self._checked_current_layout()
        if not rebuilt.is_ok():
            return rebuilt.status
        var safe = rebuilt.payload.value()
        if not safe.is_proven_unique():
            return TensorStatus.OVERLAPPING
        var pos = safe._offset_from_validated(indices)
        if not pos.is_ok():
            return pos.status
        var at = pos.payload.value()
        if at < 0 or at >= len(self.data):
            return TensorStatus.OUT_OF_BOUNDS
        self.data[at] = value
        return TensorStatus.OK


    # Mutable borrowed views are capabilities over an existing owner. Queries
    # never allocate and scalar writes remain checked for mapping uniqueness.
    def shape(self) -> TensorShape[Self.rank]:
        return self.layout.shape.copy()

    def ndim(self) -> Int:
        return Self.rank

    def is_scalar(self) -> Bool:
        return Self.rank == 0

    def storage_offset(self) -> Int:
        return self.layout.offset

    def dim(self, axis: Int) -> TensorResult[Int]:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Int](checked.status)
        return checked.payload.value().shape.dim(axis)

    def size(self, axis: Int) -> TensorResult[Int]:
        return self.dim(axis)

    def internal_overlap_kind(self) -> InternalOverlapKind:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return InternalOverlapKind.UNKNOWN
        return checked.payload.value().overlap

    def numel(self) -> Int:
        return self.layout.shape.numel()

    def checked_numel(self) -> TensorResult[Int]:
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Int](proof.status)
        return TensorResult[Int](proof.payload.value().shape.numel())

    def checked_nbytes(self) -> TensorResult[Int]:
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Int](proof.status)
        return TensorResult[Int](proof.payload.value().shape.nbytes())

    def nbytes(self) -> Int:
        return self.layout.shape.nbytes()

    def itemsize(self) -> Int:
        return 4

    def is_empty(self) -> Bool:
        # Like Tensor and immutable TensorView, mutable borrowed metadata may
        # contain a stale cached element count. Logical extents determine
        # emptiness; the checked shape query reports malformed metadata.
        return self.layout.shape.is_empty()

    def is_unique(self) -> Bool:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return False
        return checked.payload.value().is_proven_unique()

    def is_c_contiguous(self) -> Bool:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return False
        return checked.payload.value().is_c_contiguous()

    def is_f_contiguous(self) -> Bool:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return False
        return checked.payload.value().is_f_contiguous()

    def is_contiguous(self, order: ContiguousOrder) -> Bool:
        # C/F contiguity remains independently defined. Unsupported orders
        # are not silently treated as either memory format.
        if order == ContiguousOrder.C:
            return self.is_c_contiguous()
        if order == ContiguousOrder.F:
            return self.is_f_contiguous()
        return False

    def is_exhaustive(self) -> Bool:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return False
        return checked.payload.value().is_exhaustive()

    def stride(self, axis: Int) -> TensorResult[Int]:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Int](checked.status)
        var a = axis
        if a < 0:
            if a < -Self.rank:
                return TensorResult[Int](TensorStatus.INVALID_AXIS)
            a += Self.rank
        if a >= Self.rank:
            return TensorResult[Int](TensorStatus.INVALID_AXIS)
        return TensorResult[Int](checked.payload.value().strides.get(a))

    def item(self) -> TensorResult[Float32]:
        # A one-element strided view has a single physical address even for
        # negative or non-unit strides. One structural proof is sufficient;
        # re-entering get() would revalidate the entire layout unnecessarily.
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Float32](checked.status)
        var safe = checked.payload.value()
        if safe.shape.numel() != 1:
            return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
        var at = safe.offset
        if at < 0 or at >= len(self.data):
            return TensorResult[Float32](TensorStatus.OUT_OF_BOUNDS)
        return TensorResult[Float32](self.data[at])

    def set_item(mut self, value: Float32) -> TensorStatus:
        # The writable mapping must be unique, even when the public layout
        # metadata was modified after construction. Check once and write the
        # proven single element directly instead of re-entering set().
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return checked.status
        var safe = checked.payload.value()
        if not safe.is_proven_unique():
            return TensorStatus.OVERLAPPING
        if safe.shape.numel() != 1:
            return TensorStatus.SIZE_MISMATCH
        var at = safe.offset
        if at < 0 or at >= len(self.data):
            return TensorStatus.OUT_OF_BOUNDS
        self.data[at] = value
        return TensorStatus.OK

    def __getitem__(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Float32]:
        return self.get(indices)

    def __getitem__(self, index: Int) -> TensorResult[Float32] where Self.rank == 1:
        var coords = StaticRankArray[Self.rank](0)
        coords.set(0, index)
        return self.get(coords)

    def __getitem__(
        mut self, index: Int,
    ) -> TensorResult[TensorMutView[Self.rank - 1, Self.origin]] where Self.rank >= 2:
        return self._index_basic_mut_as[Self.rank - 1](index)

    def __setitem__(mut self, indices: StaticRankArray[Self.rank], value: Float32) -> TensorStatus:
        return self.set(indices, value)

    def __setitem__(mut self, index: Int, value: Float32) -> TensorStatus where Self.rank == 1:
        var coords = StaticRankArray[Self.rank](0)
        coords.set(0, index)
        return self.set(coords, value)

    def __setitem__(mut self, index: Int, value: Float32) -> TensorStatus where Self.rank >= 2:
        var selected = self._index_basic_mut_as[Self.rank - 1](index)
        if not selected.is_ok():
            return selected.status
        var writable = selected.payload.take()
        return writable.fill(value)

    # A mutable subview is granted only after reconstructing bounds and
    # proving the SELECTED mapping unique. These methods never allocate or
    # read a different device implicitly. Scalar indexing remains get/set.
    def index_mut_view[out_rank: Int, count: Int](
        mut self, terms: Array[BasicIndex, count],
    ) -> TensorResult[TensorMutView[out_rank, Self.origin]]:
        comptime assert out_rank >= 0 and out_rank <= 8, "mutable index output rank must be 0..8"
        comptime assert count >= 1 and count <= 17, "a fixed Array index pack must contain 1..17 terms"
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[TensorMutView[out_rank, Self.origin]](source.status)
        var mapped = basic_index_layout[Self.rank, out_rank, count](
            source.payload.value(), terms, len(self.data),
        )
        if not mapped.is_ok():
            return TensorResult[TensorMutView[out_rank, Self.origin]](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[TensorMutView[out_rank, Self.origin]](TensorStatus.OVERLAPPING)
        return TensorResult[TensorMutView[out_rank, Self.origin]](
            TensorMutView[out_rank, Self.origin](self.data, selected)
        )

    def _index_basic_mut_as[out_rank: Int, *IndexTypes: AnyType](
        mut self, *terms: *IndexTypes,
    ) -> TensorResult[TensorMutView[out_rank, Self.origin]]:
        comptime count = terms.__len__()
        comptime assert count <= 17, "mutable basic index pack must have at most 17 terms"
        comptime if count == 0:
            # Rank zero and identity indexing do not construct Array[T, 0].
            comptime assert out_rank == Self.rank, "identity index must preserve rank"
            return TensorMutView[out_rank, Self.origin].from_span_checked(
                self.data, rebind[TensorLayout[out_rank]](self.layout)
            )
        else:
            var packed = Array[BasicIndex, count](fill=BasicIndex.full_slice())
            comptime for i in range(count):
                packed[i] = _pack_basic_index(terms[i])
            return self.index_mut_view[out_rank, count](packed)

    def index_basic_mut[*IndexTypes: AnyType](
        mut self, *terms: *IndexTypes,
    ) -> TensorResult[TensorMutView[inferred_basic_index_rank[Self.rank, *IndexTypes](), Self.origin]]:
        comptime out_rank = inferred_basic_index_rank[Self.rank, *IndexTypes]()
        comptime assert out_rank >= 0 and out_rank <= 8, "invalid mutable index signature"
        return self._index_basic_mut_as[out_rank](*terms)

    def __getitem__(
        mut self, spec: SliceSpec,
    ) -> TensorResult[TensorMutView[Self.rank, Self.origin]] where Self.rank >= 1:
        return self._index_basic_mut_as[Self.rank](spec)

    def __getitem__(
        mut self, spec: EllipsisIndex,
    ) -> TensorResult[TensorMutView[Self.rank, Self.origin]]:
        return self._index_basic_mut_as[Self.rank](spec)

    def __getitem__(
        mut self, spec: NewAxis,
    ) -> TensorResult[TensorMutView[Self.rank + 1, Self.origin]] where Self.rank < 8:
        return self._index_basic_mut_as[Self.rank + 1](spec)

    def __setitem__(
        mut self, spec: SliceSpec, value: Float32,
    ) -> TensorStatus where Self.rank >= 1:
        var selected = self._index_basic_mut_as[Self.rank](spec)
        if not selected.is_ok():
            return selected.status
        var writable = selected.payload.take()
        return writable.fill(value)

    def slice_axis_mut(mut self, axis: Int, slice_spec: SliceSpec) -> TensorResult[Self]:
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[Self](source.status)
        var next = slice_layout_axis(
            source.payload.value(), axis, slice_spec, len(self.data),
        )
        if not next.is_ok():
            return TensorResult[Self](next.status)
        var selected = next.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[Self](TensorStatus.OVERLAPPING)
        return TensorResult[Self](Self(self.data, selected))

    def transpose_mut(mut self, first: Int, second: Int) -> TensorResult[Self]:
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[Self](source.status)
        var mapped = source.payload.value().transpose(first, second, len(self.data))
        if not mapped.is_ok():
            return TensorResult[Self](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[Self](TensorStatus.OVERLAPPING)
        return TensorResult[Self](Self(self.data, selected))

    def permute_mut(
        mut self, axes: StaticRankArray[Self.rank],
    ) -> TensorResult[Self]:
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[Self](source.status)
        var mapped = source.payload.value().permute(axes, len(self.data))
        if not mapped.is_ok():
            return TensorResult[Self](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[Self](TensorStatus.OVERLAPPING)
        return TensorResult[Self](Self(self.data, selected))

    def reshape_mut_view[new_rank: Int](
        mut self, target: TensorShape[new_rank],
    ) -> TensorResult[TensorMutView[new_rank, Self.origin]]:
        comptime assert new_rank >= 0 and new_rank <= 8, "reshape output rank must be 0..8"
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[TensorMutView[new_rank, Self.origin]](source.status)
        var mapped = reshape_c_view[Self.rank, new_rank](
            source.payload.value(), target, len(self.data),
        )
        if not mapped.is_ok():
            return TensorResult[TensorMutView[new_rank, Self.origin]](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[TensorMutView[new_rank, Self.origin]](TensorStatus.OVERLAPPING)
        return TensorResult[TensorMutView[new_rank, Self.origin]](
            TensorMutView[new_rank, Self.origin](self.data, selected)
        )

    def flip_mut(mut self, axis: Int) -> TensorResult[Self]:
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[Self](source.status)
        var mapped = source.payload.value().flip(axis, len(self.data))
        if not mapped.is_ok():
            return TensorResult[Self](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[Self](TensorStatus.OVERLAPPING)
        return TensorResult[Self](Self(self.data, selected))

    def narrow_mut(mut self, axis: Int, start: Int, length: Int) -> TensorResult[Self]:
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[Self](source.status)
        var normalized = axis
        if normalized < 0:
            if normalized < -Self.rank:
                return TensorResult[Self](TensorStatus.INVALID_AXIS)
            normalized += Self.rank
        if normalized >= Self.rank:
            return TensorResult[Self](TensorStatus.INVALID_AXIS)
        var extent = source.payload.value().shape.dims.get(normalized)
        if start < 0 or length < 0 or start > extent or length > extent - start:
            return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
        var mapped = slice_layout_axis(
            source.payload.value(), normalized, SliceSpec(start, start + length, 1),
            len(self.data),
        )
        if not mapped.is_ok():
            return TensorResult[Self](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[Self](TensorStatus.OVERLAPPING)
        return TensorResult[Self](Self(self.data, selected))

    def movedim_mut(mut self, source_axis: Int, destination: Int) -> TensorResult[Self]:
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[Self](source.status)
        var mapped = movedim_layout(
            source.payload.value(), source_axis, destination, len(self.data),
        )
        if not mapped.is_ok():
            return TensorResult[Self](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[Self](TensorStatus.OVERLAPPING)
        return TensorResult[Self](Self(self.data, selected))

    def reshape_f_mut_view[new_rank: Int](
        mut self, target: TensorShape[new_rank],
    ) -> TensorResult[TensorMutView[new_rank, Self.origin]]:
        comptime assert new_rank >= 0 and new_rank <= 8, "reshape output rank must be 0..8"
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[TensorMutView[new_rank, Self.origin]](source.status)
        var mapped = reshape_f_layout[Self.rank, new_rank](
            source.payload.value(), target, len(self.data),
        )
        if not mapped.is_ok():
            return TensorResult[TensorMutView[new_rank, Self.origin]](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[TensorMutView[new_rank, Self.origin]](TensorStatus.OVERLAPPING)
        return TensorResult[TensorMutView[new_rank, Self.origin]](
            TensorMutView[new_rank, Self.origin](self.data, selected)
        )

    def unsqueeze_mut[axis: Int](
        mut self,
    ) -> TensorResult[TensorMutView[Self.rank + 1, Self.origin]] where Self.rank < 8:
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[TensorMutView[Self.rank + 1, Self.origin]](source.status)
        comptime assert Self.rank >= 0 and Self.rank < 8 and axis >= 0 and axis <= Self.rank
        var mapped = unsqueeze[Self.rank, axis](source.payload.value(), len(self.data))
        if not mapped.is_ok():
            return TensorResult[TensorMutView[Self.rank + 1, Self.origin]](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[TensorMutView[Self.rank + 1, Self.origin]](TensorStatus.OVERLAPPING)
        return TensorResult[TensorMutView[Self.rank + 1, Self.origin]](
            TensorMutView[Self.rank + 1, Self.origin](self.data, selected)
        )

    def squeeze_axis_mut[axis: Int](
        mut self,
    ) -> TensorResult[TensorMutView[Self.rank - 1, Self.origin]] where Self.rank > 0:
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[TensorMutView[Self.rank - 1, Self.origin]](source.status)
        comptime assert Self.rank >= 1 and Self.rank <= 8 and axis >= 0 and axis < Self.rank
        var mapped = squeeze_axis[Self.rank, axis](source.payload.value(), len(self.data))
        if not mapped.is_ok():
            return TensorResult[TensorMutView[Self.rank - 1, Self.origin]](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[TensorMutView[Self.rank - 1, Self.origin]](TensorStatus.OVERLAPPING)
        return TensorResult[TensorMutView[Self.rank - 1, Self.origin]](
            TensorMutView[Self.rank - 1, Self.origin](self.data, selected)
        )

    def squeeze_axes_mut[count: Int](
        mut self, axes: StaticRankArray[count],
    ) -> TensorResult[TensorMutView[Self.rank - count, Self.origin]] where count >= 0 and count <= Self.rank:
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[TensorMutView[Self.rank - count, Self.origin]](source.status)
        comptime assert Self.rank >= 0 and Self.rank <= 8 and count >= 0 and count <= Self.rank
        var mapped = squeeze_axes_layout[Self.rank, count](
            source.payload.value(), axes, len(self.data),
        )
        if not mapped.is_ok():
            return TensorResult[TensorMutView[Self.rank - count, Self.origin]](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[TensorMutView[Self.rank - count, Self.origin]](TensorStatus.OVERLAPPING)
        return TensorResult[TensorMutView[Self.rank - count, Self.origin]](
            TensorMutView[Self.rank - count, Self.origin](self.data, selected)
        )

    def expand_mut_view[new_rank: Int](
        mut self, target: TensorShape[new_rank],
    ) -> TensorResult[TensorMutView[new_rank, Self.origin]] where new_rank >= Self.rank and new_rank <= 8:
        # Expanding an extent-1 axis creates zero-stride aliases and is NOT a
        # writable tensor mapping. Identity/singleton-only expansions are fine.
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[TensorMutView[new_rank, Self.origin]](source.status)
        comptime assert Self.rank >= 0
        var mapped = broadcast_layout[Self.rank, new_rank](
            source.payload.value(), target, len(self.data),
        )
        if not mapped.is_ok():
            return TensorResult[TensorMutView[new_rank, Self.origin]](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[TensorMutView[new_rank, Self.origin]](TensorStatus.OVERLAPPING)
        return TensorResult[TensorMutView[new_rank, Self.origin]](
            TensorMutView[new_rank, Self.origin](self.data, selected)
        )

    def as_strided_mut_view[new_rank: Int](
        mut self, shape: TensorShape[new_rank],
        strides: StaticRankArray[new_rank], offset: Int,
    ) -> TensorResult[TensorMutView[new_rank, Self.origin]]:
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[TensorMutView[new_rank, Self.origin]](source.status)
        var mapped = as_strided[Self.rank, new_rank](
            source.payload.value(), shape, strides, offset, len(self.data),
        )
        if not mapped.is_ok():
            return TensorResult[TensorMutView[new_rank, Self.origin]](mapped.status)
        var selected = mapped.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[TensorMutView[new_rank, Self.origin]](TensorStatus.OVERLAPPING)
        return TensorResult[TensorMutView[new_rank, Self.origin]](
            TensorMutView[new_rank, Self.origin](self.data, selected)
        )

    def ravel_mut_view(
        mut self,
    ) -> TensorResult[TensorMutView[1, Self.origin]]:
        var count = self.layout.shape.checked_numel()
        if not count.is_ok():
            return TensorResult[TensorMutView[1, Self.origin]](count.status)
        var target = TensorShape[1].checked(
            StaticRankArray[1](count.payload.value()),
        )
        if not target.is_ok():
            return TensorResult[TensorMutView[1, Self.origin]](target.status)
        return self.reshape_mut_view[1](target.payload.value())

    def ravel_f_mut_view(
        mut self,
    ) -> TensorResult[TensorMutView[1, Self.origin]]:
        var count = self.layout.shape.checked_numel()
        if not count.is_ok():
            return TensorResult[TensorMutView[1, Self.origin]](count.status)
        var target = TensorShape[1].checked(
            StaticRankArray[1](count.payload.value()),
        )
        if not target.is_ok():
            return TensorResult[TensorMutView[1, Self.origin]](target.status)
        return self.reshape_f_mut_view[1](target.payload.value())

    def flatten_mut_view[start_dim: Int, end_dim: Int](
        mut self,
    ) -> TensorResult[TensorMutView[Self.rank - (end_dim - start_dim), Self.origin]] where Self.rank >= 1:
        comptime assert start_dim >= 0 and end_dim >= start_dim and end_dim < Self.rank, "invalid flatten axes"
        comptime out_rank = Self.rank - (end_dim - start_dim)
        var target = flatten_shape[Self.rank, start_dim, end_dim](self.layout.shape)
        if not target.is_ok():
            return TensorResult[TensorMutView[out_rank, Self.origin]](target.status)
        return self.reshape_mut_view[out_rank](target.payload.value())

    def fill(mut self, value: Float32) -> TensorStatus:
        # Validate only once, before starting a fast unchecked loop. The
        # rebuilt metadata rather than caller-mutable cached span flags is
        # the basis for uniqueness and memory-bound proofs.
        var layout_result = self._checked_current_layout()
        if not layout_result.is_ok():
            return layout_result.status
        var safe = layout_result.payload.value()
        if not safe.is_proven_unique():
            return TensorStatus.OVERLAPPING
        if safe.shape.numel() == 0:
            return TensorStatus.OK
        # Exhaustive/unique means the tensor writes each element of one
        # physical contiguous interval exactly once, even after a transpose
        # or a negative-stride flip. Use wide, unchecked FP32 stores after
        # the single allocation/span proof, not per-element Span bounds checks.
        var raw = self.data.unsafe_ptr()
        if safe.is_exhaustive():
            comptime width = simd_width_of[DType.float32]()
            var n = safe.shape.numel()
            var end = n - n % width
            var packed = SIMD[DType.float32, width](value)
            for i in range(0, end, width):
                raw.unsafe_store(safe.span_lo + i, packed)
            for i in range(end, n):
                raw.unsafe_store(safe.span_lo + i, value)
            return TensorStatus.OK
        # Signed and gapped mappings traverse their prevalidated affine
        # offsets; there can be no out-of-bounds address or overlapping write.
        var cursor = StaticRankArray[Self.rank](0)
        for _ in range(safe.shape.numel()):
            var at = safe.physical_offset_unchecked(cursor)
            raw.unsafe_store(at, value)
            comptime if Self.rank > 0:
                for j in range(Self.rank):
                    var axis = Self.rank - 1 - j
                    cursor.set(axis, cursor.get(axis) + 1)
                    if cursor.get(axis) < safe.shape.dims.get(axis):
                        break
                    cursor.set(axis, 0)
        return TensorStatus.OK
