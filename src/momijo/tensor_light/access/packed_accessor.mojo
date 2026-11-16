# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR ACCESS / PACKED TENSOR ACCESSORS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Access and References
# Module:        momijo.tensor_light.access.packed_accessor
# Source:        src/momijo/tensor_light/access/packed_accessor.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines packed tensor accessors in the access and references layer of the
#   FP32 Light tensor implementation.
#
#   Public declarations include `PackedHostAccessor`, `can_use_i32_indexing`,
#   `pack_accessor`, `unpack_shape`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Provide checked tensor element access without exceeding storage bounds.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `PackedHostAccessor`,
#     `can_use_i32_indexing`, `pack_accessor`, `unpack_shape`,
#     `offset_unchecked`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate index bounds, backing capacity, and access mutability.
#   - Never expose an element reference beyond its owning storage lifetime.
#
# Performance Objectives:
#   - Keep checked metadata setup outside element-wise hot loops.
#   - Avoid copying tensor storage when a verified accessor is sufficient.
#
# Maintainability:
#   - Document borrowing, aliasing, and raw-pointer preconditions.
#   - Test strided, empty, and offset access paths.
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

# Compact rank-static metadata for host-side FP32 kernel arguments.
# GPU lowering must construct a native device accessor, not smuggle host Span.
from std.collections import Span
from ..shape.shape_storage import StaticRankArray
from ..shape.shape import TensorShape
from ..layout.layout import TensorLayout
from ..view.view import TensorView
from ..internal.status import TensorResult

struct PackedHostAccessor[rank: Int, origin: Origin](Copyable):
    var span: Span[False, Float32, Self.origin]
    var dims: StaticRankArray[Self.rank]
    var strides: StaticRankArray[Self.rank]
    var offset: Int

    def __init__(out self, view: TensorView[Self.rank, Self.origin]):
        self.span = view.data
        self.dims = view.layout.shape.dims.copy()
        self.strides = view.layout.strides.copy()
        self.offset = view.layout.offset

    def __init__(out self, *, copy: Self):
        self.span = copy.span
        self.dims = copy.dims.copy()
        self.strides = copy.strides.copy()
        self.offset = copy.offset

    @staticmethod
    def checked(view: TensorView[Self.rank, Self.origin]) -> TensorResult[Self]:
        # Reject metadata fabricated or modified after view construction.
        # This is the one-time validation boundary before an unchecked AOT
        # kernel loop, not an allocation or a device-address-space bridge.
        var proof = view._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Self](proof.status)
        # Capture the newly verified snapshot, not the mutable original
        # descriptor whose cached flags/span may have been tampered with.
        var trusted_view = TensorView[Self.rank, Self.origin](
            view.data, proof.payload.value(),
        )
        return TensorResult[Self](Self(trusted_view))

    def offset_checked(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Int]:
        # This packed descriptor is Copyable with public mutable fields.
        # Even a previously validated accessor can therefore be changed by
        # its caller. Re-prove the *complete* shape/stride/capacity contract
        # before using those fields at a checked element-access boundary.
        # `load_unchecked` remains a separate hot path requiring a proof.
        var canonical = TensorShape[Self.rank].checked(self.dims)
        if not canonical.is_ok():
            return TensorResult[Int](canonical.status)
        var verified = TensorLayout[Self.rank].checked(
            canonical.payload.value(), self.strides, self.offset,
            len(self.span),
        )
        if not verified.is_ok():
            return TensorResult[Int](verified.status)
        return verified.payload.value()._offset_from_validated(indices)

    def load(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Float32]:
        var position = self.offset_checked(indices)
        if not position.is_ok():
            return TensorResult[Float32](position.status)
        return TensorResult[Float32](self.span[position.payload.value()])

    def load_unchecked(self, indices: StaticRankArray[Self.rank]) -> Float32:
        # Used only after full shape/span/index proof at the kernel boundary.
        var position = self.offset
        comptime for i in range(Self.rank):
            position += indices.get(i) * self.strides.get(i)
        return self.span.unsafe_ptr().unsafe_load(position)[0]


def can_use_i32_indexing[rank: Int, origin: Origin](view: TensorView[rank, origin]) -> Bool:
    # Conservative 32-bit GPU index-space proof. A GPU kernel must still
    # construct a native device accessor; Span is a host borrow only.
    var checked = view._checked_current_layout()
    if not checked.is_ok():
        return False
    var layout = checked.payload.value()
    if layout.offset < 0 or layout.offset > 2147483647:
        return False
    if layout.span_lo < 0 or layout.span_hi > 2147483647:
        return False
    comptime for i in range(rank):
        if layout.shape.dims.get(i) > 2147483647:
            return False
        if layout.strides.get(i) < -2147483648 or layout.strides.get(i) > 2147483647:
            return False
    return True


# Functional aliases use the same checked host representation; they do not
# introduce a second GPU-accessor type or erase pointer origin/address space.
def pack_accessor[rank: Int, origin: Origin](
    view: TensorView[rank, origin],
) -> TensorResult[PackedHostAccessor[rank, origin]]:
    return PackedHostAccessor[rank, origin].checked(view)


def unpack_shape[rank: Int, origin: Origin](
    accessor: PackedHostAccessor[rank, origin],
) -> TensorResult[TensorShape[rank]]:
    # Packed metadata can be modified after construction, so expose its shape
    # only after rebuilding the original logical layout through a checked view.
    var source_shape = TensorShape[rank].checked(accessor.dims)
    if not source_shape.is_ok():
        return TensorResult[TensorShape[rank]](source_shape.status)
    var layout = TensorLayout[rank].checked(
        source_shape.payload.value(), accessor.strides,
        accessor.offset, len(accessor.span),
    )
    if not layout.is_ok():
        return TensorResult[TensorShape[rank]](layout.status)
    return TensorResult[TensorShape[rank]](source_shape.payload.value().copy())


@inline(.always)
def offset_unchecked[rank: Int, origin: Origin](
    accessor: PackedHostAccessor[rank, origin],
    indices: StaticRankArray[rank],
) -> Int:
    # Kernel-only: parent view/capacity and logical indices are already valid.
    var position = accessor.offset
    comptime for i in range(rank):
        position += indices.get(i) * accessor.strides.get(i)
    return position
