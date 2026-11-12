# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR ACCESS / ACCESSOR
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Access and References
# Module:        momijo.tensor_light.access.accessor
# Source:        src/momijo/tensor_light/access/accessor.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines accessor in the access and references layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `TensorAccessor`, `TensorMutAccessor`,
#   `make_accessor`, `make_mut_accessor`; preconditions and supported targets
#   are determined by their implementations.
#
# Responsibilities:
#   - Provide checked tensor element access without exceeding storage bounds.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `TensorAccessor`,
#     `TensorMutAccessor`, `make_accessor`, `make_mut_accessor`,
#     `offset_unchecked` and related symbols.
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

# Rank-static checked FP32 accessors. Views own no memory; ownership remains
# with Tensor. All host dereferences retain the original borrow lifetime.

from ..view.view import TensorView, TensorMutView
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus
from std.collections import Span

struct TensorAccessor[rank: Int, origin: Origin](Copyable):
    var view: TensorView[Self.rank, Self.origin]

    def __init__(out self, view: TensorView[Self.rank, Self.origin]):
        self.view = view

    def __init__(out self, *, copy: Self):
        self.view = copy.view.copy()

    def load(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Float32]:
        return self.view.get(indices)

    def load_unchecked(self, indices: StaticRankArray[Self.rank]) -> Float32:
        # PRECONDITION: view layout, bounds and host lifetime were already
        # checked once at the API boundary; GPU uses native packed metadata.
        return self.view.data.unsafe_ptr().unsafe_load(self.view.layout.physical_offset_unchecked(indices))[0]


struct TensorMutAccessor[rank: Int, origin: Origin[mut=True]](Movable):
    var view: TensorMutView[Self.rank, Self.origin]

    def __init__(out self, var view: TensorMutView[Self.rank, Self.origin]):
        self.view = view

    def load(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Float32]:
        return self.view.get(indices)

    def store(mut self, indices: StaticRankArray[Self.rank], value: Float32) -> TensorStatus:
        return self.view.set(indices, value)

    def load_unchecked(self, indices: StaticRankArray[Self.rank]) -> Float32:
        # The mutable borrow and proven-unique mapping are part of the
        # caller's established fast-path proof, never inferred from dtype.
        return self.view.data.unsafe_ptr().unsafe_load(self.view.layout.physical_offset_unchecked(indices))[0]

    def store_unchecked(mut self, indices: StaticRankArray[Self.rank], value: Float32):
        # PRECONDITION: layout checked, destination unique/writable, and
        # index in bounds. Only an internal AOT kernel may use this method.
        self.view.data.unsafe_ptr().unsafe_store[alignment=4](self.view.layout.physical_offset_unchecked(indices), value)


def make_accessor[rank: Int, origin: Origin](view: TensorView[rank, origin]) -> TensorResult[TensorAccessor[rank, origin]]:
    var validated = view._checked_current_layout()
    if not validated.is_ok():
        return TensorResult[TensorAccessor[rank, origin]](validated.status)
    # The checked accessor carries exactly the layout snapshot just proved;
    # it must not preserve an independently mutable, stale source descriptor.
    # This changes neither the allocation owner nor the Span borrow origin.
    var canonical_view = TensorView[rank, origin](
        view.data, validated.payload.value(),
    )
    return TensorResult[TensorAccessor[rank, origin]](
        TensorAccessor[rank, origin](canonical_view),
    )


def make_mut_accessor[rank: Int, origin: Origin[mut=True]](
    var view: TensorMutView[rank, origin]
) -> TensorResult[TensorMutAccessor[rank, origin]]:
    # Consume the unique mutable view; never duplicate an exclusive borrow.
    var checked = view._checked_current_layout()
    if not checked.is_ok():
        return TensorResult[TensorMutAccessor[rank, origin]](checked.status)
    if not checked.payload.value().is_proven_unique():
        return TensorResult[TensorMutAccessor[rank, origin]](TensorStatus.OVERLAPPING)
    # Preserve the exclusive mutable Span, but replace caller-writable cached
    # layout fields with the same validated snapshot proven above. In
    # particular, unchecked kernel access must not inherit a stale span or
    # overlap flag from the pre-validated source descriptor.
    var canonical_view = TensorMutView[rank, origin](
        view.data, checked.payload.take(),
    )
    return TensorResult[TensorMutAccessor[rank, origin]](
        TensorMutAccessor[rank, origin](canonical_view),
    )


# Kernel-only AOT access primitives. These functions intentionally do not
# recheck bounds in the inner loop. The checked TensorView/Storage boundary
# must prove every vector lane and physical offset before calling them.

@inline(.always)
def offset_unchecked[rank: Int, origin: Origin](
    accessor: TensorAccessor[rank, origin], indices: StaticRankArray[rank]
) -> Int:
    return accessor.view.layout.physical_offset_unchecked(indices)


@inline(.always)
def load_scalar_unchecked[rank: Int, origin: Origin](
    accessor: TensorAccessor[rank, origin], indices: StaticRankArray[rank]
) -> Float32:
    return accessor.load_unchecked(indices)


@inline(.always)
def store_scalar_unchecked[rank: Int, origin: Origin[mut=True]](
    mut accessor: TensorMutAccessor[rank, origin],
    indices: StaticRankArray[rank], value: Float32,
):
    accessor.store_unchecked(indices, value)


@inline(.always)
def load_contiguous_unchecked[width: Int, origin: Origin](
    source: Span[False, Float32, origin], start: Int,
) -> SIMD[DType.float32, width]:
    # The caller proves [start, start + width) is in the live storage span.
    return source.unsafe_ptr().unsafe_load[width=width, alignment=4](start)


@inline(.always)
def store_contiguous_unchecked[width: Int, origin: Origin[mut=True]](
    destination: Span[True, Float32, origin], start: Int,
    value: SIMD[DType.float32, width],
):
    destination.unsafe_ptr().unsafe_store[alignment=4](start, value)


def masked_load_tail[width: Int, origin: Origin](
    source: Span[False, Float32, origin], start: Int,
    valid_lanes: Int,
) -> TensorResult[SIMD[DType.float32, width]]:
    # Public tail boundary is checked once; masked-off lanes are never read,
    # even on hardware that lacks native predicated vector memory access.
    if valid_lanes < 0 or valid_lanes > width or start < 0 or start > len(source):
        return TensorResult[SIMD[DType.float32, width]](TensorStatus.OUT_OF_BOUNDS)
    if valid_lanes > len(source) - start:
        return TensorResult[SIMD[DType.float32, width]](TensorStatus.OUT_OF_BOUNDS)
    var lanes = SIMD[DType.float32, width](0.0)
    for lane in range(valid_lanes):
        lanes[lane] = source[start + lane]
    return TensorResult[SIMD[DType.float32, width]](lanes)


def masked_store_tail[width: Int, origin: Origin[mut=True]](
    destination: Span[True, Float32, origin], start: Int,
    values: SIMD[DType.float32, width], valid_lanes: Int,
) -> TensorStatus:
    if valid_lanes < 0 or valid_lanes > width or start < 0 or start > len(destination):
        return TensorStatus.OUT_OF_BOUNDS
    if valid_lanes > len(destination) - start:
        return TensorStatus.OUT_OF_BOUNDS
    for lane in range(valid_lanes):
        destination[start + lane] = values[lane]
    return TensorStatus.OK


@inline(.always)
def load_strided_unchecked[width: Int, origin: Origin](
    source: Span[False, Float32, origin], start: Int, step: Int,
) -> SIMD[DType.float32, width]:
    # AOT irregular-stride fallback: shape/layout span proof is upstream.
    # Do not vector-load across holes or reversed physical storage.
    var result = SIMD[DType.float32, width](0.0)
    var ptr = source.unsafe_ptr()
    for lane in range(width):
        result[lane] = ptr.unsafe_load(start + lane * step)[0]
    return result


@inline(.always)
def store_strided_unchecked[width: Int, origin: Origin[mut=True]](
    destination: Span[True, Float32, origin], start: Int, step: Int,
    value: SIMD[DType.float32, width],
):
    # PRECONDITION: unique physical output offsets and all lanes in bounds.
    var ptr = destination.unsafe_ptr()
    for lane in range(width):
        ptr.unsafe_store[alignment=4](start + lane * step, value[lane])


@inline(.always)
def gather_unchecked[width: Int, origin: Origin](
    source: Span[False, Float32, origin],
    offsets: SIMD[DType.int64, width],
) -> SIMD[DType.float32, width]:
    var result = SIMD[DType.float32, width](0.0)
    var ptr = source.unsafe_ptr()
    for lane in range(width):
        result[lane] = ptr.unsafe_load(Int(offsets[lane]))[0]
    return result


@inline(.always)
def scatter_unchecked[width: Int, origin: Origin[mut=True]](
    destination: Span[True, Float32, origin],
    offsets: SIMD[DType.int64, width],
    values: SIMD[DType.float32, width],
):
    # PRECONDITION: validated conflict policy, no two active lanes target the
    # same physical address, and no concurrent writer owns the same region.
    var ptr = destination.unsafe_ptr()
    for lane in range(width):
        ptr.unsafe_store[alignment=4](Int(offsets[lane]), values[lane])
