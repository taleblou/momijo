# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR SHAPE / SHAPE METADATA
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Shapes
# Module:        momijo.tensor_light.shape.shape
# Source:        src/momijo/tensor_light/shape/shape.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines shape metadata in the tensor shapes layer of the FP32 Light tensor
#   implementation.
#
#   Public declarations include `TensorShape`, `TrailingReductionGeometry`,
#   `checked_trailing_reduction_geometry`, `validate_extent`; preconditions
#   and supported targets are determined by their implementations.
#
# Responsibilities:
#   - Construct, normalize, or validate tensor shape metadata.
#   - Preserve documented behavior for shape compatibility, rank boundaries,
#     and empty dimensions.
#   - Provide source-declared interfaces such as `TensorShape`,
#     `TrailingReductionGeometry`, `checked_trailing_reduction_geometry`,
#     `validate_extent`, `is_zero_extent` and related symbols.
#   - Handle scalar, empty, and rank-dependent edge cases explicitly.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate extents, axes, rank, and element-count overflow.
#   - Preserve rank-zero and zero-sized-shape semantics.
#   - Handle scalar, rank-zero, and zero-length dimensions without silent
#     mismatch.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Keep shape arithmetic in metadata-only code.
#   - Avoid allocating shape arrays unnecessarily in hot paths.
#   - Reuse validated output-shape metadata before data traversal.
#
# Maintainability:
#   - Share one broadcasting and axis normalization policy.
#   - Test invalid ranks, mismatched shapes, and inferred dimensions.
#   - Exercise incompatible dimensions and rank-limit regressions.
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

# Rank is static, extents are runtime; scalar numel is exactly one.
from ..internal.status import TensorResult, TensorStatus
from ..internal.checked_math import checked_mul_nonnegative, checked_nbytes
from .shape_storage import StaticRankArray
from .axis import normalize_axis_pair

struct TensorShape[rank: Int](Copyable):
    var dims: StaticRankArray[Self.rank]
    var elements: Int

    def __init__(out self, dims: StaticRankArray[Self.rank], elements: Int):
        self.dims = dims
        self.elements = elements

    def __init__(out self, *, copy: Self):
        self.dims = copy.dims.copy()
        self.elements = copy.elements

    @staticmethod
    def checked(dims: StaticRankArray[Self.rank]) -> TensorResult[Self]:
        comptime assert Self.rank >= 0 and Self.rank <= 8, "Self.rank must be 0..8"
        # Build new canonical metadata containing ONLY logical dimensions.
        # In particular, a forged rank-0 physical padding slot is discarded
        # rather than being accidentally copied into a validated descriptor.
        # This is metadata-only and allocates no heap storage (rank <= 8).
        var canonical_dims = StaticRankArray[Self.rank](0)
        comptime for axis in range(Self.rank):
            canonical_dims.set(axis, dims.get(axis))
        var empty = False
        comptime for i in range(Self.rank):
            if canonical_dims.get(i) < 0:
                return TensorResult[Self](TensorStatus.INVALID_SHAPE)
            if canonical_dims.get(i) == 0:
                empty = True
        if empty:
            return TensorResult[Self](Self(canonical_dims, 0))
        var n = 1
        comptime for i in range(Self.rank):
            var result = checked_mul_nonnegative(n, canonical_dims.get(i))
            if not result.is_ok():
                return TensorResult[Self](result.status)
            n = result.payload.value()
        if n > Int.MAX // 4:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        return TensorResult[Self](Self(canonical_dims, n))

    def numel(self) -> Int:
        return self.elements

    def nbytes(self) -> Int:
        # Public metadata can be constructed or mutated with an invalid cached
        # element count. Never let an overflowing count become a plausible
        # allocation size. Use checked_nbytes() to receive the exact error;
        # the legacy Int-returning accessor reports invalid metadata as -1.
        var proven = self.checked_nbytes()
        if not proven.is_ok():
            return -1
        return proven.payload.value()

    def checked_numel(self) -> TensorResult[Int]:
        var canonical = Self.checked(self.dims)
        if not canonical.is_ok():
            return TensorResult[Int](canonical.status)
        var actual = canonical.payload.value().numel()
        if actual != self.elements:
            return TensorResult[Int](TensorStatus.INVALID_SHAPE)
        return TensorResult[Int](actual)

    def checked_nbytes(self) -> TensorResult[Int]:
        var count = self.checked_numel()
        if not count.is_ok():
            return TensorResult[Int](count.status)
        # Self.checked() has already proved actual <= Int.MAX // 4.
        return TensorResult[Int](count.payload.value() * 4)

    def is_empty(self) -> Bool:
        # An empty-result fast path may bypass every payload access. It must
        # never rely on a forged zero extent with a stale cached numel.
        # The checked constructor is metadata-only (rank <= 8) and detects
        # overflow, negative extents and inconsistent cached element counts.
        var count = self.checked_numel()
        if not count.is_ok():
            return False
        # Rank zero has one logical scalar; no empty rank-zero tensor exists.
        return Self.rank > 0 and count.payload.value() == 0

    def is_scalar(self) -> Bool:
        return Self.rank == 0

    def dim(self, axis: Int) -> TensorResult[Int]:
        # The public TensorShape is a mutable value. An axis query is checked
        # only if it rejects invalid extents and a forged cached numel too;
        # internal hot code reads the already-validated StaticRankArray.
        var canonical = Self.checked(self.dims)
        if not canonical.is_ok():
            return TensorResult[Int](canonical.status)
        if canonical.payload.value().numel() != self.elements:
            return TensorResult[Int](TensorStatus.INVALID_SHAPE)
        var a = axis
        if a < 0:
            if a < -Self.rank:
                return TensorResult[Int](TensorStatus.INVALID_AXIS)
            a += Self.rank
        if a >= Self.rank:
            return TensorResult[Int](TensorStatus.INVALID_AXIS)
        return TensorResult[Int](canonical.payload.value().dims.get(a))

    def same_extents(self, other: Self) -> Bool:
        # Equality is public and is used to authorize copy, mask and SIMD
        # fast paths. A pair of malformed shapes with identical dimensions
        # but forged cached element counts is NOT a compatible shape proof.
        # Perform this once at the operation boundary, never per element.
        var left = self.checked_numel()
        if not left.is_ok():
            return False
        var right = other.checked_numel()
        if not right.is_ok():
            return False
        if left.payload.value() != right.payload.value():
            return False
        comptime for i in range(Self.rank):
            if self.dims.get(i) != other.dims.get(i):
                return False
        return True



# Contiguous trailing-axis reductions share a single, rank-static geometry.
# A [d0, ..., d{rank-axes-1}, k0, ..., k{axes-1}] C-order Tensor has
# `rows` independent output elements, each reducing `columns` adjacent
# Float32 elements. Neither a second Tensor owner nor a reshaped allocation
# is created. This mapping is applicable to CPU and optional GPU/Hybrid
# lowerings of the SAME primitive operation.
@fieldwise_init
struct TrailingReductionGeometry(Copyable):
    var rows: Int
    var columns: Int


def checked_trailing_reduction_geometry[rank: Int, axes: Int](
    shape: TensorShape[rank],
) -> TensorResult[TrailingReductionGeometry]:
    comptime assert rank >= 1 and rank <= 8, "trailing reduction rank must be 1..8"
    comptime assert axes >= 1 and axes <= rank, "trailing reduction axis count out of range"
    var verified = shape.checked_numel()
    if not verified.is_ok():
        return TensorResult[TrailingReductionGeometry](verified.status)
    # An empty reduction produces identities/errors through the ordinary
    # CPU/GPU Tensor reduction API. The concurrent split path requires a
    # positive row length and at least two independently writable rows.
    if verified.payload.value() == 0:
        return TensorResult[TrailingReductionGeometry](TensorStatus.NOT_REPRESENTABLE)
    var rows = 1
    var columns = 1
    comptime for i in range(rank):
        var dimension = shape.dims.get(i)
        comptime if i < rank - axes:
            var next_rows = checked_mul_nonnegative(rows, dimension)
            if not next_rows.is_ok():
                return TensorResult[TrailingReductionGeometry](next_rows.status)
            rows = next_rows.payload.value()
        else:
            var next_columns = checked_mul_nonnegative(columns, dimension)
            if not next_columns.is_ok():
                return TensorResult[TrailingReductionGeometry](next_columns.status)
            columns = next_columns.payload.value()
    if rows <= 0 or columns <= 0 or rows > Int.MAX // columns:
        return TensorResult[TrailingReductionGeometry](TensorStatus.OVERFLOW)
    if rows * columns != verified.payload.value():
        return TensorResult[TrailingReductionGeometry](TensorStatus.INVALID_SHAPE)
    return TensorResult[TrailingReductionGeometry](
        TrailingReductionGeometry(rows, columns)
    )

# Explicit checked shape utilities. No secondary tensor ownership exists.
def validate_extent(extent: Int) -> TensorStatus:
    if extent < 0:
        return TensorStatus.INVALID_SHAPE
    return TensorStatus.OK


def is_zero_extent(extent: Int) -> Bool:
    return extent == 0


def is_singleton_extent(extent: Int) -> Bool:
    return extent == 1


def checked_extent_product(a: Int, b: Int) -> TensorResult[Int]:
    return checked_mul_nonnegative(a, b)


def try_make_shape[rank: Int](dims: StaticRankArray[rank]) -> TensorResult[TensorShape[rank]]:
    return TensorShape[rank].checked(dims)


def scalar_shape() -> TensorResult[TensorShape[0]]:
    return TensorShape[0].checked(StaticRankArray[0](0))


def same_shape[rank: Int](a: TensorShape[rank], b: TensorShape[rank]) -> Bool:
    var checked_a = TensorShape[rank].checked(a.dims)
    if not checked_a.is_ok() or checked_a.payload.value().numel() != a.numel():
        return False
    var checked_b = TensorShape[rank].checked(b.dims)
    if not checked_b.is_ok() or checked_b.payload.value().numel() != b.numel():
        return False
    return checked_a.payload.value().same_extents(checked_b.payload.value())

# Shape-only operations never allocate payload storage or change the dtype.
# The original owner validates its TensorLayout/Storage separately.
def dim_unchecked[rank: Int](shape: TensorShape[rank], axis: Int) -> Int:
    # PRECONDITION: 0 <= axis < rank and shape has a checked construction.
    return shape.dims.get(axis)


def permute_shape[rank: Int](shape: TensorShape[rank], axes: StaticRankArray[rank]) -> TensorResult[TensorShape[rank]]:
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok():
        return canonical
    if canonical.payload.value().numel() != shape.numel():
        return TensorResult[TensorShape[rank]](TensorStatus.INVALID_SHAPE)
    var used = Array[Bool, 8](fill=False)
    var dims = StaticRankArray[rank](0)
    comptime for i in range(rank):
        var axis = axes.get(i)
        if axis < 0 or axis >= rank:
            return TensorResult[TensorShape[rank]](TensorStatus.INVALID_AXIS)
        if used[axis]:
            return TensorResult[TensorShape[rank]](TensorStatus.INVALID_AXIS)
        used[axis] = True
        dims.set(i, shape.dims.get(axis))
    return TensorShape[rank].checked(dims)


def transpose_shape[rank: Int](shape: TensorShape[rank], first: Int, second: Int) -> TensorResult[TensorShape[rank]]:
    var axes = normalize_axis_pair[rank](first, second)
    if not axes.is_ok():
        return TensorResult[TensorShape[rank]](axes.status)
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok():
        return canonical
    if canonical.payload.value().numel() != shape.numel():
        return TensorResult[TensorShape[rank]](TensorStatus.INVALID_SHAPE)
    var dims = shape.dims.copy()
    var a = axes.payload.value().get(0)
    var b = axes.payload.value().get(1)
    var hold = dims.get(a)
    dims.set(a, dims.get(b))
    dims.set(b, hold)
    return TensorShape[rank].checked(dims)


def unsqueeze_shape[rank: Int, axis: Int](shape: TensorShape[rank]) -> TensorResult[TensorShape[rank + 1]]:
    comptime assert rank >= 0 and rank < 8, "unsqueeze requires rank 0..7"
    comptime assert axis >= 0 and axis <= rank, "inserted axis out of bounds"
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok():
        return TensorResult[TensorShape[rank + 1]](canonical.status)
    if canonical.payload.value().numel() != shape.numel():
        return TensorResult[TensorShape[rank + 1]](TensorStatus.INVALID_SHAPE)
    var dims = StaticRankArray[rank + 1](1)
    comptime for i in range(rank + 1):
        comptime if i < axis:
            dims.set(i, shape.dims.get(i))
        elif i > axis:
            dims.set(i, shape.dims.get(i - 1))
    return TensorShape[rank + 1].checked(dims)


def squeeze_axis_shape[rank: Int, axis: Int](shape: TensorShape[rank]) -> TensorResult[TensorShape[rank - 1]]:
    comptime assert rank >= 1 and rank <= 8, "squeeze requires rank 1..8"
    comptime assert axis >= 0 and axis < rank, "squeeze axis out of bounds"
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok():
        return TensorResult[TensorShape[rank - 1]](canonical.status)
    if canonical.payload.value().numel() != shape.numel():
        return TensorResult[TensorShape[rank - 1]](TensorStatus.INVALID_SHAPE)
    if shape.dims.get(axis) != 1:
        return TensorResult[TensorShape[rank - 1]](TensorStatus.SIZE_MISMATCH)
    var dims = StaticRankArray[rank - 1](0)
    comptime for i in range(rank - 1):
        comptime if i < axis:
            dims.set(i, shape.dims.get(i))
        else:
            dims.set(i, shape.dims.get(i + 1))
    return TensorShape[rank - 1].checked(dims)


def flatten_shape[rank: Int, start_dim: Int, end_dim: Int](shape: TensorShape[rank]) -> TensorResult[TensorShape[rank - (end_dim - start_dim)]]:
    comptime assert rank >= 1 and rank <= 8, "flatten requires rank 1..8"
    comptime assert start_dim >= 0 and start_dim <= end_dim and end_dim < rank, "invalid flatten axes"
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok():
        return TensorResult[TensorShape[rank - (end_dim - start_dim)]](canonical.status)
    if canonical.payload.value().numel() != shape.numel():
        return TensorResult[TensorShape[rank - (end_dim - start_dim)]](TensorStatus.INVALID_SHAPE)
    var has_zero = False
    comptime for i in range(start_dim, end_dim + 1):
        if shape.dims.get(i) == 0:
            has_zero = True
    var extent = 0
    if not has_zero:
        extent = 1
        comptime for i in range(start_dim, end_dim + 1):
            var product = checked_mul_nonnegative(extent, shape.dims.get(i))
            if not product.is_ok():
                return TensorResult[TensorShape[rank - (end_dim - start_dim)]](product.status)
            extent = product.payload.value()
    var dims = StaticRankArray[rank - (end_dim - start_dim)](0)
    comptime for i in range(rank - (end_dim - start_dim)):
        comptime if i < start_dim:
            dims.set(i, shape.dims.get(i))
        elif i == start_dim:
            dims.set(i, extent)
        else:
            dims.set(i, shape.dims.get(i + (end_dim - start_dim)))
    return TensorShape[rank - (end_dim - start_dim)].checked(dims)
