# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR INDEXING / SLICE SPECIFICATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Indexing
# Module:        momijo.tensor_light.indexing.slice_spec
# Source:        src/momijo/tensor_light/indexing/slice_spec.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines slice specifications in the tensor indexing layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `NewAxis`, `EllipsisIndex`, `newaxis`,
#   `ellipsis_index`; preconditions and supported targets are determined by
#   their implementations.
#
# Responsibilities:
#   - Normalize and validate index or slicing operations.
#   - Preserve documented behavior for bounds, strides, view provenance, and
#     write safety.
#   - Provide source-declared interfaces such as `NewAxis`, `EllipsisIndex`,
#     `newaxis`, `ellipsis_index`, `BasicIndexKind` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Normalize and bounds-check indices before memory access.
#   - Preserve documented negative-index and slicing semantics.
#   - Check index bounds before converting indices into physical offsets.
#
# Performance Objectives:
#   - Perform checked offset computation outside repeated element access.
#   - Avoid unnecessary index-tensor copies.
#   - Reuse normalized index ranges when the operation permits.
#
# Maintainability:
#   - Centralize slicing and index normalization.
#   - Test empty, negative-step, out-of-range, and high-rank cases.
#   - Test invalid indices, zero extents, and boundary conditions.
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

# Basic-index vocabulary. Each term is a trivially copied static-rank descriptor.
# The tagged terms can be placed in a fixed Array with no heap allocation.
from ..indexing.slice import SliceSpec
from ..internal.status import TensorResult, TensorStatus

@fieldwise_init
struct NewAxis(Copyable):
    var marker: Int

@fieldwise_init
struct EllipsisIndex(Copyable):
    var marker: Int

def newaxis() -> NewAxis:
    return NewAxis(0)

def ellipsis_index() -> EllipsisIndex:
    return EllipsisIndex(0)

@fieldwise_init
struct BasicIndexKind(Copyable, Equatable):
    var code: Int
    comptime INTEGER = Self(0)
    comptime SLICE = Self(1)
    comptime NEW_AXIS = Self(2)
    comptime ELLIPSIS = Self(3)

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

struct BasicIndex(Copyable):
    var kind: BasicIndexKind
    var integer_value: Int
    var slice_value: SliceSpec

    def __init__(out self, kind: BasicIndexKind, integer_value: Int, slice_value: SliceSpec):
        self.kind = kind.copy()
        self.integer_value = integer_value
        self.slice_value = slice_value.copy()

    def __init__(out self, *, copy: Self):
        self.kind = copy.kind.copy()
        self.integer_value = copy.integer_value
        self.slice_value = copy.slice_value.copy()

    @staticmethod
    def integer(value: Int) -> Self:
        return Self(materialize[BasicIndexKind.INTEGER](), value, SliceSpec(None, None, 1))

    @staticmethod
    def sliced(value: SliceSpec) -> Self:
        return Self(materialize[BasicIndexKind.SLICE](), 0, value)

    @staticmethod
    def full_slice() -> Self:
        return Self(materialize[BasicIndexKind.SLICE](), 0, SliceSpec(None, None, 1))

    @staticmethod
    def new_axis() -> Self:
        return Self(materialize[BasicIndexKind.NEW_AXIS](), 0, SliceSpec(None, None, 1))

    @staticmethod
    def ellipsis() -> Self:
        return Self(materialize[BasicIndexKind.ELLIPSIS](), 0, SliceSpec(None, None, 1))


# Public specification vocabulary: IndexSpec is the canonical checked
# runtime basic-index tagged value, not a second representation with separate
# copy/ownership and layout interpretation rules.
comptime IndexSpec = BasicIndex


# Exact overloads allow the VariadicPack caller to normalize heterogeneous
# runtime terms at compile time. Unsupported term types fail to resolve an
# overload; they cannot silently become an unchecked integer index.
def as_basic_index(value: Int) -> BasicIndex:
    return BasicIndex.integer(value)

def as_basic_index(value: SliceSpec) -> BasicIndex:
    return BasicIndex.sliced(value.copy())

def as_basic_index(value: NewAxis) -> BasicIndex:
    return BasicIndex.new_axis()

def as_basic_index(value: EllipsisIndex) -> BasicIndex:
    return BasicIndex.ellipsis()

def as_basic_index(value: BasicIndex) -> BasicIndex:
    return value.copy()

# Computes the static rank of a *typed* heterogeneous basic index. The
# integer terms remove one axis; NewAxis inserts one. SliceSpec and ellipsis
# preserve rank. BasicIndex, whose kind is a runtime tag, is intentionally
# excluded: use index_view[out_rank] for a dynamic-tagged index array.
def inferred_basic_index_rank[src_rank: Int, *IndexTypes: AnyType]() -> Int:
    var consumed = 0
    var result = src_rank
    var ellipses = 0
    comptime for i in range(IndexTypes.length):
        comptime if IndexTypes[i] == Int:
            consumed += 1
            result -= 1
        elif IndexTypes[i] == SliceSpec:
            consumed += 1
        elif IndexTypes[i] == NewAxis:
            result += 1
        elif IndexTypes[i] == EllipsisIndex:
            ellipses += 1
        else:
            # A compile-time failure keeps dynamic BasicIndex out of an
            # index signature whose return rank must be known statically.
            return -1
    if ellipses > 1 or consumed > src_rank:
        return -1
    if result < 0 or result > 8:
        return -1
    return result


# One canonical, fixed-size basic-index term validation for all entry points.
# Metadata-only: no storage, pointer arithmetic or dynamic index container.
@fieldwise_init
struct BasicIndexPackSummary(Copyable):
    var consumed_source_axes: Int
    var produced_output_axes: Int
    var ellipsis_count: Int


def _summarize_basic_index_pack[count: Int](
    terms: Array[BasicIndex, count],
) -> TensorResult[BasicIndexPackSummary]:
    comptime assert count >= 1 and count <= 17, "BasicIndex pack size must be 1..17"
    var consumed = 0
    var produced = 0
    var ellipses = 0
    comptime for i in range(count):
        var term = terms[i].copy()
        if term.kind == materialize[BasicIndexKind.INTEGER]():
            consumed += 1
        elif term.kind == materialize[BasicIndexKind.SLICE]():
            # Slice length/endpoint checks need the indexed extent, but a
            # zero/Int.MIN step is invalid independently of that extent.
            if term.slice_value.step == 0 or term.slice_value.step == Int.MIN:
                return TensorResult[BasicIndexPackSummary](TensorStatus.INVALID_LAYOUT)
            consumed += 1
            produced += 1
        elif term.kind == materialize[BasicIndexKind.NEW_AXIS]():
            produced += 1
        elif term.kind == materialize[BasicIndexKind.ELLIPSIS]():
            ellipses += 1
        else:
            return TensorResult[BasicIndexPackSummary](TensorStatus.UNSUPPORTED)
    if ellipses > 1:
        return TensorResult[BasicIndexPackSummary](TensorStatus.INVALID_AXIS)
    return TensorResult[BasicIndexPackSummary](
        BasicIndexPackSummary(consumed, produced, ellipses)
    )


def count_ellipsis[count: Int](
    terms: Array[BasicIndex, count],
) -> TensorResult[Int]:
    var result = _summarize_basic_index_pack(terms)
    if not result.is_ok():
        return TensorResult[Int](result.status)
    return TensorResult[Int](result.payload.value().ellipsis_count)


def count_consumed_axes[count: Int](
    terms: Array[BasicIndex, count],
) -> TensorResult[Int]:
    var result = _summarize_basic_index_pack(terms)
    if not result.is_ok():
        return TensorResult[Int](result.status)
    return TensorResult[Int](result.payload.value().consumed_source_axes)


def result_rank_of_basic_index_pack[src_rank: Int, count: Int](
    terms: Array[BasicIndex, count],
) -> TensorResult[Int]:
    comptime assert src_rank >= 0 and src_rank <= 8, "source rank must be 0..8"
    var inspected = _summarize_basic_index_pack(terms)
    if not inspected.is_ok():
        return TensorResult[Int](inspected.status)
    var metadata = inspected.payload.value().copy()
    if metadata.consumed_source_axes > src_rank:
        return TensorResult[Int](TensorStatus.INVALID_AXIS)
    # The ellipsis, when present, stands for exactly the unconsumed axes.
    # Without an ellipsis, the same trailing axes are implicitly full slices.
    var result_rank = metadata.produced_output_axes + (
        src_rank - metadata.consumed_source_axes
    )
    if result_rank < 0 or result_rank > 8:
        return TensorResult[Int](TensorStatus.INVALID_RANK)
    return TensorResult[Int](result_rank)


def validate_basic_index_pack[src_rank: Int, out_rank: Int, count: Int](
    terms: Array[BasicIndex, count],
) -> TensorStatus:
    comptime assert out_rank >= 0 and out_rank <= 8, "output rank must be 0..8"
    var expected = result_rank_of_basic_index_pack[src_rank, count](terms)
    if not expected.is_ok():
        return expected.status
    if expected.payload.value() != out_rank:
        return TensorStatus.INVALID_RANK
    return TensorStatus.OK


# The specification also requires a TYPED, compile-time VariadicPack API.
# These overloads deliberately do not construct runtime BasicIndex arrays:
# the returned rank/counts are decided while specializing the caller, and an
# unsupported component type is rejected by comptime assertion; invalid
# cardinality/rank combinations return -1 (or False for validation).
# Runtime-tagged BasicIndex arrays use the checked overloads above instead.
def validate_basic_index_pack[*IndexTypes: AnyType]() -> Bool:
    comptime for i in range(IndexTypes.length):
        comptime assert (
            IndexTypes[i] == Int
            or IndexTypes[i] == SliceSpec
            or IndexTypes[i] == NewAxis
            or IndexTypes[i] == EllipsisIndex
        ), "Only Int, SliceSpec, NewAxis and EllipsisIndex are valid typed indices"
    var ellipses = 0
    comptime for i in range(IndexTypes.length):
        comptime if IndexTypes[i] == EllipsisIndex:
            ellipses += 1
    if ellipses > 1:
        return False
    return True


def count_ellipsis[*IndexTypes: AnyType]() -> Int:
    var n = 0
    comptime for i in range(IndexTypes.length):
        comptime assert (
            IndexTypes[i] == Int
            or IndexTypes[i] == SliceSpec
            or IndexTypes[i] == NewAxis
            or IndexTypes[i] == EllipsisIndex
        ), "Unsupported typed basic index"
        comptime if IndexTypes[i] == EllipsisIndex:
            n += 1
    if n > 1:
        return -1
    return n


def count_consumed_axes[*IndexTypes: AnyType]() -> Int:
    var n = 0
    comptime for i in range(IndexTypes.length):
        comptime assert (
            IndexTypes[i] == Int
            or IndexTypes[i] == SliceSpec
            or IndexTypes[i] == NewAxis
            or IndexTypes[i] == EllipsisIndex
        ), "Unsupported typed basic index"
        comptime if IndexTypes[i] == Int or IndexTypes[i] == SliceSpec:
            n += 1
    return n


def result_rank_of_basic_index_pack[src_rank: Int, *IndexTypes: AnyType]() -> Int:
    comptime assert src_rank >= 0 and src_rank <= 8, "Tensor rank must be 0..8"
    var consumed = 0
    var retained_slices = 0
    var inserted = 0
    var ellipses = 0
    comptime for i in range(IndexTypes.length):
        comptime assert (
            IndexTypes[i] == Int
            or IndexTypes[i] == SliceSpec
            or IndexTypes[i] == NewAxis
            or IndexTypes[i] == EllipsisIndex
        ), "Unsupported typed basic index"
        comptime if IndexTypes[i] == Int:
            consumed += 1
        elif IndexTypes[i] == SliceSpec:
            consumed += 1
            retained_slices += 1
        elif IndexTypes[i] == NewAxis:
            inserted += 1
        elif IndexTypes[i] == EllipsisIndex:
            ellipses += 1
    if ellipses > 1 or consumed > src_rank:
        return -1
    # Slices consume an input axis but preserve one result axis. Only
    # integer indices lower rank; NewAxis raises it, ellipsis preserves it.
    var out_rank = src_rank - consumed + retained_slices + inserted
    if out_rank < 0 or out_rank > 8:
        return -1
    return out_rank
