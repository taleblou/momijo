# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR PARALLEL / WORK PARTITIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / CPU Parallel Work
# Module:        momijo.tensor_light.parallel.partition
# Source:        src/momijo/tensor_light/parallel/partition.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines work partitions in the CPU parallel work layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `WorkRange`, `checked_range`, `split_at`,
#   `checked_range_count`; preconditions and supported targets are determined
#   by their implementations.
#
# Responsibilities:
#   - Partition CPU work and validate independent ranges.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `WorkRange`, `checked_range`,
#     `split_at`, `checked_range_count`, `checked_ranges_disjoint` and related
#     symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Partition work into valid, disjoint ranges.
#   - Preserve deterministic reductions when required by the API.
#
# Performance Objectives:
#   - Limit scheduling overhead for small workloads.
#   - Align partitions with profitable memory and vector boundaries.
#
# Maintainability:
#   - Keep work distribution separate from kernel semantics.
#   - Test irregular tails and empty partitions.
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

# Disjoint half-open work regions are required for concurrent CPU/GPU writes.
from ..internal.status import TensorStatus, TensorResult
from ..shape.shape import TensorShape
from ..shape.axis import normalize_axis
from ..layout.layout import TensorLayout

@fieldwise_init
struct WorkRange(Copyable):
    var begin: Int
    var end: Int

    def count(self) -> Int:
        # Raw WorkRange is public mutable metadata. An invalid half-open
        # interval is not a legal scheduler grain; return a negative sentinel.
        if self.begin < 0 or self.end < self.begin:
            return -1
        return self.end - self.begin

    def disjoint(self, other: Self) -> Bool:
        # Conservative even if called without checked_ranges_disjoint().
        # A forged empty/reversed region must never prove write independence.
        if self.begin < 0 or other.begin < 0:
            return False
        if self.end < self.begin or other.end < other.begin:
            return False
        # Valid empty half-open regions never contain a writable element,
        # even when their zero-length endpoint lies inside another region.
        if self.end == self.begin or other.end == other.begin:
            return True
        return self.end <= other.begin or other.end <= self.begin

def checked_range(begin: Int, end: Int, domain_size: Int) -> TensorResult[WorkRange]:
    if domain_size < 0 or begin < 0 or end < begin or end > domain_size:
        return TensorResult[WorkRange](TensorStatus.OUT_OF_BOUNDS)
    return TensorResult[WorkRange](WorkRange(begin, end))

def split_at(count: Int, boundary: Int) -> TensorResult[Tuple[WorkRange, WorkRange]]:
    if count < 0 or boundary < 0 or boundary > count:
        return TensorResult[Tuple[WorkRange, WorkRange]](TensorStatus.OUT_OF_BOUNDS)
    return TensorResult[Tuple[WorkRange, WorkRange]]((WorkRange(0, boundary), WorkRange(boundary, count)))


def checked_range_count(span: WorkRange, domain_size: Int) -> TensorResult[Int]:
    # WorkRange is mutable metadata. Do not consume an unchecked range as
    # a trusted scheduler grain, especially in a hybrid write partition.
    var proof = checked_range(span.begin, span.end, domain_size)
    if not proof.is_ok():
        return TensorResult[Int](proof.status)
    return TensorResult[Int](span.end - span.begin)


def checked_ranges_disjoint(
    a: WorkRange, b: WorkRange, domain_size: Int,
) -> TensorResult[Bool]:
    # Reject forged negative/reversed ranges BEFORE proving write disjointness.
    var a_proof = checked_range(a.begin, a.end, domain_size)
    if not a_proof.is_ok():
        return TensorResult[Bool](a_proof.status)
    var b_proof = checked_range(b.begin, b.end, domain_size)
    if not b_proof.is_ok():
        return TensorResult[Bool](b_proof.status)
    return TensorResult[Bool](a_proof.payload.value().disjoint(b_proof.payload.value()))


def split_range_aligned(
    begin: Int, end: Int, requested_split: Int, alignment_elems: Int,
) -> TensorResult[Tuple[WorkRange, WorkRange]]:
    # Return non-overlapping, full-coverage segments of [begin,end). The
    # boundary aligns relative to the first element, preserving the source
    # domain; the scheduler can choose whether an empty shard is useful.
    if begin < 0 or end < begin or requested_split < begin or requested_split > end:
        return TensorResult[Tuple[WorkRange, WorkRange]](TensorStatus.OUT_OF_BOUNDS)
    if alignment_elems <= 0 or alignment_elems & (alignment_elems - 1) != 0:
        return TensorResult[Tuple[WorkRange, WorkRange]](TensorStatus.INVALID_ALIGNMENT)
    var relative = requested_split - begin
    var aligned = requested_split - relative % alignment_elems
    return TensorResult[Tuple[WorkRange, WorkRange]]((
        WorkRange(begin, aligned), WorkRange(aligned, end),
    ))


def checked_partition_cover[parts: Int](
    ranges: Array[WorkRange, parts], begin: Int, end: Int,
) -> TensorStatus:
    # A caller-supplied static partition receives a full-coverage proof
    # before concurrent writes. Contiguous ordering proves both no gaps
    # and no intersections without allocating any runtime metadata.
    comptime assert parts > 0, "static partition count must be positive"
    if begin < 0 or end < begin:
        return TensorStatus.OUT_OF_BOUNDS
    var cursor = begin
    for i in range(parts):
        var part = ranges[i].copy()
        if part.begin != cursor or part.end < part.begin or part.end > end:
            return TensorStatus.HYBRID_PARTITION_INVALID
        cursor = part.end
    if cursor != end:
        return TensorStatus.HYBRID_PARTITION_INVALID
    return TensorStatus.OK


def split_range_parts[parts: Int](
    begin: Int, end: Int,
) -> TensorResult[Array[WorkRange, parts]]:
    # AOT-static number of partitions and fixed Array metadata: no heap
    # per dispatch, no round-up multiplication and no uncovered tail.
    comptime assert parts > 0, "static partition count must be positive"
    if begin < 0 or end < begin:
        return TensorResult[Array[WorkRange, parts]](TensorStatus.OUT_OF_BOUNDS)
    var total = end - begin
    var size = total // parts
    var extra = total % parts
    var cursor = begin
    var result = Array[WorkRange, parts](fill=WorkRange(begin, begin))
    for i in range(parts):
        var take = size
        if i < extra:
            take += 1
        var boundary = cursor + take
        result[i] = WorkRange(cursor, boundary)
        cursor = boundary
    return TensorResult[Array[WorkRange, parts]](result^)


# A fixed number of chunks keeps all scheduling metadata on the stack. The
# public alias retains the source specification's split_range vocabulary.
def split_range[parts: Int](
    begin: Int, end: Int,
) -> TensorResult[Array[WorkRange, parts]]:
    return split_range_parts[parts](begin, end)


# Fetch one grain-sized shard without building a runtime List of ranges.
# The number of shards is ceil((end - begin) / grain); multiplication is only
# performed after proving part_index is within the total domain.
def split_range_grain(
    begin: Int, end: Int, grain: Int, part_index: Int,
) -> TensorResult[WorkRange]:
    if begin < 0 or end < begin or grain <= 0 or part_index < 0:
        return TensorResult[WorkRange](TensorStatus.OUT_OF_BOUNDS)
    var count = end - begin
    var pieces = count // grain
    if count % grain != 0:
        pieces += 1
    if part_index >= pieces:
        return TensorResult[WorkRange](TensorStatus.OUT_OF_BOUNDS)
    var offset = part_index * grain
    var remaining = count - offset
    var take = min(remaining, grain)
    return TensorResult[WorkRange](WorkRange(begin + offset, begin + offset + take))


# ND work partitioning returns ranges on one compile-time-rank axis rather
# than a heap-allocated list of dynamic-rank tile descriptors. Rank 0 has no
# partitionable axis; its one scalar is assigned to a single scalar kernel.
def split_nd[rank: Int, parts: Int](
    shape: TensorShape[rank], axis: Int,
) -> TensorResult[Array[WorkRange, parts]]:
    comptime assert rank >= 1 and rank <= 8, "ND partitions require rank 1..8"
    comptime assert parts > 0, "parts must be positive"
    var canonical = TensorShape[rank].checked(shape.dims)
    if not canonical.is_ok():
        return TensorResult[Array[WorkRange, parts]](canonical.status)
    if canonical.payload.value().numel() != shape.numel():
        return TensorResult[Array[WorkRange, parts]](TensorStatus.INVALID_SHAPE)
    var normalized = normalize_axis[rank](axis)
    if not normalized.is_ok():
        return TensorResult[Array[WorkRange, parts]](normalized.status)
    var extent = shape.dims.get(normalized.payload.value())
    return split_range_parts[parts](0, extent)


@fieldwise_init
struct HybridPartition(Copyable):
    var cpu: WorkRange
    var gpu: WorkRange


def make_hybrid_partition(
    total_items: Int, gpu_items: Int, gpu_first: Bool,
) -> TensorResult[HybridPartition]:
    if total_items < 0 or gpu_items <= 0 or gpu_items >= total_items:
        return TensorResult[HybridPartition](TensorStatus.HYBRID_PARTITION_INVALID)
    var cpu_items = total_items - gpu_items
    if gpu_first:
        return TensorResult[HybridPartition](HybridPartition(
            WorkRange(gpu_items, total_items), WorkRange(0, gpu_items),
        ))
    return TensorResult[HybridPartition](HybridPartition(
        WorkRange(0, cpu_items), WorkRange(cpu_items, total_items),
    ))


# An input reduction may be split across devices only when each physical
# partition is read by one lane. The caller combines partials according to
# its ReductionAccuracyMode; this routine does no cross-device transfer.
def partition_reduction_input(
    total_items: Int, gpu_items: Int,
) -> TensorResult[HybridPartition]:
    return make_hybrid_partition(total_items, gpu_items, False)


# Checked AOT parallel-axis selection for unique Float32 strided layouts.
# The metadata is a scheduling hint, NEVER evidence of physical noalias:
# the caller separately proves each output shard writes a disjoint region.
# Favor an outer/large-stride axis so a unit-stride inner axis remains local
# to a worker for contiguous SIMD and GPU-coalesced access. Rank zero has no
# axes; singleton/empty layouts have no profitable parallel split.
def choose_partition_axis[rank: Int](
    layout: TensorLayout[rank], capacity: Int,
) -> TensorResult[Int] where rank >= 0 and rank <= 8:
    comptime if rank == 0:
        return TensorResult[Int](TensorStatus.INVALID_AXIS)
    var checked = layout.validated_snapshot(capacity)
    if not checked.is_ok():
        return TensorResult[Int](checked.status)
    var trusted = checked.payload.value()
    if trusted.shape.is_empty():
        return TensorResult[Int](TensorStatus.SIZE_MISMATCH)
    # Any overlapping output mapping is illegal for independent writers.
    if not trusted.is_proven_unique():
        return TensorResult[Int](TensorStatus.OVERLAPPING)
    var selected = -1
    var largest_stride = -1
    var largest_extent = 0
    comptime for axis in range(rank):
        var extent = trusted.shape.dims.get(axis)
        if extent > 1:
            var stride = trusted.strides.get(axis)
            if stride == Int.MIN:
                return TensorResult[Int](TensorStatus.OVERFLOW)
            if stride < 0:
                stride = -stride
            if stride > largest_stride or (stride == largest_stride and extent > largest_extent):
                selected = axis
                largest_stride = stride
                largest_extent = extent
    if selected < 0:
        return TensorResult[Int](TensorStatus.SIZE_MISMATCH)
    return TensorResult[Int](selected)
