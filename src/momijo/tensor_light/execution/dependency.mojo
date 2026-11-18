# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR EXECUTION / EXECUTION DEPENDENCIES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Execution Planning and AOT
# Module:        momijo.tensor_light.execution.dependency
# Source:        src/momijo/tensor_light/execution/dependency.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines execution dependencies within checked AOT execution planning for
#   the FP32 Light implementation.
#
#   Public declarations include `ExecutionDependency`, `ExecutionStageKind`,
#   `ExecutionAccess`, `intervals_intersect`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Define dependency or memory-access metadata and check legal ordering,
#     conflicts, or bounds.
#   - Expose module-level declarations including `ExecutionDependency`,
#     `ExecutionStageKind`, `ExecutionAccess`, `intervals_intersect`.
#   - Preserve shape, dtype, target, provenance, and capacity constraints
#     where required by the declarations.
#   - Keep planning and validation separate from proof of successful kernel
#     execution.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate the declared policy, dependency, shape, capacity, and target
#     constraints before dispatch.
#   - Do not treat a selected, prepared, or sealed plan as evidence of a
#     completed native execution.
#
# Performance Objectives:
#   - Keep plan construction, policy checks, and metadata queries inexpensive.
#   - Avoid repeated target selection and unnecessary allocations in
#     dispatch-critical paths.
#
# Maintainability:
#   - Keep metadata schemas synchronized with the binding and execution
#     consumers.
#   - Test invalid plans, stale bindings, unsupported targets, and policy
#     boundary cases.
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

# Static AOT dataflow schedule with allocation-aware physical access records.
# Every active access has an identity assigned at the checked ownership boundary.
# Two IDs may be distinct only when storage lifetimes/provenance prove this;
# untracked external aliasing must be resolved or snapshotted before execution.
# No runtime graph scheduler, callbacks, JIT compilation or global registry.
from ..internal.status import TensorStatus, TensorResult

@fieldwise_init
struct ExecutionDependency(Copyable):
    var kind: Int
    var stream_id: Int
    var event_id: Int

    comptime IMMEDIATE = 0
    comptime HOST_COMPLETE = 1
    comptime DEVICE_EVENT = 2
    comptime JOINED = 3

    def is_immediate(self) -> Bool:
        return self.kind == Self.IMMEDIATE


@fieldwise_init
struct ExecutionStageKind(Copyable, Equatable):
    var code: Int
    comptime CPU_PREPARE = Self(0)
    comptime H2D = Self(1)
    comptime GPU_KERNEL = Self(2)
    comptime D2H = Self(3)
    comptime CPU_COMPUTE = Self(4)
    comptime HOST_JOIN = Self(5)

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code


@fieldwise_init
struct ExecutionAccess(Copyable):
    # space: 0=host, 1=device, 2=staging (pinned or CPU-local); -1=no access.
    # buffer_id: an allocation identity, NOT a stage/stream/domain identifier.
    var space: Int
    var buffer_id: Int
    var begin: Int
    var end: Int

    @staticmethod
    def none() -> Self:
        return Self(-1, -1, 0, 0)

    def active(self) -> Bool:
        return self.space >= 0


def intervals_intersect(a_begin: Int, a_end: Int, b_begin: Int, b_end: Int) -> Bool:
    # This helper can be called with untrusted mutable access records. If an
    # interval is negative or reversed, conservatively report a conflict:
    # malformed metadata must not authorize unsynchronized memory access.
    if a_begin < 0 or b_begin < 0:
        return True
    if a_end < a_begin or b_end < b_begin:
        return True
    if a_begin == a_end or b_begin == b_end:
        return False
    return a_begin < b_end and b_begin < a_end


def valid_access(access: ExecutionAccess, extent: Int) -> Bool:
    if not access.active():
        return access.space == -1 and access.buffer_id == -1 and access.begin == 0 and access.end == 0
    if access.space > 2 or access.buffer_id < 0:
        return False
    # Host offsets are logical tensor positions and must fit total_elements.
    # Staging/device offsets are PHYSICAL allocation-local offsets; those
    # may be larger than the total logical work domain (e.g. padded staging
    # buffers). bind_schedule_buffers_checked() is the mandatory physical
    # capacity proof for those memory spaces before any launch.
    if access.begin < 0 or access.end <= access.begin:
        return False
    # Every physical interval is in Float32 element units. No later kernel
    # may turn an unrepresentable element offset into a wrapped byte offset.
    if access.end > Int.MAX // 4:
        return False
    if access.space == 0 and access.end > extent:
        return False
    return True


def access_overlaps(a: ExecutionAccess, b: ExecutionAccess) -> Bool:
    # Evaluate BOTH untrusted access descriptions before treating either
    # canonical `none` as a proof that two stages cannot race. In particular,
    # a valid inactive a with malformed active b must NOT return False early.
    var a_inactive = not a.active()
    var b_inactive = not b.active()
    if a_inactive:
        if a.space != -1 or a.buffer_id != -1 or a.begin != 0 or a.end != 0:
            return True
    else:
        if a.space > 2 or a.buffer_id < 0:
            return True
        if a.begin < 0 or a.end <= a.begin or a.end > Int.MAX // 4:
            return True
    if b_inactive:
        if b.space != -1 or b.buffer_id != -1 or b.begin != 0 or b.end != 0:
            return True
    else:
        if b.space > 2 or b.buffer_id < 0:
            return True
        if b.begin < 0 or b.end <= b.begin or b.end > Int.MAX // 4:
            return True
    if a_inactive or b_inactive:
        return False
    if a.space != b.space or a.buffer_id != b.buffer_id:
        # This result is only usable after owner-backed allocation/domain
        # identities have been bound and proven by the schedule binder.
        return False
    return intervals_intersect(a.begin, a.end, b.begin, b.end)


@fieldwise_init
struct ExecutionStage(Copyable):
    var identifier: Int
    var kind: ExecutionStageKind
    # Bit j means stage j must complete before this stage can begin.
    var depends_on: Int
    # Two independently identified reads are needed for FP32 binary kernels.
    var read_a: ExecutionAccess
    var read_b: ExecutionAccess
    var write: ExecutionAccess
    # 0=host compute, 1=device compute, 2=transfer, 3=host join.
    var lane: Int


def stage_access_conflict(current: ExecutionStage, prior: ExecutionStage) -> Bool:
    if access_overlaps(current.write, prior.write):
        return True
    if access_overlaps(current.write, prior.read_a):
        return True
    if access_overlaps(current.write, prior.read_b):
        return True
    if access_overlaps(current.read_a, prior.write):
        return True
    return access_overlaps(current.read_b, prior.write)


def _read_has_ordered_producer[max_stages: Int](
    stages: Array[ExecutionStage, max_stages], current_index: Int,
    reachable_predecessors: Int, access: ExecutionAccess,
) -> Bool:
    # Initial host input owners 0 and 1 are fully initialized by the native
    # owner binder. Every other read must be covered by earlier WRITERS that
    # happen-before the consumer. One writer need not cover the entire range:
    # a legal staged transfer can assemble it from disjoint partial writes.
    # The checked DAG separately rejects all unordered RAW/WAR/WAW hazards.
    # This is a bounded metadata-only proof, never an element-loop traversal.
    if not access.active():
        return True
    if access.space == 0 and (access.buffer_id == 0 or access.buffer_id == 1):
        return True
    # Self.checked() validates every active interval before calling here.
    # Grow the covered prefix only through real ordered predecessor writes.
    var covered = access.begin
    while covered < access.end:
        var next_end = covered
        for earlier in range(current_index):
            if reachable_predecessors & (Int(1) << earlier) == 0:
                continue
            var written = stages[earlier].write
            if not written.active():
                continue
            if written.space != access.space or written.buffer_id != access.buffer_id:
                continue
            if written.begin <= covered and written.end > next_end:
                next_end = written.end
        if next_end == covered:
            return False
        covered = min(next_end, access.end)
    return True


def equal_access_lengths(a: ExecutionAccess, b: ExecutionAccess) -> Bool:
    # This helper is also callable on raw, mutable stage metadata. Validate
    # both signed endpoints BEFORE subtraction; otherwise forged Int.MIN/MAX
    # endpoints could wrap a length and bless an invalid AOT stage signature.
    if not a.active() or not b.active():
        return False
    if a.space > 2 or b.space > 2 or a.buffer_id < 0 or b.buffer_id < 0:
        return False
    if a.begin < 0 or b.begin < 0 or a.end <= a.begin or b.end <= b.begin:
        return False
    if a.end > Int.MAX // 4 or b.end > Int.MAX // 4:
        return False
    return a.end - a.begin == b.end - b.begin


def valid_stage_signature(stage: ExecutionStage) -> Bool:
    # Bind each stage to the expected target mapping before launch.
    if stage.kind == ExecutionStageKind.CPU_PREPARE:
        # In the fixed host-binary ABI the only initially readable owners
        # are the two host inputs (0, 1). Scratch allocations use IDs >= 3.
        return (
            stage.lane == 0 and stage.read_a.space == 0
            and (stage.read_a.buffer_id == 0 or stage.read_a.buffer_id == 1)
            and not stage.read_b.active()
            and stage.write.space == 2 and stage.write.buffer_id >= 3
            and equal_access_lengths(stage.read_a, stage.write)
        )
    if stage.kind == ExecutionStageKind.H2D:
        return (
            stage.lane == 2 and stage.read_a.space == 2
            and stage.read_a.buffer_id >= 3 and not stage.read_b.active()
            and stage.write.space == 1 and stage.write.buffer_id >= 3
            and equal_access_lengths(stage.read_a, stage.write)
        )
    if stage.kind == ExecutionStageKind.GPU_KERNEL:
        # The current checked schedule models a binary elementwise FP32
        # kernel. Matmul/reduction need a separately typed operation contract.
        return (
            stage.lane == 1 and stage.read_a.space == 1 and stage.read_b.space == 1
            and stage.read_a.buffer_id >= 3 and stage.read_b.buffer_id >= 3
            and stage.write.space == 1 and stage.write.buffer_id >= 3
            and equal_access_lengths(stage.read_a, stage.write)
            and equal_access_lengths(stage.read_b, stage.write)
        )
    if stage.kind == ExecutionStageKind.D2H:
        return (
            stage.lane == 2 and stage.read_a.space == 1
            and stage.read_a.buffer_id >= 3 and not stage.read_b.active()
            and stage.write.space == 2 and stage.write.buffer_id >= 3
            and equal_access_lengths(stage.read_a, stage.write)
        )
    if stage.kind == ExecutionStageKind.CPU_COMPUTE:
        # A checked CPU stage may read initial host inputs (0, 1) or
        # snapshotted scratch (IDs >= 3), but it can only write OUTPUT 2.
        # Otherwise a forged schedule could overwrite a live host input.
        var a_host = (
            (stage.read_a.space == 0 and
             (stage.read_a.buffer_id == 0 or stage.read_a.buffer_id == 1))
            or (stage.read_a.space == 2 and stage.read_a.buffer_id >= 3)
        )
        var b_host = (
            (stage.read_b.space == 0 and
             (stage.read_b.buffer_id == 0 or stage.read_b.buffer_id == 1))
            or (stage.read_b.space == 2 and stage.read_b.buffer_id >= 3)
        )
        return (
            stage.lane == 0 and a_host and b_host
            and stage.write.space == 0 and stage.write.buffer_id == 2
            and equal_access_lengths(stage.read_a, stage.write)
            and equal_access_lengths(stage.read_b, stage.write)
        )
    if stage.kind == ExecutionStageKind.HOST_JOIN:
        return (
            stage.lane == 3 and stage.read_a.space == 2
            and stage.read_a.buffer_id >= 3 and not stage.read_b.active()
            and stage.write.space == 0 and stage.write.buffer_id == 2
            and equal_access_lengths(stage.read_a, stage.write)
        )
    return False


struct CheckedStageSchedule[max_stages: Int](Movable):
    var stages: Array[ExecutionStage, Self.max_stages]
    var count: Int
    var total_elements: Int

    def __init__(out self, var stages: Array[ExecutionStage, Self.max_stages], count: Int, total_elements: Int):
        self.stages = stages
        self.count = count
        self.total_elements = total_elements

    @staticmethod
    def checked(var stages: Array[ExecutionStage, Self.max_stages], count: Int, total_elements: Int) -> TensorResult[Self]:
        comptime assert Self.max_stages >= 1 and Self.max_stages <= 16, "AOT schedule capacity is 1..16"
        if count < 0 or count > Self.max_stages or total_elements < 0:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if total_elements > Int.MAX // 4:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        # A zero-element operation never needs a kernel, DMA stage, or a
        # synthetic host join. Conversely, positive output cannot be produced
        # by an empty graph. This agrees with SealedStaticDataflowPlan and
        # prevents side-effect-only stages from passing a zero-work boundary.
        if (count == 0 and total_elements > 0) or (count > 0 and total_elements == 0):
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        var transitive = Array[Int, Self.max_stages](fill=0)
        for i in range(count):
            var node = stages[i]
            if node.identifier != i or node.kind.code < 0 or node.kind.code > 5:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            # Validate every endpoint BEFORE any signature helper subtracts
            # lengths. Stage structs are public mutable metadata, not proofs.
            if not valid_access(node.read_a, total_elements) or not valid_access(node.read_b, total_elements) or not valid_access(node.write, total_elements):
                return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
            if not valid_stage_signature(node):
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if node.write.begin == node.write.end:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            # Stage-level dependencies cannot fix a write-after-read hazard
            # occurring INSIDE a single stage. This schedule is conservatively
            # out-of-place: in-place kernels require a separate proven contract.
            if access_overlaps(node.write, node.read_a):
                return TensorResult[Self](TensorStatus.OVERLAPPING)
            if access_overlaps(node.write, node.read_b):
                return TensorResult[Self](TensorStatus.OVERLAPPING)
            var earlier = (Int(1) << i) - 1
            if node.depends_on < 0 or node.depends_on & ~earlier != 0:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            var reachable = node.depends_on
            for j in range(i):
                if node.depends_on & (Int(1) << j) != 0:
                    reachable |= transitive[j]
            transitive[i] = reachable
            # RAW hazard checking alone is insufficient: a scratch read with
            # no previous writer has no conflict and would previously pass.
            # Reject uninitialized device/staging input before any enqueue.
            if not _read_has_ordered_producer[Self.max_stages](
                stages, i, reachable, node.read_a
            ):
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if not _read_has_ordered_producer[Self.max_stages](
                stages, i, reachable, node.read_b
            ):
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            # WAW, RAW and WAR require a dependency when accesses share
            # the SAME allocation identity and overlapping physical ranges.
            for j in range(i):
                if stage_access_conflict(node, stages[j]):
                    if reachable & (Int(1) << j) == 0:
                        return TensorResult[Self](TensorStatus.OVERLAPPING)
        return TensorResult[Self](Self(stages, count, total_elements))

    def checked_host_output_coverage(self, expected_begin: Int, expected_end: Int) -> TensorStatus:
        # DAG hazards prove ordering, not that all requested output elements
        # have a producer. This narrow host-binary ABI writes logical output
        # via CPU_COMPUTE or HOST_JOIN to host buffer 2. Prove an exact,
        # nonoverlapping partition of [expected_begin, expected_end).
        # The pipeline schedule covers only its two reusable staging windows,
        # not the entire later logical workload; callers supply that prefix.
        var rebuilt = Self.checked(self.stages.copy(), self.count, self.total_elements)
        if not rebuilt.is_ok():
            return rebuilt.status
        var fixed = rebuilt.payload.value()
        if expected_begin < 0 or expected_end < expected_begin or expected_end > fixed.total_elements:
            return TensorStatus.OUT_OF_BOUNDS
        var producer_count = 0
        for i in range(fixed.count):
            var node = fixed.stages[i]
            var written = node.write
            if written.space == 0 and written.buffer_id == 2:
                if (node.kind != ExecutionStageKind.CPU_COMPUTE
                    and node.kind != ExecutionStageKind.HOST_JOIN):
                    return TensorStatus.INVALID_LAYOUT
                if written.begin < expected_begin or written.end > expected_end:
                    return TensorStatus.OUT_OF_BOUNDS
                producer_count += 1
        if expected_begin == expected_end:
            if producer_count != 0:
                return TensorStatus.OVERLAPPING
            return TensorStatus.OK
        # Each producer must begin precisely where its predecessor ended.
        # This is O(stages^2) for at most 16 fixed AOT nodes, outside the hot
        # execution path; it does not sort, allocate or depend on buffer IDs
        # from an untrusted physical owner.
        var cursor = expected_begin
        var visited = 0
        while cursor < expected_end:
            var successor = cursor
            var matches = 0
            for i in range(fixed.count):
                var written = fixed.stages[i].write
                if written.space == 0 and written.buffer_id == 2 and written.begin == cursor:
                    matches += 1
                    successor = written.end
            if matches != 1 or successor <= cursor or successor > expected_end:
                return TensorStatus.INVALID_LAYOUT
            cursor = successor
            visited += 1
        if visited != producer_count:
            return TensorStatus.OVERLAPPING
        return TensorStatus.OK


# A checked stage schedule may contain allocations much smaller than its total
# logical work domain. Validate their own physical spans before backend launch.
@fieldwise_init
struct ExecutionBufferBounds(Copyable):
    var space: Int
    var buffer_id: Int
    var extent: Int


def _access_in_buffer_bounds[n: Int](
    access: ExecutionAccess, bounds: Array[ExecutionBufferBounds, n]
) -> TensorStatus:
    if not access.active():
        # An unused access is represented by exactly one canonical sentinel.
        # The public schedule fields must never bypass this proof.
        if access.space == -1 and access.buffer_id == -1 and access.begin == 0 and access.end == 0:
            return TensorStatus.OK
        return TensorStatus.INVALID_LAYOUT
    if access.space > 2 or access.buffer_id < 0 or access.begin < 0 or access.end <= access.begin:
        return TensorStatus.INVALID_LAYOUT
    for i in range(n):
        var entry = bounds[i]
        if entry.space == access.space and entry.buffer_id == access.buffer_id:
            if entry.extent < 0 or access.begin < 0 or access.end > entry.extent:
                return TensorStatus.OUT_OF_BOUNDS
            return TensorStatus.OK
    return TensorStatus.INVALID_LAYOUT


# Carries a validated DAG together with the EXACT buffer bounds used for its
# proof. Earlier versions returned only CheckedStageSchedule and silently
# discarded the physical bounds. This value deliberately contains NO kernel
# or allocation handle: native ownership must still be bound by the backend.
# Mojo 1.1 var fields are not language-enforced immutable; lowerings consume
# the revalidated owned snapshot, not the caller's potentially edited value.
struct BoundStageSchedule[stage_capacity: Int, buffer_capacity: Int](Movable):
    var _schedule: CheckedStageSchedule[Self.stage_capacity]
    var _bounds: Array[ExecutionBufferBounds, Self.buffer_capacity]

    def __init__(
        out self, var schedule: CheckedStageSchedule[Self.stage_capacity],
        bounds: Array[ExecutionBufferBounds, Self.buffer_capacity],
    ):
        self._schedule = schedule
        self._bounds = bounds.copy()

    def checked_snapshot(self) -> TensorResult[Self]:
        # The original schedules and their bound metadata are mutable
        # source-level values. Rebuild all DAG, read/write hazard, physical
        # interval and duplicate identity checks as one owned value.
        var copy_of_schedule = CheckedStageSchedule[Self.stage_capacity](
            self._schedule.stages.copy(),
            self._schedule.count,
            self._schedule.total_elements,
        )
        return bind_schedule_buffers_checked[Self.stage_capacity, Self.buffer_capacity](
            copy_of_schedule, self._bounds.copy()
        )

    def validate_for_launch(self, required_elements: Int) -> TensorStatus:
        if required_elements < 0 or required_elements != self._schedule.total_elements:
            return TensorStatus.SIZE_MISMATCH
        return self.checked_snapshot().status

    def validate_host_result_launch(
        self, required_elements: Int, expected_begin: Int, expected_end: Int,
    ) -> TensorStatus:
        # Reconstruct the owner-independent stage/buffer proof AND confirm
        # that this invocation's output partition has exactly one writer per
        # element. Never substitute this numerical binding for real allocation
        # provenance or event-fenced native ownership.
        var proof = self.checked_snapshot()
        if not proof.is_ok():
            return proof.status
        var fixed = proof.payload.value()
        if fixed._schedule.total_elements != required_elements:
            return TensorStatus.SIZE_MISMATCH
        return fixed._schedule.checked_host_output_coverage(expected_begin, expected_end)

    def stage_count(self) -> Int:
        # Metadata-only query; not a reusable proof of validation.
        return self._schedule.count


def bind_schedule_buffers_checked[stages: Int, buffers: Int](
    schedule: CheckedStageSchedule[stages],
    bounds: Array[ExecutionBufferBounds, buffers],
) -> TensorResult[BoundStageSchedule[stages, buffers]]:
    # A mutable schedule type alone is never a durable validation proof.
    # Rebind the owner-independent DAG to an owned snapshot of its exact
    # physical allocation bounds, keeping the two proofs together.
    # Revalidate stage signatures, ordering, RAW/WAR/WAW hazards and logical
    # bounds at the launch boundary, outside the kernel hot path.
    var rebuilt = CheckedStageSchedule[stages].checked(
        schedule.stages.copy(), schedule.count, schedule.total_elements
    )
    if not rebuilt.is_ok():
        return TensorResult[BoundStageSchedule[stages, buffers]](rebuilt.status)
    # Consume only the checked copy. A caller can mutate the original
    # schedule's var fields after construction; never reread those fields
    # when establishing physical access bounds for launch.
    var safe_schedule = rebuilt.payload.take()
    # Duplicate buffer identities could create conflicting capacity proofs.
    for i in range(buffers):
        var a = bounds[i]
        if a.extent < 0:
            return TensorResult[BoundStageSchedule[stages, buffers]](TensorStatus.INVALID_LAYOUT)
        if a.extent > Int.MAX // 4:
            return TensorResult[BoundStageSchedule[stages, buffers]](TensorStatus.OVERFLOW)
        if a.space < 0:
            if a.space != -1 or a.buffer_id != -1 or a.extent != 0:
                return TensorResult[BoundStageSchedule[stages, buffers]](TensorStatus.INVALID_LAYOUT)
            continue
        if a.space > 2 or a.buffer_id < 0:
            return TensorResult[BoundStageSchedule[stages, buffers]](TensorStatus.INVALID_LAYOUT)
        # This checked schedule has exactly two external host inputs and one
        # host output; all device/staging buffers are independent scratch.
        # A caller-controlled global buffer ID cannot silently change roles.
        if a.space == 0:
            if a.buffer_id > 2:
                return TensorResult[BoundStageSchedule[stages, buffers]](TensorStatus.INVALID_LAYOUT)
            if a.extent != safe_schedule.total_elements:
                return TensorResult[BoundStageSchedule[stages, buffers]](TensorStatus.SIZE_MISMATCH)
        elif a.buffer_id < 3:
            return TensorResult[BoundStageSchedule[stages, buffers]](TensorStatus.INVALID_LAYOUT)
        for j in range(i):
            var b = bounds[j]
            # Buffer IDs are allocation identities across ALL memory spaces,
            # not separate numbering namespaces. Reusing one identity for two
            # domains makes disjointness and ownership proofs ambiguous.
            if a.buffer_id == b.buffer_id and a.buffer_id >= 0:
                return TensorResult[BoundStageSchedule[stages, buffers]](TensorStatus.INVALID_LAYOUT)
    for i in range(safe_schedule.count):
        var stage = safe_schedule.stages[i]
        var read_a = _access_in_buffer_bounds(stage.read_a, bounds)
        if read_a != TensorStatus.OK:
            return TensorResult[BoundStageSchedule[stages, buffers]](read_a)
        var read_b = _access_in_buffer_bounds(stage.read_b, bounds)
        if read_b != TensorStatus.OK:
            return TensorResult[BoundStageSchedule[stages, buffers]](read_b)
        var write = _access_in_buffer_bounds(stage.write, bounds)
        if write != TensorStatus.OK:
            return TensorResult[BoundStageSchedule[stages, buffers]](write)
    return TensorResult[BoundStageSchedule[stages, buffers]](
        BoundStageSchedule[stages, buffers](safe_schedule, bounds)
    )


def validate_schedule_buffers[stages: Int, buffers: Int](
    schedule: CheckedStageSchedule[stages],
    bounds: Array[ExecutionBufferBounds, buffers],
) -> TensorStatus:
    # Compatibility status entry point; lowerings requiring a validated plan
    # must consume the owned snapshot from bind_schedule_buffers_checked().
    return bind_schedule_buffers_checked[stages, buffers](schedule, bounds).status
