# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR EXECUTION / STATIC DATAFLOW
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Execution Planning and AOT
# Module:        momijo.tensor_light.execution.static_dataflow
# Source:        src/momijo/tensor_light/execution/static_dataflow.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines static dataflow within checked AOT execution planning for the FP32
#   Light implementation.
#
#   Public declarations include `DataflowMemorySlot`, `DataflowRegion`,
#   `DataflowNode`, `SealedStaticDataflowPlan`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Represent and validate prepared plans, typed dataflow metadata,
#     placement, and immutable snapshots.
#   - Expose module-level declarations including `DataflowMemorySlot`,
#     `DataflowRegion`, `DataflowNode`, `SealedStaticDataflowPlan`.
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

# Unified static-dataflow *plan description* for fixed-FP32/rank-0..8 work.
# CPU, GPU and overlapped CPU/GPU stages are mappings of ONE node/edge model.
# This is the bounded, heap-free structural IR consumed by an AOT lowering.
# It is NOT an executable GPU kernel, a trusted artifact loader, a graph JIT,
# an alias-proven native allocation, or an asynchronous lifetime token.
# Such native proofs must be attached to the checked owner/loader boundary.
# There is no dtype field: all byte spans contain Float32 data exclusively.
from ..internal.status import TensorResult, TensorStatus
from .execution_plan import ExecutionPlan, ExecutionPlacement
from .dependency import BoundStageSchedule, ExecutionAccess, ExecutionStageKind


@fieldwise_init
struct DataflowMemorySlot(Copyable):
    # Logical owner index must equal its position in the fixed slot table.
    var identifier: Int
    # 0=host, 1=device, 2=host-pinned/staging. Device ordinal/real pointer
    # is bound separately from the source-level schedule.
    var domain: Int
    var capacity_elements: Int
    # Source declaration, NOT a trustable initialization proof. Backend must
    # bind an initialized owned Tensor/borrow with the declared region.
    var source_input: Bool
    var writable: Bool


@fieldwise_init
struct DataflowRegion(Copyable):
    var slot: Int
    var begin: Int
    var end: Int

    @staticmethod
    def none() -> Self:
        return Self(-1, 0, 0)

    def active(self) -> Bool:
        return self.slot >= 0


@fieldwise_init
struct DataflowNode(Copyable):
    var identifier: Int
    # Stage family: 0=CPU compute, 1=GPU compute, 2=H2D, 3=D2H,
    # 4=D2D, 5=host prepare, 6=host join, 7=host-to-staging copy.
    var kind: Int
    # Primitive family: 0=copy, 1=unary, 2=binary, 3=ternary,
    # 4=reduction, 5=scan, 6=matrix, 7=flat concatenation, 8=FP32 fill.
    var primitive: Int
    # AOT operation WITHIN the primitive family; this is semantic identity,
    # not the kernel variant. Binary: 0 add, 1 sub, 2 mul, 3 div, 4 max,
    # 5 min. The native loader must bind both opcode and compiled symbol.
    var opcode: Int
    # Bit j means the entire preceding node j must complete first.
    var predecessors: Int
    # Two reads plus optional third for one-pass FMA/where; no callback ABI.
    var read_a: DataflowRegion
    var read_b: DataflowRegion
    var read_c: DataflowRegion
    var write: DataflowRegion
    # Shape rank and math policy are fixed before native dispatch.
    var rank: Int
    var math_mode: Int
    var aot_variant: Int
    # Exact logical GEMM geometry for primitive=6, opcode=0, rank=2.
    # These MUST be zero for other primitive families. A flat element count
    # alone cannot prove the A/B/C shapes of a matrix multiplication.
    # Only dense row-major rank-2 GEMM is representable by this bounded IR;
    # batched/strided variants require shape/stride descriptors in their AOT ABI.
    var matrix_m: Int
    var matrix_k: Int
    var matrix_n: Int
    # Explicit contiguous reduction/scan geometry. For a reduction the
    # input run is output_count * reduction_extent FP32 values. For a scan,
    # scan_row_length must divide the equal input/output run length. Native
    # lowering must additionally bind the real shape, axis and strides.
    # These MUST be zero for unrelated primitive families.
    var reduction_extent: Int
    var scan_row_length: Int
    # Scratch uses an explicit, bounded FP32 region in the SAME slot table.
    # An integer scratch_bytes alone cannot establish allocation, isolation,
    # stream lifetime or even that scratch fits anywhere. All scratch stores
    # and loads remain private to the selected native AOT kernel.
    var scratch: DataflowRegion
    var scratch_bytes: Int
    # Only primitive=8 consumes this FP32 immediate. No dtype dispatch,
    # numeric scalar conversion or hidden external constant buffer is used.
    var fill_value: Float32


def _region_valid[slots: Int](
    region: DataflowRegion, memory: Array[DataflowMemorySlot, slots],
) -> Bool:
    if not region.active():
        return region.slot == -1 and region.begin == 0 and region.end == 0
    if region.slot >= slots or region.begin < 0 or region.end <= region.begin:
        return False
    if region.end > Int.MAX // 4:
        return False
    return region.end <= memory[region.slot].capacity_elements


def _same_allocation_overlap(a: DataflowRegion, b: DataflowRegion) -> Bool:
    # Call only AFTER checking each region against the exact slot table.
    if not a.active() or not b.active() or a.slot != b.slot:
        return False
    return a.begin < b.end and b.begin < a.end


def _same_region_length(a: DataflowRegion, b: DataflowRegion) -> Bool:
    # Called only on capacity-checked, positive-length regions. Subtracting
    # these endpoints cannot overflow because both lie in [0, Int.MAX//4].
    if not a.active() or not b.active():
        return False
    return a.end - a.begin == b.end - b.begin


def _same_region_length_or_singleton(a: DataflowRegion, b: DataflowRegion) -> Bool:
    # A rank-0 / scalar-like elementwise input may broadcast across a result
    # run. This does NOT represent nontrivial N-D broadcasts: the native
    # lowered descriptor must explicitly prove its shape and strides.
    if not a.active() or not b.active():
        return False
    var input_count = a.end - a.begin
    var output_count = b.end - b.begin
    return input_count == 1 or input_count == output_count


def _node_signature_valid(node: DataflowNode) -> Bool:
    # Structural AOT ABI: every stage must declare the input arity that its
    # primitive actually consumes. This prevents an absent read from being
    # treated as an initialized implicit zero and rejects forged DMA nodes.
    # Matrix geometry and output layout still need a trusted native binder.
    var a = node.read_a.active()
    var b = node.read_b.active()
    var c = node.read_c.active()
    # FP32 full/zeros and a K=0 GEMM output need a true zero-input fill.
    # Treating an absent input as an implicit zero in *another* primitive
    # would read uninitialized memory, so only this explicit opcode permits
    # arity zero. The actual lowering must bind the native fill instruction.
    if node.primitive == 8:
        return (node.opcode == 0 and not a and not b and not c
                and (node.kind == 0 or node.kind == 1)
                and node.matrix_m == 0 and node.matrix_k == 0
                and node.matrix_n == 0 and node.reduction_extent == 0
                and node.scan_row_length == 0 and node.scratch_bytes == 0
                and (node.rank != 0 or node.write.end - node.write.begin == 1))
    if not a:
        return False
    if c and not b:
        return False
    # Reject unrepresentable operation identities, not just illegal arity.
    # A primitive family does not identify the arithmetic performed by its
    # AOT artifact; the per-node opcode is part of the immutable IR contract.
    if node.opcode < 0:
        return False
    if node.primitive == 0 and node.opcode != 0:
        return False
    if node.primitive == 1 and node.opcode > 8:
        return False
    if node.primitive == 2 and node.opcode > 5:
        return False
    # The bounded IR has exactly three FP32 operands. It can express fused
    # multiply-add but NOT a `where` whose predicate is a separate TensorMask
    # type. A numeric opcode cannot invent a fourth/masked operand ABI.
    if node.primitive == 3 and node.opcode != 0:
        return False
    # STRICT_FP32 disallows silent contraction; this AOT ternary opcode is
    # explicitly FMA and may only be selected in an FMA-permitting mode.
    if node.primitive == 3 and node.math_mode == 0:
        return False
    if node.primitive == 4 and node.opcode > 6:
        return False
    if node.primitive == 5 and node.opcode > 3:
        return False
    # The only currently typed matrix ABI is dense, rank-2 GEMM. Every
    # other matrix opcode would need explicit geometry and ABI contracts.
    if node.primitive == 6 and node.opcode != 0:
        return False
    # A prefix scan needs an explicit sequence axis. A rank-zero scalar has
    # no axis over which an AOT scan could establish ordered dependencies.
    if node.primitive == 5 and node.rank == 0:
        return False
    # Integer argmin/argmax output indices do not have the FP32 element
    # representation of DataflowMemorySlot. Such results need a separate
    # non-Tensor scalar/index output contract, not forged Float32 storage.
    if node.primitive == 4 and node.opcode >= 5:
        return False
    # The only representable structural transform in this bounded flat ABI
    # is ordered, rank-one concatenation of 1..3 complete FP32 input runs.
    # Repeat/tile, transpose, stack, gather and arbitrary reshapes require
    # shape/stride/index descriptors that are NOT present in this node.
    if node.primitive == 7 and (node.opcode != 0 or node.rank != 1):
        return False
    if node.rank == 0 and node.write.end - node.write.begin != 1:
        return False
    if node.primitive != 6:
        # Unused matrix dimensions are canonical zeros, not latent mutable
        # geometry that could disagree with a selected operation identity.
        if node.matrix_m != 0 or node.matrix_k != 0 or node.matrix_n != 0:
            return False
    if node.primitive != 4 and node.reduction_extent != 0:
        return False
    if node.primitive != 5 and node.scan_row_length != 0:
        return False
    # Transfers and joins are one-input whole-FP32-element copies. There
    # is no out-of-place dtype conversion in the fixed-FP32 architecture.
    if node.kind == 2 or node.kind == 3 or node.kind == 4 or node.kind == 7 or node.kind == 6:
        return node.primitive == 0 and node.opcode == 0 and not b and not c and _same_region_length(node.read_a, node.write)
    # Host packing/preparation is restricted to explicit one-source copies
    # until a native pack ABI with shape/stride descriptors has been bound.
    if node.kind == 5:
        return node.primitive == 0 and node.opcode == 0 and not b and not c and _same_region_length(node.read_a, node.write)
    if node.primitive == 0 or node.primitive == 1:
        return not b and not c and _same_region_length(node.read_a, node.write)
    if node.primitive == 5:
        # One contiguous prefix-scan run or multiple complete, equally sized
        # rows. Without axis/stride binding the plan MUST NOT pretend it can
        # scan an arbitrary strided N-D dimension.
        if b or c or not _same_region_length(node.read_a, node.write):
            return False
        var input_count = node.read_a.end - node.read_a.begin
        var row_length = node.scan_row_length
        return row_length > 0 and input_count % row_length == 0
    if node.primitive == 2:
        # Allow one-element operands used in scalar broadcasting. The
        # general N-D broadcast ABI is a separate, required native proof.
        return b and not c and _same_region_length_or_singleton(node.read_a, node.write) and _same_region_length_or_singleton(node.read_b, node.write)
    if node.primitive == 3:
        return b and c and _same_region_length_or_singleton(node.read_a, node.write) and _same_region_length_or_singleton(node.read_b, node.write) and _same_region_length_or_singleton(node.read_c, node.write)
    if node.primitive == 4:
        # A flat row-wise reduction has exactly one output per complete input
        # segment. Merely allowing any smaller output silently admitted
        # incompatible segment sizes, non-integral axes and wrong AOT ABIs.
        # Division, rather than multiplication, avoids a count overflow.
        if b or c or node.reduction_extent <= 0:
            return False
        var input_count = node.read_a.end - node.read_a.begin
        var output_count = node.write.end - node.write.begin
        if output_count <= 0 or input_count % output_count != 0:
            return False
        return input_count // output_count == node.reduction_extent
    if node.primitive == 6:
        # Exact rank-2 dense GEMM: A[M,K] @ B[K,N] -> C[M,N]. Nonempty
        # regions are compulsory: K=0 has a separate zero-fill lowering,
        # and M=0/N=0 have no executable nodes in this graph contract.
        if not b or c or node.rank != 2:
            return False
        var m = node.matrix_m
        var k = node.matrix_k
        var n = node.matrix_n
        if m <= 0 or k <= 0 or n <= 0:
            return False
        # All counts are FP32 elements, so prove products are byte-safe
        # BEFORE multiplying. Equal lengths alone cannot identify GEMM.
        var maximum_elements = Int.MAX // 4
        if m > maximum_elements // k or k > maximum_elements // n:
            return False
        if m > maximum_elements // n:
            return False
        var a_count = node.read_a.end - node.read_a.begin
        var b_count = node.read_b.end - node.read_b.begin
        var c_count = node.write.end - node.write.begin
        return a_count == m * k and b_count == k * n and c_count == m * n
    if node.primitive == 7:
        # This descriptive IR has no repeat factors, affine permutation,
        # slice geometry or N-D shape operands. Until those appear in a
        # *trusted* native AOT binder, only size-preserving one-input views/
        # copies and concatenation/stack of 2-3 complete input runs can be
        # represented truthfully. Reject unproved tile/repeat/gather lengths
        # instead of blessing arbitrary output bytes as initialized FP32.
        var output_count = node.write.end - node.write.begin
        var remaining = output_count
        var count_a = node.read_a.end - node.read_a.begin
        if count_a > remaining:
            return False
        remaining -= count_a
        if b:
            var count_b = node.read_b.end - node.read_b.begin
            if count_b > remaining:
                return False
            remaining -= count_b
        if c:
            var count_c = node.read_c.end - node.read_c.begin
            if count_c > remaining:
                return False
            remaining -= count_c
        return remaining == 0
    return False


def _has_dataflow_hazard(a: DataflowNode, b: DataflowNode) -> Bool:
    # Covers WAW, WAR and RAW. Scratch is a READ/WRITE region owned by the
    # kernel from its launch until its completion fence, so it conflicts with
    # every overlapping read, write, or scratch region of another node.
    # Different slot IDs are NOT a native noalias proof: the owner binding
    # must also prove that distinct logical slots have disjoint allocations.
    if _same_allocation_overlap(a.write, b.write):
        return True
    if _same_allocation_overlap(a.write, b.read_a):
        return True
    if _same_allocation_overlap(a.write, b.read_b):
        return True
    if _same_allocation_overlap(a.write, b.read_c):
        return True
    if _same_allocation_overlap(a.read_a, b.write):
        return True
    if _same_allocation_overlap(a.read_b, b.write):
        return True
    if _same_allocation_overlap(a.read_c, b.write):
        return True
    if _same_allocation_overlap(a.scratch, b.scratch):
        return True
    if _same_allocation_overlap(a.scratch, b.write):
        return True
    if _same_allocation_overlap(a.write, b.scratch):
        return True
    if _same_allocation_overlap(a.scratch, b.read_a):
        return True
    if _same_allocation_overlap(a.scratch, b.read_b):
        return True
    if _same_allocation_overlap(a.scratch, b.read_c):
        return True
    if _same_allocation_overlap(a.read_a, b.scratch):
        return True
    if _same_allocation_overlap(a.read_b, b.scratch):
        return True
    return _same_allocation_overlap(a.read_c, b.scratch)


def _ordered_read_is_produced[nodes: Int, slots: Int](
    graph: Array[DataflowNode, nodes], memory: Array[DataflowMemorySlot, slots],
    index: Int, transitive_before: Int, read: DataflowRegion,
) -> Bool:
    if not read.active():
        return True
    if memory[read.slot].source_input:
        # The native binder must supply a real initialized source owner.
        return True
    # A scratch read may be covered by multiple ordered, disjoint writes.
    # Greedily extend the covered prefix using ordered prior writers; this
    # bounded O(nodes^2) operation executes at the checked boundary only.
    var covered = read.begin
    while covered < read.end:
        var next_end = covered
        for before in range(index):
            if transitive_before & (Int(1) << before) == 0:
                continue
            var written = graph[before].write
            if written.slot == read.slot and written.active():
                if written.begin <= covered and written.end > next_end:
                    next_end = written.end
        if next_end == covered:
            return False
        covered = min(next_end, read.end)
    return True


def _node_domains_valid[slots: Int](
    node: DataflowNode, memory: Array[DataflowMemorySlot, slots],
) -> Bool:
    # All active inputs need domain-compatible target lowering. The identity
    # of the actual CPU/GPU device remains a separate trusted AOT binding.
    var output_domain = memory[node.write.slot].domain
    if node.kind == 0 or node.kind == 5 or node.kind == 6:
        if output_domain == 1:
            return False
        if node.read_a.active() and memory[node.read_a.slot].domain == 1:
            return False
        if node.read_b.active() and memory[node.read_b.slot].domain == 1:
            return False
        if node.read_c.active() and memory[node.read_c.slot].domain == 1:
            return False
        return True
    if node.kind == 1:
        if output_domain != 1:
            return False
        if node.read_a.active() and memory[node.read_a.slot].domain != 1:
            return False
        if node.read_b.active() and memory[node.read_b.slot].domain != 1:
            return False
        if node.read_c.active() and memory[node.read_c.slot].domain != 1:
            return False
        return True
    # Transfers have one source and one destination; no multi-source DMA.
    if not node.read_a.active() or node.read_b.active() or node.read_c.active():
        return False
    var source_domain = memory[node.read_a.slot].domain
    if node.kind == 2:
        return source_domain != 1 and output_domain == 1
    if node.kind == 3:
        return source_domain == 1 and output_domain != 1
    if node.kind == 4:
        return source_domain == 1 and output_domain == 1
    if node.kind == 7:
        return source_domain == 0 and output_domain == 2
    return False


# Validation is separate from the executor to avoid a runtime graph runtime.
# A compiled AOT lowering may keep the checked snapshot as immutable policy;
# just passing an artifact number does NOT authenticate the executable.
struct SealedStaticDataflowPlan[max_nodes: Int, max_slots: Int](Movable):
    var _nodes: Array[DataflowNode, Self.max_nodes]
    var _memory: Array[DataflowMemorySlot, Self.max_slots]
    var _count: Int
    var _logical_elements: Int
    var _output_slot: Int
    var _used_slots: Int

    def __init__(
        out self, nodes: Array[DataflowNode, Self.max_nodes],
        memory: Array[DataflowMemorySlot, Self.max_slots], count: Int,
        logical_elements: Int, output_slot: Int, used_slots: Int,
    ):
        # Only the checked factory is a valid way to instantiate this value.
        # Revalidation at the native boundary still protects against forged
        # public structs and unsafe caller mutation of their `var` fields.
        self._nodes = nodes.copy()
        self._memory = memory.copy()
        self._count = count
        self._logical_elements = logical_elements
        self._output_slot = output_slot
        self._used_slots = used_slots

    @staticmethod
    def checked(
        nodes: Array[DataflowNode, Self.max_nodes],
        memory: Array[DataflowMemorySlot, Self.max_slots], count: Int,
        logical_elements: Int, output_slot: Int, used_slots: Int,
    ) -> TensorResult[Self]:
        # Int is a signed 64-bit dependency mask on the supported targets.
        # Keep the sign bit clear and leave one spare bit for safe full-mask
        # algebra; this permits 62 AOT-specialized nodes without allocating
        # a heap graph, while preserving the original compact Int bitsets.
        # Normal 1..16-node specializations retain their original footprint.
        comptime assert Self.max_nodes >= 1 and Self.max_nodes <= 62, "bounded AOT DAG <=62 nodes"
        comptime assert Self.max_slots >= 1 and Self.max_slots <= 64, "bounded AOT storage <=64 slots"
        if logical_elements < 0 or logical_elements > Int.MAX // 4:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        if count < 0 or count > Self.max_nodes or used_slots <= 0 or used_slots > Self.max_slots:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if output_slot < 0 or output_slot >= used_slots:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        # No element exists to write in a zero-sized result. A nonempty DAG
        # for such an output would have only unrelated side effects and may
        # falsely advertise completion of an output the graph never wrote.
        if logical_elements == 0 and count != 0:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if logical_elements > 0 and count == 0:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        for i in range(used_slots):
            var slot = memory[i]
            if slot.identifier != i or slot.domain < 0 or slot.domain > 2:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if slot.capacity_elements < 0 or slot.capacity_elements > Int.MAX // 4:
                return TensorResult[Self](TensorStatus.OVERFLOW)
            # This IR is out-of-place. An input is a borrowed *read-only*
            # source even when the actual owner has mutable storage. Reject
            # contradictory slot flags at the plan boundary: a native slot
            # binder must never infer write permission from a forged metadata
            # record that simultaneously claims to be an input.
            if slot.source_input and slot.writable:
                return TensorResult[Self](TensorStatus.READ_ONLY)
        # A declared source_input describes readable caller-owned input
        # state, never a scratch/output allocation that the dataflow may
        # silently overwrite. In-place updates require a separately typed
        # mutable-borrow/alias plan with explicit original-value lifetime.
        if memory[output_slot].source_input:
            return TensorResult[Self](TensorStatus.READ_ONLY)
        if not memory[output_slot].writable:
            return TensorResult[Self](TensorStatus.READ_ONLY)
        if logical_elements > memory[output_slot].capacity_elements:
            return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
        # Scratch is kernel-private workspace, not an initialized operand
        # or a logical output. Reserving its slots here prevents a later
        # ordinary read from incorrectly treating an earlier *data* write as
        # initialized after an intervening opaque scratch overwrite. Reuse
        # between ordered scratch-using kernels remains legal. A future
        # alias-aware allocator may relax this only with owner-backed
        # lifetime and content-preservation proofs.
        var reserved_scratch_slots = Array[Bool, Self.max_slots](fill=False)
        for i in range(count):
            var scratch = nodes[i].scratch
            if scratch.active() and scratch.slot < used_slots:
                reserved_scratch_slots[scratch.slot] = True
        var closures = Array[Int, Self.max_nodes](fill=0)
        for i in range(count):
            var node = nodes[i]
            if node.identifier != i or node.kind < 0 or node.kind > 7:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if node.primitive < 0 or node.primitive > 8 or node.rank < 0 or node.rank > 8:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if node.math_mode < 0 or node.math_mode > 3 or node.aot_variant < 0:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if node.scratch_bytes < 0 or node.scratch_bytes > Int.MAX // 4 * 4:
                return TensorResult[Self](TensorStatus.OVERFLOW)
            if node.scratch_bytes % 4 != 0:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            # Scratch metadata cannot silently promise temporary capacity.
            # The bound region is byte-exact and cannot alias either input,
            # the output, or the caller-owned public result allocation.
            # Transfers, host joins and preparation in this version do not
            # have a native scratch ABI; compute stages alone may use it.
            if node.scratch_bytes == 0:
                if node.scratch.active() or not _region_valid[Self.max_slots](node.scratch, memory):
                    return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            else:
                if node.kind != 0 and node.kind != 1:
                    return TensorResult[Self](TensorStatus.UNSUPPORTED)
                if not node.scratch.active() or node.scratch.slot >= used_slots:
                    return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
                if not _region_valid[Self.max_slots](node.scratch, memory):
                    return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
                var scratch_slot = memory[node.scratch.slot]
                if scratch_slot.source_input or not scratch_slot.writable:
                    return TensorResult[Self](TensorStatus.READ_ONLY)
                if node.scratch.slot == output_slot:
                    return TensorResult[Self](TensorStatus.OVERLAPPING)
                if node.kind == 1 and scratch_slot.domain != 1:
                    return TensorResult[Self](TensorStatus.DEVICE_MISMATCH)
                if node.kind == 0 and scratch_slot.domain == 1:
                    return TensorResult[Self](TensorStatus.DEVICE_MISMATCH)
                if node.scratch.end - node.scratch.begin != node.scratch_bytes // 4:
                    return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
                if (_same_allocation_overlap(node.scratch, node.read_a)
                    or _same_allocation_overlap(node.scratch, node.read_b)
                    or _same_allocation_overlap(node.scratch, node.read_c)
                    or _same_allocation_overlap(node.scratch, node.write)):
                    return TensorResult[Self](TensorStatus.OVERLAPPING)
            if not node.write.active() or node.write.slot >= used_slots:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if not _region_valid[Self.max_slots](node.write, memory):
                return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
            # A caller must not make a source allocation writable just by
            # toggling its flag: this out-of-place DAG never writes inputs.
            if memory[node.write.slot].source_input:
                return TensorResult[Self](TensorStatus.READ_ONLY)
            if not memory[node.write.slot].writable:
                return TensorResult[Self](TensorStatus.READ_ONLY)
            if not _region_valid[Self.max_slots](node.read_a, memory):
                return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
            if not _region_valid[Self.max_slots](node.read_b, memory):
                return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
            if not _region_valid[Self.max_slots](node.read_c, memory):
                return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
            if node.read_a.active() and node.read_a.slot >= used_slots:
                return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
            if node.read_b.active() and node.read_b.slot >= used_slots:
                return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
            if node.read_c.active() and node.read_c.slot >= used_slots:
                return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
            # A private scratch slot must not silently become an FP32 result
            # or readable operand. The native kernel is free to overwrite
            # every byte of its scratch region with unspecified values.
            if reserved_scratch_slots[node.write.slot]:
                return TensorResult[Self](TensorStatus.OVERLAPPING)
            if node.read_a.active() and reserved_scratch_slots[node.read_a.slot]:
                return TensorResult[Self](TensorStatus.OVERLAPPING)
            if node.read_b.active() and reserved_scratch_slots[node.read_b.slot]:
                return TensorResult[Self](TensorStatus.OVERLAPPING)
            if node.read_c.active() and reserved_scratch_slots[node.read_c.slot]:
                return TensorResult[Self](TensorStatus.OVERLAPPING)
            if not _node_domains_valid[Self.max_slots](node, memory):
                return TensorResult[Self](TensorStatus.DEVICE_MISMATCH)
            # A source-level DAG cannot invent input values for an incomplete
            # unary/binary/ternary signature or invoke a DMA stage as GEMM.
            if not _node_signature_valid(node):
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if _same_allocation_overlap(node.write, node.read_a):
                return TensorResult[Self](TensorStatus.OVERLAPPING)
            if _same_allocation_overlap(node.write, node.read_b):
                return TensorResult[Self](TensorStatus.OVERLAPPING)
            if _same_allocation_overlap(node.write, node.read_c):
                return TensorResult[Self](TensorStatus.OVERLAPPING)
            var earlier_mask = (Int(1) << i) - 1
            if node.predecessors < 0 or node.predecessors & ~earlier_mask != 0:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            var reachable = node.predecessors
            for before in range(i):
                if node.predecessors & (Int(1) << before) != 0:
                    reachable |= closures[before]
            closures[i] = reachable
            if not _ordered_read_is_produced[Self.max_nodes, Self.max_slots](
                nodes, memory, i, reachable, node.read_a,
            ):
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if not _ordered_read_is_produced[Self.max_nodes, Self.max_slots](
                nodes, memory, i, reachable, node.read_b,
            ):
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if not _ordered_read_is_produced[Self.max_nodes, Self.max_slots](
                nodes, memory, i, reachable, node.read_c,
            ):
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            for before in range(i):
                if _has_dataflow_hazard(node, nodes[before]):
                    if reachable & (Int(1) << before) == 0:
                        return TensorResult[Self](TensorStatus.OVERLAPPING)
        # The public output must be populated by disjoint final writes. We
        # prohibit partial overwritten output regions in this conservative
        # source-level plan; data reuse is valid for non-output scratch slots.
        if logical_elements > 0:
            var covered = 0
            var producers = 0
            while covered < logical_elements:
                var next_end = covered
                var matches = 0
                for i in range(count):
                    var out = nodes[i].write
                    if out.slot == output_slot and out.begin == covered:
                        next_end = out.end
                        matches += 1
                if matches != 1 or next_end <= covered or next_end > logical_elements:
                    return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
                covered = next_end
                producers += 1
            var all_producers = 0
            for i in range(count):
                if nodes[i].write.slot == output_slot:
                    all_producers += 1
            if all_producers != producers:
                return TensorResult[Self](TensorStatus.OVERLAPPING)
            # One canonical, side-effect-free workload has ONE public output.
            # Every executable stage must be on a predecessor path to a final
            # output writer. A dead independent kernel/transfer would consume
            # resources, violate thin deterministic lowering assumptions and
            # may keep a scratch owner alive after the advertised join.
            var required_nodes = 0
            for i in range(count):
                if nodes[i].write.slot == output_slot:
                    required_nodes |= (Int(1) << i) | closures[i]
            var complete_mask = (Int(1) << count) - 1
            if required_nodes != complete_mask:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            # A synchronization-only dependency is not a producer/consumer
            # relationship. Each non-output write must be read by a later,
            # dependency-ordered stage. This rejects dead packing/compute
            # stages that merely appear as predecessors of the public output.
            # Native allocation identity remains an independent obligation.
            for produced_index in range(count):
                var produced = nodes[produced_index].write
                if produced.slot == output_slot:
                    continue
                var consumed = False
                for reader_index in range(produced_index + 1, count):
                    var reader = nodes[reader_index]
                    if closures[reader_index] & (Int(1) << produced_index) == 0:
                        continue
                    if (_same_allocation_overlap(produced, reader.read_a)
                        or _same_allocation_overlap(produced, reader.read_b)
                        or _same_allocation_overlap(produced, reader.read_c)):
                        consumed = True
                        break
                if not consumed:
                    return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        # A bounded Array can carry entries beyond the active node/slot
        # counts. They were never validated above and must never survive a
        # successful checked_snapshot as latent native selector/owner data.
        # Normalize every inactive record to a deterministic inert value;
        # only the active prefix is semantically part of the sealed plan.
        var canonical_nodes = nodes.copy()
        var unused_region = DataflowRegion.none()
        for i in range(count, Self.max_nodes):
            canonical_nodes[i] = DataflowNode(
                i, 5, 0, 0, 0,
                unused_region, unused_region, unused_region, unused_region,
                1, 0, 0, 0, 0, 0, 0, 0,
                unused_region, 0, Float32(0.0),
            )
        var canonical_memory = memory.copy()
        for i in range(used_slots, Self.max_slots):
            canonical_memory[i] = DataflowMemorySlot(i, 0, 0, False, False)
        return TensorResult[Self](Self(
            canonical_nodes, canonical_memory, count,
            logical_elements, output_slot, used_slots,
        ))

    def checked_snapshot(self) -> TensorResult[Self]:
        # Rebuild an owned value from ALL fields so the backend cannot rely
        # on a stale validation performed before mutable metadata changed.
        return Self.checked(
            self._nodes.copy(), self._memory.copy(), self._count,
            self._logical_elements, self._output_slot, self._used_slots,
        )

    # Narrow unchecked accessors are exclusively for a local lowering that
    # already holds a successful checked_snapshot() of THIS plan value.
    # They avoid re-running the O(nodes^2) graph validator per kernel stage.
    def _cpu_lowering_used_slots(self) -> Int:
        return self._used_slots

    def _cpu_lowering_slot_unchecked(self, index: Int) -> DataflowMemorySlot:
        return self._memory[index].copy()

    def _cpu_lowering_node_unchecked(self, index: Int) -> DataflowNode:
        return self._nodes[index].copy()

    def node_count(self) -> Int:
        return self._count

    def output_count(self) -> Int:
        return self._logical_elements

    def checked_node(self, index: Int) -> TensorResult[DataflowNode]:
        if index < 0 or index >= self._count:
            return TensorResult[DataflowNode](TensorStatus.OUT_OF_BOUNDS)
        var proof = self.checked_snapshot()
        if not proof.is_ok():
            return TensorResult[DataflowNode](proof.status)
        return TensorResult[DataflowNode](proof.payload.value()._nodes[index].copy())

    def checked_scratch_reuse_eligibility(
        self, released_slot: Int, next_slot: Int,
    ) -> TensorResult[Bool]:
        # Proves a necessary static liveness/order precondition for offline
        # allocation reuse. It does NOT release memory, authorize pointer
        # aliasing, or replace device-event/stream ownership at execution.
        var plan = self.checked_snapshot()
        if not plan.is_ok():
            return TensorResult[Bool](plan.status)
        var safe = plan.payload.value()
        if released_slot < 0 or next_slot < 0:
            return TensorResult[Bool](TensorStatus.OUT_OF_BOUNDS)
        if released_slot >= safe._used_slots or next_slot >= safe._used_slots:
            return TensorResult[Bool](TensorStatus.OUT_OF_BOUNDS)
        if released_slot == next_slot:
            return TensorResult[Bool](False)
        if released_slot == safe._output_slot or next_slot == safe._output_slot:
            return TensorResult[Bool](False)
        var released = safe._memory[released_slot]
        var upcoming = safe._memory[next_slot]
        if released.source_input or upcoming.source_input:
            return TensorResult[Bool](False)
        if released.domain != upcoming.domain:
            return TensorResult[Bool](False)
        if released.capacity_elements < upcoming.capacity_elements:
            return TensorResult[Bool](False)
        # A lexical "last use" is NOT a safe fence in a concurrent DAG.
        # Example: old uses A and B are independent, but only B precedes the
        # first new use C. C may launch while A is still using the old block.
        # EVERY old use must happen-before EVERY new use, including separate
        # GPU streams, CPU assists and apparently disjoint later new writes.
        var transitive = Array[Int, Self.max_nodes](fill=0)
        for i in range(safe._count):
            var deps = safe._nodes[i].predecessors
            for j in range(i):
                if safe._nodes[i].predecessors & (Int(1) << j) != 0:
                    deps |= transitive[j]
            transitive[i] = deps
        var saw_old = False
        var saw_new = False
        for old_index in range(safe._count):
            var old_node = safe._nodes[old_index]
            var old_used = (
                old_node.read_a.slot == released_slot
                or old_node.read_b.slot == released_slot
                or old_node.read_c.slot == released_slot
                or old_node.write.slot == released_slot
                or old_node.scratch.slot == released_slot
            )
            if not old_used:
                continue
            saw_old = True
            for new_index in range(safe._count):
                var new_node = safe._nodes[new_index]
                var new_used = (
                    new_node.read_a.slot == next_slot
                    or new_node.read_b.slot == next_slot
                    or new_node.read_c.slot == next_slot
                    or new_node.write.slot == next_slot
                    or new_node.scratch.slot == next_slot
                )
                if not new_used:
                    continue
                saw_new = True
                if new_index <= old_index:
                    return TensorResult[Bool](False)
                if transitive[new_index] & (Int(1) << old_index) == 0:
                    return TensorResult[Bool](False)
        return TensorResult[Bool](saw_old and saw_new)


# Conservative *logical* allocation requirements for the same checked plan.
# This is a source-level bound for offline memory scheduling, NOT a native
# allocation, pointer-identity proof, or permission to reuse a block before
# its final completion fence. Input slots may be externally owned, and a
# scratch slot is already included in total_declared_capacity_bytes.
@fieldwise_init
struct StaticDataflowMemoryRequirements(Copyable):
    var host_capacity_bytes: Int
    var device_capacity_bytes: Int
    var pinned_capacity_bytes: Int
    var total_declared_capacity_bytes: Int
    var source_input_capacity_bytes: Int
    var scratch_slot_capacity_bytes: Int
    var maximum_node_scratch_bytes: Int


def static_dataflow_memory_requirements[nodes: Int, slots: Int](
    plan: SealedStaticDataflowPlan[nodes, slots],
) -> TensorResult[StaticDataflowMemoryRequirements]:
    # Validation is outside all execution loops. Every declared slot is
    # counted once, including source inputs and staging; native planning can
    # subsequently decide which allocations are externally provided.
    var checked = plan.checked_snapshot()
    if not checked.is_ok():
        return TensorResult[StaticDataflowMemoryRequirements](checked.status)
    var safe = checked.payload.value()
    var max_elements = Int.MAX // 4
    var host_elements = 0
    var device_elements = 0
    var pinned_elements = 0
    var total_elements = 0
    var input_elements = 0
    var scratch_elements = 0
    var largest_scratch_bytes = 0
    var seen_scratch = Array[Bool, slots](fill=False)
    for index in range(safe._used_slots):
        var slot = safe._memory[index]
        var capacity = slot.capacity_elements
        # Checked construction proves each capacity is non-negative and <=
        # Int.MAX/4, but the sum of multiple slots can still overflow.
        if capacity > max_elements - total_elements:
            return TensorResult[StaticDataflowMemoryRequirements](
                TensorStatus.OVERFLOW
            )
        total_elements += capacity
        if slot.domain == 0:
            host_elements += capacity
        elif slot.domain == 1:
            device_elements += capacity
        else:
            pinned_elements += capacity
        if slot.source_input:
            input_elements += capacity
    for index in range(safe._count):
        var node = safe._nodes[index]
        if node.scratch.active():
            var scratch_slot = node.scratch.slot
            if not seen_scratch[scratch_slot]:
                seen_scratch[scratch_slot] = True
                # Scratch slot capacities are included in total_elements.
                # Count each even when several ordered kernels reuse it.
                scratch_elements += safe._memory[scratch_slot].capacity_elements
            largest_scratch_bytes = max(
                largest_scratch_bytes, node.scratch_bytes
            )
    return TensorResult[StaticDataflowMemoryRequirements](
        StaticDataflowMemoryRequirements(
            host_elements * 4, device_elements * 4, pinned_elements * 4,
            total_elements * 4, input_elements * 4, scratch_elements * 4,
            largest_scratch_bytes,
        )
    )


# Immutable-by-contract AOT scheduling summary for the SAME canonical
# SealedStaticDataflowPlan; not another graph model, scheduler or backend.
# A wave contains only DAG-independent nodes. This is a lowering hint, NOT
# proof that distinct logical slot identifiers represent disjoint native
# allocations or that CPU/GPU streams can actually execute concurrently.
@fieldwise_init
struct StaticDataflowWaveAnalysis[capacity: Int](Copyable):
    var wave_of_node: Array[Int, Self.capacity]
    var nodes_in_wave: Array[Int, Self.capacity]
    var wave_count: Int
    var max_wave_width: Int
    var output_writers_mask: Int
    var all_terminal_nodes_mask: Int
    var candidate_host_device_overlap: Bool


def analyze_static_dataflow_waves[nodes: Int, slots: Int](
    plan: SealedStaticDataflowPlan[nodes, slots],
) -> TensorResult[StaticDataflowWaveAnalysis[nodes]]:
    # Validate and snapshot the exact original model at the checked boundary.
    # The dependency masks are strictly lower-triangular, so one forward
    # traversal produces the minimum wave number respecting every edge.
    var checked = plan.checked_snapshot()
    if not checked.is_ok():
        return TensorResult[StaticDataflowWaveAnalysis[nodes]](checked.status)
    var safe = checked.payload.value()
    var waves = Array[Int, nodes](fill=0)
    var masks = Array[Int, nodes](fill=0)
    var occupied = Array[Int, nodes](fill=0)
    var host_lane = Array[Bool, nodes](fill=False)
    var gpu_lane = Array[Bool, nodes](fill=False)
    var all_predecessors_mask = 0
    var output_writers = 0
    var total_waves = 0
    var largest_wave = 0
    for index in range(safe._count):
        var node = safe._nodes[index]
        var wave = 0
        for before in range(index):
            if node.predecessors & (Int(1) << before) != 0:
                wave = max(wave, waves[before] + 1)
        waves[index] = wave
        masks[wave] = masks[wave] | (Int(1) << index)
        occupied[wave] = occupied[wave] + 1
        largest_wave = max(largest_wave, occupied[wave])
        total_waves = max(total_waves, wave + 1)
        all_predecessors_mask |= node.predecessors
        if node.write.slot == safe._output_slot:
            output_writers |= Int(1) << index
        if node.kind == 0 or node.kind == 5 or node.kind == 6 or node.kind == 7:
            host_lane[wave] = True
        elif node.kind == 1:
            gpu_lane[wave] = True
    var possible_overlap = False
    for wave in range(total_waves):
        if host_lane[wave] and gpu_lane[wave]:
            possible_overlap = True
    # A terminal node has no outgoing prerequisite edge. Releasing any
    # scratch owner before the terminal completion join is never permitted.
    var terminal_mask = 0
    for index in range(safe._count):
        if all_predecessors_mask & (Int(1) << index) == 0:
            terminal_mask |= Int(1) << index
    return TensorResult[StaticDataflowWaveAnalysis[nodes]](
        StaticDataflowWaveAnalysis[nodes](
            waves, masks, total_waves, largest_wave,
            output_writers, terminal_mask, possible_overlap,
        )
    )


# A single source-level agreement proof between the canonical FP32 DAG and
# its target lowering. The two existing plan values must describe THE SAME
# output, backend mapping, disjoint CPU/GPU result partitions and explicit
# transfer traffic before dispatch. This evidence is not a native allocator
# binding, ISA attestation, DeviceEvent or executable AOT artifact.
@fieldwise_init
struct DataflowExecutionEvidence(Copyable):
    var cpu_output_elements: Int
    var gpu_output_elements: Int
    var cpu_nodes: Int
    var gpu_nodes: Int
    var host_prepare_nodes: Int
    var transfer_nodes: Int
    var required_transfer_bytes: Int
    var output_domain: Int


def _validate_static_dataflow_execution_plan[
    per_node_artifacts: Bool, nodes: Int, slots: Int,
](
    graph: SealedStaticDataflowPlan[nodes, slots], placement: ExecutionPlan,
) -> TensorResult[DataflowExecutionEvidence]:
    # Never compare unchecked mutable fields. Each snapshot validates all
    # logical bounds and edges exactly once; nothing runs on the hot path.
    var graph_proof = graph.checked_snapshot()
    if not graph_proof.is_ok():
        return TensorResult[DataflowExecutionEvidence](graph_proof.status)
    var plan_proof = placement.checked_snapshot()
    if not plan_proof.is_ok():
        return TensorResult[DataflowExecutionEvidence](plan_proof.status)
    var safe = graph_proof.payload.value()
    var chosen = plan_proof.payload.value()
    if safe._logical_elements != chosen.total_items:
        return TensorResult[DataflowExecutionEvidence](TensorStatus.SIZE_MISMATCH)
    var final_domain = safe._memory[safe._output_slot].domain
    if chosen.output_device == 1:
        if final_domain != 1:
            return TensorResult[DataflowExecutionEvidence](TensorStatus.DEVICE_MISMATCH)
    elif final_domain == 1:
        return TensorResult[DataflowExecutionEvidence](TensorStatus.DEVICE_MISMATCH)

    var cpu_min = safe._logical_elements
    var cpu_max = 0
    var gpu_min = safe._logical_elements
    var gpu_max = 0
    var cpu_written = 0
    var gpu_written = 0
    var cpu_compute = 0
    var gpu_compute = 0
    var host_prepare = 0
    var transfer_count = 0
    var gpu_dma_count = 0
    var h2d_count = 0
    var d2h_count = 0
    var h2d_elements = 0
    var d2h_elements = 0
    var transferred_elements = 0

    for i in range(safe._count):
        var item = safe._nodes[i]
        if item.kind == 0:
            # Legacy single-variant CPU-only plans may not advertise a
            # different executable specialization from any of their CPU
            # compute nodes. Hybrid CPU assistance is a separately typed
            # lowering and its CPU opcode need not share the GPU variant.
            comptime if not per_node_artifacts:
                if (chosen.placement == ExecutionPlacement.CPU_ONLY
                    and item.aot_variant != chosen.aot_variant):
                    return TensorResult[DataflowExecutionEvidence](
                        TensorStatus.INVALID_LAYOUT
                    )
            cpu_compute += 1
        elif item.kind == 1:
            # Legacy single-artifact callers must retain their ONE variant.
            # A heterogeneous plan is permitted only through the explicit
            # per-node AOT binding boundary, where every compute node has
            # its own authenticated (elsewhere) symbol/ABI/target selection.
            comptime if not per_node_artifacts:
                if item.aot_variant != chosen.aot_variant:
                    return TensorResult[DataflowExecutionEvidence](TensorStatus.INVALID_LAYOUT)
            gpu_compute += 1
        elif item.kind == 5:
            host_prepare += 1
        if item.kind == 2 or item.kind == 3 or item.kind == 4 or item.kind == 7:
            transfer_count += 1
            if item.kind == 2:
                h2d_count += 1
            if item.kind == 3:
                d2h_count += 1
            # Host-to-pinned preparation (kind 7) is a CPU memory copy;
            # ExecutionPlan.transfer_bytes is the explicit GPU DMA budget.
            # Do not accidentally charge the host preparation run as H2D.
            if item.kind != 7:
                gpu_dma_count += 1
                var length = item.write.end - item.write.begin
                # Accumulate in ELEMENTS so a final exact x4 can be checked.
                if length > Int.MAX // 4 - transferred_elements:
                    return TensorResult[DataflowExecutionEvidence](TensorStatus.OVERFLOW)
                transferred_elements += length
                if item.kind == 2:
                    if length > Int.MAX // 4 - h2d_elements:
                        return TensorResult[DataflowExecutionEvidence](TensorStatus.OVERFLOW)
                    h2d_elements += length
                elif item.kind == 3:
                    if length > Int.MAX // 4 - d2h_elements:
                        return TensorResult[DataflowExecutionEvidence](TensorStatus.OVERFLOW)
                    d2h_elements += length
        if item.write.slot == safe._output_slot:
            # A preparation-only copy is never an authoritative public
            # output producer; use the real CPU compute/copy or final join.
            if item.kind == 5:
                return TensorResult[DataflowExecutionEvidence](TensorStatus.UNSUPPORTED)
            # An output must be produced by a stage with the correct native
            # result semantics, not an unrelated copy or host preparation.
            if (chosen.placement == ExecutionPlacement.HYBRID_SPLIT
                or chosen.placement == ExecutionPlacement.HYBRID_CPU_ASSIST):
                if item.kind != 0 and item.kind != 3 and item.kind != 6:
                    return TensorResult[DataflowExecutionEvidence](TensorStatus.UNSUPPORTED)
            elif chosen.placement == ExecutionPlacement.HYBRID_PIPELINE:
                if item.kind != 3 and item.kind != 6:
                    return TensorResult[DataflowExecutionEvidence](TensorStatus.UNSUPPORTED)
            var written = item.write.end - item.write.begin
            if (item.kind == 0 or item.kind == 5
                or (item.kind == 6 and chosen.placement == ExecutionPlacement.CPU_ONLY)):
                # A host join/copy is ordinary CPU work in CPU_ONLY plans.
                # In Hybrid plans the same host join is the completion of an
                # accelerator-dependent result, not a CPU compute partition.
                # Misclassifying CPU_ONLY joins used to reject a valid plan
                # although its CPU executor has a checked host-copy lowering.
                cpu_min = min(cpu_min, item.write.begin)
                cpu_max = max(cpu_max, item.write.end)
                cpu_written += written
            else:
                # DMA, GPU stages and GPU-dependent Hybrid host joins are
                # attributed to the accelerator portion of the final output.
                gpu_min = min(gpu_min, item.write.begin)
                gpu_max = max(gpu_max, item.write.end)
                gpu_written += written

    # The canonical DAG already establishes a disjoint, full output cover.
    # Consequently, count + min/max is also an exact contiguous partition
    # proof (not merely an approximate bounding-box comparison).
    if cpu_written > 0 and cpu_written != cpu_max - cpu_min:
        return TensorResult[DataflowExecutionEvidence](TensorStatus.HYBRID_PARTITION_INVALID)
    if gpu_written > 0 and gpu_written != gpu_max - gpu_min:
        return TensorResult[DataflowExecutionEvidence](TensorStatus.HYBRID_PARTITION_INVALID)
    if cpu_written + gpu_written != safe._logical_elements:
        return TensorResult[DataflowExecutionEvidence](TensorStatus.INVALID_LAYOUT)
    var transfer_bytes = transferred_elements * 4
    if transfer_bytes > chosen.transfer_bytes:
        return TensorResult[DataflowExecutionEvidence](TensorStatus.SIZE_MISMATCH)

    if chosen.placement == ExecutionPlacement.CPU_ONLY:
        if (gpu_compute != 0 or transfer_count != 0 or gpu_written != 0
            or cpu_written != chosen.total_items):
            return TensorResult[DataflowExecutionEvidence](TensorStatus.DEVICE_MISMATCH)
        if chosen.total_items > 0 and (cpu_min != 0 or cpu_max != chosen.total_items):
            return TensorResult[DataflowExecutionEvidence](TensorStatus.HYBRID_PARTITION_INVALID)
    elif chosen.placement == ExecutionPlacement.GPU_ONLY:
        if cpu_compute != 0 or cpu_written != 0 or gpu_written != chosen.total_items:
            return TensorResult[DataflowExecutionEvidence](TensorStatus.DEVICE_MISMATCH)
        if chosen.total_items > 0:
            if gpu_min != 0 or gpu_max != chosen.total_items:
                return TensorResult[DataflowExecutionEvidence](TensorStatus.HYBRID_PARTITION_INVALID)
            # Host-only staging does not establish accelerator execution.
            # A GPU mapping needs GPU computation or an actual GPU DMA.
            if gpu_compute == 0 and gpu_dma_count == 0:
                return TensorResult[DataflowExecutionEvidence](TensorStatus.UNSUPPORTED)
    elif (chosen.placement == ExecutionPlacement.HYBRID_SPLIT
          or chosen.placement == ExecutionPlacement.HYBRID_CPU_ASSIST):
        # Work partitions are proven from the same output slot regardless of
        # primitive arity. A unary operation needs one input, a binary needs
        # two, and a resident GEMM input may need no H2D at all. The sealed
        # producer/consumer graph establishes initialized device input reads;
        # the owner-bound native ABI validates its own transfer requirements.
        # For this HOST-result split, an explicit D2H stage is mandatory.
        if (cpu_compute < 1 or gpu_compute < 1 or d2h_count < 1
            or cpu_written == 0 or gpu_written == 0):
            return TensorResult[DataflowExecutionEvidence](TensorStatus.UNSUPPORTED)
        if (cpu_min != chosen.cpu_begin or cpu_max != chosen.cpu_end
            or gpu_min != chosen.gpu_begin or gpu_max != chosen.gpu_end):
            return TensorResult[DataflowExecutionEvidence](TensorStatus.HYBRID_PARTITION_INVALID)
        if (cpu_written != chosen.cpu_end - chosen.cpu_begin
            or gpu_written != chosen.gpu_end - chosen.gpu_begin):
            return TensorResult[DataflowExecutionEvidence](TensorStatus.HYBRID_PARTITION_INVALID)
        # Every GPU-owned result element must pass through an explicit D2H
        # producer before host publication. Input byte requirements are NOT
        # proportional to output length for GEMM, reduction or reused resident
        # tensors. No binary-only H2D lower bound belongs in this shared IR.
        # Binary/unary/matrix operation binders check their exact ABI transfer
        # budgets separately; the graph validator checks declared stage bytes
        # against the immutable plan above, with no hidden transfers.
        if d2h_elements < gpu_written:
            return TensorResult[DataflowExecutionEvidence](TensorStatus.SIZE_MISMATCH)
    elif chosen.placement == ExecutionPlacement.HYBRID_PIPELINE:
        # In the current pipeline ABI the host prepares the NEXT chunk, but
        # never computes a disjoint arithmetic output partition.
        if (cpu_compute != 0 or cpu_written != 0 or gpu_written != chosen.total_items
            or gpu_compute < 2 or host_prepare < 1 or h2d_count < 1 or d2h_count < 1):
            return TensorResult[DataflowExecutionEvidence](TensorStatus.UNSUPPORTED)
        if gpu_min != 0 or gpu_max != chosen.total_items:
            return TensorResult[DataflowExecutionEvidence](TensorStatus.HYBRID_PARTITION_INVALID)
        # Some stages may reuse an earlier device-resident operand or produce
        # an intermediate with a different element count. The host-result D2H
        # coverage is universal; binary two-upload accounting is specific to
        # the checked host-binary pipeline adapter, not to UASDEA itself.
        if d2h_elements < chosen.total_items:
            return TensorResult[DataflowExecutionEvidence](TensorStatus.SIZE_MISMATCH)
    else:
        return TensorResult[DataflowExecutionEvidence](TensorStatus.INVALID_LAYOUT)

    return TensorResult[DataflowExecutionEvidence](DataflowExecutionEvidence(
        cpu_written, gpu_written, cpu_compute, gpu_compute, host_prepare,
        transfer_count, transfer_bytes, final_domain,
    ))


def validate_static_dataflow_execution_plan[nodes: Int, slots: Int](
    graph: SealedStaticDataflowPlan[nodes, slots], placement: ExecutionPlan,
) -> TensorResult[DataflowExecutionEvidence]:
    # Backwards-compatible single-artifact lowering remains fail-closed.
    return _validate_static_dataflow_execution_plan[False, nodes, slots](graph, placement)


def validate_static_dataflow_execution_plan_per_node_artifacts[
    nodes: Int, slots: Int,
](
    graph: SealedStaticDataflowPlan[nodes, slots], placement: ExecutionPlan,
) -> TensorResult[DataflowExecutionEvidence]:
    # Necessary, not sufficient: the caller MUST also provide an exact
    # checked per-node catalog binding before lowering any compute node.
    # This is NOT an entry point for executing unbound native functions.
    return _validate_static_dataflow_execution_plan[True, nodes, slots](graph, placement)


# Bridge the existing native host-binary stage schedule into the SAME bounded
# FP32 dataflow IR. A validated stage schedule by itself does not establish
# graph/ExecutionPlan agreement; the caller must also call the canonical
# validate_static_dataflow_execution_plan() on this returned owned snapshot.
#
# The current native host-binary ABI gives allocation IDs 0/1 to the two
# initial host sources, 2 to the host result, and 3+ to unique staging/device
# owners. IDs are allocation identities, not interchangeable device names.
# This converter deliberately rejects schedules with any other numbering or
# operation family rather than interpreting an arbitrary stage as a kernel.
def _bound_access_to_dataflow_region(access: ExecutionAccess) -> DataflowRegion:
    if not access.active():
        return DataflowRegion.none()
    return DataflowRegion(access.buffer_id, access.begin, access.end)


def canonical_host_binary_dataflow_from_schedule[
    stages: Int, buffers: Int,
](
    schedule: BoundStageSchedule[stages, buffers],
    plan: ExecutionPlan, output_elements: Int, operation: Int,
) -> TensorResult[SealedStaticDataflowPlan[stages, buffers]]:
    comptime assert stages >= 1 and stages <= 16, "bounded canonical stage capacity"
    comptime assert buffers >= 3 and buffers <= 32, "bounded canonical buffer capacity"
    if operation < 0 or operation > 5:
        return TensorResult[SealedStaticDataflowPlan[stages, buffers]](
            TensorStatus.UNSUPPORTED
        )
    var validated_plan = plan.checked_snapshot()
    if not validated_plan.is_ok():
        return TensorResult[SealedStaticDataflowPlan[stages, buffers]](
            validated_plan.status
        )
    var chosen = validated_plan.payload.value()
    var validated_schedule = schedule.checked_snapshot()
    if not validated_schedule.is_ok():
        return TensorResult[SealedStaticDataflowPlan[stages, buffers]](
            validated_schedule.status
        )
    var native = validated_schedule.payload.value()
    if output_elements <= 0 or output_elements > chosen.total_items:
        return TensorResult[SealedStaticDataflowPlan[stages, buffers]](
            TensorStatus.SIZE_MISMATCH
        )
    if native._schedule.count < 1 or native._schedule.count > stages:
        return TensorResult[SealedStaticDataflowPlan[stages, buffers]](
            TensorStatus.INVALID_LAYOUT
        )
    if native._schedule.total_elements != chosen.total_items:
        return TensorResult[SealedStaticDataflowPlan[stages, buffers]](
            TensorStatus.SIZE_MISMATCH
        )
    var empty_slot = DataflowMemorySlot(0, 0, 0, False, False)
    var memory = Array[DataflowMemorySlot, buffers](fill=empty_slot)
    for i in range(buffers):
        var entry = native._bounds[i]
        # The native ABI has a dense table, one distinct allocation per ID.
        # A missing entry, foreign domain, or permuted ID is not a binding.
        if entry.buffer_id != i or entry.space < 0 or entry.space > 2:
            return TensorResult[SealedStaticDataflowPlan[stages, buffers]](
                TensorStatus.INVALID_LAYOUT
            )
        if i < 3 and entry.space != 0:
            return TensorResult[SealedStaticDataflowPlan[stages, buffers]](
                TensorStatus.DEVICE_MISMATCH
            )
        if entry.extent < 0 or entry.extent > Int.MAX // 4:
            return TensorResult[SealedStaticDataflowPlan[stages, buffers]](
                TensorStatus.OVERFLOW
            )
        memory[i] = DataflowMemorySlot(
            i, entry.space, entry.extent, i < 2, i >= 2
        )
    if memory[2].capacity_elements < output_elements:
        return TensorResult[SealedStaticDataflowPlan[stages, buffers]](
            TensorStatus.OUT_OF_BOUNDS
        )
    var none = DataflowRegion.none()
    var unused = DataflowNode(0, 5, 0, 0, 0, none, none, none, none, 1, 0, 0, 0, 0, 0, 0, 0, none, 0, Float32(0.0))
    var graph = Array[DataflowNode, stages](fill=unused)
    for i in range(native._schedule.count):
        var stage = native._schedule.stages[i]
        var family = -1
        var primitive = 0
        if stage.kind == ExecutionStageKind.CPU_PREPARE:
            family = 5
        elif stage.kind == ExecutionStageKind.H2D:
            family = 2
        elif stage.kind == ExecutionStageKind.GPU_KERNEL:
            family = 1
            primitive = 2
        elif stage.kind == ExecutionStageKind.D2H:
            family = 3
        elif stage.kind == ExecutionStageKind.CPU_COMPUTE:
            family = 0
            primitive = 2
        elif stage.kind == ExecutionStageKind.HOST_JOIN:
            family = 6
        else:
            return TensorResult[SealedStaticDataflowPlan[stages, buffers]](
                TensorStatus.UNSUPPORTED
            )
        # One flat FP32 binary ABI: all stages operate on 1-D segments and
        # every binary compute node uses the single selected offline variant.
        var opcode = 0
        if primitive == 2:
            opcode = operation
        graph[i] = DataflowNode(
            stage.identifier, family, primitive, opcode, stage.depends_on,
            _bound_access_to_dataflow_region(stage.read_a),
            _bound_access_to_dataflow_region(stage.read_b),
            DataflowRegion.none(),
            _bound_access_to_dataflow_region(stage.write),
            1, 0, chosen.aot_variant, 0, 0, 0, 0, 0, none, 0, Float32(0.0),
        )
    return SealedStaticDataflowPlan[stages, buffers].checked(
        graph, memory, native._schedule.count, output_elements, 2, buffers,
    )


def validate_host_binary_native_schedule_agreement[
    stages: Int, buffers: Int,
](
    schedule: BoundStageSchedule[stages, buffers], plan: ExecutionPlan,
    operation: Int,
) -> TensorStatus:
    # The native split/assist schedule is COMPLETE; the two-stage pipeline
    # schedule instead describes a reusable prefix, not the full workload.
    # Never claim graph-wide agreement for a partial pipeline window.
    var selected = plan.checked_snapshot()
    if not selected.is_ok():
        return selected.status
    var fixed = selected.payload.value()
    if (fixed.placement != ExecutionPlacement.HYBRID_SPLIT
        and fixed.placement != ExecutionPlacement.HYBRID_CPU_ASSIST):
        return TensorStatus.UNSUPPORTED
    var converted = canonical_host_binary_dataflow_from_schedule[
        stages, buffers,
    ](schedule, fixed, fixed.total_items, operation)
    if not converted.is_ok():
        return converted.status
    var agreed = validate_static_dataflow_execution_plan[
        stages, buffers,
    ](converted.payload.value(), fixed)
    return agreed.status


# A complete, immutable-by-construction logical replay range for EVERY
# pipeline chunk, including the final short tail. The two fixed GPU slots
# are reused only for full windows; the tail uses a third, exactly-sized
# native owner. This is the same ExecutionPlan, not another scheduler or IR.
@fieldwise_init
struct CheckedHostBinaryPipelineWindow(Copyable):
    var begin: Int
    var elements: Int
    var slot: Int
    var reuses_slot: Bool
    var tail: Bool


def checked_host_binary_pipeline_window(
    whole_plan: ExecutionPlan, capacity: Int, index: Int,
) -> TensorResult[CheckedHostBinaryPipelineWindow]:
    var proof = whole_plan.checked_snapshot()
    if not proof.is_ok():
        return TensorResult[CheckedHostBinaryPipelineWindow](proof.status)
    var selected = proof.payload.value()
    if selected.placement != ExecutionPlacement.HYBRID_PIPELINE:
        return TensorResult[CheckedHostBinaryPipelineWindow](
            TensorStatus.DEVICE_MISMATCH
        )
    # The canonical two-window template requires at least two FULL chunks.
    # The count is already bounded to representable FP32 bytes by the plan.
    if (capacity <= 0 or capacity > selected.total_items // 2
        or index < 0):
        return TensorResult[CheckedHostBinaryPipelineWindow](
            TensorStatus.NOT_REPRESENTABLE
        )
    # Avoid a multiplication by an untrusted index before proving bounds.
    if index > (selected.total_items - 1) // capacity:
        return TensorResult[CheckedHostBinaryPipelineWindow](
            TensorStatus.OUT_OF_BOUNDS
        )
    var begin = index * capacity
    var length = min(capacity, selected.total_items - begin)
    if length <= 0:
        return TensorResult[CheckedHostBinaryPipelineWindow](
            TensorStatus.INVALID_LAYOUT
        )
    var short_tail = length < capacity
    var slot = index % 2
    if short_tail:
        # The native executor must allocate a separate exact-size owner.
        slot = 2
    return TensorResult[CheckedHostBinaryPipelineWindow](
        CheckedHostBinaryPipelineWindow(
            begin, length, slot, index >= 2 and not short_tail, short_tail,
        )
    )


# A pipeline reuses an AOT *window template*. Its 14-stage schedule describes
# precisely two complete windows, not every later queued chunk. Validate that
# prefix against an ExecutionPlan of the same FP32 shape/placement semantics.
# This is a necessary reusable-template proof; ownership/reuse fences for the
# repeated windows are still verified at each native stream/event boundary.
def validate_host_binary_pipeline_window_agreement[
    stages: Int, buffers: Int,
](
    schedule: BoundStageSchedule[stages, buffers],
    whole_plan: ExecutionPlan, window_elements: Int, operation: Int,
) -> TensorStatus:
    var selected = whole_plan.checked_snapshot()
    if not selected.is_ok():
        return selected.status
    var whole = selected.payload.value()
    if whole.placement != ExecutionPlacement.HYBRID_PIPELINE:
        return TensorStatus.DEVICE_MISMATCH
    if (window_elements < 2 or window_elements > whole.total_items
        or window_elements > Int.MAX // 12):
        return TensorStatus.SIZE_MISMATCH
    var prefix = ExecutionPlan.checked(
        ExecutionPlacement.HYBRID_PIPELINE,
        window_elements, 0, 0, 0, window_elements,
        window_elements * 12, whole.aot_variant, 2, 0,
    )
    if not prefix.is_ok():
        return prefix.status
    var converted = canonical_host_binary_dataflow_from_schedule[
        stages, buffers,
    ](schedule, whole, window_elements, operation)
    if not converted.is_ok():
        return converted.status
    var matched = validate_static_dataflow_execution_plan[
        stages, buffers,
    ](converted.payload.value(), prefix.payload.value())
    return matched.status
