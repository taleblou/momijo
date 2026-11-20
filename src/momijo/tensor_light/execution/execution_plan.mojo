# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR EXECUTION / PREPARED EXECUTION PLANS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Execution Planning and AOT
# Module:        momijo.tensor_light.execution.execution_plan
# Source:        src/momijo/tensor_light/execution/execution_plan.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines prepared execution plans within checked AOT execution planning for
#   the FP32 Light implementation.
#
#   Public declarations include `ExecutionPlacement`, `ExecutionPlan`,
#   `StaticAOTExecutionPlan`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Select or prepare validated launch, placement, and execution-plan
#     policies; do not claim that policy selection performs device work.
#   - Expose module-level declarations including `ExecutionPlacement`,
#     `ExecutionPlan`, `StaticAOTExecutionPlan`.
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

# UASDEA: one immutable-by-contract dataflow plan, five target mappings.
# Lowering and transfer decisions are completed before this value is executed.
from ..internal.status import TensorStatus, TensorResult

@fieldwise_init
struct ExecutionPlacement(Copyable, Equatable):
    var code: Int
    comptime CPU_ONLY = Self(0)
    comptime GPU_ONLY = Self(1)
    comptime HYBRID_SPLIT = Self(2)
    comptime HYBRID_PIPELINE = Self(3)
    comptime HYBRID_CPU_ASSIST = Self(4)

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

struct ExecutionPlan(Copyable):
    var placement: ExecutionPlacement
    var total_items: Int
    var cpu_begin: Int
    var cpu_end: Int
    var gpu_begin: Int
    var gpu_end: Int
    var transfer_bytes: Int
    var aot_variant: Int
    # Mapping-level completion/join obligations (0 CPU, 1 GPU, 2 Split/Assist).
    # This is NOT the number of edges in CheckedStageSchedule.
    var dependency_count: Int
    var output_device: Int

    def __init__(out self, placement: ExecutionPlacement, total_items: Int, cpu_begin: Int, cpu_end: Int, gpu_begin: Int, gpu_end: Int, transfer_bytes: Int, aot_variant: Int, dependency_count: Int, output_device: Int):
        self.placement = placement.copy()
        self.total_items = total_items
        self.cpu_begin = cpu_begin
        self.cpu_end = cpu_end
        self.gpu_begin = gpu_begin
        self.gpu_end = gpu_end
        self.transfer_bytes = transfer_bytes
        self.aot_variant = aot_variant
        self.dependency_count = dependency_count
        self.output_device = output_device

    def __init__(out self, *, copy: Self):
        self.placement = copy.placement.copy()
        self.total_items = copy.total_items
        self.cpu_begin = copy.cpu_begin
        self.cpu_end = copy.cpu_end
        self.gpu_begin = copy.gpu_begin
        self.gpu_end = copy.gpu_end
        self.transfer_bytes = copy.transfer_bytes
        self.aot_variant = copy.aot_variant
        self.dependency_count = copy.dependency_count
        self.output_device = copy.output_device

    @staticmethod
    def checked(placement: ExecutionPlacement, count: Int, cpu_begin: Int, cpu_end: Int, gpu_begin: Int, gpu_end: Int, transfer_bytes: Int, variant: Int, dependency_count: Int, output_device: Int) -> TensorResult[Self]:
        if count < 0 or transfer_bytes < 0 or variant < 0 or dependency_count < 0:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        # Every plan in this Core addresses four-byte Float32 elements. A
        # forged CPU-only plan with more elements than a representable FP32
        # byte range must not bypass the shape/allocation overflow contract.
        if count > Int.MAX // 4:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        if placement.code < 0 or placement.code > 4:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        # FP32 transfers must contain whole elements. Output residency is
        # explicit: 0 means host, 1 means target device.
        if transfer_bytes % 4 != 0 or output_device < 0 or output_device > 1:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if cpu_begin < 0 or cpu_end < cpu_begin or cpu_end > count:
            return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
        if gpu_begin < 0 or gpu_end < gpu_begin or gpu_end > count:
            return TensorResult[Self](TensorStatus.OUT_OF_BOUNDS)
        if placement == materialize[ExecutionPlacement.CPU_ONLY]():
            if cpu_begin != 0 or cpu_end != count or gpu_begin != 0 or gpu_end != 0:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            # An ordinary CPU-only lowering cannot produce device-resident
            # output or schedule an implicit accelerator transfer.
            if transfer_bytes != 0 or output_device != 0 or dependency_count != 0:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        elif placement == materialize[ExecutionPlacement.GPU_ONLY]():
            # Empty GPU-resident tensor metadata requires no kernel enqueue,
            # transfer, or artificial completion event. The execution path
            # must skip native launch when total_items == 0.
            if count == 0:
                if transfer_bytes != 0 or dependency_count != 0 or output_device != 1:
                    return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            elif dependency_count != 1:
                # One completion event is mandatory for real GPU work.
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if gpu_begin != 0 or gpu_end != count or cpu_begin != 0 or cpu_end != 0:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            # Host-resident GPU results require an explicit D2H transfer.
            # Device-resident outputs do not require that transfer.
            if count > 0 and output_device == 0 and (count > Int.MAX // 4 or transfer_bytes < count * 4):
                return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
        elif placement == materialize[ExecutionPlacement.HYBRID_SPLIT]() or placement == materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]():
            # Host-result split mappings share one strict row/item partition,
            # disjoint output intervals and an event-joined GPU segment.
            # The common 8-byte/item bound is a conservative FLOOR for a
            # unary host-result ABI. Each lowering validates its own actual
            # transfer budget before any native work is enqueued.
            if dependency_count != 2:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            # The currently implemented split joins to a HOST output only.
            # Device-resident split output needs its own proven CPU-to-device
            # transfer and joined owner/event lowering, which is not present.
            if output_device != 0:
                return TensorResult[Self](TensorStatus.UNSUPPORTED)
            # A shared static plan cannot assume the BINARY operand count.
            # The least expensive out-of-place host-result FP32 primitive is
            # unary: one 4-byte H2D input and one 4-byte D2H output per GPU
            # item. Its operation-specific checked boundary proves 8 bytes;
            # binary, reduction and matrix lowerings independently prove their
            # exact, usually larger transfer budgets before enqueue.
            var gpu_items = gpu_end - gpu_begin
            if gpu_items > Int.MAX // 8:
                return TensorResult[Self](TensorStatus.OVERFLOW)
            if transfer_bytes < gpu_items * 8:
                return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
            # Zero-work lanes are not hybrid; select the corresponding single backend.
            if cpu_end == cpu_begin or gpu_end == gpu_begin:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            var cpu_first = cpu_begin == 0 and cpu_end == gpu_begin and gpu_end == count
            var gpu_first = gpu_begin == 0 and gpu_end == cpu_begin and cpu_end == count
            if placement == materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]():
                # CPU assist is strictly a CPU tail paired with a GPU bulk
                # prefix. Larger CPU partitions belong to HYBRID_SPLIT.
                if not gpu_first or cpu_end - cpu_begin > gpu_end - gpu_begin:
                    return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            elif not cpu_first and not gpu_first:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        elif placement == materialize[ExecutionPlacement.HYBRID_PIPELINE]():
            # Narrow, host-result, fixed-capacity FP32 binary double-buffered
            # pipeline. Its CPU work is PREPARATION, not disjoint arithmetic:
            # the GPU computes all elements. Buffer reuse is event-fenced.
            # The AOT lowering additionally checks count/chunk_capacity and
            # constructs a two-slot static stage DAG before any enqueue.
            if output_device != 0 or dependency_count != 2:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if count <= 1 or cpu_begin != 0 or cpu_end != 0:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if gpu_begin != 0 or gpu_end != count:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
            if count > Int.MAX // 12:
                return TensorResult[Self](TensorStatus.OVERFLOW)
            if transfer_bytes < count * 12:
                return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
        else:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        return TensorResult[Self](Self(placement, count, cpu_begin, cpu_end, gpu_begin, gpu_end, transfer_bytes, variant, dependency_count, output_device))

    def checked_snapshot(self) -> TensorResult[Self]:
        # The returned value owns a distinct copy of the fully checked plan.
        # Lowering must use this snapshot, never re-read caller-owned mutable
        # fields between capability validation and a native kernel launch.
        # A normal Mojo immutable borrow prevents concurrent mutation during
        # the call; unsafe external aliases are outside this contract.
        return Self.checked(
            self.placement, self.total_items, self.cpu_begin, self.cpu_end,
            self.gpu_begin, self.gpu_end, self.transfer_bytes, self.aot_variant,
            self.dependency_count, self.output_device,
        )

    def checked_host_binary_snapshot(self) -> TensorResult[Self]:
        # A generic GPU plan proves only a device/host completion policy. An
        # OUT-OF-PLACE host-result FP32 binary operation has a stricter ABI:
        # two H2D operands and one D2H result, all in four-byte elements.
        # Bind those transfer obligations before the first native allocation.
        var proof = self.checked_snapshot()
        if not proof.is_ok():
            return proof^
        var selected = proof.payload.value().copy()
        if selected.output_device != 0:
            return TensorResult[Self](TensorStatus.DEVICE_MISMATCH)
        if selected.placement == materialize[ExecutionPlacement.CPU_ONLY]():
            return proof^
        if selected.total_items <= 0:
            # Host-result zero work should select the empty CPU path; the
            # GPU-only empty plan is device-result by construction.
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if selected.placement == materialize[ExecutionPlacement.GPU_ONLY]():
            if selected.total_items > Int.MAX // 12:
                return TensorResult[Self](TensorStatus.OVERFLOW)
            if selected.transfer_bytes < selected.total_items * 12:
                return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
            return proof^
        if (selected.placement == materialize[ExecutionPlacement.HYBRID_SPLIT]()
            or selected.placement == materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]()
            or selected.placement == materialize[ExecutionPlacement.HYBRID_PIPELINE]()):
            # A generic plan only proves the 8-byte unary floor. The binary
            # ABI has TWO inputs and one output; enforce its 12-byte budget
            # here, after disjoint work regions and dependencies are proved.
            var gpu_items = selected.gpu_end - selected.gpu_begin
            if gpu_items > Int.MAX // 12:
                return TensorResult[Self](TensorStatus.OVERFLOW)
            if selected.transfer_bytes < gpu_items * 12:
                return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
            return proof^
        return TensorResult[Self](TensorStatus.UNSUPPORTED)

    def checked_host_unary_snapshot(self) -> TensorResult[Self]:
        # One input and one host output. Host-only, GPU-only and disjoint
        # CPU/GPU splits all use the SAME immutable ExecutionPlan semantics.
        # This ABI is separate from the binary 12-byte-per-item contract.
        var proof = self.checked_snapshot()
        if not proof.is_ok():
            return proof^
        var selected = proof.payload.value().copy()
        if selected.output_device != 0:
            return TensorResult[Self](TensorStatus.DEVICE_MISMATCH)
        # The immutable plan identifies the exact unary opcode family even
        # when the offline selector chooses CPU_ONLY. A caller may not
        # substitute an operation by passing a different native function.
        if selected.aot_variant < 20 or selected.aot_variant > 28:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        if selected.placement == materialize[ExecutionPlacement.CPU_ONLY]():
            return proof^
        if selected.placement == materialize[ExecutionPlacement.GPU_ONLY]():
            if selected.total_items <= 0 or selected.total_items > Int.MAX // 8:
                return TensorResult[Self](TensorStatus.OVERFLOW)
            if selected.transfer_bytes < selected.total_items * 8:
                return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
            return proof^
        if (selected.placement == materialize[ExecutionPlacement.HYBRID_SPLIT]()
            or selected.placement == materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]()):
            var gpu_items = selected.gpu_end - selected.gpu_begin
            if gpu_items <= 0 or gpu_items > Int.MAX // 8:
                return TensorResult[Self](TensorStatus.HYBRID_PARTITION_INVALID)
            if selected.transfer_bytes < gpu_items * 8:
                return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
            return proof^
        # Unary pipeline is not falsely advertised as a binary stage DAG.
        return TensorResult[Self](TensorStatus.UNSUPPORTED)

    def checked_host_reduce_rows_split_snapshot(
        self, columns: Int,
    ) -> TensorResult[Self]:
        # Same canonical ExecutionPlan: the work-item unit is one OUTPUT row.
        # A two-dimensional FP32 input row has `columns` elements, requiring
        # (columns+1)*4 bytes H2D+D2H per GPU-owned output row.
        var proof = self.checked_snapshot()
        if not proof.is_ok():
            return proof^
        var selected = proof.payload.value().copy()
        if (selected.placement != materialize[ExecutionPlacement.HYBRID_SPLIT]()
            and selected.placement != materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]()):
            return TensorResult[Self](TensorStatus.DEVICE_MISMATCH)
        if selected.output_device != 0 or selected.aot_variant < 30 or selected.aot_variant > 34:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        var rows = selected.total_items
        if rows <= 1 or columns <= 0:
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        if rows > Int.MAX // 4 or columns > (Int.MAX // 4) // rows:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        var gpu_rows = selected.gpu_end - selected.gpu_begin
        if gpu_rows <= 0:
            return TensorResult[Self](TensorStatus.HYBRID_PARTITION_INVALID)
        # No binary-only 3-element minimum: a single-column row reduction
        # legitimately transfers one FP32 input and one FP32 output.
        var transfer_elements_per_row = columns + 1
        if gpu_rows > (Int.MAX // 4) // transfer_elements_per_row:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        var needed_bytes = gpu_rows * transfer_elements_per_row * 4
        if selected.transfer_bytes < needed_bytes:
            return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
        return proof^

    def checked_host_gemm_rows_split_snapshot(
        self, k: Int, n: Int,
    ) -> TensorResult[Self]:
        # Fixed rank-two C-order FP32 GEMM: work unit = one OUTPUT row.
        # The GPU receives A[gpu_rows,K], the ENTIRE shared B[K,N], then
        # returns C[gpu_rows,N]. This is NOT the binary 12-byte/item ABI.
        var proof = self.checked_snapshot()
        if not proof.is_ok():
            return proof^
        var selected = proof.payload.value().copy()
        if (selected.placement != materialize[ExecutionPlacement.HYBRID_SPLIT]()
            and selected.placement != materialize[ExecutionPlacement.HYBRID_CPU_ASSIST]()):
            return TensorResult[Self](TensorStatus.DEVICE_MISMATCH)
        if selected.output_device != 0 or selected.aot_variant != 14:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        var m = selected.total_items
        if m <= 1 or k <= 0 or n <= 0:
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var maximum_elements = Int.MAX // 4
        if m > maximum_elements // k or k > maximum_elements // n:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        if m > maximum_elements // n:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        var gpu_rows = selected.gpu_end - selected.gpu_begin
        if gpu_rows <= 0 or gpu_rows > maximum_elements // k:
            return TensorResult[Self](TensorStatus.HYBRID_PARTITION_INVALID)
        if gpu_rows > maximum_elements // n:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        var upload_a = gpu_rows * k
        var upload_b = k * n
        var download_c = gpu_rows * n
        if upload_a > maximum_elements - upload_b:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        var combined = upload_a + upload_b
        if download_c > maximum_elements - combined:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        # Input A slice, shared B and output C are the exact GEMM traffic.
        # Their verified byte count is not the binary elementwise ABI floor.
        var exact_bytes = (combined + download_c) * 4
        if selected.transfer_bytes < exact_bytes:
            return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
        return proof^

    def checked_native_device_snapshot(self) -> TensorResult[Self]:
        # One canonical, owner-independent contract for a PRE-RESIDENT GPU
        # kernel. These adapters neither stage operands nor copy output back
        # to the host, so a valid generic GPU plan is not sufficient.
        # The caller must still prove buffer shape, ownership and the AOT ABI.
        var snapshot = self.checked_snapshot()
        if not snapshot.is_ok():
            return snapshot^
        var fixed = snapshot.payload.value().copy()
        if fixed.placement != materialize[ExecutionPlacement.GPU_ONLY]() or fixed.output_device != 1:
            return TensorResult[Self](TensorStatus.DEVICE_MISMATCH)
        if fixed.transfer_bytes != 0:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        return snapshot^

    def checked_native_cpu_snapshot(self) -> TensorResult[Self]:
        # A pre-resident CPU-only primitive cannot silently migrate a device
        # buffer or consume an accelerator event. Bind the exact CPU mapping.
        var proof = self.checked_snapshot()
        if not proof.is_ok():
            return proof^
        var native = proof.payload.value().copy()
        if native.placement != materialize[ExecutionPlacement.CPU_ONLY]():
            return TensorResult[Self](TensorStatus.DEVICE_MISMATCH)
        if native.output_device != 0 or native.transfer_bytes != 0:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        return proof^

    def validated(self) -> TensorStatus:
        return self.checked_snapshot().status


# For a workload whose shape, placement, reuse and transfer obligations are
# statically known, the *specialization identity* has no runtime mutable
# fields at all. This is a distinct AOT-only form of the same dataflow plan,
# not a new executor or a dtype/device-specific tensor family. Runtime-shape
# callers retain ExecutionPlan.checked_snapshot() at their checked boundary.
# Every parameter below is a compile-time value baked into the native image.
struct StaticAOTExecutionPlan[
    placement_code: Int, total_items: Int,
    cpu_begin: Int, cpu_end: Int, gpu_begin: Int, gpu_end: Int,
    transfer_bytes: Int, aot_variant: Int, dependency_count: Int,
    output_device: Int,
](Copyable):
    comptime PLACEMENT = Self.placement_code
    comptime TOTAL_ITEMS = Self.total_items
    comptime VARIANT = Self.aot_variant

    def __init__(out self):
        # Compile-time validation of primitive value ranges. Cross-field
        # CPU/GPU/Hybrid legality still has the single authoritative checker
        # in ExecutionPlan.checked(), which can be constant-folded for this
        # specialization. No mutable plan data is stored in this struct.
        comptime assert Self.placement_code >= 0 and Self.placement_code <= 4, "invalid AOT placement"
        comptime assert Self.total_items >= 0 and Self.total_items <= Int.MAX // 4, "invalid AOT item count"
        comptime assert Self.cpu_begin >= 0 and Self.cpu_end >= Self.cpu_begin, "invalid AOT CPU range"
        comptime assert Self.gpu_begin >= 0 and Self.gpu_end >= Self.gpu_begin, "invalid AOT GPU range"
        comptime assert Self.transfer_bytes >= 0 and Self.transfer_bytes % 4 == 0, "invalid AOT transfer"
        comptime assert Self.aot_variant >= 0 and Self.dependency_count >= 0, "invalid AOT binding"
        comptime assert Self.output_device >= 0 and Self.output_device <= 1, "invalid AOT result device"
        comptime assert Self.cpu_end <= Self.total_items and Self.gpu_end <= Self.total_items, "AOT region outside logical domain"
        # The target mappings are lowered from the SAME static dataflow
        # contract. Reject an invalid fixed configuration at specialization
        # time rather than letting it become a runtime scheduling choice.
        comptime if Self.placement_code == 0:
            comptime assert Self.cpu_begin == 0 and Self.cpu_end == Self.total_items, "CPU-only AOT coverage"
            comptime assert Self.gpu_begin == 0 and Self.gpu_end == 0, "CPU-only AOT GPU region"
            comptime assert Self.transfer_bytes == 0 and Self.dependency_count == 0 and Self.output_device == 0, "CPU-only AOT ownership"
        elif Self.placement_code == 1:
            comptime assert Self.cpu_begin == 0 and Self.cpu_end == 0, "GPU-only AOT CPU region"
            comptime assert Self.gpu_begin == 0 and Self.gpu_end == Self.total_items, "GPU-only AOT coverage"
            comptime if Self.total_items == 0:
                comptime assert Self.transfer_bytes == 0 and Self.dependency_count == 0 and Self.output_device == 1, "empty GPU AOT plan"
            else:
                comptime assert Self.dependency_count == 1, "GPU AOT completion event"
                comptime if Self.output_device == 0:
                    comptime assert Self.transfer_bytes >= Self.total_items * 4, "GPU AOT host readback bytes"
        elif Self.placement_code == 2 or Self.placement_code == 4:
            comptime assert Self.output_device == 0 and Self.dependency_count == 2, "hybrid AOT ownership"
            comptime assert Self.cpu_end > Self.cpu_begin and Self.gpu_end > Self.gpu_begin, "hybrid AOT nonempty lanes"
            # The shared AOT plan has one FP32 source/readback minimum for
            # its host-result split. Operation-specific typed ABI binders
            # enforce binary (12 bytes/item), unary (8 bytes/item), reduction
            # and GEMM exact traffic independently before any native launch.
            comptime assert Self.gpu_end - Self.gpu_begin <= Int.MAX // 8, "hybrid AOT transfer overflow"
            comptime assert Self.transfer_bytes >= (Self.gpu_end - Self.gpu_begin) * 8, "hybrid AOT transfer bytes"
            comptime if Self.placement_code == 4:
                comptime assert Self.gpu_begin == 0 and Self.gpu_end == Self.cpu_begin and Self.cpu_end == Self.total_items, "CPU-assist AOT partition"
                comptime assert Self.cpu_end - Self.cpu_begin <= Self.gpu_end - Self.gpu_begin, "CPU-assist AOT tail"
            else:
                comptime assert (Self.cpu_begin == 0 and Self.cpu_end == Self.gpu_begin and Self.gpu_end == Self.total_items) or (Self.gpu_begin == 0 and Self.gpu_end == Self.cpu_begin and Self.cpu_end == Self.total_items), "split AOT partition"
        else:
            comptime assert Self.total_items > 1 and Self.total_items <= Int.MAX // 12, "pipeline AOT work domain"
            comptime assert Self.cpu_begin == 0 and Self.cpu_end == 0 and Self.gpu_begin == 0 and Self.gpu_end == Self.total_items, "pipeline AOT lanes"
            comptime assert Self.output_device == 0 and Self.dependency_count == 2, "pipeline AOT completion"
            comptime assert Self.transfer_bytes >= Self.total_items * 12, "pipeline AOT transfer bytes"

    def checked_execution_plan(self) -> TensorResult[ExecutionPlan]:
        # Materialize a fully validated value for an existing lowering ABI.
        # The caller must not mutate that ephemeral value after validation.
        return ExecutionPlan.checked(
            ExecutionPlacement(Self.placement_code), Self.total_items,
            Self.cpu_begin, Self.cpu_end, Self.gpu_begin, Self.gpu_end,
            Self.transfer_bytes, Self.aot_variant, Self.dependency_count,
            Self.output_device,
        )
