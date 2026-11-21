# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DISPATCH / DISPATCH POLICY
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Dispatch and Specialization
# Module:        momijo.tensor_light.dispatch.dispatch
# Source:        src/momijo/tensor_light/dispatch/dispatch.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines dispatch policy in the dispatch and specialization layer of the
#   FP32 Light tensor implementation.
#
#   Public declarations include `validate_dispatch_target`,
#   `validate_dataflow_dispatch_target`, `bind_finite_aot_strategy`,
#   `validate_finite_aot_dataflow_key`; preconditions and supported targets
#   are determined by their implementations.
#
# Responsibilities:
#   - Validate dispatch choices against target and artifact contracts.
#   - Preserve documented behavior for immutable plan metadata and trusted
#     dispatch preconditions.
#   - Provide source-declared interfaces such as `validate_dispatch_target`,
#     `validate_dataflow_dispatch_target`, `bind_finite_aot_strategy`,
#     `validate_finite_aot_dataflow_key`, `bind_finite_aot_dataflow_strategy`
#     and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate dispatch keys and trusted artifact identities before selection.
#   - Treat tuning metadata as hints, not proof of executable code.
#
# Performance Objectives:
#   - Prefer precomputed finite strategies over runtime trial-and-error.
#   - Avoid scheduling or kernel compilation in pure selection utilities.
#
# Maintainability:
#   - Keep release schemas and validation checks versioned together.
#   - Test negative cases for unbound or inconsistent specialization data.
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

# AOT dispatch consumes a prevalidated plan, no dynamic dtype registry.

from ..execution.execution_plan import ExecutionPlan, ExecutionPlacement
from ..internal.status import TensorStatus, TensorResult
from ..execution.static_dataflow import SealedStaticDataflowPlan, validate_static_dataflow_execution_plan, validate_static_dataflow_execution_plan_per_node_artifacts
from .dispatch_key import DispatchKey, dispatch_key_matches
from .release_node_index import checked_release_compute_node_binding
from .specialization import FiniteAOTStrategyCatalog

def validate_dispatch_target(plan: ExecutionPlan, gpu_backend_available: Bool) -> TensorStatus:
    # Validate the complete immutable-plan contract before target selection.
    # Dispatch must never turn malformed metadata into a native kernel launch.
    var proof = plan.checked_snapshot()
    if not proof.is_ok():
        return proof.status
    var selected = proof.payload.value().copy()
    if selected.placement == materialize[ExecutionPlacement.CPU_ONLY]():
        return TensorStatus.OK
    if not gpu_backend_available:
        return TensorStatus.BACKEND_UNAVAILABLE
    return TensorStatus.OK


def validate_dataflow_dispatch_target[nodes: Int, slots: Int](
    graph: SealedStaticDataflowPlan[nodes, slots],
    plan: ExecutionPlan, gpu_backend_available: Bool,
) -> TensorStatus:
    # Validate correspondence of the ONE dataflow model to the selected
    # CPU/GPU/Hybrid lowering before checking the target's availability.
    # Actual owner/stream/artifact handles must still be bound natively.
    var matched = validate_static_dataflow_execution_plan(graph, plan)
    if not matched.is_ok():
        return matched.status
    return validate_dispatch_target(plan, gpu_backend_available)




def bind_finite_aot_strategy[capacity: Int](
    plan: ExecutionPlan, key: DispatchKey,
    catalog: FiniteAOTStrategyCatalog[capacity],
    expected_abi_version: Int,
) -> TensorResult[ExecutionPlan]:
    # Single AOT variant-selection boundary. A keyed, finite catalog must be
    # supplied by the offline plan builder: a naked integer variant cannot
    # authorize an executable kernel. Native ABI/code authentication remains
    # a distinct and mandatory checked backend operation.
    var prepared = plan.checked_snapshot()
    if not prepared.is_ok():
        return TensorResult[ExecutionPlan](prepared.status)
    if expected_abi_version <= 0 or expected_abi_version > 65535:
        return TensorResult[ExecutionPlan](TensorStatus.UNSUPPORTED)
    var selected = catalog.select_variant_checked(key)
    if not selected.is_ok():
        return TensorResult[ExecutionPlan](selected.status)
    if selected.payload.value().abi_version != expected_abi_version:
        return TensorResult[ExecutionPlan](TensorStatus.UNSUPPORTED)
    var prior = prepared.payload.value().copy()
    return ExecutionPlan.checked(
        prior.placement, prior.total_items,
        prior.cpu_begin, prior.cpu_end, prior.gpu_begin, prior.gpu_end,
        prior.transfer_bytes, selected.payload.value().decision.aot_variant,
        prior.dependency_count, prior.output_device,
    )



def _flat_dataflow_aot_key_is_legal(key: DispatchKey, is_gpu: Bool) -> Bool:
    # This IR has FP32 element offsets and a bounded flat C-order native
    # interface; it carries NO per-axis stride/shape, nor an aligned native
    # base-address certificate. A caller-selected AOT key may not enlarge
    # that ABI by asserting a different layout or SIMD alignment.
    # Layout class 0 is canonical contiguous row-major in this source ABI.
    if key.layout_class != 0:
        return False
    if key.alignment_class != 0 and key.alignment_class != 4:
        return False
    if is_gpu:
        return key.target_arch > 0
    return key.target_arch == 0


def validate_finite_aot_dataflow_key[nodes: Int, slots: Int](
    graph: SealedStaticDataflowPlan[nodes, slots], key: DispatchKey,
) -> TensorStatus:
    # One DispatchKey denotes ONE compiled operation/shape/math ABI. A DAG
    # can contain several DIFFERENT operations; binding a single kernel key
    # to such a graph would be an incorrect artifact-identity proof. Until
    # an immutable per-node native-artifact table exists, refuse that case.
    # Copy and validate ALL untrusted metadata before reading any node.
    var checked_key = key.checked_snapshot()
    if not checked_key.is_ok():
        return checked_key.status
    var checked_graph = graph.checked_snapshot()
    if not checked_graph.is_ok():
        return checked_graph.status
    var request = checked_key.payload.value().copy()
    var sealed = checked_graph.payload.take()
    var saw_compute = False
    var saw_gpu = False
    var saw_cpu = False
    for i in range(sealed.node_count()):
        var node = sealed._cpu_lowering_node_unchecked(i)
        if node.kind != 0 and node.kind != 1:
            # DMA, packing and joins have separate stage/transfer contracts.
            # They must not be reinterpreted as the compiled compute kernel.
            continue
        saw_compute = True
        if node.kind == 1:
            saw_gpu = True
        else:
            saw_cpu = True
        if not _flat_dataflow_aot_key_is_legal(request, node.kind == 1):
            return TensorStatus.DEVICE_MISMATCH
        # Stable operation encoding for this bounded IR: family * 16 +
        # family opcode. Both are bounded in checked_snapshot(), so there
        # is no signed overflow or collision among supported operations.
        var semantic_operation = node.primitive * 16 + node.opcode
        if request.operation != semantic_operation:
            return TensorStatus.UNSUPPORTED
        if request.rank != node.rank or request.math_mode != node.math_mode:
            return TensorStatus.INVALID_LAYOUT
    if not saw_compute:
        # An all-transfer plan has no compute artifact to bind.
        return TensorStatus.UNSUPPORTED
    if saw_gpu and saw_cpu:
        # CPU and GPU symbols are separate native target lowerings even if
        # the operation/rank are identical. Bind one key per compute node.
        return TensorStatus.NOT_REPRESENTABLE
    return TensorStatus.OK



def bind_finite_aot_dataflow_strategy[
    nodes: Int, slots: Int, capacity: Int,
](
    graph: SealedStaticDataflowPlan[nodes, slots],
    plan: ExecutionPlan, key: DispatchKey,
    catalog: FiniteAOTStrategyCatalog[capacity],
    expected_abi_version: Int, gpu_backend_available: Bool,
) -> TensorResult[ExecutionPlan]:
    # Both the source IR and the selected AOT plan must describe the exact
    # SAME node variants and placement. Never repair an IR/planner mismatch
    # by dispatching another undeclared kernel or silently changing targets.
    var semantic_status = validate_finite_aot_dataflow_key(graph, key)
    if semantic_status != TensorStatus.OK:
        return TensorResult[ExecutionPlan](semantic_status)
    var selected = bind_finite_aot_strategy(
        plan, key, catalog, expected_abi_version,
    )
    if not selected.is_ok():
        return selected^
    var status = validate_dataflow_dispatch_target(
        graph, selected.payload.value(), gpu_backend_available,
    )
    if status != TensorStatus.OK:
        return TensorResult[ExecutionPlan](status)
    return selected^


# One immutable-by-contract AOT identity selection for EACH compute node of
# the SAME static graph. Transfer/host-prepare nodes have variant=-1 because
# they have no compute-kernel symbol. These arrays are not callable handles;
# the actual native loader must verify code hash, ABI and target separately.
@fieldwise_init
struct DataflowPerNodeAOTSelection[count: Int](Copyable):
    # Preserve ALL identity dimensions, not just the numeric variant.
    # An opcode must never silently borrow another operation's symbol.
    var keys: Array[DispatchKey, Self.count]
    var is_compute: Array[Bool, Self.count]
    var variant: Array[Int, Self.count]
    var abi_version: Array[Int, Self.count]
    var selected_compute_nodes: Int
    var selected_gpu_nodes: Int


def _bind_finite_per_node_aot_catalog_any_abi[
    nodes: Int, slots: Int, capacity: Int,
](
    graph: SealedStaticDataflowPlan[nodes, slots],
    plan: ExecutionPlan,
    node_keys: Array[DispatchKey, nodes],
    catalog: FiniteAOTStrategyCatalog[capacity],
    required_abi_version: Int,
    gpu_backend_available: Bool,
) -> TensorResult[DataflowPerNodeAOTSelection[nodes]]:
    # Supports heterogeneous primitive families without inventing a second
    # executor/IR or applying one variant to every node. The offline plan
    # supplies one key per compute node; noncompute nodes have no key usage.
    # This returns only a checked selection TABLE, not an authenticated GPU
    # binary or permission to enqueue an arbitrary native function pointer.
    if required_abi_version < 0 or required_abi_version > 65535:
        return TensorResult[DataflowPerNodeAOTSelection[nodes]](
            TensorStatus.UNSUPPORTED
        )
    var graph_proof = graph.checked_snapshot()
    if not graph_proof.is_ok():
        return TensorResult[DataflowPerNodeAOTSelection[nodes]](
            graph_proof.status
        )
    var plan_proof = plan.checked_snapshot()
    if not plan_proof.is_ok():
        return TensorResult[DataflowPerNodeAOTSelection[nodes]](
            plan_proof.status
        )
    var selected_plan = plan_proof.payload.value().copy()
    var placement_status = validate_dispatch_target(
        selected_plan, gpu_backend_available
    )
    if placement_status != TensorStatus.OK:
        return TensorResult[DataflowPerNodeAOTSelection[nodes]](
            placement_status
        )
    var agreement = validate_static_dataflow_execution_plan_per_node_artifacts(
        graph_proof.payload.value(), selected_plan,
    )
    if not agreement.is_ok():
        return TensorResult[DataflowPerNodeAOTSelection[nodes]](
            agreement.status
        )
    var catalog_proof = catalog.checked_snapshot()
    if not catalog_proof.is_ok():
        return TensorResult[DataflowPerNodeAOTSelection[nodes]](
            catalog_proof.status
        )
    var validated_graph = graph_proof.payload.take()
    var validated_catalog = catalog_proof.payload.take()
    # A noncompute stage must never carry a caller-provided AOT symbol key.
    # A DMA/host stage has no compiled compute identity, and a key outside
    # node_count must not leak into an otherwise checked fixed-size table.
    # Canonicalize ALL inactive entries before binding the real compute
    # stages from the finite, checked catalog.  This avoids preserving an
    # unchecked DispatchKey in a result described as a sealed selection.
    var neutral_key = DispatchKey(0, 0, 0, 0, 0, 0, 0, 0)
    var bound_keys = Array[DispatchKey, nodes](fill=neutral_key)
    var compute_mask = Array[Bool, nodes](fill=False)
    var variants = Array[Int, nodes](fill=-1)
    var abi_versions = Array[Int, nodes](fill=0)
    var compute_count = 0
    var gpu_count = 0
    for i in range(validated_graph.node_count()):
        var node = validated_graph._cpu_lowering_node_unchecked(i)
        if node.kind != 0 and node.kind != 1:
            continue
        var key_proof = node_keys[i].checked_snapshot()
        if not key_proof.is_ok():
            return TensorResult[DataflowPerNodeAOTSelection[nodes]](
                key_proof.status
            )
        var key = key_proof.payload.value().copy()
        if key.operation != node.primitive * 16 + node.opcode:
            return TensorResult[DataflowPerNodeAOTSelection[nodes]](
                TensorStatus.UNSUPPORTED
            )
        if key.rank != node.rank or key.math_mode != node.math_mode:
            return TensorResult[DataflowPerNodeAOTSelection[nodes]](
                TensorStatus.INVALID_LAYOUT
            )
        if not _flat_dataflow_aot_key_is_legal(key, node.kind == 1):
            return TensorResult[DataflowPerNodeAOTSelection[nodes]](
                TensorStatus.NOT_REPRESENTABLE
            )
        if node.kind == 1:
            gpu_count += 1
            if not gpu_backend_available:
                return TensorResult[DataflowPerNodeAOTSelection[nodes]](
                    TensorStatus.BACKEND_UNAVAILABLE
                )
        var selected = validated_catalog.select_variant_checked(key)
        if not selected.is_ok():
            return TensorResult[DataflowPerNodeAOTSelection[nodes]](
                selected.status
            )
        var entry = selected.payload.value().copy()
        # ABI version zero is INTERNAL to the heterogeneous per-node binder.
        # Each node's actual catalog ABI is still checked against the signed
        # release table (for GPU) and an explicit expected-node ABI array.
        if required_abi_version != 0 and entry.abi_version != required_abi_version:
            return TensorResult[DataflowPerNodeAOTSelection[nodes]](
                TensorStatus.UNSUPPORTED
            )
        if entry.decision.aot_variant != node.aot_variant:
            return TensorResult[DataflowPerNodeAOTSelection[nodes]](
                TensorStatus.INVALID_LAYOUT
            )
        if node.kind == 1:
            # A finite numeric strategy key is NOT a native symbol/ABI proof.
            # The build-bound signed release index must bind this precise
            # primitive+opcode, variant, target and ABI BEFORE any GPU stage
            # can be lowered. The source-only index intentionally denies all.
            var native_release = checked_release_compute_node_binding(
                entry.key.operation, entry.decision.aot_variant,
                entry.key.target_arch, entry.abi_version,
            )
            if native_release != TensorStatus.OK:
                return TensorResult[DataflowPerNodeAOTSelection[nodes]](
                    native_release
                )
        # Use the catalog-owned checked key rather than the untrusted
        # caller input: artifact loading must match this entire identity.
        bound_keys[i] = entry.key.copy()
        compute_mask[i] = True
        variants[i] = entry.decision.aot_variant
        abi_versions[i] = entry.abi_version
        compute_count += 1
    # An empty compute graph is legal for a verified explicit copy/transfer,
    # but such a plan has no native compute artifact to bind here.
    if compute_count == 0:
        return TensorResult[DataflowPerNodeAOTSelection[nodes]](
            TensorStatus.UNSUPPORTED
        )
    return TensorResult[DataflowPerNodeAOTSelection[nodes]](
        DataflowPerNodeAOTSelection[nodes](
            bound_keys^, compute_mask^, variants^, abi_versions^,
            compute_count, gpu_count,
        )
    )



# Public homogeneous binder preserves the original single-ABI contract.
# ABI 0 is NEVER a public permission to accept an arbitrary catalog entry.
def bind_finite_per_node_aot_catalog[
    nodes: Int, slots: Int, capacity: Int,
](
    graph: SealedStaticDataflowPlan[nodes, slots],
    plan: ExecutionPlan,
    node_keys: Array[DispatchKey, nodes],
    catalog: FiniteAOTStrategyCatalog[capacity],
    required_abi_version: Int,
    gpu_backend_available: Bool,
) -> TensorResult[DataflowPerNodeAOTSelection[nodes]]:
    if required_abi_version <= 0 or required_abi_version > 65535:
        return TensorResult[DataflowPerNodeAOTSelection[nodes]](
            TensorStatus.UNSUPPORTED
        )
    return _bind_finite_per_node_aot_catalog_any_abi(
        graph, plan, node_keys, catalog, required_abi_version,
        gpu_backend_available,
    )

def _validate_per_node_aot_selection_against_catalog_any_abi[
    nodes: Int, slots: Int, capacity: Int,
](
    candidate: DataflowPerNodeAOTSelection[nodes],
    graph: SealedStaticDataflowPlan[nodes, slots],
    plan: ExecutionPlan,
    original_keys: Array[DispatchKey, nodes],
    catalog: FiniteAOTStrategyCatalog[capacity],
    required_abi_version: Int,
    gpu_backend_available: Bool,
) -> TensorStatus:
    # A mutable selection returned from a prior check is not a launch token.
    # Reconstruct the authoritative per-node identity from the sealed graph
    # and the finite checked catalog BEFORE using any metadata in the
    # concrete target binder. This runs once at the owner/binder boundary.
    var fresh = _bind_finite_per_node_aot_catalog_any_abi(
        graph, plan, original_keys, catalog, required_abi_version,
        gpu_backend_available,
    )
    if not fresh.is_ok():
        return fresh.status
    var authoritative = fresh.payload.value().copy()
    if candidate.selected_compute_nodes != authoritative.selected_compute_nodes:
        return TensorStatus.INVALID_LAYOUT
    if candidate.selected_gpu_nodes != authoritative.selected_gpu_nodes:
        return TensorStatus.DEVICE_MISMATCH
    for i in range(nodes):
        if candidate.is_compute[i] != authoritative.is_compute[i]:
            return TensorStatus.INVALID_LAYOUT
        if candidate.variant[i] != authoritative.variant[i]:
            return TensorStatus.UNSUPPORTED
        if candidate.abi_version[i] != authoritative.abi_version[i]:
            return TensorStatus.UNSUPPORTED
        # Compare EVERY key. The binder canonicalizes unused slots to an
        # inert identity so that a forged noncompute key cannot be confused
        # with a later native symbol or carried into an external launch table.
        # This is source-level hygiene, not native artifact authentication.
        if not dispatch_key_matches(candidate.keys[i], authoritative.keys[i]):
            return TensorStatus.DEVICE_MISMATCH
    return TensorStatus.OK


# A single UASDEA graph may lower CPU SIMD, GPU binary and GPU matrix nodes
# through different AOT ABIs. Do not force one arbitrary version onto EVERY
# compute node. The expected array is static-rank-owned data, not a registry.

# Public homogeneous validation cannot be bypassed with an ABI-0 sentinel.
def validate_per_node_aot_selection_against_catalog[
    nodes: Int, slots: Int, capacity: Int,
](
    candidate: DataflowPerNodeAOTSelection[nodes],
    graph: SealedStaticDataflowPlan[nodes, slots],
    plan: ExecutionPlan,
    original_keys: Array[DispatchKey, nodes],
    catalog: FiniteAOTStrategyCatalog[capacity],
    required_abi_version: Int,
    gpu_backend_available: Bool,
) -> TensorStatus:
    if required_abi_version <= 0 or required_abi_version > 65535:
        return TensorStatus.UNSUPPORTED
    return _validate_per_node_aot_selection_against_catalog_any_abi(
        candidate, graph, plan, original_keys, catalog,
        required_abi_version, gpu_backend_available,
    )

def bind_finite_per_node_aot_catalog_mixed[
    nodes: Int, slots: Int, capacity: Int,
](
    graph: SealedStaticDataflowPlan[nodes, slots],
    plan: ExecutionPlan,
    node_keys: Array[DispatchKey, nodes],
    catalog: FiniteAOTStrategyCatalog[capacity],
    expected_node_abi: Array[Int, nodes],
    gpu_backend_available: Bool,
) -> TensorResult[DataflowPerNodeAOTSelection[nodes]]:
    # 0 means "check exact ABI per node", NOT "trust any ABI". This code
    # proves each one matches the caller's checked release plan. GPU code
    # additionally goes through checked_release_compute_node_binding().
    for i in range(nodes):
        if expected_node_abi[i] < 0 or expected_node_abi[i] > 65535:
            return TensorResult[DataflowPerNodeAOTSelection[nodes]](
                TensorStatus.UNSUPPORTED
            )
    var selected = _bind_finite_per_node_aot_catalog_any_abi(
        graph, plan, node_keys, catalog, 0, gpu_backend_available,
    )
    if not selected.is_ok():
        return selected^
    var binding = selected.payload.value().copy()
    for i in range(nodes):
        if binding.is_compute[i]:
            if (expected_node_abi[i] <= 0
                or expected_node_abi[i] != binding.abi_version[i]):
                return TensorResult[DataflowPerNodeAOTSelection[nodes]](
                    TensorStatus.UNSUPPORTED
                )
        elif expected_node_abi[i] != 0:
            # DMA/host-prepare/join stages never carry compute-kernel ABIs.
            return TensorResult[DataflowPerNodeAOTSelection[nodes]](
                TensorStatus.INVALID_LAYOUT
            )
    return selected^


def validate_per_node_aot_selection_against_catalog_mixed[
    nodes: Int, slots: Int, capacity: Int,
](
    candidate: DataflowPerNodeAOTSelection[nodes],
    graph: SealedStaticDataflowPlan[nodes, slots],
    plan: ExecutionPlan,
    original_keys: Array[DispatchKey, nodes],
    catalog: FiniteAOTStrategyCatalog[capacity],
    expected_node_abi: Array[Int, nodes],
    gpu_backend_available: Bool,
) -> TensorStatus:
    # First rebind ALL node identities and release gates from the graph,
    # then prove the caller's expected ABI vector, including zero on every
    # noncompute stage. No mutable cached tuple authorizes native launch.
    var authoritative = _validate_per_node_aot_selection_against_catalog_any_abi(
        candidate, graph, plan, original_keys, catalog, 0,
        gpu_backend_available,
    )
    if authoritative != TensorStatus.OK:
        return authoritative
    for i in range(nodes):
        if candidate.is_compute[i]:
            if (expected_node_abi[i] <= 0 or expected_node_abi[i] > 65535
                or expected_node_abi[i] != candidate.abi_version[i]):
                return TensorStatus.UNSUPPORTED
        elif expected_node_abi[i] != 0:
            return TensorStatus.INVALID_LAYOUT
    return TensorStatus.OK
