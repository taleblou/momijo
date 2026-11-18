# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR EXECUTION / BOUND AOT RELEASE PLANS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Execution Planning and AOT
# Module:        momijo.tensor_light.execution.aot_release_plan
# Source:        src/momijo/tensor_light/execution/aot_release_plan.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines bound AOT release plans within checked AOT execution planning for
#   the FP32 Light implementation.
#
#   Binds checked, immutable execution metadata; it does not load or execute a
#   native binary.
#
# Responsibilities:
#   - Represent and validate prepared plans, typed dataflow metadata,
#     placement, and immutable snapshots.
#   - Expose module-level declarations including `BoundAOTDataflowPlan`.
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

# AOT release binder for the ONE fixed-FP32 static dataflow architecture.
# CPU/GPU/Hybrid are placement lowerings, not independent graph formats.
# This value owns checked copies of graph, placement, node dispatch keys and
# the finite strategy catalog, so a caller cannot change those separately
# AFTER a successful checked binding. No runtime JIT, registration or scheduler.
# This is a source-level sealed plan, NOT a native binary loader/executor.
from ..internal.status import TensorResult, TensorStatus
from ..dispatch.dispatch_key import DispatchKey
from ..dispatch.specialization import FiniteAOTStrategyCatalog
from ..dispatch.dispatch import (
    DataflowPerNodeAOTSelection,
    bind_finite_per_node_aot_catalog,
    validate_per_node_aot_selection_against_catalog,
    bind_finite_per_node_aot_catalog_mixed,
    validate_per_node_aot_selection_against_catalog_mixed,
)
from .execution_plan import ExecutionPlan
from .static_dataflow import SealedStaticDataflowPlan


struct BoundAOTDataflowPlan[
    nodes: Int, slots: Int, variants: Int,
](Movable):
    var _graph: SealedStaticDataflowPlan[Self.nodes, Self.slots]
    var _placement: ExecutionPlan
    var _keys: Array[DispatchKey, Self.nodes]
    var _catalog: FiniteAOTStrategyCatalog[Self.variants]
    var _bindings: DataflowPerNodeAOTSelection[Self.nodes]
    var _abi_version: Int
    var _expected_node_abi: Array[Int, Self.nodes]
    var _gpu_available: Bool

    def __init__(
        out self,
        var graph: SealedStaticDataflowPlan[Self.nodes, Self.slots],
        placement: ExecutionPlan,
        keys: Array[DispatchKey, Self.nodes],
        catalog: FiniteAOTStrategyCatalog[Self.variants],
        bindings: DataflowPerNodeAOTSelection[Self.nodes],
        abi_version: Int,
        gpu_available: Bool,
        expected_node_abi: Array[Int, Self.nodes],
    ):
        self._graph = graph
        self._placement = placement.copy()
        self._keys = keys.copy()
        self._catalog = catalog.copy()
        self._bindings = bindings.copy()
        self._abi_version = abi_version
        self._expected_node_abi = expected_node_abi.copy()
        self._gpu_available = gpu_available

    @staticmethod
    def checked(
        graph: SealedStaticDataflowPlan[Self.nodes, Self.slots],
        placement: ExecutionPlan,
        keys: Array[DispatchKey, Self.nodes],
        catalog: FiniteAOTStrategyCatalog[Self.variants],
        abi_version: Int,
        gpu_available: Bool,
    ) -> TensorResult[Self]:
        # The homogeneous constructor never accepts the mixed-ABI sentinel.
        # Use checked_mixed() and an explicit static expected-ABI vector.
        if abi_version <= 0 or abi_version > 65535:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        # Rebuild ALL source identities once, before anything can be launched.
        var graph_check = graph.checked_snapshot()
        if not graph_check.is_ok():
            return TensorResult[Self](graph_check.status)
        var plan_check = placement.checked_snapshot()
        if not plan_check.is_ok():
            return TensorResult[Self](plan_check.status)
        var cat_check = catalog.checked_snapshot()
        if not cat_check.is_ok():
            return TensorResult[Self](cat_check.status)
        var native_graph = graph_check.payload.take()
        var native_plan = plan_check.payload.take()
        var native_catalog = cat_check.payload.take()
        var native_keys = keys.copy()
        var selected = bind_finite_per_node_aot_catalog(
            native_graph, native_plan, native_keys, native_catalog,
            abi_version, gpu_available,
        )
        if not selected.is_ok():
            return TensorResult[Self](selected.status)
        var checked_binding = selected.payload.take()
        var expected = Array[Int, Self.nodes](fill=0)
        for i in range(Self.nodes):
            if checked_binding.is_compute[i]:
                expected[i] = abi_version
        return TensorResult[Self](Self(
            native_graph, native_plan, native_keys, native_catalog,
            checked_binding, abi_version, gpu_available, expected,
        ))

    @staticmethod
    def checked_mixed(
        graph: SealedStaticDataflowPlan[Self.nodes, Self.slots],
        placement: ExecutionPlan,
        keys: Array[DispatchKey, Self.nodes],
        catalog: FiniteAOTStrategyCatalog[Self.variants],
        expected_node_abi: Array[Int, Self.nodes],
        gpu_available: Bool,
    ) -> TensorResult[Self]:
        # Mixed-ABI DAG still has ONE graph, placement and finite AOT plan.
        # A checked array describes exactly one ABI for each COMPUTE node.
        var graph_check = graph.checked_snapshot()
        if not graph_check.is_ok():
            return TensorResult[Self](graph_check.status)
        var plan_check = placement.checked_snapshot()
        if not plan_check.is_ok():
            return TensorResult[Self](plan_check.status)
        var cat_check = catalog.checked_snapshot()
        if not cat_check.is_ok():
            return TensorResult[Self](cat_check.status)
        var native_graph = graph_check.payload.take()
        var native_plan = plan_check.payload.take()
        var native_catalog = cat_check.payload.take()
        var native_keys = keys.copy()
        var expected = expected_node_abi.copy()
        var selected = bind_finite_per_node_aot_catalog_mixed(
            native_graph, native_plan, native_keys, native_catalog,
            expected, gpu_available,
        )
        if not selected.is_ok():
            return TensorResult[Self](selected.status)
        return TensorResult[Self](Self(
            native_graph, native_plan, native_keys, native_catalog,
            selected.payload.take(), 0, gpu_available, expected,
        ))

    def validate_against_owned_catalog(self) -> TensorStatus:
        # A borrowed/mutable metadata object alone is not an executable
        # capability. Recheck node identities and artifact release allowlists
        # immediately before a trusted native adapter consumes the plan.
        if self._abi_version == 0:
            return validate_per_node_aot_selection_against_catalog_mixed(
                self._bindings, self._graph, self._placement,
                self._keys, self._catalog, self._expected_node_abi,
                self._gpu_available,
            )
        return validate_per_node_aot_selection_against_catalog(
            self._bindings, self._graph, self._placement,
            self._keys, self._catalog, self._abi_version,
            self._gpu_available,
        )

    def checked_cpu_graph_snapshot(
        self,
    ) -> TensorResult[SealedStaticDataflowPlan[Self.nodes, Self.slots]]:
        # A bound finite catalog is the authoritative source of variant and
        # ABI identity. A naked SealedStaticDataflowPlan is not evidence that
        # its declared AOT variants agree with this immutable-by-contract
        # execution plan. Never silently lower GPU/DMA stages onto the CPU.
        var status = self.validate_against_owned_catalog()
        if status != TensorStatus.OK:
            return TensorResult[SealedStaticDataflowPlan[Self.nodes, Self.slots]](
                status
            )
        var placement = self._placement.checked_native_cpu_snapshot()
        if not placement.is_ok():
            return TensorResult[SealedStaticDataflowPlan[Self.nodes, Self.slots]](
                placement.status
            )
        if self._bindings.selected_gpu_nodes != 0:
            return TensorResult[SealedStaticDataflowPlan[Self.nodes, Self.slots]](
                TensorStatus.DEVICE_MISMATCH
            )
        var graph = self._graph.checked_snapshot()
        if not graph.is_ok():
            return TensorResult[SealedStaticDataflowPlan[Self.nodes, Self.slots]](
                graph.status
            )
        var sealed = graph.payload.take()
        for i in range(sealed._cpu_lowering_used_slots()):
            if sealed._cpu_lowering_slot_unchecked(i).domain != 0:
                return TensorResult[SealedStaticDataflowPlan[Self.nodes, Self.slots]](
                    TensorStatus.DEVICE_MISMATCH
                )
        for i in range(sealed.node_count()):
            var node = sealed._cpu_lowering_node_unchecked(i)
            if node.kind != 0 and node.kind != 5 and node.kind != 6:
                return TensorResult[SealedStaticDataflowPlan[Self.nodes, Self.slots]](
                    TensorStatus.UNSUPPORTED
                )
            if node.scratch_bytes != 0 or node.aot_variant != 0:
                return TensorResult[SealedStaticDataflowPlan[Self.nodes, Self.slots]](
                    TensorStatus.UNSUPPORTED
                )
            if node.kind == 0:
                if not self._bindings.is_compute[i]:
                    return TensorResult[SealedStaticDataflowPlan[Self.nodes, Self.slots]](
                        TensorStatus.INVALID_LAYOUT
                    )
                if self._bindings.variant[i] != node.aot_variant:
                    return TensorResult[SealedStaticDataflowPlan[Self.nodes, Self.slots]](
                        TensorStatus.UNSUPPORTED
                    )
        return TensorResult[SealedStaticDataflowPlan[Self.nodes, Self.slots]](sealed)

    def checked_placement(self) -> TensorResult[ExecutionPlan]:
        var status = self.validate_against_owned_catalog()
        if status != TensorStatus.OK:
            return TensorResult[ExecutionPlan](status)
        return self._placement.checked_snapshot()

    def compute_stage_count(self) -> TensorResult[Int]:
        var status = self.validate_against_owned_catalog()
        if status != TensorStatus.OK:
            return TensorResult[Int](status)
        return TensorResult[Int](self._bindings.selected_compute_nodes)

    def gpu_stage_count(self) -> TensorResult[Int]:
        var status = self.validate_against_owned_catalog()
        if status != TensorStatus.OK:
            return TensorResult[Int](status)
        return TensorResult[Int](self._bindings.selected_gpu_nodes)
