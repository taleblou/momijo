# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR BACKEND / BACKEND SELECTION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Backend Availability
# Module:        momijo.tensor_light.backend.backend
# Source:        src/momijo/tensor_light/backend/backend.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines backend selection in the backend availability layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `BackendSelection`, `backend_for_plan`,
#   `backend_for_plan_checked`, `backend_for_static_dataflow`; preconditions
#   and supported targets are determined by their implementations.
#
# Responsibilities:
#   - Select or report supported execution backends explicitly.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `BackendSelection`,
#     `backend_for_plan`, `backend_for_plan_checked`,
#     `backend_for_static_dataflow`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Reject unsupported targets rather than silently falling back.
#   - Keep backend selection consistent with actual runtime capability.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid device initialization during pure capability queries.
#   - Preserve direct CPU paths when validated and applicable.
#
# Maintainability:
#   - Separate availability checks from native launch authorization.
#   - Make missing GPU/hybrid implementations explicit.
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

# One execution architecture; no runtime virtual TensorBackend registry.
from ..execution.execution_plan import ExecutionPlan, ExecutionPlacement
from ..internal.status import TensorResult, TensorStatus
from ..execution.static_dataflow import SealedStaticDataflowPlan, validate_static_dataflow_execution_plan

@fieldwise_init
struct BackendSelection(Copyable):
    var code: Int
    comptime CPU = Self(0)
    comptime GPU = Self(1)
    comptime HYBRID = Self(2)

def _backend_for_validated_plan(plan: ExecutionPlan) -> BackendSelection:
    # Internal only: this function must receive a snapshot that has already
    # passed ExecutionPlan.checked(). There is no implicit hybrid fallback for
    # unknown execution placement values.
    if plan.placement == ExecutionPlacement.CPU_ONLY:
        return BackendSelection.CPU
    if plan.placement == ExecutionPlacement.GPU_ONLY:
        return BackendSelection.GPU
    return BackendSelection.HYBRID


def backend_for_plan(plan: ExecutionPlan) -> TensorResult[BackendSelection]:
    # The ordinary public API must not silently accept a forged or mutated
    # placement and mistake it for HYBRID. Validate once; dispatch solely from
    # the separately owned snapshot, not the caller's mutable value.
    var proof = plan.checked_snapshot()
    if not proof.is_ok():
        return TensorResult[BackendSelection](proof.status)
    var checked_plan = proof.payload.value()
    return TensorResult[BackendSelection](_backend_for_validated_plan(checked_plan))


def backend_for_plan_checked(plan: ExecutionPlan) -> TensorResult[BackendSelection]:
    # Backward-compatible checked spelling: both names now enforce the same
    # checked-boundary contract and yield TensorResult.
    return backend_for_plan(plan)


def backend_for_static_dataflow[nodes: Int, slots: Int](
    graph: SealedStaticDataflowPlan[nodes, slots], plan: ExecutionPlan,
) -> TensorResult[BackendSelection]:
    # An ExecutionPlan alone does not prove it lowers the supplied DAG.
    # This checks both snapshots and their exact mapping-level agreement.
    var matched = validate_static_dataflow_execution_plan(graph, plan)
    if not matched.is_ok():
        return TensorResult[BackendSelection](matched.status)
    return backend_for_plan(plan)
