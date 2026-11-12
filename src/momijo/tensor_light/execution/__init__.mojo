# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR EXECUTION / EXECUTION PLANNING AND AOT INTERFACE
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Execution Planning and AOT
# Module:        momijo.tensor_light.execution
# Source:        src/momijo/tensor_light/execution/__init__.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines the execution planning and AOT package boundary for the FP32 Light
#   implementation.
#
#   Re-exports explicitly imported interfaces without declaring an independent
#   implementation.
#
# Responsibilities:
#   - Define the execution planning and AOT import and namespace boundary.
#   - Keep imported public names consistent with their owning modules.
#   - Avoid introducing duplicate implementations or initialization side
#     effects.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Expose only symbols actually defined by imported modules.
#   - Preserve the established package API and import graph.
#
# Performance Objectives:
#   - Avoid initialization side effects or eager heavy allocations.
#   - Keep executable hot-path implementation in dedicated modules.
#
# Maintainability:
#   - Update curated exports when public APIs move or change.
#   - Do not add import cycles or duplicate implementations.
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

# Curated package exports. No initialization side effects.
from .execution_plan import ExecutionPlan, ExecutionPlacement, StaticAOTExecutionPlan
from .launch_policy import HybridTuningProfile
from .dependency import ExecutionDependency
from .dependency import ExecutionStage, ExecutionStageKind, CheckedStageSchedule, BoundStageSchedule
from .launch_policy import build_host_add_split_schedule

# Generic bounded fixed-FP32 dataflow description (native owner/ISA binding separate).
from .static_dataflow import DataflowMemorySlot, DataflowRegion, DataflowNode, SealedStaticDataflowPlan

from .static_dataflow import DataflowExecutionEvidence, validate_static_dataflow_execution_plan

# Expert AOT integration: converts the existing host-binary native stage DAG
# to the ONE canonical dataflow IR, then checks split/pipeline agreement.
# No second public tensor owner, scheduler, kernel registry or JIT is created.
from .static_dataflow import canonical_host_binary_dataflow_from_schedule
from .static_dataflow import validate_host_binary_native_schedule_agreement, validate_host_binary_pipeline_window_agreement

# Checked offline capacity requirements; not an allocator or owner proof.
from .static_dataflow import StaticDataflowMemoryRequirements, static_dataflow_memory_requirements
# Checked row-oriented CPU/GPU split of ONE ExecutionPlan, explicitly sized
# for one H2D input and one D2H output per GPU-owned reduction row.
from .launch_policy import build_host_reduce_rows_split_plan
from .launch_policy import build_host_gemm_rows_split_plan
from .launch_policy import HybridGemmTuningProfile, validate_hybrid_gemm_profile
from .launch_policy import choose_host_gemm_rows_split_plan_for_capacity
from .launch_policy import choose_host_reduce_rows_split_plan_for_capacity
from .launch_policy import HybridRowReductionTuningProfile, validate_hybrid_row_reduction_profile
# Immutable-by-contract checked ownership of a finite native release plan.
# AOT binary authentication and the actual executor remain separate.
from .aot_release_plan import BoundAOTDataflowPlan

# Optional PREPARE-time admission with explicit RAM, pinned-host and VRAM caps.
from .launch_policy import admit_host_binary_plan_with_total_ram_budget
from .launch_policy import choose_host_binary_pipeline_with_total_ram_budget
