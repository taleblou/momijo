# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DISPATCH / RELEASE NODE INDEX
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Dispatch and Specialization
# Module:        momijo.tensor_light.dispatch.release_node_index
# Source:        src/momijo/tensor_light/dispatch/release_node_index.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Validates the generated, fail-closed FP32 compute-node release allowlist
#   used by AOT dispatch.
#
#   A source catalog entry is not proof that matching target-native machine
#   code is present or authenticated.
#
# Responsibilities:
#   - Expose `checked_release_compute_node_binding` for checked compute-node
#     release binding decisions.
#   - Reject missing, mismatched, or unauthorized native artifact identities.
#   - Keep release catalog validation distinct from loading and executing
#     device code.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Default to denying unmatched release artifacts and target capabilities.
#   - Validate the source-defined release identity, ABI, and binding
#     constraints.
#   - Do not treat catalog membership alone as authentication of loaded native
#     bytes.
#
# Performance Objectives:
#   - Use static allowlist lookup without runtime JIT or benchmark-based
#     selection.
#   - Keep release checks outside per-element execution hot paths.
#
# Maintainability:
#   - Keep generated release indices and offline artifact verification in
#     sync.
#   - Exercise rejected or inconsistent bindings in regression tests.
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

# Fail-closed native AOT allowlist for GPU compute nodes in UASDEA static DAG.
# A source-level finite strategy catalog is not a trusted native binary.
# An external build may REPLACE this module only with a deterministic checked
# table generated from independently signed, SHA256-verified AOT artifacts.
# The bound native handle must still be loaded from the exact signed image.
from ..internal.status import TensorStatus


def checked_release_compute_node_binding(
    encoded_operation: Int, variant: Int, target_arch: Int,
    abi_version: Int,
) -> TensorStatus:
    # No general GPU static-DAG native kernel images ship in source-only v197.
    # In particular, a CUDA-capable device or compatible numeric key is NOT
    # evidence that the requested opcode has a precompiled correct symbol.
    return TensorStatus.UNSUPPORTED
