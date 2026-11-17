# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DISPATCH / RELEASE ARTIFACT INDEX
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Dispatch and Specialization
# Module:        momijo.tensor_light.dispatch.release_artifact_index
# Source:        src/momijo/tensor_light/dispatch/release_artifact_index.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Validates the generated, fail-closed FP32 GEMM release allowlist used by
#   AOT dispatch.
#
#   A source catalog entry is not proof that matching target-native machine
#   code is present or authenticated.
#
# Responsibilities:
#   - Expose `checked_release_gemm_binding` for checked GEMM release binding
#     decisions.
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

# Generated native-release allowlist. Default is DENY ALL native TF32 MMA.
# Future native-release packages require a separately reviewed offline
# export/ABI gate. This source-only snapshot includes no such verifier or
# executable binder; artifact hashes alone cannot attest loaded-image ABI,
# ISA support, or ownership and lifetime correctness.
# This module is a policy gate, NOT an executable loader. The binder must
# still load the exact checked symbol and retain its DeviceContext/lifetimes.
from ..internal.status import TensorStatus


def checked_release_gemm_binding(
    flavor: Int, variant: Int, target_arch: Int,
    abi_version: Int, threads_per_block: Int,
) -> TensorStatus:
    # The unbuilt source distribution contains zero precompiled TF32 images.
    # Never infer native instructions from TF32_ALLOWED, sm89, or a Bool flag.
    return TensorStatus.UNSUPPORTED
