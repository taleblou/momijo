# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DISPATCH / CPU TUNING PROFILES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Dispatch and Specialization
# Module:        momijo.tensor_light.dispatch.cpu_tuning_profile
# Source:        src/momijo/tensor_light/dispatch/cpu_tuning_profile.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines CPU tuning profiles in the dispatch and specialization layer of
#   the FP32 Light tensor implementation.
#
#   Public declarations include `CpuOfflineAOTProfile`,
#   `choose_cpu_elementwise_with_offline_profile`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Validate dispatch choices against target and artifact contracts.
#   - Preserve documented behavior for immutable plan metadata and trusted
#     dispatch preconditions.
#   - Provide source-declared interfaces such as `CpuOfflineAOTProfile`,
#     `choose_cpu_elementwise_with_offline_profile`.
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

# UASDEA CPU target policy: immutable, offline-supplied finite AOT thresholds.
# This is a source-level speed policy, NOT an empirical proof of optimality.
# Only exact compiled scalar/unaligned-AVX2/aligned-AVX2 variants may run.
# The alignment/alias/ISA proof remains in specialization.mojo; a numeric
# profile cannot enable a vector instruction or assert noalias on its own.
from std.collections import Span
from ..internal.status import TensorResult, TensorStatus
from ..device.capabilities import CpuCapabilities
from .dispatch_key import DispatchKey
from .specialization import (
    SpecializationDecision, choose_avx2_elementwise_for_spans,
    choose_checked_scalar_elementwise_for_spans,
)


@fieldwise_init
struct CpuOfflineAOTProfile(Copyable):
    # Minimum element counts to amortize SIMD and a more strongly aligned
    # memory path on the explicitly targeted CPU. Offline measurements may
    # supply these once per release; NO runtime benchmark/JIT is involved.
    var vector_from: Int
    var aligned_from: Int
    # Finite upper bound prevents nonsensical corrupted policy metadata.
    var target_vector_lanes: Int
    var target_alignment_bytes: Int

    @staticmethod
    def checked(
        vector_from: Int, aligned_from: Int,
        target_vector_lanes: Int, target_alignment_bytes: Int,
    ) -> TensorResult[Self]:
        # Current AVX2/FMA AOT ABI has 8 FP32 lanes, aligned variant uses
        # 32-byte aligned output AND both input pointers. New ISA families
        # require their own separately compiled ABI and checked selector.
        if target_vector_lanes != 8 or target_alignment_bytes != 32:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        if (vector_from < 8 or aligned_from < vector_from
            or aligned_from > Int.MAX // 4):
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        return TensorResult[Self](Self(
            vector_from, aligned_from, target_vector_lanes,
            target_alignment_bytes,
        ))

    def checked_snapshot(self) -> TensorResult[Self]:
        return Self.checked(
            self.vector_from, self.aligned_from,
            self.target_vector_lanes, self.target_alignment_bytes,
        )


def choose_cpu_elementwise_with_offline_profile[
    out_origin: Origin[mut=True], left_origin: Origin, right_origin: Origin,
](
    key: DispatchKey, capabilities: CpuCapabilities,
    result: Span[True, Float32, out_origin],
    left: Span[False, Float32, left_origin],
    right: Span[False, Float32, right_origin],
    profile: CpuOfflineAOTProfile,
) -> TensorResult[SpecializationDecision]:
    var policy = profile.checked_snapshot()
    if not policy.is_ok():
        return TensorResult[SpecializationDecision](policy.status)
    # One scalar fallback works on CPUs without AVX2/FMA too, but it MUST
    # prove that a forward store cannot overwrite an unread input. The proof
    # is shared with AVX2; no alias-sensitive early return on a tiny count.
    var scalar = choose_checked_scalar_elementwise_for_spans(
        key, capabilities, result, left, right,
    )
    if not scalar.is_ok():
        return scalar
    var count = len(result)
    var thresholds = policy.payload.value()
    if count < thresholds.vector_from:
        return scalar
    if not capabilities.has_avx2 or not capabilities.has_fma:
        return scalar
    # Native AVX2 is selected only after actual pointer alignment and ISA
    # proof; a numeric profile cannot authorize a compiled instruction.
    var baseline = choose_avx2_elementwise_for_spans(
        key, capabilities, result, left, right,
    )
    if not baseline.is_ok():
        return baseline
    var selected = baseline.payload.value()
    if (selected.aot_variant == 2
        and count < thresholds.aligned_from):
        return SpecializationDecision.checked(1, 8, 1, 1, 1)
    return selected.checked_snapshot()
