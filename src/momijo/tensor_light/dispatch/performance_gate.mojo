# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DISPATCH / PERFORMANCE GATE
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Dispatch and Specialization
# Module:        momijo.tensor_light.dispatch.performance_gate
# Source:        src/momijo/tensor_light/dispatch/performance_gate.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines performance gate in the dispatch and specialization layer of the
#   FP32 Light tensor implementation.
#
#   Public declarations include `Fp32CandidateEvidence`, `Fp32OfflineWinner`,
#   `choose_verified_fp32_candidate`,
#   `choose_verified_fp32_candidate_with_ram_caps`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Validate dispatch choices against target and artifact contracts.
#   - Preserve documented behavior for immutable plan metadata and trusted
#     dispatch preconditions.
#   - Provide source-declared interfaces such as `Fp32CandidateEvidence`,
#     `Fp32OfflineWinner`, `choose_verified_fp32_candidate`,
#     `choose_verified_fp32_candidate_with_ram_caps`.
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

# Immutable OFFLINE candidate ranking for the SAME fixed-Float32 workload.
# Run during PREPARE, never once per REPLAY or GEMM tile. This is deliberately
# not a kernel, profiler, benchmark runner, artifact signature verifier or GPU
# authorization. Evidence records must be backed by independent actual raw
# runs plus the trusted loader checks before the selected variant is shipped.
from std.utils.numerics import isfinite
from ..internal.status import TensorStatus, TensorResult


@fieldwise_init
struct Fp32CandidateEvidence(Copyable):
    var variant: Int
    # Same exact canonical workload fingerprint, output residency, target
    # class and arithmetic mode required for an apples-to-apples decision.
    var workload_id: Int
    # Nonzero manifest-bound environment key: exact hardware/OS/driver,
    # power profile, compiler and benchmark timing boundary. An ID is only
    # an equality guard, NEVER independent proof that a device was measured.
    var target_environment_id: Int
    # Nonzero held-out input fixture identifier, including data/seed hashes.
    # Prevent cross-input comparisons under a single shape/workload ID.
    var input_fixture_id: Int
    var output_residency: Int
    var math_mode_code: Int
    var rank: Int
    # Execution evidence metadata only: NOT a trust/owner proof by itself.
    var correctness_verified: Bool
    var independent_holdout_verified: Bool
    var artifact_verified: Bool
    var p95_ns: Float64
    var p95_ci_low_ns: Float64
    var p95_ci_high_ns: Float64
    var p99_ns: Float64
    var device_peak_bytes: Int
    var host_pinned_peak_bytes: Int

    def validate(self) -> TensorStatus:
        if (self.variant < 0 or self.workload_id < 0
            or self.target_environment_id <= 0 or self.input_fixture_id <= 0
            or self.output_residency < 0 or self.output_residency > 2
            or self.math_mode_code < 0 or self.math_mode_code > 3
            or self.rank < 0 or self.rank > 8
            or self.device_peak_bytes < 0 or self.host_pinned_peak_bytes < 0):
            return TensorStatus.INVALID_LAYOUT
        if (not Bool(isfinite(self.p95_ns))
            or not Bool(isfinite(self.p95_ci_low_ns))
            or not Bool(isfinite(self.p95_ci_high_ns))
            or not Bool(isfinite(self.p99_ns))):
            return TensorStatus.INVALID_LAYOUT
        if (self.p95_ci_low_ns <= 0.0 or self.p95_ns < self.p95_ci_low_ns
            or self.p95_ci_high_ns < self.p95_ns
            or self.p99_ns < self.p95_ns):
            return TensorStatus.INVALID_LAYOUT
        return TensorStatus.OK


@fieldwise_init
struct Fp32OfflineWinner(Copyable):
    var variant: Int
    var challenger_selected: Bool
    # False: lack of sufficient evidence/margin, incumbent retained.
    var conclusive: Bool


def choose_verified_fp32_candidate(
    incumbent: Fp32CandidateEvidence,
    challenger: Fp32CandidateEvidence,
    max_device_bytes: Int, max_pinned_host_bytes: Int,
    max_p99_ns: Float64, required_gain_bp: Int,
) -> TensorResult[Fp32OfflineWinner]:
    # Compare release candidates only under the IDENTICAL Float32 rank and
    # math-mode contract; a TF32 win cannot replace strict FP32 results.
    if (max_device_bytes < 0 or max_pinned_host_bytes < 0
        or required_gain_bp < 0 or required_gain_bp > 9999
        or not Bool(isfinite(max_p99_ns)) or max_p99_ns <= 0.0):
        return TensorResult[Fp32OfflineWinner](TensorStatus.INVALID_LAYOUT)
    var valid_base = incumbent.validate()
    if valid_base != TensorStatus.OK:
        return TensorResult[Fp32OfflineWinner](valid_base)
    if (not incumbent.correctness_verified
        or not incumbent.independent_holdout_verified
        or not incumbent.artifact_verified):
        return TensorResult[Fp32OfflineWinner](TensorStatus.UNSUPPORTED)
    if (incumbent.device_peak_bytes > max_device_bytes
        or incumbent.host_pinned_peak_bytes > max_pinned_host_bytes
        or incumbent.p99_ns > max_p99_ns):
        return TensorResult[Fp32OfflineWinner](TensorStatus.NOT_REPRESENTABLE)
    var fallback = Fp32OfflineWinner(incumbent.variant, False, False)
    if challenger.validate() != TensorStatus.OK:
        return TensorResult[Fp32OfflineWinner](fallback)
    # The variant is the executable candidate identity in this selector.
    # A repeated observation of the SAME ID is not a new deployable kernel,
    # even if the supplied timing numbers appear better. Treat this as
    # inconclusive instead of emitting a fictitious challenger promotion.
    if challenger.variant == incumbent.variant:
        return TensorResult[Fp32OfflineWinner](fallback)
    if (challenger.workload_id != incumbent.workload_id
        or challenger.target_environment_id != incumbent.target_environment_id
        or challenger.input_fixture_id != incumbent.input_fixture_id
        or challenger.output_residency != incumbent.output_residency
        or challenger.math_mode_code != incumbent.math_mode_code
        or challenger.rank != incumbent.rank):
        return TensorResult[Fp32OfflineWinner](fallback)
    if (not challenger.correctness_verified
        or not challenger.independent_holdout_verified
        or not challenger.artifact_verified):
        return TensorResult[Fp32OfflineWinner](fallback)
    if (challenger.device_peak_bytes > max_device_bytes
        or challenger.host_pinned_peak_bytes > max_pinned_host_bytes
        or challenger.p99_ns > max_p99_ns):
        return TensorResult[Fp32OfflineWinner](fallback)
    # Uncertainty-aware conservative threshold: challenger's UPPER p95 CI
    # must beat the incumbent's LOWER CI by a user-specified practical margin.
    # A faster mean, warm-up sample or isolated kernel cannot win this gate.
    var fraction = 1.0 - Float64(required_gain_bp) / 10000.0
    if challenger.p95_ci_high_ns < incumbent.p95_ci_low_ns * fraction:
        return TensorResult[Fp32OfflineWinner](Fp32OfflineWinner(
            challenger.variant, True, True,
        ))
    return TensorResult[Fp32OfflineWinner](fallback)


def choose_verified_fp32_candidate_with_ram_caps(
    incumbent: Fp32CandidateEvidence,
    challenger: Fp32CandidateEvidence,
    incumbent_peak_ram_bytes: Int,
    challenger_peak_ram_bytes: Int,
    max_ram_bytes: Int,
    max_device_bytes: Int,
    max_pinned_host_bytes: Int,
    max_p99_ns: Float64,
    required_gain_bp: Int,
) -> TensorResult[Fp32OfflineWinner]:
    # BEN-115/TEST-140: the existing selector enforces device and PINNED
    # host-memory ceilings. Pinned memory is NOT total RAM, so this additive
    # offline policy also requires separate, measured TOTAL host RAM peaks.
    # This function cannot certify the authenticity of the supplied evidence.
    # Existing ABI/call sites of choose_verified_fp32_candidate are unchanged.
    if (max_ram_bytes <= 0 or incumbent_peak_ram_bytes < 0
        or challenger_peak_ram_bytes < 0):
        return TensorResult[Fp32OfflineWinner](TensorStatus.INVALID_LAYOUT)
    if incumbent_peak_ram_bytes < incumbent.host_pinned_peak_bytes:
        return TensorResult[Fp32OfflineWinner](TensorStatus.INVALID_LAYOUT)
    var base = choose_verified_fp32_candidate(
        incumbent, challenger, max_device_bytes, max_pinned_host_bytes,
        max_p99_ns, required_gain_bp,
    )
    if not base.is_ok():
        return base
    if incumbent_peak_ram_bytes > max_ram_bytes:
        return TensorResult[Fp32OfflineWinner](TensorStatus.NOT_REPRESENTABLE)
    if (challenger_peak_ram_bytes < challenger.host_pinned_peak_bytes
        or challenger_peak_ram_bytes > max_ram_bytes):
        return TensorResult[Fp32OfflineWinner](
            Fp32OfflineWinner(incumbent.variant, False, False)
        )
    return base
