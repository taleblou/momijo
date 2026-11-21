# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DTYPE / FLOATING-POINT MATH MODES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / FP32 DType and Math Policy
# Module:        momijo.tensor_light.dtype.math_mode
# Source:        src/momijo/tensor_light/dtype/math_mode.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines floating-point math modes in the FP32 dtype and math policy layer
#   of the FP32 Light tensor implementation.
#
#   Public declarations include `Fp32MathMode`, `ReductionAccuracyMode`,
#   `tf32_permitted`, `allows_fma`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Represent declared Float32 computation and reduction-accuracy modes.
#   - Determine when FMA contraction, reassociation, or TF32 is permitted.
#   - Require vendor-bound device capabilities before authorizing TF32
#     selection.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Treat mode selection as a permission policy, not as arithmetic
#     execution.
#   - Do not enable TF32 from untrusted numeric architecture IDs alone.
#   - Keep reassociation permission distinct from FMA contraction.
#
# Performance Objectives:
#   - Keep math-mode and device-capability selection inexpensive.
#   - Allow AOT specialization without adding per-element policy branches.
#
# Maintainability:
#   - Keep reduction and arithmetic policies aligned with their consuming
#     kernels.
#   - Test strict/FMA/TF32/fast-mode behavior and unsupported target
#     capability cases.
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

# Storage is always Float32; TF32 is only an optional compute precision.
from ..device.capabilities import GpuCapabilities
from ..internal.status import TensorStatus, TensorResult
@fieldwise_init
struct Fp32MathMode(ImplicitlyCopyable, Equatable):
    var code: Int
    comptime STRICT_FP32 = Self(0)
    comptime FMA_FP32 = Self(1)
    comptime TF32_ALLOWED = Self(2)
    comptime FASTEST = Self(3)
    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

@fieldwise_init
struct ReductionAccuracyMode(Copyable):
    var code: Int
    comptime FAST_TREE = Self(0)
    comptime DETERMINISTIC_PAIRWISE = Self(1)

def tf32_permitted(mode: Fp32MathMode, capabilities: GpuCapabilities) -> Bool:
    # An untyped numeric architecture ID is not evidence of NVIDIA identity.
    # Use the vendor-bound capability contract before enabling TF32 compute.
    return capabilities.supports_tf32() and (
        mode == Fp32MathMode.TF32_ALLOWED or mode == Fp32MathMode.FASTEST
    )

# Reductions with strict reproducibility must not select a reassociating SIMD
# path merely because FASTEST is enabled. NaN/tie behavior is an op contract.

def allows_fma(mode: Fp32MathMode) -> Bool:
    return mode == Fp32MathMode.FMA_FP32 or mode == Fp32MathMode.TF32_ALLOWED or mode == Fp32MathMode.FASTEST


def allows_reassociation(mode: Fp32MathMode) -> Bool:
    # FMA contraction alone is not a blanket permission to reorder reductions.
    return mode == Fp32MathMode.FASTEST


def allows_tf32(mode: Fp32MathMode, capabilities: GpuCapabilities) -> Bool:
    # Numeric architecture IDs alone are insufficient: the target must also
    # be NVIDIA (as enforced by supports_tf32 on this capability type).
    return capabilities.supports_tf32() and (
        mode == Fp32MathMode.TF32_ALLOWED or mode == Fp32MathMode.FASTEST
    )


def normalize_math_mode(mode: Fp32MathMode, capabilities: GpuCapabilities) -> Fp32MathMode:
    # Compatibility function: normalization must NOT rewrite the caller's
    # declared numerical contract (TF32 -> FMA was a silent mode change).
    # Use checked_fp32_math_mode() to reject unavailable modes with status.
    return mode


def checked_fp32_math_mode(
    requested: Fp32MathMode, capabilities: GpuCapabilities,
) -> TensorResult[Fp32MathMode]:
    # An explicit precision request is never silently downgraded or promoted.
    # This is ONLY a vendor/ISA candidate check; the release loader must
    # independently attest its executable image, ABI and numeric oracle.
    if requested.code < 0 or requested.code > 3:
        return TensorResult[Fp32MathMode](TensorStatus.MATH_MODE_UNSUPPORTED)
    if requested == Fp32MathMode.TF32_ALLOWED:
        if not capabilities.supports_tf32():
            return TensorResult[Fp32MathMode](TensorStatus.MATH_MODE_UNSUPPORTED)
    # FASTEST allows the runtime to CHOOSE among individually attested
    # CPU/GPU variants; it does not REQUIRE any GPU or TF32 capability.
    return TensorResult[Fp32MathMode](requested)

# Pure offline policy boundary: the caller chooses whether stable pairwise
# association is required. Tensor payload stays Float32 in both cases.
def reduction_mode_for(deterministic: Bool) -> ReductionAccuracyMode:
    if deterministic:
        return materialize[ReductionAccuracyMode.DETERMINISTIC_PAIRWISE]()
    return materialize[ReductionAccuracyMode.FAST_TREE]()
