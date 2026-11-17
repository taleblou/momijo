# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DTYPE / ELEMENT TRAITS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / FP32 DType and Math Policy
# Module:        momijo.tensor_light.dtype.traits
# Source:        src/momijo/tensor_light/dtype/traits.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines element traits in the FP32 dtype and math policy layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `Fp32Traits`, `default_fp32_traits`,
#   `preferred_vector_width`, `is_trivially_copyable_element`; preconditions
#   and supported targets are determined by their implementations.
#
# Responsibilities:
#   - Describe fixed-FP32 element size, SIMD lane count, and CPU FMA
#     capability.
#   - Expose checked alignment-dependent load-width hints.
#   - Keep native load recommendations separate from proof of pointer
#     alignment.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Derive FMA support from validated CPU capability metadata.
#   - Reject invalid alignment hints rather than certifying unproven aligned
#     loads.
#   - Do not infer device eligibility from the 4-byte element representation
#     alone.
#
# Performance Objectives:
#   - Use compile-time SIMD width where available.
#   - Keep capability and alignment checks outside repeated element
#     operations.
#
# Maintainability:
#   - Keep load-width hints consistent with CPU kernel alignment
#     preconditions.
#   - Test invalid, non-power-of-two, and supported alignment boundaries.
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

# Compile-time native FP32 SIMD policy, always element size 4.
from std.sys.info import simd_width_of
from ..device.capabilities import CpuCapabilities, GpuCapabilities
from ..internal.status import TensorStatus, TensorResult

@fieldwise_init
struct Fp32Traits(Copyable):
    var simd_width: Int
    var byte_width: Int
    var supports_fma: Bool

def default_fp32_traits(capabilities: CpuCapabilities) -> Fp32Traits:
    # ISA-dependent FMA support must never default to True on every host.
    var lanes = simd_width_of[DType.float32]()
    var allowed_fma = capabilities.validate() == TensorStatus.OK and capabilities.has_fma
    return Fp32Traits(lanes, lanes * 4, allowed_fma)

# No runtime dtype branch: all elements have the same 4-byte representation.
def preferred_vector_width() -> Int:
    return simd_width_of[DType.float32]()


def is_trivially_copyable_element() -> Bool:
    return True


def preferred_load_width_bytes(alignment: Int) -> Int:
    # This PUBLIC SIMD hint receives caller-controlled alignment metadata.
    # Returning a four-byte *aligned* load for 0, 1, 2, 3, a negative number
    # or a non-power-of-two value incorrectly certifies a native alignment
    # that was never established. Zero is a fail-closed, no-kernel sentinel;
    # a checked caller can use preferred_load_width_bytes_checked() for the
    # exact status. For valid alignments the prior fast AOT policy is unchanged.
    if alignment < 4 or (alignment & (alignment - 1)) != 0:
        return 0
    if alignment >= 16:
        return 16
    if alignment >= 8:
        return 8
    return 4


def preferred_load_width_bytes_checked(alignment: Int) -> TensorResult[Int]:
    # Reuse the exact same structural gate as the nonfallible optimizer hint.
    # Neither function proves that an arbitrary host pointer is actually
    # mapped or aligned; that still belongs to the owning Span boundary.
    var selected = preferred_load_width_bytes(alignment)
    if selected == 0:
        return TensorResult[Int](TensorStatus.INVALID_ALIGNMENT)
    return TensorResult[Int](selected)

# Checked/AOT candidate gates; these do not prove a native binary was built.
def supports_native_fma(capabilities: CpuCapabilities) -> Bool:
    return capabilities.validate() == TensorStatus.OK and capabilities.has_fma


def supports_tf32_matrix_engine(capabilities: GpuCapabilities) -> Bool:
    return capabilities.supports_tf32()


def supports_fp32_matrix_engine(capabilities: GpuCapabilities, allow_tf32: Bool) -> Bool:
    # Do not confuse strict-FP32 SIMT with strict-FP32 matrix-engine support.
    # This target contract only has a declared TF32-capable matrix engine;
    # the actual TF32 kernel and trusted AOT ABI remain release gaps.
    return allow_tf32 and capabilities.supports_tf32()
