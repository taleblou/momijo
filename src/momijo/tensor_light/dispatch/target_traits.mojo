# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DISPATCH / TARGET TRAITS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Dispatch and Specialization
# Module:        momijo.tensor_light.dispatch.target_traits
# Source:        src/momijo/tensor_light/dispatch/target_traits.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines target traits in the dispatch and specialization layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `supports_avx2_fma`, `supports_avx512`,
#   `supports_tf32_mma`, `supports_wgmma`; preconditions and supported targets
#   are determined by their implementations.
#
# Responsibilities:
#   - Validate dispatch choices against target and artifact contracts.
#   - Preserve documented behavior for immutable plan metadata and trusted
#     dispatch preconditions.
#   - Provide source-declared interfaces such as `supports_avx2_fma`,
#     `supports_avx512`, `supports_tf32_mma`, `supports_wgmma`, `supports_tma`
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

# Architecture-specific instruction gating for AOT strategy selection.
# The GPU capability includes a vendor identity. An integer ISA number alone
# must not authorize an NVIDIA-only native instruction on another GPU family.
from ..device.capabilities import GpuCapabilities, CpuCapabilities
from ..device.target import TargetKind
from ..internal.status import TensorStatus, TensorResult
def supports_avx2_fma(isa_features: Int) -> Bool:
    return (isa_features & 3) == 3

def supports_avx512(isa_features: Int) -> Bool:
    return (isa_features & 4) != 0

def supports_tf32_mma(capabilities: GpuCapabilities) -> Bool:
    return capabilities.supports_tf32()

def supports_wgmma(capabilities: GpuCapabilities) -> Bool:
    return capabilities.supports_wgmma()

def supports_tma(capabilities: GpuCapabilities) -> Bool:
    return capabilities.supports_tma()

def preferred_vector_width(capabilities: CpuCapabilities) -> Int:
    # FP32 lanes: one 32-bit element per lane. Only a verified target record
    # may select an AOT width; return scalar for invalid capability metadata.
    if capabilities.validate() != TensorStatus.OK:
        return 1
    if capabilities.has_avx512:
        return 16
    if capabilities.has_avx2:
        return 8
    # Portable 128-bit FP32 SIMD candidates remain useful on SSE/NEON
    # targets when their AOT package actually contains that lowering.
    if capabilities.vector_bits >= 128:
        return 4
    return 1


def supports_gpu_async_copy(capabilities: GpuCapabilities) -> Bool:
    return (capabilities.validate() == TensorStatus.OK
            and capabilities.supports_async_copy)


# These closed-world source gates select only target-native instruction
# families that the AOT profile explicitly recognizes. ISA numbers are not
# monotone capability promises: sm_89 must never receive sm_90 WGMMA/TMA;
# a future sm_100+ requires separately authenticated machine-code lowerings.
# The caller MUST still bind declared metadata to the physical DeviceContext
# and its already-built artifact (a metadata record cannot authenticate PTX).
@fieldwise_init
struct NvidiaInstructionFamily(Copyable, Equatable):
    var code: Int
    comptime SIMT_FP32 = Self(0)
    comptime CP_ASYNC = Self(1)
    comptime TF32_MMA = Self(2)
    comptime HOPPER_WGMMA = Self(3)
    comptime HOPPER_TMA = Self(4)

    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code


def recognized_nvidia_sm_aot_target(capabilities: GpuCapabilities) -> Bool:
    if capabilities.validate() != TensorStatus.OK:
        return False
    if capabilities.target_kind.code != TargetKind.NVIDIA_GPU.code:
        return False
    var sm = capabilities.architecture_id
    # Deliberately conservative allowlist, rather than >= 80 / >= 90:
    # additional sm targets need their own AOT-validated feature profiles.
    return (sm == 80 or sm == 86 or sm == 87 or sm == 89 or sm == 90)


def checked_nvidia_fp32_feature(
    capabilities: GpuCapabilities, family: NvidiaInstructionFamily,
    fp32_math_mode: Int,
) -> TensorResult[Bool]:
    # 0 strict FP32, 1 FMA FP32, 2 TF32_ALLOWED, 3 FASTEST.
    if fp32_math_mode < 0 or fp32_math_mode > 3:
        return TensorResult[Bool](TensorStatus.MATH_MODE_UNSUPPORTED)
    if family.code < 0 or family.code > 4:
        return TensorResult[Bool](TensorStatus.UNSUPPORTED)
    if capabilities.validate() != TensorStatus.OK:
        return TensorResult[Bool](TensorStatus.INVALID_LAYOUT)
    if not recognized_nvidia_sm_aot_target(capabilities):
        return TensorResult[Bool](False)
    var sm = capabilities.architecture_id
    if family == NvidiaInstructionFamily.SIMT_FP32:
        return TensorResult[Bool](True)
    if family == NvidiaInstructionFamily.CP_ASYNC:
        # The target must also have a confirmed asynchronous-copy lowering.
        return TensorResult[Bool](capabilities.supports_async_copy)
    if family == NvidiaInstructionFamily.TF32_MMA:
        return TensorResult[Bool](
            fp32_math_mode >= 2 and capabilities.supports_tf32()
        )
    if family == NvidiaInstructionFamily.HOPPER_WGMMA:
        return TensorResult[Bool](
            sm == 90 and fp32_math_mode >= 2 and capabilities.supports_wgmma()
        )
    # sm90-only Hopper TMA descriptor ABI; no automatic future ISA inheritance.
    return TensorResult[Bool](sm == 90 and capabilities.supports_tma())
