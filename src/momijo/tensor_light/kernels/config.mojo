# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR KERNEL / RUNTIME CONFIGURATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Kernels
# Module:        momijo.tensor_light.kernels.config
# Source:        src/momijo/tensor_light/kernels/config.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines checked, finite CPU/FP32 kernel configurations and matrix tuning
#   defaults.
#
#   This file defines or validates metadata; it does not by itself prove that
#   a native artifact is loaded, authenticated, or executed.
#
# Responsibilities:
#   - Define and validate the declared configuration, launch, or ABI
#     constraints for this module.
#   - Expose source-defined declarations including `CpuKernelConfig`,
#     `MatrixKernelConfig`, `checked_cpu_kernel_config`,
#     `checked_matrix_kernel_config`.
#   - Keep metadata validation separate from trusted native binding and
#     execution.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate target, shape, ABI, or configuration invariants implemented by
#     these declarations.
#   - Treat an identifier, signature, or valid plan as metadata rather than
#     proof of executable native code.
#
# Performance Objectives:
#   - Keep static configuration and validation inexpensive at plan and launch
#     boundaries.
#   - Avoid kernel compilation, device migration, or per-element work in
#     metadata utilities.
#
# Maintainability:
#   - Keep signatures, launch constraints, and device-specific consumers
#     consistent.
#   - Test invalid signatures, target mismatches, and unsupported
#     configuration cases.
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

# Small finite family of precompiled FP32 kernel shapes.
from ..internal.status import TensorResult, TensorStatus
from ..device.capabilities import CpuCapabilities
@fieldwise_init
struct CpuKernelConfig(Copyable):
    var vector_width: Int
    var unroll: Int
    var mc: Int
    var nc: Int
    var kc: Int

@fieldwise_init
struct MatrixKernelConfig(Copyable):
    var tile_m: Int
    var tile_n: Int
    var tile_k: Int
    var tf32_allowed: Bool

def checked_cpu_kernel_config(
    candidate: CpuKernelConfig, target: CpuCapabilities,
) -> TensorResult[CpuKernelConfig]:
    var valid = target.validate()
    if valid != TensorStatus.OK:
        return TensorResult[CpuKernelConfig](valid)
    if candidate.vector_width <= 0 or candidate.unroll <= 0:
        return TensorResult[CpuKernelConfig](TensorStatus.INVALID_LAYOUT)
    if candidate.vector_width & (candidate.vector_width - 1) != 0:
        # Unsupported SIMD vectors must never enter the target-native AOT path.
        return TensorResult[CpuKernelConfig](TensorStatus.INVALID_LAYOUT)
    if candidate.mc <= 0 or candidate.nc <= 0 or candidate.kc <= 0:
        return TensorResult[CpuKernelConfig](TensorStatus.INVALID_SHAPE)
    if candidate.vector_width > Int.MAX // 32:
        return TensorResult[CpuKernelConfig](TensorStatus.OVERFLOW)
    if candidate.vector_width * 32 > target.vector_bits:
        return TensorResult[CpuKernelConfig](TensorStatus.UNSUPPORTED)
    # An unrolled vector group is a single AOT scheduling unit; even a
    # malformed offline tuning record must not overflow its element/byte span.
    if candidate.unroll > (Int.MAX // 4) // candidate.vector_width:
        return TensorResult[CpuKernelConfig](TensorStatus.OVERFLOW)
    # Packed FP32 microkernel panels and result tiles must be representable.
    if candidate.mc > (Int.MAX // 4) // candidate.kc:
        return TensorResult[CpuKernelConfig](TensorStatus.OVERFLOW)
    if candidate.kc > (Int.MAX // 4) // candidate.nc:
        return TensorResult[CpuKernelConfig](TensorStatus.OVERFLOW)
    if candidate.mc > (Int.MAX // 4) // candidate.nc:
        return TensorResult[CpuKernelConfig](TensorStatus.OVERFLOW)
    return TensorResult[CpuKernelConfig](candidate.copy())


def checked_matrix_kernel_config(candidate: MatrixKernelConfig) -> TensorStatus:
    if candidate.tile_m <= 0 or candidate.tile_n <= 0 or candidate.tile_k <= 0:
        return TensorStatus.INVALID_SHAPE
    if candidate.tile_m > (Int.MAX // 4) // candidate.tile_n:
        return TensorStatus.OVERFLOW
    if candidate.tile_m > (Int.MAX // 4) // candidate.tile_k:
        return TensorStatus.OVERFLOW
    if candidate.tile_n > (Int.MAX // 4) // candidate.tile_k:
        return TensorStatus.OVERFLOW
    return TensorStatus.OK


def avx2_fp32_gemm_defaults() -> CpuKernelConfig:
    # Offline-known AVX2 candidate. The caller must gate actual ISA use
    # against CpuCapabilities before loading any target-native artifact.
    return CpuKernelConfig(8, 4, 128, 128, 256)
