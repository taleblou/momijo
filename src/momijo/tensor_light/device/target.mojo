# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DEVICE / TARGET METADATA
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Device and Stream Control
# Module:        momijo.tensor_light.device.target
# Source:        src/momijo/tensor_light/device/target.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines target metadata in the device and stream control layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `TargetKind`, `TargetSpec`,
#   `validate_target_spec`, `target_is_cpu`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Represent checked device/stream/event state and compatibility.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `TargetKind`, `TargetSpec`,
#     `validate_target_spec`, `target_is_cpu`, `target_is_gpu` and related
#     symbols.
#   - Reject unsupported target or unverified completion states.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Match device ordinals, contexts, streams, and events explicitly.
#   - Do not treat an unrecorded event as proof of completion.
#   - Reject unsupported or unauthenticated execution paths explicitly.
#
# Performance Objectives:
#   - Avoid unnecessary synchronization outside required boundaries.
#   - Keep capability queries independent from data movement.
#   - Avoid implicit host/device migration or compilation at execution time.
#
# Maintainability:
#   - Differentiate structural device metadata from active resources.
#   - Test mismatched device/stream/context paths.
#   - Keep device-specific capability statements conservative.
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

# CPU/GPU are target mappings of the same static dataflow architecture.
@fieldwise_init
struct TargetKind(Copyable):
    var code: Int
    comptime CPU = Self(0)
    comptime NVIDIA_GPU = Self(1)
    comptime AMD_GPU = Self(2)
    comptime APPLE_GPU = Self(3)

@fieldwise_init
struct TargetSpec(Copyable):
    var kind: TargetKind
    var architecture_id: Int
    var simd_lanes: Int
    var shared_memory_bytes: Int


def validate_target_spec(spec: TargetSpec) -> Bool:
    if spec.kind.code < TargetKind.CPU.code or spec.kind.code > TargetKind.APPLE_GPU.code:
        return False
    if spec.architecture_id < 0 or spec.simd_lanes < 0 or spec.shared_memory_bytes < 0:
        return False
    if spec.kind.code == TargetKind.CPU.code:
        # An AOT CPU target uses scalar or native power-of-two FP32 vectors;
        # arbitrary lane counts cannot identify a valid SIMD specialization.
        if spec.shared_memory_bytes != 0 or spec.simd_lanes <= 0:
            return False
        if spec.simd_lanes > Int.MAX // 32:
            return False
        return spec.simd_lanes & (spec.simd_lanes - 1) == 0
    return spec.architecture_id > 0


def target_is_cpu(spec: TargetSpec) -> Bool:
    return validate_target_spec(spec) and spec.kind.code == TargetKind.CPU.code


def target_is_gpu(spec: TargetSpec) -> Bool:
    return validate_target_spec(spec) and spec.kind.code != TargetKind.CPU.code


def target_vector_bits(spec: TargetSpec) -> Int:
    # Exposes declared AOT target vector width, never probes the active CPU.
    if not target_is_cpu(spec) or spec.simd_lanes > Int.MAX // 32:
        return 0
    return spec.simd_lanes * 32
