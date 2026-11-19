# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | GPU KERNEL / RUNTIME CONFIGURATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / GPU Tensor Kernels
# Module:        momijo.tensor_light.kernels.gpu.config
# Source:        src/momijo/tensor_light/kernels/gpu/config.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines static AOT GPU kernel configuration and target capability
#   metadata.
#
#   This file defines or validates metadata; it does not by itself prove that
#   a native artifact is loaded, authenticated, or executed.
#
# Responsibilities:
#   - Define and validate the declared configuration, launch, or ABI
#     constraints for this module.
#   - Expose source-defined declarations including `GpuKernelConfig`.
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

# Static AOT GPU kernel configuration. Hardware vendor is part of the
# target identity: an AMD gfx number must never pass an NVIDIA SM gate.
from ...internal.status import TensorResult, TensorStatus
from ...device.capabilities import GpuCapabilities
from ...device.target import TargetKind

@fieldwise_init
struct GpuKernelConfig(Copyable):
    # 1=NVIDIA; 2=AMD; 3=other. Never infer vendor from an integer arch.
    var vendor_code: Int
    var sm_arch: Int
    var threads_per_block: Int
    var warps_per_block: Int
    var tile_m: Int
    var tile_n: Int
    var tile_k: Int
    var shared_memory_bytes: Int
    var allow_tf32: Bool

    @staticmethod
    def checked_nvidia(
        sm_arch: Int, threads_per_block: Int, warps_per_block: Int,
        tile_m: Int, tile_n: Int, tile_k: Int, shared_memory_bytes: Int,
        allow_tf32: Bool,
    ) -> TensorResult[Self]:
        if sm_arch < 50 or sm_arch > 120:
            return TensorResult[Self](TensorStatus.UNSUPPORTED)
        if threads_per_block < 32 or threads_per_block > 1024:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if warps_per_block < 1 or warps_per_block > 32:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        # The current native ABI allocates only complete 32-thread NVIDIA
        # warps. Metadata and launch geometry must describe the same block.
        if threads_per_block % 32 != 0:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if warps_per_block != threads_per_block // 32:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if tile_m <= 0 or tile_n <= 0 or tile_k <= 0:
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        # Tile products are in FP32 elements. A checked kernel configuration
        # must never authorize a wrapped work/tile size or shared-memory span.
        # This proof happens once when selecting an offline/AOT variant.
        if tile_m > (Int.MAX // 4) // tile_n:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        if tile_m > (Int.MAX // 4) // tile_k:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        if tile_n > (Int.MAX // 4) // tile_k:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        if shared_memory_bytes < 0:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if allow_tf32 and sm_arch < 80:
            return TensorResult[Self](TensorStatus.MATH_MODE_UNSUPPORTED)
        return TensorResult[Self](Self(
            1, sm_arch, threads_per_block, warps_per_block, tile_m,
            tile_n, tile_k, shared_memory_bytes, allow_tf32,
        ))

    def validated_snapshot(self) -> TensorResult[Self]:
        # @fieldwise_init metadata can be mutated by the caller. Reconstruct
        # one canonical checked configuration before capability-based lowering.
        if self.vendor_code != 1:
            return TensorResult[Self](TensorStatus.DEVICE_MISMATCH)
        return Self.checked_nvidia(
            self.sm_arch, self.threads_per_block, self.warps_per_block,
            self.tile_m, self.tile_n, self.tile_k, self.shared_memory_bytes,
            self.allow_tf32,
        )

    def validate_for(self, capabilities: GpuCapabilities) -> TensorStatus:
        # This binds declared launch geometry to declared target capabilities,
        # NOT to a physical DeviceContext. The trusted AOT loader must still
        # authenticate the actual executable target and its ISA.
        var checked = self.validated_snapshot()
        if not checked.is_ok():
            return checked.status
        var target = capabilities.validate()
        if target != TensorStatus.OK:
            return target
        if capabilities.target_kind.code != TargetKind.NVIDIA_GPU.code:
            return TensorStatus.DEVICE_MISMATCH
        if capabilities.architecture_id != self.sm_arch:
            return TensorStatus.DEVICE_MISMATCH
        if self.threads_per_block > capabilities.max_threads_per_block:
            return TensorStatus.INVALID_LAYOUT
        if self.shared_memory_bytes > capabilities.shared_memory_bytes:
            return TensorStatus.INVALID_LAYOUT
        if self.allow_tf32 and not capabilities.supports_tf32():
            return TensorStatus.MATH_MODE_UNSUPPORTED
        return TensorStatus.OK

    def supports_hopper_wgmma(self) -> Bool:
        # Restrict the *FP32* candidate to SM90 and a mode permitting TF32.
        # A generic SM >= 90 shortcut would produce invalid Ada/Blackwell PTX.
        if not self.validated_snapshot().is_ok():
            return False
        return self.sm_arch == 90 and self.allow_tf32

    def supports_hopper_tma(self) -> Bool:
        # Hopper's TMA descriptor/barrier ABI is not implied by larger SM IDs.
        if not self.validated_snapshot().is_ok():
            return False
        return self.sm_arch == 90

    def supports_ada_cp_async(self) -> Bool:
        if not self.validated_snapshot().is_ok():
            return False
        return self.sm_arch == 89

    def supports_ada_tf32(self) -> Bool:
        if not self.validated_snapshot().is_ok():
            return False
        return self.sm_arch == 89 and self.allow_tf32
