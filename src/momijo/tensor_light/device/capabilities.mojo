# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DEVICE / DEVICE CAPABILITIES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Device and Stream Control
# Module:        momijo.tensor_light.device.capabilities
# Source:        src/momijo/tensor_light/device/capabilities.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines device capabilities in the device and stream control layer of the
#   FP32 Light tensor implementation.
#
#   Public declarations include `CpuCapabilities`, `GpuCapabilities`,
#   `supports_fp32_simd`, `supports_fma`; preconditions and supported targets
#   are determined by their implementations.
#
# Responsibilities:
#   - Represent checked device/stream/event state and compatibility.
#   - Preserve documented behavior for explicit device/context identity and
#     completion requirements.
#   - Provide source-declared interfaces such as `CpuCapabilities`,
#     `GpuCapabilities`, `supports_fp32_simd`, `supports_fma`,
#     `cpu_vector_bits` and related symbols.
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

# Backend capability facts are selected before AOT lowering.
from .target import TargetKind
from ..internal.status import TensorStatus, TensorResult
@fieldwise_init
struct CpuCapabilities(Copyable):
    var vector_bits: Int
    var has_avx2: Bool
    var has_avx512: Bool
    var has_fma: Bool
    var workers: Int

    def validate(self) -> TensorStatus:
        if self.vector_bits < 0 or self.workers < 1:
            return TensorStatus.INVALID_LAYOUT
        # A vector length is an integer count of FP32 lanes; malformed ISA
        # metadata must not select an impossible AOT SIMD width.
        if self.vector_bits % 32 != 0:
            return TensorStatus.INVALID_LAYOUT
        if self.has_avx2 and self.vector_bits < 256:
            return TensorStatus.INVALID_LAYOUT
        if self.has_avx512 and self.vector_bits < 512:
            return TensorStatus.INVALID_LAYOUT
        return TensorStatus.OK

@fieldwise_init
struct GpuCapabilities(Copyable):
    # Architecture numbers alone cannot distinguish NVIDIA sm90 from a
    # differently numbered AMD/Apple target. Feature flags MUST bind vendor.
    var target_kind: TargetKind
    var architecture_id: Int
    var warp_size: Int
    var max_threads_per_block: Int
    var shared_memory_bytes: Int
    var supports_async_copy: Bool
    var supports_concurrent_copy_compute: Bool

    def validate(self) -> TensorStatus:
        # This verifies the SHAPE of declared capabilities, not the physical
        # identity of a DeviceContext. Binding to the actual device remains
        # a responsibility of the trusted AOT loader/backend adapter.
        if self.target_kind.code < 1 or self.target_kind.code > 3:
            return TensorStatus.DEVICE_MISMATCH
        if self.architecture_id <= 0 or self.warp_size <= 0:
            return TensorStatus.INVALID_LAYOUT
        if self.warp_size & (self.warp_size - 1) != 0:
            return TensorStatus.INVALID_LAYOUT
        if self.max_threads_per_block < self.warp_size or self.max_threads_per_block > 4096:
            return TensorStatus.INVALID_LAYOUT
        if self.shared_memory_bytes < 0:
            return TensorStatus.INVALID_LAYOUT
        if self.target_kind.code == TargetKind.NVIDIA_GPU.code:
            if self.warp_size != 32:
                return TensorStatus.INVALID_LAYOUT
            # The declared hardware ceiling is at most 1024 threads/block
            # on the supported NVIDIA lowering. A fake larger ceiling must
            # not authorize an invalid native grid/block shape.
            if self.max_threads_per_block > 1024:
                return TensorStatus.INVALID_LAYOUT
            if self.max_threads_per_block % 32 != 0:
                return TensorStatus.INVALID_LAYOUT
        return TensorStatus.OK

    def supports_tf32(self) -> Bool:
        return (self.validate() == TensorStatus.OK
                and self.target_kind.code == TargetKind.NVIDIA_GPU.code
                and self.architecture_id >= 80)

    def supports_wgmma(self) -> Bool:
        # Warp-group MMA is a Hopper SM90-family feature. A future NVIDIA
        # architecture does NOT automatically implement the same PTX ISA:
        # Blackwell uses its own matrix-engine lowering (e.g. tcgen05).
        # A separately authenticated AOT target/feature probe is still needed.
        return (self.validate() == TensorStatus.OK
                and self.target_kind.code == TargetKind.NVIDIA_GPU.code
                and self.architecture_id == 90)

    def supports_tma(self) -> Bool:
        # The Hopper TMA instruction contract is not a monotonic property
        # of numerical SM IDs. In particular, do not infer a compatible
        # descriptor/mbarrier ABI for future or Blackwell architectures.
        # Other families require a separately authenticated lowering.
        return (self.validate() == TensorStatus.OK
                and self.target_kind.code == TargetKind.NVIDIA_GPU.code
                and self.architecture_id == 90)


# Checked, vendor-aware capability queries over an explicitly supplied target
# descriptor. This is policy metadata, NOT a hardware discovery or artifact
# authentication mechanism. Callers must bind it to their actual device.
def supports_fp32_simd(capabilities: CpuCapabilities, lanes: Int) -> Bool:
    if capabilities.validate() != TensorStatus.OK or lanes < 1:
        return False
    # The AOT vector family uses scalar or power-of-two lane counts.
    # Arbitrary lane counts are not a legal native SIMD specialization key.
    if lanes & (lanes - 1) != 0:
        return False
    if lanes > Int.MAX // 32:
        return False
    return lanes * 32 <= capabilities.vector_bits


def supports_fma(capabilities: CpuCapabilities) -> Bool:
    return capabilities.validate() == TensorStatus.OK and capabilities.has_fma


def cpu_vector_bits(capabilities: CpuCapabilities) -> Int:
    if capabilities.validate() != TensorStatus.OK:
        return 0
    return capabilities.vector_bits


def gpu_warp_size(capabilities: GpuCapabilities) -> Int:
    if capabilities.validate() != TensorStatus.OK:
        return 0
    return capabilities.warp_size


def gpu_max_threads_per_block(capabilities: GpuCapabilities) -> Int:
    if capabilities.validate() != TensorStatus.OK:
        return 0
    return capabilities.max_threads_per_block


def gpu_shared_memory_limit(capabilities: GpuCapabilities) -> Int:
    if capabilities.validate() != TensorStatus.OK:
        return 0
    return capabilities.shared_memory_bytes


def supports_async_copy(capabilities: GpuCapabilities) -> Bool:
    # An advertised feature is useful only after the host backend confirms
    # this record refers to its actual accelerator/compiled artifact.
    return capabilities.validate() == TensorStatus.OK and capabilities.supports_async_copy


def supports_concurrent_copy_compute(capabilities: GpuCapabilities) -> Bool:
    return (capabilities.validate() == TensorStatus.OK
            and capabilities.supports_concurrent_copy_compute)


def supports_fp32_matrix_engine(capabilities: GpuCapabilities, allow_tf32: Bool) -> Bool:
    # Native strict-FP32 matrix instructions are not inferred from the
    # presence of a TF32 tensor core. Only advertise the known TF32 path.
    return allow_tf32 and capabilities.supports_tf32()


def supports_concurrent_host_device_execution(capabilities: GpuCapabilities) -> Bool:
    # CPU compute can overlap an asynchronously enqueued GPU kernel even on
    # devices WITHOUT a dedicated independent DMA copy engine. The older
    # condition wrongly equated host/device compute overlap with simultaneous
    # GPU copy/compute overlap, incorrectly disabling legal HYBRID_SPLIT.
    # Transfer-heavy HYBRID_PIPELINE is still gated separately by
    # supports_concurrent_copy_compute(). A trusted DeviceContext/backend
    # binder must establish that native enqueue is asynchronous; a freely
    # writable capability record is only a planning hint, not execution proof.
    return (capabilities.validate() == TensorStatus.OK
            and capabilities.max_threads_per_block >= capabilities.warp_size)
