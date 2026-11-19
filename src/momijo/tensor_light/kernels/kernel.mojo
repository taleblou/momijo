# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR KERNEL / KERNEL CONTRACTS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Kernels
# Module:        momijo.tensor_light.kernels.kernel
# Source:        src/momijo/tensor_light/kernels/kernel.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines fixed-FP32 kernel signature metadata and operation-signature
#   matching.
#
#   This file defines or validates metadata; it does not by itself prove that
#   a native artifact is loaded, authenticated, or executed.
#
# Responsibilities:
#   - Define and validate the declared configuration, launch, or ABI
#     constraints for this module.
#   - Expose source-defined declarations including `KernelSignature`,
#     `make_kernel_signature`, `kernel_signature_matches`.
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

# Fixed-FP32 AOT signature metadata. Signatures are checked descriptions,
# NOT proof that a native kernel artifact has been loaded or authenticated.
# CPU, GPU and Hybrid are target mappings of the same operation signature.
from ..internal.status import TensorResult, TensorStatus
from ..device.target import TargetKind

@fieldwise_init
struct KernelSignature(Copyable):
    var operation: Int
    var rank: Int
    var math_mode: Int
    var target_kind: Int
    var shape_bucket: Int
    var alignment_class: Int

    @staticmethod
    def checked(
        operation: Int, rank: Int, math_mode: Int,
        target_kind: Int, shape_bucket: Int, alignment_class: Int,
    ) -> TensorResult[Self]:
        if rank < 0 or rank > 8:
            return TensorResult[Self](TensorStatus.INVALID_RANK)
        if operation < 0 or shape_bucket < 0 or shape_bucket > 4:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        # STRICT_FP32=0, FMA_FP32=1, TF32_ALLOWED=2, FASTEST=3.
        if math_mode < 0 or math_mode > 3:
            return TensorResult[Self](TensorStatus.MATH_MODE_UNSUPPORTED)
        if target_kind < TargetKind.CPU.code or target_kind > TargetKind.APPLE_GPU.code:
            return TensorResult[Self](TensorStatus.DEVICE_MISMATCH)
        # Zero denotes unknown alignment, not permission to assume aligned
        # SIMD/accelerator accesses. A known FP32 base alignment must be
        # at least 4 bytes and power-of-two.
        if alignment_class != 0:
            if alignment_class < 4 or alignment_class & (alignment_class - 1) != 0:
                return TensorResult[Self](TensorStatus.INVALID_ALIGNMENT)
        return TensorResult[Self](Self(
            operation, rank, math_mode, target_kind,
            shape_bucket, alignment_class,
        ))

    def checked_snapshot(self) -> TensorResult[Self]:
        # @fieldwise_init and mutable fields are not a validation proof.
        return Self.checked(
            self.operation, self.rank, self.math_mode,
            self.target_kind, self.shape_bucket, self.alignment_class,
        )


def make_kernel_signature(
    operation: Int, rank: Int, math_mode: Int,
    target_kind: Int, shape_bucket: Int, alignment_class: Int,
) -> TensorResult[KernelSignature]:
    return KernelSignature.checked(
        operation, rank, math_mode, target_kind, shape_bucket, alignment_class,
    )


def kernel_signature_matches(a: KernelSignature, b: KernelSignature) -> Bool:
    var checked_a = a.checked_snapshot()
    if not checked_a.is_ok():
        return False
    var checked_b = b.checked_snapshot()
    if not checked_b.is_ok():
        return False
    var x = checked_a.payload.value()
    var y = checked_b.payload.value()
    return (
        x.operation == y.operation and x.rank == y.rank
        and x.math_mode == y.math_mode and x.target_kind == y.target_kind
        and x.shape_bucket == y.shape_bucket
        and x.alignment_class == y.alignment_class
    )
