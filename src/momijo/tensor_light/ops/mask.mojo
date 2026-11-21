# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / BOOLEAN MASK OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Boolean Mask Operations
# Module:        momijo.tensor_light.ops.mask
# Source:        src/momijo/tensor_light/ops/mask.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines operations over Boolean masks without treating masks as ordinary
#   Float32 values.
#
#   Public declarations include `mask_and`, `mask_or`, `mask_xor`, `mask_not`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Implement operation-specific tensor semantics.
#   - Preserve documented behavior for Boolean mask semantics and tensor shape
#     compatibility.
#   - Provide source-declared interfaces such as `mask_and`, `mask_or`,
#     `mask_xor`, `mask_not`, `mask_any` and related symbols.
#   - Support mask logic and any/all/count reductions.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate operand shape, dtype, and storage access.
#   - Preserve operation-specific broadcasting and result contracts.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid materializing broadcasts unless output storage needs it.
#   - Prefer specialized or fused paths only when semantics match.
#
# Maintainability:
#   - Keep public operations aligned with underlying kernel semantics.
#   - Cover edge cases and maintain relevant regression tests.
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

# Public mask operations re-export the low-level auxiliary mask type.
# A core Tensor must never depend on the higher-level ops package.
from ..core.tensor_mask import TensorMask
from ..shape.shape import TensorShape
from ..internal.status import TensorStatus, TensorResult

# Logical mask ops are deliberately distinct from Float32 Tensor arithmetic.
def mask_and[rank: Int](a: TensorMask[rank], b: TensorMask[rank]) -> TensorResult[TensorMask[rank]]:
    var av = a.metadata_status()
    if av != TensorStatus.OK:
        return TensorResult[TensorMask[rank]](av)
    var bv = b.metadata_status()
    if bv != TensorStatus.OK:
        return TensorResult[TensorMask[rank]](bv)
    if not a.shape.same_extents(b.shape):
        return TensorResult[TensorMask[rank]](TensorStatus.SIZE_MISMATCH)
    var constructed = TensorMask[rank].zeros_checked(a.shape)
    if not constructed.is_ok():
        return TensorResult[TensorMask[rank]](constructed.status)
    var out = constructed.payload.take()
    for i in range(a.shape.numel()):
        var left = a.bytes[i]
        var right = b.bytes[i]
        if (left != 0 and left != 1) or (right != 0 and right != 1):
            return TensorResult[TensorMask[rank]](TensorStatus.INVALID_LAYOUT)
        if left != 0 and right != 0:
            out.bytes[i] = 1
    return TensorResult[TensorMask[rank]](out^)

def mask_or[rank: Int](a: TensorMask[rank], b: TensorMask[rank]) -> TensorResult[TensorMask[rank]]:
    var av = a.metadata_status()
    if av != TensorStatus.OK:
        return TensorResult[TensorMask[rank]](av)
    var bv = b.metadata_status()
    if bv != TensorStatus.OK:
        return TensorResult[TensorMask[rank]](bv)
    if not a.shape.same_extents(b.shape):
        return TensorResult[TensorMask[rank]](TensorStatus.SIZE_MISMATCH)
    var constructed = TensorMask[rank].zeros_checked(a.shape)
    if not constructed.is_ok():
        return TensorResult[TensorMask[rank]](constructed.status)
    var out = constructed.payload.take()
    for i in range(a.shape.numel()):
        var left = a.bytes[i]
        var right = b.bytes[i]
        if (left != 0 and left != 1) or (right != 0 and right != 1):
            return TensorResult[TensorMask[rank]](TensorStatus.INVALID_LAYOUT)
        if left != 0 or right != 0:
            out.bytes[i] = 1
    return TensorResult[TensorMask[rank]](out^)

def mask_xor[rank: Int](a: TensorMask[rank], b: TensorMask[rank]) -> TensorResult[TensorMask[rank]]:
    var av = a.metadata_status()
    if av != TensorStatus.OK:
        return TensorResult[TensorMask[rank]](av)
    var bv = b.metadata_status()
    if bv != TensorStatus.OK:
        return TensorResult[TensorMask[rank]](bv)
    if not a.shape.same_extents(b.shape):
        return TensorResult[TensorMask[rank]](TensorStatus.SIZE_MISMATCH)
    var constructed = TensorMask[rank].zeros_checked(a.shape)
    if not constructed.is_ok():
        return TensorResult[TensorMask[rank]](constructed.status)
    var out = constructed.payload.take()
    for i in range(a.shape.numel()):
        var left = a.bytes[i]
        var right = b.bytes[i]
        if (left != 0 and left != 1) or (right != 0 and right != 1):
            return TensorResult[TensorMask[rank]](TensorStatus.INVALID_LAYOUT)
        if (left != 0) != (right != 0):
            out.bytes[i] = 1
    return TensorResult[TensorMask[rank]](out^)

def mask_not[rank: Int](a: TensorMask[rank]) -> TensorResult[TensorMask[rank]]:
    var status = a.metadata_status()
    if status != TensorStatus.OK:
        return TensorResult[TensorMask[rank]](status)
    var constructed = TensorMask[rank].zeros_checked(a.shape)
    if not constructed.is_ok():
        return TensorResult[TensorMask[rank]](constructed.status)
    var out = constructed.payload.take()
    for i in range(a.shape.numel()):
        var predicate = a.bytes[i]
        if predicate != 0 and predicate != 1:
            return TensorResult[TensorMask[rank]](TensorStatus.INVALID_LAYOUT)
        if predicate == 0:
            out.bytes[i] = 1
    return TensorResult[TensorMask[rank]](out^)


def mask_any[rank: Int](mask: TensorMask[rank]) -> TensorResult[Bool]:
    return mask.any()


def mask_all[rank: Int](mask: TensorMask[rank]) -> TensorResult[Bool]:
    return mask.all()


def mask_count_true[rank: Int](mask: TensorMask[rank]) -> TensorResult[Int]:
    var valid = mask.metadata_status()
    if valid != TensorStatus.OK:
        return TensorResult[Int](valid)
    var count = 0
    for i in range(mask.shape.numel()):
        var predicate = mask.bytes[i]
        if predicate != 0 and predicate != 1:
            return TensorResult[Int](TensorStatus.INVALID_LAYOUT)
        if predicate == 1:
            count += 1
    return TensorResult[Int](count)
