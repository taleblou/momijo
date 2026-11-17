# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DTYPE / FP32 ARITHMETIC SEMANTICS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / FP32 DType and Math Policy
# Module:        momijo.tensor_light.dtype.fp32
# Source:        src/momijo/tensor_light/dtype/fp32.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines FP32 arithmetic semantics in the FP32 dtype and math policy layer
#   of the FP32 Light tensor implementation.
#
#   Public declarations include `tensor_zero`, `tensor_one`, `canonical_nan`,
#   `canonical_inf`; preconditions and supported targets are determined by
#   their implementations.
#
# Responsibilities:
#   - Define consistent Float32 dtype and arithmetic policy.
#   - Preserve documented behavior for floating-point policy, dtype
#     preservation, and rounding.
#   - Provide source-declared interfaces such as `tensor_zero`, `tensor_one`,
#     `canonical_nan`, `canonical_inf`, `fp32_is_negative_zero` and related
#     symbols.
#   - Keep floating-point corner-case behavior explicit and testable.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Preserve the declared FP32 dtype and arithmetic mode.
#   - Handle non-finite values, signed zeros, and rounding intentionally.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid unnecessary intermediate conversions and extra rounding.
#   - Use fixed policies that can be specialized ahead of time.
#
# Maintainability:
#   - Keep scalar semantics shared with kernels and operators.
#   - Cover IEEE corner cases with targeted regression tests.
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

# The sole supported tensor element type and its storage invariants.
from std.math import copysign

comptime TENSOR_DTYPE = DType.float32
comptime TENSOR_ITEMSIZE = 4

def tensor_zero() -> Float32:
    return 0.0

def tensor_one() -> Float32:
    return 1.0


def canonical_nan() -> Float32:
    # Native IEEE-754 Float32 NaN without an FP64 intermediate.
    return Float32(FloatLiteral.nan)


def canonical_inf() -> Float32:
    # Positive IEEE-754 Float32 infinity, preserving the only payload dtype.
    return Float32(FloatLiteral.infinity)


# IEEE-754 FP32 extrema with NaN propagation and signed-zero stability.
# For equal finite nonzero operands the left operand is retained. For zeros,
# minimum selects -0.0 and maximum selects +0.0 regardless of input order.
# This is independent of compiler/platform min/max intrinsic NaN behavior.

@inline(.always)
def fp32_is_negative_zero(value: Float32) -> Bool:
    return value == Float32(0.0) and copysign(Float32(1.0), value) < Float32(0.0)

@inline(.always)
def fp32_minimum(left: Float32, right: Float32) -> Float32:
    if left != left:
        return left
    if right != right:
        return right
    if left < right:
        return left
    if right < left:
        return right
    if left == Float32(0.0) and fp32_is_negative_zero(right):
        return right
    return left

@inline(.always)
def fp32_maximum(left: Float32, right: Float32) -> Float32:
    if left != left:
        return left
    if right != right:
        return right
    if left > right:
        return left
    if right > left:
        return right
    if left == Float32(0.0) and fp32_is_negative_zero(left) and not fp32_is_negative_zero(right):
        return right
    return left

# Stable arg-extrema tie ordering. Equal nonzero values retain the earliest
# logical index; NaNs are handled by the caller as the first NaN encountered.
# IEEE-754 compares +0 and -0 as equal, but this package's minimum/maximum
# contract orders -0 below +0. Argmin/argmax must select the same value.
@inline(.always)
def fp32_argmin_prefers(candidate: Float32, incumbent: Float32) -> Bool:
    if candidate < incumbent:
        return True
    if candidate == Float32(0.0) and incumbent == Float32(0.0):
        return fp32_is_negative_zero(candidate) and not fp32_is_negative_zero(incumbent)
    return False


@inline(.always)
def fp32_argmax_prefers(candidate: Float32, incumbent: Float32) -> Bool:
    if candidate > incumbent:
        return True
    if candidate == Float32(0.0) and incumbent == Float32(0.0):
        return not fp32_is_negative_zero(candidate) and fp32_is_negative_zero(incumbent)
    return False


# Float32-only element properties, specialized at compile time.
def tensor_dtype() -> DType:
    return DType.float32


def itemsize() -> Int:
    return 4


def natural_alignment() -> Int:
    return 4


def tensor_itemsize() -> Int:
    # Named alias for callers that cannot overload a generic itemsize symbol.
    return TENSOR_ITEMSIZE
