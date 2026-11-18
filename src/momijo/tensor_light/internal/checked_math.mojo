# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR INTERNAL / CHECKED ARITHMETIC
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Internal Validation Utilities
# Module:        momijo.tensor_light.internal.checked_math
# Source:        src/momijo/tensor_light/internal/checked_math.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines checked arithmetic in the internal validation utilities layer of
#   the FP32 Light tensor implementation.
#
#   Public declarations include `checked_mul_nonnegative`, `checked_add`,
#   `checked_sub`, `checked_mul`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Provide reusable checked primitives and status contracts.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `checked_mul_nonnegative`,
#     `checked_add`, `checked_sub`, `checked_mul`, `checked_nbytes` and
#     related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Detect integer overflow and invalid metadata before dereference.
#   - Return explicit statuses for unsupported or invalid states.
#
# Performance Objectives:
#   - Use simple inlinable checked primitives where appropriate.
#   - Keep validation work proportional to required metadata.
#
# Maintainability:
#   - Reuse one error vocabulary across dependent modules.
#   - Test boundary arithmetic and invalid-state propagation.
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

# All potentially overflowing arithmetic is checked before a hot loop.
from .status import TensorResult, TensorStatus

def checked_mul_nonnegative(a: Int, b: Int) -> TensorResult[Int]:
    if a < 0 or b < 0:
        return TensorResult[Int](TensorStatus.INVALID_SHAPE)
    if a == 0 or b == 0:
        return TensorResult[Int](0)
    if a > Int.MAX // b:
        return TensorResult[Int](TensorStatus.OVERFLOW)
    return TensorResult[Int](a * b)

def checked_add(a: Int, b: Int) -> TensorResult[Int]:
    if b > 0 and a > Int.MAX - b:
        return TensorResult[Int](TensorStatus.OVERFLOW)
    if b < 0 and a < Int.MIN - b:
        return TensorResult[Int](TensorStatus.OVERFLOW)
    return TensorResult[Int](a + b)

def checked_sub(a: Int, b: Int) -> TensorResult[Int]:
    if b == Int.MIN:
        if a >= 0:
            return TensorResult[Int](TensorStatus.OVERFLOW)
        return TensorResult[Int](a - b)
    return checked_add(a, -b)

def checked_mul(a: Int, b: Int) -> TensorResult[Int]:
    # Mojo's // rounds toward NEGATIVE infinity, not toward zero. Signed
    # overflow bounds cannot compare against a raw negative floor quotient:
    # e.g. Int.MIN // 3 admits one too-negative multiplicand if used as the
    # lower limit. Work out the exact allowed interval BEFORE multiplying.
    if a == 0 or b == 0:
        return TensorResult[Int](0)
    if a > 0 and b > 0:
        if a > Int.MAX // b:
            return TensorResult[Int](TensorStatus.OVERFLOW)
    elif a < 0 and b < 0:
        # The positive result has to fit MAX (not abs(MIN)). Handle the
        # unrepresentable absolute value of Int.MIN before negating either
        # operand; the remaining absolute values are both positive Ints.
        if a == Int.MIN or b == Int.MIN:
            return TensorResult[Int](TensorStatus.OVERFLOW)
        var positive_a = -a
        var positive_b = -b
        if positive_a > Int.MAX // positive_b:
            return TensorResult[Int](TensorStatus.OVERFLOW)
    elif a < 0:
        # Here b > 0. The minimum permitted multiplier is ceil(MIN / b).
        # Mojo's // gives floor, so increment when the remainder is nonzero.
        var minimum_a = Int.MIN // b
        if Int.MIN % b != 0:
            minimum_a += 1
        if a < minimum_a:
            return TensorResult[Int](TensorStatus.OVERFLOW)
    else:
        # Here a > 0 and b < 0. This is the symmetric signed lower bound.
        var minimum_b = Int.MIN // a
        if Int.MIN % a != 0:
            minimum_b += 1
        if b < minimum_b:
            return TensorResult[Int](TensorStatus.OVERFLOW)
    # All operand combinations, including MIN * 1 and negative/negative,
    # have passed exact bounds; wrapped multiplication is now impossible.
    return TensorResult[Int](a * b)


def checked_nbytes(numel: Int) -> TensorResult[Int]:
    if numel < 0:
        return TensorResult[Int](TensorStatus.INVALID_SHAPE)
    return checked_mul_nonnegative(numel, 4)


def checked_byte_count(numel: Int, itemsize: Int) -> TensorResult[Int]:
    # Fixed FP32 tensor bytes. Explicitly reject a foreign element size.
    if itemsize != 4:
        return TensorResult[Int](TensorStatus.UNSUPPORTED)
    return checked_nbytes(numel)


def checked_offset_add(base: Int, delta: Int) -> TensorResult[Int]:
    # Physical element offset arithmetic, including negative strides.
    return checked_add(base, delta)

def checked_product[count: Int](values: Array[Int, count], logical_count: Int) -> TensorResult[Int]:
    # The logical count is separate from fixed Array capacity so that a
    # rank-0 padded metadata array correctly has an empty product of one.
    # The zero prescan makes an otherwise overflowing intermediate product
    # irrelevant for an empty tensor, matching TensorShape.checked().
    if logical_count < 0 or logical_count > count or logical_count > 8:
        return TensorResult[Int](TensorStatus.INVALID_RANK)
    var has_zero = False
    for i in range(logical_count):
        if values[i] < 0:
            return TensorResult[Int](TensorStatus.INVALID_SHAPE)
        if values[i] == 0:
            has_zero = True
    if has_zero:
        return TensorResult[Int](0)
    var product = 1
    for i in range(logical_count):
        var part = checked_mul_nonnegative(product, values[i])
        if not part.is_ok():
            return TensorResult[Int](part.status)
        product = part.payload.value()
    return TensorResult[Int](product)
