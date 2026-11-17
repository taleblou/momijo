# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR DTYPE / SCALAR TYPE UTILITIES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / FP32 DType and Math Policy
# Module:        momijo.tensor_light.dtype.scalar
# Source:        src/momijo/tensor_light/dtype/scalar.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines scalar type utilities in the FP32 dtype and math policy layer of
#   the FP32 Light tensor implementation.
#
#   Public declarations include `normalize_scalar`, `scalar_zero`,
#   `scalar_one`, `scalar_is_nan`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Define consistent Float32 dtype and arithmetic policy.
#   - Preserve documented behavior for floating-point policy, dtype
#     preservation, and rounding.
#   - Provide source-declared interfaces such as `normalize_scalar`,
#     `scalar_zero`, `scalar_one`, `scalar_is_nan`, `scalar_is_finite`.
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

# Explicit Float32 scalar boundary; no dtype conversion/promotions.
from std.utils.numerics import isfinite

def normalize_scalar(value: Float32) -> Float32:
    return value

def scalar_zero() -> Float32:
    return 0.0

def scalar_one() -> Float32:
    return 1.0

def scalar_is_nan(value: Float32) -> Bool:
    return value != value



def scalar_is_finite(value: Float32) -> Bool:
    return Bool(isfinite(value))
