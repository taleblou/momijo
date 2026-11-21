# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR FACTORY / SEEDED RANDOM OPERATIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Factories
# Module:        momijo.tensor_light.factory.random
# Source:        src/momijo/tensor_light/factory/random.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines seeded random operations in the tensor factories layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `random_uniform`, `random_normal`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Create tensors with checked layouts, allocation, and initialization.
#   - Preserve documented behavior for reproducible seeds, index domains, and
#     deterministic outputs.
#   - Provide source-declared interfaces such as `random_uniform`,
#     `random_normal`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate output shape, allocation size, and initialization state.
#   - Do not expose partially initialized storage as a completed tensor.
#   - Honor deterministic seeded streams and documented sampling bounds.
#   - Respect the fixed-FP32 type and supported-rank policy where applicable.
#
# Performance Objectives:
#   - Avoid redundant zeroing and intermediate allocations when safe.
#   - Use one checked allocation boundary per created result where possible.
#   - Avoid unnecessary reseeding or intermediate index buffers.
#
# Maintainability:
#   - Keep factory names consistent with storage/ownership contracts.
#   - Test invalid shapes and allocation failures.
#   - Test reproducibility and edge-size behavior.
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

# Local deterministic CPU Float32 input constructors for QA fixtures.
# This is a VERSIONED Park-Miller/Box-Muller generator (local-f32-pm-v1),
# NOT a substitute for the cross-engine canonical SHA-256 byte generator.
# It produces no integer-valued Tensor and supports the original rank 0..8.
from std.math import log as math_log, sqrt as math_sqrt, cos as math_cos, sin as math_sin
from ..core.tensor import Tensor
from ..shape.shape import TensorShape
from ..storage.host_storage import HostStorage
from .creation import _complete_overwrite_output
from ..internal.status import TensorResult, TensorStatus


@inline(.always)
def _seed_state_pm_v1(seed: Int) -> Int:
    # Reduce before adding to avoid overflow for extreme signed seeds.
    var state = seed % 2147483646
    if state < 0:
        state += 2147483646
    return state + 1


@inline(.always)
def _next_state_pm_v1(state: Int) -> Int:
    # 16807*2147483646 < 2**46; safe for supported 64-bit Mojo Int hosts.
    return (state * 16807) % 2147483647


@inline(.always)
def _unit_float_pm_v1(state: Int) -> Float32:
    # Exactly representable 24-bit mantissa fraction in [0,1).
    return Float32(state >> 7) * Float32(0.000000059604644775390625)


def random_uniform[rank: Int](
    shape: TensorShape[rank], seed: Int
) -> TensorResult[Tensor[rank, HostStorage]]:
    var valid = TensorShape[rank].checked(shape.dims)
    if not valid.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](valid.status)
    if valid.payload.value().numel() != shape.numel():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.INVALID_SHAPE)
    var built = _complete_overwrite_output[rank](valid.payload.value())
    if not built.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](built.status)
    var output = built.payload.take()
    var dst = output.impl.storage.borrow_mut_span()
    var state = _seed_state_pm_v1(seed)
    for index in range(shape.numel()):
        state = _next_state_pm_v1(state)
        dst[index] = _unit_float_pm_v1(state)
    return (output^)._finish_complete_overwrite()


def random_normal[rank: Int](
    shape: TensorShape[rank], seed: Int
) -> TensorResult[Tensor[rank, HostStorage]]:
    var valid = TensorShape[rank].checked(shape.dims)
    if not valid.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](valid.status)
    if valid.payload.value().numel() != shape.numel():
        return TensorResult[Tensor[rank, HostStorage]](TensorStatus.INVALID_SHAPE)
    var built = _complete_overwrite_output[rank](valid.payload.value())
    if not built.is_ok():
        return TensorResult[Tensor[rank, HostStorage]](built.status)
    var output = built.payload.take()
    var dst = output.impl.storage.borrow_mut_span()
    var state = _seed_state_pm_v1(seed)
    var index = 0
    while index < shape.numel():
        state = _next_state_pm_v1(state)
        # 24-bit fraction strictly positive for the log() argument.
        var u1 = Float32((state >> 7) + 1) * Float32(0.000000059604644775390625)
        state = _next_state_pm_v1(state)
        var u2 = _unit_float_pm_v1(state)
        var radius = math_sqrt(Float32(-2.0) * math_log(u1))
        var angle = Float32(6.283185307179586) * u2
        dst[index] = radius * math_cos(angle)
        if index + 1 < shape.numel():
            dst[index + 1] = radius * math_sin(angle)
        index += 2
    return (output^)._finish_complete_overwrite()
