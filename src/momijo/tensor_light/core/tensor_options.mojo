# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR CORE / OPTIONS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Core Tensor Types
# Module:        momijo.tensor_light.core.tensor_options
# Source:        src/momijo/tensor_light/core/tensor_options.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor options and the related core tensor representation
#   contracts.
#
#   Public declarations include `TensorOptions`, `default_host_options`,
#   `validate_tensor_options`, `with_order`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Maintain tensor representation and metadata invariants.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `TensorOptions`,
#     `default_host_options`, `validate_tensor_options`, `with_order`,
#     `with_alignment` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Preserve owning-tensor invariants and initialized-storage contracts.
#   - Check shape/layout compatibility before changing tensor metadata.
#
# Performance Objectives:
#   - Prefer metadata-only transformations when ownership permits.
#   - Keep hot-path operations specialized without weakening checks.
#
# Maintainability:
#   - Keep the primary tensor ownership model unambiguous.
#   - Document public constructors, moves, and borrow boundaries.
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

# Construction policy, not persistent Tensor identity.
from ..memory.memory_space import MemorySpace
from ..device.device import Device
from ..internal.status import TensorResult, TensorStatus
from ..layout.memory_format import ContiguousOrder

@fieldwise_init
struct TensorOptions(Copyable):
    var device_ordinal: Int
    var memory_space: MemorySpace
    var f_order: Bool
    var requested_alignment: Int
    # False selects an allocation-only, initially UNINITIALIZED builder.
    # True is served by the explicit zeros_with_options() constructor;
    # empty_with_options() rejects the conflicting request rather than
    # exposing an initialized Tensor through an uninitialized return type.
    var initialize_to_zero: Bool

def default_host_options() -> TensorOptions:
    return TensorOptions(-1, materialize[MemorySpace.HOST_PAGEABLE](), False, 64, False)

def validate_tensor_options(options: TensorOptions) -> TensorStatus:
    if options.requested_alignment < 4 or (options.requested_alignment & (options.requested_alignment - 1)) != 0:
        return TensorStatus.INVALID_ALIGNMENT
    if options.memory_space.code < 0 or options.memory_space.code > 3:
        return TensorStatus.INVALID_LAYOUT
    if options.memory_space == materialize[MemorySpace.HOST_PAGEABLE]():
        if options.device_ordinal != -1:
            return TensorStatus.DEVICE_MISMATCH
    elif options.device_ordinal < 0:
        return TensorStatus.DEVICE_MISMATCH
    return TensorStatus.OK


def with_order(options: TensorOptions, f_order: Bool) -> TensorOptions:
    var updated = options.copy()
    updated.f_order = f_order
    return updated^


def with_alignment(options: TensorOptions, alignment: Int) -> TensorResult[TensorOptions]:
    var updated = options.copy()
    updated.requested_alignment = alignment
    var status = validate_tensor_options(updated)
    if status != TensorStatus.OK:
        return TensorResult[TensorOptions](status)
    return TensorResult[TensorOptions](updated^)


def with_device(options: TensorOptions, ordinal: Int) -> TensorResult[TensorOptions]:
    # The Int overload has an unambiguous policy: -1 selects CPU pageable
    # memory; any nonnegative ordinal selects the corresponding GPU. Merely
    # changing the ordinal while leaving HOST_PAGEABLE selected would create
    # an invalid cross-field tuple for every GPU request.
    if ordinal < -1:
        return TensorResult[TensorOptions](TensorStatus.DEVICE_MISMATCH)
    var updated = options.copy()
    updated.device_ordinal = ordinal
    if ordinal == -1:
        updated.memory_space = materialize[MemorySpace.HOST_PAGEABLE]()
    else:
        updated.memory_space = materialize[MemorySpace.DEVICE_LOCAL]()
    var status = validate_tensor_options(updated)
    if status != TensorStatus.OK:
        return TensorResult[TensorOptions](status)
    return TensorResult[TensorOptions](updated^)


def with_memory_space(options: TensorOptions, memory_space: MemorySpace) -> TensorResult[TensorOptions]:
    var updated = options.copy()
    updated.memory_space = memory_space.copy()
    var status = validate_tensor_options(updated)
    if status != TensorStatus.OK:
        return TensorResult[TensorOptions](status)
    return TensorResult[TensorOptions](updated^)


def with_initialization_policy(options: TensorOptions, initialize_to_zero: Bool) -> TensorOptions:
    var updated = options.copy()
    updated.initialize_to_zero = initialize_to_zero
    return updated^

def default_tensor_options[rank: Int]() -> TensorOptions:
    comptime assert rank >= 0 and rank <= 8, "tensor rank must be 0..8"
    return default_host_options()

def with_device(options: TensorOptions, device: Device) -> TensorResult[TensorOptions]:
    var updated = options.copy()
    if device.is_cpu():
        if device.ordinal != 0:
            return TensorResult[TensorOptions](TensorStatus.DEVICE_MISMATCH)
        updated.device_ordinal = -1
        updated.memory_space = materialize[MemorySpace.HOST_PAGEABLE]()
    elif device.is_gpu():
        if device.ordinal < 0:
            return TensorResult[TensorOptions](TensorStatus.DEVICE_MISMATCH)
        updated.device_ordinal = device.ordinal
        updated.memory_space = materialize[MemorySpace.DEVICE_LOCAL]()
    else:
        return TensorResult[TensorOptions](TensorStatus.DEVICE_MISMATCH)
    var valid = validate_tensor_options(updated)
    if valid != TensorStatus.OK:
        return TensorResult[TensorOptions](valid)
    return TensorResult[TensorOptions](updated^)


def with_order(options: TensorOptions, order: ContiguousOrder) -> TensorResult[TensorOptions]:
    if order != ContiguousOrder.C and order != ContiguousOrder.F:
        return TensorResult[TensorOptions](TensorStatus.INVALID_LAYOUT)
    var updated = options.copy()
    updated.f_order = order == ContiguousOrder.F
    return TensorResult[TensorOptions](updated^)
