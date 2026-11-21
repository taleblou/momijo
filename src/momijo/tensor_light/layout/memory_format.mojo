# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR LAYOUT / MEMORY FORMATS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Layout and Strides
# Module:        momijo.tensor_light.layout.memory_format
# Source:        src/momijo/tensor_light/layout/memory_format.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines memory formats in the tensor layout and strides layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `ContiguousOrder`, `MemoryFormat`,
#   `default_memory_format`, `memory_format_from_order`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Compute and validate shape-to-storage layout mappings.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `ContiguousOrder`,
#     `MemoryFormat`, `default_memory_format`, `memory_format_from_order`,
#     `is_contiguous_format` and related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Preserve correct offsets and spans for arbitrary legal strides.
#   - Detect overflow, invalid extents, and internal overlap.
#
# Performance Objectives:
#   - Prefer metadata-only layout analysis.
#   - Cache or reuse validated stride computations where safe.
#
# Maintainability:
#   - Use consistent C/F contiguity terminology.
#   - Test negative, zero, and unusually large strides.
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

# Order vocabulary is separate from concrete stride values.
from ..internal.status import TensorResult, TensorStatus
@fieldwise_init
struct ContiguousOrder(ImplicitlyCopyable, Equatable):
    var code: Int
    comptime C = Self(0)
    comptime F = Self(1)
    def __eq__(self, other: Self) -> Bool:
        return self.code == other.code

@fieldwise_init
struct MemoryFormat(Copyable):
    var code: Int
    comptime CONTIGUOUS_C = Self(0)
    comptime CONTIGUOUS_F = Self(1)
    comptime STRIDED = Self(2)


def default_memory_format() -> MemoryFormat:
    return materialize[MemoryFormat.CONTIGUOUS_C]()


def memory_format_from_order(order: ContiguousOrder) -> TensorResult[MemoryFormat]:
    if order == ContiguousOrder.C:
        return TensorResult[MemoryFormat](materialize[MemoryFormat.CONTIGUOUS_C]())
    if order == ContiguousOrder.F:
        return TensorResult[MemoryFormat](materialize[MemoryFormat.CONTIGUOUS_F]())
    return TensorResult[MemoryFormat](TensorStatus.INVALID_LAYOUT)


def is_contiguous_format(format: MemoryFormat) -> Bool:
    return format.code == 0 or format.code == 1


def normalize_memory_format(format: MemoryFormat) -> TensorResult[MemoryFormat]:
    if format.code < 0 or format.code > 2:
        return TensorResult[MemoryFormat](TensorStatus.INVALID_LAYOUT)
    return TensorResult[MemoryFormat](format.copy())
