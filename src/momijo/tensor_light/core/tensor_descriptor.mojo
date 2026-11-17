# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR CORE / DESCRIPTORS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Core Tensor Types
# Module:        momijo.tensor_light.core.tensor_descriptor
# Source:        src/momijo/tensor_light/core/tensor_descriptor.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor descriptors and the related core tensor representation
#   contracts.
#
#   Public declarations include `TensorDescriptor`, `make_descriptor`,
#   `descriptor_from_layout`, `replace_layout`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Maintain tensor representation and metadata invariants.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `TensorDescriptor`,
#     `make_descriptor`, `descriptor_from_layout`, `replace_layout`.
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

# Compact FP32 tensor descriptor. No runtime dtype or heap indirection.
# Public entry points validate structural metadata against an actual owner
# capacity; raw writable TensorLayout fields alone are NOT safety evidence.
from ..layout.layout import TensorLayout
from ..internal.status import TensorResult, TensorStatus

@fieldwise_init
struct TensorDescriptor[rank: Int](Copyable):
    var mapping: TensorLayout[Self.rank]

    def numel(self) -> Int:
        # Fast query; valid only for a checked descriptor/owner pair.
        return self.mapping.shape.numel()

    def nbytes(self) -> Int:
        return self.mapping.shape.nbytes()

    def required_span_size(self) -> Int:
        return self.mapping.span_size()

    def checked_snapshot(self, storage_capacity: Int) -> TensorResult[Self]:
        var verified = self.mapping.validated_snapshot(storage_capacity)
        if not verified.is_ok():
            return TensorResult[Self](verified.status)
        return TensorResult[Self](Self(verified.payload.value()))

    def checked_numel(self, storage_capacity: Int) -> TensorResult[Int]:
        var verified = self.checked_snapshot(storage_capacity)
        if not verified.is_ok():
            return TensorResult[Int](verified.status)
        return TensorResult[Int](verified.payload.value().numel())

    def checked_nbytes(self, storage_capacity: Int) -> TensorResult[Int]:
        var verified = self.checked_snapshot(storage_capacity)
        if not verified.is_ok():
            return TensorResult[Int](verified.status)
        return TensorResult[Int](verified.payload.value().nbytes())

    def checked_required_span_size(self, storage_capacity: Int) -> TensorResult[Int]:
        var verified = self.checked_snapshot(storage_capacity)
        if not verified.is_ok():
            return TensorResult[Int](verified.status)
        return TensorResult[Int](verified.payload.value().required_span_size())


def make_descriptor[rank: Int](
    layout: TensorLayout[rank], storage_capacity: Int,
) -> TensorResult[TensorDescriptor[rank]]:
    # Always validate the live allocation's physical extent before exposing
    # metadata as an owning tensor descriptor.
    var checked = layout.validated_snapshot(storage_capacity)
    if not checked.is_ok():
        return TensorResult[TensorDescriptor[rank]](checked.status)
    return TensorResult[TensorDescriptor[rank]](
        TensorDescriptor[rank](checked.payload.value())
    )


def descriptor_from_layout[rank: Int](
    layout: TensorLayout[rank], storage_capacity: Int,
) -> TensorResult[TensorDescriptor[rank]]:
    return make_descriptor(layout, storage_capacity)


def replace_layout[rank: Int](
    descriptor: TensorDescriptor[rank], layout: TensorLayout[rank],
    storage_capacity: Int,
) -> TensorResult[TensorDescriptor[rank]]:
    # The old descriptor has no authority to bless the new layout; the new
    # mapping must independently fit the same live allocation capacity.
    var old = descriptor.checked_snapshot(storage_capacity)
    if not old.is_ok():
        return TensorResult[TensorDescriptor[rank]](old.status)
    return make_descriptor(layout, storage_capacity)
