# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR CORE / REFERENCES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Core Tensor Types
# Module:        momijo.tensor_light.core.tensor_ref
# Source:        src/momijo/tensor_light/core/tensor_ref.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor references and the related core tensor representation
#   contracts.
#
#   Public declarations include `TensorRef`, `TensorMutRef`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Maintain tensor representation and metadata invariants.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `TensorRef`, `TensorMutRef`.
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

# Lightweight borrowing facade; no deep-copy ownership.
from std.collections import Span
from ..layout.layout import TensorLayout
from ..layout.memory_format import ContiguousOrder
from ..shape.shape import TensorShape
from ..view.view import TensorView, TensorMutView
from ..shape.shape_storage import StaticRankArray
from ..internal.status import TensorResult, TensorStatus

struct TensorRef[rank: Int, origin: Origin](Copyable):
    var view_data: TensorView[Self.rank, Self.origin]

    def __init__(out self, view: TensorView[Self.rank, Self.origin]):
        self.view_data = view

    def __init__(out self, *, copy: Self):
        self.view_data = copy.view_data.copy()

    def numel(self) -> Int:
        return self.view_data.numel()

    def checked_numel(self) -> TensorResult[Int]:
        return self.view_data.checked_numel()

    def checked_nbytes(self) -> TensorResult[Int]:
        return self.view_data.checked_nbytes()

    def get(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Float32]:
        return self.view_data.get(indices)

    # Borrowed metadata accessors do not allocate or take ownership.
    def shape(self) -> TensorShape[Self.rank]:
        return self.view_data.layout.shape.copy()

    def layout(self) -> TensorLayout[Self.rank]:
        return self.view_data.layout.copy()

    def storage_offset(self) -> Int:
        return self.view_data.layout.offset

    def dim(self, axis: Int) -> TensorResult[Int]:
        var proven = self.view_data.checked_layout()
        if not proven.is_ok():
            return TensorResult[Int](proven.status)
        return proven.payload.value().shape.dim(axis)

    def itemsize(self) -> Int:
        return 4

    def is_empty(self) -> Bool:
        return self.view_data.is_empty()

    def is_unique(self) -> Bool:
        return self.view_data.is_unique()

    def is_exhaustive(self) -> Bool:
        return self.view_data.is_exhaustive()

    def stride(self, axis: Int) -> TensorResult[Int]:
        # The borrowed view owns the checked mapping and bounds contract.
        return self.view_data.stride(axis)

    def is_c_contiguous(self) -> Bool:
        return self.view_data.is_c_contiguous()

    def is_f_contiguous(self) -> Bool:
        return self.view_data.is_f_contiguous()

    def is_contiguous(self, order: ContiguousOrder) -> Bool:
        if order == ContiguousOrder.C:
            return self.is_c_contiguous()
        if order == ContiguousOrder.F:
            return self.is_f_contiguous()
        return False

    def as_view(self) -> TensorView[Self.rank, Self.origin]:
        return self.view_data.copy()


# Expert borrowed mutable capability. It never owns storage or outlives its
# originating Tensor. A mutable view is created only after overlap proof.
struct TensorMutRef[rank: Int, origin: Origin[mut=True]](Movable):
    var view_data: TensorMutView[Self.rank, Self.origin]

    def __init__(out self, var view: TensorMutView[Self.rank, Self.origin]):
        self.view_data = view

    def shape(self) -> TensorShape[Self.rank]:
        return self.view_data.shape()

    def layout(self) -> TensorLayout[Self.rank]:
        return self.view_data.layout.copy()

    def numel(self) -> Int:
        return self.view_data.numel()

    def checked_numel(self) -> TensorResult[Int]:
        return self.view_data.checked_numel()

    def checked_nbytes(self) -> TensorResult[Int]:
        return self.view_data.checked_nbytes()

    def get(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Float32]:
        return self.view_data.get(indices)

    def set(mut self, indices: StaticRankArray[Self.rank], value: Float32) -> TensorStatus:
        return self.view_data.set(indices, value)

    def fill(mut self, value: Float32) -> TensorStatus:
        return self.view_data.fill(value)

    def is_unique(self) -> Bool:
        return self.view_data.is_unique()

    def storage_offset(self) -> Int:
        return self.view_data.storage_offset()
