# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR CORE / BOOLEAN TENSOR MASKS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Core Tensor Types
# Module:        momijo.tensor_light.core.tensor_mask
# Source:        src/momijo/tensor_light/core/tensor_mask.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines boolean tensor masks and the related core tensor representation
#   contracts.
#
#   Public declarations include `TensorMask`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Maintain tensor representation and metadata invariants.
#   - Preserve documented behavior for Boolean mask semantics and tensor shape
#     compatibility.
#   - Provide source-declared interfaces such as `TensorMask`.
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

# Low-level Float32 predicate companion type, not a Tensor dtype.
from std.collections.list import List
from ..shape.shape import TensorShape
from ..internal.status import TensorStatus, TensorResult

struct TensorMask[rank: Int](Movable):
    var shape: TensorShape[Self.rank]
    var bytes: List[UInt8]

    def __init__(out self, shape: TensorShape[Self.rank], var buffer: List[UInt8]):
        self.shape = shape
        self.bytes = buffer^

    @staticmethod
    def zeros_checked(shape: TensorShape[Self.rank]) -> TensorResult[Self]:
        # Verify cached numel before allocation. Rank zero has one mask element.
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        if canonical.payload.value().numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var bytes = List[UInt8](length=shape.numel(), fill=0)
        return TensorResult[Self](Self(canonical.payload.value(), bytes^))

    @staticmethod
    def from_buffer_checked(shape: TensorShape[Self.rank], var buffer: List[UInt8]) -> TensorResult[Self]:
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        if canonical.payload.value().numel() != shape.numel() or len(buffer) != shape.numel():
            return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
        for i in range(len(buffer)):
            if buffer[i] != 0 and buffer[i] != 1:
                return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        return TensorResult[Self](Self(canonical.payload.value(), buffer^))

    def metadata_status(self) -> TensorStatus:
        # Cheap structural proof for operations that also traverse and check
        # every predicate byte while producing a NEW, private output.
        # In-place operations must use validated() before their first write.
        var canonical = TensorShape[Self.rank].checked(self.shape.dims)
        if not canonical.is_ok():
            return canonical.status
        if canonical.payload.value().numel() != self.shape.numel():
            return TensorStatus.INVALID_SHAPE
        if len(self.bytes) != self.shape.numel():
            return TensorStatus.SIZE_MISMATCH
        return TensorStatus.OK

    def validated(self) -> TensorStatus:
        var status = self.metadata_status()
        if status != TensorStatus.OK:
            return status
        # This full scan is required for in-place mutation: no destination
        # element may change if a later mask byte is malformed.
        for i in range(len(self.bytes)):
            if self.bytes[i] != 0 and self.bytes[i] != 1:
                return TensorStatus.INVALID_LAYOUT
        return TensorStatus.OK

    def get(self, linear_index: Int) -> TensorResult[Bool]:
        var valid = self.metadata_status()
        if valid != TensorStatus.OK:
            return TensorResult[Bool](valid)
        if linear_index < 0 or linear_index >= self.shape.numel():
            return TensorResult[Bool](TensorStatus.OUT_OF_BOUNDS)
        if self.bytes[linear_index] != 0 and self.bytes[linear_index] != 1:
            return TensorResult[Bool](TensorStatus.INVALID_LAYOUT)
        return TensorResult[Bool](self.bytes[linear_index] != 0)

    def set(mut self, linear_index: Int, value: Bool) -> TensorStatus:
        # This is a PUBLIC in-place mutation boundary. Validate all existing
        # bytes before the first write; unlike out-of-place mask operations,
        # a malformed later predicate must never leave a partially edited mask.
        # Bulk kernels should validate once and use their own unchecked loop.
        var valid = self.validated()
        if valid != TensorStatus.OK:
            return valid
        if linear_index < 0 or linear_index >= self.shape.numel():
            return TensorStatus.OUT_OF_BOUNDS
        if value:
            self.bytes[linear_index] = 1
        else:
            self.bytes[linear_index] = 0
        return TensorStatus.OK

    def any(self) -> TensorResult[Bool]:
        var status = self.metadata_status()
        if status != TensorStatus.OK:
            return TensorResult[Bool](status)
        # One pass verifies every byte, even after the answer is already true.
        # An invalid byte must not be hidden by short-circuit evaluation.
        var result = False
        for i in range(self.shape.numel()):
            var predicate = self.bytes[i]
            if predicate != 0 and predicate != 1:
                return TensorResult[Bool](TensorStatus.INVALID_LAYOUT)
            if predicate == 1:
                result = True
        return TensorResult[Bool](result)

    def all(self) -> TensorResult[Bool]:
        var status = self.metadata_status()
        if status != TensorStatus.OK:
            return TensorResult[Bool](status)
        var result = True
        for i in range(self.shape.numel()):
            var predicate = self.bytes[i]
            if predicate != 0 and predicate != 1:
                return TensorResult[Bool](TensorStatus.INVALID_LAYOUT)
            if predicate == 0:
                result = False
        return TensorResult[Bool](result)

