# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR MEMORY / DATA POINTERS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Memory and Allocation
# Module:        momijo.tensor_light.memory.data_ptr
# Source:        src/momijo/tensor_light/memory/data_ptr.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines data pointers in the memory and allocation layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `HostDataPtr`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Represent or validate typed data pointers and their offset/address-space
#     metadata.
#   - Expose source-defined interfaces including `HostDataPtr`,
#     `element_count`, `byte_count_checked`, `from_span_checked`.
#   - Enforce the source-defined pointer provenance, bounds, and memory-domain
#     contracts.
#   - Propagate failed validation or allocation results without silently
#     changing memory domains.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate sizes, alignment, pointer access, and memory-domain contracts
#     as applicable.
#   - Reject invalid inputs without asserting unrelated pool ownership or
#     completion semantics.
#
# Performance Objectives:
#   - Keep pointer, size, domain, and alignment checks inexpensive.
#   - Avoid introducing allocations or copies in metadata and validation
#     paths.
#
# Maintainability:
#   - Document the exact allocation, pointer, and address-space contracts
#     exposed here.
#   - Test overflow, invalid alignment, and allocation failures where
#     supported.
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

# Non-owning FP32 host data handle. Origin and physical range remain explicit.
from std.collections import Span
from ..internal.status import TensorResult, TensorStatus
from ..memory.alignment import is_valid_alignment
from ..internal.checked_math import checked_nbytes

struct HostDataPtr[origin: Origin](Copyable):
    var span: Span[False, Float32, Self.origin]
    var alignment: Int

    def __init__(out self, span: Span[False, Float32, Self.origin], alignment: Int):
        self.span = span
        self.alignment = alignment

    def __init__(out self, *, copy: Self):
        self.span = copy.span
        self.alignment = copy.alignment

    def element_count(self) -> Int:
        return len(self.span)

    def byte_count_checked(self) -> TensorResult[Int]:
        return checked_nbytes(len(self.span))

    @staticmethod
    def from_span_checked(span: Span[False, Float32, Self.origin], alignment: Int) -> TensorResult[Self]:
        # A power-of-two integer is NOT an alignment proof for external data.
        # The base address must actually satisfy the advertised alignment;
        # otherwise an aligned AOT SIMD load can have undefined behavior.
        if not is_valid_alignment(alignment):
            return TensorResult[Self](TensorStatus.INVALID_ALIGNMENT)
        var bytes = checked_nbytes(len(span))
        if not bytes.is_ok():
            return TensorResult[Self](bytes.status)
        if len(span) > 0:
            var base = Int(span.unsafe_ptr())
            if base <= 0 or base % alignment != 0:
                return TensorResult[Self](TensorStatus.INVALID_ALIGNMENT)
            # An address + the complete physical byte range must be
            # representable, even though this check never dereferences it.
            if base > Int.MAX - bytes.payload.value():
                return TensorResult[Self](TensorStatus.OVERFLOW)
        # Empty spans have no dereferenceable element; do not materialize a
        # possibly invalid empty-span pointer just to check alignment.
        return TensorResult[Self](Self(span, alignment))

    def checked_snapshot(self) -> TensorResult[Self]:
        # The raw initializer and var fields remain available for internal
        # metadata, but they do not certify the pointer/alignment contract.
        # Recheck immediately before using this value to dispatch a kernel.
        return Self.from_span_checked(self.span, self.alignment)

    def load_checked(self, index: Int) -> TensorResult[Float32]:
        if index < 0 or index >= len(self.span):
            return TensorResult[Float32](TensorStatus.OUT_OF_BOUNDS)
        return TensorResult[Float32](self.span[index])
