# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR STORAGE / HOST STORAGE
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Storage and Ownership
# Module:        momijo.tensor_light.storage.host_storage
# Source:        src/momijo/tensor_light/storage/host_storage.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines host storage in the storage and ownership layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `HostStorage`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Enforce tensor storage lifetime and initialized-data contracts.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `HostStorage`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Preserve storage provenance and initialization state.
#   - Prevent use-after-release and unsafe mutable aliasing.
#   - Never promote unknown provenance to proven non-aliasing.
#
# Performance Objectives:
#   - Avoid storage duplication when a safe borrow is sufficient.
#   - Reclaim storage only after ownership/completion conditions hold.
#   - Do not recycle memory before lifetime and completion proofs hold.
#
# Maintainability:
#   - Document ownership transitions and destructor behavior.
#   - Test external-borrow lifetime and partially initialized buffers.
#   - Cover recycled blocks, ownership failures, and overlapping buffers.
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

# Unique ownership of an FP32 host allocation, independent of shape and layout.
# Allocation and deallocation have one canonical owner in memory/allocation.
from std.collections import Span
from std.memory import unsafe_memcpy
from std.sys.info import simd_width_of
from ..internal.status import TensorStatus, TensorResult
from ..memory.allocation import HostAllocation, _allocate_host_fp32_uninitialized
from .storage import TensorStorage, MutableTensorStorage, HostSpanStorage

# Storage-local vector fill for a verified contiguous writable FP32 region.
# All callers validate bounds/writeability before invoking this unchecked loop.
@inline(.always)
def _fill_fp32_contiguous[width: Int, o: Origin[mut=True]](
    dest: Span[True, Float32, o], value: Float32
):
    comptime assert width > 0, "Float32 SIMD width must be positive"
    var n = len(dest)
    var end = n - n % width
    var ptr = dest.unsafe_ptr()
    var lanes = SIMD[DType.float32, width](value)
    for i in range(0, end, width):
        ptr.unsafe_store[alignment=4](i, lanes)
    for i in range(end, n):
        ptr.unsafe_store[alignment=4](i, value)

struct HostStorage(Movable, HostSpanStorage):
    var owned: HostAllocation

    def __init__(out self, var owned: HostAllocation):
        self.owned = owned

    @staticmethod
    def full(count: Int, value: Float32) -> TensorResult[Self]:
        var allocation = _allocate_host_fp32_uninitialized(count)
        if not allocation.is_ok():
            return TensorResult[Self](allocation.status)
        var result = Self(allocation.payload.take())
        var status = result.fill_host(value)
        if status != TensorStatus.OK:
            return TensorResult[Self](status)
        return TensorResult[Self](result)

    @staticmethod
    def zeros(count: Int) -> TensorResult[Self]:
        return Self.full(count, 0.0)

    @staticmethod
    def _allocate_for_complete_overwrite(count: Int) -> TensorResult[Self]:
        # Internal complete-overwrite callers must initialize every element.
        # No PUBLIC factory returns this as a readable owning Tensor before
        # a proven complete write. Public Tensor.empty(), including historic
        # unsafe_empty_* spellings, returns an unreadable init builder.
        var allocation = _allocate_host_fp32_uninitialized(count)
        if not allocation.is_ok():
            return TensorResult[Self](allocation.status)
        return TensorResult[Self](Self(allocation.payload.take()))

    def capacity(self) -> Int:
        # Negative capacity makes every checked TensorLayout construction fail
        # before it can borrow a raw HostStorage Span.
        if self.owned.validate_metadata() != TensorStatus.OK:
            return -1
        return self.owned.length()

    def is_host_accessible(self) -> Bool:
        return True

    def is_initialized(self) -> Bool:
        return self.owned.is_initialized()

    def _mark_complete_overwrite(mut self) -> TensorStatus:
        return self.owned._mark_complete_overwrite()

    def _invalidate_after_failed_write(mut self):
        # Only a later full, validated overwrite may make this owner readable.
        # Existing allocations remain owned and are never released on error.
        self.owned._invalidate_after_failed_write()

    def read_host(self, index: Int) -> TensorResult[Float32]:
        return self.owned.read_checked(index)

    def write_host(mut self, index: Int, value: Float32) -> TensorStatus:
        return self.owned.write_checked(index, value)

    def fill_host(mut self, value: Float32) -> TensorStatus:
        var valid = self.owned.validate_metadata()
        if valid != TensorStatus.OK:
            return valid
        _fill_fp32_contiguous[simd_width_of[DType.float32]()](self.borrow_mut_span(), value)
        return self.owned._mark_complete_overwrite()

    def _copy_from_disjoint_span_unchecked[origin: Origin](
        mut self, source: Span[False, Float32, origin]
    ) -> TensorStatus:
        # The caller establishes source/destination disjointness (a fresh
        # owned allocation or a separately proven address range). Recheck
        # byte-count and owner metadata ONCE at this trusted copy boundary:
        # an unchecked memcpy must never copy a stale/mutated logical count.
        # This method never runs in the per-element hot loop.
        var valid = self.owned.validate_metadata()
        if valid != TensorStatus.OK:
            return valid
        if len(source) != self.owned.logical_count:
            return TensorStatus.SIZE_MISMATCH
        if self.owned.logical_count > 0:
            unsafe_memcpy(
                dest=self.borrow_mut_span().unsafe_ptr(),
                src=source.unsafe_ptr(),
                count=self.owned.logical_count,
            )
        # Do not discard a failed publication: a partial/invalid owner must
        # not become a readable Tensor via an unconditional constructor.
        return self.owned._mark_complete_overwrite()

    # Internal fast-path precondition: call capacity()/validated_snapshot()
    # before borrowing; the returned Span cannot express fallible validation.
    def borrow_span(ref self) -> Span[False, Float32, origin_of(self)]:
        return Span[False, Float32, origin_of(self)](
            unsafe_ptr=self.owned.memory.unsafe_ptr(), length=self.owned.logical_count
        )

    # Internal fast-path precondition: metadata and exclusive mapping checked.
    def borrow_mut_span(mut self) -> Span[True, Float32, origin_of(self)]:
        return Span[True, Float32, origin_of(self)](
            unsafe_ptr=self.owned.memory.unsafe_ptr(), length=self.owned.logical_count
        )
