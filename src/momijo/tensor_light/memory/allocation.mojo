# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR MEMORY / MEMORY ALLOCATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Memory and Allocation
# Module:        momijo.tensor_light.memory.allocation
# Source:        src/momijo/tensor_light/memory/allocation.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines memory allocation in the memory and allocation layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `HostAllocation`, `allocate_host_fp32`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Delegate checked allocation to the owning storage implementation.
#   - Expose source-defined interfaces including `HostAllocation`,
#     `is_initialized`, `validate_metadata`, `length`.
#   - Preserve the owning allocator's error and lifetime contracts; do not
#     introduce a second allocator.
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
#   - Limit allocation overhead while satisfying size, alignment, and
#     ownership rules.
#   - Avoid redundant initialization, allocation, and copying of storage.
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

# Owning FP32 allocations. Payload count is logical; physical allocation is
# at least one element so an empty tensor does not require a fabricated pointer.
from std.memory.alloc import alloc, Layout, ManagedAllocation
from ..internal.status import TensorResult, TensorStatus
from std.sys.info import simd_width_of

struct HostAllocation(Movable):
    var memory: ManagedAllocation[Float32]
    var logical_count: Int
    var physical_count: Int
    # Only a COMPLETE logical overwrite can transition this owner to readable.
    # A public count/shape validation never implies initialized payload bytes.
    var initialized: Bool
    # The readable proof is tied to EXACT logical extent at completion.
    # Growing a mutable logical_count after a fill must invalidate the proof.
    var initialized_elements: Int

    def __init__(out self, var memory: ManagedAllocation[Float32], logical_count: Int, physical_count: Int):
        self.memory = memory
        self.logical_count = logical_count
        self.physical_count = physical_count
        # Zero logical elements have no readable payload. Physical padding
        # remains outside the logical bounds, including for rank > 0 empties.
        self.initialized = logical_count == 0
        self.initialized_elements = -1
        if logical_count == 0:
            self.initialized_elements = 0

    def is_initialized(self) -> Bool:
        return (self.validate_metadata() == TensorStatus.OK
                and self.initialized
                and self.initialized_elements == self.logical_count)

    def _invalidate_after_failed_write(mut self):
        # A native/Hybrid execution may have modified a proper subset of a
        # previously initialized owner. The bytes still belong to us, but
        # no public reader, view, clone or AOT input may observe them until
        # a separately proven COMPLETE overwrite restores initialization.
        # Poison the WHOLE logical allocation: a partial region is not an
        # acceptable ownership-level initialization certificate.
        self.initialized = False
        self.initialized_elements = -1

    def _mark_complete_overwrite(mut self) -> TensorStatus:
        # INTERNAL trusted writer contract: every logical Float32 element
        # must already have been written. Do not call for partial writes,
        # unjoined device copies or only a partially constructed result.
        var metadata = self.validate_metadata()
        if metadata != TensorStatus.OK:
            return metadata
        self.initialized = True
        self.initialized_elements = self.logical_count
        return TensorStatus.OK

    def validate_metadata(self) -> TensorStatus:
        # Only allocations from allocate_host_fp32 or a checked HostPool lease
        # carry this proof. Direct access to mutable allocation metadata can
        # invalidate it; callers must never trust cached counts without checking.
        if self.logical_count < 0 or self.physical_count < 1:
            return TensorStatus.INVALID_LAYOUT
        if self.logical_count > self.physical_count:
            return TensorStatus.SIZE_MISMATCH
        if self.physical_count > Int.MAX // 4:
            return TensorStatus.OVERFLOW
        # The ManagedAllocation owns the REAL allocation Layout. Metadata
        # counters are writable values and cannot prove physical capacity.
        # Match against the authoritative owner before any unsafe pointer or
        # Span access; never trust a forged physical_count after construction.
        if self.physical_count != self.memory.layout().count():
            return TensorStatus.SIZE_MISMATCH
        return TensorStatus.OK

    def length(self) -> Int:
        return self.logical_count

    def allocated_elements(self) -> Int:
        return self.physical_count

    def read_checked(self, index: Int) -> TensorResult[Float32]:
        var valid = self.validate_metadata()
        if valid != TensorStatus.OK:
            return TensorResult[Float32](valid)
        if index < 0 or index >= self.logical_count:
            return TensorResult[Float32](TensorStatus.OUT_OF_BOUNDS)
        # validate_metadata() already ran above; do not repeat its owner
        # Layout check in every read_host scalar fallback element.
        if not self.initialized or self.initialized_elements != self.logical_count:
            return TensorResult[Float32](TensorStatus.UNINITIALIZED)
        return TensorResult[Float32](self.memory.unsafe_ptr().unsafe_load(index)[0])

    def write_checked(mut self, index: Int, value: Float32) -> TensorStatus:
        var valid = self.validate_metadata()
        if valid != TensorStatus.OK:
            return valid
        if index < 0 or index >= self.logical_count:
            return TensorStatus.OUT_OF_BOUNDS
        self.memory.unsafe_ptr().unsafe_store(index, value)
        return TensorStatus.OK

# Allocation-only path for an INTERNAL complete overwrite. These bytes MUST
# NOT be read or exported until an owning constructor has written every
# logical FP32 element. Keep this separate from the safe public allocator.
def _allocate_host_fp32_uninitialized(count: Int) -> TensorResult[HostAllocation]:
    if count < 0:
        return TensorResult[HostAllocation](TensorStatus.INVALID_SHAPE)
    if count > Int.MAX // 4:
        return TensorResult[HostAllocation](TensorStatus.OVERFLOW)
    var physical = max(count, 1)
    var allocation = alloc(Layout[Float32].aligned[64](count=physical)).into_managed()
    return TensorResult[HostAllocation](HostAllocation(allocation, count, physical))


def allocate_host_fp32(count: Int) -> TensorResult[HostAllocation]:
    # Public allocation MUST be initialized, because HostAllocation exposes
    # read_checked(). The previous version returned readable uninitialized
    # floats. Materialization paths continue using the explicitly internal
    # complete-overwrite allocation, with no redundant zeroing pass.
    var acquired = _allocate_host_fp32_uninitialized(count)
    if not acquired.is_ok():
        return acquired
    var result = acquired.payload.take()
    var ptr = result.memory.unsafe_ptr()
    comptime width = simd_width_of[DType.float32]()
    var full = count - count % width
    var zeros = SIMD[DType.float32, width](Float32(0.0))
    for i in range(0, full, width):
        ptr.unsafe_store(i, zeros)
    for i in range(full, count):
        ptr.unsafe_store(i, Float32(0.0))
    # No logical element of a zero-element allocation is readable. Its single
    # physical padding slot remains inaccessible through checked APIs.
    var finalized = result._mark_complete_overwrite()
    if finalized != TensorStatus.OK:
        return TensorResult[HostAllocation](finalized)
    return TensorResult[HostAllocation](result)
