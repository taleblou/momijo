# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR STORAGE / STORAGE IMPLEMENTATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Storage and Ownership
# Module:        momijo.tensor_light.storage.storage_impl
# Source:        src/momijo/tensor_light/storage/storage_impl.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines storage implementation in the storage and ownership layer of the
#   FP32 Light tensor implementation.
#
#   Public declarations include `OwnedStorageImpl`; preconditions and
#   supported targets are determined by their implementations.
#
# Responsibilities:
#   - Enforce tensor storage lifetime and initialized-data contracts.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `OwnedStorageImpl`.
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

# Concrete owned FP32 storage. Layout and tensor metadata never live here.
# HostAllocation is move-only; this wrapper never duplicates allocation rights.
from ..memory.allocation import HostAllocation, _allocate_host_fp32_uninitialized
from ..internal.status import TensorResult, TensorStatus
from .storage import MutableTensorStorage

struct OwnedStorageImpl(Movable, MutableTensorStorage):
    var owned: HostAllocation

    def __init__(out self, var owned: HostAllocation):
        self.owned = owned

    @staticmethod
    def zeros(count: Int) -> TensorResult[Self]:
        var a = _allocate_host_fp32_uninitialized(count)
        if not a.is_ok():
            return TensorResult[Self](a.status)
        var storage = Self(a.payload.take())
        var initialized = storage.fill_host(0.0)
        if initialized != TensorStatus.OK:
            return TensorResult[Self](initialized)
        return TensorResult[Self](storage)

    def capacity(self) -> Int:
        if self.owned.validate_metadata() != TensorStatus.OK:
            return -1
        return self.owned.length()

    def is_host_accessible(self) -> Bool:
        return True

    def is_initialized(self) -> Bool:
        return self.owned.is_initialized()

    def read_host(self, index: Int) -> TensorResult[Float32]:
        return self.owned.read_checked(index)

    def write_host(mut self, index: Int, value: Float32) -> TensorStatus:
        return self.owned.write_checked(index, value)

    def fill_host(mut self, value: Float32) -> TensorStatus:
        # Recheck the move-only allocation's mutable count metadata before
        # writing through its unchecked pointer. No per-element status needed.
        var valid = self.owned.validate_metadata()
        if valid != TensorStatus.OK:
            return valid
        var ptr = self.owned.memory.unsafe_ptr()
        for i in range(self.owned.length()):
            ptr.unsafe_store(i, value)
        return self.owned._mark_complete_overwrite()
