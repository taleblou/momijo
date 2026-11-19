# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR STORAGE / EXTERNAL BUFFER OWNERSHIP
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Storage and Ownership
# Module:        momijo.tensor_light.storage.external_buffer
# Source:        src/momijo/tensor_light/storage/external_buffer.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines external buffer ownership in the storage and ownership layer of
#   the FP32 Light tensor implementation.
#
#   Public declarations include `borrow_external_fp32`,
#   `borrow_mut_external_fp32`, `ExternalFp32Deleter`, `ExternalOwnedStorage`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Enforce tensor storage lifetime and initialized-data contracts.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `borrow_external_fp32`,
#     `borrow_mut_external_fp32`, `ExternalFp32Deleter`,
#     `ExternalOwnedStorage`.
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

# Explicit ownership boundary for FP32 host buffers supplied by foreign code.
# Borrowing and adoption are different operations: borrowed spans NEVER free
# their source, whereas an adopted buffer has exactly one move-only owner.
# The unsafe adoption contract cannot prove a foreign allocator's provenance;
# the caller MUST supply an exclusive allocation and its matching deleter.
from std.collections import Span
from std.memory import Pointer
from ..layout.layout import TensorLayout
from ..view.view import TensorView, TensorMutView
from ..internal.status import TensorResult, TensorStatus
from ..memory.alignment import is_valid_alignment
from .storage import MutableTensorStorage, HostSpanStorage


def borrow_external_fp32[rank: Int, origin: Origin](
    data: Span[False, Float32, origin], layout: TensorLayout[rank]
) -> TensorResult[TensorView[rank, origin]]:
    return TensorView[rank, origin].from_span_checked(data, layout)


def borrow_mut_external_fp32[rank: Int, origin: Origin[mut=True]](
    data: Span[True, Float32, origin], layout: TensorLayout[rank]
) -> TensorResult[TensorMutView[rank, origin]]:
    return TensorMutView[rank, origin].from_span_checked(data, layout)


# The callback is a CONCRETE, move-only value, selected through a compile-time
# trait. No closure, dynamic virtual table, mutable global allocator or
# common-free assumption is introduced. Implementations MUST be non-raising.
# The exact original allocation size, alignment and allocator belong to D.
trait ExternalFp32Deleter(Movable):
    def release(mut self, address: Int, allocation_elements: Int, alignment: Int): ...


# A custom-deleter host-storage capability. It is not a new public Tensor
# family: applications create Tensor[rank, ExternalOwnedStorage[D]].
# The address is untracked because ownership originates outside Mojo. It is
# NEVER an alias/noalias proof or permission to use the CPU for device memory.
struct ExternalOwnedStorage[D: ExternalFp32Deleter](Movable, HostSpanStorage):
    var _address: Int
    var _elements: Int
    var _alignment: Int
    var _deleter: Self.D
    var _initialized: Bool
    # The raw constructor is not an adoption proof. Even a forged pointer
    # with plausible numeric alignment MUST NOT become a readable owner or
    # invoke a foreign deleter unless unsafe_adopt_checked committed it.
    var _adopted: Bool

    def __init__(
        out self, address: Int, allocation_elements: Int,
        alignment: Int, var deleter: Self.D,
    ):
        self._address = address
        self._elements = allocation_elements
        self._alignment = alignment
        self._deleter = deleter
        # Fail closed on direct construction. Adoption and readable-state
        # publication occur together only in unsafe_adopt_checked below.
        self._initialized = False
        self._adopted = False

    def __deinit__(deinit self):
        # The direct constructor carries no committed foreign ownership.
        # Only the explicit unsafe checked adoption enables the deleter;
        # this also prevents an invalid intermediate from freeing caller data.
        # No implicit malloc/free substitution is permitted.
        if self._adopted:
            self._deleter.release(
                self._address, self._elements, self._alignment
            )

    @staticmethod
    def unsafe_adopt_checked(
        address: Int, allocation_elements: Int,
        alignment: Int, var deleter: Self.D,
    ) -> TensorResult[Self]:
        # PRECONDITION (supplied by caller, not mechanically verifiable):
        # * address begins one exclusive live HOST allocation of at least
        #   allocation_elements initialized Float32 values (zero is legal
        #   only with a real, non-null allocator-owned address);
        # * the deleter matches the exact allocator, count and alignment;
        # * no other owner will release it; no unfinished foreign I/O/DMA;
        # * the caller relinquishes ownership ONLY after success;
        # * D.release() is the ONLY function that frees the foreign buffer.
        # Reject bad metadata before moving ownership into the storage.
        # A zero-element allocation may back an empty rank-1..8 Tensor.
        # Adoption still requires one real, non-null, allocator-owned HOST
        # address and the matching deleter. This NEVER manufactures a null
        # Pointer or permits a zero-length span to be dereferenced.
        if allocation_elements < 0:
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        if allocation_elements > Int.MAX // 4:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        if not is_valid_alignment(alignment):
            return TensorResult[Self](TensorStatus.INVALID_ALIGNMENT)
        if address <= 0 or address % alignment != 0:
            return TensorResult[Self](TensorStatus.INVALID_ALIGNMENT)
        if address > Int.MAX - allocation_elements * 4:
            return TensorResult[Self](TensorStatus.OVERFLOW)
        # Commit ownership only AFTER all fallible numeric preflight.
        # This cannot establish that the foreign pointer is mapped or that
        # a caller-selected deleter is correct; the explicit unsafe contract
        # remains mandatory. It DOES block the ordinary constructor bypass.
        var adopted = Self(address, allocation_elements, alignment, deleter)
        adopted._adopted = True
        adopted._initialized = True
        return TensorResult[Self](adopted)

    def capacity(self) -> Int:
        # Recheck mutable metadata at every checked boundary. This is NOT a
        # provenance proof: forging the fields or violating the foreign
        # allocation contract remains unsafe and is expressly unsupported.
        if not self._adopted:
            return -1
        if self._elements < 0 or self._elements > Int.MAX // 4:
            return -1
        if not is_valid_alignment(self._alignment):
            return -1
        if self._address <= 0 or self._address % self._alignment != 0:
            return -1
        if self._address > Int.MAX - self._elements * 4:
            return -1
        return self._elements

    def is_host_accessible(self) -> Bool:
        # An uncommitted raw constructor does not prove a usable host owner.
        # The foreign allocator/address-space claim remains an unsafe caller
        # obligation even after checked adoption.
        return self.capacity() >= 0

    def is_initialized(self) -> Bool:
        return self._initialized and self.capacity() >= 0

    def _invalidate_after_failed_write(mut self):
        # External memory stays with its exact caller-supplied deleter; only
        # the initialized/readable certification is revoked after an unsafe
        # operation may have partially overwritten this owner.
        self._initialized = False

    def read_host(self, index: Int) -> TensorResult[Float32]:
        var count = self.capacity()
        if count < 0:
            return TensorResult[Float32](TensorStatus.INVALID_LAYOUT)
        if not self._initialized:
            return TensorResult[Float32](TensorStatus.UNINITIALIZED)
        if index < 0 or index >= count:
            return TensorResult[Float32](TensorStatus.OUT_OF_BOUNDS)
        var ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=self._address)
        return TensorResult[Float32](ptr.unsafe_load(index)[0])

    def write_host(mut self, index: Int, value: Float32) -> TensorStatus:
        var count = self.capacity()
        if count < 0:
            return TensorStatus.INVALID_LAYOUT
        if index < 0 or index >= count:
            return TensorStatus.OUT_OF_BOUNDS
        var ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=self._address)
        ptr.unsafe_store(index, value)
        return TensorStatus.OK

    def fill_host(mut self, value: Float32) -> TensorStatus:
        var count = self.capacity()
        if count < 0:
            return TensorStatus.INVALID_LAYOUT
        var ptr = Pointer[Float32, MutUntrackedOrigin](unsafe_from_address=self._address)
        for index in range(count):
            ptr.unsafe_store(index, value)
        self._initialized = True
        return TensorStatus.OK

    # Checked entry points for consumers that have not already proven an
    # owning Tensor descriptor. In contrast, the HostSpanStorage trait's
    # borrow_span() primitives are explicitly unchecked after owner proof.
    def try_borrow_span(
        ref self,
    ) -> TensorResult[Span[False, Float32, origin_of(self)]]:
        if self.capacity() < 0:
            return TensorResult[Span[False, Float32, origin_of(self)]](
                TensorStatus.INVALID_LAYOUT
            )
        if not self._initialized:
            return TensorResult[Span[False, Float32, origin_of(self)]](
                TensorStatus.UNINITIALIZED
            )
        return TensorResult[Span[False, Float32, origin_of(self)]](
            self.borrow_span()
        )

    def try_borrow_mut_span(
        mut self,
    ) -> TensorResult[Span[True, Float32, origin_of(self)]]:
        if self.capacity() < 0:
            return TensorResult[Span[True, Float32, origin_of(self)]](
                TensorStatus.INVALID_LAYOUT
            )
        # A mutable owner may be deliberately uninitialized immediately
        # before a complete overwrite. This method proves only the storage
        # lifetime and bounds, never that its previous payload was readable.
        return TensorResult[Span[True, Float32, origin_of(self)]](
            self.borrow_mut_span()
        )

    def borrow_span(ref self) -> Span[False, Float32, origin_of(self)]:
        # INTERNAL ONLY: caller has checked capacity() and full initialization
        # while holding the owner borrow. The foreign-memory contract must
        # remain valid until the borrow ends; no ownership transfer occurs.
        var ptr = Pointer[Float32, origin_of(self)](
            unsafe_from_address=self._address
        )
        return Span[False, Float32, origin_of(self)](
            unsafe_ptr=ptr, length=self._elements
        )

    def borrow_mut_span(mut self) -> Span[True, Float32, origin_of(self)]:
        # INTERNAL ONLY: checked, unique, writable owner borrow required.
        var ptr = Pointer[Float32, origin_of(self)](
            unsafe_from_address=self._address
        )
        return Span[True, Float32, origin_of(self)](
            unsafe_ptr=ptr, length=self._elements
        )
