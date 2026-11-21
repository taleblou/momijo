# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR CORE / IMPLEMENTATION
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Core Tensor Types
# Module:        momijo.tensor_light.core.tensor_impl
# Source:        src/momijo/tensor_light/core/tensor_impl.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor implementation and the related core tensor representation
#   contracts.
#
#   Public declarations include `TensorImpl`; preconditions and supported
#   targets are determined by their implementations.
#
# Responsibilities:
#   - Maintain tensor representation and metadata invariants.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `TensorImpl`.
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

# Value-type owned implementation, no mandatory heap Pimpl or intrusive refcount.
from ..storage.storage import TensorStorage, HostSpanStorage
from ..layout.layout import TensorLayout
from .tensor_descriptor import TensorDescriptor, make_descriptor
from ..view.view import TensorView, TensorMutView
from ..storage.host_storage import HostStorage
from ..storage.storage_ref import StorageRef, MutableStorageRef
from ..internal.status import TensorResult, TensorStatus

struct TensorImpl[rank: Int, S: TensorStorage](Movable):
    var storage: Self.S
    var layout: TensorLayout[Self.rank]

    def __init__(out self, var owned: Self.S, layout: TensorLayout[Self.rank]):
        self.storage = owned^
        self.layout = layout

    def _take_storage(deinit self) -> Self.S:
        (self.layout^).__deinit__()
        return self.storage^

    # Every native/borrowed projection gets its proof from this owner rather
    # than trusting a caller-mutable TensorLayout or cached span flags.
    def checked_descriptor(self) -> TensorResult[TensorDescriptor[Self.rank]]:
        # An in-bounds descriptor is not a valid readable view while its
        # backing allocation is still awaiting a complete initialization.
        if not self.storage.is_initialized():
            return TensorResult[TensorDescriptor[Self.rank]](TensorStatus.UNINITIALIZED)
        return make_descriptor[Self.rank](self.layout, self.storage.capacity())

    def descriptor(self) -> TensorResult[TensorDescriptor[Self.rank]]:
        # Retain checked ownership bounds; mutable impl fields cannot be an
        # unchecked guarantee about the storage or cached layout.
        return self.checked_descriptor()

    def storage_ref(
        ref self,
    ) -> TensorResult[StorageRef[origin_of(self.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var proof = self.checked_descriptor()
        if not proof.is_ok():
            return TensorResult[StorageRef[origin_of(self.storage)]](proof.status)
        return TensorResult[StorageRef[origin_of(self.storage)]](
            StorageRef[origin_of(self.storage)](self.storage.borrow_span())
        )

    def mut_storage_ref(
        mut self,
    ) -> TensorResult[MutableStorageRef[origin_of(self.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var proof = self.checked_descriptor()
        if not proof.is_ok():
            return TensorResult[MutableStorageRef[origin_of(self.storage)]](proof.status)
        return TensorResult[MutableStorageRef[origin_of(self.storage)]](
            MutableStorageRef[origin_of(self.storage)](self.storage.borrow_mut_span())
        )

    def view(
        ref self
    ) -> TensorResult[TensorView[Self.rank, origin_of(self.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var descriptor = self.checked_descriptor()
        if not descriptor.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.storage)]](
                descriptor.status
            )
        # A single checked descriptor has already validated shape, strides,
        # physical extent and the live owner's capacity. Repeating that proof
        # in from_span_checked would add only metadata work on this hot path.
        return TensorResult[TensorView[Self.rank, origin_of(self.storage)]](
            TensorView[Self.rank, origin_of(self.storage)](
                self.storage.borrow_span(), descriptor.payload.value().mapping
            )
        )

    def mut_view(
        mut self
    ) -> TensorResult[TensorMutView[Self.rank, origin_of(self.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var descriptor = self.checked_descriptor()
        if not descriptor.is_ok():
            return TensorResult[TensorMutView[Self.rank, origin_of(self.storage)]](
                descriptor.status
            )
        var mapping = descriptor.payload.value().mapping
        if not mapping.is_proven_unique():
            return TensorResult[TensorMutView[Self.rank, origin_of(self.storage)]](
                TensorStatus.OVERLAPPING
            )
        # The verified owner guarantees physical bounds; uniqueness is
        # proven above. Avoid a second full metadata validation here.
        return TensorResult[TensorMutView[Self.rank, origin_of(self.storage)]](
            TensorMutView[Self.rank, origin_of(self.storage)](
                self.storage.borrow_mut_span(), mapping
            )
        )
