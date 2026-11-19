# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR STORAGE / STORAGE REFERENCES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Storage and Ownership
# Module:        momijo.tensor_light.storage.storage_ref
# Source:        src/momijo/tensor_light/storage/storage_ref.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines storage references in the storage and ownership layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `StorageRef`, `MutableStorageRef`;
#   preconditions and supported targets are determined by their
#   implementations.
#
# Responsibilities:
#   - Enforce tensor storage lifetime and initialized-data contracts.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `StorageRef`,
#     `MutableStorageRef`.
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

# Borrowed FP32 capacity + provenance. Does not own the underlying bytes.
from std.collections import Span

struct StorageRef[origin: Origin](Copyable):
    var elements: Span[False, Float32, Self.origin]
    var known_allocation_id: Int

    def __init__(out self, elements: Span[False, Float32, Self.origin], known_allocation_id: Int = -1):
        self.elements = elements
        self.known_allocation_id = known_allocation_id

    def __init__(out self, *, copy: Self):
        self.elements = copy.elements
        self.known_allocation_id = copy.known_allocation_id

    def capacity(self) -> Int:
        return len(self.elements)


# A mutable STORAGE capability is not a mutable Tensor view: only TensorImpl
# can create one from an exclusive checked host owner. It cannot outlive that
# owner and is move-only, preventing accidental copying of write capabilities.
struct MutableStorageRef[origin: Origin[mut=True]](Movable):
    var elements: Span[True, Float32, Self.origin]

    def __init__(out self, elements: Span[True, Float32, Self.origin]):
        self.elements = elements

    def capacity(self) -> Int:
        return len(self.elements)
