# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR STORAGE / STORAGE INTERFACES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Storage and Ownership
# Module:        momijo.tensor_light.storage.storage
# Source:        src/momijo/tensor_light/storage/storage.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines storage interfaces in the storage and ownership layer of the FP32
#   Light tensor implementation.
#
#   Public declarations include `TensorStorage`, `MutableTensorStorage`,
#   `HostSpanStorage`; preconditions and supported targets are determined by
#   their implementations.
#
# Responsibilities:
#   - Enforce tensor storage lifetime and initialized-data contracts.
#   - Preserve documented behavior for ownership, alias provenance, lifetime,
#     and initialized storage.
#   - Provide source-declared interfaces such as `TensorStorage`,
#     `MutableTensorStorage`, `HostSpanStorage`.
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

from std.collections import Span
# Storage is a compile-time capability, never a runtime virtual base class.
# A non-host backend can return BACKEND_UNAVAILABLE from checked host access.
from ..internal.status import TensorStatus, TensorResult

# Every storage held by an owning Tensor has move semantics. Requiring this
# at the capability boundary rules out non-movable handles as tensor owners.
trait TensorStorage(Movable, Deinitable):
    def capacity(self) -> Int: ...
    def read_host(self, index: Int) -> TensorResult[Float32]: ...
    def is_host_accessible(self) -> Bool: ...
    # Owner-level initialized-state precondition for readable Tensor APIs.
    def is_initialized(self) -> Bool: ...

trait MutableTensorStorage(TensorStorage):
    # Mutating capabilities are statically constrained by the trait.
    def write_host(mut self, index: Int, value: Float32) -> TensorStatus: ...
    def fill_host(mut self, value: Float32) -> TensorStatus: ...


# Host-only, allocation-lifetime-bound Span capability. DeviceStorage does
# not implement this trait; generic Tensor methods cannot accidentally
# dereference GPU-local buffers from the CPU.
trait HostSpanStorage(MutableTensorStorage):
    def borrow_span[origin: Origin[mut=False]](ref[origin] self) -> Span[Float32, origin]: ...
    def borrow_mut_span[origin: Origin[mut=True]](ref[origin] self) -> Span[Float32, origin]: ...
    # A partial native/Hybrid CPU failure cannot preserve the old full-owner
    # initialization certificate. Every concrete host-span owner implements
    # this fail-closed transition, including adopted external host memory.
    def _invalidate_after_failed_write(mut self): ...
