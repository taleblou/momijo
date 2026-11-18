# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR INTERNAL / INVARIANT CHECKS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Internal Validation Utilities
# Module:        momijo.tensor_light.internal.invariants
# Source:        src/momijo/tensor_light/internal/invariants.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor invariant checks in the internal validation utilities layer
#   of the FP32 Light tensor implementation.
#
#   Public declarations include `check_shape_invariants`,
#   `check_storage_invariants`, `check_layout_invariants`,
#   `check_view_invariants`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Provide reusable checked primitives and status contracts.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `check_shape_invariants`,
#     `check_storage_invariants`, `check_layout_invariants`,
#     `check_view_invariants`, `check_tensor_invariants`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Detect integer overflow and invalid metadata before dereference.
#   - Return explicit statuses for unsupported or invalid states.
#
# Performance Objectives:
#   - Use simple inlinable checked primitives where appropriate.
#   - Keep validation work proportional to required metadata.
#
# Maintainability:
#   - Reuse one error vocabulary across dependent modules.
#   - Test boundary arithmetic and invalid-state propagation.
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

# Expensive, owner-backed boundary/debug invariant audits only.
# No unchecked memory access, no runtime data sampling and no hot-kernel work.
from ..core.tensor import Tensor
from ..storage.storage import TensorStorage
from ..shape.shape import TensorShape
from ..layout.layout import TensorLayout
from ..view.view import TensorView, TensorMutView
from .status import TensorStatus


def check_shape_invariants[rank: Int](shape: TensorShape[rank]) -> TensorStatus:
    # Previously checked only negative/empty extents: a nonempty descriptor
    # such as shape=(2,3), cached elements=7 was incorrectly accepted.
    # TensorShape.checked_numel rebuilds the exact product with overflow
    # detection and verifies its agreement with the cached value.
    var verified = shape.checked_numel()
    return verified.status


def check_storage_invariants[S: TensorStorage](
    ref storage: S, require_readable: Bool = False,
) -> TensorStatus:
    # This is a generic capability-level audit. HostStorage and
    # DeviceStorage.capacity() independently validate their real allocation
    # metadata; an external unsafe raw owner still cannot authenticate its
    # foreign allocator, pointer mapping or deleter from source inspection.
    var physical = storage.capacity()
    if physical < 0:
        return TensorStatus.INVALID_LAYOUT
    if physical > Int.MAX // 4:
        return TensorStatus.OVERFLOW
    if require_readable and not storage.is_initialized():
        return TensorStatus.UNINITIALIZED
    return TensorStatus.OK


def check_layout_invariants[rank: Int](
    layout: TensorLayout[rank], capacity: Int,
) -> TensorStatus:
    # Recompute AND compare cached span, overlap and shape metadata, not
    # merely the previously claimed inclusive physical bounds.
    var checked = layout.validated_snapshot(capacity)
    return checked.status


def check_view_invariants[rank: Int, origin: Origin](
    ref view: TensorView[rank, origin],
) -> TensorStatus:
    # Immutable borrowed data need not have an injective mapping, but all
    # reachable offsets must be within its borrowed Span.
    var checked = view.checked_layout()
    return checked.status


def check_view_invariants[rank: Int, origin: Origin[mut=True]](
    ref view: TensorMutView[rank, origin],
) -> TensorStatus:
    # A mutable borrowed view additionally needs unique addresses.
    # A forged cached overlap flag is rejected by validated_snapshot().
    var checked = view.checked_layout()
    if not checked.is_ok():
        return checked.status
    if not checked.payload.value().is_proven_unique():
        return TensorStatus.OVERLAPPING
    return TensorStatus.OK


def check_tensor_invariants[rank: Int, S: TensorStorage](
    ref tensor: Tensor[rank, S], require_readable: Bool = False,
) -> TensorStatus:
    # The owner is borrowed, never moved into a debug predicate. A partially
    # poisoned output remains structurally inspectable unless the caller
    # explicitly requests proof of a readable tensor payload.
    var owner = check_storage_invariants(tensor.impl.storage, require_readable)
    if owner != TensorStatus.OK:
        return owner
    return check_layout_invariants(tensor.impl.layout, tensor.impl.storage.capacity())
