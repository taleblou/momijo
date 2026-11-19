# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR TRANSFER / COPYING
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Transfers
# Module:        momijo.tensor_light.transfer.copy
# Source:        src/momijo/tensor_light/transfer/copy.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines tensor copying in the tensor transfers layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `copy_from`, `copy_disjoint_owned`,
#   `copy_borrowed_overlap_safe`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Plan or perform data copies and device transfers safely.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `copy_from`,
#     `copy_disjoint_owned`, `copy_borrowed_overlap_safe`.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate host/device accessibility, bounds, and transfer direction.
#   - Preserve required completion and ownership lifetimes.
#
# Performance Objectives:
#   - Use contiguous transfer paths when layouts permit.
#   - Minimize redundant staging or host/device round trips.
#
# Maintainability:
#   - Keep checked transfer and unsafe enqueue APIs distinct.
#   - Test overlapping regions, failures, and asynchronous completion.
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

# Unified owning and borrowed FP32 host copy entry points. Every borrowed
# view-to-view transfer delegates to the canonical checked CPU copy engine;
# there is no duplicate alias policy or second strided copy implementation.
from ..core.tensor import Tensor
from ..storage.host_storage import HostStorage
from ..view.view import TensorView, TensorMutView
from ..internal.status import TensorStatus
from ..kernels.cpu.copy import cpu_copy_strided


def copy_from[rank: Int](
    mut destination: Tensor[rank, HostStorage],
    ref source: Tensor[rank, HostStorage],
) -> TensorStatus:
    return destination.copy_from(source)


def copy_disjoint_owned[rank: Int](
    mut destination: Tensor[rank, HostStorage],
    ref source: Tensor[rank, HostStorage],
) -> TensorStatus:
    # Unique HostAllocation owners are disjoint by the ownership contract.
    return destination.copy_from(source)


def copy_borrowed_overlap_safe[
    rank: Int, do: Origin[mut=True], so: Origin,
](
    mut destination: TensorMutView[rank, do], source: TensorView[rank, so],
) -> TensorStatus:
    # Checked copy preserves the source logical order for C/F, negative and
    # zero input strides. Its snapshot path handles unknown physical aliases
    # before any write; destination mapping must be one-to-one.
    return cpu_copy_strided(source, destination)
