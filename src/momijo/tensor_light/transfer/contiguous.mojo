# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR TRANSFER / CONTIGUOUS COPIES
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Transfers
# Module:        momijo.tensor_light.transfer.contiguous
# Source:        src/momijo/tensor_light/transfer/contiguous.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines contiguous copies in the tensor transfers layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `contiguous_copy`, `contiguous_copy_order`,
#   `reshape_copy`, `flatten_copy`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Plan or perform data copies and device transfers safely.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `contiguous_copy`,
#     `contiguous_copy_order`, `reshape_copy`, `flatten_copy`, `ravel_copy`.
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

# Explicit host C-order materialization. No implicit device readback or copy.
# Core Tensor owns the canonical deep clone path; transfer never imports ops.
from ..core.tensor import Tensor
from ..storage.host_storage import HostStorage
from ..internal.status import TensorResult
from ..shape.shape import TensorShape
from ..layout.memory_format import ContiguousOrder


def contiguous_copy[rank: Int](ref source: Tensor[rank, HostStorage]) -> TensorResult[Tensor[rank, HostStorage]]:
    return source.clone()


def contiguous_copy_order[rank: Int](
    ref source: Tensor[rank, HostStorage], order: ContiguousOrder,
) -> TensorResult[Tensor[rank, HostStorage]]:
    return source.contiguous_copy_order(order)


def reshape_copy[src_rank: Int, out_rank: Int](
    ref source: Tensor[src_rank, HostStorage], shape: TensorShape[out_rank],
) -> TensorResult[Tensor[out_rank, HostStorage]]:
    return source.reshape_copy[out_rank](shape)


def flatten_copy[rank: Int, start_dim: Int, end_dim: Int](
    ref source: Tensor[rank, HostStorage],
) -> TensorResult[Tensor[rank - (end_dim - start_dim), HostStorage]] where rank >= 1:
    return source.flatten_copy[start_dim, end_dim]()


def ravel_copy[rank: Int](
    ref source: Tensor[rank, HostStorage],
) -> TensorResult[Tensor[1, HostStorage]]:
    return source.ravel_copy()
