# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR OPS / OPERATIONS INTERFACE
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Operations
# Module:        momijo.tensor_light.ops
# Source:        src/momijo/tensor_light/ops/__init__.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines the tensor operations package boundary for the FP32 Light
#   implementation.
#
#   Re-exports explicitly imported interfaces without declaring an independent
#   implementation.
#
# Responsibilities:
#   - Define the tensor operations import and namespace boundary.
#   - Keep imported public names consistent with their owning modules.
#   - Avoid introducing duplicate implementations or initialization side
#     effects.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Expose only symbols actually defined by imported modules.
#   - Preserve the established package API and import graph.
#
# Performance Objectives:
#   - Avoid initialization side effects or eager heavy allocations.
#   - Keep executable hot-path implementation in dedicated modules.
#
# Maintainability:
#   - Update curated exports when public APIs move or change.
#   - Do not add import cycles or duplicate implementations.
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

# Curated package exports. No initialization side effects.
from .arithmetic import add, sub, mul, div, maximum, minimum, add_scalar, sub_scalar, mul_scalar, div_scalar, fma, fma_nd
from .comparison import eq, ne, lt, le, gt, ge, isnan, isinf, isfinite
from .mask import TensorMask, mask_and, mask_or, mask_xor, mask_not, mask_any, mask_all
from .copy import clone, copy, contiguous, reshape_copy, ravel_copy, flatten_copy, copy_view_into, fill_view, zero_view, to_device, to_host
from .reduction import sum_all, mean_all, prod_all, min_all, max_all, argmin_all, argmax_all
from .reduction import KeepDims, DropDims, sum, prod, min, max, mean, argmin, argmax
from .scan import cumsum, cumprod, cummin, cummax
from .unary import unary, UnaryKind, neg, abs, sqrt, exp, log, sin, cos, tanh, reciprocal
from .where import where, where_nd, masked_fill
from .matrix import mm, dot, outer, gemv, vecmat
from .reduction import sum_axes_keepdims, prod_axes_keepdims, min_axes_keepdims, max_axes_keepdims, mean_axes_keepdims
from .transform import tile
from .transform import transpose_copy2
from .transform import concatenate2, stack2, repeat_axis, tile_axis
from .transform import split, chunk, repeat
from .indexing import gather, scatter, narrow, ScatterConflictPolicy
from .arithmetic import add_nd, sub_nd, mul_nd, div_nd
from .matrix import bmm, matmul
from .comparison import compare_nd
from .matrix import matmul_nd, matmul_mixed
# The rank-preserving axis reductions are also public, not only their
# multi-axis counterparts. They are thin delegates to the owning Tensor.
from .reduction import sum_axis_keepdims, prod_axis_keepdims, min_axis_keepdims, max_axis_keepdims, mean_axis_keepdims

# Zero-copy views remain borrowed and never materialize an owning Tensor.
from .view import transpose, reshape_view, reshape_f_view, permute
from .view import swapaxes, movedim, unsqueeze, squeeze_axis, squeeze_axes
from .view import expand, broadcast_to, flatten_view, ravel_view, ravel_f_view, flip

# Rank-reducing operators use the same FP32 Tensor implementation as keepdims.
from .reduction import sum_axis, prod_axis, min_axis, max_axis, mean_axis
from .reduction import sum_axes, prod_axes, min_axes, max_axes, mean_axes

from .reduction import argmin_axis, argmax_axis, argmin_axis_keepdims, argmax_axis_keepdims
from .logical import logical_and, logical_or, logical_xor, logical_not, any_true, all_true, count_true

# FP32 CPU application operators: no additional storage dtype or rank support.
from .nn import relu, gelu_tanh, softmax_last_axis, logsumexp_vector
from .nn import layernorm_last_axis, scaled_dot_product_attention_2d
