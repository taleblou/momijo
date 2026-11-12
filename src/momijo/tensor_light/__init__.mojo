# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR / PUBLIC TENSOR API INTERFACE
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Public Tensor API
# Module:        momijo.tensor_light
# Source:        src/momijo/tensor_light/__init__.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines the public tensor API package boundary for the FP32 Light
#   implementation.
#
#   Curates public FP32 interfaces while leaving expert kernel and launcher
#   APIs in their dedicated modules.
#
# Responsibilities:
#   - Define the public tensor API import and namespace boundary.
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

# Curated public FP32 tensor surface. Tensor is the primary owning value.
# Backend-specific launchers, raw spans and mutable schedules are expert APIs
# and must be imported from their modules, never from this root package.
from .core.tensor import Tensor, UninitializedHostTensor, UninitializedDeviceTensor
from .core.tensor_options import TensorOptions, default_host_options, default_tensor_options, with_order, with_alignment, with_device, with_memory_space, with_initialization_policy
from .shape.shape import TensorShape
from .shape.shape_storage import StaticRankArray
from .storage.host_storage import HostStorage
from .storage.device_storage import DeviceStorage
from .device.device import Device, DeviceType, cpu_device, gpu_device, gpu_device_checked, validate_device, same_device
from .layout.layout import TensorLayout
from .layout.memory_format import ContiguousOrder
from .internal.status import TensorStatus, TensorStatusCode, TensorResult
from .config import MAX_TENSOR_RANK, TENSOR_DTYPE, TENSOR_ITEMSIZE
# Source-spec configuration and target policy. Target profiles are structural
# metadata, never evidence of an authenticated native executable.
from .config import TensorCoreConfig, AlignmentPolicy, TargetTuningProfile
from .config import validate_rank_config, default_contiguous_order
from .config import preferred_alignment, preferred_simd_width
from .config import default_fp32_math_mode, default_reduction_accuracy_mode
from .config import default_execution_preference, hybrid_execution_enabled
from .config import target_tuning_profile, target_tuning_profile_from_offline
from .config import hybrid_tuning_profile, hybrid_tuning_profile_from_offline
from .execution.execution_context import ExecutionPreference
from .execution.launch_policy import HybridTuningProfile
from .factory.creation import empty, zeros, ones, full
from .factory.creation import empty_with_options, full_with_options, zeros_with_options, ones_with_options
from .factory.like import empty_like, zeros_like, ones_like, full_like
from .ops.arithmetic import add, sub, mul, div, maximum, minimum, add_scalar, sub_scalar, mul_scalar, div_scalar, fma, fma_nd
from .dtype.math_mode import Fp32MathMode, ReductionAccuracyMode
from .ops.reduction import sum_all, mean_all, prod_all, min_all, max_all, argmin_all, argmax_all
from .ops.reduction import KeepDims, DropDims, sum, prod, min, max, mean, argmin, argmax
from .ops.matrix import mm, dot, outer, gemv, vecmat
from .ops.copy import clone, copy, contiguous, reshape_copy, ravel_copy, flatten_copy
from .ops.mask import TensorMask
from .ops.comparison import eq
from .execution.execution_plan import ExecutionPlan, ExecutionPlacement, StaticAOTExecutionPlan
from .execution.execution_context import ExecutionContext, make_cpu_context, make_gpu_context, make_hybrid_context
from .factory.from_buffer import from_copy_span, from_copy_scalar
from .ops.transform import tile
from .ops.transform import transpose_copy2
from .ops.reduction import sum_axes_keepdims, prod_axes_keepdims, min_axes_keepdims, max_axes_keepdims, mean_axes_keepdims
from .ops.transform import concatenate2, stack2, repeat_axis, tile_axis
from .ops.transform import split, chunk, repeat
from .ops.comparison import ne, lt, le, gt, ge
from .ops.comparison import isnan, isinf, isfinite
from .ops.scan import cumsum, cumprod, cummin, cummax, exclusive_cumsum, exclusive_cumprod
from .ops.where import where, where_nd, masked_fill
from .view.view import TensorView, TensorMutView
from .core.tensor_ref import TensorRef, TensorMutRef

from .ops.mask import mask_and, mask_or, mask_xor, mask_not, mask_any, mask_all
from .ops.unary import neg, abs, sqrt, exp, log, sin, cos, tanh, reciprocal

# Heterogeneous compile-time-rank basic indexing descriptors.
from .indexing.slice_spec import (
    BasicIndex, IndexSpec, BasicIndexKind, NewAxis, EllipsisIndex,
    newaxis, ellipsis_index, validate_basic_index_pack,
    result_rank_of_basic_index_pack, count_ellipsis, count_consumed_axes,
)
from .indexing.slice import SliceSpec
from .ops.matrix import bmm, matmul
from .ops.arithmetic import add_nd, sub_nd, mul_nd, div_nd
from .ops.comparison import compare_nd
from .ops.indexing import gather, scatter, ScatterConflictPolicy
from .ops.reduction import sum_axis_keepdims, prod_axis_keepdims, min_axis_keepdims, max_axis_keepdims, mean_axis_keepdims
# Rank-reducing FP32 reductions: output rank known at compile time.
from .ops.reduction import sum_axis, prod_axis, min_axis, max_axis, mean_axis
from .ops.reduction import sum_axes, prod_axes, min_axes, max_axes, mean_axes

# Rank-static matrix operation with broadcastable leading batch dimensions.
from .ops.matrix import matmul_nd

# Different compile-time input ranks, right-aligned broadcast batch dimensions.
from .ops.matrix import matmul_mixed

# AOT-specialized host-resident FP32 binary lowering across all mappings.
# For GPU/Split, supply the offline artifact for the selected operation/variant.

# Ownership-proven CPU_ONLY lowering: SIMD dense and strided reference paths.

# Safe public GPU kernel entry points wait through the completion event.
# Low-level enqueue_* functions remain advanced APIs under backend.gpu;
# they are NOT re-exported from the owning Tensor package surface because
# a DeviceEvent alone does not retain the caller's buffer allocations.


# Offline AOT artifact identity must match FP32 opcode, architecture and ABI.
from .dispatch.specialization import BinaryAOTArtifactIdentity

# Strict FP32 SIMT device GEMM, rank-2 contiguous reference lowering.


# Validated host-output AOT lowering (CPU, GPU, Split, CPU Assist).

# Cost-gated two-slot FP32 CPU-prepare/GPU pipeline (host result only).

# Device-owner scoped synchronous AOT binary execution; no dangling async Tensor.

from .view.adapters import try_as_static_c_layout_tensor_1d, try_as_mut_static_c_layout_tensor_1d

from .view.adapters import try_as_static_c_layout_tensor_2d, try_as_mut_static_c_layout_tensor_2d

# Explicit zero-cast FP32 placement and narrow completion-synchronized readback.
from .ops.copy import to_device, to_host

# Intrinsic AOT FP32 GPU unary kernel with synchronous ownership scope.

# Exact C-contiguous flattened rank 0..8 MAX zero-copy bridges.
from .view.adapters import try_as_static_flat_c_layout_tensor
from .view.adapters import try_as_mut_static_flat_c_layout_tensor

# Rank-preserving, nonmaterializing AOT C-order MAX LayoutTensor bridges.
# All dimensions must match the requested static shape and be nonzero.
from .view.adapters import try_as_static_c_layout_tensor_3d, try_as_mut_static_c_layout_tensor_3d
from .view.adapters import try_as_static_c_layout_tensor_4d, try_as_mut_static_c_layout_tensor_4d
from .view.adapters import try_as_static_c_layout_tensor_5d, try_as_mut_static_c_layout_tensor_5d
from .view.adapters import try_as_static_c_layout_tensor_6d, try_as_mut_static_c_layout_tensor_6d
from .view.adapters import try_as_static_c_layout_tensor_7d, try_as_mut_static_c_layout_tensor_7d
from .view.adapters import try_as_static_c_layout_tensor_8d, try_as_mut_static_c_layout_tensor_8d

# Offline trusted AOT GEMM metadata; use Tensor.device_mm_into() for ownership.
from .dispatch.specialization import GemmAOTArtifactIdentity

# Bound unary AOT metadata; use Tensor.device_unary_into() for owner safety.
from .dispatch.specialization import UnaryAOTArtifactIdentity

# Optional functional spellings for structural views. Tensor methods remain
# the canonical external API and the sole implementation owner.
from .ops.view import transpose, reshape_view, reshape_f_view, permute
from .ops.view import swapaxes, movedim, unsqueeze, squeeze_axis
from .ops.view import expand, broadcast_to, flatten_view, ravel_view, ravel_f_view
from .ops.indexing import narrow
from .ops.view import flip

# Checked external AOT identity for FP32 contiguous row reductions.
from .dispatch.specialization import ReduceRowsAOTArtifactIdentity
from .dispatch.specialization import ScanRowsAOTArtifactIdentity

# Rank-reducing reduction exports are declared once above; Tensor remains
# the canonical external API. No second public tensor owner is introduced.

# Explicit multi-axis zero-copy singleton removal with static result rank.
from .ops.view import squeeze_axes

# Foreign allocations are adopted explicitly; borrowed buffers stay borrowed.
from .storage.external_buffer import ExternalFp32Deleter, ExternalOwnedStorage
from .factory.from_buffer import unsafe_adopt_external_fp32_tensor

# Optional MAX TileTensor experts: every returned tile borrows from the one
# public Tensor owner. The primary public type remains `Tensor` itself.
from .view.adapters import try_as_static_flat_c_tile_tensor, try_as_mut_static_flat_c_tile_tensor
from .view.adapters import try_as_static_c_tile_tensor_2d, try_as_mut_static_c_tile_tensor_2d
from .view.adapters import try_as_static_f_tile_tensor_2d, try_as_mut_static_f_tile_tensor_2d

# Rank-preserving, borrowed AOT MAX TileTensor bridges, C-contiguous ranks 1..8.
from .view.adapters import try_as_static_c_tile_tensor_1d, try_as_mut_static_c_tile_tensor_1d, try_as_static_c_tile_tensor_3d, try_as_mut_static_c_tile_tensor_3d
from .view.adapters import try_as_static_c_tile_tensor_4d, try_as_mut_static_c_tile_tensor_4d, try_as_static_c_tile_tensor_5d, try_as_mut_static_c_tile_tensor_5d
from .view.adapters import try_as_static_c_tile_tensor_6d, try_as_mut_static_c_tile_tensor_6d, try_as_static_c_tile_tensor_7d, try_as_mut_static_c_tile_tensor_7d
from .view.adapters import try_as_static_c_tile_tensor_8d, try_as_mut_static_c_tile_tensor_8d

# AOT F-order rank-preserving zero-copy MAX adapters for ranks 3..8.
from .view.adapters import try_as_static_f_layout_tensor_3d, try_as_static_f_tile_tensor_3d, try_as_mut_static_f_layout_tensor_3d, try_as_mut_static_f_tile_tensor_3d
from .view.adapters import try_as_static_f_layout_tensor_4d, try_as_static_f_tile_tensor_4d, try_as_mut_static_f_layout_tensor_4d, try_as_mut_static_f_tile_tensor_4d
from .view.adapters import try_as_static_f_layout_tensor_5d, try_as_static_f_tile_tensor_5d, try_as_mut_static_f_layout_tensor_5d, try_as_mut_static_f_tile_tensor_5d
from .view.adapters import try_as_static_f_layout_tensor_6d, try_as_static_f_tile_tensor_6d, try_as_mut_static_f_layout_tensor_6d, try_as_mut_static_f_tile_tensor_6d
from .view.adapters import try_as_static_f_layout_tensor_7d, try_as_static_f_tile_tensor_7d, try_as_mut_static_f_layout_tensor_7d, try_as_mut_static_f_tile_tensor_7d
from .view.adapters import try_as_static_f_layout_tensor_8d, try_as_static_f_tile_tensor_8d, try_as_mut_static_f_layout_tensor_8d, try_as_mut_static_f_tile_tensor_8d

# AOT-checked zero-copy strided MAX bridges (rank 1/2, nonnegative strides).
from .view.adapters import (
    try_as_static_strided_layout_tensor_1d, try_as_mut_static_strided_layout_tensor_1d,
    try_as_static_strided_layout_tensor_2d, try_as_mut_static_strided_layout_tensor_2d,
    try_as_static_strided_tile_tensor_1d, try_as_mut_static_strided_tile_tensor_1d,
    try_as_static_strided_tile_tensor_2d, try_as_mut_static_strided_tile_tensor_2d,
)

# AOT-specialized zero-copy nonnegative strided MAX bridges, ranks 3..8.
from .view.adapters import try_as_static_strided_layout_tensor_3d, try_as_mut_static_strided_layout_tensor_3d, try_as_static_strided_tile_tensor_3d, try_as_mut_static_strided_tile_tensor_3d
from .view.adapters import try_as_static_strided_layout_tensor_4d, try_as_mut_static_strided_layout_tensor_4d, try_as_static_strided_tile_tensor_4d, try_as_mut_static_strided_tile_tensor_4d
from .view.adapters import try_as_static_strided_layout_tensor_5d, try_as_mut_static_strided_layout_tensor_5d, try_as_static_strided_tile_tensor_5d, try_as_mut_static_strided_tile_tensor_5d
from .view.adapters import try_as_static_strided_layout_tensor_6d, try_as_mut_static_strided_layout_tensor_6d, try_as_static_strided_tile_tensor_6d, try_as_mut_static_strided_tile_tensor_6d
from .view.adapters import try_as_static_strided_layout_tensor_7d, try_as_mut_static_strided_layout_tensor_7d, try_as_static_strided_tile_tensor_7d, try_as_mut_static_strided_tile_tensor_7d
from .view.adapters import try_as_static_strided_layout_tensor_8d, try_as_mut_static_strided_layout_tensor_8d, try_as_static_strided_tile_tensor_8d, try_as_mut_static_strided_tile_tensor_8d

from .indexing.index_tensor import IndexTensor
from .ops.reduction import argmin_axis, argmax_axis, argmin_axis_keepdims, argmax_axis_keepdims
from .ops.logical import logical_and, logical_or, logical_xor, logical_not, any_true, all_true, count_true

# CPU neural-network reference operators: Float32 tensors, ranks <= 8.
from .ops.nn import relu, gelu_tanh, softmax_last_axis, logsumexp_vector
from .ops.nn import layernorm_last_axis, scaled_dot_product_attention_2d
from .factory.random import random_uniform, random_normal
