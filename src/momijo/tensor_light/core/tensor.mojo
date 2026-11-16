# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR CORE / OWNING TENSOR
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Core Tensor Types
# Module:        momijo.tensor_light.core.tensor
# Source:        src/momijo/tensor_light/core/tensor.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines owning tensor and the related core tensor representation
#   contracts.
#
#   Public declarations include `Tensor`, `UninitializedHostTensor`,
#   `UninitializedDeviceTensor`; preconditions and supported targets are
#   determined by their implementations.
#
# Responsibilities:
#   - Maintain tensor representation and metadata invariants.
#   - Preserve documented behavior for shape, dtype, storage, and API
#     preconditions.
#   - Provide source-declared interfaces such as `Tensor`,
#     `UninitializedHostTensor`, `UninitializedDeviceTensor`.
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

# PRIMARY AND ONLY PUBLIC OWNING TENSOR TYPE. FP32 payload, rank 0..8.
# AOT: rank and storage capability are compile-time parameters; extents runtime.
from std.utils.numerics import isnan as fp32_isnan, isinf as fp32_isinf, isfinite as fp32_isfinite
from .tensor_mask import TensorMask
from ..shape.shape import TensorShape, flatten_shape, checked_trailing_reduction_geometry
from ..shape.shape_storage import StaticRankArray
from ..shape.axis import normalize_axis
from ..layout.layout import TensorLayout
from ..layout.contiguity import InternalOverlapKind
from ..alias.alias_info import _owner_bound_mapping_disjoint, _two_owner_mappings_disjoint
from ..alias.overlap import host_output_disjoint_from_input, host_output_disjoint_from_indices
from ..storage.storage import TensorStorage, MutableTensorStorage, HostSpanStorage
from ..internal.status import TensorResult, TensorStatus, device_write_failure_may_have_enqueued
from .tensor_impl import TensorImpl
from .tensor_descriptor import TensorDescriptor
from ..storage.host_storage import HostStorage, _fill_fp32_contiguous
from ..storage.external_buffer import ExternalFp32Deleter, ExternalOwnedStorage
from std.sys.info import simd_width_of
from std.math import fma as std_fma
from ..dtype.math_mode import Fp32MathMode, allows_fma
from ..memory.allocation import _allocate_host_fp32_uninitialized
from std.collections import Span
from std.memory import unsafe_memcpy, forget_deinit
from ..memory.memory_pool import HostPool
from ..view.view import TensorView, TensorMutView
from layout import Layout, LayoutTensor, IntTuple
from layout.tile_tensor import TileTensor, LTToTTLayout
from ..view.adapters import (
    try_as_static_strided_layout_tensor_1d, try_as_mut_static_strided_layout_tensor_1d,
    try_as_static_strided_layout_tensor_2d, try_as_mut_static_strided_layout_tensor_2d,
    try_as_static_strided_tile_tensor_1d, try_as_mut_static_strided_tile_tensor_1d,
    try_as_static_strided_tile_tensor_2d, try_as_mut_static_strided_tile_tensor_2d,
)
from ..view.adapters import (
    try_as_static_flat_c_tile_tensor, try_as_mut_static_flat_c_tile_tensor,
    try_as_static_c_tile_tensor_2d, try_as_mut_static_c_tile_tensor_2d,
    try_as_static_f_tile_tensor_2d, try_as_mut_static_f_tile_tensor_2d,
)
from ..view.adapters import (
    try_as_static_c_tile_tensor_1d,
    try_as_mut_static_c_tile_tensor_1d,
    try_as_static_c_tile_tensor_3d,
    try_as_mut_static_c_tile_tensor_3d,
    try_as_static_c_tile_tensor_4d,
    try_as_mut_static_c_tile_tensor_4d,
    try_as_static_c_tile_tensor_5d,
    try_as_mut_static_c_tile_tensor_5d,
    try_as_static_c_tile_tensor_6d,
    try_as_mut_static_c_tile_tensor_6d,
    try_as_static_c_tile_tensor_7d,
    try_as_mut_static_c_tile_tensor_7d,
    try_as_static_c_tile_tensor_8d,
    try_as_mut_static_c_tile_tensor_8d
)
from ..view.adapters import (
    try_as_static_f_tile_tensor_3d,
    try_as_mut_static_f_tile_tensor_3d,
    try_as_static_f_tile_tensor_4d,
    try_as_mut_static_f_tile_tensor_4d,
    try_as_static_f_tile_tensor_5d,
    try_as_mut_static_f_tile_tensor_5d,
    try_as_static_f_tile_tensor_6d,
    try_as_mut_static_f_tile_tensor_6d,
    try_as_static_f_tile_tensor_7d,
    try_as_mut_static_f_tile_tensor_7d,
    try_as_static_f_tile_tensor_8d,
    try_as_mut_static_f_tile_tensor_8d
)
from .tensor_ref import TensorRef, TensorMutRef
from ..view.reshape import reshape_c_view, reshape_f_view
from ..view.unsqueeze import unsqueeze as unsqueeze_layout, squeeze_axis as squeeze_axis_layout
from ..view.expand import broadcast_layout, squeeze_axes as squeeze_axes_layout
from ..view.transpose import movedim as movedim_layout
from ..internal.checked_math import checked_add, checked_mul_nonnegative
from ..iterator.nd_iterator import NDIndexIterator
from ..view.as_strided import slice_axis as slice_layout_axis
from ..indexing.slice import SliceSpec
from ..indexing.index import normalize_index
from ..indexing.slice_spec import BasicIndex, NewAxis, EllipsisIndex, as_basic_index, inferred_basic_index_rank
from ..layout.memory_format import ContiguousOrder
from ..layout.layout_analysis import is_reverse_c_contiguous, is_reverse_f_contiguous
from ..kernels.cpu.reduction import cpu_sum_contiguous, cpu_sum_contiguous_parallel_fast, cpu_sum_strided_parallel_fast, cpu_sum_c_strict_fp32, cpu_prod_c_fp32, cpu_extremum_c_fp32, cpu_argextremum_c_fp32, cpu_reduce_strided_fp32, cpu_reduce_c_trailing_axes_fp32, cpu_reduce_c_axis_block_fp32, cpu_scan_strided_fp32
from ..kernels.cpu.transform import cpu_transpose_copy2_strided
from ..kernels.cpu.indexing import cpu_gather_kernel, cpu_scatter_kernel
from ..kernels.cpu.matrix import (
    cpu_dot_contiguous, cpu_mm_contiguous, cpu_mm_packed_fp32,
    cpu_gemv_contiguous, cpu_vecmat_contiguous, cpu_outer_contiguous,
    cpu_mm_autoselect_fp32, cpu_mm_packed_workspace_fp32, cpu_dot_checked_fp32,
    cpu_gemv_checked_fp32, cpu_vecmat_checked_fp32, cpu_outer_checked_fp32,
)
from ..kernels.cpu.unary import cpu_unary_scalar_fp32, cpu_unary_contiguous_fp32
from ..storage.device_storage import DeviceStorage, _retain_context_on_unproven_gpu_completion
from ..dtype.fp32 import fp32_minimum, fp32_maximum, fp32_argmin_prefers, fp32_argmax_prefers
from max.gpu.host import DeviceExternalFunction
from ..dispatch.specialization import BinaryAOTArtifactIdentity, GemmAOTArtifactIdentity, UnaryAOTArtifactIdentity, ReduceRowsAOTArtifactIdentity, ScanRowsAOTArtifactIdentity
from ..execution.execution_plan import ExecutionPlan, ExecutionPlacement
from ..execution.launch_policy import HybridGemmTuningProfile, HybridRowReductionTuningProfile, choose_host_gemm_rows_split_plan_for_capacity, choose_host_reduce_rows_split_plan_for_capacity
from ..execution.launch_policy import HybridTuningProfile, choose_host_unary_plan_for_capacity
from ..execution.static_dataflow import SealedStaticDataflowPlan
from ..execution.aot_release_plan import BoundAOTDataflowPlan
from ..execution.cpu_aot_executor import _run_cpu_static_dataflow_in_arena
from ..backend.hybrid_reduction import unsafe_execute_hybrid_reduce_rows_fp32
from ..backend.hybrid_matrix import unsafe_execute_hybrid_gemm_rows_fp32
from ..backend.hybrid import unsafe_execute_host_unary_fp32, execute_host_binary_fp32
from ..kernels.launch import run_owned_bound_gpu_binary_fp32, run_owned_bound_gpu_gemm_fp32, run_owned_bound_gpu_unary_fp32, run_owned_bound_gpu_sum_rows_fp32, run_owned_bound_gpu_reduce_rows_fp32, run_owned_bound_gpu_scan_rows_fp32, run_owned_gpu_binary_strided_fp32, run_owned_gpu_unary_strided_fp32, run_owned_gpu_reduce_axis_strided_fp32, run_owned_gpu_scan_axis_strided_fp32, run_owned_gpu_reduce_axes_strided_fp32
from max.gpu.host import DeviceContext
from .tensor_options import TensorOptions, validate_tensor_options
from ..memory.memory_space import MemorySpace
from ..kernels.cpu.elementwise import (
    cpu_add_contiguous, cpu_sub_contiguous, cpu_mul_contiguous,
    cpu_div_contiguous, cpu_maximum_contiguous, cpu_minimum_contiguous,
    cpu_shift_contiguous, cpu_scale_contiguous,
    cpu_sub_scalar_contiguous, cpu_div_scalar_contiguous,
    cpu_fma_contiguous, cpu_scalar_broadcast_binary_contiguous, cpu_binary_contiguous_checked,
)
from ..indexing.index_math import basic_index_layout
from ..transfer.device_transfer import copy_d2d_fp32, device_fp32_regions_disjoint
from ..access.span import copy_to_span as access_copy_to_span, copy_from_span as access_copy_from_span


# Complete fixed-rank (3..8) specialization surface for statically strided MAX views.
from ..view.adapters import try_as_static_flat_f_layout_tensor, try_as_mut_static_flat_f_layout_tensor
from ..view.adapters import try_as_static_flat_f_tile_tensor, try_as_mut_static_flat_f_tile_tensor
from ..view.adapters import try_as_static_strided_layout_tensor_3d, try_as_mut_static_strided_layout_tensor_3d, try_as_static_strided_tile_tensor_3d, try_as_mut_static_strided_tile_tensor_3d
from ..view.adapters import try_as_static_strided_layout_tensor_4d, try_as_mut_static_strided_layout_tensor_4d, try_as_static_strided_tile_tensor_4d, try_as_mut_static_strided_tile_tensor_4d
from ..view.adapters import try_as_static_strided_layout_tensor_5d, try_as_mut_static_strided_layout_tensor_5d, try_as_static_strided_tile_tensor_5d, try_as_mut_static_strided_tile_tensor_5d
from ..view.adapters import try_as_static_strided_layout_tensor_6d, try_as_mut_static_strided_layout_tensor_6d, try_as_static_strided_tile_tensor_6d, try_as_mut_static_strided_tile_tensor_6d
from ..view.adapters import try_as_static_strided_layout_tensor_7d, try_as_mut_static_strided_layout_tensor_7d, try_as_static_strided_tile_tensor_7d, try_as_mut_static_strided_tile_tensor_7d
from ..view.adapters import try_as_static_strided_layout_tensor_8d, try_as_mut_static_strided_layout_tensor_8d, try_as_static_strided_tile_tensor_8d, try_as_mut_static_strided_tile_tensor_8d

struct Tensor[rank: Int, S: TensorStorage = HostStorage](Movable):
    var impl: TensorImpl[Self.rank, Self.S]

    def __init__(out self, var impl: TensorImpl[Self.rank, Self.S]):
        comptime assert Self.rank >= 0 and Self.rank <= 8, "Self.rank must be 0..8"
        self.impl = impl

    @staticmethod
    def _from_parts_unchecked(
        var owned: Self.S, layout: TensorLayout[Self.rank]
    ) -> Self:
        # Internal zero-check assembly. Requires a checked rank, shape,
        # span and allocation-capacity proof from the caller. Never expose
        # this unchecked constructor as a public untrusted-buffer factory.
        return Self(TensorImpl[Self.rank, Self.S](owned, layout))

    @staticmethod
    def from_parts_checked(var owned: Self.S, layout: TensorLayout[Self.rank]) -> TensorResult[Self]:
        # Revalidate untrusted structural metadata at the public ownership boundary.
        # Never trust cached span or overlap flags constructed outside checked APIs.
        var true_shape = TensorShape[Self.rank].checked(layout.shape.dims)
        if not true_shape.is_ok():
            return TensorResult[Self](true_shape.status)
        if true_shape.payload.value().numel() != layout.shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        if not owned.is_initialized():
            return TensorResult[Self](TensorStatus.UNINITIALIZED)
        var validated = TensorLayout[Self.rank].checked(
            true_shape.payload.value(), layout.strides, layout.offset, owned.capacity()
        )
        if not validated.is_ok():
            return TensorResult[Self](validated.status)
        return TensorResult[Self](Self._from_parts_unchecked(owned, validated.payload.value()))

    @staticmethod
    def unsafe_adopt_external_checked[D: ExternalFp32Deleter](
        address: Int, allocation_elements: Int, alignment: Int,
        var deleter: D, layout: TensorLayout[Self.rank],
    ) -> TensorResult[Self] where identical(Self.S, ExternalOwnedStorage[D]):
        # Complete ALL fallible shape/layout checks before adopting the
        # foreign allocation. Otherwise a failed from_parts_checked would
        # destroy the temporary storage and free the caller's buffer despite
        # returning an error, risking an accidental second free by the caller.
        # No hidden copy, alias promotion or migration occurs.
        var canonical_shape = TensorShape[Self.rank].checked(layout.shape.dims)
        if not canonical_shape.is_ok():
            return TensorResult[Self](canonical_shape.status)
        if canonical_shape.payload.value().numel() != layout.shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var valid_layout = TensorLayout[Self.rank].checked(
            canonical_shape.payload.value(), layout.strides,
            layout.offset, allocation_elements
        )
        if not valid_layout.is_ok():
            return TensorResult[Self](valid_layout.status)
        # The foreign owner is constructed only after layout validation.
        # If pointer/alignment metadata is invalid, adoption returns an
        # error before constructing an owned-storage object.
        var adopted = ExternalOwnedStorage[D].unsafe_adopt_checked(
            address, allocation_elements, alignment, deleter
        )
        if not adopted.is_ok():
            return TensorResult[Self](adopted.status)
        # Every remaining operation is an infallible checked-parts assembly.
        # The destructor invokes the exact supplied deleter after success.
        return TensorResult[Self](
            Self._from_parts_unchecked(
                adopted.payload.take(), valid_layout.payload.value()
            )
        )

    @staticmethod
    def full_checked(shape: TensorShape[Self.rank], value: Float32) -> TensorResult[Self] where identical(Self.S, HostStorage):
        # The primary Tensor type is also the public owning factory. Validate
        # before allocation, then build one canonical C-order layout.
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        if canonical.payload.value().numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var trusted = canonical.payload.value()
        var layout = TensorLayout[Self.rank].c_contiguous(trusted, trusted.numel())
        if not layout.is_ok():
            return TensorResult[Self](layout.status)
        var owned = HostStorage.full(trusted.numel(), value)
        if not owned.is_ok():
            return TensorResult[Self](owned.status)
        return TensorResult[Self](Self._from_parts_unchecked(owned.payload.take(), layout.payload.value()))

    @staticmethod
    def full_from_pool_checked(
        mut pool: HostPool, shape: TensorShape[Self.rank], value: Float32
    ) -> TensorResult[Self] where identical(Self.S, HostStorage):
        # Canonical validation precedes leasing; one writer initializes every
        # element before the owning Tensor is exposed to the caller.
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        if canonical.payload.value().numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var trusted = canonical.payload.value()
        var layout = TensorLayout[Self.rank].c_contiguous(trusted, trusted.numel())
        if not layout.is_ok():
            return TensorResult[Self](layout.status)
        var leased = pool.acquire_filled(trusted.numel(), value)
        if not leased.is_ok():
            return TensorResult[Self](leased.status)
        var owned = HostStorage(leased.payload.take())
        return TensorResult[Self](Self._from_parts_unchecked(owned, layout.payload.value()))

    @staticmethod
    def full_from_pool(mut pool: HostPool, shape: TensorShape[Self.rank], value: Float32) -> TensorResult[Self] where identical(Self.S, HostStorage):
        return Self.full_from_pool_checked(pool, shape, value)

    def recycle_into_host_pool(deinit self, mut pool: HostPool) -> TensorStatus where identical(Self.S, HostStorage):
        # A pooled Tensor must have a symmetric, explicit return path; simply
        # dropping an owned Tensor frees its allocation instead of caching it.
        # Consuming self proves that no safe Tensor/View borrow survives. The
        # pool validates the physical owner and takes its allocation exactly
        # once, so the following full_from_pool() can reuse it without a malloc.
        # Caller-issued unsafe asynchronous borrows must already be complete.
        return pool.release(self.impl.storage.owned)

    @staticmethod
    def from_copy_span_checked[origin: Origin](
        source: Span[False, Float32, origin], shape: TensorShape[Self.rank]
    ) -> TensorResult[Self] where identical(Self.S, HostStorage):
        # Validate before allocation; the destination is a different owning
        # allocation and every accessible Float32 is initialized by the copy.
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        var trusted = canonical.payload.value()
        if trusted.numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        if len(source) != trusted.numel():
            return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
        var layout = TensorLayout[Self.rank].c_contiguous(trusted, trusted.numel())
        if not layout.is_ok():
            return TensorResult[Self](layout.status)
        var allocation = _allocate_host_fp32_uninitialized(trusted.numel())
        if not allocation.is_ok():
            return TensorResult[Self](allocation.status)
        var owned = HostStorage(allocation.payload.take())
        # Fresh allocation proves the source/destination are disjoint.
        var copied = owned._copy_from_disjoint_span_unchecked(source)
        if copied != TensorStatus.OK:
            return TensorResult[Self](copied)
        return TensorResult[Self](Self._from_parts_unchecked(owned, layout.payload.value()))

    @staticmethod
    def from_copy_span[origin: Origin](
        source: Span[False, Float32, origin], shape: TensorShape[Self.rank]
    ) -> TensorResult[Self] where identical(Self.S, HostStorage):
        return Self.from_copy_span_checked(source, shape)

    @staticmethod
    def _empty_for_complete_overwrite_checked(
        shape: TensorShape[Self.rank]
    ) -> TensorResult[Self] where identical(Self.S, HostStorage):
        # INTERNAL-ONLY: caller must initialize EVERY logical element before
        # returning this Tensor. Error paths consume and destroy the owner.
        # Only internal kernels with a proven complete-write path may use
        # this readable owner constructor. Public allocation-only APIs always
        # return unreadable initialization builders instead.
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        if canonical.payload.value().numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var trusted = canonical.payload.value()
        var layout = TensorLayout[Self.rank].c_contiguous(trusted, trusted.numel())
        if not layout.is_ok():
            return TensorResult[Self](layout.status)
        var owned = HostStorage._allocate_for_complete_overwrite(trusted.numel())
        if not owned.is_ok():
            return TensorResult[Self](owned.status)
        return TensorResult[Self](Self._from_parts_unchecked(owned.payload.take(), layout.payload.value()))

    @staticmethod
    def unsafe_empty_uninitialized_checked(
        shape: TensorShape[Self.rank]
    ) -> TensorResult[UninitializedHostTensor[Self.rank]] where identical(Self.S, HostStorage):
        # The historical expert spelling must not expose a readable Tensor
        # backed by uninitialized storage. This is an allocation-only builder;
        # initialize_fill() or initialize_from_physical_span() consumes it and
        # yields the first readable Tensor without a redundant zero-fill pass.
        return Self.empty_checked(shape)

    @staticmethod
    def empty_checked(
        shape: TensorShape[Self.rank]
    ) -> TensorResult[UninitializedHostTensor[Self.rank]] where identical(Self.S, HostStorage):
        # A fresh uninitialized owner must NOT be returned as a readable
        # Tensor. Explicit full initialization consumes this builder.
        # This preserves the O(1) allocation-only path for speed-critical code.
        return UninitializedHostTensor[Self.rank].allocate_checked(shape, ContiguousOrder.C)

    @staticmethod
    def zeros_checked(shape: TensorShape[Self.rank]) -> TensorResult[Self] where identical(Self.S, HostStorage):
        return Self.full_checked(shape, 0.0)

    @staticmethod
    def ones_checked(shape: TensorShape[Self.rank]) -> TensorResult[Self] where identical(Self.S, HostStorage):
        return Self.full_checked(shape, 1.0)

    @staticmethod
    def scalar_checked(value: Float32) -> TensorResult[Self] where identical(Self.S, HostStorage) and Self.rank == 0:
        var shape = TensorShape[0].checked(StaticRankArray[0](0))
        if not shape.is_ok():
            return TensorResult[Self](shape.status)
        return Self.full_checked(shape.payload.value(), value)

    # Order-aware constructors keep fixed-rank shape and FP32 ownership.
    # C and F are physical layout guarantees; arbitrary strides require views.
    @staticmethod
    def full_with_order_checked(
        shape: TensorShape[Self.rank], value: Float32, order: ContiguousOrder
    ) -> TensorResult[Self] where identical(Self.S, HostStorage):
        if order == ContiguousOrder.C:
            return Self.full_checked(shape, value)
        if order != ContiguousOrder.F:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        var trusted = canonical.payload.value()
        if trusted.numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var mapping = TensorLayout[Self.rank].f_contiguous(trusted, trusted.numel())
        if not mapping.is_ok():
            return TensorResult[Self](mapping.status)
        var storage = HostStorage.full(trusted.numel(), value)
        if not storage.is_ok():
            return TensorResult[Self](storage.status)
        return TensorResult[Self](Self._from_parts_unchecked(storage.payload.take(), mapping.payload.value()))

    @staticmethod
    def unsafe_empty_with_order_uninitialized_checked(
        shape: TensorShape[Self.rank], order: ContiguousOrder
    ) -> TensorResult[UninitializedHostTensor[Self.rank]] where identical(Self.S, HostStorage):
        # C/F layout selection happens once, before the allocation. Neither
        # order creates an uninitialized readable Tensor or duplicates the
        # underlying fast allocation path.
        return Self.empty_with_order_checked(shape, order)

    @staticmethod
    def empty_with_order_checked(
        shape: TensorShape[Self.rank], order: ContiguousOrder
    ) -> TensorResult[UninitializedHostTensor[Self.rank]] where identical(Self.S, HostStorage):
        # The returned uninitialized owner cannot be used as a readable
        # Tensor until initialize_fill()/initialize_from_physical_span().
        return UninitializedHostTensor[Self.rank].allocate_checked(shape, order)

    @staticmethod
    def zeros_with_order_checked(
        shape: TensorShape[Self.rank], order: ContiguousOrder
    ) -> TensorResult[Self] where identical(Self.S, HostStorage):
        return Self.full_with_order_checked(shape, 0.0, order)

    @staticmethod
    def ones_with_order_checked(
        shape: TensorShape[Self.rank], order: ContiguousOrder
    ) -> TensorResult[Self] where identical(Self.S, HostStorage):
        return Self.full_with_order_checked(shape, 1.0, order)

    @staticmethod
    def _validate_host_options(options: TensorOptions) -> TensorStatus:
        var checked = validate_tensor_options(options)
        if checked != TensorStatus.OK:
            return checked
        # HostStorage is currently a 64-byte-aligned pageable allocation.
        # Never pretend it satisfies pinned/device or >64-byte alignment.
        if options.device_ordinal != -1:
            return TensorStatus.DEVICE_MISMATCH
        if options.memory_space != MemorySpace.HOST_PAGEABLE:
            return TensorStatus.UNSUPPORTED
        if options.requested_alignment > 64:
            return TensorStatus.INVALID_ALIGNMENT
        return TensorStatus.OK

    @staticmethod
    def full_with_options_checked(
        shape: TensorShape[Self.rank], value: Float32, options: TensorOptions
    ) -> TensorResult[Self] where identical(Self.S, HostStorage):
        var checked = Self._validate_host_options(options)
        if checked != TensorStatus.OK:
            return TensorResult[Self](checked)
        if options.f_order:
            return Self.full_with_order_checked(shape, value, ContiguousOrder.F)
        return Self.full_checked(shape, value)

    @staticmethod
    def empty_with_options_checked(
        shape: TensorShape[Self.rank], options: TensorOptions
    ) -> TensorResult[UninitializedHostTensor[Self.rank]] where identical(Self.S, HostStorage):
        var checked = Self._validate_host_options(options)
        if checked != TensorStatus.OK:
            return TensorResult[UninitializedHostTensor[Self.rank]](checked)
        # Initialization-to-zero returns a readable Tensor and therefore
        # belongs to zeros_with_options_checked(), not the uninit builder API.
        # Reject contradictory options rather than returning the wrong type.
        if options.initialize_to_zero:
            return TensorResult[UninitializedHostTensor[Self.rank]](TensorStatus.UNSUPPORTED)
        if options.f_order:
            return Self.empty_with_order_checked(shape, ContiguousOrder.F)
        return Self.empty_checked(shape)

    @staticmethod
    def unsafe_empty_with_options_uninitialized_checked(
        shape: TensorShape[Self.rank], options: TensorOptions
    ) -> TensorResult[UninitializedHostTensor[Self.rank]] where identical(Self.S, HostStorage):
        # The historical unsafe alias now has the same non-readable typestate
        # as every other allocation-only constructor. Initialization-to-zero
        # is deliberately handled by zeros_with_options_checked() instead.
        return Self.empty_with_options_checked(shape, options)

    @staticmethod
    def zeros_with_options_checked(
        shape: TensorShape[Self.rank], options: TensorOptions
    ) -> TensorResult[Self] where identical(Self.S, HostStorage):
        return Self.full_with_options_checked(shape, 0.0, options)

    @staticmethod
    def ones_with_options_checked(
        shape: TensorShape[Self.rank], options: TensorOptions
    ) -> TensorResult[Self] where identical(Self.S, HostStorage):
        return Self.full_with_options_checked(shape, 1.0, options)

    # Tensor is the canonical public factory; the *_checked spellings are
    # retained for existing callers. These thin aliases never allocate
    # metadata or add a second Tensor ownership abstraction.
    @staticmethod
    def empty(
        shape: TensorShape[Self.rank]
    ) -> TensorResult[UninitializedHostTensor[Self.rank]] where identical(Self.S, HostStorage):
        # The canonical Tensor remains the ONLY readable owning tensor type.
        # Empty storage uses a restricted initialization builder instead.
        return Self.empty_checked(shape)

    @staticmethod
    def full(shape: TensorShape[Self.rank], value: Float32) -> TensorResult[Self] where identical(Self.S, HostStorage):
        return Self.full_checked(shape, value)

    @staticmethod
    def zeros(shape: TensorShape[Self.rank]) -> TensorResult[Self] where identical(Self.S, HostStorage):
        return Self.zeros_checked(shape)

    @staticmethod
    def ones(shape: TensorShape[Self.rank]) -> TensorResult[Self] where identical(Self.S, HostStorage):
        return Self.ones_checked(shape)

    @staticmethod
    def scalar(value: Float32) -> TensorResult[Self] where identical(Self.S, HostStorage) and Self.rank == 0:
        return Self.scalar_checked(value)

    # Tensor-first factories. Each returns a fresh FP32 owning allocation;
    # unique F-order inputs retain F order, other layouts use explicit C order.
    def empty_like(
        ref self
    ) -> TensorResult[UninitializedHostTensor[Self.rank]] where identical(Self.S, HostStorage):
        # No payload may be read before explicit complete initialization.
        # The source contributes only shape/order metadata, never values.
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[UninitializedHostTensor[Self.rank]](checked.status)
        var source_layout = checked.payload.value()
        if source_layout.is_f_contiguous() and not source_layout.is_c_contiguous():
            return Tensor[Self.rank, HostStorage].empty_with_order_checked(
                source_layout.shape, ContiguousOrder.F
            )
        return Tensor[Self.rank, HostStorage].empty_checked(source_layout.shape)

    def zeros_like(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](checked.status)
        var source_layout = checked.payload.value()
        if source_layout.is_f_contiguous() and not source_layout.is_c_contiguous():
            return Tensor[Self.rank, HostStorage].zeros_with_order_checked(
                source_layout.shape, ContiguousOrder.F
            )
        return Tensor[Self.rank, HostStorage].zeros_checked(source_layout.shape)

    def ones_like(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](checked.status)
        var source_layout = checked.payload.value()
        if source_layout.is_f_contiguous() and not source_layout.is_c_contiguous():
            return Tensor[Self.rank, HostStorage].ones_with_order_checked(
                source_layout.shape, ContiguousOrder.F
            )
        return Tensor[Self.rank, HostStorage].ones_checked(source_layout.shape)

    def full_like(ref self, value: Float32) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](checked.status)
        var source_layout = checked.payload.value()
        if source_layout.is_f_contiguous() and not source_layout.is_c_contiguous():
            return Tensor[Self.rank, HostStorage].full_with_order_checked(
                source_layout.shape, value, ContiguousOrder.F
            )
        return Tensor[Self.rank, HostStorage].full_checked(source_layout.shape, value)

    # Explicit GPU-native Float32 construction. The sole public owner is
    # Tensor[rank, DeviceStorage]; no intermediate CPU Tensor or host transfer
    # is introduced. MAX fills the newly allocated device buffer in place.
    @staticmethod
    def unsafe_empty_device_uninitialized_checked(
        ctx: DeviceContext, shape: TensorShape[Self.rank]
    ) -> TensorResult[UninitializedDeviceTensor[Self.rank]] where identical(Self.S, DeviceStorage):
        # Never project an uninitialized GPU buffer into readable Tensor
        # ownership. This compatibility alias preserves explicit event-joined
        # allocation and the guarded initialize_fill() transition.
        return Self.empty_device_checked(ctx, shape)

    @staticmethod
    def empty_device_checked(
        ctx: DeviceContext, shape: TensorShape[Self.rank]
    ) -> TensorResult[UninitializedDeviceTensor[Self.rank]] where identical(Self.S, DeviceStorage):
        # No readable device Tensor exists until a complete device fill has
        # completed and its completion event has been joined.
        return UninitializedDeviceTensor[Self.rank].allocate_checked(ctx, shape)

    @staticmethod
    def empty_device(
        ctx: DeviceContext, shape: TensorShape[Self.rank]
    ) -> TensorResult[UninitializedDeviceTensor[Self.rank]] where identical(Self.S, DeviceStorage):
        return Self.empty_device_checked(ctx, shape)

    @staticmethod
    def full_device_checked(
        ctx: DeviceContext, shape: TensorShape[Self.rank], value: Float32
    ) -> TensorResult[Self] where identical(Self.S, DeviceStorage):
        # Canonical checked FP32 allocation: allocation and fill share one
        # stream and one joined Event. Unlike allocating an empty builder
        # and filling later, there is no redundant synchronization barrier.
        # On unrecoverable failure, DeviceStorage quarantines the allocation.
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        var trusted = canonical.payload.value()
        if trusted.numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var layout = TensorLayout[Self.rank].c_contiguous(trusted, trusted.numel())
        if not layout.is_ok():
            return TensorResult[Self](layout.status)
        var filled = DeviceStorage._allocate_owned_checked(
            ctx, trusted.numel(), True, value,
        )
        if not filled.is_ok():
            return TensorResult[Self](filled.status)
        return Self.from_parts_checked(filled.payload.take(), layout.payload.value())

    @staticmethod
    def zeros_device(
        ctx: DeviceContext, shape: TensorShape[Self.rank]
    ) -> TensorResult[Self] where identical(Self.S, DeviceStorage):
        return Self.full_device_checked(ctx, shape, 0.0)

    @staticmethod
    def ones_device(
        ctx: DeviceContext, shape: TensorShape[Self.rank]
    ) -> TensorResult[Self] where identical(Self.S, DeviceStorage):
        return Self.full_device_checked(ctx, shape, 1.0)

    @staticmethod
    def full_device(
        ctx: DeviceContext, shape: TensorShape[Self.rank], value: Float32
    ) -> TensorResult[Self] where identical(Self.S, DeviceStorage):
        return Self.full_device_checked(ctx, shape, value)

    # Explicit H2D ownership transition. No implicit fallback or migration.
    # The borrowed input remains alive throughout host staging, transfer and
    # its stream-scoped completion. MAX handles only Float32 payload data;
    # any noncontiguous host view is packed in logical C order explicitly.
    # The source is host RAM, not an in-flight DMA allocation; failed GPU
    # copies retain their OWN staging/device buffers and DeviceContext.
    def to_device(
        ref self, ctx: DeviceContext
    ) -> TensorResult[Tensor[Self.rank, DeviceStorage]] where identical(Self.S, HostStorage):
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](proof.status)
        var source_layout = proof.payload.value()
        var count = source_layout.shape.numel()
        var device_layout = TensorLayout[Self.rank].c_contiguous(source_layout.shape, count)
        if not device_layout.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](device_layout.status)
        try:
            # This owner must be retained after an unsuccessful asynchronous
            # enqueue until either an Event or the stream proves quiescence.
            var device_buffer = ctx.enqueue_create_buffer[DType.float32](max(count, 1))
            if count > 0:
                try:
                    var staging = ctx.enqueue_create_host_buffer[DType.float32](count)
                    try:
                        # HostBuffer allocation is asynchronous; do not write
                        # pinned memory before its allocation Event completes.
                        var allocation_event = ctx.create_event()
                        ctx.stream().record_event(allocation_event)
                        allocation_event.synchronize()
                        var source = self.impl.storage.borrow_span()
                        if source_layout.is_c_contiguous() and source_layout.is_exhaustive():
                            var first = source_layout.span_lo
                            var source_run = source[first:first + count]
                            unsafe_memcpy(
                                dest=staging.unsafe_ptr(), src=source_run.unsafe_ptr(),
                                count=count,
                            )
                        else:
                            var cursor = NDIndexIterator[Self.rank](source_layout.shape)
                            var linear = 0
                            while not cursor.done():
                                staging[linear] = source[
                                    source_layout.physical_offset_unchecked(cursor.coordinates)
                                ]
                                linear += 1
                                cursor.advance()
                        ctx.enqueue_copy(dst_buf=device_buffer, src_buf=staging)
                        var copied = ctx.create_event()
                        ctx.stream().record_event(copied)
                        copied.synchronize()
                    except:
                        try:
                            ctx.stream().synchronize()
                        except:
                            # A DMA request may still reference BOTH owners.
                            # Freeing either on this fatal path would be a UAF.
                            forget_deinit(staging)
                            forget_deinit(device_buffer)
                            _retain_context_on_unproven_gpu_completion(ctx)
                            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                                TensorStatus.GPU_QUIESCENCE_UNPROVEN
                            )
                        # The stream join made DMA safe, but buffer drops
                        # enqueue later native frees; drain before the caller
                        # can destroy its last allocating-context reference.
                        _ = staging
                        _ = device_buffer
                        try:
                            ctx.stream().synchronize()
                        except:
                            _retain_context_on_unproven_gpu_completion(ctx)
                            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                                TensorStatus.GPU_QUIESCENCE_UNPROVEN
                            )
                        return TensorResult[Tensor[Self.rank, DeviceStorage]](
                            TensorStatus.TRANSFER_UNAVAILABLE
                        )
                except:
                    # The pinned allocation itself may have failed after the
                    # device allocation was queued. Quiesce before releasing
                    # the already-created device owner.
                    try:
                        ctx.stream().synchronize()
                    except:
                        forget_deinit(device_buffer)
                        _retain_context_on_unproven_gpu_completion(ctx)
                        return TensorResult[Tensor[Self.rank, DeviceStorage]](
                            TensorStatus.GPU_QUIESCENCE_UNPROVEN
                        )
                    # The device allocation may be released asynchronously.
                    _ = device_buffer
                    try:
                        ctx.stream().synchronize()
                    except:
                        _retain_context_on_unproven_gpu_completion(ctx)
                        return TensorResult[Tensor[Self.rank, DeviceStorage]](
                            TensorStatus.GPU_QUIESCENCE_UNPROVEN
                        )
                    return TensorResult[Tensor[Self.rank, DeviceStorage]](
                        TensorStatus.ALLOCATION_FAILED
                    )
            else:
                try:
                    # Empty tensors have a physical padding element, but no
                    # logical payload. Join only the pending allocation.
                    var allocated = ctx.create_event()
                    ctx.stream().record_event(allocated)
                    allocated.synchronize()
                except:
                    try:
                        ctx.stream().synchronize()
                    except:
                        forget_deinit(device_buffer)
                        _retain_context_on_unproven_gpu_completion(ctx)
                        return TensorResult[Tensor[Self.rank, DeviceStorage]](
                            TensorStatus.GPU_QUIESCENCE_UNPROVEN
                        )
                    # Even an empty device Tensor owns a physical buffer.
                    _ = device_buffer
                    try:
                        ctx.stream().synchronize()
                    except:
                        _retain_context_on_unproven_gpu_completion(ctx)
                        return TensorResult[Tensor[Self.rank, DeviceStorage]](
                            TensorStatus.GPU_QUIESCENCE_UNPROVEN
                        )
                    return TensorResult[Tensor[Self.rank, DeviceStorage]](
                        TensorStatus.TRANSFER_UNAVAILABLE
                    )
            # Both allocation and transfer (when nonempty) have completed.
            # Ownership may now move into the readable Tensor exactly once.
            var storage = DeviceStorage._unsafe_from_completed_buffer_checked(device_buffer, count, ctx)
            if not storage.is_ok():
                return TensorResult[Tensor[Self.rank, DeviceStorage]](storage.status)
            var initialized_owner = storage.payload.take()
            var sealed = initialized_owner._mark_complete_overwrite()
            if sealed != TensorStatus.OK:
                return TensorResult[Tensor[Self.rank, DeviceStorage]](sealed)
            return Tensor[Self.rank, DeviceStorage].from_parts_checked(
                initialized_owner, device_layout.payload.value()
            )
        except:
            # No completed allocation owner was returned by the outer MAX
            # allocation call. Pending context work is handled by MAX;
            # locally created owners are covered by the nested handlers.
            try:
                ctx.stream().synchronize()
            except:
                # The backend may still own an in-flight allocation even
                # though enqueue_create_buffer returned no buffer object.
                _retain_context_on_unproven_gpu_completion(ctx)
                return TensorResult[Tensor[Self.rank, DeviceStorage]](
                    TensorStatus.GPU_QUIESCENCE_UNPROVEN
                )
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.ALLOCATION_FAILED
            )

    # Explicit D2H materialization. Device tensors cannot be dereferenced by
    # the CPU: readback is a narrow event-ordered copy before owned output is
    # exposed. Arbitrary device views do not silently pack or migrate.
    # BORROWED NATIVE INPUT: on GPU_QUIESCENCE_UNPROVEN the caller MUST keep
    # this Tensor alive until external device reset/recovery. The staging
    # allocation and context are retained locally; neither retains self.
    # to_host_consuming() is the owner-safe failure alternative.
    def to_host(
        ref self, ctx: DeviceContext
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, DeviceStorage):
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](proof.status)
        var source_layout = proof.payload.value()
        if not source_layout.is_c_contiguous() or not source_layout.is_exhaustive():
            return TensorResult[Tensor[Self.rank, HostStorage]](
                TensorStatus.NOT_REPRESENTABLE
            )
        if source_layout.span_lo != 0:
            return TensorResult[Tensor[Self.rank, HostStorage]](
                TensorStatus.NOT_REPRESENTABLE
            )
        # Reading device memory is legal only after the source owner's
        # original native queue has completed. An unrelated caller context
        # cannot validate this allocation or its producer stream.
        if (not self.impl.storage.lifetime_context
            or self.impl.storage.quiescence_unproven
            or self.impl.storage.quarantined_context):
            return TensorResult[Tensor[Self.rank, HostStorage]](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        var issuing = self.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorResult[Tensor[Self.rank, HostStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var count = source_layout.shape.numel()
        if count > 0 and len(self.impl.storage.buffer) != count:
            return TensorResult[Tensor[Self.rank, HostStorage]](
                TensorStatus.NOT_REPRESENTABLE
            )
        var prepared = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(
            source_layout.shape
        )
        if not prepared.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](prepared.status)
        var result = prepared.payload.take()
        if count == 0:
            return result._finish_complete_overwrite()
        try:
            var staging = issuing.enqueue_create_host_buffer[DType.float32](count)
            try:
                issuing.enqueue_copy(dst_buf=staging, src_buf=self.impl.storage.buffer)
                var received = issuing.create_event()
                issuing.stream().record_event(received)
                received.synchronize()
                # Only a joined D2H completion may initialize the readable
                # host result; its own allocation is fresh and disjoint.
                var output = result.impl.storage.borrow_mut_span()
                unsafe_memcpy(
                    dest=output.unsafe_ptr(), src=staging.unsafe_ptr(), count=count,
                )
                return result._finish_complete_overwrite()
            except:
                try:
                    issuing.stream().synchronize()
                except:
                    # The GPU may still write pinned staging memory. Keep
                    # that owner out of the allocator/pool on device failure.
                    # The SOURCE remains borrowed: the caller must preserve
                    # its DeviceStorage owner until device recovery.
                    forget_deinit(staging)
                    _retain_context_on_unproven_gpu_completion(issuing)
                    return TensorResult[Tensor[Self.rank, HostStorage]](
                        TensorStatus.GPU_QUIESCENCE_UNPROVEN
                    )
                # The D2H join precedes staging destruction; finish the
                # queued pinned-memory free while issuing still owns ctx.
                _ = staging
                try:
                    issuing.stream().synchronize()
                except:
                    _retain_context_on_unproven_gpu_completion(issuing)
                    return TensorResult[Tensor[Self.rank, HostStorage]](
                        TensorStatus.GPU_QUIESCENCE_UNPROVEN
                    )
                return TensorResult[Tensor[Self.rank, HostStorage]](
                    TensorStatus.TRANSFER_UNAVAILABLE
                )
        except:
            # An allocation error did not yield a HostBuffer owner that can
            # be inspected or released here. Ask the backend to quiesce any
            # partially enqueued allocation before returning an error.
            try:
                issuing.stream().synchronize()
            except:
                _retain_context_on_unproven_gpu_completion(issuing)
                return TensorResult[Tensor[Self.rank, HostStorage]](
                    TensorStatus.GPU_QUIESCENCE_UNPROVEN
                )
            return TensorResult[Tensor[Self.rank, HostStorage]](
                TensorStatus.ALLOCATION_FAILED
            )

    def to_host_consuming(
        deinit self, ctx: DeviceContext
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, DeviceStorage):
        # Ownership-retaining alternative to borrowed to_host(). A failed
        # GPU event AND stream fence can leave the input device allocation
        # readable by an in-flight DMA; preserve its physical owner until
        # the device runtime is reset. Success drains the copy before the
        # consuming source owner is destroyed. The caller is still responsible
        # for the lifetime of its borrowed DeviceContext.
        var outcome = self.to_host(ctx)
        if outcome.status == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(self.impl.storage)
        return outcome

    # In-place, GPU-native FP32 fill. Only a fully addressable, unique,
    # exhaustive physical allocation can use the whole-buffer enqueue_fill
    # primitive. Irregular views are explicitly NOT_REPRESENTABLE.
    def fill_device(
        mut self, ctx: DeviceContext, value: Float32
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        # A full, owner-bounded GPU overwrite may RECOVER an uninitialized
        # allocation after a previous drained native failure. Therefore do
        # NOT call _checked_current_layout(): that read-only boundary rejects
        # the very unreadable output whose entire content is being replaced.
        # A poisoned/unjoined owner, however, must never enqueue new writes.
        if (self.impl.storage.quiescence_unproven
            or self.impl.storage.quarantined_context
            or self.impl.storage._native_update_pending
            or not self.impl.storage.lifetime_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        var proof = self.impl.layout.validated_snapshot(
            self.impl.storage.capacity()
        )
        if not proof.is_ok():
            return proof.status
        var mapping = proof.payload.value()
        if not mapping.is_proven_unique():
            return TensorStatus.OVERLAPPING
        if not mapping.is_c_contiguous() or not mapping.is_exhaustive():
            return TensorStatus.NOT_REPRESENTABLE
        if mapping.span_lo != 0:
            return TensorStatus.NOT_REPRESENTABLE
        var count = mapping.shape.numel()
        if count != self.impl.storage.count:
            return TensorStatus.NOT_REPRESENTABLE
        # MAX pads a logically empty DeviceBuffer allocation to one element;
        # the empty tensor still needs an allocation/completion event fence.
        if count > 0 and len(self.impl.storage.buffer) != count:
            return TensorStatus.NOT_REPRESENTABLE
        # An unrelated caller context cannot order the issuing stream.
        # Reject BEFORE revoking a previously valid payload.
        var issuing = self.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        # Use the SAME checked begin/complete transaction as native AOT
        # out= operations. Without this gate, an overlapping full fill could
        # bypass a pending native writer while its output remained unreadable.
        var update = self.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        try:
            if count > 0:
                self.impl.storage.buffer.enqueue_fill(value)
            var completion = issuing.create_event()
            issuing.stream().record_event(completion)
            completion.synchronize()
            return self.impl.storage._complete_native_output_update(
                issuing, TensorStatus.OK, write_ticket, True
            )
        except:
            try:
                issuing.stream().synchronize()
            except:
                # A completion failure quarantines this actual output owner
                # with its native context, not only an error-code snapshot.
                return self.impl.storage._complete_native_output_update(
                    issuing, TensorStatus.GPU_QUIESCENCE_UNPROVEN,
                    write_ticket, True,
                )
            # A drained but possibly partial fill remains unreadable.
            return self.impl.storage._complete_native_output_update(
                issuing, TensorStatus.TRANSFER_UNAVAILABLE,
                write_ticket, True,
            )

    # UNSAFE borrowed-input D2D transfer. If neither the native event nor
    # the fallback stream join establishes quiescence, this function cannot
    # retain the caller's borrowed input after returning. The caller MUST
    # keep it alive until independent native recovery. Ordinary Tensor users
    # should call copy_from_device(), which consumes and safely quarantines
    # the source owner on an unproven completion. No implicit CPU fallback.
    def unsafe_copy_from_device_borrowed(
        mut self, ctx: DeviceContext, ref source: Tensor[Self.rank, DeviceStorage]
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        # D2D is a complete overwrite of the destination owner. A freshly
        # allocated destination can be uninitialized; checking its *readable*
        # descriptor would incorrectly prevent initializing it by copying.
        # In contrast, the source MUST already be fully joined and readable.
        if (self.impl.storage.quiescence_unproven
            or self.impl.storage.quarantined_context
            or self.impl.storage._native_update_pending
            or not self.impl.storage.lifetime_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        var dst_proof = self.impl.layout.validated_snapshot(
            self.impl.storage.capacity()
        )
        if not dst_proof.is_ok():
            return dst_proof.status
        var src_proof = source._checked_current_layout()
        if not src_proof.is_ok():
            return src_proof.status
        var dst = dst_proof.payload.value()
        var src = src_proof.payload.value()
        if not dst.shape.same_extents(src.shape):
            return TensorStatus.SIZE_MISMATCH
        if not dst.is_proven_unique():
            return TensorStatus.OVERLAPPING
        if (not dst.is_c_contiguous() or not src.is_c_contiguous()
            or not dst.is_exhaustive() or not src.is_exhaustive()
            or dst.span_lo != 0 or src.span_lo != 0):
            return TensorStatus.NOT_REPRESENTABLE
        var count = dst.shape.numel()
        if count != self.impl.storage.count or count != source.impl.storage.count:
            return TensorStatus.SIZE_MISMATCH
        # An empty logical copy still has physical allocation ownership.
        # Validate context identity before publishing the destination ready.
        if not source.impl.storage._shares_issuing_context(self.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        var issuing = self.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        if count > 0:
            if (len(self.impl.storage.buffer) != count
                or len(source.impl.storage.buffer) != count):
                return TensorStatus.NOT_REPRESENTABLE
            if not device_fp32_regions_disjoint(
                self.impl.storage.buffer, count, source.impl.storage.buffer, count,
            ):
                return TensorStatus.ALIAS_UNKNOWN
        # Both full D2D and an empty logical D2D must observe the pending
        # writer gate. The empty owner was already allocation-event joined
        # by the checked device allocation/adoption path.
        var update = self.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        if count == 0:
            return self.impl.storage._complete_native_output_update(
                issuing, TensorStatus.OK, write_ticket, True
            )
        var outcome = copy_d2d_fp32(
            issuing, self.impl.storage.buffer, source.impl.storage.buffer, count,
        )
        return self.impl.storage._complete_native_output_update(
            issuing, outcome, write_ticket, True
        )

    def copy_from_device_consuming(
        mut self, ctx: DeviceContext,
        var source: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        # A consuming source makes error lifetime explicit. The destination
        # is already quarantined by unsafe_copy_from_device_borrowed() if no GPU completion
        # proof exists; the source owner must be retained as well, because
        # it may still be read by in-flight DMA. On success the copy is fully
        # joined before the local source owner is destroyed.
        var outcome = self.unsafe_copy_from_device_borrowed(ctx, source)
        if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(source.impl.storage)
        return outcome

    def copy_from_device(
        mut self, ctx: DeviceContext,
        var source: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        # Safe-by-default D2D owner API: source ownership is passed into the
        # call rather than borrowing a buffer that might outlive the call on
        # failed native quiescence. The existing consuming implementation
        # retains (quarantines) both owners if an in-flight DMA cannot join.
        # This is a source compatibility break for borrowed callers: those
        # must choose the explicitly unsafe borrowed entry point.
        return self.copy_from_device_consuming(ctx, source)

    # The device analogue of Tensor.clone(): one new independent owned FP32
    # allocation, explicit same-device transfer, no materialization on CPU.
    # A noncontiguous device view returns NOT_REPRESENTABLE instead of silently
    # migrating or launching a kernel not present in the source package.
    # UNSAFE BORROWED NATIVE INPUT: on GPU_QUIESCENCE_UNPROVEN the caller
    # MUST retain self until external reset. The safe default clone_device()
    # consumes and quarantines the source owner on unknown quiescence.
    def unsafe_clone_device_borrowed(
        ref self, ctx: DeviceContext
    ) -> TensorResult[Tensor[Self.rank, DeviceStorage]] where identical(Self.S, DeviceStorage):
        # The caller context MUST equal the source owner's retained context;
        # an unrelated native queue cannot certify producer completion.
        # An externally produced buffer on a different stream requires its
        # own joined producer event before this borrowed call is legal.
        var source_proof = self._checked_current_layout()
        if not source_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](source_proof.status)
        var mapping = source_proof.payload.value()
        if not mapping.is_c_contiguous() or not mapping.is_exhaustive() or mapping.span_lo != 0:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.NOT_REPRESENTABLE
            )
        var count = mapping.shape.numel()
        if count > 0 and len(self.impl.storage.buffer) != count:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.NOT_REPRESENTABLE
            )
        var target_layout = TensorLayout[Self.rank].c_contiguous(mapping.shape, count)
        if not target_layout.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](target_layout.status)
        # Keep allocation errors separate from errors AFTER ownership exists.
        # The latter must retain the actual returned owner when the stream
        # cannot establish quiescence; the former has no owner to quarantine.
        # The source's retained allocation context, not an arbitrary caller
        # context, is the only proven queue owner for the source device
        # buffer. Allocation, D2D enqueue, completion and adoption all use
        # THIS SAME native context so a foreign event cannot be mistaken
        # for completion of a different stream/device. External producers on
        # independent streams remain outside this explicit contract.
        if not self.impl.storage.lifetime_context:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        var issuing = self.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var new_buffer: DeviceBuffer[DType.float32]
        try:
            # MAX may require a nonzero physical allocation for a zero-size
            # tensor. The public Tensor capacity remains exactly zero.
            new_buffer = issuing.enqueue_create_buffer[DType.float32](max(count, 1))
        except:
            try:
                issuing.stream().synchronize()
            except:
                _retain_context_on_unproven_gpu_completion(issuing)
                return TensorResult[Tensor[Self.rank, DeviceStorage]](
                    TensorStatus.GPU_QUIESCENCE_UNPROVEN
                )
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.ALLOCATION_FAILED
            )
        try:
            if count > 0:
                var status = copy_d2d_fp32(
                    issuing, new_buffer, self.impl.storage.buffer, count,
                )
                if status != TensorStatus.OK:
                    # A fatal D2D copy reports GPU_QUIESCENCE_UNPROVEN when
                    # neither its event nor stream can prove completion. The new
                    # allocation may still be written by the GPU: leaking its
                    # owner is safer than a premature deallocation/UAF.
                    if status == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
                        forget_deinit(new_buffer)
                        _retain_context_on_unproven_gpu_completion(issuing)
                        return TensorResult[Tensor[Self.rank, DeviceStorage]](status)
                    # The joined/unsent copy still owns an allocation. Drain
                    # the native free after dropping that rejected output.
                    _ = new_buffer
                    try:
                        issuing.stream().synchronize()
                    except:
                        _retain_context_on_unproven_gpu_completion(issuing)
                        return TensorResult[Tensor[Self.rank, DeviceStorage]](
                            TensorStatus.GPU_QUIESCENCE_UNPROVEN
                        )
                    return TensorResult[Tensor[Self.rank, DeviceStorage]](status)
            else:
                var allocated = issuing.create_event()
                issuing.stream().record_event(allocated)
                allocated.synchronize()
        except:
            # This handler runs only after new_buffer was actually acquired.
            # If completion cannot be proved, quarantine that owner instead
            # of freeing a potentially in-flight GPU/DMA allocation.
            try:
                issuing.stream().synchronize()
            except:
                forget_deinit(new_buffer)
                _retain_context_on_unproven_gpu_completion(issuing)
                return TensorResult[Tensor[Self.rank, DeviceStorage]](
                    TensorStatus.GPU_QUIESCENCE_UNPROVEN
                )
            _ = new_buffer
            try:
                issuing.stream().synchronize()
            except:
                _retain_context_on_unproven_gpu_completion(issuing)
                return TensorResult[Tensor[Self.rank, DeviceStorage]](
                    TensorStatus.GPU_QUIESCENCE_UNPROVEN
                )
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.TRANSFER_UNAVAILABLE
            )
        # All raising/MAX operations have completed before moving the buffer
        # into DeviceStorage. A later error handler cannot reference a moved
        # allocation owner or accidentally quarantine an uninitialized local.
        var owner = DeviceStorage._unsafe_from_completed_buffer_checked(new_buffer, count, issuing)
        if not owner.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](owner.status)
        var initialized_owner = owner.payload.take()
        var sealed = initialized_owner._mark_complete_overwrite()
        if sealed != TensorStatus.OK:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](sealed)
        return Tensor[Self.rank, DeviceStorage].from_parts_checked(
            initialized_owner, target_layout.payload.value()
        )

    def clone_device_consuming(
        deinit self, ctx: DeviceContext
    ) -> TensorResult[Tensor[Self.rank, DeviceStorage]] where identical(Self.S, DeviceStorage):
        # Prefer this variant when the caller can transfer source ownership.
        # On fatal stream failure, the read-only source may still be in use
        # and must not be freed even when its destination was quarantined.
        # This does not claim to own the caller's DeviceContext or artifact.
        var outcome = self.unsafe_clone_device_borrowed(ctx)
        if outcome.status == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(self.impl.storage)
        return outcome

    def clone_device(
        deinit self, ctx: DeviceContext
    ) -> TensorResult[Tensor[Self.rank, DeviceStorage]] where identical(Self.S, DeviceStorage):
        # Safe-by-default deep device copy. Retain a source owner on an
        # unprovably unfinished transfer; otherwise destroy it only after
        # native completion has been joined. Borrowing callers explicitly
        # opt into unsafe_clone_device_borrowed() and its lease contract.
        var outcome = self.unsafe_clone_device_borrowed(ctx)
        if outcome.status == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(self.impl.storage)
        return outcome

    # Tensor remains the primary public API for device-resident operations.
    # This owner-scoped out= form leaves allocation policy explicit, supports
    # rank 0..8, and never moves a payload between host and GPU implicitly.
    # The sole GPU binary ABI is out-of-place contiguous FP32; other layouts
    # require explicit packing or a different AOT-specialized adapter.
    def device_binary_into[operation: Int](
        ref self,
        ctx: DeviceContext,
        artifact: DeviceExternalFunction,
        identity: BinaryAOTArtifactIdentity,
        target_arch: Int,
        selected: ExecutionPlan,
        right: Tensor[Self.rank, DeviceStorage],
        mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        comptime assert operation >= 0 and operation <= 5, "FP32 device opcode must be 0..5"
        var prepared = selected.checked_snapshot()
        if not prepared.is_ok():
            return prepared.status
        var plan = prepared.payload.value()
        if plan.placement != ExecutionPlacement.GPU_ONLY or plan.output_device != 1:
            return TensorStatus.DEVICE_MISMATCH
        var left_check = self._checked_current_layout()
        if not left_check.is_ok():
            return left_check.status
        var right_check = right._checked_current_layout()
        if not right_check.is_ok():
            return right_check.status
        var output_check = result._checked_device_complete_overwrite_layout()
        if not output_check.is_ok():
            return output_check.status
        var a = left_check.payload.value()
        var b = right_check.payload.value()
        var o = output_check.payload.value()
        if not a.shape.same_extents(b.shape) or not a.shape.same_extents(o.shape):
            return TensorStatus.SIZE_MISMATCH
        if plan.total_items != a.shape.numel():
            return TensorStatus.SIZE_MISMATCH
        if not o.is_proven_unique():
            return TensorStatus.OVERLAPPING
        if not (a.is_c_contiguous() and b.is_c_contiguous() and o.is_c_contiguous()):
            return TensorStatus.NOT_REPRESENTABLE
        if not (a.is_exhaustive() and b.is_exhaustive() and o.is_exhaustive()):
            return TensorStatus.NOT_REPRESENTABLE
        if a.offset != 0 or b.offset != 0 or o.offset != 0:
            return TensorStatus.NOT_REPRESENTABLE
        # The borrowed owners remain alive while the low-level adapter waits
        # on the event; DeviceStorage adoption forbids duplicate ownership.
        # The output owner supplies the issuing stream. Reject a caller
        # DeviceContext without matching native context identity before
        # starting a write or revoking old output readability.
        if (not result.impl.storage.lifetime_context
            or result.impl.storage.quiescence_unproven
            or result.impl.storage.quarantined_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        if not right.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        var issuing = result.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        var update = result.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        var outcome = run_owned_bound_gpu_binary_fp32[operation](
            issuing, artifact, identity, target_arch, plan,
            result.impl.storage.buffer, self.impl.storage.buffer,
            right.impl.storage.buffer,
        )
        # Rejected AOT identities, shapes and aliases never enqueue. Keep
        # an already-initialized out= owner readable on those errors. Only
        # a failure that may have reached the GPU invalidates its contents.
        return result.impl.storage._complete_native_output_update(
            issuing, outcome, write_ticket, True
        )

    def unsafe_device_binary_broadcast_into[operation: Int, right_rank: Int, out_rank: Int](
        ref self, ctx: DeviceContext, selected: ExecutionPlan,
        ref rhs: Tensor[right_rank, DeviceStorage],
        mut result: Tensor[out_rank, DeviceStorage],
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        # UNSAFE BORROWED API: if GPU_QUIESCENCE_UNPROVEN is returned, the
        # caller MUST retain both inputs and the DeviceContext until native
        # device reset/recovery. The result's backing allocation is retained
        # by DeviceStorage; that guarantee does not cover borrowed inputs.
        # Prefer the consuming counterpart below to retain inputs on failure.
        var left_check = self._checked_current_layout()
        if not left_check.is_ok():
            return left_check.status
        var right_check = rhs._checked_current_layout()
        if not right_check.is_ok():
            return right_check.status
        var output_check = result._checked_device_strided_write_layout()
        if not output_check.is_ok():
            return output_check.status
        var dst = output_check.payload.value()
        var writes_entire_owner = (
            dst.is_exhaustive() and dst.span_lo == 0
            and dst.shape.numel() == result.impl.storage.count
        )
        # The output owner supplies the issuing stream. Reject a caller
        # DeviceContext without matching native context identity before
        # starting a write or revoking old output readability.
        if (not result.impl.storage.lifetime_context
            or result.impl.storage.quiescence_unproven
            or result.impl.storage.quarantined_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        if not rhs.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        var issuing = result.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        var update = result.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        var outcome = run_owned_gpu_binary_strided_fp32[
            out_rank, Self.rank, right_rank, operation,
        ](
            issuing, selected, development_allow_runtime_compilation, result.impl.storage.buffer,
            self.impl.storage.buffer, rhs.impl.storage.buffer,
            output_check.payload.value(), left_check.payload.value(),
            right_check.payload.value(),
        )
        return result.impl.storage._complete_native_output_update(
            issuing, outcome, write_ticket, writes_entire_owner
        )

    def unsafe_device_unary_strided_into[operation: Int](
        ref self, ctx: DeviceContext, selected: ExecutionPlan,
        mut result: Tensor[Self.rank, DeviceStorage],
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        # UNSAFE BORROWED API: caller retains source and DeviceContext until
        # the GPU has quiesced, including an unproven-quiescence failure.
        # Prefer device_unary_strided_consuming when ownership can move.
        var input_check = self._checked_current_layout()
        if not input_check.is_ok():
            return input_check.status
        var output_check = result._checked_device_strided_write_layout()
        if not output_check.is_ok():
            return output_check.status
        var dst = output_check.payload.value()
        var writes_entire_owner = (
            dst.is_exhaustive() and dst.span_lo == 0
            and dst.shape.numel() == result.impl.storage.count
        )
        # The output owner supplies the issuing stream. Reject a caller
        # DeviceContext without matching native context identity before
        # starting a write or revoking old output readability.
        if (not result.impl.storage.lifetime_context
            or result.impl.storage.quiescence_unproven
            or result.impl.storage.quarantined_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        var issuing = result.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        var update = result.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        var outcome = run_owned_gpu_unary_strided_fp32[Self.rank, operation](
            issuing, selected, development_allow_runtime_compilation, result.impl.storage.buffer,
            self.impl.storage.buffer,
            output_check.payload.value(), input_check.payload.value(),
        )
        return result.impl.storage._complete_native_output_update(
            issuing, outcome, write_ticket, writes_entire_owner
        )

    @staticmethod
    def device_binary_broadcast_consuming[
        operation: Int, right_rank: Int, out_rank: Int,
    ](
        ctx: DeviceContext, selected: ExecutionPlan,
        var left: Self,
        var right: Tensor[right_rank, DeviceStorage],
        mut result: Tensor[out_rank, DeviceStorage],
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        # An owner-consuming entry point prevents early destruction of both
        # inputs if neither GPU event nor fallback stream wait can prove
        # quiescence. DeviceContext lifetime remains a caller obligation.
        var outcome = left.unsafe_device_binary_broadcast_into[
            operation, right_rank, out_rank,
        ](ctx, selected, right, result, development_allow_runtime_compilation)
        if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(left)
            forget_deinit(right)
        return outcome

    @staticmethod
    def device_unary_strided_consuming[operation: Int](
        ctx: DeviceContext, selected: ExecutionPlan,
        var source: Self,
        mut result: Tensor[Self.rank, DeviceStorage],
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        var outcome = source.unsafe_device_unary_strided_into[operation](
            ctx, selected, result, development_allow_runtime_compilation
        )
        if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(source)
        return outcome

    def unsafe_device_reduce_axis_strided_into[operation: Int, axis: Int](
        ref self, ctx: DeviceContext, selected: ExecutionPlan,
        mut result: Tensor[Self.rank, DeviceStorage],
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        # Keepdims result: the selected axis has extent 1; all other axes
        # preserve input dimensions. Arbitrary signed strides are accepted.
        # Borrowed source/context must survive unknown GPU quiescence.
        var input_check = self._checked_current_layout()
        if not input_check.is_ok():
            return input_check.status
        var output_check = result._checked_device_strided_write_layout()
        if not output_check.is_ok():
            return output_check.status
        var dst = output_check.payload.value()
        var writes_entire_owner = (
            dst.is_exhaustive() and dst.span_lo == 0
            and dst.shape.numel() == result.impl.storage.count
        )
        # The output owner supplies the issuing stream. Reject a caller
        # DeviceContext without matching native context identity before
        # starting a write or revoking old output readability.
        if (not result.impl.storage.lifetime_context
            or result.impl.storage.quiescence_unproven
            or result.impl.storage.quarantined_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        var issuing = result.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        var update = result.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        var outcome = run_owned_gpu_reduce_axis_strided_fp32[
            Self.rank, axis, operation,
        ](
            issuing, selected, development_allow_runtime_compilation, result.impl.storage.buffer,
            self.impl.storage.buffer,
            output_check.payload.value(), input_check.payload.value(),
        )
        return result.impl.storage._complete_native_output_update(
            issuing, outcome, write_ticket, writes_entire_owner
        )

    @staticmethod
    def device_reduce_axis_strided_consuming[operation: Int, axis: Int](
        ctx: DeviceContext, selected: ExecutionPlan,
        var source: Self,
        mut result: Tensor[Self.rank, DeviceStorage],
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        var outcome = source.unsafe_device_reduce_axis_strided_into[
            operation, axis,
        ](ctx, selected, result, development_allow_runtime_compilation)
        if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(source)
        return outcome

    def unsafe_device_reduce_axes_strided_into[operation: Int](
        ref self, ctx: DeviceContext, selected: ExecutionPlan,
        mut result: Tensor[Self.rank, DeviceStorage], axes_mask: Int,
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        # Runtime bitmask selects one or more axes of a statically known rank.
        # Every selected axis must be a size-one axis in the output mapping.
        # Inputs are BORROWED; the caller must preserve their lifetimes if
        # event and stream quiescence cannot be proven.
        var source_layout = self._checked_current_layout()
        if not source_layout.is_ok():
            return source_layout.status
        var output_layout = result._checked_device_strided_write_layout()
        if not output_layout.is_ok():
            return output_layout.status
        var dst = output_layout.payload.value()
        var writes_entire_owner = (
            dst.is_exhaustive() and dst.span_lo == 0
            and dst.shape.numel() == result.impl.storage.count
        )
        # The output owner supplies the issuing stream. Reject a caller
        # DeviceContext without matching native context identity before
        # starting a write or revoking old output readability.
        if (not result.impl.storage.lifetime_context
            or result.impl.storage.quiescence_unproven
            or result.impl.storage.quarantined_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        var issuing = result.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        var update = result.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        var outcome = run_owned_gpu_reduce_axes_strided_fp32[
            Self.rank, operation,
        ](
            issuing, selected, development_allow_runtime_compilation,
            result.impl.storage.buffer, self.impl.storage.buffer,
            output_layout.payload.value(), source_layout.payload.value(),
            axes_mask,
        )
        return result.impl.storage._complete_native_output_update(
            issuing, outcome, write_ticket, writes_entire_owner
        )

    @staticmethod
    def device_reduce_axes_strided_consuming[operation: Int](
        ctx: DeviceContext, selected: ExecutionPlan,
        var source: Self, mut result: Tensor[Self.rank, DeviceStorage],
        axes_mask: Int, development_allow_runtime_compilation: Bool = False,
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        var outcome = source.unsafe_device_reduce_axes_strided_into[operation](
            ctx, selected, result, axes_mask,
            development_allow_runtime_compilation,
        )
        if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(source)
        return outcome

    def unsafe_device_scan_axis_strided_into[operation: Int, axis: Int](
        ref self, ctx: DeviceContext, selected: ExecutionPlan,
        mut result: Tensor[Self.rank, DeviceStorage],
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        # Scan follows logical axis order regardless of positive/negative
        # physical strides. Borrowing remains unsafe on unproven quiescence.
        var input_check = self._checked_current_layout()
        if not input_check.is_ok():
            return input_check.status
        var output_check = result._checked_device_strided_write_layout()
        if not output_check.is_ok():
            return output_check.status
        var dst = output_check.payload.value()
        var writes_entire_owner = (
            dst.is_exhaustive() and dst.span_lo == 0
            and dst.shape.numel() == result.impl.storage.count
        )
        # The output owner supplies the issuing stream. Reject a caller
        # DeviceContext without matching native context identity before
        # starting a write or revoking old output readability.
        if (not result.impl.storage.lifetime_context
            or result.impl.storage.quiescence_unproven
            or result.impl.storage.quarantined_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        var issuing = result.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        var update = result.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        var outcome = run_owned_gpu_scan_axis_strided_fp32[
            Self.rank, axis, operation,
        ](
            issuing, selected, development_allow_runtime_compilation, result.impl.storage.buffer,
            self.impl.storage.buffer,
            output_check.payload.value(), input_check.payload.value(),
        )
        return result.impl.storage._complete_native_output_update(
            issuing, outcome, write_ticket, writes_entire_owner
        )

    @staticmethod
    def device_scan_axis_strided_consuming[operation: Int, axis: Int](
        ctx: DeviceContext, selected: ExecutionPlan,
        var source: Self,
        mut result: Tensor[Self.rank, DeviceStorage],
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        var outcome = source.unsafe_device_scan_axis_strided_into[
            operation, axis,
        ](ctx, selected, result, development_allow_runtime_compilation)
        if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(source)
        return outcome

    def device_unary_into[operation: Int](
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: UnaryAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        comptime assert operation >= 0 and operation <= 8, "invalid unary opcode"
        var lhs_check = self._checked_current_layout()
        if not lhs_check.is_ok():
            return lhs_check.status
        var out_check = result._checked_device_complete_overwrite_layout()
        if not out_check.is_ok():
            return out_check.status
        var lhs = lhs_check.payload.value()
        var dst = out_check.payload.value()
        if not lhs.shape.same_extents(dst.shape):
            return TensorStatus.SIZE_MISMATCH
        # A trusted buffer capacity alone cannot prove the logical operation
        # covers the entire output. Reject truncated AOT work schedules.
        var checked_plan = plan.checked_snapshot()
        if not checked_plan.is_ok():
            return checked_plan.status
        if checked_plan.payload.value().total_items != lhs.shape.numel():
            return TensorStatus.SIZE_MISMATCH
        if not lhs.is_c_contiguous() or not lhs.is_exhaustive() or lhs.offset != 0:
            return TensorStatus.NOT_REPRESENTABLE
        if not dst.is_c_contiguous() or not dst.is_exhaustive() or dst.offset != 0:
            return TensorStatus.NOT_REPRESENTABLE
        if not dst.is_proven_unique():
            return TensorStatus.OVERLAPPING
        # Pass the same checked snapshot used to prove the Tensor extents.
        # Never re-read the caller-owned mutable plan after boundary checks.
        var safe_plan = checked_plan.payload.take()
        # The output owner supplies the issuing stream. Reject a caller
        # DeviceContext without matching native context identity before
        # starting a write or revoking old output readability.
        if (not result.impl.storage.lifetime_context
            or result.impl.storage.quiescence_unproven
            or result.impl.storage.quarantined_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        var issuing = result.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        var update = result.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        var outcome = run_owned_bound_gpu_unary_fp32[operation](
            issuing, artifact, identity, target_arch, safe_plan,
            result.impl.storage.buffer, self.impl.storage.buffer,
        )
        # Rejected AOT identities, shapes and aliases never enqueue. Keep
        # an already-initialized out= owner readable on those errors. Only
        # a failure that may have reached the GPU invalidates its contents.
        return result.impl.storage._complete_native_output_update(
            issuing, outcome, write_ticket, True
        )

    # Rank-reducing native FP32 GPU SUM/PROD/MIN/MAX/MEAN along the
    # innermost axis; the operation is a compile-time specialization. The output
    # Tensor carries rank-1 (rank 0 is the scalar result for rank-1 input).
    # Both sides must be flat, exhaustive, separately-owned device buffers.
    # No hidden allocation, materialization, or host transfer occurs.
    def device_reduce_last_axis_into[operation: Int](
        ref self,
        ctx: DeviceContext,
        artifact: DeviceExternalFunction,
        identity: ReduceRowsAOTArtifactIdentity,
        target_arch: Int,
        plan: ExecutionPlan,
        mut result: Tensor[Self.rank - 1, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        comptime if operation < 0 or operation > 4:
            return TensorStatus.UNSUPPORTED
        var source_proof = self._checked_current_layout()
        if not source_proof.is_ok():
            return source_proof.status
        var result_proof = result._checked_device_complete_overwrite_layout()
        if not result_proof.is_ok():
            return result_proof.status
        var source_layout = source_proof.payload.value()
        var dest_layout = result_proof.payload.value()
        if not source_layout.is_c_contiguous() or not source_layout.is_exhaustive():
            return TensorStatus.NOT_REPRESENTABLE
        if not dest_layout.is_c_contiguous() or not dest_layout.is_exhaustive():
            return TensorStatus.NOT_REPRESENTABLE
        if source_layout.offset != 0 or dest_layout.offset != 0:
            return TensorStatus.NOT_REPRESENTABLE
        if not dest_layout.is_proven_unique():
            return TensorStatus.OVERLAPPING
        for i in range(Self.rank - 1):
            if source_layout.shape.dims.get(i) != dest_layout.shape.dims.get(i):
                return TensorStatus.SIZE_MISMATCH
        var rows = dest_layout.shape.numel()
        var cols = source_layout.shape.dims.get(Self.rank - 1)
        if rows < 0 or cols < 0:
            return TensorStatus.INVALID_SHAPE
        if rows > 0 and cols > Int.MAX // rows:
            return TensorStatus.OVERFLOW
        if rows * cols != source_layout.shape.numel():
            return TensorStatus.SIZE_MISMATCH
        comptime if operation >= 2:
            if rows > 0 and cols == 0:
                return TensorStatus.SIZE_MISMATCH
        var selected = plan.checked_snapshot()
        if not selected.is_ok():
            return selected.status
        var safe_plan = selected.payload.take()
        if safe_plan.total_items != rows:
            return TensorStatus.SIZE_MISMATCH
        # The output owner supplies the issuing stream. Reject a caller
        # DeviceContext without matching native context identity before
        # starting a write or revoking old output readability.
        if (not result.impl.storage.lifetime_context
            or result.impl.storage.quiescence_unproven
            or result.impl.storage.quarantined_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        var issuing = result.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        var update = result.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        var outcome = run_owned_bound_gpu_reduce_rows_fp32[operation](
            issuing, artifact, identity, target_arch, safe_plan,
            result.impl.storage.buffer, self.impl.storage.buffer, rows, cols,
        )
        # Rejected AOT identities, shapes and aliases never enqueue. Keep
        # an already-initialized out= owner readable on those errors. Only
        # a failure that may have reached the GPU invalidates its contents.
        return result.impl.storage._complete_native_output_update(
            issuing, outcome, write_ticket, True
        )

    # Stable convenience entry points all use the same checked Tensor
    # layout/owner boundary. The requested opcode is bound at compile time.
    def device_sum_last_axis_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: ReduceRowsAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank - 1, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        return self.device_reduce_last_axis_into[0](
            ctx, artifact, identity, target_arch, plan, result,
        )

    def device_prod_last_axis_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: ReduceRowsAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank - 1, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        return self.device_reduce_last_axis_into[1](
            ctx, artifact, identity, target_arch, plan, result,
        )

    def device_min_last_axis_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: ReduceRowsAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank - 1, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        return self.device_reduce_last_axis_into[2](
            ctx, artifact, identity, target_arch, plan, result,
        )

    def device_max_last_axis_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: ReduceRowsAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank - 1, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        return self.device_reduce_last_axis_into[3](
            ctx, artifact, identity, target_arch, plan, result,
        )

    def device_mean_last_axis_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: ReduceRowsAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank - 1, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        return self.device_reduce_last_axis_into[4](
            ctx, artifact, identity, target_arch, plan, result,
        )

    # Row-wise inclusive FP32 native scan. Last-axis scan preserves rank and
    # extents; each row is independent. AOT identity is operation-specific.
    # Requires zero-offset, C-contiguous, exhaustive and disjoint storage.
    def device_scan_last_axis_into[operation: Int](
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: ScanRowsAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        comptime if operation < 0 or operation > 3:
            return TensorStatus.UNSUPPORTED
        var source_proof = self._checked_current_layout()
        if not source_proof.is_ok():
            return source_proof.status
        var output_proof = result._checked_device_complete_overwrite_layout()
        if not output_proof.is_ok():
            return output_proof.status
        var source = source_proof.payload.value()
        var output = output_proof.payload.value()
        if not source.shape.same_extents(output.shape):
            return TensorStatus.SIZE_MISMATCH
        if (not source.is_c_contiguous() or not source.is_exhaustive()
            or source.offset != 0):
            return TensorStatus.NOT_REPRESENTABLE
        if (not output.is_c_contiguous() or not output.is_exhaustive()
            or output.offset != 0):
            return TensorStatus.NOT_REPRESENTABLE
        if not output.is_proven_unique():
            return TensorStatus.OVERLAPPING
        var count = output.shape.numel()
        var cols = source.shape.dims.get(Self.rank - 1)
        var rows = 0
        if cols > 0:
            rows = count // cols
            if rows * cols != count:
                return TensorStatus.SIZE_MISMATCH
        elif count != 0:
            return TensorStatus.INVALID_SHAPE
        var checked = plan.checked_snapshot()
        if not checked.is_ok():
            return checked.status
        var snapshot = checked.payload.take()
        if snapshot.total_items != count:
            return TensorStatus.SIZE_MISMATCH
        # The output owner supplies the issuing stream. Reject a caller
        # DeviceContext without matching native context identity before
        # starting a write or revoking old output readability.
        if (not result.impl.storage.lifetime_context
            or result.impl.storage.quiescence_unproven
            or result.impl.storage.quarantined_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        var issuing = result.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        var update = result.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        var outcome = run_owned_bound_gpu_scan_rows_fp32[operation](
            issuing, artifact, identity, target_arch, snapshot,
            result.impl.storage.buffer, self.impl.storage.buffer, rows, cols,
        )
        # Rejected AOT identities, shapes and aliases never enqueue. Keep
        # an already-initialized out= owner readable on those errors. Only
        # a failure that may have reached the GPU invalidates its contents.
        return result.impl.storage._complete_native_output_update(
            issuing, outcome, write_ticket, True
        )

    def device_cumsum_last_axis_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: ScanRowsAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        return self.device_scan_last_axis_into[0](
            ctx, artifact, identity, target_arch, plan, result,
        )

    def device_cumprod_last_axis_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: ScanRowsAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        return self.device_scan_last_axis_into[1](
            ctx, artifact, identity, target_arch, plan, result,
        )

    def device_cummin_last_axis_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: ScanRowsAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        return self.device_scan_last_axis_into[2](
            ctx, artifact, identity, target_arch, plan, result,
        )

    def device_cummax_last_axis_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: ScanRowsAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        return self.device_scan_last_axis_into[3](
            ctx, artifact, identity, target_arch, plan, result,
        )

    def device_mm_into(
        ref self,
        ctx: DeviceContext,
        artifact: DeviceExternalFunction,
        identity: GemmAOTArtifactIdentity,
        target_arch: Int,
        math_mode: Int,
        plan: ExecutionPlan,
        rhs: Tensor[2, DeviceStorage],
        mut result: Tensor[2, DeviceStorage],
    ) -> TensorStatus where Self.rank == 2 and identical(Self.S, DeviceStorage):
        # This operation consumes NO host memory and makes NO implicit copy.
        # A read-only borrow keeps both operands live until the event join.
        var lhs_check = self._checked_current_layout()
        if not lhs_check.is_ok():
            return lhs_check.status
        var rhs_check = rhs._checked_current_layout()
        if not rhs_check.is_ok():
            return rhs_check.status
        var out_check = result._checked_device_complete_overwrite_layout()
        if not out_check.is_ok():
            return out_check.status
        var lhs = lhs_check.payload.value()
        var right = rhs_check.payload.value()
        var dst = out_check.payload.value()
        if not lhs.is_c_contiguous() or not lhs.is_exhaustive() or lhs.offset != 0:
            return TensorStatus.NOT_REPRESENTABLE
        if not right.is_c_contiguous() or not right.is_exhaustive() or right.offset != 0:
            return TensorStatus.NOT_REPRESENTABLE
        if not dst.is_c_contiguous() or not dst.is_exhaustive() or dst.offset != 0:
            return TensorStatus.NOT_REPRESENTABLE
        if not dst.is_proven_unique():
            return TensorStatus.OVERLAPPING
        var m = lhs.shape.dims.get(0)
        var k = lhs.shape.dims.get(1)
        var n = right.shape.dims.get(1)
        if right.shape.dims.get(0) != k:
            return TensorStatus.SIZE_MISMATCH
        if dst.shape.dims.get(0) != m or dst.shape.dims.get(1) != n:
            return TensorStatus.SIZE_MISMATCH
        var selected = plan.checked_snapshot()
        if not selected.is_ok():
            return selected.status
        var safe_plan = selected.payload.take()
        if safe_plan.total_items != dst.shape.numel():
            return TensorStatus.SIZE_MISMATCH
        # The output owner supplies the issuing stream. Reject a caller
        # DeviceContext without matching native context identity before
        # starting a write or revoking old output readability.
        if (not result.impl.storage.lifetime_context
            or result.impl.storage.quiescence_unproven
            or result.impl.storage.quarantined_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        if not rhs.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        var issuing = result.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        var update = result.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        var outcome = run_owned_bound_gpu_gemm_fp32(
            issuing, artifact, identity, target_arch, math_mode, safe_plan,
            result.impl.storage.buffer, self.impl.storage.buffer,
            rhs.impl.storage.buffer, m, k, n,
        )
        # Rejected AOT identities, shapes and aliases never enqueue. Keep
        # an already-initialized out= owner readable on those errors. Only
        # a failure that may have reached the GPU invalidates its contents.
        return result.impl.storage._complete_native_output_update(
            issuing, outcome, write_ticket, True
        )

    # Device-resident GEMV through the already validated strict-FP32 AOT
    # GEMM ABI with N=1. This avoids a new unbound kernel handle or any
    # implicit host migration; an offline profile may choose a dedicated
    # GEMV artifact later. AOT and kernel ownership remain explicit.
    def device_gemv_into(
        ref self,
        ctx: DeviceContext,
        artifact: DeviceExternalFunction,
        identity: GemmAOTArtifactIdentity,
        target_arch: Int,
        math_mode: Int,
        plan: ExecutionPlan,
        vector: Tensor[1, DeviceStorage],
        mut result: Tensor[1, DeviceStorage],
    ) -> TensorStatus where Self.rank == 2 and identical(Self.S, DeviceStorage):
        var matrix_proof = self._checked_current_layout()
        if not matrix_proof.is_ok():
            return matrix_proof.status
        var vector_proof = vector._checked_current_layout()
        if not vector_proof.is_ok():
            return vector_proof.status
        var output_proof = result._checked_device_complete_overwrite_layout()
        if not output_proof.is_ok():
            return output_proof.status
        var matrix = matrix_proof.payload.value()
        var x = vector_proof.payload.value()
        var output = output_proof.payload.value()
        if not matrix.is_c_contiguous() or not matrix.is_exhaustive() or matrix.offset != 0:
            return TensorStatus.NOT_REPRESENTABLE
        if not x.is_c_contiguous() or not x.is_exhaustive() or x.offset != 0:
            return TensorStatus.NOT_REPRESENTABLE
        if not output.is_c_contiguous() or not output.is_exhaustive() or output.offset != 0:
            return TensorStatus.NOT_REPRESENTABLE
        if not output.is_proven_unique():
            return TensorStatus.OVERLAPPING
        var m = matrix.shape.dims.get(0)
        var k = matrix.shape.dims.get(1)
        if x.shape.dims.get(0) != k or output.shape.dims.get(0) != m:
            return TensorStatus.SIZE_MISMATCH
        var checked_plan = plan.checked_snapshot()
        if not checked_plan.is_ok():
            return checked_plan.status
        var frozen = checked_plan.payload.take()
        if frozen.total_items != m:
            return TensorStatus.SIZE_MISMATCH
        # Reuse a typed and identity-bound AOT adapter, never an unbound
        # native function pointer. GEMM's B argument is exactly Kx1 here.
        # The output owner supplies the issuing stream. Reject a caller
        # DeviceContext without matching native context identity before
        # starting a write or revoking old output readability.
        if (not result.impl.storage.lifetime_context
            or result.impl.storage.quiescence_unproven
            or result.impl.storage.quarantined_context):
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        if not self.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        if not vector.impl.storage._shares_issuing_context(result.impl.storage):
            return TensorStatus.DEVICE_MISMATCH
        var issuing = result.impl.storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorStatus.DEVICE_MISMATCH
        var update = result.impl.storage._begin_native_output_update()
        if not update.is_ok():
            return update.status
        var write_ticket = update.payload.value()
        var outcome = run_owned_bound_gpu_gemm_fp32(
            issuing, artifact, identity, target_arch, math_mode, frozen,
            result.impl.storage.buffer, self.impl.storage.buffer,
            vector.impl.storage.buffer, m, k, 1,
        )
        # Rejected AOT identities, shapes and aliases never enqueue. Keep
        # an already-initialized out= owner readable on those errors. Only
        # a failure that may have reached the GPU invalidates its contents.
        return result.impl.storage._complete_native_output_update(
            issuing, outcome, write_ticket, True
        )


    # Native borrowed-source operations above cannot OWN a source after a
    # failed DeviceEvent AND failed stream wait. Consuming wrappers provide
    # the safe ownership contract: all read-only inputs remain allocated if
    # the GPU may still reference them. Their lost buffers are deliberately
    # quarantined, never returned to a stream-unsafe pool. A foreign AOT
    # symbol/stream still requires an explicit trusted backend contract.

    @staticmethod
    def device_binary_into_consuming[operation: Int](
        ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: BinaryAOTArtifactIdentity, target_arch: Int,
        selected: ExecutionPlan, var left: Self,
        var right: Tensor[Self.rank, DeviceStorage],
        mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        var status = left.device_binary_into[operation](
            ctx, artifact, identity, target_arch, selected, right, result,
        )
        if status == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(left)
            forget_deinit(right)
        return status

    @staticmethod
    def device_unary_into_consuming[operation: Int](
        ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: UnaryAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, var source: Self,
        mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        var status = source.device_unary_into[operation](
            ctx, artifact, identity, target_arch, plan, result,
        )
        if status == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(source)
        return status

    @staticmethod
    def device_reduce_last_axis_into_consuming[operation: Int](
        ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: ReduceRowsAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, var source: Self,
        mut result: Tensor[Self.rank - 1, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        var status = source.device_reduce_last_axis_into[operation](
            ctx, artifact, identity, target_arch, plan, result,
        )
        if status == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(source)
        return status

    @staticmethod
    def device_scan_last_axis_into_consuming[operation: Int](
        ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: ScanRowsAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, var source: Self,
        mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where Self.rank >= 1 and identical(Self.S, DeviceStorage):
        var status = source.device_scan_last_axis_into[operation](
            ctx, artifact, identity, target_arch, plan, result,
        )
        if status == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(source)
        return status

    @staticmethod
    def device_mm_into_consuming(
        ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: GemmAOTArtifactIdentity, target_arch: Int,
        math_mode: Int, plan: ExecutionPlan, var left: Self,
        var right: Tensor[2, DeviceStorage],
        mut result: Tensor[2, DeviceStorage],
    ) -> TensorStatus where Self.rank == 2 and identical(Self.S, DeviceStorage):
        var status = left.device_mm_into(
            ctx, artifact, identity, target_arch, math_mode, plan, right, result,
        )
        if status == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(left)
            forget_deinit(right)
        return status

    @staticmethod
    def device_gemv_into_consuming(
        ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: GemmAOTArtifactIdentity, target_arch: Int,
        math_mode: Int, plan: ExecutionPlan, var matrix: Self,
        var vector: Tensor[1, DeviceStorage],
        mut result: Tensor[1, DeviceStorage],
    ) -> TensorStatus where Self.rank == 2 and identical(Self.S, DeviceStorage):
        var status = matrix.device_gemv_into(
            ctx, artifact, identity, target_arch, math_mode, plan, vector, result,
        )
        if status == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
            forget_deinit(matrix)
            forget_deinit(vector)
        return status

    def device_add_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: BinaryAOTArtifactIdentity, target_arch: Int,
        selected: ExecutionPlan, right: Tensor[Self.rank, DeviceStorage],
        mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        return self.device_binary_into[0](
            ctx, artifact, identity, target_arch, selected, right, result,
        )

    def device_sub_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: BinaryAOTArtifactIdentity, target_arch: Int,
        selected: ExecutionPlan, right: Tensor[Self.rank, DeviceStorage],
        mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        return self.device_binary_into[1](
            ctx, artifact, identity, target_arch, selected, right, result,
        )

    def device_mul_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: BinaryAOTArtifactIdentity, target_arch: Int,
        selected: ExecutionPlan, right: Tensor[Self.rank, DeviceStorage],
        mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        return self.device_binary_into[2](
            ctx, artifact, identity, target_arch, selected, right, result,
        )

    def device_div_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: BinaryAOTArtifactIdentity, target_arch: Int,
        selected: ExecutionPlan, right: Tensor[Self.rank, DeviceStorage],
        mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        return self.device_binary_into[3](
            ctx, artifact, identity, target_arch, selected, right, result,
        )

    def device_maximum_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: BinaryAOTArtifactIdentity, target_arch: Int,
        selected: ExecutionPlan, right: Tensor[Self.rank, DeviceStorage],
        mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        return self.device_binary_into[4](
            ctx, artifact, identity, target_arch, selected, right, result,
        )

    def device_minimum_into(
        ref self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: BinaryAOTArtifactIdentity, target_arch: Int,
        selected: ExecutionPlan, right: Tensor[Self.rank, DeviceStorage],
        mut result: Tensor[Self.rank, DeviceStorage],
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        return self.device_binary_into[5](
            ctx, artifact, identity, target_arch, selected, right, result,
        )

    def shape(self) -> TensorShape[Self.rank]:
        return self.impl.layout.shape.copy()

    def layout(self) -> TensorLayout[Self.rank]:
        return self.impl.layout.copy()

    def _checked_current_layout(self) -> TensorResult[TensorLayout[Self.rank]]:
        # A readable owning Tensor must have a COMPLETELY initialized buffer.
        # An internal complete-overwrite result may be assembled before its
        # data is written, but it cannot pass a checked reading/view boundary.
        if not self.impl.storage.is_initialized():
            return TensorResult[TensorLayout[Self.rank]](TensorStatus.UNINITIALIZED)
        return self.impl.layout.validated_snapshot(self.impl.storage.capacity())

    def _checked_device_strided_write_layout(
        ref self,
    ) -> TensorResult[TensorLayout[Self.rank]] where identical(Self.S, DeviceStorage):
        # A strided native writer may initialize an unreadable GPU owner only
        # when the validated UNIQUE layout covers every logical FP32 element
        # of that owner. Partial views require previously initialized memory;
        # otherwise they would publish unwritten, unspecified payload bytes.
        # This proof is a source-level bounds/coverage proof, NOT native
        # event/stream ordering. Native writes still need a joined completion.
        if (self.impl.storage.quiescence_unproven
            or self.impl.storage.quarantined_context
            or self.impl.storage._native_update_pending
            or not self.impl.storage.lifetime_context):
            return TensorResult[TensorLayout[Self.rank]](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        var checked = self.impl.layout.validated_snapshot(
            self.impl.storage.capacity()
        )
        if not checked.is_ok():
            return checked
        var mapping = checked.payload.value()
        if not mapping.is_proven_unique():
            return TensorResult[TensorLayout[Self.rank]](
                TensorStatus.OVERLAPPING
            )
        var writes_entire_owner = (
            mapping.is_exhaustive() and mapping.span_lo == 0
            and mapping.shape.numel() == self.impl.storage.count
        )
        if not writes_entire_owner and not self.impl.storage.is_initialized():
            return TensorResult[TensorLayout[Self.rank]](
                TensorStatus.UNINITIALIZED
            )
        return checked

    def _checked_device_complete_overwrite_layout(
        ref self,
    ) -> TensorResult[TensorLayout[Self.rank]] where identical(Self.S, DeviceStorage):
        # An out= kernel is a complete writer, not a reader of the previous
        # destination. Validate its structural and allocation proof without
        # demanding that the prior FP32 payload was initialized. Incomplete
        # or foreign-stream-quarantined owners are never writable this way.
        if (self.impl.storage.quiescence_unproven
            or self.impl.storage.quarantined_context
            or self.impl.storage._native_update_pending
            or not self.impl.storage.lifetime_context):
            return TensorResult[TensorLayout[Self.rank]](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        var verified = self.impl.layout.validated_snapshot(
            self.impl.storage.capacity()
        )
        if not verified.is_ok():
            return verified
        var mapping = verified.payload.value()
        if not mapping.is_proven_unique():
            return TensorResult[TensorLayout[Self.rank]](
                TensorStatus.OVERLAPPING
            )
        if (not mapping.is_c_contiguous() or not mapping.is_exhaustive()
            or mapping.offset != 0):
            return TensorResult[TensorLayout[Self.rank]](
                TensorStatus.NOT_REPRESENTABLE
            )
        # The AOT writer must overwrite the entire *owner*, not merely a
        # view or prefix, before initialized_elements may become count.
        if mapping.shape.numel() != self.impl.storage.count:
            return TensorResult[TensorLayout[Self.rank]](
                TensorStatus.SIZE_MISMATCH
            )
        return verified

    def _finish_complete_overwrite(deinit self) -> TensorResult[Self] where identical(Self.S, HostStorage):
        # The only completion path for an INTERNAL allocation-only output.
        # Caller has written all elements in one of the checked CPU kernels.
        # This check is performed once outside any elementwise/matrix loop.
        var verified = self.impl.layout.validated_snapshot(self.impl.storage.capacity())
        if not verified.is_ok():
            return TensorResult[Self](verified.status)
        var mapping = verified.payload.value()
        if not mapping.is_exhaustive() or mapping.span_lo != 0:
            return TensorResult[Self](TensorStatus.NOT_EXHAUSTIVE)
        if mapping.shape.numel() != self.impl.storage.capacity():
            return TensorResult[Self](TensorStatus.SIZE_MISMATCH)
        var status = self.impl.storage._mark_complete_overwrite()
        if status != TensorStatus.OK:
            return TensorResult[Self](status)
        return TensorResult[Self](self)

    def checked_layout(self) -> TensorResult[TensorLayout[Self.rank]]:
        # A public, validated COPY of the strided descriptor. No raw pointer,
        # storage mutation or unchecked allocation capacity is exposed.
        return self._checked_current_layout()

    def checked_descriptor(self) -> TensorResult[TensorDescriptor[Self.rank]]:
        # A verified snapshot of the owner's FP32 layout and physical bounds.
        # This never transfers memory or introduces a secondary Tensor type.
        return self.impl.checked_descriptor()

    def checked_shape(self) -> TensorResult[TensorShape[Self.rank]]:
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[TensorShape[Self.rank]](proof.status)
        return TensorResult[TensorShape[Self.rank]](proof.payload.value().shape.copy())

    def checked_storage_offset(self) -> TensorResult[Int]:
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Int](proof.status)
        return TensorResult[Int](proof.payload.value().offset)

    def numel(self) -> Int:
        return self.impl.layout.shape.numel()

    def nbytes(self) -> Int:
        return self.impl.layout.shape.nbytes()

    def checked_numel(self) -> TensorResult[Int]:
        # A mutably exposed descriptor is never a trusted structural proof.
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Int](checked.status)
        return TensorResult[Int](checked.payload.value().shape.numel())

    def checked_nbytes(self) -> TensorResult[Int]:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Int](checked.status)
        # TensorShape.checked has already bounded numel to Int.MAX // 4.
        return TensorResult[Int](checked.payload.value().shape.nbytes())

    def ndim(self) -> Int:
        return Self.rank

    def itemsize(self) -> Int:
        return 4

    def dtype(self) -> DType:
        # Constant metadata query: public elements are always Float32.
        # The Tensor identity contains no runtime dtype discriminator.
        return DType.float32

    def is_empty(self) -> Bool:
        # Logical emptiness depends on extents, not the mutable cached numel.
        # The checked_shape()/checked_numel() methods report invalid metadata.
        return self.impl.layout.shape.is_empty()

    def is_scalar(self) -> Bool:
        return Self.rank == 0

    def storage_offset(self) -> Int:
        return self.impl.layout.offset

    def is_c_contiguous(self) -> Bool:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return False
        return checked.payload.value().is_c_contiguous()

    def is_f_contiguous(self) -> Bool:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return False
        return checked.payload.value().is_f_contiguous()

    def is_unique(self) -> Bool:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return False
        return checked.payload.value().is_proven_unique()

    def internal_overlap_kind(self) -> InternalOverlapKind:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return InternalOverlapKind.UNKNOWN
        return checked.payload.value().overlap

    def is_exhaustive(self) -> Bool:
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return False
        return checked.payload.value().is_exhaustive()

    def dim(self, axis: Int) -> TensorResult[Int]:
        # A checked metadata query must not bypass the validated descriptor.
        # This has no effect on the unchecked, AOT-specialized element loops.
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Int](proof.status)
        return proof.payload.value().shape.dim(axis)

    def stride(self, axis: Int) -> TensorResult[Int]:
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Int](proof.status)
        var a = axis
        if a < 0:
            if a < -Self.rank:
                return TensorResult[Int](TensorStatus.INVALID_AXIS)
            a += Self.rank
        if a >= Self.rank:
            return TensorResult[Int](TensorStatus.INVALID_AXIS)
        return TensorResult[Int](proof.payload.value().strides.get(a))

    def is_contiguous(self, order: ContiguousOrder) -> Bool:
        # Contiguous C and F properties are independent; rank-0 and some
        # degenerate tensors satisfy both. Unknown order never succeeds.
        if order == ContiguousOrder.C:
            return self.is_c_contiguous()
        if order == ContiguousOrder.F:
            return self.is_f_contiguous()
        return False

    def is_host_accessible(self) -> Bool:
        # Device-local Tensor remains a Tensor, but CPU scalar access is
        # unavailable until an explicit transfer or target-native lowering.
        return self.impl.storage.is_host_accessible()

    def storage_capacity(self) -> Int:
        # Capacity is physical logical-storage capacity, not tensor numel.
        return self.impl.storage.capacity()

    def get(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Float32]:
        # A checked scalar boundary must not trust mutable/cached metadata.
        # Bulk kernels use validated packed accessors and avoid this cost.
        var checked_layout = self._checked_current_layout()
        if not checked_layout.is_ok():
            return TensorResult[Float32](checked_layout.status)
        var pos = checked_layout.payload.value()._offset_from_validated(indices)
        if not pos.is_ok():
            return TensorResult[Float32](pos.status)
        var at = pos.payload.value()
        if at < 0 or at >= self.impl.storage.capacity():
            return TensorResult[Float32](TensorStatus.OUT_OF_BOUNDS)
        return self.impl.storage.read_host(at)

    def set(mut self, indices: StaticRankArray[Self.rank], value: Float32) -> TensorStatus where conforms_to(Self.S, MutableTensorStorage):
        # A scalar indexed write still requires a real host owner and a
        # validated, unique mapping.  Validate the physical index BEFORE
        # attempting poisoned-owner recovery; a bad index must not mutate an
        # otherwise recoverable allocation.
        if not self.impl.storage.is_host_accessible():
            return TensorStatus.BACKEND_UNAVAILABLE
        var checked_layout = self.impl.layout.validated_snapshot(
            self.impl.storage.capacity()
        )
        if not checked_layout.is_ok():
            return checked_layout.status
        var safe_layout = checked_layout.payload.value()
        if not safe_layout.is_proven_unique():
            return TensorStatus.OVERLAPPING
        var pos = safe_layout._offset_from_validated(indices)
        if not pos.is_ok():
            return pos.status
        var at = pos.payload.value()
        if at < 0 or at >= self.impl.storage.capacity():
            return TensorStatus.OUT_OF_BOUNDS
        if not self.impl.storage.is_initialized():
            # Match set_item(): only a uniquely addressable, one-element
            # owner is fully rewritten by this scalar operation.  A scalar
            # subview of an otherwise poisoned owner cannot restore the
            # other bytes and must fail before its first store.
            if (safe_layout.shape.numel() != 1
                or not safe_layout.is_exhaustive()
                or safe_layout.span_lo != 0
                or self.impl.storage.capacity() != 1):
                return TensorStatus.UNINITIALIZED
            var recovered = self.impl.storage.fill_host(value)
            if recovered != TensorStatus.OK:
                comptime if conforms_to(Self.S, HostSpanStorage):
                    self.impl.storage._invalidate_after_failed_write()
            return recovered
        var outcome = self.impl.storage.write_host(at, value)
        if outcome != TensorStatus.OK:
            # A fallible foreign-storage writer can have modified its bytes
            # before reporting failure.  Never keep the old initialized flag
            # on a possibly partial or externally interrupted update.
            comptime if conforms_to(Self.S, HostSpanStorage):
                self.impl.storage._invalidate_after_failed_write()
        return outcome

    def data_ptr(
        ref self
    ) -> TensorResult[Pointer[Float32, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        # Borrow the pointer at the logical origin, not the allocation base.
        # Its origin ties the pointer lifetime to this Tensor owner.
        # This pointer is NOT a claim of contiguity, alignment above Float32,
        # writeability or disjointness. Use layout() to interpret its strides.
        var checked_layout = self._checked_current_layout()
        if not checked_layout.is_ok():
            return TensorResult[Pointer[Float32, origin_of(self.impl.storage)]](
                checked_layout.status
            )
        # An empty tensor has NO logical Float32 element. Its offset may
        # legally be one-past-capacity, which must not be advertised as an
        # addressable data pointer. Use as_c_span() for a zero-length view.
        if checked_layout.payload.value().shape.numel() == 0:
            return TensorResult[Pointer[Float32, origin_of(self.impl.storage)]](
                TensorStatus.NOT_REPRESENTABLE
            )
        var logical_origin = checked_layout.payload.value().offset
        var whole = self.impl.storage.borrow_span()
        return TensorResult[Pointer[Float32, origin_of(self.impl.storage)]](
            whole.unsafe_ptr().unsafe_offset(logical_origin)
        )

    def as_c_span(
        ref self
    ) -> TensorResult[Span[False, Float32, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        # Contiguous logical C-order view without allocation or hidden copy.
        # Rank zero contains exactly one Float32; an empty tensor returns an
        # empty borrowed Span, even though its owner has a physical sentinel.
        var checked_layout = self._checked_current_layout()
        if not checked_layout.is_ok():
            return TensorResult[Span[False, Float32, origin_of(self.impl.storage)]](
                checked_layout.status
            )
        var mapping = checked_layout.payload.value()
        if not mapping.is_c_contiguous() or not mapping.is_exhaustive():
            return TensorResult[Span[False, Float32, origin_of(self.impl.storage)]](
                TensorStatus.NOT_REPRESENTABLE
            )
        var whole = self.impl.storage.borrow_span()
        var begin = mapping.span_lo
        return TensorResult[Span[False, Float32, origin_of(self.impl.storage)]](
            whole[begin:begin + mapping.shape.numel()]
        )

    def as_mut_c_span(
        mut self
    ) -> TensorResult[Span[True, Float32, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        # A mutable contiguous borrow additionally requires injective mapping.
        # Do not expose writable broadcast/overlapping mappings.
        var checked_layout = self._checked_current_layout()
        if not checked_layout.is_ok():
            return TensorResult[Span[True, Float32, origin_of(self.impl.storage)]](
                checked_layout.status
            )
        var mapping = checked_layout.payload.value()
        if not mapping.is_proven_unique():
            return TensorResult[Span[True, Float32, origin_of(self.impl.storage)]](
                TensorStatus.OVERLAPPING
            )
        if not mapping.is_c_contiguous() or not mapping.is_exhaustive():
            return TensorResult[Span[True, Float32, origin_of(self.impl.storage)]](
                TensorStatus.NOT_REPRESENTABLE
            )
        var whole = self.impl.storage.borrow_mut_span()
        var begin = mapping.span_lo
        return TensorResult[Span[True, Float32, origin_of(self.impl.storage)]](
            whole[begin:begin + mapping.shape.numel()]
        )

    # Public Tensor is the primary entry point for borrowed, owner-scoped
    # MAX zero-copy bridges. Static shape and stride must exactly match the
    # checked Tensor layout. These helpers cannot move data or take ownership.


    # Owner-scoped F-order flatten MAX bridges, rank 0..8. The result
    # views contiguous physical FP32 storage in F order (NOT logical C order).
    def as_static_flat_f_layout_tensor[extent: Int](
        ref self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(extent), origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var source = self.view_checked()
        if not source.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout.row_major(extent), origin_of(self.impl.storage)]](source.status)
        return try_as_static_flat_f_layout_tensor[Self.rank, extent](source.payload.take())

    def as_mut_static_flat_f_layout_tensor[extent: Int](
        mut self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(extent), origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var source = self.mut_view_checked()
        if not source.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout.row_major(extent), origin_of(self.impl.storage)]](source.status)
        return try_as_mut_static_flat_f_layout_tensor[Self.rank, extent](source.payload.take())

    def as_static_flat_f_tile_tensor[extent: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var source = self.view_checked()
        if not source.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], origin_of(self.impl.storage)]](source.status)
        return try_as_static_flat_f_tile_tensor[Self.rank, extent](source.payload.take())

    def as_mut_static_flat_f_tile_tensor[extent: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var source = self.mut_view_checked()
        if not source.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], origin_of(self.impl.storage)]](source.status)
        return try_as_mut_static_flat_f_tile_tensor[Self.rank, extent](source.payload.take())

    # Native/MAX AOT layouts with statically supplied exact strides.
    def as_static_strided_layout_tensor_3d[d0: Int, d1: Int, d2: Int, s0: Int, s1: Int, s2: Int](
        ref self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2)), origin_of(self.impl.storage)]] where Self.rank == 3 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view_checked()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2)), origin_of(self.impl.storage)]](borrowed.status)
        return try_as_static_strided_layout_tensor_3d[d0, d1, d2, s0, s1, s2](borrowed.payload.take())

    def as_mut_static_strided_layout_tensor_3d[d0: Int, d1: Int, d2: Int, s0: Int, s1: Int, s2: Int](
        mut self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2)), origin_of(self.impl.storage)]] where Self.rank == 3 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view_checked()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2)), origin_of(self.impl.storage)]](borrowed.status)
        return try_as_mut_static_strided_layout_tensor_3d[d0, d1, d2, s0, s1, s2](borrowed.payload.take())

    def as_static_strided_tile_tensor_3d[d0: Int, d1: Int, d2: Int, s0: Int, s1: Int, s2: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2))], origin_of(self.impl.storage)]] where Self.rank == 3 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view_checked()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2))], origin_of(self.impl.storage)]](borrowed.status)
        return try_as_static_strided_tile_tensor_3d[d0, d1, d2, s0, s1, s2](borrowed.payload.take())

    def as_mut_static_strided_tile_tensor_3d[d0: Int, d1: Int, d2: Int, s0: Int, s1: Int, s2: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2))], origin_of(self.impl.storage)]] where Self.rank == 3 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view_checked()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2))], origin_of(self.impl.storage)]](borrowed.status)
        return try_as_mut_static_strided_tile_tensor_3d[d0, d1, d2, s0, s1, s2](borrowed.payload.take())

    def as_static_strided_layout_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, s0: Int, s1: Int, s2: Int, s3: Int](
        ref self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3)), origin_of(self.impl.storage)]] where Self.rank == 4 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view_checked()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3)), origin_of(self.impl.storage)]](borrowed.status)
        return try_as_static_strided_layout_tensor_4d[d0, d1, d2, d3, s0, s1, s2, s3](borrowed.payload.take())

    def as_mut_static_strided_layout_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, s0: Int, s1: Int, s2: Int, s3: Int](
        mut self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3)), origin_of(self.impl.storage)]] where Self.rank == 4 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view_checked()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3)), origin_of(self.impl.storage)]](borrowed.status)
        return try_as_mut_static_strided_layout_tensor_4d[d0, d1, d2, d3, s0, s1, s2, s3](borrowed.payload.take())

    def as_static_strided_tile_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, s0: Int, s1: Int, s2: Int, s3: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3))], origin_of(self.impl.storage)]] where Self.rank == 4 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view_checked()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3))], origin_of(self.impl.storage)]](borrowed.status)
        return try_as_static_strided_tile_tensor_4d[d0, d1, d2, d3, s0, s1, s2, s3](borrowed.payload.take())

    def as_mut_static_strided_tile_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, s0: Int, s1: Int, s2: Int, s3: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3))], origin_of(self.impl.storage)]] where Self.rank == 4 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view_checked()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3))], origin_of(self.impl.storage)]](borrowed.status)
        return try_as_mut_static_strided_tile_tensor_4d[d0, d1, d2, d3, s0, s1, s2, s3](borrowed.payload.take())

    def as_static_strided_layout_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int](
        ref self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4)), origin_of(self.impl.storage)]] where Self.rank == 5 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view_checked()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4)), origin_of(self.impl.storage)]](borrowed.status)
        return try_as_static_strided_layout_tensor_5d[d0, d1, d2, d3, d4, s0, s1, s2, s3, s4](borrowed.payload.take())

    def as_mut_static_strided_layout_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int](
        mut self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4)), origin_of(self.impl.storage)]] where Self.rank == 5 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view_checked()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4)), origin_of(self.impl.storage)]](borrowed.status)
        return try_as_mut_static_strided_layout_tensor_5d[d0, d1, d2, d3, d4, s0, s1, s2, s3, s4](borrowed.payload.take())

    def as_static_strided_tile_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4))], origin_of(self.impl.storage)]] where Self.rank == 5 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view_checked()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4))], origin_of(self.impl.storage)]](borrowed.status)
        return try_as_static_strided_tile_tensor_5d[d0, d1, d2, d3, d4, s0, s1, s2, s3, s4](borrowed.payload.take())

    def as_mut_static_strided_tile_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4))], origin_of(self.impl.storage)]] where Self.rank == 5 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view_checked()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4))], origin_of(self.impl.storage)]](borrowed.status)
        return try_as_mut_static_strided_tile_tensor_5d[d0, d1, d2, d3, d4, s0, s1, s2, s3, s4](borrowed.payload.take())

    def as_static_strided_layout_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int](
        ref self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5)), origin_of(self.impl.storage)]] where Self.rank == 6 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view_checked()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5)), origin_of(self.impl.storage)]](borrowed.status)
        return try_as_static_strided_layout_tensor_6d[d0, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5](borrowed.payload.take())

    def as_mut_static_strided_layout_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int](
        mut self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5)), origin_of(self.impl.storage)]] where Self.rank == 6 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view_checked()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5)), origin_of(self.impl.storage)]](borrowed.status)
        return try_as_mut_static_strided_layout_tensor_6d[d0, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5](borrowed.payload.take())

    def as_static_strided_tile_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5))], origin_of(self.impl.storage)]] where Self.rank == 6 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view_checked()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5))], origin_of(self.impl.storage)]](borrowed.status)
        return try_as_static_strided_tile_tensor_6d[d0, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5](borrowed.payload.take())

    def as_mut_static_strided_tile_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5))], origin_of(self.impl.storage)]] where Self.rank == 6 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view_checked()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5))], origin_of(self.impl.storage)]](borrowed.status)
        return try_as_mut_static_strided_tile_tensor_6d[d0, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5](borrowed.payload.take())

    def as_static_strided_layout_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int](
        ref self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6)), origin_of(self.impl.storage)]] where Self.rank == 7 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view_checked()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6)), origin_of(self.impl.storage)]](borrowed.status)
        return try_as_static_strided_layout_tensor_7d[d0, d1, d2, d3, d4, d5, d6, s0, s1, s2, s3, s4, s5, s6](borrowed.payload.take())

    def as_mut_static_strided_layout_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int](
        mut self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6)), origin_of(self.impl.storage)]] where Self.rank == 7 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view_checked()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6)), origin_of(self.impl.storage)]](borrowed.status)
        return try_as_mut_static_strided_layout_tensor_7d[d0, d1, d2, d3, d4, d5, d6, s0, s1, s2, s3, s4, s5, s6](borrowed.payload.take())

    def as_static_strided_tile_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6))], origin_of(self.impl.storage)]] where Self.rank == 7 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view_checked()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6))], origin_of(self.impl.storage)]](borrowed.status)
        return try_as_static_strided_tile_tensor_7d[d0, d1, d2, d3, d4, d5, d6, s0, s1, s2, s3, s4, s5, s6](borrowed.payload.take())

    def as_mut_static_strided_tile_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6))], origin_of(self.impl.storage)]] where Self.rank == 7 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view_checked()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6))], origin_of(self.impl.storage)]](borrowed.status)
        return try_as_mut_static_strided_tile_tensor_7d[d0, d1, d2, d3, d4, d5, d6, s0, s1, s2, s3, s4, s5, s6](borrowed.payload.take())

    def as_static_strided_layout_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int, s7: Int](
        ref self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7)), origin_of(self.impl.storage)]] where Self.rank == 8 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view_checked()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7)), origin_of(self.impl.storage)]](borrowed.status)
        return try_as_static_strided_layout_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6, s7](borrowed.payload.take())

    def as_mut_static_strided_layout_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int, s7: Int](
        mut self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7)), origin_of(self.impl.storage)]] where Self.rank == 8 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view_checked()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7)), origin_of(self.impl.storage)]](borrowed.status)
        return try_as_mut_static_strided_layout_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6, s7](borrowed.payload.take())

    def as_static_strided_tile_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int, s7: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7))], origin_of(self.impl.storage)]] where Self.rank == 8 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view_checked()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7))], origin_of(self.impl.storage)]](borrowed.status)
        return try_as_static_strided_tile_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6, s7](borrowed.payload.take())

    def as_mut_static_strided_tile_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int, s7: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7))], origin_of(self.impl.storage)]] where Self.rank == 8 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view_checked()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7))], origin_of(self.impl.storage)]](borrowed.status)
        return try_as_mut_static_strided_tile_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6, s7](borrowed.payload.take())


    def as_static_strided_layout_tensor_1d[extent: Int, step: Int](
        ref self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(extent), IntTuple(step)), origin_of(self.impl.storage)]] where Self.rank == 1 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(extent), IntTuple(step)), origin_of(self.impl.storage)]](
                borrowed.status
            )
        return try_as_static_strided_layout_tensor_1d[extent, step](borrowed.payload.take())


    def as_mut_static_strided_layout_tensor_1d[extent: Int, step: Int](
        mut self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(extent), IntTuple(step)), origin_of(self.impl.storage)]] where Self.rank == 1 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(extent), IntTuple(step)), origin_of(self.impl.storage)]](
                borrowed.status
            )
        return try_as_mut_static_strided_layout_tensor_1d[extent, step](borrowed.payload.take())


    def as_static_strided_tile_tensor_1d[extent: Int, step: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(extent), IntTuple(step))], origin_of(self.impl.storage)]] where Self.rank == 1 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(extent), IntTuple(step))], origin_of(self.impl.storage)]](
                borrowed.status
            )
        return try_as_static_strided_tile_tensor_1d[extent, step](borrowed.payload.take())


    def as_mut_static_strided_tile_tensor_1d[extent: Int, step: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(extent), IntTuple(step))], origin_of(self.impl.storage)]] where Self.rank == 1 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(extent), IntTuple(step))], origin_of(self.impl.storage)]](
                borrowed.status
            )
        return try_as_mut_static_strided_tile_tensor_1d[extent, step](borrowed.payload.take())


    def as_static_strided_layout_tensor_2d[rows: Int, cols: Int, row_step: Int, col_step: Int](
        ref self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(rows, cols), IntTuple(row_step, col_step)), origin_of(self.impl.storage)]] where Self.rank == 2 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(rows, cols), IntTuple(row_step, col_step)), origin_of(self.impl.storage)]](
                borrowed.status
            )
        return try_as_static_strided_layout_tensor_2d[rows, cols, row_step, col_step](borrowed.payload.take())


    def as_mut_static_strided_layout_tensor_2d[rows: Int, cols: Int, row_step: Int, col_step: Int](
        mut self,
    ) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(rows, cols), IntTuple(row_step, col_step)), origin_of(self.impl.storage)]] where Self.rank == 2 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view()
        if not borrowed.is_ok():
            return TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(rows, cols), IntTuple(row_step, col_step)), origin_of(self.impl.storage)]](
                borrowed.status
            )
        return try_as_mut_static_strided_layout_tensor_2d[rows, cols, row_step, col_step](borrowed.payload.take())


    def as_static_strided_tile_tensor_2d[rows: Int, cols: Int, row_step: Int, col_step: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(rows, cols), IntTuple(row_step, col_step))], origin_of(self.impl.storage)]] where Self.rank == 2 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.view()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(rows, cols), IntTuple(row_step, col_step))], origin_of(self.impl.storage)]](
                borrowed.status
            )
        return try_as_static_strided_tile_tensor_2d[rows, cols, row_step, col_step](borrowed.payload.take())


    def as_mut_static_strided_tile_tensor_2d[rows: Int, cols: Int, row_step: Int, col_step: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(rows, cols), IntTuple(row_step, col_step))], origin_of(self.impl.storage)]] where Self.rank == 2 and conforms_to(Self.S, HostSpanStorage):
        var borrowed = self.mut_view()
        if not borrowed.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(rows, cols), IntTuple(row_step, col_step))], origin_of(self.impl.storage)]](
                borrowed.status
            )
        return try_as_mut_static_strided_tile_tensor_2d[rows, cols, row_step, col_step](borrowed.payload.take())

    # Rank-preserving static MAX TileTensor facade (same owning Tensor).
    def as_c_tile_tensor_1d[extent: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], origin_of(self.impl.storage)]] where Self.rank == 1 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_static_c_tile_tensor_1d[extent](proof.payload.take())

    def as_mut_c_tile_tensor_1d[extent: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], origin_of(self.impl.storage)]] where Self.rank == 1 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_mut_static_c_tile_tensor_1d[extent](proof.payload.take())

    def as_c_tile_tensor_3d[d0: Int, d1: Int, d2: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2)], origin_of(self.impl.storage)]] where Self.rank == 3 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_static_c_tile_tensor_3d[d0, d1, d2](proof.payload.take())

    def as_mut_c_tile_tensor_3d[d0: Int, d1: Int, d2: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2)], origin_of(self.impl.storage)]] where Self.rank == 3 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_mut_static_c_tile_tensor_3d[d0, d1, d2](proof.payload.take())

    def as_c_tile_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3)], origin_of(self.impl.storage)]] where Self.rank == 4 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_static_c_tile_tensor_4d[d0, d1, d2, d3](proof.payload.take())

    def as_mut_c_tile_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3)], origin_of(self.impl.storage)]] where Self.rank == 4 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_mut_static_c_tile_tensor_4d[d0, d1, d2, d3](proof.payload.take())

    def as_c_tile_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4)], origin_of(self.impl.storage)]] where Self.rank == 5 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_static_c_tile_tensor_5d[d0, d1, d2, d3, d4](proof.payload.take())

    def as_mut_c_tile_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4)], origin_of(self.impl.storage)]] where Self.rank == 5 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_mut_static_c_tile_tensor_5d[d0, d1, d2, d3, d4](proof.payload.take())

    def as_c_tile_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5)], origin_of(self.impl.storage)]] where Self.rank == 6 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_static_c_tile_tensor_6d[d0, d1, d2, d3, d4, d5](proof.payload.take())

    def as_mut_c_tile_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5)], origin_of(self.impl.storage)]] where Self.rank == 6 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_mut_static_c_tile_tensor_6d[d0, d1, d2, d3, d4, d5](proof.payload.take())

    def as_c_tile_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6)], origin_of(self.impl.storage)]] where Self.rank == 7 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_static_c_tile_tensor_7d[d0, d1, d2, d3, d4, d5, d6](proof.payload.take())

    def as_mut_c_tile_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6)], origin_of(self.impl.storage)]] where Self.rank == 7 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_mut_static_c_tile_tensor_7d[d0, d1, d2, d3, d4, d5, d6](proof.payload.take())

    def as_c_tile_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6, d7)], origin_of(self.impl.storage)]] where Self.rank == 8 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6, d7)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_static_c_tile_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7](proof.payload.take())

    def as_mut_c_tile_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6, d7)], origin_of(self.impl.storage)]] where Self.rank == 8 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6, d7)], origin_of(self.impl.storage)]](
                proof.status
            )
        return try_as_mut_static_c_tile_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7](proof.payload.take())

    # Rank-preserving F-order MAX TileTensor bridges, preserving physical strides.
    def as_f_tile_tensor_3d[d0: Int, d1: Int, d2: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2)], origin_of(self.impl.storage)]] where Self.rank == 3 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2)], origin_of(self.impl.storage)]](proof.status)
        return try_as_static_f_tile_tensor_3d[d0, d1, d2](proof.payload.take())

    def as_mut_f_tile_tensor_3d[d0: Int, d1: Int, d2: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2)], origin_of(self.impl.storage)]] where Self.rank == 3 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2)], origin_of(self.impl.storage)]](proof.status)
        return try_as_mut_static_f_tile_tensor_3d[d0, d1, d2](proof.payload.take())

    def as_f_tile_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3)], origin_of(self.impl.storage)]] where Self.rank == 4 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3)], origin_of(self.impl.storage)]](proof.status)
        return try_as_static_f_tile_tensor_4d[d0, d1, d2, d3](proof.payload.take())

    def as_mut_f_tile_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3)], origin_of(self.impl.storage)]] where Self.rank == 4 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3)], origin_of(self.impl.storage)]](proof.status)
        return try_as_mut_static_f_tile_tensor_4d[d0, d1, d2, d3](proof.payload.take())

    def as_f_tile_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4)], origin_of(self.impl.storage)]] where Self.rank == 5 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4)], origin_of(self.impl.storage)]](proof.status)
        return try_as_static_f_tile_tensor_5d[d0, d1, d2, d3, d4](proof.payload.take())

    def as_mut_f_tile_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4)], origin_of(self.impl.storage)]] where Self.rank == 5 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4)], origin_of(self.impl.storage)]](proof.status)
        return try_as_mut_static_f_tile_tensor_5d[d0, d1, d2, d3, d4](proof.payload.take())

    def as_f_tile_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5)], origin_of(self.impl.storage)]] where Self.rank == 6 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5)], origin_of(self.impl.storage)]](proof.status)
        return try_as_static_f_tile_tensor_6d[d0, d1, d2, d3, d4, d5](proof.payload.take())

    def as_mut_f_tile_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5)], origin_of(self.impl.storage)]] where Self.rank == 6 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5)], origin_of(self.impl.storage)]](proof.status)
        return try_as_mut_static_f_tile_tensor_6d[d0, d1, d2, d3, d4, d5](proof.payload.take())

    def as_f_tile_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6)], origin_of(self.impl.storage)]] where Self.rank == 7 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6)], origin_of(self.impl.storage)]](proof.status)
        return try_as_static_f_tile_tensor_7d[d0, d1, d2, d3, d4, d5, d6](proof.payload.take())

    def as_mut_f_tile_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6)], origin_of(self.impl.storage)]] where Self.rank == 7 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6)], origin_of(self.impl.storage)]](proof.status)
        return try_as_mut_static_f_tile_tensor_7d[d0, d1, d2, d3, d4, d5, d6](proof.payload.take())

    def as_f_tile_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6, d7)], origin_of(self.impl.storage)]] where Self.rank == 8 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6, d7)], origin_of(self.impl.storage)]](proof.status)
        return try_as_static_f_tile_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7](proof.payload.take())

    def as_mut_f_tile_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6, d7)], origin_of(self.impl.storage)]] where Self.rank == 8 and conforms_to(Self.S, HostSpanStorage):
        var proof = self.mut_view()
        if not proof.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6, d7)], origin_of(self.impl.storage)]](proof.status)
        return try_as_mut_static_f_tile_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7](proof.payload.take())

    # MAX views are thin, zero-copy delegates on the canonical Tensor owner.
    # The general flattened form covers rank 0..8; the 2-D form preserves
    # matrix geometry for tile-native kernels. No AOT dispatch or implicit
    # materialization is performed by these adapters.
    def as_flat_c_tile_tensor[extent: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var source = self.view()
        if not source.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], origin_of(self.impl.storage)]](
                source.status
            )
        return try_as_static_flat_c_tile_tensor[Self.rank, extent](
            source.payload.take()
        )

    def as_mut_flat_c_tile_tensor[extent: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var source = self.mut_view()
        if not source.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], origin_of(self.impl.storage)]](
                source.status
            )
        return try_as_mut_static_flat_c_tile_tensor[Self.rank, extent](
            source.payload.take()
        )

    def as_c_tile_tensor_2d[rows: Int, cols: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(rows, cols)], origin_of(self.impl.storage)]] where Self.rank == 2 and conforms_to(Self.S, HostSpanStorage):
        var source = self.view()
        if not source.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(rows, cols)], origin_of(self.impl.storage)]](
                source.status
            )
        return try_as_static_c_tile_tensor_2d[rows, cols](source.payload.take())

    def as_mut_c_tile_tensor_2d[rows: Int, cols: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(rows, cols)], origin_of(self.impl.storage)]] where Self.rank == 2 and conforms_to(Self.S, HostSpanStorage):
        var source = self.mut_view()
        if not source.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(rows, cols)], origin_of(self.impl.storage)]](
                source.status
            )
        return try_as_mut_static_c_tile_tensor_2d[rows, cols](
            source.payload.take()
        )

    def as_f_tile_tensor_2d[rows: Int, cols: Int](
        ref self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(rows, cols)], origin_of(self.impl.storage)]] where Self.rank == 2 and conforms_to(Self.S, HostSpanStorage):
        var source = self.view()
        if not source.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(rows, cols)], origin_of(self.impl.storage)]](
                source.status
            )
        return try_as_static_f_tile_tensor_2d[rows, cols](source.payload.take())

    def as_mut_f_tile_tensor_2d[rows: Int, cols: Int](
        mut self,
    ) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(rows, cols)], origin_of(self.impl.storage)]] where Self.rank == 2 and conforms_to(Self.S, HostSpanStorage):
        var source = self.mut_view()
        if not source.is_ok():
            return TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(rows, cols)], origin_of(self.impl.storage)]](
                source.status
            )
        return try_as_mut_static_f_tile_tensor_2d[rows, cols](
            source.payload.take()
        )

    def as_f_span(
        ref self
    ) -> TensorResult[Span[False, Float32, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        # Physical F-order Span: no materialization and no claim that its
        # iteration order matches C-order logical traversal.
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Span[False, Float32, origin_of(self.impl.storage)]](proof.status)
        var layout = proof.payload.value()
        if not layout.is_f_contiguous() or not layout.is_exhaustive():
            return TensorResult[Span[False, Float32, origin_of(self.impl.storage)]](TensorStatus.NOT_REPRESENTABLE)
        var storage = self.impl.storage.borrow_span()
        var begin = layout.span_lo
        return TensorResult[Span[False, Float32, origin_of(self.impl.storage)]](
            storage[begin:begin + layout.shape.numel()]
        )

    def as_mut_f_span(
        mut self
    ) -> TensorResult[Span[True, Float32, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Span[True, Float32, origin_of(self.impl.storage)]](proof.status)
        var layout = proof.payload.value()
        if not layout.is_proven_unique():
            return TensorResult[Span[True, Float32, origin_of(self.impl.storage)]](TensorStatus.OVERLAPPING)
        if not layout.is_f_contiguous() or not layout.is_exhaustive():
            return TensorResult[Span[True, Float32, origin_of(self.impl.storage)]](TensorStatus.NOT_REPRESENTABLE)
        var storage = self.impl.storage.borrow_mut_span()
        var begin = layout.span_lo
        return TensorResult[Span[True, Float32, origin_of(self.impl.storage)]](
            storage[begin:begin + layout.shape.numel()]
        )

    def item(self) -> TensorResult[Float32]:
        # Validate once, then use the unique physical origin of a one-element
        # layout. This avoids building zero coordinates and revalidating the
        # owner in get() on the hot scalar boundary (including rank 0).
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Float32](checked.status)
        var safe = checked.payload.value()
        if safe.shape.numel() != 1:
            return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
        if not self.impl.storage.is_host_accessible():
            return TensorResult[Float32](TensorStatus.BACKEND_UNAVAILABLE)
        return self.impl.storage.read_host(safe.offset)

    def set_item(mut self, value: Float32) -> TensorStatus where conforms_to(Self.S, MutableTensorStorage):
        # A one-element *owner* can be safely restored from UNINITIALIZED by
        # overwriting its only logical FP32.  A one-element SUBVIEW cannot:
        # other owner bytes might still be poisoned, so preserve the original
        # full-readability precondition in that case.
        var checked = self.impl.layout.validated_snapshot(
            self.impl.storage.capacity()
        )
        if not checked.is_ok():
            return checked.status
        var safe = checked.payload.value()
        if safe.shape.numel() != 1:
            return TensorStatus.SIZE_MISMATCH
        if not safe.is_proven_unique():
            return TensorStatus.OVERLAPPING
        if not self.impl.storage.is_host_accessible():
            return TensorStatus.BACKEND_UNAVAILABLE
        if (safe.is_exhaustive() and safe.span_lo == 0
            and self.impl.storage.capacity() == 1):
            var recovered = self.impl.storage.fill_host(value)
            if recovered != TensorStatus.OK:
                comptime if conforms_to(Self.S, HostSpanStorage):
                    self.impl.storage._invalidate_after_failed_write()
            return recovered
        if not self.impl.storage.is_initialized():
            return TensorStatus.UNINITIALIZED
        var result = self.impl.storage.write_host(safe.offset, value)
        if result != TensorStatus.OK:
            comptime if conforms_to(Self.S, HostSpanStorage):
                self.impl.storage._invalidate_after_failed_write()
        return result

    def fill(mut self, value: Float32) -> TensorStatus where conforms_to(Self.S, MutableTensorStorage):
        if not self.impl.storage.is_host_accessible():
            # A device-local tensor needs an explicit GPU fill lowering.
            # Even a zero-element device tensor must not pretend CPU success.
            return TensorStatus.BACKEND_UNAVAILABLE
        # A full overwrite is a recovery operation, not a read.  It must be
        # possible to fill a host owner invalidated by an earlier failed CPU
        # or Hybrid writer.  Unlike _checked_current_layout(), a structural
        # snapshot proves bounds without demanding OLD payload readability.
        var checked_layout = self.impl.layout.validated_snapshot(
            self.impl.storage.capacity()
        )
        if not checked_layout.is_ok():
            return checked_layout.status
        var safe_layout = checked_layout.payload.value()
        if not safe_layout.is_proven_unique():
            return TensorStatus.OVERLAPPING
        # The storage capability owns publication of the initialized flag.
        # A unique/exhaustive mapping covering the entire allocation can
        # delegate to fill_host() even if logical traversal is F-order or
        # reversed.  Every physical Float32 receives the SAME value, so this
        # loses no logical semantics and does not require a staging tensor.
        if (safe_layout.is_exhaustive() and safe_layout.span_lo == 0
            and safe_layout.shape.numel() == self.impl.storage.capacity()):
            var outcome = self.impl.storage.fill_host(value)
            if outcome != TensorStatus.OK:
                comptime if conforms_to(Self.S, HostSpanStorage):
                    self.impl.storage._invalidate_after_failed_write()
            return outcome
        # A partial write cannot restore an unreadable owner's certificate.
        # Reject it BEFORE the first store; explicit full-owner writes above
        # are the safe recovery path. This includes empty subviews of a
        # poisoned nonempty allocation.
        if not self.impl.storage.is_initialized():
            return TensorStatus.UNINITIALIZED
        if safe_layout.shape.numel() == 0:
            return TensorStatus.OK
        # Only HostStorage promises a borrowed contiguous Span. Other
        # MutableTensorStorage implementations satisfy the scalar trait API,
        # not the HostStorage-specific Span API.
        comptime if identical(Self.S, HostStorage):
            if safe_layout.is_exhaustive():
                var data = self.impl.storage.borrow_mut_span()
                var first = safe_layout.span_lo
                var dense = data[first:first + safe_layout.shape.numel()]
                _fill_fp32_contiguous[simd_width_of[DType.float32]()](dense, value)
                return TensorStatus.OK
        # HostStorage has a known, previously validated physical span. Walk
        # an irregular but unique layout without allocating TensorResult or
        # revalidating every element. Borrowing ties the data to its owner.
        comptime if identical(Self.S, HostStorage):
            var host_data = self.impl.storage.borrow_mut_span()
            var host_cursor = NDIndexIterator[Self.rank](safe_layout.shape)
            while not host_cursor.done():
                var at = safe_layout.physical_offset_unchecked(host_cursor.coordinates)
                host_data[at] = value
                host_cursor.advance()
            return TensorStatus.OK
        # Generic host-accessible storage has already passed shape, capacity,
        # overlap and mutability checks. Avoid revalidating metadata at every
        # element; the trait write still enforces storage-specific rules.
        var cursor = NDIndexIterator[Self.rank](safe_layout.shape)
        while not cursor.done():
            var at = safe_layout.physical_offset_unchecked(cursor.coordinates)
            var outcome = self.impl.storage.write_host(at, value)
            if outcome != TensorStatus.OK:
                # The status may come after writes to earlier coordinates.
                # Revoke the whole-owner readable certificate where the
                # storage capability provides the required poison operation.
                comptime if conforms_to(Self.S, HostSpanStorage):
                    self.impl.storage._invalidate_after_failed_write()
                return outcome
            cursor.advance()
        return TensorStatus.OK

    def zero(mut self) -> TensorStatus where conforms_to(Self.S, MutableTensorStorage):
        return self.fill(0.0)


    # Public numerical facade: the owner's name remains Tensor. Operations
    # delegate to fixed-FP32 CPU kernels and one checked shape/layout boundary;
    # ops/arithmetic reuses these methods for owned host/host operands.
    # High-level unary functions are methods of the primary Tensor type.
    # The opcode is a compile-time parameter, so the scalar inner loop has
    # no operation switch, no dtype branch, and no runtime virtual dispatch.
    def unary_host[operation: Int](
        ref self
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        comptime assert operation >= 0 and operation <= 8, "invalid unary FP32 operation"
        var verified = self._checked_current_layout()
        if not verified.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](verified.status)
        var safe_layout = verified.payload.value()
        var allocated = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(safe_layout.shape)
        if not allocated.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](allocated.status)
        var output = allocated.payload.take()
        var source = self.impl.storage.borrow_span()
        var destination = output.impl.storage.borrow_mut_span()
        if safe_layout.is_c_contiguous() and safe_layout.is_exhaustive():
            var start = safe_layout.span_lo
            var run = source[start:start + self.numel()]
            var unary_status = cpu_unary_contiguous_fp32[
                operation, simd_width_of[DType.float32](),
            ](run, destination)
            if unary_status != TensorStatus.OK:
                return TensorResult[Tensor[Self.rank, HostStorage]](unary_status)
            return output._finish_complete_overwrite()
        var cursor = NDIndexIterator[Self.rank](safe_layout.shape)
        var linear = 0
        while not cursor.done():
            var at = safe_layout.physical_offset_unchecked(cursor.coordinates)
            destination[linear] = cpu_unary_scalar_fp32[operation](source[at])
            linear += 1
            cursor.advance()
        return output._finish_complete_overwrite()

    def abs(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.unary_host[1]()

    def sqrt(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.unary_host[2]()

    def exp(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.unary_host[3]()

    def log(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.unary_host[4]()

    def sin(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.unary_host[5]()

    def cos(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.unary_host[6]()

    def tanh(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.unary_host[7]()

    def reciprocal(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.unary_host[8]()

    def _scalar_binary_host[operation: Int](
        ref self, rhs: Float32
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        comptime assert operation >= 0 and operation <= 3, "invalid FP32 scalar arithmetic opcode"
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](checked.status)
        var mapping = checked.payload.value()
        var allocated = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(mapping.shape)
        if not allocated.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](allocated.status)
        var output = allocated.payload.take()
        var src = self.impl.storage.borrow_span()
        var dst = output.impl.storage.borrow_mut_span()
        if mapping.is_c_contiguous() and mapping.is_exhaustive():
            var contiguous = src[mapping.span_lo:mapping.span_lo + mapping.shape.numel()]
            var scalar_status = TensorStatus.UNSUPPORTED
            if operation == 0:
                scalar_status = cpu_shift_contiguous[
                    simd_width_of[DType.float32](),
                ](dst, contiguous, rhs)
            elif operation == 1:
                scalar_status = cpu_sub_scalar_contiguous[
                    simd_width_of[DType.float32](),
                ](dst, contiguous, rhs)
            elif operation == 2:
                scalar_status = cpu_scale_contiguous[
                    simd_width_of[DType.float32](),
                ](dst, contiguous, rhs)
            else:
                scalar_status = cpu_div_scalar_contiguous[
                    simd_width_of[DType.float32](),
                ](dst, contiguous, rhs)
            if scalar_status != TensorStatus.OK:
                return TensorResult[Tensor[Self.rank, HostStorage]](scalar_status)
            return output._finish_complete_overwrite()
        var cursor = NDIndexIterator[Self.rank](mapping.shape)
        var linear = 0
        while not cursor.done():
            var x = src[mapping.physical_offset_unchecked(cursor.coordinates)]
            if operation == 0:
                dst[linear] = x + rhs
            elif operation == 1:
                dst[linear] = x - rhs
            elif operation == 2:
                dst[linear] = x * rhs
            else:
                dst[linear] = x / rhs
            linear += 1
            cursor.advance()
        return output._finish_complete_overwrite()

    def add_scalar(ref self, value: Float32) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._scalar_binary_host[0](value)

    def sub_scalar(ref self, value: Float32) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._scalar_binary_host[1](value)

    def mul_scalar(ref self, value: Float32) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._scalar_binary_host[2](value)

    def div_scalar(ref self, value: Float32) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._scalar_binary_host[3](value)

    def _binary_host[operation: Int](
        ref self, other: Tensor[Self.rank, HostStorage]
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        comptime assert operation >= 0 and operation <= 5, "invalid FP32 binary arithmetic opcode"
        var lhs_proof = self._checked_current_layout()
        if not lhs_proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](lhs_proof.status)
        var rhs_proof = other._checked_current_layout()
        if not rhs_proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](rhs_proof.status)
        var a_layout = lhs_proof.payload.value()
        var b_layout = rhs_proof.payload.value()
        var dims = StaticRankArray[Self.rank](0)
        comptime for axis in range(Self.rank):
            var da = a_layout.shape.dims.get(axis)
            var db = b_layout.shape.dims.get(axis)
            if da == db:
                dims.set(axis, da)
            elif da == 1:
                dims.set(axis, db)
            elif db == 1:
                dims.set(axis, da)
            else:
                return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var shape = TensorShape[Self.rank].checked(dims)
        if not shape.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](shape.status)
        var allocated = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(shape.payload.value())
        if not allocated.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](allocated.status)
        var result = allocated.payload.take()
        var lhs = self.impl.storage.borrow_span()
        var rhs = other.impl.storage.borrow_span()
        var dst = result.impl.storage.borrow_mut_span()
        if a_layout.shape.same_extents(b_layout.shape):
            if a_layout.is_c_contiguous() and b_layout.is_c_contiguous():
                if a_layout.is_exhaustive() and b_layout.is_exhaustive():
                    var aa = lhs[a_layout.span_lo:a_layout.span_lo + self.numel()]
                    var bb = rhs[b_layout.span_lo:b_layout.span_lo + other.numel()]
                    var leaf_status = cpu_binary_contiguous_checked[
                        operation, simd_width_of[DType.float32]()
                    ](dst, aa, bb)
                    if leaf_status != TensorStatus.OK:
                        return TensorResult[Tensor[Self.rank, HostStorage]](leaf_status)
                    return result._finish_complete_overwrite()
        # Singleton-axis broadcasting is extremely common. A source with
        # exactly one FP32 element is a scalar even when its STATIC rank > 0;
        # the other operand may be an arbitrary 1..8D C-dense tensor. Eliminate
        # coordinate carry and affine offset multiplication from this hot path.
        var output_count = shape.payload.value().numel()
        if output_count > 1:
            if (a_layout.shape.numel() == 1 and
                b_layout.shape.numel() == output_count and
                b_layout.is_c_contiguous() and b_layout.is_exhaustive()):
                var dense = rhs[b_layout.span_lo:b_layout.span_lo + output_count]
                var leaf_status = cpu_scalar_broadcast_binary_contiguous[
                    operation, True, simd_width_of[DType.float32]()
                ](dst, dense, lhs[a_layout.offset])
                if leaf_status != TensorStatus.OK:
                    return TensorResult[Tensor[Self.rank, HostStorage]](leaf_status)
                return result._finish_complete_overwrite()
            if (b_layout.shape.numel() == 1 and
                a_layout.shape.numel() == output_count and
                a_layout.is_c_contiguous() and a_layout.is_exhaustive()):
                var dense = lhs[a_layout.span_lo:a_layout.span_lo + output_count]
                var leaf_status = cpu_scalar_broadcast_binary_contiguous[
                    operation, False, simd_width_of[DType.float32]()
                ](dst, dense, rhs[b_layout.offset])
                if leaf_status != TensorStatus.OK:
                    return TensorResult[Tensor[Self.rank, HostStorage]](leaf_status)
                return result._finish_complete_overwrite()
        # Lower broadcasting into zero strides once, outside the hot loop.
        # Each output coordinate can then be used directly for both inputs;
        # there is no per-element axis test or temporary index array.
        var a_steps = a_layout.strides.copy()
        var b_steps = b_layout.strides.copy()
        comptime for axis in range(Self.rank):
            if a_layout.shape.dims.get(axis) == 1:
                a_steps.set(axis, 0)
            if b_layout.shape.dims.get(axis) == 1:
                b_steps.set(axis, 0)
        var cursor = NDIndexIterator[Self.rank](shape.payload.value())
        var linear = 0
        while not cursor.done():
            var a_at = a_layout.offset
            var b_at = b_layout.offset
            comptime for axis in range(Self.rank):
                var logical = cursor.coordinates.get(axis)
                a_at += logical * a_steps.get(axis)
                b_at += logical * b_steps.get(axis)
            var x = lhs[a_at]
            var y = rhs[b_at]
            comptime if operation == 0:
                dst[linear] = x + y
            elif operation == 1:
                dst[linear] = x - y
            elif operation == 2:
                dst[linear] = x * y
            elif operation == 3:
                dst[linear] = x / y
            elif operation == 4:
                dst[linear] = fp32_maximum(x, y)
            else:
                dst[linear] = fp32_minimum(x, y)
            linear += 1
            cursor.advance()
        return result._finish_complete_overwrite()

    # Right-aligned FP32 broadcasting is available from the primary Tensor API.
    # The output rank is static, never a runtime dtype/rank dispatch value.
    # All source layout checks and broadcast decisions precede the hot loop.
    def _binary_nd_host[operation: Int, other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[Tensor[out_rank, HostStorage]] where identical(Self.S, HostStorage):
        comptime assert operation >= 0 and operation <= 5, "invalid binary opcode"
        comptime assert other_rank >= 0 and other_rank <= 8, "invalid right operand rank"
        comptime assert out_rank == max(Self.rank, other_rank) and out_rank <= 8, "invalid output rank"
        comptime if other_rank == Self.rank:
            return self._binary_host[operation](other)
        var left_proof = self._checked_current_layout()
        if not left_proof.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](left_proof.status)
        var right_proof = other._checked_current_layout()
        if not right_proof.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](right_proof.status)
        var left_layout = left_proof.payload.value()
        var right_layout = right_proof.payload.value()
        var dims = StaticRankArray[out_rank](1)
        var left_steps = StaticRankArray[out_rank](0)
        var right_steps = StaticRankArray[out_rank](0)
        comptime for i in range(out_rank):
            var da = 1
            var db = 1
            comptime if i >= out_rank - Self.rank:
                da = left_layout.shape.dims.get(i - (out_rank - Self.rank))
            comptime if i >= out_rank - other_rank:
                db = right_layout.shape.dims.get(i - (out_rank - other_rank))
            if da == db:
                dims.set(i, da)
            elif da == 1:
                dims.set(i, db)
            elif db == 1:
                dims.set(i, da)
            else:
                return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var shape = TensorShape[out_rank].checked(dims)
        if not shape.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](shape.status)
        comptime for i in range(Self.rank):
            comptime at = out_rank - Self.rank + i
            if left_layout.shape.dims.get(i) != 1:
                left_steps.set(at, left_layout.strides.get(i))
        comptime for i in range(other_rank):
            comptime at = out_rank - other_rank + i
            if right_layout.shape.dims.get(i) != 1:
                right_steps.set(at, right_layout.strides.get(i))
        var allocated = Tensor[out_rank, HostStorage]._empty_for_complete_overwrite_checked(shape.payload.value())
        if not allocated.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](allocated.status)
        var result = allocated.payload.take()
        var lhs = self.impl.storage.borrow_span()
        var rhs = other.impl.storage.borrow_span()
        var dst = result.impl.storage.borrow_mut_span()
        # For differently ranked operands, implicit leading singleton axes
        # do not change the logical C-order element sequence. If both owners
        # already contain the full output count, use an AOT SIMD binary pass.
        # Only a real singleton-vs-many broadcast needs a dedicated kernel.
        var output_count = shape.payload.value().numel()
        if output_count > 0:
            if (left_layout.shape.numel() == output_count and
                right_layout.shape.numel() == output_count and
                left_layout.is_c_contiguous() and right_layout.is_c_contiguous() and
                left_layout.is_exhaustive() and right_layout.is_exhaustive()):
                var dense_left = lhs[left_layout.span_lo:left_layout.span_lo + output_count]
                var dense_right = rhs[right_layout.span_lo:right_layout.span_lo + output_count]
                var leaf_status = cpu_binary_contiguous_checked[
                    operation, simd_width_of[DType.float32]()
                ](dst, dense_left, dense_right)
                if leaf_status != TensorStatus.OK:
                    return TensorResult[Tensor[out_rank, HostStorage]](leaf_status)
                return result._finish_complete_overwrite()
            if (left_layout.shape.numel() == 1 and
                right_layout.shape.numel() == output_count and
                right_layout.is_c_contiguous() and right_layout.is_exhaustive()):
                var dense = rhs[right_layout.span_lo:right_layout.span_lo + output_count]
                var leaf_status = cpu_scalar_broadcast_binary_contiguous[
                    operation, True, simd_width_of[DType.float32]()
                ](dst, dense, lhs[left_layout.offset])
                if leaf_status != TensorStatus.OK:
                    return TensorResult[Tensor[out_rank, HostStorage]](leaf_status)
                return result._finish_complete_overwrite()
            if (right_layout.shape.numel() == 1 and
                left_layout.shape.numel() == output_count and
                left_layout.is_c_contiguous() and left_layout.is_exhaustive()):
                var dense = lhs[left_layout.span_lo:left_layout.span_lo + output_count]
                var leaf_status = cpu_scalar_broadcast_binary_contiguous[
                    operation, False, simd_width_of[DType.float32]()
                ](dst, dense, rhs[right_layout.offset])
                if leaf_status != TensorStatus.OK:
                    return TensorResult[Tensor[out_rank, HostStorage]](leaf_status)
                return result._finish_complete_overwrite()
        var cursor = NDIndexIterator[out_rank](shape.payload.value())
        while not cursor.done():
            var lo = left_layout.offset
            var ro = right_layout.offset
            comptime for axis in range(out_rank):
                var coord = cursor.coordinates.get(axis)
                lo += coord * left_steps.get(axis)
                ro += coord * right_steps.get(axis)
            var x = lhs[lo]
            var y = rhs[ro]
            comptime if operation == 0:
                dst[cursor.produced] = x + y
            elif operation == 1:
                dst[cursor.produced] = x - y
            elif operation == 2:
                dst[cursor.produced] = x * y
            elif operation == 3:
                dst[cursor.produced] = x / y
            elif operation == 4:
                dst[cursor.produced] = fp32_maximum(x, y)
            else:
                dst[cursor.produced] = fp32_minimum(x, y)
            cursor.advance()
        return result._finish_complete_overwrite()

    def add_nd[other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[Tensor[out_rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._binary_nd_host[0, other_rank, out_rank](other)

    def sub_nd[other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[Tensor[out_rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._binary_nd_host[1, other_rank, out_rank](other)

    def mul_nd[other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[Tensor[out_rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._binary_nd_host[2, other_rank, out_rank](other)

    def div_nd[other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[Tensor[out_rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._binary_nd_host[3, other_rank, out_rank](other)

    def maximum_nd[other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[Tensor[out_rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._binary_nd_host[4, other_rank, out_rank](other)

    def minimum_nd[other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[Tensor[out_rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._binary_nd_host[5, other_rank, out_rank](other)


    # Fused multiply-add is one primitive, not mul() followed by add() and
    # materializing an intermediate Tensor. All three inputs are Float32.
    # Singleton dimensions broadcast (including the 1->0 empty case).
    # STRICT_FP32 never deliberately selects the explicit FMA instruction;
    # compile settings must also prohibit implicit reassociation/contraction
    # to demand exact non-fused bitwise results on every CPU target.
    def fma(
        ref self, multiplier: Tensor[Self.rank, HostStorage],
        addend: Tensor[Self.rank, HostStorage], mode: Fp32MathMode = Fp32MathMode.STRICT_FP32,
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        if mode.code < 0 or mode.code > 3:
            return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.UNSUPPORTED)
        var aa = self._checked_current_layout()
        if not aa.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](aa.status)
        var bb = multiplier._checked_current_layout()
        if not bb.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](bb.status)
        var cc = addend._checked_current_layout()
        if not cc.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](cc.status)
        var a_layout = aa.payload.value()
        var b_layout = bb.payload.value()
        var c_layout = cc.payload.value()
        var dims = StaticRankArray[Self.rank](1)
        comptime for axis in range(Self.rank):
            var da = a_layout.shape.dims.get(axis)
            var db = b_layout.shape.dims.get(axis)
            var dc = c_layout.shape.dims.get(axis)
            var result_dim = da
            if da != db:
                if da == 1:
                    result_dim = db
                elif db != 1:
                    return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
            if result_dim != dc:
                if result_dim == 1:
                    result_dim = dc
                elif dc != 1:
                    return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
            dims.set(axis, result_dim)
        var inferred = TensorShape[Self.rank].checked(dims)
        if not inferred.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](inferred.status)
        var allocated = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(inferred.payload.value())
        if not allocated.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](allocated.status)
        var output = allocated.payload.take()
        var dst = output.impl.storage.borrow_mut_span()
        var lhs = self.impl.storage.borrow_span()
        var rhs = multiplier.impl.storage.borrow_span()
        var acc = addend.impl.storage.borrow_span()
        var fused = allows_fma(mode)
        if a_layout.shape.same_extents(inferred.payload.value()) and b_layout.shape.same_extents(inferred.payload.value()) and c_layout.shape.same_extents(inferred.payload.value()):
            if a_layout.is_c_contiguous() and b_layout.is_c_contiguous() and c_layout.is_c_contiguous():
                if a_layout.is_exhaustive() and b_layout.is_exhaustive() and c_layout.is_exhaustive():
                    var count = inferred.payload.value().numel()
                    var left_run = lhs[a_layout.span_lo:a_layout.span_lo + count]
                    var right_run = rhs[b_layout.span_lo:b_layout.span_lo + count]
                    var add_run = acc[c_layout.span_lo:c_layout.span_lo + count]
                    var leaf_status = cpu_fma_contiguous[simd_width_of[DType.float32]()](dst, left_run, right_run, add_run, fused)
                    if leaf_status != TensorStatus.OK:
                        return TensorResult[Tensor[Self.rank, HostStorage]](leaf_status)
                    return output._finish_complete_overwrite()
        var sa = a_layout.strides.copy()
        var sb = b_layout.strides.copy()
        var sc = c_layout.strides.copy()
        comptime for axis in range(Self.rank):
            if a_layout.shape.dims.get(axis) == 1:
                sa.set(axis, 0)
            if b_layout.shape.dims.get(axis) == 1:
                sb.set(axis, 0)
            if c_layout.shape.dims.get(axis) == 1:
                sc.set(axis, 0)
        var cursor = NDIndexIterator[Self.rank](inferred.payload.value())
        while not cursor.done():
            var pa = a_layout.offset
            var pb = b_layout.offset
            var pc = c_layout.offset
            comptime for axis in range(Self.rank):
                var index = cursor.coordinates.get(axis)
                pa += index * sa.get(axis)
                pb += index * sb.get(axis)
                pc += index * sc.get(axis)
            if fused:
                dst[cursor.produced] = std_fma(lhs[pa], rhs[pb], acc[pc])
            else:
                dst[cursor.produced] = lhs[pa] * rhs[pb] + acc[pc]
            cursor.advance()
        return output._finish_complete_overwrite()

    # Mixed-rank ternary broadcasting is static: out_rank is max of the
    # compile-time operand ranks, so rank=0 scalars work without a dispatcher.
    def fma_nd[brank: Int, crank: Int, out_rank: Int](
        ref self, multiplier: Tensor[brank, HostStorage],
        addend: Tensor[crank, HostStorage], mode: Fp32MathMode = Fp32MathMode.STRICT_FP32,
    ) -> TensorResult[Tensor[out_rank, HostStorage]] where identical(Self.S, HostStorage):
        comptime assert brank >= 0 and brank <= 8 and crank >= 0 and crank <= 8, "rank 0..8 only"
        comptime assert out_rank == max(Self.rank, max(brank, crank)) and out_rank <= 8, "output rank mismatch"
        comptime if Self.rank == brank and Self.rank == crank:
            return self.fma(multiplier, addend, mode)
        if mode.code < 0 or mode.code > 3:
            return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.UNSUPPORTED)
        var aa = self._checked_current_layout()
        if not aa.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](aa.status)
        var bb = multiplier._checked_current_layout()
        if not bb.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](bb.status)
        var cc = addend._checked_current_layout()
        if not cc.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](cc.status)
        var a_layout = aa.payload.value()
        var b_layout = bb.payload.value()
        var c_layout = cc.payload.value()
        var dims = StaticRankArray[out_rank](1)
        var sa = StaticRankArray[out_rank](0)
        var sb = StaticRankArray[out_rank](0)
        var sc = StaticRankArray[out_rank](0)
        comptime for axis in range(out_rank):
            var da = 1
            var db = 1
            var dc = 1
            comptime if axis >= out_rank - Self.rank:
                da = a_layout.shape.dims.get(axis - (out_rank - Self.rank))
            comptime if axis >= out_rank - brank:
                db = b_layout.shape.dims.get(axis - (out_rank - brank))
            comptime if axis >= out_rank - crank:
                dc = c_layout.shape.dims.get(axis - (out_rank - crank))
            var result_dim = da
            if da != db:
                if da == 1:
                    result_dim = db
                elif db != 1:
                    return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
            if result_dim != dc:
                if result_dim == 1:
                    result_dim = dc
                elif dc != 1:
                    return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
            dims.set(axis, result_dim)
        var inferred = TensorShape[out_rank].checked(dims)
        if not inferred.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](inferred.status)
        comptime for axis in range(Self.rank):
            comptime dest = out_rank - Self.rank + axis
            if a_layout.shape.dims.get(axis) != 1:
                sa.set(dest, a_layout.strides.get(axis))
        comptime for axis in range(brank):
            comptime dest = out_rank - brank + axis
            if b_layout.shape.dims.get(axis) != 1:
                sb.set(dest, b_layout.strides.get(axis))
        comptime for axis in range(crank):
            comptime dest = out_rank - crank + axis
            if c_layout.shape.dims.get(axis) != 1:
                sc.set(dest, c_layout.strides.get(axis))
        var allocated = Tensor[out_rank, HostStorage]._empty_for_complete_overwrite_checked(inferred.payload.value())
        if not allocated.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](allocated.status)
        var output = allocated.payload.take()
        var lhs = self.impl.storage.borrow_span()
        var rhs = multiplier.impl.storage.borrow_span()
        var acc = addend.impl.storage.borrow_span()
        var dst = output.impl.storage.borrow_mut_span()
        var fused = allows_fma(mode)
        var cursor = NDIndexIterator[out_rank](inferred.payload.value())
        while not cursor.done():
            var pa = a_layout.offset
            var pb = b_layout.offset
            var pc = c_layout.offset
            comptime for axis in range(out_rank):
                var index = cursor.coordinates.get(axis)
                pa += index * sa.get(axis)
                pb += index * sb.get(axis)
                pc += index * sc.get(axis)
            if fused:
                dst[cursor.produced] = std_fma(lhs[pa], rhs[pb], acc[pc])
            else:
                dst[cursor.produced] = lhs[pa] * rhs[pb] + acc[pc]
            cursor.advance()
        return output._finish_complete_overwrite()

    # Comparison masks use a separate auxiliary representation, not a bool Tensor.
    # Rank-zero is scalar and emits exactly one predicate byte.
    def _compare_nd_host[relation: Int, other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[TensorMask[out_rank]] where identical(Self.S, HostStorage):
        comptime assert relation >= 0 and relation <= 5, "invalid predicate"
        comptime assert other_rank >= 0 and other_rank <= 8, "invalid input rank"
        comptime assert out_rank == max(Self.rank, other_rank) and out_rank <= 8, "invalid output rank"
        comptime if other_rank == Self.rank:
            return self._compare_host[relation](other)
        var left_proof = self._checked_current_layout()
        if not left_proof.is_ok():
            return TensorResult[TensorMask[out_rank]](left_proof.status)
        var right_proof = other._checked_current_layout()
        if not right_proof.is_ok():
            return TensorResult[TensorMask[out_rank]](right_proof.status)
        var left_layout = left_proof.payload.value()
        var right_layout = right_proof.payload.value()
        var dims = StaticRankArray[out_rank](1)
        var left_steps = StaticRankArray[out_rank](0)
        var right_steps = StaticRankArray[out_rank](0)
        comptime for i in range(out_rank):
            var da = 1
            var db = 1
            comptime if i >= out_rank - Self.rank:
                da = left_layout.shape.dims.get(i - (out_rank - Self.rank))
            comptime if i >= out_rank - other_rank:
                db = right_layout.shape.dims.get(i - (out_rank - other_rank))
            if da == db:
                dims.set(i, da)
            elif da == 1:
                dims.set(i, db)
            elif db == 1:
                dims.set(i, da)
            else:
                return TensorResult[TensorMask[out_rank]](TensorStatus.SIZE_MISMATCH)
        var shape = TensorShape[out_rank].checked(dims)
        if not shape.is_ok():
            return TensorResult[TensorMask[out_rank]](shape.status)
        comptime for i in range(Self.rank):
            comptime at = out_rank - Self.rank + i
            if left_layout.shape.dims.get(i) != 1:
                left_steps.set(at, left_layout.strides.get(i))
        comptime for i in range(other_rank):
            comptime at = out_rank - other_rank + i
            if right_layout.shape.dims.get(i) != 1:
                right_steps.set(at, right_layout.strides.get(i))
        var created = TensorMask[out_rank].zeros_checked(shape.payload.value())
        if not created.is_ok():
            return TensorResult[TensorMask[out_rank]](created.status)
        var mask = created.payload.take()
        var lhs = self.impl.storage.borrow_span()
        var rhs = other.impl.storage.borrow_span()
        var cursor = NDIndexIterator[out_rank](shape.payload.value())
        while not cursor.done():
            var lo = left_layout.offset
            var ro = right_layout.offset
            comptime for axis in range(out_rank):
                var coord = cursor.coordinates.get(axis)
                lo += coord * left_steps.get(axis)
                ro += coord * right_steps.get(axis)
            var x = lhs[lo]
            var y = rhs[ro]
            var predicate = False
            comptime if relation == 0:
                predicate = x == y
            elif relation == 1:
                predicate = x != y
            elif relation == 2:
                predicate = x < y
            elif relation == 3:
                predicate = x <= y
            elif relation == 4:
                predicate = x > y
            else:
                predicate = x >= y
            if predicate:
                mask.bytes[cursor.produced] = 1
            else:
                mask.bytes[cursor.produced] = 0
            cursor.advance()
        return TensorResult[TensorMask[out_rank]](mask)

    def eq_nd[other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[TensorMask[out_rank]] where identical(Self.S, HostStorage):
        return self._compare_nd_host[0, other_rank, out_rank](other)

    def ne_nd[other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[TensorMask[out_rank]] where identical(Self.S, HostStorage):
        return self._compare_nd_host[1, other_rank, out_rank](other)

    def lt_nd[other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[TensorMask[out_rank]] where identical(Self.S, HostStorage):
        return self._compare_nd_host[2, other_rank, out_rank](other)

    def le_nd[other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[TensorMask[out_rank]] where identical(Self.S, HostStorage):
        return self._compare_nd_host[3, other_rank, out_rank](other)

    def gt_nd[other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[TensorMask[out_rank]] where identical(Self.S, HostStorage):
        return self._compare_nd_host[4, other_rank, out_rank](other)

    def ge_nd[other_rank: Int, out_rank: Int](
        ref self, other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[TensorMask[out_rank]] where identical(Self.S, HostStorage):
        return self._compare_nd_host[5, other_rank, out_rank](other)

    def add(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._binary_host[0](other)

    def sub(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._binary_host[1](other)

    def mul(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._binary_host[2](other)

    def div(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._binary_host[3](other)

    def maximum(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._binary_host[4](other)

    def minimum(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._binary_host[5](other)


    # Tensor owns the primary CPU predicate surface; auxiliary masks are not
    # an alternative Tensor dtype. Only fixed rank and Float32 are supported.
    def _compare_host[relation: Int](
        ref self, other: Tensor[Self.rank, HostStorage]
    ) -> TensorResult[TensorMask[Self.rank]] where identical(Self.S, HostStorage):
        comptime assert relation >= 0 and relation <= 5, "invalid FP32 predicate"
        var left_proof = self._checked_current_layout()
        if not left_proof.is_ok():
            return TensorResult[TensorMask[Self.rank]](left_proof.status)
        var right_proof = other._checked_current_layout()
        if not right_proof.is_ok():
            return TensorResult[TensorMask[Self.rank]](right_proof.status)
        var left_layout = left_proof.payload.value()
        var right_layout = right_proof.payload.value()
        var dims = StaticRankArray[Self.rank](0)
        var lhs_steps = left_layout.strides.copy()
        var rhs_steps = right_layout.strides.copy()
        comptime for axis in range(Self.rank):
            var da = left_layout.shape.dims.get(axis)
            var db = right_layout.shape.dims.get(axis)
            if da == db:
                dims.set(axis, da)
            elif da == 1:
                dims.set(axis, db)
            elif db == 1:
                dims.set(axis, da)
            else:
                return TensorResult[TensorMask[Self.rank]](TensorStatus.SIZE_MISMATCH)
            if da == 1:
                lhs_steps.set(axis, 0)
            if db == 1:
                rhs_steps.set(axis, 0)
        var shape = TensorShape[Self.rank].checked(dims)
        if not shape.is_ok():
            return TensorResult[TensorMask[Self.rank]](shape.status)
        var created = TensorMask[Self.rank].zeros_checked(shape.payload.value())
        if not created.is_ok():
            return TensorResult[TensorMask[Self.rank]](created.status)
        var mask = created.payload.take()
        var lhs = self.impl.storage.borrow_span()
        var rhs = other.impl.storage.borrow_span()
        var cursor = NDIndexIterator[Self.rank](shape.payload.value())
        while not cursor.done():
            var lo = left_layout.offset
            var ro = right_layout.offset
            comptime for axis in range(Self.rank):
                var position = cursor.coordinates.get(axis)
                lo += position * lhs_steps.get(axis)
                ro += position * rhs_steps.get(axis)
            var x = lhs[lo]
            var y = rhs[ro]
            var predicate = False
            comptime if relation == 0:
                predicate = x == y
            elif relation == 1:
                predicate = x != y
            elif relation == 2:
                predicate = x < y
            elif relation == 3:
                predicate = x <= y
            elif relation == 4:
                predicate = x > y
            else:
                predicate = x >= y
            if predicate:
                mask.bytes[cursor.produced] = 1
            else:
                mask.bytes[cursor.produced] = 0
            cursor.advance()
        return TensorResult[TensorMask[Self.rank]](mask)

    def eq(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[TensorMask[Self.rank]] where identical(Self.S, HostStorage):
        return self._compare_host[0](other)

    def ne(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[TensorMask[Self.rank]] where identical(Self.S, HostStorage):
        return self._compare_host[1](other)

    def lt(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[TensorMask[Self.rank]] where identical(Self.S, HostStorage):
        return self._compare_host[2](other)

    def le(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[TensorMask[Self.rank]] where identical(Self.S, HostStorage):
        return self._compare_host[3](other)

    def gt(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[TensorMask[Self.rank]] where identical(Self.S, HostStorage):
        return self._compare_host[4](other)

    def ge(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[TensorMask[Self.rank]] where identical(Self.S, HostStorage):
        return self._compare_host[5](other)

    def _predicate_host[kind: Int](ref self) -> TensorResult[TensorMask[Self.rank]] where identical(Self.S, HostStorage):
        comptime assert kind >= 0 and kind <= 2, "invalid FP32 unary predicate"
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[TensorMask[Self.rank]](proof.status)
        var safe = proof.payload.value()
        var created = TensorMask[Self.rank].zeros_checked(safe.shape)
        if not created.is_ok():
            return TensorResult[TensorMask[Self.rank]](created.status)
        var mask = created.payload.take()
        var source = self.impl.storage.borrow_span()
        var cursor = NDIndexIterator[Self.rank](safe.shape)
        while not cursor.done():
            var value = source[safe.physical_offset_unchecked(cursor.coordinates)]
            var predicate = False
            comptime if kind == 0:
                predicate = Bool(fp32_isnan(value))
            elif kind == 1:
                predicate = Bool(fp32_isinf(value))
            else:
                predicate = Bool(fp32_isfinite(value))
            if predicate:
                mask.bytes[cursor.produced] = 1
            else:
                mask.bytes[cursor.produced] = 0
            cursor.advance()
        return TensorResult[TensorMask[Self.rank]](mask)

    def isnan(ref self) -> TensorResult[TensorMask[Self.rank]] where identical(Self.S, HostStorage):
        return self._predicate_host[0]()

    def isinf(ref self) -> TensorResult[TensorMask[Self.rank]] where identical(Self.S, HostStorage):
        return self._predicate_host[1]()

    def isfinite(ref self) -> TensorResult[TensorMask[Self.rank]] where identical(Self.S, HostStorage):
        return self._predicate_host[2]()

    def where(ref self, mask: TensorMask[Self.rank], other: Tensor[Self.rank, HostStorage]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        # The result is a NEW complete overwrite; a malformed mask aborts
        # without mutating either input or exposing an incomplete result.
        var valid = mask.metadata_status()
        if valid != TensorStatus.OK:
            return TensorResult[Tensor[Self.rank, HostStorage]](valid)
        var left_proof = self._checked_current_layout()
        if not left_proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](left_proof.status)
        var right_proof = other._checked_current_layout()
        if not right_proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](right_proof.status)
        var la = left_proof.payload.value()
        var lb = right_proof.payload.value()
        var lsteps = la.strides.copy()
        var rsteps = lb.strides.copy()
        comptime for i in range(Self.rank):
            var dest_extent = mask.shape.dims.get(i)
            var a_extent = la.shape.dims.get(i)
            var b_extent = lb.shape.dims.get(i)
            if (a_extent != dest_extent and a_extent != 1) or (b_extent != dest_extent and b_extent != 1):
                return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
            if a_extent == 1:
                lsteps.set(i, 0)
            if b_extent == 1:
                rsteps.set(i, 0)
        var created = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(mask.shape)
        if not created.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](created.status)
        var result = created.payload.take()
        var left_data = self.impl.storage.borrow_span()
        var right_data = other.impl.storage.borrow_span()
        var out = result.impl.storage.borrow_mut_span()
        var cursor = NDIndexIterator[Self.rank](mask.shape)
        while not cursor.done():
            var predicate = mask.bytes[cursor.produced]
            if predicate != 0 and predicate != 1:
                return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.INVALID_LAYOUT)
            var at_a = la.offset
            var at_b = lb.offset
            comptime for i in range(Self.rank):
                var at = cursor.coordinates.get(i)
                at_a += at * lsteps.get(i)
                at_b += at * rsteps.get(i)
            if predicate == 1:
                out[cursor.produced] = left_data[at_a]
            else:
                out[cursor.produced] = right_data[at_b]
            cursor.advance()
        return result._finish_complete_overwrite()

    # Mixed-rank where is materialized through the owning Tensor facade.
    # The mask fixes the output rank; operands broadcast from the right.
    def where_nd[other_rank: Int, out_rank: Int](
        ref self, mask: TensorMask[out_rank], other: Tensor[other_rank, HostStorage]
    ) -> TensorResult[Tensor[out_rank, HostStorage]] where identical(Self.S, HostStorage):
        comptime assert other_rank >= 0 and other_rank <= 8, "invalid RHS rank"
        comptime assert out_rank >= Self.rank and out_rank >= other_rank and out_rank <= 8, "invalid mask rank"
        comptime if Self.rank == out_rank and other_rank == out_rank:
            return self.where(mask, other)
        var checked_mask = mask.metadata_status()
        if checked_mask != TensorStatus.OK:
            return TensorResult[Tensor[out_rank, HostStorage]](checked_mask)
        var left_proof = self._checked_current_layout()
        if not left_proof.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](left_proof.status)
        var right_proof = other._checked_current_layout()
        if not right_proof.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](right_proof.status)
        var al = left_proof.payload.value()
        var bl = right_proof.payload.value()
        var a_steps = StaticRankArray[out_rank](0)
        var b_steps = StaticRankArray[out_rank](0)
        comptime for i in range(out_rank):
            var da = 1
            var db = 1
            comptime if i >= out_rank - Self.rank:
                comptime ai = i - (out_rank - Self.rank)
                da = al.shape.dims.get(ai)
                if da != 1:
                    a_steps.set(i, al.strides.get(ai))
            comptime if i >= out_rank - other_rank:
                comptime bi = i - (out_rank - other_rank)
                db = bl.shape.dims.get(bi)
                if db != 1:
                    b_steps.set(i, bl.strides.get(bi))
            var requested = mask.shape.dims.get(i)
            if (da != requested and da != 1) or (db != requested and db != 1):
                return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var output_shape = TensorShape[out_rank].checked(mask.shape.dims)
        if not output_shape.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](output_shape.status)
        var created = Tensor[out_rank, HostStorage]._empty_for_complete_overwrite_checked(output_shape.payload.value())
        if not created.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](created.status)
        var output = created.payload.take()
        var lhs = self.impl.storage.borrow_span()
        var rhs = other.impl.storage.borrow_span()
        var dst = output.impl.storage.borrow_mut_span()
        var cursor = NDIndexIterator[out_rank](output_shape.payload.value())
        while not cursor.done():
            var predicate = mask.bytes[cursor.produced]
            if predicate != 0 and predicate != 1:
                return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.INVALID_LAYOUT)
            var ai = al.offset
            var bi = bl.offset
            comptime for axis in range(out_rank):
                var position = cursor.coordinates.get(axis)
                ai += position * a_steps.get(axis)
                bi += position * b_steps.get(axis)
            if predicate == 1:
                dst[cursor.produced] = lhs[ai]
            else:
                dst[cursor.produced] = rhs[bi]
            cursor.advance()
        return output._finish_complete_overwrite()

    def masked_fill(mut self, mask: TensorMask[Self.rank], value: Float32) -> TensorStatus where identical(Self.S, HostStorage):
        # In-place mutation is atomic w.r.t. invalid mask/shape/overlap:
        # validate ALL predicates before the first write.
        var mask_status = mask.validated()
        if mask_status != TensorStatus.OK:
            return mask_status
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return proof.status
        var safe = proof.payload.value()
        if not mask.shape.same_extents(safe.shape):
            return TensorStatus.SIZE_MISMATCH
        if not safe.is_proven_unique():
            return TensorStatus.OVERLAPPING
        var data = self.impl.storage.borrow_mut_span()
        var cursor = NDIndexIterator[Self.rank](safe.shape)
        while not cursor.done():
            if mask.bytes[cursor.produced] == 1:
                data[safe.physical_offset_unchecked(cursor.coordinates)] = value
            cursor.advance()
        return TensorStatus.OK

    # Operators on the single public owning type remain checked and always
    # return TensorResult. They never implicitly transfer, mutate inputs,
    # erase FP32 metadata, or allocate an untracked intermediate.
    def __add__(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.add(other)

    def __sub__(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.sub(other)

    def __mul__(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.mul(other)

    def __truediv__(ref self, other: Tensor[Self.rank, HostStorage]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.div(other)

    def neg(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        # Unary sign flip is not multiplication by -1: preserve IEEE-754
        # signed zeros, infinities and NaN sign in the chosen FP32 mode.
        return self.unary_host[0]()

    def __neg__(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self.neg()

    # The primary Tensor API owns the rank-2 HostStorage GEMM path. The
    # functional ops.mm wrapper delegates here instead of duplicating kernels.
    # No implicit CPU/GPU transfer, second dtype, or hidden reshape.
    def mm(
        ref self, right: Tensor[2, HostStorage]
    ) -> TensorResult[Tensor[2, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 2:
        var checked_left = self._checked_current_layout()
        if not checked_left.is_ok():
            return TensorResult[Tensor[2, HostStorage]](checked_left.status)
        var checked_right = right._checked_current_layout()
        if not checked_right.is_ok():
            return TensorResult[Tensor[2, HostStorage]](checked_right.status)
        var left_layout = checked_left.payload.value()
        var right_layout = checked_right.payload.value()
        var m = left_layout.shape.dims.get(0)
        var k = left_layout.shape.dims.get(1)
        if k != right_layout.shape.dims.get(0):
            return TensorResult[Tensor[2, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var n = right_layout.shape.dims.get(1)
        var dims = StaticRankArray[2](0)
        dims.set(0, m)
        dims.set(1, n)
        var shape = TensorShape[2].checked(dims)
        if not shape.is_ok():
            return TensorResult[Tensor[2, HostStorage]](shape.status)
        var allocated = Tensor[2, HostStorage]._empty_for_complete_overwrite_checked(
            shape.payload.value()
        )
        if not allocated.is_ok():
            return TensorResult[Tensor[2, HostStorage]](allocated.status)
        var output = allocated.payload.take()
        var lhs = self.impl.storage.borrow_span()
        var rhs = right.impl.storage.borrow_span()
        var dst = output.impl.storage.borrow_mut_span()
        if left_layout.is_c_contiguous() and right_layout.is_c_contiguous():
            if left_layout.is_exhaustive() and right_layout.is_exhaustive():
                var left_run = lhs[
                    left_layout.span_lo:left_layout.span_lo + left_layout.shape.numel()
                ]
                var right_run = rhs[
                    right_layout.span_lo:right_layout.span_lo + right_layout.shape.numel()
                ]
                var selected_kernel = cpu_mm_autoselect_fp32[
                    simd_width_of[DType.float32]()
                ](dst, left_run, right_run, m, k, n)
                if selected_kernel != TensorStatus.OK:
                    return TensorResult[Tensor[2, HostStorage]](selected_kernel)
                return output._finish_complete_overwrite()
        # A layout may be signed-stride or transposed while still mapping
        # uniquely. Address proofs above cover the entire K-reduction span.
        var row_stride_a = left_layout.strides.get(0)
        var depth_stride_a = left_layout.strides.get(1)
        var depth_stride_b = right_layout.strides.get(0)
        var col_stride_b = right_layout.strides.get(1)
        for row in range(m):
            var left_row = left_layout.offset + row * row_stride_a
            for col in range(n):
                var right_col = right_layout.offset + col * col_stride_b
                var acc = Float32(0.0)
                for depth in range(k):
                    acc += lhs[left_row + depth * depth_stride_a] * rhs[
                        right_col + depth * depth_stride_b
                    ]
                dst[row * n + col] = acc
        return output._finish_complete_overwrite()

    # Allocation-free repeated GEMM into an EXISTING initialized Host Tensor.
    # The caller explicitly owns both the destination and the FP32 scratch
    # span; neither allocation, prepacking, worker creation for tiny cases,
    # nor device probing is forced by this API. This is intentionally limited
    # to C-order rank-2 matrices; mm() remains the fully general fallback.
    # Existing initialized destination values are overwritten completely on
    # success. Alias/capacity checks happen before the first output write.
    def mm_into_workspace[workspace_origin: Origin[mut=True]](
        ref self, right: Tensor[2, HostStorage],
        mut destination: Tensor[2, HostStorage],
        mut workspace: Span[True, Float32, workspace_origin],
    ) -> TensorStatus where identical(Self.S, HostStorage) and Self.rank == 2:
        var left_checked = self._checked_current_layout()
        if not left_checked.is_ok():
            return left_checked.status
        var right_checked = right._checked_current_layout()
        if not right_checked.is_ok():
            return right_checked.status
        var dest_checked = destination._checked_current_layout()
        if not dest_checked.is_ok():
            return dest_checked.status
        var left_layout = left_checked.payload.value()
        var right_layout = right_checked.payload.value()
        var dest_layout = dest_checked.payload.value()
        var m = left_layout.shape.dims.get(0)
        var k = left_layout.shape.dims.get(1)
        var n = right_layout.shape.dims.get(1)
        if right_layout.shape.dims.get(0) != k:
            return TensorStatus.SIZE_MISMATCH
        if (dest_layout.shape.dims.get(0) != m
            or dest_layout.shape.dims.get(1) != n):
            return TensorStatus.SIZE_MISMATCH
        if (not left_layout.is_c_contiguous()
            or not right_layout.is_c_contiguous()
            or not dest_layout.is_c_contiguous()
            or not left_layout.is_exhaustive()
            or not right_layout.is_exhaustive()
            or not dest_layout.is_exhaustive()
            or not dest_layout.is_proven_unique()):
            return TensorStatus.NOT_REPRESENTABLE
        # All products below are validated through canonical TensorShape
        # metadata and the owning HostStorage capacity proofs above.
        var lhs = self.impl.storage.borrow_span()
        var rhs = right.impl.storage.borrow_span()
        var dst = destination.impl.storage.borrow_mut_span()
        var a_start = left_layout.span_lo
        var b_start = right_layout.span_lo
        var c_start = dest_layout.span_lo
        var a_run = lhs[a_start:a_start + m * k]
        var b_run = rhs[b_start:b_start + k * n]
        var c_run = dst[c_start:c_start + m * n]
        comptime width = simd_width_of[DType.float32]()
        var completed = TensorStatus.OK
        if m >= 64 and n >= 64 and k >= 128:
            # The standard precompiled panel is 128x64 Float32 elements.
            # The caller must supply at least 8192 scratch elements, safely
            # disjoint from A, B and C. The kernel verifies this atomically.
            completed = cpu_mm_packed_workspace_fp32[width, 128, 64, 64](
                c_run, a_run, b_run, workspace, m, k, n,
            )
        else:
            # The selector chooses only the unallocated small/narrow paths.
            # K=0 writes zeros without dereferencing inputs.
            completed = cpu_mm_autoselect_fp32[width](
                c_run, a_run, b_run, m, k, n,
            )
        if completed != TensorStatus.OK:
            # The checked kernel only returns SIZE_MISMATCH / ALIAS_UNKNOWN
            # on its pre-write buffer-capacity and disjointness gates. The
            # destination stays readable for those rejected calls, so users
            # can supply a corrected scratch span and retry.
            # Any OTHER kernel error is conservatively treated as possibly
            # post-write: never expose a partially overwritten FP32 owner.
            if (completed != TensorStatus.SIZE_MISMATCH
                and completed != TensorStatus.ALIAS_UNKNOWN):
                destination.impl.storage._invalidate_after_failed_write()
        return completed

    def vecmat(
        ref self, matrix: Tensor[2, HostStorage]
    ) -> TensorResult[Tensor[1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 1:
        var vector_checked = self._checked_current_layout()
        if not vector_checked.is_ok():
            return TensorResult[Tensor[1, HostStorage]](vector_checked.status)
        var matrix_checked = matrix._checked_current_layout()
        if not matrix_checked.is_ok():
            return TensorResult[Tensor[1, HostStorage]](matrix_checked.status)
        var vector_layout = vector_checked.payload.value()
        var matrix_layout = matrix_checked.payload.value()
        var k = matrix_layout.shape.dims.get(0)
        var n = matrix_layout.shape.dims.get(1)
        if vector_layout.shape.numel() != k:
            return TensorResult[Tensor[1, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var result_dims = StaticRankArray[1](n)
        var result_shape = TensorShape[1].checked(result_dims)
        if not result_shape.is_ok():
            return TensorResult[Tensor[1, HostStorage]](result_shape.status)
        var allocation = Tensor[1, HostStorage]._empty_for_complete_overwrite_checked(
            result_shape.payload.value()
        )
        if not allocation.is_ok():
            return TensorResult[Tensor[1, HostStorage]](allocation.status)
        var output = allocation.payload.take()
        var vec = self.impl.storage.borrow_span()
        var rhs = matrix.impl.storage.borrow_span()
        var dst = output.impl.storage.borrow_mut_span()
        if vector_layout.is_c_contiguous() and matrix_layout.is_c_contiguous():
            if vector_layout.is_exhaustive() and matrix_layout.is_exhaustive():
                var vector_run = vec[
                    vector_layout.span_lo:vector_layout.span_lo + k
                ]
                var matrix_run = rhs[
                    matrix_layout.span_lo:matrix_layout.span_lo + k * n
                ]
                var fast_status = cpu_vecmat_checked_fp32[simd_width_of[DType.float32]()](
                    dst, vector_run, matrix_run, k, n
                )
                if fast_status != TensorStatus.OK:
                    return TensorResult[Tensor[1, HostStorage]](fast_status)
                return output._finish_complete_overwrite()
        var vector_stride = vector_layout.strides.get(0)
        var row_stride = matrix_layout.strides.get(0)
        var col_stride = matrix_layout.strides.get(1)
        for col in range(n):
            var acc = Float32(0.0)
            var rhs_col = matrix_layout.offset + col * col_stride
            for inner in range(k):
                acc += vec[vector_layout.offset + inner * vector_stride] * rhs[
                    rhs_col + inner * row_stride
                ]
            dst[col] = acc
        return output._finish_complete_overwrite()

    def outer(
        ref self, other: Tensor[1, HostStorage]
    ) -> TensorResult[Tensor[2, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 1:
        var left_checked = self._checked_current_layout()
        if not left_checked.is_ok():
            return TensorResult[Tensor[2, HostStorage]](left_checked.status)
        var right_checked = other._checked_current_layout()
        if not right_checked.is_ok():
            return TensorResult[Tensor[2, HostStorage]](right_checked.status)
        var left_layout = left_checked.payload.value()
        var right_layout = right_checked.payload.value()
        var m = left_layout.shape.numel()
        var n = right_layout.shape.numel()
        var result_dims = StaticRankArray[2](0)
        result_dims.set(0, m)
        result_dims.set(1, n)
        var result_shape = TensorShape[2].checked(result_dims)
        if not result_shape.is_ok():
            return TensorResult[Tensor[2, HostStorage]](result_shape.status)
        var allocation = Tensor[2, HostStorage]._empty_for_complete_overwrite_checked(
            result_shape.payload.value()
        )
        if not allocation.is_ok():
            return TensorResult[Tensor[2, HostStorage]](allocation.status)
        var result = allocation.payload.take()
        var dst = result.impl.storage.borrow_mut_span()
        var lhs = self.impl.storage.borrow_span()
        var rhs = other.impl.storage.borrow_span()
        if left_layout.is_c_contiguous() and right_layout.is_c_contiguous():
            if left_layout.is_exhaustive() and right_layout.is_exhaustive():
                var fast_status = cpu_outer_checked_fp32[simd_width_of[DType.float32]()](
                    dst,
                    lhs[left_layout.span_lo:left_layout.span_lo + m],
                    rhs[right_layout.span_lo:right_layout.span_lo + n],
                    m, n
                )
                if fast_status != TensorStatus.OK:
                    return TensorResult[Tensor[2, HostStorage]](fast_status)
                return result._finish_complete_overwrite()
        var left_stride = left_layout.strides.get(0)
        var right_stride = right_layout.strides.get(0)
        for row in range(m):
            var scalar = lhs[left_layout.offset + row * left_stride]
            for col in range(n):
                dst[row * n + col] = scalar * rhs[
                    right_layout.offset + col * right_stride
                ]
        return result._finish_complete_overwrite()

    # Mojo's matrix multiplication operator is a zero-overhead alias of the
    # existing rank-specialized owning Tensor kernels. No extra allocation,
    # shape traversal, layout change or CPU/GPU migration is introduced.
    def __matmul__(
        ref self, right: Tensor[2, HostStorage]
    ) -> TensorResult[Tensor[2, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 2:
        return self.mm(right)

    def __matmul__(
        ref self, right: Tensor[1, HostStorage]
    ) -> TensorResult[Tensor[1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 2:
        return self.gemv(right)

    def __matmul__(
        ref self, right: Tensor[1, HostStorage]
    ) -> TensorResult[Float32] where identical(Self.S, HostStorage) and Self.rank == 1:
        return self.dot(right)

    def __matmul__(
        ref self, right: Tensor[2, HostStorage]
    ) -> TensorResult[Tensor[1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 1:
        return self.vecmat(right)

    def __matmul__(
        ref self, right: Tensor[Self.rank, HostStorage]
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage) and Self.rank >= 3:
        return self.matmul_nd(right)

    def matmul(
        ref self, right: Tensor[Self.rank, HostStorage]
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage) and Self.rank >= 3:
        return self.matmul_nd(right)

    # General batched matmul with DIFFERENT static input ranks (2..8).
    # Right-aligned batch broadcasting is resolved ONCE at the boundary.
    # Tensor owns the HostStorage implementation; ops/matrix is a thin
    # functional delegate. CPU fast path calls packed/SIMD-aware rank-two
    # microkernels per broadcasted batch for standard contiguous layouts.
    def matmul_mixed[other_rank: Int, out_rank: Int](
        ref self, right: Tensor[other_rank, HostStorage]
    ) -> TensorResult[Tensor[out_rank, HostStorage]] where identical(Self.S, HostStorage) and Self.rank >= 2 and other_rank >= 2 and other_rank <= 8 and out_rank == max(Self.rank, other_rank):
        comptime if Self.rank == other_rank:
            return self.matmul_nd(right)
        var checked_a = self._checked_current_layout()
        if not checked_a.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](checked_a.status)
        var checked_b = right._checked_current_layout()
        if not checked_b.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](checked_b.status)
        var a_layout = checked_a.payload.value()
        var b_layout = checked_b.payload.value()
        var m = a_layout.shape.dims.get(Self.rank - 2)
        var k = a_layout.shape.dims.get(Self.rank - 1)
        if k != b_layout.shape.dims.get(other_rank - 2):
            return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var n = b_layout.shape.dims.get(other_rank - 1)
        var dims = StaticRankArray[out_rank](1)
        comptime for axis in range(out_rank - 2):
            var da = 1
            var db = 1
            comptime if axis >= out_rank - Self.rank:
                da = a_layout.shape.dims.get(axis - (out_rank - Self.rank))
            comptime if axis >= out_rank - other_rank:
                db = b_layout.shape.dims.get(axis - (out_rank - other_rank))
            if da == db:
                dims.set(axis, da)
            elif da == 1:
                dims.set(axis, db)
            elif db == 1:
                dims.set(axis, da)
            else:
                return TensorResult[Tensor[out_rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        dims.set(out_rank - 2, m)
        dims.set(out_rank - 1, n)
        var shape = TensorShape[out_rank].checked(dims)
        if not shape.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](shape.status)
        # K=0: the additive identity is defined, even for a nonempty result.
        # Never read an uninitialized operand or instantiate a GEMM with K=0.
        if k == 0:
            return Tensor[out_rank, HostStorage].zeros_checked(shape.payload.value())
        var made = Tensor[out_rank, HostStorage]._empty_for_complete_overwrite_checked(
            shape.payload.value()
        )
        if not made.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](made.status)
        var output = made.payload.take()
        if output.numel() == 0:
            return output._finish_complete_overwrite()
        var lhs = self.impl.storage.borrow_span()
        var rhs = right.impl.storage.borrow_span()
        var dst = output.impl.storage.borrow_mut_span()
        var a_depth = a_layout.strides.get(Self.rank - 1)
        var b_depth = b_layout.strides.get(other_rank - 2)
        if a_layout.is_c_contiguous() and b_layout.is_c_contiguous():
            if a_layout.is_exhaustive() and b_layout.is_exhaustive():
                # A contiguous batch broadcast reuses whole packed matrix
                # panels rather than scalar-by-scalar N-D reduction loops.
                var batch_count = 1
                comptime for axis in range(out_rank - 2):
                    batch_count *= dims.get(axis)
                var a_matrix = m * k
                var b_matrix = k * n
                var c_matrix = m * n
                for batch in range(batch_count):
                    var residue = batch
                    var coordinate = StaticRankArray[out_rank](0)
                    comptime for reverse_axis in range(out_rank - 2):
                        var axis = out_rank - 3 - reverse_axis
                        var extent = dims.get(axis)
                        coordinate.set(axis, residue % extent)
                        residue //= extent
                    var a_start = a_layout.offset
                    var b_start = b_layout.offset
                    comptime for axis in range(out_rank - 2):
                        comptime if axis >= out_rank - Self.rank:
                            var source_axis = axis - (out_rank - Self.rank)
                            if a_layout.shape.dims.get(source_axis) > 1:
                                a_start += coordinate.get(axis) * a_layout.strides.get(source_axis)
                        comptime if axis >= out_rank - other_rank:
                            var source_axis_b = axis - (out_rank - other_rank)
                            if b_layout.shape.dims.get(source_axis_b) > 1:
                                b_start += coordinate.get(axis) * b_layout.strides.get(source_axis_b)
                    var out_start = batch * c_matrix
                    var selected_kernel = cpu_mm_autoselect_fp32[
                        simd_width_of[DType.float32]()
                    ](
                        dst[out_start:out_start + c_matrix],
                        lhs[a_start:a_start + a_matrix],
                        rhs[b_start:b_start + b_matrix],
                        m, k, n,
                    )
                    if selected_kernel != TensorStatus.OK:
                        return TensorResult[Tensor[out_rank, HostStorage]](selected_kernel)
                return output._finish_complete_overwrite()
        # Arbitrary positive/negative/zero-stride layouts retain exact
        # right-aligned broadcasting and a single physical bounds proof.
        var cursor = NDIndexIterator[out_rank](shape.payload.value())
        while not cursor.done():
            var ai = StaticRankArray[Self.rank](0)
            var bi = StaticRankArray[other_rank](0)
            comptime for axis in range(Self.rank - 2):
                var value = cursor.coordinates.get(out_rank - Self.rank + axis)
                if a_layout.shape.dims.get(axis) == 1:
                    value = 0
                ai.set(axis, value)
            comptime for axis in range(other_rank - 2):
                var value = cursor.coordinates.get(out_rank - other_rank + axis)
                if b_layout.shape.dims.get(axis) == 1:
                    value = 0
                bi.set(axis, value)
            ai.set(Self.rank - 2, cursor.coordinates.get(out_rank - 2))
            bi.set(other_rank - 1, cursor.coordinates.get(out_rank - 1))
            ai.set(Self.rank - 1, 0)
            bi.set(other_rank - 2, 0)
            var a_base = a_layout.physical_offset_unchecked(ai)
            var b_base = b_layout.physical_offset_unchecked(bi)
            var total = Float32(0.0)
            for inner in range(k):
                total += lhs[a_base + inner * a_depth] * rhs[b_base + inner * b_depth]
            dst[cursor.produced] = total
            cursor.advance()
        return output._finish_complete_overwrite()

    # Strict rank-three BMM has no implicit batch broadcasting.
    # Both contiguous operands reuse the rank-two SIMD microkernel per batch.
    def bmm(ref self, right: Tensor[3, HostStorage]) -> TensorResult[Tensor[3, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 3:
        var left_proof = self._checked_current_layout()
        if not left_proof.is_ok():
            return TensorResult[Tensor[3, HostStorage]](left_proof.status)
        var right_proof = right._checked_current_layout()
        if not right_proof.is_ok():
            return TensorResult[Tensor[3, HostStorage]](right_proof.status)
        var al = left_proof.payload.value()
        var bl = right_proof.payload.value()
        var batches = al.shape.dims.get(0)
        var m = al.shape.dims.get(1)
        var k = al.shape.dims.get(2)
        var n = bl.shape.dims.get(2)
        if bl.shape.dims.get(0) != batches or bl.shape.dims.get(1) != k:
            return TensorResult[Tensor[3, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var dims = StaticRankArray[3](0)
        dims.set(0, batches)
        dims.set(1, m)
        dims.set(2, n)
        var shape = TensorShape[3].checked(dims)
        if not shape.is_ok():
            return TensorResult[Tensor[3, HostStorage]](shape.status)
        # For empty K every live output must be zero; no source is read.
        if k == 0:
            return Tensor[3, HostStorage].zeros_checked(shape.payload.value())
        var made = Tensor[3, HostStorage]._empty_for_complete_overwrite_checked(shape.payload.value())
        if not made.is_ok():
            return TensorResult[Tensor[3, HostStorage]](made.status)
        var result = made.payload.take()
        if result.numel() == 0:
            return result._finish_complete_overwrite()
        var lhs = self.impl.storage.borrow_span()
        var rhs = right.impl.storage.borrow_span()
        var dst = result.impl.storage.borrow_mut_span()
        if al.is_c_contiguous() and bl.is_c_contiguous():
            if al.is_exhaustive() and bl.is_exhaustive():
                var a_length = m * k
                var b_length = k * n
                var c_length = m * n
                for batch in range(batches):
                    var a_at = al.span_lo + batch * a_length
                    var b_at = bl.span_lo + batch * b_length
                    var c_at = batch * c_length
                    var a_run = lhs[a_at:a_at + a_length]
                    var b_run = rhs[b_at:b_at + b_length]
                    var c_run = dst[c_at:c_at + c_length]
                    var selected_kernel = cpu_mm_autoselect_fp32[
                        simd_width_of[DType.float32]()
                    ](c_run, a_run, b_run, m, k, n)
                    if selected_kernel != TensorStatus.OK:
                        return TensorResult[Tensor[3, HostStorage]](selected_kernel)
                return result._finish_complete_overwrite()
        var a_batch = al.strides.get(0)
        var a_row = al.strides.get(1)
        var a_k = al.strides.get(2)
        var b_batch = bl.strides.get(0)
        var b_k = bl.strides.get(1)
        var b_col = bl.strides.get(2)
        for batch in range(batches):
            var ai = al.offset + batch * a_batch
            var bi = bl.offset + batch * b_batch
            for row in range(m):
                var row_at = ai + row * a_row
                for col in range(n):
                    var col_at = bi + col * b_col
                    var total = Float32(0.0)
                    for inner in range(k):
                        total += lhs[row_at + inner * a_k] * rhs[col_at + inner * b_k]
                    dst[(batch * m + row) * n + col] = total
        return result._finish_complete_overwrite()

    # Same-rank N-D matmul supports rank 2..8 and singleton batch broadcasting.
    # This reference lowering stays CPU-only; GPU requires an AOT artifact.
    def matmul_nd(ref self, right: Tensor[Self.rank, HostStorage]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage) and Self.rank >= 2:
        comptime if Self.rank == 2:
            return self.mm(right)
        var left_proof = self._checked_current_layout()
        if not left_proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](left_proof.status)
        var right_proof = right._checked_current_layout()
        if not right_proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](right_proof.status)
        var al = left_proof.payload.value()
        var bl = right_proof.payload.value()
        var m = al.shape.dims.get(Self.rank - 2)
        var k = al.shape.dims.get(Self.rank - 1)
        var n = bl.shape.dims.get(Self.rank - 1)
        if bl.shape.dims.get(Self.rank - 2) != k:
            return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var dims = StaticRankArray[Self.rank](0)
        comptime for axis in range(Self.rank - 2):
            var da = al.shape.dims.get(axis)
            var db = bl.shape.dims.get(axis)
            if da == db:
                dims.set(axis, da)
            elif da == 1:
                dims.set(axis, db)
            elif db == 1:
                dims.set(axis, da)
            else:
                return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        dims.set(Self.rank - 2, m)
        dims.set(Self.rank - 1, n)
        var shape = TensorShape[Self.rank].checked(dims)
        if not shape.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](shape.status)
        comptime if Self.rank == 3:
            if al.shape.dims.get(0) == bl.shape.dims.get(0):
                return self.bmm(right)
        if k == 0:
            return Tensor[Self.rank, HostStorage].zeros_checked(shape.payload.value())
        var allocated = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(shape.payload.value())
        if not allocated.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](allocated.status)
        var result = allocated.payload.take()
        if result.numel() == 0:
            return result._finish_complete_overwrite()
        var lhs = self.impl.storage.borrow_span()
        var rhs = right.impl.storage.borrow_span()
        var dst = result.impl.storage.borrow_mut_span()
        # When batch axes match, entire contiguous batches can reuse the
        # specialized rank-two SIMD GEMM, avoiding per-output N-D indexing.
        var identical_batches = True
        comptime for axis in range(Self.rank - 2):
            if al.shape.dims.get(axis) != bl.shape.dims.get(axis):
                identical_batches = False
        if identical_batches and al.is_c_contiguous() and bl.is_c_contiguous():
            if al.is_exhaustive() and bl.is_exhaustive():
                var batch_count = 1
                comptime for axis in range(Self.rank - 2):
                    batch_count *= shape.payload.value().dims.get(axis)
                var a_length = m * k
                var b_length = k * n
                var c_length = m * n
                for batch in range(batch_count):
                    var a_start = al.span_lo + batch * a_length
                    var b_start = bl.span_lo + batch * b_length
                    var c_start = batch * c_length
                    var a_run = lhs[a_start:a_start + a_length]
                    var b_run = rhs[b_start:b_start + b_length]
                    var c_run = dst[c_start:c_start + c_length]
                    var selected_kernel = cpu_mm_autoselect_fp32[
                        simd_width_of[DType.float32]()
                    ](c_run, a_run, b_run, m, k, n)
                    if selected_kernel != TensorStatus.OK:
                        return TensorResult[Tensor[Self.rank, HostStorage]](selected_kernel)
                return result._finish_complete_overwrite()
        var a_k = al.strides.get(Self.rank - 1)
        var b_k = bl.strides.get(Self.rank - 2)
        var cursor = NDIndexIterator[Self.rank](shape.payload.value())
        while not cursor.done():
            var ac = cursor.coordinates.copy()
            var bc = cursor.coordinates.copy()
            comptime for axis in range(Self.rank - 2):
                if al.shape.dims.get(axis) == 1:
                    ac.set(axis, 0)
                if bl.shape.dims.get(axis) == 1:
                    bc.set(axis, 0)
            ac.set(Self.rank - 1, 0)
            bc.set(Self.rank - 2, 0)
            var a_base = al.physical_offset_unchecked(ac)
            var b_base = bl.physical_offset_unchecked(bc)
            var total = Float32(0.0)
            for inner in range(k):
                total += lhs[a_base + inner * a_k] * rhs[b_base + inner * b_k]
            dst[cursor.produced] = total
            cursor.advance()
        return result._finish_complete_overwrite()

    def matmul(
        ref self, right: Tensor[2, HostStorage]
    ) -> TensorResult[Tensor[1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 1:
        return self.vecmat(right)

    def matmul(
        ref self, right: Tensor[1, HostStorage]
    ) -> TensorResult[Tensor[1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 2:
        return self.gemv(right)

    def matmul(
        ref self, right: Tensor[1, HostStorage]
    ) -> TensorResult[Float32] where identical(Self.S, HostStorage) and Self.rank == 1:
        return self.dot(right)

    # Rank-2 matrix-vector multiplication: strict default K-order; caller
    # may explicitly select gemv_fast() for SIMD horizontal reduction.
    def matmul(
        ref self, right: Tensor[2, HostStorage]
    ) -> TensorResult[Tensor[2, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 2:
        # Rank-2 Tensor method uses the same GEMM path as mm().
        # N-D batched matmul is also exposed through self.matmul_nd().
        return self.mm(right)

    def _gemv_with_math_policy[fast_tree: Bool](
        ref self, vector: Tensor[1, HostStorage]
    ) -> TensorResult[Tensor[1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 2:
        var matrix_proof = self._checked_current_layout()
        if not matrix_proof.is_ok():
            return TensorResult[Tensor[1, HostStorage]](matrix_proof.status)
        var vector_proof = vector._checked_current_layout()
        if not vector_proof.is_ok():
            return TensorResult[Tensor[1, HostStorage]](vector_proof.status)
        var matrix_layout = matrix_proof.payload.value()
        var vector_layout = vector_proof.payload.value()
        var m = matrix_layout.shape.dims.get(0)
        var k = matrix_layout.shape.dims.get(1)
        if k != vector_layout.shape.numel():
            return TensorResult[Tensor[1, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var out_dims = StaticRankArray[1](m)
        var out_shape = TensorShape[1].checked(out_dims)
        if not out_shape.is_ok():
            return TensorResult[Tensor[1, HostStorage]](out_shape.status)
        var allocated = Tensor[1, HostStorage]._empty_for_complete_overwrite_checked(
            out_shape.payload.value()
        )
        if not allocated.is_ok():
            return TensorResult[Tensor[1, HostStorage]](allocated.status)
        var output = allocated.payload.take()
        var lhs = self.impl.storage.borrow_span()
        var vec = vector.impl.storage.borrow_span()
        var dst = output.impl.storage.borrow_mut_span()
        # The SIMD K-axis horizontal reduction reassociates additions.
        # Keep it opt-in; the default scalar recurrence is strictly ordered.
        if fast_tree and matrix_layout.is_c_contiguous() and vector_layout.is_c_contiguous():
            if matrix_layout.is_exhaustive() and vector_layout.is_exhaustive():
                var matrix_run = lhs[
                    matrix_layout.span_lo:matrix_layout.span_lo + matrix_layout.shape.numel()
                ]
                var vector_run = vec[
                    vector_layout.span_lo:vector_layout.span_lo + vector_layout.shape.numel()
                ]
                var fast_status = cpu_gemv_checked_fp32[simd_width_of[DType.float32]()](
                    dst, matrix_run, vector_run, m, k
                )
                if fast_status != TensorStatus.OK:
                    return TensorResult[Tensor[1, HostStorage]](fast_status)
                return output._finish_complete_overwrite()
        var matrix_row_stride = matrix_layout.strides.get(0)
        var matrix_col_stride = matrix_layout.strides.get(1)
        var vector_stride = vector_layout.strides.get(0)
        for row in range(m):
            var acc = Float32(0.0)
            var start = matrix_layout.offset + row * matrix_row_stride
            for col in range(k):
                acc += lhs[start + col * matrix_col_stride] * vec[
                    vector_layout.offset + col * vector_stride
                ]
            dst[row] = acc
        return output._finish_complete_overwrite()

    def gemv(
        ref self, vector: Tensor[1, HostStorage]
    ) -> TensorResult[Tensor[1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 2:
        # Default: C-order accumulation in the K axis, no reassociation.
        return self._gemv_with_math_policy[False](vector)

    def gemv_fast(
        ref self, vector: Tensor[1, HostStorage]
    ) -> TensorResult[Tensor[1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 2:
        # Explicit opt-in: SIMD horizontal reduction may change last bits.
        return self._gemv_with_math_policy[True](vector)

    def _dot_with_math_policy[fast_tree: Bool](
        ref self, other: Tensor[1, HostStorage]
    ) -> TensorResult[Float32] where identical(Self.S, HostStorage) and Self.rank == 1:
        # Matrix/vector primitives are public methods of Tensor as well as
        # convenient functions in ops/matrix. One checked layout proof per
        # operand; only explicit fast-tree mode uses the tuned SIMD dot,
        # while the strict default uses an ordered direct host traversal.
        var left_checked = self._checked_current_layout()
        if not left_checked.is_ok():
            return TensorResult[Float32](left_checked.status)
        var right_checked = other._checked_current_layout()
        if not right_checked.is_ok():
            return TensorResult[Float32](right_checked.status)
        var a = left_checked.payload.value()
        var b = right_checked.payload.value()
        var count = a.shape.numel()
        if count != b.shape.numel():
            return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
        var lhs = self.impl.storage.borrow_span()
        var rhs = other.impl.storage.borrow_span()
        # Only explicit fast-tree mode uses SIMD lane reduction; the default
        # dot() follows the same scalar K order for contiguous and strided.
        if fast_tree and a.is_c_contiguous() and b.is_c_contiguous():
            if a.is_exhaustive() and b.is_exhaustive():
                var left_run = lhs[a.span_lo:a.span_lo + count]
                var right_run = rhs[b.span_lo:b.span_lo + count]
                return cpu_dot_checked_fp32[simd_width_of[DType.float32]()](
                    left_run, right_run,
                )
        var acc = Float32(0.0)
        for i in range(count):
            acc += lhs[a.offset + i * a.strides.get(0)] * rhs[b.offset + i * b.strides.get(0)]
        return TensorResult[Float32](acc)

    def dot(
        ref self, other: Tensor[1, HostStorage]
    ) -> TensorResult[Float32] where identical(Self.S, HostStorage) and Self.rank == 1:
        return self._dot_with_math_policy[False](other)

    def dot_fast(
        ref self, other: Tensor[1, HostStorage]
    ) -> TensorResult[Float32] where identical(Self.S, HostStorage) and Self.rank == 1:
        # Explicit opt-in for SIMD reassociation without changing Float32.
        return self._dot_with_math_policy[True](other)

    def _scan_host[operation: Int](
        ref self, axis: Int
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        # One FP32 prefix-scan implementation for all four operations. The
        # opcode/rank/storage are AOT-specialized and source bounds are
        # validated once. This is an axis-wise ordered scan, not a reduction.
        comptime assert operation >= 0 and operation <= 5, "invalid FP32 scan opcode"
        var checked = self._checked_current_layout()
        if not checked.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](checked.status)
        var chosen = normalize_axis[Self.rank](axis)
        if not chosen.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](chosen.status)
        var target_axis = chosen.payload.value()
        var mapping = checked.payload.value()
        var allocated = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(mapping.shape)
        if not allocated.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](allocated.status)
        var out = allocated.payload.take()
        var src = self.impl.storage.borrow_span()
        var dst = out.impl.storage.borrow_mut_span()
        # Delegate the ordered, rank-static prefix recurrence to the CPU
        # kernel layer. The facade only validates, allocates and publishes.
        comptime if Self.rank > 0:
            var scan_status = cpu_scan_strided_fp32[operation, Self.rank](
                src, dst, mapping, target_axis,
                out.impl.layout.strides.get(target_axis),
            )
            if scan_status != TensorStatus.OK:
                return TensorResult[Tensor[Self.rank, HostStorage]](scan_status)
            return out._finish_complete_overwrite()
        else:
            # A scalar has no scan axis; normalization already rejected it.
            return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.INVALID_AXIS)

    def cumsum(ref self, axis: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._scan_host[0](axis)

    def cumprod(ref self, axis: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._scan_host[1](axis)

    def cummin(ref self, axis: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._scan_host[2](axis)

    def cummax(ref self, axis: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._scan_host[3](axis)

    # Exclusive prefix scans write the identity first on each line. They
    # share the same allocation and stride-specialized kernel as inclusive
    # scans: no temporary inclusive result or shift/copy is materialized.
    def exclusive_cumsum(ref self, axis: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._scan_host[4](axis)

    def exclusive_cumprod(ref self, axis: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._scan_host[5](axis)

    # Public Tensor-centric scalar reduction spelling. The old *_all
    # methods remain available and own the algorithm and numerical policy.
    def sum(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        return self.sum_all()

    def mean(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        return self.mean_all()

    def prod(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        return self.prod_all()

    def min(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        return self.min_all()

    def max(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        return self.max_all()

    def argmin(self) -> TensorResult[Int] where identical(Self.S, HostStorage):
        return self.argmin_all()

    def argmax(self) -> TensorResult[Int] where identical(Self.S, HostStorage):
        return self.argmax_all()

    def _sum_from_validated_layout_fast(
        ref self, mapping: TensorLayout[Self.rank]
    ) -> Float32 where identical(Self.S, HostStorage):
        # INTERNAL: only a canonical owner-validated mapping may enter. This
        # unchecked leaf is shared by sum_all_fast and mean_all_fast; no redundant
        # descriptor proof or TensorResult is built in the hot reduction.
        var source = self.impl.storage.borrow_span()
        if mapping.is_exhaustive():
            if mapping.is_c_contiguous() or mapping.is_f_contiguous():
                var run = source[mapping.span_lo:mapping.span_lo + mapping.shape.numel()]
                # FAST_TREE may combine disjoint shards; STRICT must preserve
                # the original C-index traversal and never enter this path.
                var reduced = cpu_sum_contiguous_parallel_fast[
                    simd_width_of[DType.float32]()
                ](run)
                if reduced.is_ok():
                    return reduced.payload.value()
                # A rejected parallel dispatch cannot poison a read-only
                # source; the existing deterministic SIMD leaf is safe.
                return cpu_sum_contiguous[simd_width_of[DType.float32]()](run)
        # FAST_TREE may also split large noncontiguous mappings into
        # independent logical C-index ranges. The original serial traversal
        # remains as a safe fallback when the parallel substrate is rejected.
        if mapping.shape.numel() >= 1048576:
            var parallel_result = cpu_sum_strided_parallel_fast[Self.rank](
                source, mapping,
            )
            if parallel_result.is_ok():
                return parallel_result.payload.value()
        var cursor = NDIndexIterator[Self.rank](mapping.shape)
        var total = Float32(0.0)
        while not cursor.done():
            total += source[mapping.physical_offset_unchecked(cursor.coordinates)]
            cursor.advance()
        return total

    def _sum_from_validated_layout_strict(
        ref self, mapping: TensorLayout[Self.rank]
    ) -> Float32 where identical(Self.S, HostStorage):
        # INTERNAL: logical C-index order, with no SIMD tree reassociation.
        # Strict compiler FP reassociation policy is a separate AOT contract.
        var source = self.impl.storage.borrow_span()
        if mapping.is_c_contiguous() and mapping.is_exhaustive():
            return cpu_sum_c_strict_fp32(
                source, mapping.span_lo, mapping.shape.numel(),
            )
        var cursor = NDIndexIterator[Self.rank](mapping.shape)
        var total = Float32(0.0)
        while not cursor.done():
            total += source[mapping.physical_offset_unchecked(cursor.coordinates)]
            cursor.advance()
        return total

    def sum_all(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        # Consistent with default_fp32_math_mode(): STRICT_FP32 is the default.
        return self.sum_all_strict()

    def sum_all_fast(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        # Explicit opt-in; parallel/SIMD tree may change reduction association.
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Float32](proof.status)
        return TensorResult[Float32](self._sum_from_validated_layout_fast(proof.payload.value()))

    def sum_all_strict(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Float32](proof.status)
        return TensorResult[Float32](self._sum_from_validated_layout_strict(proof.payload.value()))

    def mean_all(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        return self.mean_all_strict()

    def mean_all_fast(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        # The faster tree is a caller-selected precision/association policy.
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Float32](proof.status)
        var trusted = proof.payload.value()
        var count = trusted.shape.numel()
        if count == 0:
            return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
        return TensorResult[Float32](
            self._sum_from_validated_layout_fast(trusted) / Float32(count)
        )

    def mean_all_strict(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Float32](proof.status)
        var trusted = proof.payload.value()
        var count = trusted.shape.numel()
        if count == 0:
            return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
        return TensorResult[Float32](
            self._sum_from_validated_layout_strict(trusted) / Float32(count)
        )

    # Scalar reductions for the owning HostStorage facade. Rank-0 is one
    # element, and rank-1..8 use the same checked fixed-rank cursor. These
    # methods are also the canonical fast HostStorage implementations used
    # by ops/reduction; no second reduction loop is needed in the public API.
    def prod_all(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Float32](proof.status)
        var mapping = proof.payload.value()
        var values = self.impl.storage.borrow_span()
        if mapping.is_c_contiguous() and mapping.is_exhaustive():
            return TensorResult[Float32](cpu_prod_c_fp32(
                values, mapping.span_lo, mapping.shape.numel(),
            ))
        var cursor = NDIndexIterator[Self.rank](mapping.shape)
        var result = Float32(1.0)
        while not cursor.done():
            result *= values[mapping.physical_offset_unchecked(cursor.coordinates)]
            cursor.advance()
        return TensorResult[Float32](result)

    def _extremum_all[find_min: Bool](self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Float32](proof.status)
        var mapping = proof.payload.value()
        if mapping.shape.numel() == 0:
            return TensorResult[Float32](TensorStatus.SIZE_MISMATCH)
        var data = self.impl.storage.borrow_span()
        if mapping.is_c_contiguous() and mapping.is_exhaustive():
            return TensorResult[Float32](cpu_extremum_c_fp32[find_min](
                data, mapping.span_lo, mapping.shape.numel(),
            ))
        var cursor = NDIndexIterator[Self.rank](mapping.shape)
        var best = data[mapping.physical_offset_unchecked(cursor.coordinates)]
        if best != best:
            return TensorResult[Float32](best)
        cursor.advance()
        while not cursor.done():
            var value = data[mapping.physical_offset_unchecked(cursor.coordinates)]
            if value != value:
                return TensorResult[Float32](value)
            comptime if find_min:
                best = fp32_minimum(best, value)
            else:
                best = fp32_maximum(best, value)
            cursor.advance()
        return TensorResult[Float32](best)

    def min_all(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        return self._extremum_all[True]()

    def max_all(self) -> TensorResult[Float32] where identical(Self.S, HostStorage):
        return self._extremum_all[False]()

    def _arg_extremum_all[find_min: Bool](self) -> TensorResult[Int] where identical(Self.S, HostStorage):
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Int](proof.status)
        var mapping = proof.payload.value()
        if mapping.shape.numel() == 0:
            return TensorResult[Int](TensorStatus.SIZE_MISMATCH)
        var data = self.impl.storage.borrow_span()
        if mapping.is_c_contiguous() and mapping.is_exhaustive():
            return TensorResult[Int](cpu_argextremum_c_fp32[find_min](
                data, mapping.span_lo, mapping.shape.numel(),
            ))
        var cursor = NDIndexIterator[Self.rank](mapping.shape)
        var best = data[mapping.physical_offset_unchecked(cursor.coordinates)]
        if best != best:
            return TensorResult[Int](0)
        var best_index = 0
        cursor.advance()
        var linear_index = 1
        while not cursor.done():
            var value = data[mapping.physical_offset_unchecked(cursor.coordinates)]
            if value != value:
                return TensorResult[Int](linear_index)
            comptime if find_min:
                if fp32_argmin_prefers(value, best):
                    best = value
                    best_index = linear_index
            else:
                if fp32_argmax_prefers(value, best):
                    best = value
                    best_index = linear_index
            linear_index += 1
            cursor.advance()
        return TensorResult[Int](best_index)

    def argmin_all(self) -> TensorResult[Int] where identical(Self.S, HostStorage):
        return self._arg_extremum_all[True]()

    def argmax_all(self) -> TensorResult[Int] where identical(Self.S, HostStorage):
        return self._arg_extremum_all[False]()

    # Rank-preserving axis reductions live on the owning Tensor facade.
    # The public ops/reduction functions delegate here for HostStorage;
    # there is only one host-owned FP32 hot loop for these operations.
    # Empty reduced axes have sum/product identities, whereas extrema and
    # means have no result. Rank zero has no valid reduction axis.
    def _reduce_axis_keepdims_host[operation: Int](
        ref self, axis: Int
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        comptime assert operation >= 0 and operation <= 4, "FP32 axis reduction opcode must be 0..4"
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](proof.status)
        # The operation is checked statically above; no runtime opcode gate.
        var axis_result = normalize_axis[Self.rank](axis)
        if not axis_result.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](axis_result.status)
        var reduced = axis_result.payload.value()
        var source_layout = proof.payload.value()
        var length = source_layout.shape.dims.get(reduced)
        var result_dims = source_layout.shape.dims.copy()
        result_dims.set(reduced, 1)
        var result_shape = TensorShape[Self.rank].checked(result_dims)
        if not result_shape.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](result_shape.status)
        # Empty min/max/mean is invalid only when an output cell actually
        # needs a value. E.g. reduction of (0,0) on its second axis produces
        # a valid empty (0,1) output and must not read uninitialized data.
        if length == 0 and operation >= 2 and result_shape.payload.value().numel() > 0:
            return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var allocated = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(
            result_shape.payload.value()
        )
        if not allocated.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](allocated.status)
        var result = allocated.payload.take()
        # An empty output has no FP32 cells to reduce or publish. In
        # particular, a valid empty shape may carry very large nonzero
        # dimensions whose intermediate outer/inner products exceed Int.MAX.
        # Do not calculate those products in the dense fast path: the
        # already-checked zero-element owner is complete without a kernel.
        if result_shape.payload.value().numel() == 0:
            return result._finish_complete_overwrite()
        var source_data = self.impl.storage.borrow_span()
        var output_data = result.impl.storage.borrow_mut_span()
        if source_layout.is_c_contiguous() and source_layout.is_exhaustive():
            var outer = 1
            var inner = 1
            comptime for dim_index in range(Self.rank):
                if dim_index < reduced:
                    outer *= source_layout.shape.dims.get(dim_index)
                elif dim_index > reduced:
                    inner *= source_layout.shape.dims.get(dim_index)
            var kernel_status = cpu_reduce_c_axis_block_fp32[operation](
                source_data, output_data, source_layout.span_lo,
                outer, length, inner,
            )
            if kernel_status != TensorStatus.OK:
                return TensorResult[Tensor[Self.rank, HostStorage]](kernel_status)
            return result._finish_complete_overwrite()
        var cursor = NDIndexIterator[Self.rank](result_shape.payload.value())
        var reduced_pitch = source_layout.strides.get(reduced)
        while not cursor.done():
            # The output cursor fixes all nonreduced axes at once.
            var source_origin = source_layout.offset
            comptime for k in range(Self.rank):
                source_origin += cursor.coordinates.get(k) * source_layout.strides.get(k)
            var acc = cpu_reduce_strided_fp32[operation](
                source_data, source_origin, reduced_pitch, length
            )
            output_data[cursor.produced] = acc
            cursor.advance()
        return result._finish_complete_overwrite()

    # Shape-specialized rank-reducing FP32 reductions. Every resulting rank
    # is known at compile time (0..8); no dynamic-rank Tensor exists. Axis
    # normalization, duplicates, empty semantics and allocation are checked
    # once before any unchecked element traversal. This is a SINGLE dataflow
    # operation on the existing owning Tensor type, not a secondary executor.
    def _reduce_axes_removed_host[operation: Int, axes_count: Int](
        ref self, axes: StaticRankArray[axes_count]
    ) -> TensorResult[Tensor[Self.rank - axes_count, HostStorage]] where identical(Self.S, HostStorage):
        comptime assert operation >= 0 and operation <= 4, "invalid FP32 reduction"
        comptime assert axes_count >= 0 and axes_count <= Self.rank, "invalid reduction rank"
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[Tensor[Self.rank - axes_count, HostStorage]](checked_source.status)
        var layout = checked_source.payload.value()
        var chosen = Array[Bool, 8](fill=False)
        var axes_normalized = StaticRankArray[axes_count](0)
        for i in range(axes_count):
            var parsed = normalize_axis[Self.rank](axes.get(i))
            if not parsed.is_ok():
                return TensorResult[Tensor[Self.rank - axes_count, HostStorage]](parsed.status)
            var axis = parsed.payload.value()
            if chosen[axis]:
                return TensorResult[Tensor[Self.rank - axes_count, HostStorage]](TensorStatus.INVALID_AXIS)
            chosen[axis] = True
            axes_normalized.set(i, axis)
        # Zero selected axes mean an identity *copy*, never a borrowed alias.
        comptime if axes_count == 0:
            return self.clone()
        var dims = StaticRankArray[Self.rank - axes_count](0)
        var out_axis = 0
        comptime for src_axis in range(Self.rank):
            if not chosen[src_axis]:
                dims.set(out_axis, layout.shape.dims.get(src_axis))
                out_axis += 1
        var checked_shape = TensorShape[Self.rank - axes_count].checked(dims)
        if not checked_shape.is_ok():
            return TensorResult[Tensor[Self.rank - axes_count, HostStorage]](checked_shape.status)
        var output_shape = checked_shape.payload.value()
        # A zero-element output has nothing to initialize, including for min,
        # max and mean. No uninitialized element will become readable.
        if output_shape.is_empty():
            return Tensor[Self.rank - axes_count, HostStorage]._empty_for_complete_overwrite_checked(output_shape)
        # An empty selected dimension makes the entire reduction domain
        # empty, independently of how large other selected dimensions are.
        # Detect zero FIRST: multiplying huge nonzero extents before a later
        # zero could spuriously overflow a mathematically empty reduction.
        var count = 1
        var empty_reduction = False
        for i in range(axes_count):
            if layout.shape.dims.get(axes_normalized.get(i)) == 0:
                empty_reduction = True
        if empty_reduction:
            count = 0
        else:
            for i in range(axes_count):
                var extent = layout.shape.dims.get(axes_normalized.get(i))
                var multiplied = checked_mul_nonnegative(count, extent)
                if not multiplied.is_ok():
                    return TensorResult[Tensor[Self.rank - axes_count, HostStorage]](multiplied.status)
                count = multiplied.payload.value()
        comptime if operation >= 2:
            if count == 0:
                return TensorResult[Tensor[Self.rank - axes_count, HostStorage]](TensorStatus.SIZE_MISMATCH)
        # All trailing reduced axes of a C-contiguous source form one dense
        # physical FP32 row per output cell. Dispatch ONE specialized kernel
        # for this entire layout class instead of reconstructing up to eight
        # coordinates for every source or destination element. The operands
        # and output are already owner-validated; the kernel is unchecked.
        var trailing_axes = True
        comptime for src_axis in range(Self.rank):
            if chosen[src_axis] != (src_axis >= Self.rank - axes_count):
                trailing_axes = False
        if (trailing_axes and layout.is_c_contiguous()
            and layout.is_exhaustive()):
            var dense = Tensor[Self.rank - axes_count, HostStorage]._empty_for_complete_overwrite_checked(
                output_shape
            )
            if not dense.is_ok():
                return TensorResult[Tensor[Self.rank - axes_count, HostStorage]](dense.status)
            var reduced = dense.payload.take()
            var kernel_status = cpu_reduce_c_trailing_axes_fp32[operation](
                self.impl.storage.borrow_span(), reduced.impl.storage.borrow_mut_span(),
                layout.span_lo, output_shape.numel(), count,
            )
            if kernel_status != TensorStatus.OK:
                return TensorResult[Tensor[Self.rank - axes_count, HostStorage]](kernel_status)
            return reduced._finish_complete_overwrite()
        # Consecutive reduced axes form one C-order reduction block even
        # when retained axes follow them (leading/middle-axis reductions).
        # Rank is fixed, axes were normalized, output cells are disjoint, and
        # the checked owner proves the full physical dense allocation span.
        var first_reduced = Self.rank
        var last_reduced = -1
        comptime for src_axis in range(Self.rank):
            if chosen[src_axis]:
                if first_reduced == Self.rank:
                    first_reduced = src_axis
                last_reduced = src_axis
        var contiguous_axes = True
        comptime for src_axis in range(Self.rank):
            if src_axis >= first_reduced and src_axis <= last_reduced and not chosen[src_axis]:
                contiguous_axes = False
        if contiguous_axes and layout.is_c_contiguous() and layout.is_exhaustive():
            var outer = 1
            var inner = 1
            comptime for src_axis in range(Self.rank):
                if src_axis < first_reduced:
                    outer *= layout.shape.dims.get(src_axis)
                elif src_axis > last_reduced:
                    inner *= layout.shape.dims.get(src_axis)
            var created = Tensor[Self.rank - axes_count, HostStorage]._empty_for_complete_overwrite_checked(output_shape)
            if not created.is_ok():
                return TensorResult[Tensor[Self.rank - axes_count, HostStorage]](created.status)
            var output = created.payload.take()
            var kernel_status = cpu_reduce_c_axis_block_fp32[operation](
                self.impl.storage.borrow_span(), output.impl.storage.borrow_mut_span(),
                layout.span_lo, outer, count, inner,
            )
            if kernel_status != TensorStatus.OK:
                return TensorResult[Tensor[Self.rank - axes_count, HostStorage]](kernel_status)
            return output._finish_complete_overwrite()
        # For exactly one selected nontrailing axis, accumulate locally per
        # output cell and avoid repeated read/modify/write of the output.
        comptime if axes_count == 1:
            var allocated = Tensor[Self.rank - 1, HostStorage]._empty_for_complete_overwrite_checked(output_shape)
            if not allocated.is_ok():
                return TensorResult[Tensor[Self.rank - 1, HostStorage]](allocated.status)
            var output = allocated.payload.take()
            var source = self.impl.storage.borrow_span()
            var destination = output.impl.storage.borrow_mut_span()
            var axis = axes_normalized.get(0)
            var axis_stride = layout.strides.get(axis)
            var cursor = NDIndexIterator[Self.rank - 1](output_shape)
            while not cursor.done():
                var origin = layout.offset
                var source_axis = 0
                comptime for j in range(Self.rank - 1):
                    if source_axis == axis:
                        source_axis += 1
                    origin += cursor.coordinates.get(j) * layout.strides.get(source_axis)
                    source_axis += 1
                var accumulator = cpu_reduce_strided_fp32[operation](
                    source, origin, axis_stride, count
                )
                destination[cursor.produced] = accumulator
                cursor.advance()
            return output._finish_complete_overwrite()
        # Multiple selected axes: initialize each output cell to the reduction
        # identity and traverse the source exactly once in logical C order.
        var identity = Float32(0.0)
        comptime if operation == 1:
            identity = Float32(1.0)
        elif operation == 2:
            identity = Float32(FloatLiteral.infinity)
        elif operation == 3:
            identity = -Float32(FloatLiteral.infinity)
        var allocated = Tensor[Self.rank - axes_count, HostStorage].full_checked(output_shape, identity)
        if not allocated.is_ok():
            return allocated
        var output = allocated.payload.take()
        var source = self.impl.storage.borrow_span()
        var destination = output.impl.storage.borrow_mut_span()
        var target_layout = output.impl.layout
        var cursor = NDIndexIterator[Self.rank](layout.shape)
        while not cursor.done():
            var coordinates = StaticRankArray[Self.rank - axes_count](0)
            var target_axis = 0
            comptime for source_axis in range(Self.rank):
                if not chosen[source_axis]:
                    coordinates.set(target_axis, cursor.coordinates.get(source_axis))
                    target_axis += 1
            var dst_at = target_layout.physical_offset_unchecked(coordinates)
            var item = source[layout.physical_offset_unchecked(cursor.coordinates)]
            comptime if operation == 0 or operation == 4:
                destination[dst_at] += item
            elif operation == 1:
                destination[dst_at] *= item
            elif operation == 2:
                destination[dst_at] = fp32_minimum(destination[dst_at], item)
            else:
                destination[dst_at] = fp32_maximum(destination[dst_at], item)
            cursor.advance()
        comptime if operation == 4:
            var denominator = Float32(count)
            for i in range(output_shape.numel()):
                destination[i] /= denominator
        return output._finish_complete_overwrite()

    def sum_axes[axes_count: Int](
        ref self, axes: StaticRankArray[axes_count]
    ) -> TensorResult[Tensor[Self.rank - axes_count, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axes_removed_host[0, axes_count](axes)

    def prod_axes[axes_count: Int](
        ref self, axes: StaticRankArray[axes_count]
    ) -> TensorResult[Tensor[Self.rank - axes_count, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axes_removed_host[1, axes_count](axes)

    def min_axes[axes_count: Int](
        ref self, axes: StaticRankArray[axes_count]
    ) -> TensorResult[Tensor[Self.rank - axes_count, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axes_removed_host[2, axes_count](axes)

    def max_axes[axes_count: Int](
        ref self, axes: StaticRankArray[axes_count]
    ) -> TensorResult[Tensor[Self.rank - axes_count, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axes_removed_host[3, axes_count](axes)

    def mean_axes[axes_count: Int](
        ref self, axes: StaticRankArray[axes_count]
    ) -> TensorResult[Tensor[Self.rank - axes_count, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axes_removed_host[4, axes_count](axes)

    def sum_axis(ref self, axis: Int) -> TensorResult[Tensor[Self.rank - 1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank >= 1:
        var axes = StaticRankArray[1](axis)
        return self.sum_axes[1](axes)

    def prod_axis(ref self, axis: Int) -> TensorResult[Tensor[Self.rank - 1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank >= 1:
        var axes = StaticRankArray[1](axis)
        return self.prod_axes[1](axes)

    def min_axis(ref self, axis: Int) -> TensorResult[Tensor[Self.rank - 1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank >= 1:
        var axes = StaticRankArray[1](axis)
        return self.min_axes[1](axes)

    def max_axis(ref self, axis: Int) -> TensorResult[Tensor[Self.rank - 1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank >= 1:
        var axes = StaticRankArray[1](axis)
        return self.max_axes[1](axes)

    def mean_axis(ref self, axis: Int) -> TensorResult[Tensor[Self.rank - 1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank >= 1:
        var axes = StaticRankArray[1](axis)
        return self.mean_axes[1](axes)

    def sum_axis_keepdims(ref self, axis: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axis_keepdims_host[0](axis)

    def prod_axis_keepdims(ref self, axis: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axis_keepdims_host[1](axis)

    def min_axis_keepdims(ref self, axis: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axis_keepdims_host[2](axis)

    def max_axis_keepdims(ref self, axis: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axis_keepdims_host[3](axis)

    def mean_axis_keepdims(ref self, axis: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axis_keepdims_host[4](axis)

    # Multi-axis rank-preserving reduction, with static axis count and
    # runtime-validated axes. Does not create a dynamic-rank Tensor family.
    def _reduce_axes_keepdims_host[operation: Int, axes_count: Int](
        ref self, axes: StaticRankArray[axes_count]
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        comptime assert operation >= 0 and operation <= 4, "invalid FP32 reduction"
        comptime assert axes_count >= 0 and axes_count <= Self.rank, "axis count exceeds static rank"
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](proof.status)
        var selected = Array[Bool, 8](fill=False)
        var normalized = StaticRankArray[axes_count](0)
        for i in range(axes_count):
            var axis = normalize_axis[Self.rank](axes.get(i))
            if not axis.is_ok():
                return TensorResult[Tensor[Self.rank, HostStorage]](axis.status)
            var k = axis.payload.value()
            if selected[k]:
                return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.INVALID_AXIS)
            selected[k] = True
            normalized.set(i, k)
        comptime if axes_count == 0:
            return self.clone()
        comptime if axes_count == 1:
            return self._reduce_axis_keepdims_host[operation](normalized.get(0))
        var safe = proof.payload.value()
        var dims = safe.shape.dims.copy()
        for i in range(axes_count):
            dims.set(normalized.get(i), 1)
        var shaped = TensorShape[Self.rank].checked(dims)
        if not shaped.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](shaped.status)
        var out_shape = shaped.payload.value()
        if out_shape.numel() == 0:
            # No result cell needs initialization, including min/max/mean.
            return Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(out_shape)
        # A zero reduced axis dominates the entire reduction domain. Scan
        # first so that huge extents on earlier axes cannot cause spurious
        # overflow before the zero is encountered. The rank-dropping path
        # uses this same two-pass checked policy.
        var empty_reduction = False
        for i in range(axes_count):
            if safe.shape.dims.get(normalized.get(i)) == 0:
                empty_reduction = True
        var reduce_count = 0
        if not empty_reduction:
            reduce_count = 1
            for i in range(axes_count):
                var extent = safe.shape.dims.get(normalized.get(i))
                var next_count = checked_mul_nonnegative(reduce_count, extent)
                if not next_count.is_ok():
                    return TensorResult[Tensor[Self.rank, HostStorage]](next_count.status)
                reduce_count = next_count.payload.value()
        comptime if operation >= 2:
            if reduce_count == 0:
                return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        # Keepdims uses the same dense-row lowering as the rank-removing
        # facade: singleton reduced dimensions do not alter the physical
        # C-order output sequence. Nontrailing and signed-stride views keep
        # the existing general reduction path.
        var trailing_axes = True
        comptime for src_axis in range(Self.rank):
            if selected[src_axis] != (src_axis >= Self.rank - axes_count):
                trailing_axes = False
        if (trailing_axes and safe.is_c_contiguous()
            and safe.is_exhaustive()):
            var prepared = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(out_shape)
            if not prepared.is_ok():
                return TensorResult[Tensor[Self.rank, HostStorage]](prepared.status)
            var result = prepared.payload.take()
            var kernel_status = cpu_reduce_c_trailing_axes_fp32[operation](
                self.impl.storage.borrow_span(), result.impl.storage.borrow_mut_span(),
                safe.span_lo, out_shape.numel(), reduce_count,
            )
            if kernel_status != TensorStatus.OK:
                return TensorResult[Tensor[Self.rank, HostStorage]](kernel_status)
            return result._finish_complete_overwrite()
        # The exact same block lowering applies to keepdims because the
        # reduced axes have size 1 in the output; its physical order matches
        # the shape with reduced axes removed.
        var first_reduced = Self.rank
        var last_reduced = -1
        comptime for src_axis in range(Self.rank):
            if selected[src_axis]:
                if first_reduced == Self.rank:
                    first_reduced = src_axis
                last_reduced = src_axis
        var contiguous_axes = True
        comptime for src_axis in range(Self.rank):
            if src_axis >= first_reduced and src_axis <= last_reduced and not selected[src_axis]:
                contiguous_axes = False
        if contiguous_axes and safe.is_c_contiguous() and safe.is_exhaustive():
            var outer = 1
            var inner = 1
            comptime for src_axis in range(Self.rank):
                if src_axis < first_reduced:
                    outer *= safe.shape.dims.get(src_axis)
                elif src_axis > last_reduced:
                    inner *= safe.shape.dims.get(src_axis)
            var created = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(out_shape)
            if not created.is_ok():
                return TensorResult[Tensor[Self.rank, HostStorage]](created.status)
            var output = created.payload.take()
            var kernel_status = cpu_reduce_c_axis_block_fp32[operation](
                self.impl.storage.borrow_span(), output.impl.storage.borrow_mut_span(),
                safe.span_lo, outer, reduce_count, inner,
            )
            if kernel_status != TensorStatus.OK:
                return TensorResult[Tensor[Self.rank, HostStorage]](kernel_status)
            return output._finish_complete_overwrite()
        var initial = Float32(0.0)
        comptime if operation == 1:
            initial = Float32(1.0)
        elif operation == 2:
            initial = Float32(FloatLiteral.infinity)
        elif operation == 3:
            initial = -Float32(FloatLiteral.infinity)
        var built = Tensor[Self.rank, HostStorage].full_checked(out_shape, initial)
        if not built.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](built.status)
        var result = built.payload.take()
        var src = self.impl.storage.borrow_span()
        var dst = result.impl.storage.borrow_mut_span()
        var out_layout = result.impl.layout
        var cursor = NDIndexIterator[Self.rank](safe.shape)
        while not cursor.done():
            var out_coord = cursor.coordinates.copy()
            for i in range(axes_count):
                out_coord.set(normalized.get(i), 0)
            var out_offset = out_layout.physical_offset_unchecked(out_coord)
            var value = src[safe.physical_offset_unchecked(cursor.coordinates)]
            comptime if operation == 0 or operation == 4:
                dst[out_offset] += value
            elif operation == 1:
                dst[out_offset] *= value
            elif operation == 2:
                dst[out_offset] = fp32_minimum(dst[out_offset], value)
            else:
                dst[out_offset] = fp32_maximum(dst[out_offset], value)
            cursor.advance()
        comptime if operation == 4:
            var scale = Float32(reduce_count)
            for i in range(out_shape.numel()):
                dst[i] /= scale
        return result._finish_complete_overwrite()

    def sum_axes_keepdims[axes_count: Int](ref self, axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axes_keepdims_host[0, axes_count](axes)

    def prod_axes_keepdims[axes_count: Int](ref self, axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axes_keepdims_host[1, axes_count](axes)

    def min_axes_keepdims[axes_count: Int](ref self, axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axes_keepdims_host[2, axes_count](axes)

    def max_axes_keepdims[axes_count: Int](ref self, axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axes_keepdims_host[3, axes_count](axes)

    def mean_axes_keepdims[axes_count: Int](ref self, axes: StaticRankArray[axes_count]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._reduce_axes_keepdims_host[4, axes_count](axes)

    def size(self, axis: Int) -> TensorResult[Int]:
        return self.dim(axis)

    # Basic slicing delegates to the checked metadata layer and never copies.
    def slice_axis(ref self, axis: Int, spec: SliceSpec) -> TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](checked_source.status)
        var result = slice_layout_axis(checked_source.payload.value(), axis, spec, self.impl.storage.capacity())
        if not result.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](result.status)
        # The mapping constructor has already validated its shape, physical
        # span and allocation bounds. Rebuilding the same proof via
        # from_span_checked() would add a redundant O(rank) pass.
        return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](
            TensorView[Self.rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), result.payload.take()
            )
        )

    def sliced_regions_disjoint(
        ref self, axis: Int, first: SliceSpec, second: SliceSpec,
    ) -> TensorResult[Bool] where identical(Self.S, HostStorage):
        # Both slices are derived from THIS Tensor and hence the very same
        # live HostStorage owner; this proof cannot be forged with two arbitrary
        # StorageProvenance IDs. Even with irregular/negative strides, a true
        # result means their physical allocation intervals do not intersect.
        # A false result is conservative: their elements may still be disjoint.
        var base = self._checked_current_layout()
        if not base.is_ok():
            return TensorResult[Bool](base.status)
        var capacity = self.impl.storage.capacity()
        var lhs = slice_layout_axis(base.payload.value(), axis, first, capacity)
        if not lhs.is_ok():
            return TensorResult[Bool](lhs.status)
        var rhs = slice_layout_axis(base.payload.value(), axis, second, capacity)
        if not rhs.is_ok():
            return TensorResult[Bool](rhs.status)
        return _owner_bound_mapping_disjoint[Self.rank, Self.rank](
            self.impl.storage, lhs.payload.value(), rhs.payload.value()
        )

    def storage_regions_disjoint[other_rank: Int](
        ref self, ref other: Tensor[other_rank, HostStorage],
    ) -> TensorResult[Bool] where identical(Self.S, HostStorage):
        # Cross-Tensor proof derived from the two live host allocation owners.
        # No caller-controllable allocation ID grants noalias here. A true
        # answer is suitable for conservative copy/split strategy selection,
        # not for manufacturing an unsafe_as_noalias pointer.
        var left = self._checked_current_layout()
        if not left.is_ok():
            return TensorResult[Bool](left.status)
        var right = other._checked_current_layout()
        if not right.is_ok():
            return TensorResult[Bool](right.status)
        return _two_owner_mappings_disjoint[Self.rank, other_rank](
            self.impl.storage, left.payload.value(),
            other.impl.storage, right.payload.value(),
        )

    # Mixed basic indexing performs one checked metadata transformation.
    # It only borrows the backing allocation; it never materializes output.
    def index_view[out_rank: Int, count: Int](
        ref self, terms: Array[BasicIndex, count]
    ) -> TensorResult[TensorView[out_rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        comptime assert out_rank >= 0 and out_rank <= 8, "owning index result rank must be 0..8"
        comptime assert count >= 1 and count <= 17, "a fixed Array index pack must contain 1..17 terms"
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[out_rank, origin_of(self.impl.storage)]](checked_source.status)
        var mapped = basic_index_layout[Self.rank, out_rank, count](
            checked_source.payload.value(), terms, self.impl.storage.capacity()
        )
        if not mapped.is_ok():
            return TensorResult[TensorView[out_rank, origin_of(self.impl.storage)]](mapped.status)
        # The mapping constructor has already validated its shape, physical
        # span and allocation bounds. Rebuilding the same proof via
        # from_span_checked() would add a redundant O(rank) pass.
        return TensorResult[TensorView[out_rank, origin_of(self.impl.storage)]](
            TensorView[out_rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), mapped.payload.take()
            )
        )

    def index_basic_as[out_rank: Int, *IndexTypes: AnyType](
        ref self, *terms: *IndexTypes
    ) -> TensorResult[TensorView[out_rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        # VariadicPack is heterogeneous. Each conversion is selected at
        # compile time; the existing checked layout constructor verifies
        # ellipsis expansion, axis consumption, rank and storage bounds.
        comptime count = terms.__len__()
        comptime assert count <= 17, "basic index pack must have at most 17 terms"
        comptime assert out_rank >= 0 and out_rank <= 8, "result rank must be 0..8"
        comptime if count == 0:
            # A zero-term index is an identity view, including rank-0 scalars.
            # Avoid constructing an unsupported zero-length fixed Array.
            comptime assert out_rank == Self.rank, "identity index must preserve rank"
            return self.view()
        else:
            var packed = Array[BasicIndex, count](fill=BasicIndex.full_slice())
            comptime for i in range(count):
                packed[i] = as_basic_index(terms[i])
            return self.index_view[out_rank, count](packed)

    # The preferred typed-index API infers the result rank at AOT compile
    # time, without a caller-supplied out_rank. Only Int/SliceSpec/NewAxis/
    # EllipsisIndex are accepted; runtime-tagged BasicIndex belongs to
    # index_view[out_rank] (which performs runtime rank validation).
    def index_basic[*IndexTypes: AnyType](
        ref self, *terms: *IndexTypes
    ) -> TensorResult[TensorView[inferred_basic_index_rank[Self.rank, *IndexTypes](), origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        comptime out_rank = inferred_basic_index_rank[Self.rank, *IndexTypes]()
        comptime assert out_rank >= 0 and out_rank <= 8, "invalid heterogeneous index signature"
        comptime count = terms.__len__()
        comptime assert count <= 17, "basic index pack must have at most 17 terms"
        comptime if count == 0:
            # Valid for every rank, including rank 0, and never materializes.
            return self.view()
        else:
            var packed = Array[BasicIndex, count](fill=BasicIndex.full_slice())
            comptime for i in range(count):
                packed[i] = as_basic_index(terms[i])
            return self.index_view[out_rank, count](packed)

    # Heterogeneous basic slicing is available directly from Tensor. The
    # integer-only __getitem__ overloads still return scalar FP32 results;
    # explicit slice/ellipsis/newaxis overloads return zero-copy borrows.
    # No new owning type or silent materialization is involved.
    def __getitem__(
        ref self, spec: SliceSpec,
    ) -> TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]] where Self.rank >= 1 and conforms_to(Self.S, HostSpanStorage):
        return self.index_basic(spec)

    def __getitem__(
        ref self, spec: EllipsisIndex,
    ) -> TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        return self.index_basic(spec)

    def __getitem__(
        ref self, spec: NewAxis,
    ) -> TensorResult[TensorView[Self.rank + 1, origin_of(self.impl.storage)]] where Self.rank < 8 and conforms_to(Self.S, HostSpanStorage):
        return self.index_basic(spec)

    # Partial integer indexing reduces rank by one for ranks 2..8. The
    # existing full-arity integer overloads still read/write a Float32 scalar.
    # All indexing is zero-copy and the resulting borrowed view is bounded
    # against the same owning storage that backs this Tensor.
    def __getitem__(
        ref self, index: Int,
    ) -> TensorResult[TensorView[Self.rank - 1, origin_of(self.impl.storage)]] where Self.rank >= 2 and conforms_to(Self.S, HostSpanStorage):
        return self.index_basic(index)

    def __setitem__(
        mut self, index: Int, value: Float32,
    ) -> TensorStatus where Self.rank >= 2 and conforms_to(Self.S, HostSpanStorage):
        var selected = self.index_basic_mut(index)
        if not selected.is_ok():
            return selected.status
        var writable = selected.payload.take()
        return writable.fill(value)

    # Slice assignment goes through the same canonical mutable-view proof,
    # then through TensorMutView.fill()'s vectorized/strided implementation.
    # An overlapping destination is rejected; no per-element revalidation.
    def __setitem__(
        mut self, spec: SliceSpec, value: Float32,
    ) -> TensorStatus where Self.rank >= 1 and conforms_to(Self.S, HostSpanStorage):
        var selected = self.index_basic_mut(spec)
        if not selected.is_ok():
            return selected.status
        var writable = selected.payload.take()
        return writable.fill(value)

    # A mutable slice is a capability, never a writable bit on an immutable
    # TensorView. Rebuild the source mapping and prove the selected mapping
    # unique BEFORE borrowing mutable host storage. In particular a zero-
    # stride expanded slice cannot accidentally admit parallel writes.
    def index_mut_view[out_rank: Int, count: Int](
        mut self, terms: Array[BasicIndex, count],
    ) -> TensorResult[TensorMutView[out_rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        comptime assert out_rank >= 0 and out_rank <= 8, "mutable owning index output rank must be 0..8"
        comptime assert count >= 1 and count <= 17, "a fixed Array index pack must contain 1..17 terms"
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[TensorMutView[out_rank, origin_of(self.impl.storage)]](source.status)
        var next_layout = basic_index_layout[Self.rank, out_rank, count](
            source.payload.value(), terms, self.impl.storage.capacity(),
        )
        if not next_layout.is_ok():
            return TensorResult[TensorMutView[out_rank, origin_of(self.impl.storage)]](next_layout.status)
        var selected = next_layout.payload.take()
        if not selected.is_proven_unique():
            return TensorResult[TensorMutView[out_rank, origin_of(self.impl.storage)]](TensorStatus.OVERLAPPING)
        return TensorResult[TensorMutView[out_rank, origin_of(self.impl.storage)]](
            TensorMutView[out_rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_mut_span(), selected,
            )
        )

    def index_basic_mut[*IndexTypes: AnyType](
        mut self, *terms: *IndexTypes,
    ) -> TensorResult[TensorMutView[inferred_basic_index_rank[Self.rank, *IndexTypes](), origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        comptime out_rank = inferred_basic_index_rank[Self.rank, *IndexTypes]()
        comptime assert out_rank >= 0 and out_rank <= 8, "invalid mutable basic index signature"
        comptime count = terms.__len__()
        comptime assert count <= 17, "basic index pack must have at most 17 terms"
        comptime if count == 0:
            return self.mut_view()
        else:
            var packed = Array[BasicIndex, count](fill=BasicIndex.full_slice())
            comptime for i in range(count):
                packed[i] = as_basic_index(terms[i])
            return self.index_mut_view[out_rank, count](packed)

    def item_checked(self) -> TensorResult[Float32]:
        return self.item()

    def set_item_checked(mut self, value: Float32) -> TensorStatus where conforms_to(Self.S, MutableTensorStorage):
        return self.set_item(value)

    # A deep clone, never an alias. Source and destination own distinct host
    # allocations and every destination element is written exactly once.
    def clone(ref self) -> TensorResult[Tensor[Self.rank, HostStorage]] where conforms_to(Self.S, HostSpanStorage):
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](checked_source.status)
        var safe_source = checked_source.payload.value()
        var count = safe_source.shape.numel()
        var target_layout = TensorLayout[Self.rank].c_contiguous(safe_source.shape, count)
        if not target_layout.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](target_layout.status)
        # This internal allocation is not exposed until every element has been
        # copied. Avoid touching memory twice with a preliminary zero-fill.
        var owned = HostStorage._allocate_for_complete_overwrite(count)
        if not owned.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](owned.status)
        var target = Tensor[Self.rank, HostStorage]._from_parts_unchecked(
            owned.payload.take(), target_layout.payload.value()
        )
        var source_span = self.impl.storage.borrow_span()
        var dest_span = target.impl.storage.borrow_mut_span()
        if safe_source.is_c_contiguous() and safe_source.is_exhaustive():
            var first = safe_source.span_lo
            if count > 0:
                # Both allocation owners are disjoint, so an FP32 memcpy is
                # legal; it also avoids bounds checks inside the hot path.
                var logical_source = source_span[first:first + count]
                unsafe_memcpy(
                    dest=dest_span.unsafe_ptr(),
                    src=logical_source.unsafe_ptr(),
                    count=count,
                )
            return target._finish_complete_overwrite()
        if count > 0 and safe_source.is_exhaustive() and is_reverse_c_contiguous(safe_source):
            # Reverse-dense FP32 data has a single physical run in the
            # opposite logical C order. Complete-overwrite allocation and
            # the fresh layout proof guarantee in-bounds, disjoint access.
            var sp = source_span.unsafe_ptr()
            var dp = dest_span.unsafe_ptr()
            var last_source = safe_source.span_lo + count - 1
            for linear in range(count):
                dp.unsafe_store(linear, sp.unsafe_load(last_source - linear)[0])
            return target._finish_complete_overwrite()
        # The input layout was checked, so its affine address calculation is
        # safe without building TensorResult for each copied FP32 element.
        var cursor = NDIndexIterator[Self.rank](safe_source.shape)
        var linear = 0
        while not cursor.done():
            var at = safe_source.physical_offset_unchecked(cursor.coordinates)
            dest_span[linear] = source_span[at]
            linear += 1
            cursor.advance()
        return target._finish_complete_overwrite()

    def contiguous_copy(
        ref self
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where conforms_to(Self.S, HostSpanStorage):
        # Always materialize; never silently return a borrow or alias.
        return self.clone()

    def contiguous_f_copy(
        ref self
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        # Always allocate an independent column-major owner, including for
        # already F-contiguous sources; never return a borrowed temporary.
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](proof.status)
        var mapping = proof.payload.value()
        var count = mapping.shape.numel()
        var output_layout = TensorLayout[Self.rank].f_contiguous(mapping.shape, count)
        if not output_layout.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](output_layout.status)
        var buffer = HostStorage._allocate_for_complete_overwrite(count)
        if not buffer.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](buffer.status)
        var output = Tensor[Self.rank, HostStorage]._from_parts_unchecked(
            buffer.payload.take(), output_layout.payload.value()
        )
        var src = self.impl.storage.borrow_span()
        var dst = output.impl.storage.borrow_mut_span()
        if mapping.is_f_contiguous() and mapping.is_exhaustive():
            if count > 0:
                var first = mapping.span_lo
                var source_run = src[first:first + count]
                unsafe_memcpy(dest=dst.unsafe_ptr(), src=source_run.unsafe_ptr(), count=count)
            return output._finish_complete_overwrite()
        if count > 0 and mapping.is_exhaustive() and is_reverse_f_contiguous(mapping):
            # F-order materialization of a fully reversed F-dense mapping.
            # Avoid N-D cursor/physical-offset overhead on every element.
            var sp = src.unsafe_ptr()
            var dp = dst.unsafe_ptr()
            var last_source = mapping.span_lo + count - 1
            for linear in range(count):
                dp.unsafe_store(linear, sp.unsafe_load(last_source - linear)[0])
            return output._finish_complete_overwrite()
        var cursor = NDIndexIterator[Self.rank](mapping.shape)
        var target = output_layout.payload.value()
        while not cursor.done():
            var src_at = mapping.physical_offset_unchecked(cursor.coordinates)
            var dst_at = target.physical_offset_unchecked(cursor.coordinates)
            dst[dst_at] = src[src_at]
            cursor.advance()
        return output._finish_complete_overwrite()

    def contiguous_copy_order(
        ref self, order: ContiguousOrder
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        # Unlike ensure-contiguous APIs, this explicit method always deep copies.
        if order == ContiguousOrder.C:
            return self.clone()
        if order == ContiguousOrder.F:
            return self.contiguous_f_copy()
        return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.NOT_REPRESENTABLE)

    def contiguous(
        ref self, order: ContiguousOrder
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        # Explicit materialization, not a view-or-copy API. Shared engine:
        # C/F implementation performs exactly one allocation and data copy.
        return self.contiguous_copy_order(order)

    # Owner publication after a successful checked host copy.  A subview
    # copy writes only its selected elements; it must NEVER restore a whole-
    # allocation readability certificate.  A full unique map may do so even
    # when it is F-order or a reversed permutation of the physical owner.
    def _finish_host_copy_from(mut self, dst: TensorLayout[Self.rank]) -> TensorStatus where identical(Self.S, HostStorage):
        if (dst.is_exhaustive() and dst.span_lo == 0
            and dst.shape.numel() == self.impl.storage.capacity()):
            var published = self.impl.storage._mark_complete_overwrite()
            if published != TensorStatus.OK:
                self.impl.storage._invalidate_after_failed_write()
            return published
        return TensorStatus.OK

    # Direct copies between proven-disjoint owning host allocations.
    def copy_from(mut self, ref source: Tensor[Self.rank, HostStorage]) -> TensorStatus where identical(Self.S, HostStorage):
        # Copying FROM an owner requires initialized data; copying TO a full
        # destination does not.  Rebuilding its descriptor/capacity proof
        # allows the copy to restore an owner poisoned by a failed Hybrid
        # write instead of requiring an unrelated zero-fill first.
        var dst_check = self.impl.layout.validated_snapshot(
            self.impl.storage.capacity()
        )
        if not dst_check.is_ok():
            return dst_check.status
        var src_check = source._checked_current_layout()
        if not src_check.is_ok():
            return src_check.status
        var dst_layout = dst_check.payload.value()
        var src_layout = src_check.payload.value()
        if not dst_layout.shape.same_extents(src_layout.shape):
            return TensorStatus.SIZE_MISMATCH
        if not dst_layout.is_proven_unique():
            return TensorStatus.OVERLAPPING
        # Full-owner copy may restore a poisoned allocation, but a partial
        # copy may not advertise success while leaving the rest unreadable.
        if not self.impl.storage.is_initialized():
            if (not dst_layout.is_exhaustive() or dst_layout.span_lo != 0
                or dst_layout.shape.numel() != self.impl.storage.capacity()):
                return TensorStatus.UNINITIALIZED
        # Two independent HostStorage owners cannot share the same allocation.
        # A direct path avoids a whole-tensor staging allocation in the normal
        # host-to-host case, including identical-shape strided views.
        var matching_dense_c = dst_layout.is_c_contiguous() and src_layout.is_c_contiguous()
        var matching_dense_f = dst_layout.is_f_contiguous() and src_layout.is_f_contiguous()
        # Two entirely reversed dense layouts have the same physical ordering
        # of logical elements even though neither is forward C/F-contiguous.
        # Distinct HostStorage owners prove non-aliasing without relying on
        # forgeable public StorageProvenance identity metadata.
        var matching_reverse_c = (
            is_reverse_c_contiguous(dst_layout) and is_reverse_c_contiguous(src_layout)
        )
        var matching_reverse_f = (
            is_reverse_f_contiguous(dst_layout) and is_reverse_f_contiguous(src_layout)
        )
        if dst_layout.is_exhaustive() and src_layout.is_exhaustive():
            if matching_dense_c or matching_dense_f or matching_reverse_c or matching_reverse_f:
                # The two disjoint owners expose corresponding physical runs.
                # Subviews can start at distinct nonzero storage offsets.
                var dst_span = self.impl.storage.borrow_mut_span()
                var src_span = source.impl.storage.borrow_span()
                var dst_first = dst_layout.span_lo
                var src_first = src_layout.span_lo
                var count = dst_layout.shape.numel()
                if count > 0:
                    var destination = dst_span[dst_first:dst_first + count]
                    var source_slice = src_span[src_first:src_first + count]
                    # Two independent owning HostStorage allocations cannot
                    # alias, so the non-overlap contract is established.
                    unsafe_memcpy(
                        dest=destination.unsafe_ptr(),
                        src=source_slice.unsafe_ptr(),
                        count=count,
                    )
                return self._finish_host_copy_from(dst_layout)
            # Exactly one of the two dense layouts runs backwards. In this
            # case the contiguous physical runs correspond in reverse order.
            # Both allocations are independently owned and layout-validated;
            # neither ND coordinate carry nor a temporary staging buffer is
            # required to preserve logical copy semantics.
            var opposite_dense_c = (
                (dst_layout.is_c_contiguous() and is_reverse_c_contiguous(src_layout))
                or (is_reverse_c_contiguous(dst_layout) and src_layout.is_c_contiguous())
            )
            var opposite_dense_f = (
                (dst_layout.is_f_contiguous() and is_reverse_f_contiguous(src_layout))
                or (is_reverse_f_contiguous(dst_layout) and src_layout.is_f_contiguous())
            )
            if opposite_dense_c or opposite_dense_f:
                var count = dst_layout.shape.numel()
                if count > 0:
                    var dst_span = self.impl.storage.borrow_mut_span()
                    var src_span = source.impl.storage.borrow_span()
                    var dst_ptr = dst_span.unsafe_ptr()
                    var src_ptr = src_span.unsafe_ptr()
                    var dst_first = dst_layout.span_lo
                    var src_last = src_layout.span_lo + count - 1
                    for linear in range(count):
                        dst_ptr.unsafe_store(dst_first + linear, src_ptr.unsafe_load(src_last - linear)[0])
                return self._finish_host_copy_from(dst_layout)
        # Unique independent owners permit direct affine traversal without
        # duplicate per-element status allocation or overlap checks.
        var dst_span = self.impl.storage.borrow_mut_span()
        var src_span = source.impl.storage.borrow_span()
        var cursor = NDIndexIterator[Self.rank](dst_layout.shape)
        while not cursor.done():
            var source_at = src_layout.physical_offset_unchecked(cursor.coordinates)
            var destination_at = dst_layout.physical_offset_unchecked(cursor.coordinates)
            dst_span[destination_at] = src_span[source_at]
            cursor.advance()
        return self._finish_host_copy_from(dst_layout)

    # Tensor remains the primary external API. These thin delegates keep all
    # stride traversal, alias checks, and C-order copy rules in access/span.
    # No output materialization or implicit host/device movement occurs.
    def copy_to_span[dst_origin: Origin[mut=True]](
        ref self, mut destination: Span[True, Float32, dst_origin],
    ) -> TensorStatus where identical(Self.S, HostStorage):
        var borrowed = self.view()
        if not borrowed.is_ok():
            return borrowed.status
        return access_copy_to_span(borrowed.payload.value(), destination)

    def copy_from_span[src_origin: Origin](
        mut self, source: Span[False, Float32, src_origin],
    ) -> TensorStatus where identical(Self.S, HostStorage):
        var borrowed = self.mut_view()
        if borrowed.is_ok():
            # Normal initialized destination: retain the proven alias-safe
            # logical-copy adapter, including memmove/snapshot semantics.
            # A non-OK return is NOT an atomicity certificate: a future
            # storage/adapter lowering might fail after its first store.
            var target = borrowed.payload.take()
            # A mismatched source length is a pre-write rejection. Preserve
            # the previously readable owner on this benign caller error;
            # only a failure AFTER preflight can poison the destination.
            if len(source) != target.layout.shape.numel():
                return TensorStatus.SIZE_MISMATCH
            var copied = access_copy_from_span(target, source)
            if copied != TensorStatus.OK:
                self.impl.storage._invalidate_after_failed_write()
            return copied
        if borrowed.status != TensorStatus.UNINITIALIZED:
            return borrowed.status
        # Recovery from a failed Hybrid/CPU write requires a COMPLETE owner
        # overwrite with an independent initialized source Span.  A partial
        # or overlapping copy is not sufficient to publish the old owner.
        var proof = self.impl.layout.validated_snapshot(
            self.impl.storage.capacity()
        )
        if not proof.is_ok():
            return proof.status
        var mapping = proof.payload.value()
        if not mapping.is_proven_unique():
            return TensorStatus.OVERLAPPING
        if (not mapping.is_exhaustive() or mapping.span_lo != 0
            or mapping.shape.numel() != self.impl.storage.capacity()):
            return TensorStatus.UNINITIALIZED
        if len(source) != mapping.shape.numel():
            return TensorStatus.SIZE_MISMATCH
        var destination = self.impl.storage.borrow_mut_span()
        if not host_output_disjoint_from_input(destination, source):
            # An aliased source could read poisoned bytes from this owner.
            # Do not use memmove to fabricate a readability certificate.
            return TensorStatus.ALIAS_UNKNOWN
        var writable = TensorMutView[Self.rank, origin_of(self.impl.storage)](
            destination, mapping
        )
        var copied = access_copy_from_span(writable, source)
        if copied != TensorStatus.OK:
            self.impl.storage._invalidate_after_failed_write()
            return copied
        var published = self.impl.storage._mark_complete_overwrite()
        if published != TensorStatus.OK:
            self.impl.storage._invalidate_after_failed_write()
        return published

    def __getitem__(self, indices: StaticRankArray[Self.rank]) -> TensorResult[Float32]:
        return self.get(indices)

    def __setitem__(mut self, indices: StaticRankArray[Self.rank], value: Float32) -> TensorStatus where conforms_to(Self.S, MutableTensorStorage):
        return self.set(indices, value)

    # Host-only zero-copy borrow. Always validate at the public view boundary.
    # A borrowed View must never inherit stale cached layout/overlap flags.
    def view(ref self) -> TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        return self.impl.view()

    def view_checked(ref self) -> TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        return self.view()

    def tensor_ref(ref self) -> TensorResult[TensorRef[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        # Lightweight, lifetime-bound expert facade. Tensor stays primary.
        var borrowed = self.view()
        if not borrowed.is_ok():
            return TensorResult[TensorRef[Self.rank, origin_of(self.impl.storage)]](borrowed.status)
        return TensorResult[TensorRef[Self.rank, origin_of(self.impl.storage)]](
            TensorRef[Self.rank, origin_of(self.impl.storage)](borrowed.payload.take())
        )

    def tensor_mut_ref(mut self) -> TensorResult[TensorMutRef[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var created = self.mut_view()
        if not created.is_ok():
            return TensorResult[TensorMutRef[Self.rank, origin_of(self.impl.storage)]](created.status)
        return TensorResult[TensorMutRef[Self.rank, origin_of(self.impl.storage)]](
            TensorMutRef[Self.rank, origin_of(self.impl.storage)](created.payload.take())
        )

    def mut_view(mut self) -> TensorResult[TensorMutView[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        return self.impl.mut_view()

    def choose_hybrid_unary_plan[operation: Int](
        ref self, gpu_chunk_capacity: Int,
        profile: HybridTuningProfile, gpu_enabled: Bool,
    ) -> TensorResult[ExecutionPlan] where conforms_to(Self.S, HostSpanStorage):
        # All rank-0..8 shapes lower to one continuous FP32 run only when the
        # owner's current C-contiguous layout has been checked. Reuse the
        # immutable offline-profile placement policy, never time a live call.
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[ExecutionPlan](proof.status)
        var mapping = proof.payload.value()
        if not mapping.is_c_contiguous() or not mapping.is_exhaustive():
            return TensorResult[ExecutionPlan](TensorStatus.NOT_REPRESENTABLE)
        return choose_host_unary_plan_for_capacity(
            mapping.shape.numel(), gpu_chunk_capacity, operation,
            profile, gpu_enabled,
        )

    def unsafe_hybrid_unary_into[
        operation: Int, static_gpu_capacity: Int,
    ](
        ref self, ctx: DeviceContext,
        native_unary: DeviceExternalFunction,
        identity: UnaryAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank, HostStorage],
    ) -> TensorStatus where conforms_to(Self.S, HostSpanStorage):
        # Tensor remains the sole public owning type. Native AOT identity
        # metadata must be bound to a trusted symbol by the caller: this
        # method is explicitly unsafe until a native loader is available.
        var src_layout = self._checked_current_layout()
        if not src_layout.is_ok():
            return src_layout.status
        var dst_layout = result._checked_current_layout()
        if not dst_layout.is_ok():
            return dst_layout.status
        if not src_layout.payload.value().shape.same_extents(
            dst_layout.payload.value().shape
        ):
            return TensorStatus.SHAPE_MISMATCH
        var src = self.as_c_span()
        if not src.is_ok():
            return src.status
        var dst = result.as_mut_c_span()
        if not dst.is_ok():
            return dst.status
        var outcome = unsafe_execute_host_unary_fp32[
            operation, static_gpu_capacity,
        ](
            ctx, native_unary, identity, target_arch, plan,
            src.payload.take(), dst.payload.take(),
        )
        if outcome != TensorStatus.OK:
            # Earlier CPU/GPU windows can have committed before a late
            # rejection, even when its final status looks structural.
            # Multistage failures always invalidate the complete host owner.
            result.impl.storage._invalidate_after_failed_write()
        return outcome

    def unsafe_hybrid_binary_into[
        operation: Int, static_gpu_capacity: Int,
    ](
        ref self, right: Tensor[Self.rank, HostStorage],
        ctx: DeviceContext, native_binary: DeviceExternalFunction,
        identity: BinaryAOTArtifactIdentity, target_arch: Int,
        plan: ExecutionPlan, mut result: Tensor[Self.rank, HostStorage],
    ) -> TensorStatus where conforms_to(Self.S, HostSpanStorage):
        # One owner-aware Tensor facade for the existing bounded CPU/GPU
        # binary engine (including its two-slot pipeline). Source, right and
        # output have exactly the same logical shape. N-D broadcasting and
        # strided device kernels require separately proven ABIs; never
        # reinterpret a broadcast/strided mapping as one flat FP32 run.
        # The precompiled symbol is still foreign code: this method remains
        # unsafe until a trusted AOT loader can authenticate its ABI/target.
        comptime assert operation >= 0 and operation <= 5, "FP32 binary opcode"
        comptime assert static_gpu_capacity > 0, "positive AOT capacity"
        var a_proof = self._checked_current_layout()
        if not a_proof.is_ok():
            return a_proof.status
        var b_proof = right._checked_current_layout()
        if not b_proof.is_ok():
            return b_proof.status
        var c_proof = result._checked_current_layout()
        if not c_proof.is_ok():
            return c_proof.status
        var a_layout = a_proof.payload.value()
        var b_layout = b_proof.payload.value()
        var c_layout = c_proof.payload.value()
        if (not a_layout.shape.same_extents(b_layout.shape)
            or not a_layout.shape.same_extents(c_layout.shape)):
            return TensorStatus.SHAPE_MISMATCH
        if not c_layout.is_proven_unique():
            return TensorStatus.OVERLAPPING
        if (not a_layout.is_c_contiguous() or not a_layout.is_exhaustive()
            or not b_layout.is_c_contiguous() or not b_layout.is_exhaustive()
            or not c_layout.is_c_contiguous() or not c_layout.is_exhaustive()):
            return TensorStatus.NOT_REPRESENTABLE
        var a_span = self.as_c_span()
        if not a_span.is_ok():
            return a_span.status
        var b_span = right.as_c_span()
        if not b_span.is_ok():
            return b_span.status
        var c_span = result.as_mut_c_span()
        if not c_span.is_ok():
            return c_span.status
        # The same canonical ExecutionPlan and binary AOT identity are used
        # by all CPU_ONLY/GPU_ONLY/HYBRID_SPLIT/HYBRID_CPU_ASSIST/PIPELINE
        # paths. The borrowed owner and its native Context remain in scope
        # until the adapter returns; no independent Tensor implementation.
        var outcome = execute_host_binary_fp32[
            static_gpu_capacity, operation,
        ](
            ctx, native_binary, identity, target_arch, plan,
            a_span.payload.take(), b_span.payload.take(),
            c_span.payload.take(),
        )
        if outcome != TensorStatus.OK:
            # A late *structural* error can follow successful earlier GPU
            # windows or CPU partitions. The device-only single-enqueue
            # classifier cannot distinguish this history, so ALWAYS revoke
            # the whole owner on a failed multi-stage lowering.
            result.impl.storage._invalidate_after_failed_write()
        return outcome

    def choose_hybrid_reduce_trailing_axes_plan[
        axes: Int, operation: Int,
    ](
        ref self, gpu_row_capacity: Int,
        profile: HybridRowReductionTuningProfile, gpu_enabled: Bool,
    ) -> TensorResult[ExecutionPlan] where Self.rank >= 1 and axes >= 1 and axes <= Self.rank and conforms_to(Self.S, HostSpanStorage):
        # All rank-1..8 contiguous trailing-axis reductions use one row
        # primitive and one unified ExecutionPlan. The only specialization
        # parameters are static rank/axis count and the selected FP32 opcode.
        # No copying, flattening allocation, or implicit device migration.
        var verified = self._checked_current_layout()
        if not verified.is_ok():
            return TensorResult[ExecutionPlan](verified.status)
        var mapping = verified.payload.value()
        if not mapping.is_c_contiguous() or not mapping.is_exhaustive():
            return TensorResult[ExecutionPlan](TensorStatus.NOT_REPRESENTABLE)
        var geometry = checked_trailing_reduction_geometry[Self.rank, axes](
            mapping.shape
        )
        if not geometry.is_ok():
            return TensorResult[ExecutionPlan](geometry.status)
        var counts = geometry.payload.value()
        return choose_host_reduce_rows_split_plan_for_capacity(
            counts.rows, counts.columns, gpu_row_capacity,
            operation, profile, gpu_enabled,
        )

    def choose_hybrid_reduce_rows_plan[operation: Int](
        ref self, gpu_row_capacity: Int,
        profile: HybridRowReductionTuningProfile, gpu_enabled: Bool,
    ) -> TensorResult[ExecutionPlan] where Self.rank == 2 and conforms_to(Self.S, HostSpanStorage):
        # Offline calibrated cost selection only. Does not enqueue a kernel.
        # A CPU_ONLY result must use the ordinary CPU Tensor reduction path;
        # the expert Hybrid executor accepts split plans only.
        var verified = self._checked_current_layout()
        if not verified.is_ok():
            return TensorResult[ExecutionPlan](verified.status)
        var mapping = verified.payload.value()
        if not mapping.is_c_contiguous() or not mapping.is_exhaustive():
            return TensorResult[ExecutionPlan](TensorStatus.NOT_REPRESENTABLE)
        return choose_host_reduce_rows_split_plan_for_capacity(
            mapping.shape.dims.get(0), mapping.shape.dims.get(1),
            gpu_row_capacity, operation, profile, gpu_enabled,
        )

    def choose_hybrid_gemm_rows_plan(
        ref self, right: Tensor[2, HostStorage],
        gpu_row_capacity: Int, profile: HybridGemmTuningProfile,
        gpu_enabled: Bool,
    ) -> TensorResult[ExecutionPlan] where Self.rank == 2 and conforms_to(Self.S, HostSpanStorage):
        # The immutable plan is derived from owner-backed shapes and an
        # offline profile, not mutable shape claims or runtime benchmarking.
        var a_checked = self._checked_current_layout()
        if not a_checked.is_ok():
            return TensorResult[ExecutionPlan](a_checked.status)
        var b_checked = right._checked_current_layout()
        if not b_checked.is_ok():
            return TensorResult[ExecutionPlan](b_checked.status)
        var a_layout = a_checked.payload.value()
        var b_layout = b_checked.payload.value()
        if (not a_layout.is_c_contiguous() or not a_layout.is_exhaustive()
            or not b_layout.is_c_contiguous() or not b_layout.is_exhaustive()):
            return TensorResult[ExecutionPlan](TensorStatus.NOT_REPRESENTABLE)
        var m = a_layout.shape.dims.get(0)
        var k = a_layout.shape.dims.get(1)
        var n = b_layout.shape.dims.get(1)
        if b_layout.shape.dims.get(0) != k:
            return TensorResult[ExecutionPlan](TensorStatus.SHAPE_MISMATCH)
        return choose_host_gemm_rows_split_plan_for_capacity(
            m, k, n, gpu_row_capacity, profile, gpu_enabled,
        )

    def unsafe_hybrid_reduce_trailing_axes_into[
        axes: Int, operation: Int, static_gpu_row_capacity: Int,
    ](
        ref self,
        ctx: DeviceContext,
        artifact: DeviceExternalFunction,
        identity: ReduceRowsAOTArtifactIdentity,
        target_arch: Int,
        plan: ExecutionPlan,
        mut result: Tensor[Self.rank - axes, HostStorage],
    ) -> TensorStatus where Self.rank >= 1 and axes >= 1 and axes <= Self.rank and conforms_to(Self.S, HostSpanStorage):
        # Explicitly unsafe AOT-native binding: the caller must verify the
        # GPU symbol/target/ABI. All ownership, shape, layout, alias and
        # output-initialization rules are nevertheless checked here.
        var checked_input = self._checked_current_layout()
        if not checked_input.is_ok():
            return checked_input.status
        var checked_output = result._checked_current_layout()
        if not checked_output.is_ok():
            return checked_output.status
        var input_mapping = checked_input.payload.value()
        var output_mapping = checked_output.payload.value()
        if not input_mapping.is_c_contiguous() or not input_mapping.is_exhaustive():
            return TensorStatus.NOT_REPRESENTABLE
        if not output_mapping.is_c_contiguous() or not output_mapping.is_exhaustive():
            return TensorStatus.NOT_REPRESENTABLE
        if not output_mapping.is_proven_unique():
            return TensorStatus.OVERLAPPING
        var geometry = checked_trailing_reduction_geometry[Self.rank, axes](
            input_mapping.shape
        )
        if not geometry.is_ok():
            return geometry.status
        var counts = geometry.payload.value()
        if output_mapping.shape.numel() != counts.rows:
            return TensorStatus.SHAPE_MISMATCH
        # The output shape preserves exactly the unreduced prefix axes.
        # Matching only numel would accept an incompatible multidimensional
        # output even though this ABI defines a rank-changing reduction.
        comptime for i in range(Self.rank - axes):
            if (output_mapping.shape.dims.get(i)
                != input_mapping.shape.dims.get(i)):
                return TensorStatus.SHAPE_MISMATCH
        if counts.rows <= 1 or counts.columns <= 0:
            return TensorStatus.UNSUPPORTED
        var src = self.as_c_span()
        if not src.is_ok():
            return src.status
        var dst = result.as_mut_c_span()
        if not dst.is_ok():
            return dst.status
        # The checked source owner and result owner remain in scope through
        # the CPU/GPU event join; GPU may only own disjoint output rows.
        var outcome = unsafe_execute_hybrid_reduce_rows_fp32[
            operation, static_gpu_row_capacity,
        ](
            ctx, artifact, identity, target_arch, plan, counts.columns,
            src.payload.take(), dst.payload.take(),
        )
        if outcome != TensorStatus.OK:
            # The native adapter may have written CPU rows before a late GPU
            # failure. Never return a readable partially initialized Tensor.
            result.impl.storage._invalidate_after_failed_write()
        return outcome

    def unsafe_hybrid_reduce_rows_into[
        operation: Int, static_gpu_row_capacity: Int,
    ](
        ref self,
        ctx: DeviceContext,
        artifact: DeviceExternalFunction,
        identity: ReduceRowsAOTArtifactIdentity,
        target_arch: Int,
        plan: ExecutionPlan,
        mut result: Tensor[1, HostStorage],
    ) -> TensorStatus where Self.rank == 2 and conforms_to(Self.S, HostSpanStorage):
        # One owning Tensor input and one owning Tensor output. A checked
        # ExecutionPlan partitions OUTPUT ROWS, not individual FP32 inputs.
        # Native AOT symbol provenance is not authenticated here: the expert
        # entry point remains explicitly unsafe until a trusted loader exists.
        var checked_input = self._checked_current_layout()
        if not checked_input.is_ok():
            return checked_input.status
        var checked_output = result._checked_current_layout()
        if not checked_output.is_ok():
            return checked_output.status
        var source_layout = checked_input.payload.value()
        var result_layout = checked_output.payload.value()
        if not source_layout.is_c_contiguous() or not source_layout.is_exhaustive():
            return TensorStatus.NOT_REPRESENTABLE
        if not result_layout.is_c_contiguous() or not result_layout.is_exhaustive():
            return TensorStatus.NOT_REPRESENTABLE
        if not result_layout.is_proven_unique():
            return TensorStatus.OVERLAPPING
        var rows = source_layout.shape.dims.get(0)
        var columns = source_layout.shape.dims.get(1)
        if result_layout.shape.dims.get(0) != rows:
            return TensorStatus.SHAPE_MISMATCH
        var src_span = self.as_c_span()
        if not src_span.is_ok():
            return src_span.status
        var dst_span = result.as_mut_c_span()
        if not dst_span.is_ok():
            return dst_span.status
        var outcome = unsafe_execute_hybrid_reduce_rows_fp32[
            operation, static_gpu_row_capacity,
        ](
            ctx, artifact, identity, target_arch, plan, columns,
            src_span.payload.take(), dst_span.payload.take(),
        )
        # CPU rows can already have been committed before a GPU completion
        # event fails. Even a late structural rejection may follow partial
        # writes. Only the explicit Tensor-side checks ABOVE preserve an
        # already-readable output after rejecting preflight metadata.
        if outcome != TensorStatus.OK:
            # This is a multi-stage row lowering: a late structural failure
            # need not mean that previously completed CPU/GPU rows are intact.
            result.impl.storage._invalidate_after_failed_write()
        return outcome

    def unsafe_hybrid_gemm_rows_into[
        static_gpu_row_capacity: Int,
    ](
        ref self,
        right: Tensor[2, HostStorage],
        ctx: DeviceContext,
        artifact: DeviceExternalFunction,
        identity: GemmAOTArtifactIdentity,
        target_arch: Int,
        plan: ExecutionPlan,
        mut result: Tensor[2, HostStorage],
    ) -> TensorStatus where Self.rank == 2 and conforms_to(Self.S, HostSpanStorage):
        # Tensor-first expert API. A and B are initialized owner-backed
        # host tensors; C must be a separately initialized host allocation.
        # Caller-supplied AOT symbols are NOT authenticated by this method.
        var left_verified = self._checked_current_layout()
        if not left_verified.is_ok():
            return left_verified.status
        var right_verified = right._checked_current_layout()
        if not right_verified.is_ok():
            return right_verified.status
        var out_verified = result._checked_current_layout()
        if not out_verified.is_ok():
            return out_verified.status
        var layout_a = left_verified.payload.value()
        var layout_b = right_verified.payload.value()
        var layout_c = out_verified.payload.value()
        if not layout_a.is_c_contiguous() or not layout_a.is_exhaustive():
            return TensorStatus.NOT_REPRESENTABLE
        if not layout_b.is_c_contiguous() or not layout_b.is_exhaustive():
            return TensorStatus.NOT_REPRESENTABLE
        if not layout_c.is_c_contiguous() or not layout_c.is_exhaustive():
            return TensorStatus.NOT_REPRESENTABLE
        if not layout_c.is_proven_unique():
            return TensorStatus.OVERLAPPING
        var m = layout_a.shape.dims.get(0)
        var k = layout_a.shape.dims.get(1)
        var n = layout_b.shape.dims.get(1)
        if layout_b.shape.dims.get(0) != k:
            return TensorStatus.SHAPE_MISMATCH
        if layout_c.shape.dims.get(0) != m or layout_c.shape.dims.get(1) != n:
            return TensorStatus.SHAPE_MISMATCH
        var a_span = self.as_c_span()
        if not a_span.is_ok():
            return a_span.status
        var b_span = right.as_c_span()
        if not b_span.is_ok():
            return b_span.status
        var c_span = result.as_mut_c_span()
        if not c_span.is_ok():
            return c_span.status
        var outcome = unsafe_execute_hybrid_gemm_rows_fp32[
            static_gpu_row_capacity,
        ](
            ctx, artifact, identity, target_arch, plan,
            m, k, n, a_span.payload.take(),
            b_span.payload.take(), c_span.payload.take(),
        )
        # The CPU lane may have written result rows even if the accelerator
        # subsequently reports a failed event/launch. Hide the ENTIRE host
        # payload from checked consumers until a full overwrite is proven.
        if outcome != TensorStatus.OK:
            # This is a multi-stage row lowering: a late structural failure
            # need not mean that previously completed CPU/GPU rows are intact.
            result.impl.storage._invalidate_after_failed_write()
        return outcome

    def unsafe_recover_device_default_stream(mut self) -> TensorStatus where identical(Self.S, DeviceStorage):
        # The same Tensor remains the public owner. Only a caller who can
        # prove EXCLUSIVE use of the poisoned allocation on its retained
        # default stream may recover it. No GPU payload is reinitialized.
        # Cross-stream/foreign producer recovery is deliberately unsupported.
        return self.impl.storage.unsafe_recover_quarantined_default_stream()

    def unsafe_recover_and_reinitialize_entire_device(
        mut self, value: Float32,
    ) -> TensorStatus where identical(Self.S, DeviceStorage):
        # Explicit owner-preserving recovery from a FAILED native GPU writer.
        # UNSAFE caller proof: every outstanding use belongs to the retained
        # default stream, and no foreign queue/caller may touch this allocation.
        # A stream fence is insufficient for unknown foreign GPU producers.
        # The existing owner stays unreadable until a complete FP32 overwrite
        # is enqueued, joined and committed by the canonical write transaction.
        var recovered = self.impl.storage.unsafe_recover_quarantined_default_stream()
        if recovered != TensorStatus.OK:
            return recovered
        if not self.impl.storage.lifetime_context:
            return TensorStatus.GPU_QUIESCENCE_UNPROVEN
        var retained = self.impl.storage.lifetime_context.value().copy()
        # fill_device validates full-span C layout, count, exact issuing
        # context, allocation capacity, event completion and publish ticket.
        # Never set initialized/initialized_elements at this boundary.
        return self.fill_device(retained, value)

    def reinitialize_entire_host_storage(
        mut self, value: Float32,
    ) -> TensorStatus where conforms_to(Self.S, HostSpanStorage):
        # Recovery from a poisoned host owner requires a COMPLETE storage
        # overwrite, not a view/partial-region fill. Explicit name avoids
        # surprising materialization and preserves a single Tensor owner.
        # Validate the mapping WITHOUT demanding old data be initialized.
        var capacity = self.impl.storage.capacity()
        var structural = self.impl.layout.validated_snapshot(capacity)
        if not structural.is_ok():
            return structural.status
        # This fills the ENTIRE physical owner, including values outside a
        # narrow Tensor layout. Caller deliberately requests that behavior.
        return self.impl.storage.fill_host(value)

    def execute_static_cpu_dataflow[graph_nodes: Int, graph_slots: Int](
        mut self,
        plan: SealedStaticDataflowPlan[graph_nodes, graph_slots],
        offsets: Array[Int, graph_slots],
    ) -> TensorStatus where conforms_to(Self.S, HostSpanStorage):
        # Tensor is the sole public owning payload. This source-level AOT
        # CPU lowering operates on disjoint FP32 slots inside this Tensor's
        # proven C-contiguous host arena. Its execution neither constructs
        # another tensor family nor inserts any CPU/GPU transfer or JIT.
        var verified = self.as_mut_c_span()
        if not verified.is_ok():
            return verified.status
        var arena = verified.payload.take()
        var outcome = _run_cpu_static_dataflow_in_arena(plan, arena, offsets)
        # Checked preparation may reject an invalid slot, target or AOT
        # variant without touching the owner. Preserve its prior readable
        # state on such failures; invalidate only after a kernel could have
        # begun mutating the arena. A late stage failure poisons the full
        # allocation because this execution is not a transactional DAG.
        if outcome.status != TensorStatus.OK and outcome.writes_may_have_started:
            self.impl.storage._invalidate_after_failed_write()
        return outcome.status

    def execute_bound_static_cpu_dataflow[
        graph_nodes: Int, graph_slots: Int, aot_variants: Int,
    ](
        mut self,
        bound: BoundAOTDataflowPlan[graph_nodes, graph_slots, aot_variants],
        offsets: Array[Int, graph_slots],
    ) -> TensorStatus where conforms_to(Self.S, HostSpanStorage):
        # Public Tensor remains the single owning payload. Unlike the
        # lower-level execute_static_cpu_dataflow expert entry point, this
        # binds the CPU lowering to the checked immutable AOT strategy and
        # placement identities BEFORE touching any source/output element.
        # It cannot serve as a GPU/Hybrid lowering or a JIT fallback.
        var approved = bound.checked_cpu_graph_snapshot()
        if not approved.is_ok():
            return approved.status
        return self.execute_static_cpu_dataflow(approved.payload.take(), offsets)

    def transpose_copy2(
        ref self
    ) -> TensorResult[Tensor[2, HostStorage]] where Self.rank == 2:
        # Explicit materialization on the canonical Tensor type. A metadata
        # transpose() remains zero-copy; this method always owns a new buffer.
        if not self.impl.storage.is_host_accessible():
            return TensorResult[Tensor[2, HostStorage]](TensorStatus.BACKEND_UNAVAILABLE)
        var verified = self._checked_current_layout()
        if not verified.is_ok():
            return TensorResult[Tensor[2, HostStorage]](verified.status)
        var mapping = verified.payload.value()
        var rows = mapping.shape.dims.get(0)
        var columns = mapping.shape.dims.get(1)
        var dims = StaticRankArray[2](0)
        dims.set(0, columns)
        dims.set(1, rows)
        var transposed_shape = TensorShape[2].checked(dims)
        if not transposed_shape.is_ok():
            return TensorResult[Tensor[2, HostStorage]](transposed_shape.status)
        var acquired = Tensor[2, HostStorage]._empty_for_complete_overwrite_checked(
            transposed_shape.payload.value()
        )
        if not acquired.is_ok():
            return TensorResult[Tensor[2, HostStorage]](acquired.status)
        var output = acquired.payload.take()
        # A zero-element allocation carries a valid empty initialization
        # proof. The uniform owner gate still checks its actual layout.
        if transposed_shape.payload.value().numel() == 0:
            return output._finish_complete_overwrite()
        var dst = output.impl.storage.borrow_mut_span()
        var step0 = mapping.strides.get(0)
        var step1 = mapping.strides.get(1)
        comptime if identical(Self.S, HostStorage):
            var src = self.impl.storage.borrow_span()
            var transpose_status = cpu_transpose_copy2_strided(
                src, dst, rows, columns, mapping.offset, step0, step1,
            )
            if transpose_status != TensorStatus.OK:
                return TensorResult[Tensor[2, HostStorage]](transpose_status)
        else:
            # Other host-readable storage traits expose only fallible indexed
            # loads. Retain their exact checked semantics; no raw borrow is
            # invented for an unfamiliar storage/address space.
            for j in range(columns):
                var destination_row = j * rows
                var source_column = mapping.offset + j * step1
                for i in range(rows):
                    var element = self.impl.storage.read_host(
                        source_column + i * step0
                    )
                    if not element.is_ok():
                        return TensorResult[Tensor[2, HostStorage]](element.status)
                    dst[destination_row + i] = element.payload.value()
        return output._finish_complete_overwrite()

    def transpose(ref self, a: Int, b: Int) -> TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](checked_source.status)
        var changed = checked_source.payload.value().transpose(a, b, self.impl.storage.capacity())
        if not changed.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](changed.status)
        # The mapping constructor has already validated its shape, physical
        # span and allocation bounds. Rebuilding the same proof via
        # from_span_checked() would add a redundant O(rank) pass.
        return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](
            TensorView[Self.rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), changed.payload.take()
            )
        )

    def permute(ref self, axes: StaticRankArray[Self.rank]) -> TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](checked_source.status)
        var changed = checked_source.payload.value().permute(axes, self.impl.storage.capacity())
        if not changed.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](changed.status)
        # The mapping constructor has already validated its shape, physical
        # span and allocation bounds. Rebuilding the same proof via
        # from_span_checked() would add a redundant O(rank) pass.
        return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](
            TensorView[Self.rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), changed.payload.take()
            )
        )

    def flip(ref self, axis: Int) -> TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](checked_source.status)
        var changed = checked_source.payload.value().flip(axis, self.impl.storage.capacity())
        if not changed.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](changed.status)
        # The mapping constructor has already validated its shape, physical
        # span and allocation bounds. Rebuilding the same proof via
        # from_span_checked() would add a redundant O(rank) pass.
        return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](
            TensorView[Self.rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), changed.payload.take()
            )
        )

    def expand_view[new_rank: Int](ref self, target: TensorShape[new_rank]) -> TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage) and new_rank >= Self.rank and new_rank <= 8:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]](checked_source.status)
        var changed = broadcast_layout[Self.rank, new_rank](checked_source.payload.value(), target, self.impl.storage.capacity())
        if not changed.is_ok():
            return TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]](changed.status)
        # The mapping constructor has already validated its shape, physical
        # span and allocation bounds. Rebuilding the same proof via
        # from_span_checked() would add a redundant O(rank) pass.
        return TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]](
            TensorView[new_rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), changed.payload.take()
            )
        )

    def ravel_view(ref self) -> TensorResult[TensorView[1, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var dims = StaticRankArray[1](self.numel())
        var target = TensorShape[1].checked(dims)
        if not target.is_ok():
            return TensorResult[TensorView[1, origin_of(self.impl.storage)]](target.status)
        return self.reshape_view[1](target.payload.value())

    def __getitem__(self, i0: Int) -> TensorResult[Float32] where Self.rank == 1:
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        return self.get(coord)

    def __setitem__(mut self, i0: Int, value: Float32) -> TensorStatus where Self.rank == 1 and conforms_to(Self.S, MutableTensorStorage):
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        return self.set(coord, value)

    def __getitem__(self, i0: Int, i1: Int) -> TensorResult[Float32] where Self.rank == 2:
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        return self.get(coord)

    def __setitem__(mut self, i0: Int, i1: Int, value: Float32) -> TensorStatus where Self.rank == 2 and conforms_to(Self.S, MutableTensorStorage):
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        return self.set(coord, value)

    def __getitem__(self, i0: Int, i1: Int, i2: Int) -> TensorResult[Float32] where Self.rank == 3:
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        coord.set(2, i2)
        return self.get(coord)

    def __setitem__(mut self, i0: Int, i1: Int, i2: Int, value: Float32) -> TensorStatus where Self.rank == 3 and conforms_to(Self.S, MutableTensorStorage):
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        coord.set(2, i2)
        return self.set(coord, value)

    def __getitem__(self, i0: Int, i1: Int, i2: Int, i3: Int) -> TensorResult[Float32] where Self.rank == 4:
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        coord.set(2, i2)
        coord.set(3, i3)
        return self.get(coord)

    def __setitem__(mut self, i0: Int, i1: Int, i2: Int, i3: Int, value: Float32) -> TensorStatus where Self.rank == 4 and conforms_to(Self.S, MutableTensorStorage):
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        coord.set(2, i2)
        coord.set(3, i3)
        return self.set(coord, value)

    def __getitem__(self, i0: Int, i1: Int, i2: Int, i3: Int, i4: Int) -> TensorResult[Float32] where Self.rank == 5:
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        coord.set(2, i2)
        coord.set(3, i3)
        coord.set(4, i4)
        return self.get(coord)

    def __setitem__(mut self, i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, value: Float32) -> TensorStatus where Self.rank == 5 and conforms_to(Self.S, MutableTensorStorage):
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        coord.set(2, i2)
        coord.set(3, i3)
        coord.set(4, i4)
        return self.set(coord, value)

    def __getitem__(self, i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int) -> TensorResult[Float32] where Self.rank == 6:
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        coord.set(2, i2)
        coord.set(3, i3)
        coord.set(4, i4)
        coord.set(5, i5)
        return self.get(coord)

    def __setitem__(mut self, i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, value: Float32) -> TensorStatus where Self.rank == 6 and conforms_to(Self.S, MutableTensorStorage):
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        coord.set(2, i2)
        coord.set(3, i3)
        coord.set(4, i4)
        coord.set(5, i5)
        return self.set(coord, value)

    def __getitem__(self, i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int) -> TensorResult[Float32] where Self.rank == 7:
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        coord.set(2, i2)
        coord.set(3, i3)
        coord.set(4, i4)
        coord.set(5, i5)
        coord.set(6, i6)
        return self.get(coord)

    def __setitem__(mut self, i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int, value: Float32) -> TensorStatus where Self.rank == 7 and conforms_to(Self.S, MutableTensorStorage):
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        coord.set(2, i2)
        coord.set(3, i3)
        coord.set(4, i4)
        coord.set(5, i5)
        coord.set(6, i6)
        return self.set(coord, value)

    def __getitem__(self, i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int, i7: Int) -> TensorResult[Float32] where Self.rank == 8:
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        coord.set(2, i2)
        coord.set(3, i3)
        coord.set(4, i4)
        coord.set(5, i5)
        coord.set(6, i6)
        coord.set(7, i7)
        return self.get(coord)

    def __setitem__(mut self, i0: Int, i1: Int, i2: Int, i3: Int, i4: Int, i5: Int, i6: Int, i7: Int, value: Float32) -> TensorStatus where Self.rank == 8 and conforms_to(Self.S, MutableTensorStorage):
        var coord = StaticRankArray[Self.rank](0)
        coord.set(0, i0)
        coord.set(1, i1)
        coord.set(2, i2)
        coord.set(3, i3)
        coord.set(4, i4)
        coord.set(5, i5)
        coord.set(6, i6)
        coord.set(7, i7)
        return self.set(coord, value)

    # Explicit materialization into NEW C-contiguous FP32 ownership. This
    # operation never returns an alias of the source. It works for ranks 0..8,
    # including empty, F-contiguous and arbitrary checked negative strides.
    def reshape_copy[new_rank: Int](
        ref self, target: TensorShape[new_rank]
    ) -> TensorResult[Tensor[new_rank, HostStorage]] where identical(Self.S, HostStorage):
        comptime assert new_rank >= 0 and new_rank <= 8, "result rank must be 0..8"
        var source_check = self._checked_current_layout()
        if not source_check.is_ok():
            return TensorResult[Tensor[new_rank, HostStorage]](source_check.status)
        var destination_check = TensorShape[new_rank].checked(target.dims)
        if not destination_check.is_ok():
            return TensorResult[Tensor[new_rank, HostStorage]](destination_check.status)
        var source_layout = source_check.payload.value()
        var final_shape = destination_check.payload.value()
        if target.numel() != final_shape.numel():
            return TensorResult[Tensor[new_rank, HostStorage]](TensorStatus.INVALID_SHAPE)
        if source_layout.shape.numel() != final_shape.numel():
            return TensorResult[Tensor[new_rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var allocated = Tensor[new_rank, HostStorage]._empty_for_complete_overwrite_checked(
            final_shape
        )
        if not allocated.is_ok():
            return TensorResult[Tensor[new_rank, HostStorage]](allocated.status)
        var output = allocated.payload.take()
        var count = final_shape.numel()
        if count == 0:
            return output._finish_complete_overwrite()
        var input_data = self.impl.storage.borrow_span()
        var output_data = output.impl.storage.borrow_mut_span()
        if source_layout.is_c_contiguous() and source_layout.is_exhaustive():
            var start = source_layout.span_lo
            var logical = input_data[start:start + count]
            # The fresh output allocation proves non-aliasing.
            unsafe_memcpy(
                dest=output_data.unsafe_ptr(), src=logical.unsafe_ptr(), count=count
            )
            return output._finish_complete_overwrite()
        # Flatten source values in their logical C index order, regardless
        # of physical source strides, then reinterpret in the target shape.
        var cursor = NDIndexIterator[Self.rank](source_layout.shape)
        var position = 0
        while not cursor.done():
            output_data[position] = input_data[
                source_layout.physical_offset_unchecked(cursor.coordinates)
            ]
            position += 1
            cursor.advance()
        return output._finish_complete_overwrite()

    def flatten_scalar_copy(
        ref self
    ) -> TensorResult[Tensor[1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank == 0:
        # A scalar has one logical FP32 value; rank-changing copy is explicit.
        var one = TensorShape[1].checked(StaticRankArray[1](1))
        if not one.is_ok():
            return TensorResult[Tensor[1, HostStorage]](one.status)
        return self.reshape_copy[1](one.payload.value())

    def ravel_copy(
        ref self
    ) -> TensorResult[Tensor[1, HostStorage]] where identical(Self.S, HostStorage):
        var dims = StaticRankArray[1](self.numel())
        var target = TensorShape[1].checked(dims)
        if not target.is_ok():
            return TensorResult[Tensor[1, HostStorage]](target.status)
        return self.reshape_copy[1](target.payload.value())

    def flatten_copy[start_dim: Int, end_dim: Int](
        ref self
    ) -> TensorResult[
        Tensor[Self.rank - (end_dim - start_dim), HostStorage]
    ] where identical(Self.S, HostStorage) and Self.rank >= 1:
        # One canonical checked shape-transform invariant for both copy and
        # zero-copy flatten, including empty dimensions and overflow.
        comptime assert start_dim >= 0 and end_dim >= start_dim and end_dim < Self.rank, "invalid flatten axis interval"
        comptime out_rank = Self.rank - (end_dim - start_dim)
        # Flatten infers metadata only; reshape_copy is the one owner-bound
        # proof before it reads a Float32 element. Avoid a duplicate span pass.
        var flattened = flatten_shape[Self.rank, start_dim, end_dim](
            self.impl.layout.shape
        )
        if not flattened.is_ok():
            return TensorResult[Tensor[out_rank, HostStorage]](flattened.status)
        return self.reshape_copy[out_rank](flattened.payload.value())

    def reshape_view[new_rank: Int](ref self, target: TensorShape[new_rank]) -> TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]](checked_source.status)
        var layout = reshape_c_view(checked_source.payload.value(), target, self.impl.storage.capacity())
        if not layout.is_ok():
            return TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]](layout.status)
        # The mapping constructor has already validated its shape, physical
        # span and allocation bounds. Rebuilding the same proof via
        # from_span_checked() would add a redundant O(rank) pass.
        return TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]](
            TensorView[new_rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), layout.payload.take()
            )
        )

    def reshape_f_view[new_rank: Int](ref self, target: TensorShape[new_rank]) -> TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]](checked_source.status)
        var changed = reshape_f_view(checked_source.payload.value(), target, self.impl.storage.capacity())
        if not changed.is_ok():
            return TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]](changed.status)
        # The mapping constructor has already validated its shape, physical
        # span and allocation bounds. Rebuilding the same proof via
        # from_span_checked() would add a redundant O(rank) pass.
        return TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]](
            TensorView[new_rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), changed.payload.take()
            )
        )

    def ravel_f_view(ref self) -> TensorResult[TensorView[1, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var target = TensorShape[1].checked(StaticRankArray[1](self.numel()))
        if not target.is_ok():
            return TensorResult[TensorView[1, origin_of(self.impl.storage)]](target.status)
        return self.reshape_f_view[1](target.payload.value())

    def as_strided_view[new_rank: Int](
        ref self, shape: TensorShape[new_rank], strides: StaticRankArray[new_rank], offset: Int
    ) -> TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        # Absolute element offset into this very allocation, not an untracked pointer.
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]](checked_source.status)
        var mapping = TensorLayout[new_rank].checked(
            shape, strides, offset, self.impl.storage.capacity()
        )
        if not mapping.is_ok():
            return TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]](mapping.status)
        # The mapping constructor has already validated its shape, physical
        # span and allocation bounds. Rebuilding the same proof via
        # from_span_checked() would add a redundant O(rank) pass.
        return TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]](
            TensorView[new_rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), mapping.payload.take()
            )
        )

    def as_strided[new_rank: Int](
        ref self, shape: TensorShape[new_rank], strides: StaticRankArray[new_rank], offset: Int,
    ) -> TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage) and new_rank >= 0 and new_rank <= 8:
        # Explicit zero-copy name; rejects any mapping outside this allocation.
        return self.as_strided_view[new_rank](shape, strides, offset)

    def broadcast_to[new_rank: Int](ref self, target: TensorShape[new_rank]) -> TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage) and new_rank >= Self.rank and new_rank <= 8:
        return self.expand_view[new_rank](target)

    def unsqueeze_view[axis: Int](ref self) -> TensorResult[TensorView[Self.rank + 1, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage) and Self.rank < 8:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[Self.rank + 1, origin_of(self.impl.storage)]](checked_source.status)
        var layout = unsqueeze_layout[Self.rank, axis](checked_source.payload.value(), self.impl.storage.capacity())
        if not layout.is_ok():
            return TensorResult[TensorView[Self.rank + 1, origin_of(self.impl.storage)]](layout.status)
        # The mapping constructor has already validated its shape, physical
        # span and allocation bounds. Rebuilding the same proof via
        # from_span_checked() would add a redundant O(rank) pass.
        return TensorResult[TensorView[Self.rank + 1, origin_of(self.impl.storage)]](
            TensorView[Self.rank + 1, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), layout.payload.take()
            )
        )

    def squeeze_axis_view[axis: Int](ref self) -> TensorResult[TensorView[Self.rank - 1, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage) and Self.rank > 0:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[Self.rank - 1, origin_of(self.impl.storage)]](checked_source.status)
        var layout = squeeze_axis_layout[Self.rank, axis](checked_source.payload.value(), self.impl.storage.capacity())
        if not layout.is_ok():
            return TensorResult[TensorView[Self.rank - 1, origin_of(self.impl.storage)]](layout.status)
        # The mapping constructor has already validated its shape, physical
        # span and allocation bounds. Rebuilding the same proof via
        # from_span_checked() would add a redundant O(rank) pass.
        return TensorResult[TensorView[Self.rank - 1, origin_of(self.impl.storage)]](
            TensorView[Self.rank - 1, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), layout.payload.take()
            )
        )

    def squeeze_axes_view[count: Int](
        ref self, axes: StaticRankArray[count],
    ) -> TensorResult[TensorView[Self.rank - count, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage) and count >= 0 and count <= Self.rank:
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[Self.rank - count, origin_of(self.impl.storage)]](checked_source.status)
        var reduced = squeeze_axes_layout[Self.rank, count](
            checked_source.payload.value(), axes, self.impl.storage.capacity(),
        )
        if not reduced.is_ok():
            return TensorResult[TensorView[Self.rank - count, origin_of(self.impl.storage)]](reduced.status)
        return TensorResult[TensorView[Self.rank - count, origin_of(self.impl.storage)]](
            TensorView[Self.rank - count, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), reduced.payload.take()
            )
        )

    def squeeze_axes[count: Int](
        ref self, axes: StaticRankArray[count],
    ) -> TensorResult[TensorView[Self.rank - count, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage) and count >= 0 and count <= Self.rank:
        return self.squeeze_axes_view[count](axes)

    # Canonical names from the documented Tensor facade. These always
    # create lifetime-bound, metadata-only, zero-copy views, never owners.
    def unsqueeze[axis: Int](ref self) -> TensorResult[TensorView[Self.rank + 1, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage) and Self.rank < 8:
        return self.unsqueeze_view[axis]()

    def squeeze_axis[axis: Int](ref self) -> TensorResult[TensorView[Self.rank - 1, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage) and Self.rank > 0:
        return self.squeeze_axis_view[axis]()

    def expand[new_rank: Int](ref self, target: TensorShape[new_rank]) -> TensorResult[TensorView[new_rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage) and new_rank >= Self.rank and new_rank <= 8:
        return self.expand_view[new_rank](target)

    # Public structural conveniences remain thin wrappers: no allocation,
    # indexing loop, or backend policy is duplicated in the Tensor facade.
    def swapaxes(ref self, first: Int, second: Int) -> TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        return self.transpose(first, second)

    def movedim(ref self, source: Int, destination: Int) -> TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        var checked_source = self._checked_current_layout()
        if not checked_source.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](checked_source.status)
        var transformed = movedim_layout(checked_source.payload.value(), source, destination, self.impl.storage.capacity())
        if not transformed.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](transformed.status)
        # The mapping constructor has already validated its shape, physical
        # span and allocation bounds. Rebuilding the same proof via
        # from_span_checked() would add a redundant O(rank) pass.
        return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](
            TensorView[Self.rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), transformed.payload.take()
            )
        )

    def narrow(ref self, axis: Int, start: Int, length: Int) -> TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage):
        # One source snapshot and one normalized axis, with overflow-safe
        # endpoint checking. No extra Tensor.dim()/slice_axis() boundary.
        var source = self._checked_current_layout()
        if not source.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](source.status)
        var chosen = normalize_axis[Self.rank](axis)
        if not chosen.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](chosen.status)
        var extent = source.payload.value().shape.dims.get(chosen.payload.value())
        if start < 0 or length < 0 or start > extent or length > extent - start:
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](TensorStatus.OUT_OF_BOUNDS)
        var mapped = slice_layout_axis(
            source.payload.value(), chosen.payload.value(),
            SliceSpec(start, start + length, 1), self.impl.storage.capacity(),
        )
        if not mapped.is_ok():
            return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](mapped.status)
        return TensorResult[TensorView[Self.rank, origin_of(self.impl.storage)]](
            TensorView[Self.rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), mapped.payload.take()
            )
        )

    # A fixed number of disjoint, zero-copy borrowed sections. The caller
    # controls section extents, while all output ranks stay static and equal
    # to the input rank. No storage allocation or payload copy is performed.
    # A zero-sized section is represented as an empty view.
    def split_sections[parts: Int](
        ref self, axis: Int, sections: Array[Int, parts],
    ) -> TensorResult[Array[TensorView[Self.rank, origin_of(self.impl.storage)], parts]] where conforms_to(Self.S, HostSpanStorage):
        comptime assert parts > 0, "split requires a nonzero static section count"
        var initial = self.view_checked()
        if not initial.is_ok():
            return TensorResult[Array[TensorView[Self.rank, origin_of(self.impl.storage)], parts]](initial.status)
        var normalized = normalize_axis[Self.rank](axis)
        if not normalized.is_ok():
            return TensorResult[Array[TensorView[Self.rank, origin_of(self.impl.storage)], parts]](normalized.status)
        var selected_axis = normalized.payload.value()
        var base = initial.payload.take()
        var extent = base.layout.shape.dims.get(selected_axis)
        var covered = 0
        # Validate the complete section vector before constructing a view.
        for i in range(parts):
            var width = sections[i]
            if width < 0 or width > extent - covered:
                return TensorResult[Array[TensorView[Self.rank, origin_of(self.impl.storage)], parts]](TensorStatus.SIZE_MISMATCH)
            covered += width
        if covered != extent:
            return TensorResult[Array[TensorView[Self.rank, origin_of(self.impl.storage)], parts]](TensorStatus.SIZE_MISMATCH)
        # Retain the validated base borrow locally while the array receives
        # independent metadata copies. No allocation of Float32 payload data.
        var result = Array[TensorView[Self.rank, origin_of(self.impl.storage)], parts](fill=base.copy())
        var start = 0
        for i in range(parts):
            var piece = base.narrow(selected_axis, start, sections[i])
            if not piece.is_ok():
                return TensorResult[Array[TensorView[Self.rank, origin_of(self.impl.storage)], parts]](piece.status)
            result[i] = piece.payload.take()
            start += sections[i]
        return TensorResult[Array[TensorView[Self.rank, origin_of(self.impl.storage)], parts]](result)

    # Exactly `parts` near-equal zero-copy views, including empty trailing
    # views when the selected extent is smaller than the static part count.
    # This fixed-arity contract deliberately differs from dynamic PyTorch
    # chunk() APIs that may return fewer than the requested number of chunks.
    def chunk_fixed[parts: Int](
        ref self, axis: Int,
    ) -> TensorResult[Array[TensorView[Self.rank, origin_of(self.impl.storage)], parts]] where conforms_to(Self.S, HostSpanStorage):
        comptime assert parts > 0, "chunk requires a nonzero static part count"
        var extent = self.dim(axis)
        if not extent.is_ok():
            return TensorResult[Array[TensorView[Self.rank, origin_of(self.impl.storage)], parts]](extent.status)
        var quotient = extent.payload.value() // parts
        var remainder = extent.payload.value() % parts
        var sections = Array[Int, parts](fill=quotient)
        for i in range(parts):
            if i < remainder:
                sections[i] = quotient + 1
        return self.split_sections[parts](axis, sections)

    # Canonical public names; the implementation remains in the checked
    # owner facade above and returns borrowed, zero-copy, fixed-rank views.
    def split[parts: Int](
        ref self, axis: Int, sections: Array[Int, parts],
    ) -> TensorResult[Array[TensorView[Self.rank, origin_of(self.impl.storage)], parts]] where conforms_to(Self.S, HostSpanStorage):
        return self.split_sections[parts](axis, sections)

    def chunk[parts: Int](
        ref self, axis: Int,
    ) -> TensorResult[Array[TensorView[Self.rank, origin_of(self.impl.storage)], parts]] where conforms_to(Self.S, HostSpanStorage):
        return self.chunk_fixed[parts](axis)

    # Repeat (element-level) and tile (whole-axis repetition) are distinct.
    # Both are explicit allocations; neither silently returns a zero-copy view.
    def _repeat_or_tile_axis_host[tile_mode: Bool](
        ref self, axis: Int, copies: Int
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        if copies < 0:
            return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.INVALID_SHAPE)
        var source_proof = self._checked_current_layout()
        if not source_proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](source_proof.status)
        var normalized = normalize_axis[Self.rank](axis)
        if not normalized.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](normalized.status)
        var a = normalized.payload.value()
        var source_layout = source_proof.payload.value()
        var out_dims = source_layout.shape.dims.copy()
        var enlarged = checked_mul_nonnegative(out_dims.get(a), copies)
        if not enlarged.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](enlarged.status)
        out_dims.set(a, enlarged.payload.value())
        var shaped = TensorShape[Self.rank].checked(out_dims)
        if not shaped.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](shaped.status)
        var allocated = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(shaped.payload.value())
        if not allocated.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](allocated.status)
        var result = allocated.payload.take()
        var src = self.impl.storage.borrow_span()
        var dst = result.impl.storage.borrow_mut_span()
        var source_extent = source_layout.shape.dims.get(a)
        var cursor = NDIndexIterator[Self.rank](shaped.payload.value())
        while not cursor.done():
            var coord = cursor.coordinates.copy()
            comptime if tile_mode:
                # A non-empty output proves source_extent > 0.
                coord.set(a, coord.get(a) % source_extent)
            else:
                # A non-empty output proves copies > 0.
                coord.set(a, coord.get(a) // copies)
            dst[cursor.produced] = src[source_layout.physical_offset_unchecked(coord)]
            cursor.advance()
        return result._finish_complete_overwrite()

    # Gather is an explicitly materializing, rank-preserving indexing primitive.
    # All index values are validated before allocating any output.
    def gather[origin: Origin](
        ref self, axis: Int, indices: Span[False, Int, origin]
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        var source_result = self._checked_current_layout()
        if not source_result.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](source_result.status)
        var axis_result = normalize_axis[Self.rank](axis)
        if not axis_result.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](axis_result.status)
        var selected_axis = axis_result.payload.value()
        var source = source_result.payload.value()
        var extent = source.shape.dims.get(selected_axis)
        for i in range(len(indices)):
            var at = normalize_index(indices[i], extent)
            if not at.is_ok():
                return TensorResult[Tensor[Self.rank, HostStorage]](at.status)
        var dims = source.shape.dims.copy()
        dims.set(selected_axis, len(indices))
        var shape = TensorShape[Self.rank].checked(dims)
        if not shape.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](shape.status)
        var allocated = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(shape.payload.value())
        if not allocated.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](allocated.status)
        var result = allocated.payload.take()
        var src = self.impl.storage.borrow_span()
        var dst = result.impl.storage.borrow_mut_span()
        # A borrowed Int index view must not alias this fresh Float32 output
        # even through an unsafe external reinterpretation of memory.
        if not host_output_disjoint_from_indices(dst, indices):
            return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.ALIAS_UNKNOWN)
        var gathered = cpu_gather_kernel[Self.rank](
            src, dst, source, shape.payload.value(), selected_axis, indices,
        )
        if gathered != TensorStatus.OK:
            return TensorResult[Tensor[Self.rank, HostStorage]](gathered)
        return result._finish_complete_overwrite()

    # Scatter uses stable sequential last-write semantics for repeated indices.
    # Validate every shape/target/index before modifying the destination.
    def scatter_from[origin: Origin](
        mut self, axis: Int, indices: Span[False, Int, origin],
        ref source: Tensor[Self.rank, HostStorage], conflict_code: Int = 0,
    ) -> TensorStatus where identical(Self.S, HostStorage):
        if conflict_code != 0:
            return TensorStatus.UNSUPPORTED
        var dst_result = self._checked_current_layout()
        if not dst_result.is_ok():
            return dst_result.status
        var src_result = source._checked_current_layout()
        if not src_result.is_ok():
            return src_result.status
        var axis_result = normalize_axis[Self.rank](axis)
        if not axis_result.is_ok():
            return axis_result.status
        var selected_axis = axis_result.payload.value()
        var dst_layout = dst_result.payload.value()
        var src_layout = src_result.payload.value()
        if not dst_layout.is_proven_unique():
            return TensorStatus.OVERLAPPING
        comptime for i in range(Self.rank):
            if i == selected_axis:
                if src_layout.shape.dims.get(i) != len(indices):
                    return TensorStatus.SIZE_MISMATCH
            else:
                if src_layout.shape.dims.get(i) != dst_layout.shape.dims.get(i):
                    return TensorStatus.SIZE_MISMATCH
        var extent = dst_layout.shape.dims.get(selected_axis)
        # Strict monotonicity is a cheap, sufficient proof of no duplicates;
        # arbitrary/duplicate index sequences preserve ordered last-wins.
        var strictly_monotonic = True
        var previous = 0
        var direction = 0
        for i in range(len(indices)):
            var at = normalize_index(indices[i], extent)
            if not at.is_ok():
                return at.status
            var normalized_index = at.payload.value()
            if i > 0:
                if normalized_index == previous:
                    strictly_monotonic = False
                elif direction == 0:
                    if normalized_index > previous:
                        direction = 1
                    else:
                        direction = -1
                elif (direction == 1 and normalized_index <= previous
                      or direction == -1 and normalized_index >= previous):
                    strictly_monotonic = False
            previous = normalized_index
        if src_layout.shape.numel() == 0:
            return TensorStatus.OK
        var src = source.impl.storage.borrow_span()
        var dst = self.impl.storage.borrow_mut_span()
        # Both Float32 source AND the Int index vector must be independent
        # of the writable destination. Otherwise a write could corrupt a
        # future unread value or even alter its own address computation.
        if not host_output_disjoint_from_input(dst, src):
            return TensorStatus.ALIAS_UNKNOWN
        if not host_output_disjoint_from_indices(dst, indices):
            return TensorStatus.ALIAS_UNKNOWN
        var scattered = cpu_scatter_kernel[Self.rank](
            src, dst, src_layout, dst_layout, selected_axis, indices,
            strictly_monotonic,
        )
        if scattered != TensorStatus.OK:
            # A lower-level writer may report failure after changing a
            # destination element.  Conservatively revoke its whole-owner
            # readability certificate; an invalid index is checked above
            # before this call and cannot trigger such invalidation.
            self.impl.storage._invalidate_after_failed_write()
        return scattered

    def repeat_axis(ref self, axis: Int, copies: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._repeat_or_tile_axis_host[False](axis, copies)

    def repeat(ref self, repeats: Int, axis: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        # Repeat materializes the output; split/chunk never do.
        return self.repeat_axis(axis, repeats)

    def tile_axis(ref self, axis: Int, copies: Int) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        return self._repeat_or_tile_axis_host[True](axis, copies)

    def tile(ref self, reps: StaticRankArray[Self.rank]) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage):
        # All axes repeat as complete blocks. The scalar (rank 0) has a
        # well-defined empty repetition tuple and remains a scalar.
        var proof = self._checked_current_layout()
        if not proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](proof.status)
        var source_layout = proof.payload.value()
        var dims = source_layout.shape.dims.copy()
        comptime for i in range(Self.rank):
            var times = reps.get(i)
            if times < 0:
                return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.INVALID_SHAPE)
            var multiplied = checked_mul_nonnegative(dims.get(i), times)
            if not multiplied.is_ok():
                return TensorResult[Tensor[Self.rank, HostStorage]](multiplied.status)
            dims.set(i, multiplied.payload.value())
        var shaped = TensorShape[Self.rank].checked(dims)
        if not shaped.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](shaped.status)
        var allocated = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(shaped.payload.value())
        if not allocated.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](allocated.status)
        var result = allocated.payload.take()
        var src = self.impl.storage.borrow_span()
        var dst = result.impl.storage.borrow_mut_span()
        var cursor = NDIndexIterator[Self.rank](shaped.payload.value())
        while not cursor.done():
            var coord = cursor.coordinates.copy()
            comptime for i in range(Self.rank):
                coord.set(i, coord.get(i) % source_layout.shape.dims.get(i))
            dst[cursor.produced] = src[source_layout.physical_offset_unchecked(coord)]
            cursor.advance()
        return result._finish_complete_overwrite()

    def flatten_scalar_view(
        ref self
    ) -> TensorResult[TensorView[1, origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage) and Self.rank == 0:
        # Metadata-only scalar-to-vector reshape; rank zero has numel=1.
        var one = TensorShape[1].checked(StaticRankArray[1](1))
        if not one.is_ok():
            return TensorResult[TensorView[1, origin_of(self.impl.storage)]](one.status)
        return self.reshape_view[1](one.payload.value())

    def flatten_view[start_dim: Int, end_dim: Int](
        ref self
    ) -> TensorResult[TensorView[Self.rank - (end_dim - start_dim), origin_of(self.impl.storage)]] where conforms_to(Self.S, HostSpanStorage) and Self.rank >= 1:
        comptime assert start_dim >= 0 and end_dim >= start_dim and end_dim < Self.rank, "invalid flatten axis interval"
        comptime out_rank = Self.rank - (end_dim - start_dim)
        # Shape arithmetic validates its own metadata. The canonical
        # reshape checker independently validates layout + owner capacity
        # exactly once, before any borrowed pointer is returned.
        var flattened = flatten_shape[Self.rank, start_dim, end_dim](
            self.impl.layout.shape
        )
        if not flattened.is_ok():
            return TensorResult[TensorView[out_rank, origin_of(self.impl.storage)]](flattened.status)
        var mapped = reshape_c_view[Self.rank, out_rank](
            self.impl.layout, flattened.payload.value(),
            self.impl.storage.capacity(),
        )
        if not mapped.is_ok():
            return TensorResult[TensorView[out_rank, origin_of(self.impl.storage)]](mapped.status)
        # Reuse the owner-validated layout: no allocation, no redundant
        # second owner-bound validation, no implicit materialization.
        return TensorResult[TensorView[out_rank, origin_of(self.impl.storage)]](
            TensorView[out_rank, origin_of(self.impl.storage)](
                self.impl.storage.borrow_span(), mapped.payload.take()
            )
        )


    # Two-source concatenate is part of the owning Tensor API, not an ops-only
    # helper. Rank is static, both sources are fully validated, and the output
    # is a fresh, fully overwritten Float32 allocation.
    def concatenate(
        ref self, ref other: Tensor[Self.rank, HostStorage], axis: Int
    ) -> TensorResult[Tensor[Self.rank, HostStorage]] where identical(Self.S, HostStorage) and Self.rank >= 1:
        var left_proof = self._checked_current_layout()
        if not left_proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](left_proof.status)
        var right_proof = other._checked_current_layout()
        if not right_proof.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](right_proof.status)
        var chosen = normalize_axis[Self.rank](axis)
        if not chosen.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](chosen.status)
        var ax = chosen.payload.value()
        var left = left_proof.payload.value()
        var right = right_proof.payload.value()
        var dims = left.shape.dims.copy()
        comptime for i in range(Self.rank):
            if i != ax and left.shape.dims.get(i) != right.shape.dims.get(i):
                return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var joined = checked_add(left.shape.dims.get(ax), right.shape.dims.get(ax))
        if not joined.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](joined.status)
        dims.set(ax, joined.payload.value())
        var shape = TensorShape[Self.rank].checked(dims)
        if not shape.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](shape.status)
        var made = Tensor[Self.rank, HostStorage]._empty_for_complete_overwrite_checked(shape.payload.value())
        if not made.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](made.status)
        var output = made.payload.take()
        var src_left = self.impl.storage.borrow_span()
        var src_right = other.impl.storage.borrow_span()
        var dst = output.impl.storage.borrow_mut_span()
        # Dense C-order owners have contiguous slices on BOTH sides of any
        # concatenation axis. Copy outer slabs with only two memcpy calls per
        # slab instead of traversing one Float32 element at a time.
        if left.is_c_contiguous() and right.is_c_contiguous():
            if left.is_exhaustive() and right.is_exhaustive():
                var total = output.impl.layout.shape.numel()
                if total == 0:
                    return output._finish_complete_overwrite()
                var suffix = 1
                comptime for i in range(Self.rank):
                    if i > ax:
                        suffix *= dims.get(i)
                var left_per_slab = left.shape.dims.get(ax) * suffix
                var right_per_slab = right.shape.dims.get(ax) * suffix
                var output_per_slab = left_per_slab + right_per_slab
                # A nonempty checked output proves output_per_slab > 0 and
                # every intermediate product <= total <= Int.MAX / 4.
                var slabs = total // output_per_slab
                for slab in range(slabs):
                    var target = slab * output_per_slab
                    if left_per_slab > 0:
                        var source = left.span_lo + slab * left_per_slab
                        var left_run = src_left[source:source + left_per_slab]
                        var left_dst = dst[target:target + left_per_slab]
                        unsafe_memcpy(dest=left_dst.unsafe_ptr(), src=left_run.unsafe_ptr(), count=left_per_slab)
                    if right_per_slab > 0:
                        var source = right.span_lo + slab * right_per_slab
                        var start = target + left_per_slab
                        var right_run = src_right[source:source + right_per_slab]
                        var right_dst = dst[start:start + right_per_slab]
                        unsafe_memcpy(dest=right_dst.unsafe_ptr(), src=right_run.unsafe_ptr(), count=right_per_slab)
                return output._finish_complete_overwrite()
        var out_layout = output.impl.layout
        var cursor_left = NDIndexIterator[Self.rank](left.shape)
        while not cursor_left.done():
            var src_offset = left.physical_offset_unchecked(cursor_left.coordinates)
            var dst_offset = out_layout.physical_offset_unchecked(cursor_left.coordinates)
            dst[dst_offset] = src_left[src_offset]
            cursor_left.advance()
        var cursor_right = NDIndexIterator[Self.rank](right.shape)
        while not cursor_right.done():
            var out_coord = cursor_right.coordinates.copy()
            out_coord.set(ax, out_coord.get(ax) + left.shape.dims.get(ax))
            var src_offset = right.physical_offset_unchecked(cursor_right.coordinates)
            var dst_offset = out_layout.physical_offset_unchecked(out_coord)
            dst[dst_offset] = src_right[src_offset]
            cursor_right.advance()
        return output._finish_complete_overwrite()

    # A two-input stack inserts a size-two axis. The output is rank+1 at
    # compile time; rank-0 scalars and rank-7 inputs are fully supported.
    def stack(
        ref self, ref other: Tensor[Self.rank, HostStorage], axis: Int
    ) -> TensorResult[Tensor[Self.rank + 1, HostStorage]] where identical(Self.S, HostStorage) and Self.rank < 8:
        var left_proof = self._checked_current_layout()
        if not left_proof.is_ok():
            return TensorResult[Tensor[Self.rank + 1, HostStorage]](left_proof.status)
        var right_proof = other._checked_current_layout()
        if not right_proof.is_ok():
            return TensorResult[Tensor[Self.rank + 1, HostStorage]](right_proof.status)
        var left = left_proof.payload.value()
        var right = right_proof.payload.value()
        if not left.shape.same_extents(right.shape):
            return TensorResult[Tensor[Self.rank + 1, HostStorage]](TensorStatus.SIZE_MISMATCH)
        var chosen = normalize_axis[Self.rank + 1](axis)
        if not chosen.is_ok():
            return TensorResult[Tensor[Self.rank + 1, HostStorage]](chosen.status)
        var ax = chosen.payload.value()
        var dims = StaticRankArray[Self.rank + 1](1)
        comptime for i in range(Self.rank + 1):
            if i < ax:
                dims.set(i, left.shape.dims.get(i))
            elif i == ax:
                dims.set(i, 2)
            else:
                dims.set(i, left.shape.dims.get(i - 1))
        var shape = TensorShape[Self.rank + 1].checked(dims)
        if not shape.is_ok():
            return TensorResult[Tensor[Self.rank + 1, HostStorage]](shape.status)
        var made = Tensor[Self.rank + 1, HostStorage]._empty_for_complete_overwrite_checked(shape.payload.value())
        if not made.is_ok():
            return TensorResult[Tensor[Self.rank + 1, HostStorage]](made.status)
        var output = made.payload.take()
        var src_left = self.impl.storage.borrow_span()
        var src_right = other.impl.storage.borrow_span()
        var dst = output.impl.storage.borrow_mut_span()
        # Stacking on ANY axis of two C-contiguous tensors consists of
        # alternating contiguous source slabs in the C-order destination.
        # Scalar rank zero naturally uses one slab of one Float32 element.
        if left.is_c_contiguous() and right.is_c_contiguous():
            if left.is_exhaustive() and right.is_exhaustive():
                var total_source = left.shape.numel()
                if total_source == 0:
                    return output._finish_complete_overwrite()
                var elements_per_slab = 1
                comptime for i in range(Self.rank):
                    if i >= ax:
                        elements_per_slab *= left.shape.dims.get(i)
                var slabs = total_source // elements_per_slab
                for slab in range(slabs):
                    var left_at = left.span_lo + slab * elements_per_slab
                    var right_at = right.span_lo + slab * elements_per_slab
                    var out_at = slab * (2 * elements_per_slab)
                    var first = src_left[left_at:left_at + elements_per_slab]
                    var second = src_right[right_at:right_at + elements_per_slab]
                    var first_dst = dst[out_at:out_at + elements_per_slab]
                    var second_dst = dst[out_at + elements_per_slab:out_at + 2 * elements_per_slab]
                    unsafe_memcpy(dest=first_dst.unsafe_ptr(), src=first.unsafe_ptr(), count=elements_per_slab)
                    unsafe_memcpy(dest=second_dst.unsafe_ptr(), src=second.unsafe_ptr(), count=elements_per_slab)
                return output._finish_complete_overwrite()
        var out_layout = output.impl.layout
        var cursor = NDIndexIterator[Self.rank](left.shape)
        var output_coord = StaticRankArray[Self.rank + 1](0)
        while not cursor.done():
            comptime for i in range(Self.rank):
                if i < ax:
                    output_coord.set(i, cursor.coordinates.get(i))
                else:
                    output_coord.set(i + 1, cursor.coordinates.get(i))
            var left_offset = left.physical_offset_unchecked(cursor.coordinates)
            var right_offset = right.physical_offset_unchecked(cursor.coordinates)
            output_coord.set(ax, 0)
            dst[out_layout.physical_offset_unchecked(output_coord)] = src_left[left_offset]
            output_coord.set(ax, 1)
            dst[out_layout.physical_offset_unchecked(output_coord)] = src_right[right_offset]
            cursor.advance()
        return output._finish_complete_overwrite()


# This is an ALLOCATION/INITIALIZATION CAPABILITY, not another readable Tensor
# family. No get(), view(), data_ptr(), __getitem__() or kernel-input API is
# exposed. Tensor remains the single public owning tensor once initialized.
# Internal complete-overwrite kernels may use the explicitly unsafe private
# constructor to avoid a redundant initialization pass.
struct UninitializedHostTensor[rank: Int](Movable):
    var _storage: HostStorage
    var _layout: TensorLayout[Self.rank]

    def __init__(out self, var storage: HostStorage, layout: TensorLayout[Self.rank]):
        comptime assert Self.rank >= 0 and Self.rank <= 8, "uninitialized rank must be 0..8"
        self._storage = storage
        self._layout = layout

    @staticmethod
    def allocate_checked(
        shape: TensorShape[Self.rank], order: ContiguousOrder
    ) -> TensorResult[Self]:
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        var trusted = canonical.payload.value()
        if trusted.numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var layout: TensorResult[TensorLayout[Self.rank]]
        if order == ContiguousOrder.C:
            layout = TensorLayout[Self.rank].c_contiguous(trusted, trusted.numel())
        elif order == ContiguousOrder.F:
            layout = TensorLayout[Self.rank].f_contiguous(trusted, trusted.numel())
        else:
            return TensorResult[Self](TensorStatus.INVALID_LAYOUT)
        if not layout.is_ok():
            return TensorResult[Self](layout.status)
        var owned = HostStorage._allocate_for_complete_overwrite(trusted.numel())
        if not owned.is_ok():
            return TensorResult[Self](owned.status)
        return TensorResult[Self](Self(owned.payload.take(), layout.payload.value()))

    def checked_numel(self) -> TensorResult[Int]:
        var canonical = self._layout.validated_snapshot(self._storage.capacity())
        if not canonical.is_ok():
            return TensorResult[Int](canonical.status)
        return TensorResult[Int](canonical.payload.value().shape.numel())

    def initialize_fill(deinit self, value: Float32) -> TensorResult[Tensor[Self.rank, HostStorage]]:
        # A complete write is the transition to readable Tensor ownership.
        # There is no extra zero-initialization: the only write is the fill.
        var verified = self._layout.validated_snapshot(self._storage.capacity())
        if not verified.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](verified.status)
        var filled = self._storage.fill_host(value)
        if filled != TensorStatus.OK:
            return TensorResult[Tensor[Self.rank, HostStorage]](filled)
        return Tensor[Self.rank, HostStorage].from_parts_checked(
            self._storage, verified.payload.value()
        )

    def initialize_from_physical_span[origin: Origin](
        deinit self, source: Span[False, Float32, origin]
    ) -> TensorResult[Tensor[Self.rank, HostStorage]]:
        # Complete physical-order initialization, also valid for F-order.
        # Since this fresh allocation has never been exposed as a Span,
        # it cannot alias a user-provided source through a safe API.
        var verified = self._layout.validated_snapshot(self._storage.capacity())
        if not verified.is_ok():
            return TensorResult[Tensor[Self.rank, HostStorage]](verified.status)
        if len(source) != verified.payload.value().shape.numel():
            return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.SIZE_MISMATCH)
        # The copy engine is a proven-disjoint memcpy. Even an expert caller
        # who obtained a raw Span to this allocation must not invoke memcpy
        # with overlapping/unknown regions. Reject BEFORE the first write.
        # A zero-element tensor needs no dereference and is handled by the
        # same range predicate.
        if not host_output_disjoint_from_input(self._storage.borrow_mut_span(), source):
            return TensorResult[Tensor[Self.rank, HostStorage]](TensorStatus.OVERLAPPING)
        var copied = self._storage._copy_from_disjoint_span_unchecked(source)
        if copied != TensorStatus.OK:
            return TensorResult[Tensor[Self.rank, HostStorage]](copied)
        return Tensor[Self.rank, HostStorage].from_parts_checked(
            self._storage, verified.payload.value()
        )


# GPU counterpart of the allocation-only initialization capability above.
# No view, GPU kernel input, D2H transfer or data pointer can be obtained from
# this type. Tensor[rank, DeviceStorage] is returned only after a complete,
# explicitly synchronized Float32 fill. No implicit host migration exists.
struct UninitializedDeviceTensor[rank: Int](Movable):
    var _storage: DeviceStorage
    var _layout: TensorLayout[Self.rank]

    def __init__(out self, var storage: DeviceStorage, layout: TensorLayout[Self.rank]):
        comptime assert Self.rank >= 0 and Self.rank <= 8, "device rank must be 0..8"
        self._storage = storage
        self._layout = layout

    def _checked_full_owner_output_layout(ref self) -> TensorResult[TensorLayout[Self.rank]]:
        # Every native initializer is a claimed COMPLETE allocation write.
        # Prove the exact mapped physical owner, not just that a view is in
        # bounds: a short slice or non-exhaustive strided output must never
        # transition an otherwise unreadable allocation to initialized.
        var proof = self._layout.validated_snapshot(self._storage.capacity())
        if not proof.is_ok():
            return TensorResult[TensorLayout[Self.rank]](proof.status)
        var mapping = proof.payload.value()
        if mapping.shape.numel() != self._storage.capacity():
            return TensorResult[TensorLayout[Self.rank]](TensorStatus.SIZE_MISMATCH)
        if not mapping.is_proven_unique():
            return TensorResult[TensorLayout[Self.rank]](TensorStatus.OVERLAPPING)
        if (mapping.shape.numel() > 0 and not mapping.is_exhaustive()) or mapping.span_lo != 0:
            return TensorResult[TensorLayout[Self.rank]](TensorStatus.NOT_REPRESENTABLE)
        return TensorResult[TensorLayout[Self.rank]](mapping)

    @staticmethod
    def allocate_checked(
        ctx: DeviceContext, shape: TensorShape[Self.rank]
    ) -> TensorResult[Self]:
        var canonical = TensorShape[Self.rank].checked(shape.dims)
        if not canonical.is_ok():
            return TensorResult[Self](canonical.status)
        var trusted = canonical.payload.value()
        if trusted.numel() != shape.numel():
            return TensorResult[Self](TensorStatus.INVALID_SHAPE)
        var count = trusted.numel()
        var mapping = TensorLayout[Self.rank].c_contiguous(trusted, count)
        if not mapping.is_ok():
            return TensorResult[Self](mapping.status)
        # Identical Event/Stream quarantine contract as full_device_checked,
        # but the resulting storage is kept only in this unreadable builder.
        var owned = DeviceStorage._allocate_owned_checked(
            ctx, count, False, Float32(0.0),
        )
        if not owned.is_ok():
            return TensorResult[Self](owned.status)
        return TensorResult[Self](
            Self(owned.payload.take(), mapping.payload.value())
        )

    def initialize_strided_binary[
        operation: Int, left_rank: Int, right_rank: Int,
    ](
        deinit self, ctx: DeviceContext, selected: ExecutionPlan,
        var left: Tensor[left_rank, DeviceStorage],
        var right: Tensor[right_rank, DeviceStorage],
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorResult[Tensor[Self.rank, DeviceStorage]]:
        # The DeviceBuffer enqueues on the context which originally owns it.
        # A caller-supplied context may reference a different stream/device;
        # its event would not prove that this output write has completed.
        # The caller context must match the retained owner. Foreign producers
        # or independent streams still require separately authenticated events.
        if (not self._storage.lifetime_context
            or self._storage.quiescence_unproven
            or self._storage.quarantined_context):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        if not left.impl.storage._shares_issuing_context(self._storage):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        if not right.impl.storage._shares_issuing_context(self._storage):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var issuing = self._storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        # An uninitialized output is filled by exactly one native kernel;
        # there is NO preliminary device memset and no hidden data transfer.
        # Input owners move into this call so they can be quarantined after
        # a failure for which native stream quiescence cannot be proved.
        var target_proof = self._checked_full_owner_output_layout()
        if not target_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](target_proof.status)
        var lproof = left._checked_current_layout()
        if not lproof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](lproof.status)
        var rproof = right._checked_current_layout()
        if not rproof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](rproof.status)
        var target = target_proof.payload.value()
        var outcome = run_owned_gpu_binary_strided_fp32[
            Self.rank, left_rank, right_rank, operation,
        ](
            issuing, selected, development_allow_runtime_compilation, self._storage.buffer,
            left.impl.storage.buffer, right.impl.storage.buffer,
            target, lproof.payload.value(), rproof.payload.value(),
        )
        if outcome != TensorStatus.OK:
            if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
                forget_deinit(left)
                forget_deinit(right)
            self._storage._invalidate_after_failed_write(
                issuing, outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
            return TensorResult[Tensor[Self.rank, DeviceStorage]](outcome)
        var ready = self._storage._mark_complete_overwrite()
        if ready != TensorStatus.OK:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](ready)
        return Tensor[Self.rank, DeviceStorage].from_parts_checked(
            self._storage, target
        )

    def initialize_strided_unary[operation: Int](
        deinit self, ctx: DeviceContext, selected: ExecutionPlan,
        var source: Tensor[Self.rank, DeviceStorage],
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorResult[Tensor[Self.rank, DeviceStorage]]:
        # The DeviceBuffer enqueues on the context which originally owns it.
        # A caller-supplied context may reference a different stream/device;
        # its event would not prove that this output write has completed.
        # The caller context must match the retained owner. Foreign producers
        # or independent streams still require separately authenticated events.
        if (not self._storage.lifetime_context
            or self._storage.quiescence_unproven
            or self._storage.quarantined_context):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        if not source.impl.storage._shares_issuing_context(self._storage):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var issuing = self._storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var target_proof = self._checked_full_owner_output_layout()
        if not target_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](target_proof.status)
        var src_proof = source._checked_current_layout()
        if not src_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](src_proof.status)
        var target = target_proof.payload.value()
        var outcome = run_owned_gpu_unary_strided_fp32[Self.rank, operation](
            issuing, selected, development_allow_runtime_compilation, self._storage.buffer, source.impl.storage.buffer,
            target, src_proof.payload.value(),
        )
        if outcome != TensorStatus.OK:
            if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
                forget_deinit(source)
            self._storage._invalidate_after_failed_write(
                issuing, outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
            return TensorResult[Tensor[Self.rank, DeviceStorage]](outcome)
        var ready = self._storage._mark_complete_overwrite()
        if ready != TensorStatus.OK:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](ready)
        return Tensor[Self.rank, DeviceStorage].from_parts_checked(
            self._storage, target
        )

    def initialize_strided_reduce_axis[operation: Int, axis: Int](
        deinit self, ctx: DeviceContext, selected: ExecutionPlan,
        var source: Tensor[Self.rank, DeviceStorage],
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorResult[Tensor[Self.rank, DeviceStorage]] where Self.rank >= 1:
        # The DeviceBuffer enqueues on the context which originally owns it.
        # A caller-supplied context may reference a different stream/device;
        # its event would not prove that this output write has completed.
        # The caller context must match the retained owner. Foreign producers
        # or independent streams still require separately authenticated events.
        if (not self._storage.lifetime_context
            or self._storage.quiescence_unproven
            or self._storage.quarantined_context):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        if not source.impl.storage._shares_issuing_context(self._storage):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var issuing = self._storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var target_proof = self._checked_full_owner_output_layout()
        if not target_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](target_proof.status)
        var src_proof = source._checked_current_layout()
        if not src_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](src_proof.status)
        var target = target_proof.payload.value()
        var outcome = run_owned_gpu_reduce_axis_strided_fp32[
            Self.rank, axis, operation,
        ](
            issuing, selected, development_allow_runtime_compilation, self._storage.buffer, source.impl.storage.buffer,
            target, src_proof.payload.value(),
        )
        if outcome != TensorStatus.OK:
            if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
                forget_deinit(source)
            self._storage._invalidate_after_failed_write(
                issuing, outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
            return TensorResult[Tensor[Self.rank, DeviceStorage]](outcome)
        var ready = self._storage._mark_complete_overwrite()
        if ready != TensorStatus.OK:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](ready)
        return Tensor[Self.rank, DeviceStorage].from_parts_checked(
            self._storage, target
        )

    def initialize_strided_reduce_axes[operation: Int](
        deinit self, ctx: DeviceContext, selected: ExecutionPlan,
        var source: Tensor[Self.rank, DeviceStorage], axes_mask: Int,
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorResult[Tensor[Self.rank, DeviceStorage]] where Self.rank >= 1:
        # The DeviceBuffer enqueues on the context which originally owns it.
        # A caller-supplied context may reference a different stream/device;
        # its event would not prove that this output write has completed.
        # The caller context must match the retained owner. Foreign producers
        # or independent streams still require separately authenticated events.
        if (not self._storage.lifetime_context
            or self._storage.quiescence_unproven
            or self._storage.quarantined_context):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        if not source.impl.storage._shares_issuing_context(self._storage):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var issuing = self._storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var target_proof = self._checked_full_owner_output_layout()
        if not target_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](target_proof.status)
        var src_proof = source._checked_current_layout()
        if not src_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](src_proof.status)
        var target = target_proof.payload.value()
        var outcome = run_owned_gpu_reduce_axes_strided_fp32[
            Self.rank, operation,
        ](
            issuing, selected, development_allow_runtime_compilation,
            self._storage.buffer, source.impl.storage.buffer,
            target, src_proof.payload.value(), axes_mask,
        )
        if outcome != TensorStatus.OK:
            if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
                forget_deinit(source)
            self._storage._invalidate_after_failed_write(
                issuing, outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
            return TensorResult[Tensor[Self.rank, DeviceStorage]](outcome)
        var ready = self._storage._mark_complete_overwrite()
        if ready != TensorStatus.OK:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](ready)
        return Tensor[Self.rank, DeviceStorage].from_parts_checked(
            self._storage, target
        )

    def initialize_strided_scan_axis[operation: Int, axis: Int](
        deinit self, ctx: DeviceContext, selected: ExecutionPlan,
        var source: Tensor[Self.rank, DeviceStorage],
        development_allow_runtime_compilation: Bool = False,
    ) -> TensorResult[Tensor[Self.rank, DeviceStorage]] where Self.rank >= 1:
        # The DeviceBuffer enqueues on the context which originally owns it.
        # A caller-supplied context may reference a different stream/device;
        # its event would not prove that this output write has completed.
        # The caller context must match the retained owner. Foreign producers
        # or independent streams still require separately authenticated events.
        if (not self._storage.lifetime_context
            or self._storage.quiescence_unproven
            or self._storage.quarantined_context):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        if not source.impl.storage._shares_issuing_context(self._storage):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var issuing = self._storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var target_proof = self._checked_full_owner_output_layout()
        if not target_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](target_proof.status)
        var src_proof = source._checked_current_layout()
        if not src_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](src_proof.status)
        var target = target_proof.payload.value()
        var outcome = run_owned_gpu_scan_axis_strided_fp32[
            Self.rank, axis, operation,
        ](
            issuing, selected, development_allow_runtime_compilation, self._storage.buffer, source.impl.storage.buffer,
            target, src_proof.payload.value(),
        )
        if outcome != TensorStatus.OK:
            if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
                forget_deinit(source)
            self._storage._invalidate_after_failed_write(
                issuing, outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
            return TensorResult[Tensor[Self.rank, DeviceStorage]](outcome)
        var ready = self._storage._mark_complete_overwrite()
        if ready != TensorStatus.OK:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](ready)
        return Tensor[Self.rank, DeviceStorage].from_parts_checked(
            self._storage, target
        )

    def initialize_binary[operation: Int](
        deinit self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: BinaryAOTArtifactIdentity, target_arch: Int,
        selected: ExecutionPlan,
        var left: Tensor[Self.rank, DeviceStorage],
        var right: Tensor[Self.rank, DeviceStorage],
    ) -> TensorResult[Tensor[Self.rank, DeviceStorage]]:
        # The DeviceBuffer enqueues on the context which originally owns it.
        # A caller-supplied context may reference a different stream/device;
        # its event would not prove that this output write has completed.
        # The caller context must match the retained owner. Foreign producers
        # or independent streams still require separately authenticated events.
        if (not self._storage.lifetime_context
            or self._storage.quiescence_unproven
            or self._storage.quarantined_context):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        if not left.impl.storage._shares_issuing_context(self._storage):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        if not right.impl.storage._shares_issuing_context(self._storage):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var issuing = self._storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        # Complete AOT initialization of an allocation-only device output.
        # Do not zero the destination first: the checked binary adapter
        # writes every logical element before its joined event returns.
        comptime assert operation >= 0 and operation <= 5, "invalid binary operation"
        var verified = self._checked_full_owner_output_layout()
        if not verified.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](verified.status)
        var target = verified.payload.value()
        var left_proof = left._checked_current_layout()
        if not left_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](left_proof.status)
        var right_proof = right._checked_current_layout()
        if not right_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](right_proof.status)
        var a = left_proof.payload.value()
        var b = right_proof.payload.value()
        if not target.shape.same_extents(a.shape) or not target.shape.same_extents(b.shape):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](TensorStatus.SIZE_MISMATCH)
        if target.shape.numel() != self._storage.capacity():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](TensorStatus.SIZE_MISMATCH)
        if not target.is_proven_unique():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](TensorStatus.OVERLAPPING)
        if (not a.is_c_contiguous() or not b.is_c_contiguous()
            or not target.is_c_contiguous() or not a.is_exhaustive()
            or not b.is_exhaustive() or not target.is_exhaustive()
            or a.offset != 0 or b.offset != 0 or target.offset != 0):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](TensorStatus.NOT_REPRESENTABLE)
        var checked_plan = selected.checked_native_device_snapshot()
        if not checked_plan.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](checked_plan.status)
        var plan = checked_plan.payload.value()
        if plan.total_items != target.shape.numel():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](TensorStatus.SIZE_MISMATCH)
        var outcome = run_owned_bound_gpu_binary_fp32[operation](
            issuing, artifact, identity, target_arch, plan,
            self._storage.buffer, left.impl.storage.buffer, right.impl.storage.buffer,
        )
        if outcome != TensorStatus.OK:
            if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
                # The consuming initializer owns both input allocations.
                # A GPU without completion proof may still read them even
                # after the function returns, so retain those owners too.
                forget_deinit(left)
                forget_deinit(right)
            self._storage._invalidate_after_failed_write(
                issuing, outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
            return TensorResult[Tensor[Self.rank, DeviceStorage]](outcome)
        var completed = self._storage._mark_complete_overwrite()
        if completed != TensorStatus.OK:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](completed)
        return Tensor[Self.rank, DeviceStorage].from_parts_checked(
            self._storage, target,
        )

    def initialize_unary[operation: Int](
        deinit self, ctx: DeviceContext, artifact: DeviceExternalFunction,
        identity: UnaryAOTArtifactIdentity, target_arch: Int,
        selected: ExecutionPlan,
        var source: Tensor[Self.rank, DeviceStorage],
    ) -> TensorResult[Tensor[Self.rank, DeviceStorage]]:
        # The DeviceBuffer enqueues on the context which originally owns it.
        # A caller-supplied context may reference a different stream/device;
        # its event would not prove that this output write has completed.
        # The caller context must match the retained owner. Foreign producers
        # or independent streams still require separately authenticated events.
        if (not self._storage.lifetime_context
            or self._storage.quiescence_unproven
            or self._storage.quarantined_context):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        if not source.impl.storage._shares_issuing_context(self._storage):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var issuing = self._storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        # The prelinked unary kernel fully overwrites the new device owner;
        # no redundant allocation fill or host staging is introduced.
        comptime assert operation >= 0 and operation <= 8, "invalid unary operation"
        var verified = self._checked_full_owner_output_layout()
        if not verified.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](verified.status)
        var target = verified.payload.value()
        var input_proof = source._checked_current_layout()
        if not input_proof.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](input_proof.status)
        var src = input_proof.payload.value()
        if not target.shape.same_extents(src.shape):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](TensorStatus.SIZE_MISMATCH)
        if target.shape.numel() != self._storage.capacity():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](TensorStatus.SIZE_MISMATCH)
        if not target.is_proven_unique():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](TensorStatus.OVERLAPPING)
        if (not src.is_c_contiguous() or not target.is_c_contiguous()
            or not src.is_exhaustive() or not target.is_exhaustive()
            or src.offset != 0 or target.offset != 0):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](TensorStatus.NOT_REPRESENTABLE)
        var checked_plan = selected.checked_native_device_snapshot()
        if not checked_plan.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](checked_plan.status)
        var plan = checked_plan.payload.value()
        if plan.total_items != target.shape.numel():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](TensorStatus.SIZE_MISMATCH)
        var outcome = run_owned_bound_gpu_unary_fp32[operation](
            issuing, artifact, identity, target_arch, plan,
            self._storage.buffer, source.impl.storage.buffer,
        )
        if outcome != TensorStatus.OK:
            if outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN:
                # The consuming initializer owns its source Tensor. A
                # failed event and stream fence leave that input potentially
                # in flight: do not free it while the native GPU can read.
                forget_deinit(source)
            self._storage._invalidate_after_failed_write(
                issuing, outcome == TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
            return TensorResult[Tensor[Self.rank, DeviceStorage]](outcome)
        var completed = self._storage._mark_complete_overwrite()
        if completed != TensorStatus.OK:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](completed)
        return Tensor[Self.rank, DeviceStorage].from_parts_checked(
            self._storage, target,
        )

    def initialize_fill(
        deinit self, ctx: DeviceContext, value: Float32
    ) -> TensorResult[Tensor[Self.rank, DeviceStorage]]:
        # The DeviceBuffer enqueues on the context which originally owns it.
        # A caller-supplied context may reference a different stream/device;
        # its event would not prove that this output write has completed.
        # The caller context must match the retained owner. Foreign producers
        # or independent streams still require separately authenticated events.
        if (not self._storage.lifetime_context
            or self._storage.quiescence_unproven
            or self._storage.quarantined_context):
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.GPU_QUIESCENCE_UNPROVEN
            )
        var issuing = self._storage.lifetime_context.value().copy()
        if ctx != issuing:
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.DEVICE_MISMATCH
            )
        var verified = self._checked_full_owner_output_layout()
        if not verified.is_ok():
            return TensorResult[Tensor[Self.rank, DeviceStorage]](verified.status)
        var total = verified.payload.value().shape.numel()
        try:
            # The padded physical element of an empty allocation need not be
            # written because no logical element can ever read it.
            if total > 0:
                self._storage.buffer.enqueue_fill(value)
            # Even a zero-logical-element allocation is backed by a padded
            # physical DeviceBuffer whose asynchronous creation must join.
            var completed = issuing.create_event()
            issuing.stream().record_event(completed)
            completed.synchronize()
            var sealed = self._storage._mark_complete_overwrite()
            if sealed != TensorStatus.OK:
                return TensorResult[Tensor[Self.rank, DeviceStorage]](sealed)
            return Tensor[Self.rank, DeviceStorage].from_parts_checked(
                self._storage, verified.payload.value()
            )
        except:
            try:
                issuing.stream().synchronize()
            except:
                # DeviceStorage owns both allocation and retained context.
                # Mark it quarantined; its destructor will preserve BOTH
                # owners without an unsafe free of in-flight device memory.
                self._storage._invalidate_after_failed_write(issuing, True)
                return TensorResult[Tensor[Self.rank, DeviceStorage]](
                    TensorStatus.GPU_QUIESCENCE_UNPROVEN
                )
            # A successful fallback fence does not prove full initialization
            # after a native enqueue/event failure. Remain unreadable.
            self._storage._invalidate_after_failed_write(issuing, False)
            return TensorResult[Tensor[Self.rank, DeviceStorage]](
                TensorStatus.TRANSFER_UNAVAILABLE
            )
