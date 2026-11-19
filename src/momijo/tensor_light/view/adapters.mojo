# SPDX-FileCopyrightText: 2026 Morteza Taleblou
# SPDX-FileCopyrightText: 2026 Mitra Daneshmand
# SPDX-License-Identifier: MIT
#
# ==============================================================================
# MOMIJO | TENSOR VIEW / EXTERNAL VIEW ADAPTERS
# ==============================================================================
#
# Project:       Momijo
# Component:     Tensor
# Subcomponent:  FP32 Light / Tensor Views
# Module:        momijo.tensor_light.view.adapters
# Source:        src/momijo/tensor_light/view/adapters.mojo
# Language:      Mojo
#
# ------------------------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------------------------
#
# Description:
#   Defines external view adapters in the tensor views layer of the FP32 Light
#   tensor implementation.
#
#   Public declarations include `as_c_span`, `mut_as_c_span`, `as_f_span`,
#   `mut_as_f_span`; preconditions and supported targets are determined by
#   their implementations.
#
# Responsibilities:
#   - Build safe tensor views and layout-only transformations.
#   - Preserve documented behavior for bounds, strides, view provenance, and
#     write safety.
#   - Provide source-declared interfaces such as `as_c_span`, `mut_as_c_span`,
#     `as_f_span`, `mut_as_f_span`, `try_as_static_c_layout_tensor_1d` and
#     related symbols.
#
# ------------------------------------------------------------------------------
# DESIGN AND ENGINEERING GUIDELINES
# ------------------------------------------------------------------------------
#
# Correctness:
#   - Validate shape/stride/span invariants for borrowed views.
#   - Enforce write restrictions for broadcast or overlapping mappings.
#
# Performance Objectives:
#   - Prefer zero-copy metadata transforms when valid.
#   - Avoid view materialization unless the API explicitly requires it.
#
# Maintainability:
#   - Keep view semantics consistent with owning-tensor operations.
#   - Test permutations, negative strides, and zero-length axes.
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

# Zero-copy contiguous Span adapters. No hidden allocation or materialization.
# An offset contiguous subview is still representable as a borrowed Span.
from std.collections import Span
from layout import Layout, LayoutTensor, IntTuple
from layout.tile_tensor import TileTensor, LTToTTLayout, lt_to_tt
from .view import TensorView, TensorMutView
from ..internal.status import TensorResult, TensorStatus


# Checked adapters cannot treat cached layout flags as authority: view
# descriptors are value types with writable metadata. Rebuild the mapping
# before exposing raw borrowed spans to MAX or to external callers.
def as_c_span[rank: Int, o: Origin](
    view: TensorView[rank, o]
) -> TensorResult[Span[False, Float32, o]]:
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[Span[False, Float32, o]](proof.status)
    var mapping = proof.payload.value()
    if not mapping.is_c_contiguous() or not mapping.is_exhaustive():
        return TensorResult[Span[False, Float32, o]](TensorStatus.NOT_REPRESENTABLE)
    var begin = mapping.span_lo
    var count = mapping.shape.numel()
    if begin < 0 or begin > len(view.data) or count > len(view.data) - begin:
        return TensorResult[Span[False, Float32, o]](TensorStatus.OUT_OF_BOUNDS)
    return TensorResult[Span[False, Float32, o]](view.data[begin:begin + count])


def mut_as_c_span[rank: Int, o: Origin[mut=True]](
    mut view: TensorMutView[rank, o]
) -> TensorResult[Span[True, Float32, o]]:
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[Span[True, Float32, o]](proof.status)
    var mapping = proof.payload.value()
    if not mapping.is_proven_unique():
        return TensorResult[Span[True, Float32, o]](TensorStatus.OVERLAPPING)
    if not mapping.is_c_contiguous() or not mapping.is_exhaustive():
        return TensorResult[Span[True, Float32, o]](TensorStatus.NOT_REPRESENTABLE)
    var begin = mapping.span_lo
    var count = mapping.shape.numel()
    if begin < 0 or begin > len(view.data) or count > len(view.data) - begin:
        return TensorResult[Span[True, Float32, o]](TensorStatus.OUT_OF_BOUNDS)
    return TensorResult[Span[True, Float32, o]](view.data[begin:begin + count])


def as_f_span[rank: Int, o: Origin](
    view: TensorView[rank, o]
) -> TensorResult[Span[False, Float32, o]]:
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[Span[False, Float32, o]](proof.status)
    var mapping = proof.payload.value()
    if not mapping.is_f_contiguous() or not mapping.is_exhaustive():
        return TensorResult[Span[False, Float32, o]](TensorStatus.NOT_REPRESENTABLE)
    var begin = mapping.span_lo
    var count = mapping.shape.numel()
    if begin < 0 or begin > len(view.data) or count > len(view.data) - begin:
        return TensorResult[Span[False, Float32, o]](TensorStatus.OUT_OF_BOUNDS)
    return TensorResult[Span[False, Float32, o]](view.data[begin:begin + count])


def mut_as_f_span[rank: Int, o: Origin[mut=True]](
    mut view: TensorMutView[rank, o]
) -> TensorResult[Span[True, Float32, o]]:
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[Span[True, Float32, o]](proof.status)
    var mapping = proof.payload.value()
    if not mapping.is_proven_unique():
        return TensorResult[Span[True, Float32, o]](TensorStatus.OVERLAPPING)
    if not mapping.is_f_contiguous() or not mapping.is_exhaustive():
        return TensorResult[Span[True, Float32, o]](TensorStatus.NOT_REPRESENTABLE)
    var begin = mapping.span_lo
    var count = mapping.shape.numel()
    if begin < 0 or begin > len(view.data) or count > len(view.data) - begin:
        return TensorResult[Span[True, Float32, o]](TensorStatus.OUT_OF_BOUNDS)
    return TensorResult[Span[True, Float32, o]](view.data[begin:begin + count])


# AOT-exact, zero-copy, rank-one LayoutTensor bridge. MAX LayoutTensor
# requires a compile-time Layout. Therefore the caller must provide the
# static extent, and the actual checked runtime extent/layout MUST match.
# No pointer re-interpretation, storage adoption, unchecked dynamic strides,
# or implicit copy is performed. Borrowed origin ties it to the Core view.
def try_as_static_c_layout_tensor_1d[extent: Int, o: Origin](
    view: TensorView[1, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(extent), o]]:
    comptime assert extent > 0, "Static MAX LayoutTensor extent must be positive"
    comptime MAXLayout = Layout.row_major(extent)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    if verified.payload.value().shape.numel() != extent:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var span_result = as_c_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](span_result.status)
    var borrowed = span_result.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_mut_static_c_layout_tensor_1d[extent: Int, o: Origin[mut=True]](
    mut view: TensorMutView[1, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(extent), o]]:
    comptime assert extent > 0, "Static MAX LayoutTensor extent must be positive"
    comptime MAXLayout = Layout.row_major(extent)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    if verified.payload.value().shape.numel() != extent:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var span_result = mut_as_c_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](span_result.status)
    var borrowed = span_result.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


# Exact AOT strided MAX bridges. Dynamic shape/stride metadata cannot be
# made a compile-time MAX Layout without an explicit specialization contract.
# The offset into the owner is translated to the borrowed physical envelope;
# the MAX Layout then addresses that envelope with the SAME Float32 strides.
# Negative strides and zero-size static extents have no supported bridge here.
# Read-only zero-stride broadcasts are legal for LayoutTensor but not mutable.
def try_as_static_strided_layout_tensor_1d[extent: Int, step: Int, o: Origin](
    view: TensorView[1, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(extent), IntTuple(step)), o]]:
    comptime assert extent > 0 and step >= 0, "positive shape and nonnegative stride required"
    comptime MAXLayout = Layout(IntTuple(extent), IntTuple(step))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if mapping.shape.dims.get(0) != extent or mapping.strides.get(0) != step:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_mut_static_strided_layout_tensor_1d[extent: Int, step: Int, o: Origin[mut=True]](
    mut view: TensorMutView[1, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(extent), IntTuple(step)), o]]:
    comptime assert extent > 0 and step >= 0, "positive shape and nonnegative stride required"
    comptime MAXLayout = Layout(IntTuple(extent), IntTuple(step))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if not mapping.is_proven_unique():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OVERLAPPING)
    if mapping.shape.dims.get(0) != extent or mapping.strides.get(0) != step:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_static_strided_layout_tensor_2d[
    rows: Int, cols: Int, row_step: Int, col_step: Int, o: Origin,
](view: TensorView[2, o]) -> TensorResult[
    LayoutTensor[DType.float32, Layout(IntTuple(rows, cols), IntTuple(row_step, col_step)), o]
]:
    comptime assert rows > 0 and cols > 0, "positive static dimensions required"
    comptime assert row_step >= 0 and col_step >= 0, "nonnegative strides required"
    comptime MAXLayout = Layout(IntTuple(rows, cols), IntTuple(row_step, col_step))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if (mapping.shape.dims.get(0) != rows or mapping.shape.dims.get(1) != cols
        or mapping.strides.get(0) != row_step or mapping.strides.get(1) != col_step):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_mut_static_strided_layout_tensor_2d[
    rows: Int, cols: Int, row_step: Int, col_step: Int, o: Origin[mut=True],
](mut view: TensorMutView[2, o]) -> TensorResult[
    LayoutTensor[DType.float32, Layout(IntTuple(rows, cols), IntTuple(row_step, col_step)), o]
]:
    comptime assert rows > 0 and cols > 0, "positive static dimensions required"
    comptime assert row_step >= 0 and col_step >= 0, "nonnegative strides required"
    comptime MAXLayout = Layout(IntTuple(rows, cols), IntTuple(row_step, col_step))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if not mapping.is_proven_unique():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OVERLAPPING)
    if (mapping.shape.dims.get(0) != rows or mapping.shape.dims.get(1) != cols
        or mapping.strides.get(0) != row_step or mapping.strides.get(1) != col_step):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


# TileTensor is admitted only for the nonnegative, non-broadcasting static
# layouts the selected MAX LTToTTLayout specialization can represent. The
# LayoutTensor preflight remains the single source of truth for pointer/bounds.
def try_as_static_strided_tile_tensor_1d[extent: Int, step: Int, o: Origin](
    view: TensorView[1, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(extent), IntTuple(step))], o]]:
    comptime assert extent > 0 and step > 0, "TileTensor bridge requires positive stride"
    comptime MAXLayout = Layout(IntTuple(extent), IntTuple(step))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_static_strided_layout_tensor_1d[extent, step](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(lt.payload.take())
    )


def try_as_mut_static_strided_tile_tensor_1d[extent: Int, step: Int, o: Origin[mut=True]](
    mut view: TensorMutView[1, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(extent), IntTuple(step))], o]]:
    comptime assert extent > 0 and step > 0, "TileTensor bridge requires positive stride"
    comptime MAXLayout = Layout(IntTuple(extent), IntTuple(step))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_mut_static_strided_layout_tensor_1d[extent, step](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(lt.payload.take())
    )


def try_as_static_strided_tile_tensor_2d[
    rows: Int, cols: Int, row_step: Int, col_step: Int, o: Origin,
](view: TensorView[2, o]) -> TensorResult[
    TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(rows, cols), IntTuple(row_step, col_step))], o]
]:
    comptime assert rows > 0 and cols > 0, "positive static dimensions required"
    comptime assert row_step > 0 and col_step > 0, "TileTensor bridge requires positive strides"
    comptime MAXLayout = Layout(IntTuple(rows, cols), IntTuple(row_step, col_step))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_static_strided_layout_tensor_2d[rows, cols, row_step, col_step](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(lt.payload.take())
    )


def try_as_mut_static_strided_tile_tensor_2d[
    rows: Int, cols: Int, row_step: Int, col_step: Int, o: Origin[mut=True],
](mut view: TensorMutView[2, o]) -> TensorResult[
    TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(rows, cols), IntTuple(row_step, col_step))], o]
]:
    comptime assert rows > 0 and cols > 0, "positive static dimensions required"
    comptime assert row_step > 0 and col_step > 0, "TileTensor bridge requires positive strides"
    comptime MAXLayout = Layout(IntTuple(rows, cols), IntTuple(row_step, col_step))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_mut_static_strided_layout_tensor_2d[rows, cols, row_step, col_step](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(lt.payload.take())
    )


# The 2-D form specializes the matrix geometry at AOT compile time. A
# transposed/non-C view is never reinterpreted as row-major; that would
# silently change element addresses rather than performing a legal view.
def try_as_static_c_layout_tensor_2d[rows: Int, cols: Int, o: Origin](
    view: TensorView[2, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(rows, cols), o]]:
    comptime assert rows > 0 and cols > 0, "Static MAX matrix dimensions must be positive"
    comptime MAXLayout = Layout.row_major(rows, cols)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var verified_shape = verified.payload.value().shape
    if verified_shape.dims.get(0) != rows or verified_shape.dims.get(1) != cols:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var contiguous = as_c_span(view)
    if not contiguous.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](contiguous.status)
    var borrowed = contiguous.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_mut_static_c_layout_tensor_2d[rows: Int, cols: Int, o: Origin[mut=True]](
    mut view: TensorMutView[2, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(rows, cols), o]]:
    comptime assert rows > 0 and cols > 0, "Static MAX matrix dimensions must be positive"
    comptime MAXLayout = Layout.row_major(rows, cols)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var verified_shape = verified.payload.value().shape
    if verified_shape.dims.get(0) != rows or verified_shape.dims.get(1) != cols:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var contiguous = mut_as_c_span(view)
    if not contiguous.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](contiguous.status)
    var borrowed = contiguous.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


# Zero-copy flattening in PHYSICAL F-order, not logical C-order. A genuinely
# Fortran-contiguous rank 2..8 owner has a single physical span but its C-flat
# index order differs. This explicitly named adapter never promises C order.
# Rank-0 scalars satisfy C and F order and are represented as one FP32 value.
def try_as_static_flat_f_layout_tensor[rank: Int, extent: Int, o: Origin](
    view: TensorView[rank, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(extent), o]]:
    comptime assert rank >= 0 and rank <= 8, "rank must be 0..8"
    comptime assert extent > 0, "positive static flattened extent"
    comptime MAXLayout = Layout.row_major(extent)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    if proof.payload.value().shape.numel() != extent:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var physical = as_f_span(view)
    if not physical.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            physical.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](physical.payload.take())
    )


def try_as_mut_static_flat_f_layout_tensor[
    rank: Int, extent: Int, o: Origin[mut=True],
](mut view: TensorMutView[rank, o]) -> TensorResult[
    LayoutTensor[DType.float32, Layout.row_major(extent), o]
]:
    comptime assert rank >= 0 and rank <= 8, "rank must be 0..8"
    comptime assert extent > 0, "positive static flattened extent"
    comptime MAXLayout = Layout.row_major(extent)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    if proof.payload.value().shape.numel() != extent:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var physical = mut_as_f_span(view)
    if not physical.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            physical.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](physical.payload.take())
    )


def try_as_static_flat_f_tile_tensor[rank: Int, extent: Int, o: Origin](
    view: TensorView[rank, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], o]]:
    comptime assert extent > 0, "positive static flattened extent"
    comptime TileLayout = LTToTTLayout[Layout.row_major(extent)]
    var physical = try_as_static_flat_f_layout_tensor[rank, extent](view)
    if not physical.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](physical.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(physical.payload.take())
    )


def try_as_mut_static_flat_f_tile_tensor[
    rank: Int, extent: Int, o: Origin[mut=True],
](mut view: TensorMutView[rank, o]) -> TensorResult[
    TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], o]
]:
    comptime assert extent > 0, "positive static flattened extent"
    comptime TileLayout = LTToTTLayout[Layout.row_major(extent)]
    var physical = try_as_mut_static_flat_f_layout_tensor[rank, extent](view)
    if not physical.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](physical.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(physical.payload.take())
    )


# Rank 0..8 to rank-1 AOT MAX bridge. Flattened logical C-order addresses are
# identical for C-contiguous layouts, including scalar rank 0 (extent 1).
# Noncontiguous, negative-stride, transposed and empty views return explicit
# NOT_REPRESENTABLE; this function never allocates or repacks storage.
def try_as_static_flat_c_layout_tensor[rank: Int, extent: Int, o: Origin](
    view: TensorView[rank, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(extent), o]]:
    comptime assert rank >= 0 and rank <= 8, "rank must be 0..8"
    comptime assert extent > 0, "MAX flattened extent must be positive"
    comptime MAXLayout = Layout.row_major(extent)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            verified.status
        )
    if verified.payload.value().shape.numel() != extent:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var contiguous = as_c_span(view)
    if not contiguous.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            contiguous.status
        )
    var borrowed = contiguous.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_mut_static_flat_c_layout_tensor[rank: Int, extent: Int, o: Origin[mut=True]](
    mut view: TensorMutView[rank, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(extent), o]]:
    comptime assert rank >= 0 and rank <= 8, "rank must be 0..8"
    comptime assert extent > 0, "MAX flattened extent must be positive"
    comptime MAXLayout = Layout.row_major(extent)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            verified.status
        )
    if verified.payload.value().shape.numel() != extent:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var contiguous = mut_as_c_span(view)
    if not contiguous.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            contiguous.status
        )
    var borrowed = contiguous.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


# Rank-preserving, AOT-exact, C-order LayoutTensor bridges (ranks 3..8).
# The compiler knows every extent; runtime layout must have the SAME
# extents and a checked single C-contiguous FP32 storage range.
# Zero dimensions deliberately return NOT_REPRESENTABLE through the
# comptime-positive shape requirement: no zero-element MAX buffer is
# represented or silently allocated. Borrowed origins preserve lifetime.
def try_as_static_c_layout_tensor_3d[d0: Int, d1: Int, d2: Int, o: Origin](
    view: TensorView[3, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(d0, d1, d2), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0, "AOT MAX shape extents must be positive"
    comptime MAXLayout = Layout.row_major(d0, d1, d2)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var mapping = verified.payload.value()
    if mapping.shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var checked_span = as_c_span(view)
    if not checked_span.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](checked_span.status)
    var borrowed = checked_span.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_mut_static_c_layout_tensor_3d[d0: Int, d1: Int, d2: Int, o: Origin[mut=True]](
    mut view: TensorMutView[3, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(d0, d1, d2), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0, "AOT MAX shape extents must be positive"
    comptime MAXLayout = Layout.row_major(d0, d1, d2)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var mapping = verified.payload.value()
    if mapping.shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var checked_span = mut_as_c_span(view)
    if not checked_span.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](checked_span.status)
    var borrowed = checked_span.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_static_c_layout_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, o: Origin](
    view: TensorView[4, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(d0, d1, d2, d3), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0, "AOT MAX shape extents must be positive"
    comptime MAXLayout = Layout.row_major(d0, d1, d2, d3)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var mapping = verified.payload.value()
    if mapping.shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var checked_span = as_c_span(view)
    if not checked_span.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](checked_span.status)
    var borrowed = checked_span.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_mut_static_c_layout_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, o: Origin[mut=True]](
    mut view: TensorMutView[4, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(d0, d1, d2, d3), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0, "AOT MAX shape extents must be positive"
    comptime MAXLayout = Layout.row_major(d0, d1, d2, d3)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var mapping = verified.payload.value()
    if mapping.shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var checked_span = mut_as_c_span(view)
    if not checked_span.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](checked_span.status)
    var borrowed = checked_span.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_static_c_layout_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, o: Origin](
    view: TensorView[5, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(d0, d1, d2, d3, d4), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0, "AOT MAX shape extents must be positive"
    comptime MAXLayout = Layout.row_major(d0, d1, d2, d3, d4)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var mapping = verified.payload.value()
    if mapping.shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var checked_span = as_c_span(view)
    if not checked_span.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](checked_span.status)
    var borrowed = checked_span.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_mut_static_c_layout_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, o: Origin[mut=True]](
    mut view: TensorMutView[5, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(d0, d1, d2, d3, d4), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0, "AOT MAX shape extents must be positive"
    comptime MAXLayout = Layout.row_major(d0, d1, d2, d3, d4)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var mapping = verified.payload.value()
    if mapping.shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var checked_span = mut_as_c_span(view)
    if not checked_span.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](checked_span.status)
    var borrowed = checked_span.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_static_c_layout_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, o: Origin](
    view: TensorView[6, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(d0, d1, d2, d3, d4, d5), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0, "AOT MAX shape extents must be positive"
    comptime MAXLayout = Layout.row_major(d0, d1, d2, d3, d4, d5)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var mapping = verified.payload.value()
    if mapping.shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(5) != d5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var checked_span = as_c_span(view)
    if not checked_span.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](checked_span.status)
    var borrowed = checked_span.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_mut_static_c_layout_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, o: Origin[mut=True]](
    mut view: TensorMutView[6, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(d0, d1, d2, d3, d4, d5), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0, "AOT MAX shape extents must be positive"
    comptime MAXLayout = Layout.row_major(d0, d1, d2, d3, d4, d5)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var mapping = verified.payload.value()
    if mapping.shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(5) != d5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var checked_span = mut_as_c_span(view)
    if not checked_span.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](checked_span.status)
    var borrowed = checked_span.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_static_c_layout_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, o: Origin](
    view: TensorView[7, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(d0, d1, d2, d3, d4, d5, d6), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0, "AOT MAX shape extents must be positive"
    comptime MAXLayout = Layout.row_major(d0, d1, d2, d3, d4, d5, d6)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var mapping = verified.payload.value()
    if mapping.shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(5) != d5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(6) != d6:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var checked_span = as_c_span(view)
    if not checked_span.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](checked_span.status)
    var borrowed = checked_span.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_mut_static_c_layout_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, o: Origin[mut=True]](
    mut view: TensorMutView[7, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(d0, d1, d2, d3, d4, d5, d6), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0, "AOT MAX shape extents must be positive"
    comptime MAXLayout = Layout.row_major(d0, d1, d2, d3, d4, d5, d6)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var mapping = verified.payload.value()
    if mapping.shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(5) != d5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(6) != d6:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var checked_span = mut_as_c_span(view)
    if not checked_span.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](checked_span.status)
    var borrowed = checked_span.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_static_c_layout_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, o: Origin](
    view: TensorView[8, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(d0, d1, d2, d3, d4, d5, d6, d7), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0 and d7 > 0, "AOT MAX shape extents must be positive"
    comptime MAXLayout = Layout.row_major(d0, d1, d2, d3, d4, d5, d6, d7)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var mapping = verified.payload.value()
    if mapping.shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(5) != d5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(6) != d6:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(7) != d7:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var checked_span = as_c_span(view)
    if not checked_span.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](checked_span.status)
    var borrowed = checked_span.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )


def try_as_mut_static_c_layout_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, o: Origin[mut=True]](
    mut view: TensorMutView[8, o]
) -> TensorResult[LayoutTensor[DType.float32, Layout.row_major(d0, d1, d2, d3, d4, d5, d6, d7), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0 and d7 > 0, "AOT MAX shape extents must be positive"
    comptime MAXLayout = Layout.row_major(d0, d1, d2, d3, d4, d5, d6, d7)
    var verified = view._checked_current_layout()
    if not verified.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](verified.status)
    var mapping = verified.payload.value()
    if mapping.shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(5) != d5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(6) != d6:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(7) != d7:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var checked_span = mut_as_c_span(view)
    if not checked_span.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](checked_span.status)
    var borrowed = checked_span.payload.value()
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )




# The MAX 26.6 TileTensor conversion is a VIEW of the very same validated
# LayoutTensor bytes. Both bridges below retain the source borrow origin and
# do not allocate, change dtype, change strides or assume CPU/GPU migration.
# The intermediate LayoutTensor is a move-only borrowed wrapper; converting
# it using lt_to_tt keeps the pointer and exact static address mapping.
# All other layouts are explicitly NOT_REPRESENTABLE instead of materialized.
def try_as_static_flat_c_tile_tensor[rank: Int, extent: Int, o: Origin](
    view: TensorView[rank, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], o]]:
    comptime assert rank >= 0 and rank <= 8, "Core supports ranks 0..8"
    comptime assert extent > 0, "MAX static flat tile requires nonempty shape"
    comptime ResultLayout = LTToTTLayout[Layout.row_major(extent)]
    var checked = try_as_static_flat_c_layout_tensor[rank, extent](view)
    if not checked.is_ok():
        return TensorResult[TileTensor[DType.float32, ResultLayout, o]](
            checked.status
        )
    var borrowed = checked.payload.take()
    return TensorResult[TileTensor[DType.float32, ResultLayout, o]](
        lt_to_tt(borrowed)
    )


def try_as_mut_static_flat_c_tile_tensor[
    rank: Int, extent: Int, o: Origin[mut=True],
](
    mut view: TensorMutView[rank, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], o]]:
    comptime assert rank >= 0 and rank <= 8, "Core supports ranks 0..8"
    comptime assert extent > 0, "MAX static flat tile requires nonempty shape"
    comptime ResultLayout = LTToTTLayout[Layout.row_major(extent)]
    var checked = try_as_mut_static_flat_c_layout_tensor[rank, extent](view)
    if not checked.is_ok():
        return TensorResult[TileTensor[DType.float32, ResultLayout, o]](
            checked.status
        )
    var borrowed = checked.payload.take()
    return TensorResult[TileTensor[DType.float32, ResultLayout, o]](
        lt_to_tt(borrowed)
    )


def try_as_static_c_tile_tensor_2d[rows: Int, cols: Int, o: Origin](
    view: TensorView[2, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(rows, cols)], o]]:
    comptime assert rows > 0 and cols > 0, "MAX static tile requires nonempty matrix"
    comptime ResultLayout = LTToTTLayout[Layout.row_major(rows, cols)]
    var checked = try_as_static_c_layout_tensor_2d[rows, cols](view)
    if not checked.is_ok():
        return TensorResult[TileTensor[DType.float32, ResultLayout, o]](
            checked.status
        )
    var borrowed = checked.payload.take()
    return TensorResult[TileTensor[DType.float32, ResultLayout, o]](
        lt_to_tt(borrowed)
    )


def try_as_mut_static_c_tile_tensor_2d[
    rows: Int, cols: Int, o: Origin[mut=True],
](
    mut view: TensorMutView[2, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(rows, cols)], o]]:
    comptime assert rows > 0 and cols > 0, "MAX static tile requires nonempty matrix"
    comptime ResultLayout = LTToTTLayout[Layout.row_major(rows, cols)]
    var checked = try_as_mut_static_c_layout_tensor_2d[rows, cols](view)
    if not checked.is_ok():
        return TensorResult[TileTensor[DType.float32, ResultLayout, o]](
            checked.status
        )
    var borrowed = checked.payload.take()
    return TensorResult[TileTensor[DType.float32, ResultLayout, o]](
        lt_to_tt(borrowed)
    )


# True column-major (F-order) two-dimensional bridges. Reading physical
# bytes as C-order would silently transpose values; instead compile the
# actual F strides into the borrowed MAX layout, without a temporary copy.
def try_as_static_f_tile_tensor_2d[rows: Int, cols: Int, o: Origin](
    view: TensorView[2, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(rows, cols)], o]]:
    comptime assert rows > 0 and cols > 0, "MAX static tile requires nonempty matrix"
    comptime LegacyLayout = Layout.col_major(rows, cols)
    comptime TileLayout = LTToTTLayout[LegacyLayout]
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](proof.status)
    var safe = proof.payload.value()
    if safe.shape.dims.get(0) != rows or safe.shape.dims.get(1) != cols:
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var borrowed_span = as_f_span(view)
    if not borrowed_span.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](borrowed_span.status)
    var legacy = LayoutTensor[DType.float32, LegacyLayout, o](
        borrowed_span.payload.take()
    )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(legacy))


def try_as_mut_static_f_tile_tensor_2d[
    rows: Int, cols: Int, o: Origin[mut=True],
](
    mut view: TensorMutView[2, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(rows, cols)], o]]:
    comptime assert rows > 0 and cols > 0, "MAX static tile requires nonempty matrix"
    comptime LegacyLayout = Layout.col_major(rows, cols)
    comptime TileLayout = LTToTTLayout[LegacyLayout]
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](proof.status)
    var safe = proof.payload.value()
    if safe.shape.dims.get(0) != rows or safe.shape.dims.get(1) != cols:
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    var borrowed_span = mut_as_f_span(view)
    if not borrowed_span.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](borrowed_span.status)
    var legacy = LayoutTensor[DType.float32, LegacyLayout, o](
        borrowed_span.payload.take()
    )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(legacy))


# Rank-preserving AOT-exact C-order TileTensor bridges (rank 1 and 3..8).
def try_as_static_c_tile_tensor_1d[extent: Int, o: Origin](
    view: TensorView[1, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(extent)]
    var proof = try_as_static_c_layout_tensor_1d[extent](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_mut_static_c_tile_tensor_1d[extent: Int, o: Origin[mut=True]](
    mut view: TensorMutView[1, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(extent)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(extent)]
    var proof = try_as_mut_static_c_layout_tensor_1d[extent](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_static_c_tile_tensor_3d[d0: Int, d1: Int, d2: Int, o: Origin](
    view: TensorView[3, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(d0, d1, d2)]
    var proof = try_as_static_c_layout_tensor_3d[d0, d1, d2](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_mut_static_c_tile_tensor_3d[d0: Int, d1: Int, d2: Int, o: Origin[mut=True]](
    mut view: TensorMutView[3, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(d0, d1, d2)]
    var proof = try_as_mut_static_c_layout_tensor_3d[d0, d1, d2](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_static_c_tile_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, o: Origin](
    view: TensorView[4, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(d0, d1, d2, d3)]
    var proof = try_as_static_c_layout_tensor_4d[d0, d1, d2, d3](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_mut_static_c_tile_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, o: Origin[mut=True]](
    mut view: TensorMutView[4, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(d0, d1, d2, d3)]
    var proof = try_as_mut_static_c_layout_tensor_4d[d0, d1, d2, d3](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_static_c_tile_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, o: Origin](
    view: TensorView[5, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4)]
    var proof = try_as_static_c_layout_tensor_5d[d0, d1, d2, d3, d4](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_mut_static_c_tile_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, o: Origin[mut=True]](
    mut view: TensorMutView[5, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4)]
    var proof = try_as_mut_static_c_layout_tensor_5d[d0, d1, d2, d3, d4](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_static_c_tile_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, o: Origin](
    view: TensorView[6, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5)]
    var proof = try_as_static_c_layout_tensor_6d[d0, d1, d2, d3, d4, d5](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_mut_static_c_tile_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, o: Origin[mut=True]](
    mut view: TensorMutView[6, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5)]
    var proof = try_as_mut_static_c_layout_tensor_6d[d0, d1, d2, d3, d4, d5](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_static_c_tile_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, o: Origin](
    view: TensorView[7, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6)]
    var proof = try_as_static_c_layout_tensor_7d[d0, d1, d2, d3, d4, d5, d6](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_mut_static_c_tile_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, o: Origin[mut=True]](
    mut view: TensorMutView[7, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6)]
    var proof = try_as_mut_static_c_layout_tensor_7d[d0, d1, d2, d3, d4, d5, d6](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_static_c_tile_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, o: Origin](
    view: TensorView[8, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6, d7)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6, d7)]
    var proof = try_as_static_c_layout_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

def try_as_mut_static_c_tile_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, o: Origin[mut=True]](
    mut view: TensorMutView[8, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6, d7)], o]]:
    # The existing LayoutTensor adapter checks exact static shape, owner
    # bounds, C-contiguity and write uniqueness before exposing a borrow.
    # `lt_to_tt` preserves data pointer, strides, origin, and Float32 type.
    comptime TileLayout = LTToTTLayout[Layout.row_major(d0, d1, d2, d3, d4, d5, d6, d7)]
    var proof = try_as_mut_static_c_layout_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    var borrowed = proof.payload.take()
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(borrowed)
    )

# Rank-preserving Fortran-order LayoutTensor and TileTensor bridges (rank 3..8).
def try_as_static_f_layout_tensor_3d[d0: Int, d1: Int, d2: Int, o: Origin](
    view: TensorView[3, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.col_major(d0, d1, d2), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0, "AOT MAX F-order extents must be positive"
    comptime MAXLayout = Layout.col_major(d0, d1, d2)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            proof.status
        )
    var shape = proof.payload.value().shape
    if shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var span_result = as_f_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            span_result.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](span_result.payload.value())
    )

def try_as_static_f_tile_tensor_3d[d0: Int, d1: Int, d2: Int, o: Origin](
    view: TensorView[3, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2)], o]]:
    comptime TileLayout = LTToTTLayout[Layout.col_major(d0, d1, d2)]
    var proof = try_as_static_f_layout_tensor_3d[d0, d1, d2](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(proof.payload.take())
    )

def try_as_mut_static_f_layout_tensor_3d[d0: Int, d1: Int, d2: Int, o: Origin[mut=True]](
    mut view: TensorMutView[3, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.col_major(d0, d1, d2), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0, "AOT MAX F-order extents must be positive"
    comptime MAXLayout = Layout.col_major(d0, d1, d2)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            proof.status
        )
    var shape = proof.payload.value().shape
    if shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var span_result = mut_as_f_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            span_result.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](span_result.payload.value())
    )

def try_as_mut_static_f_tile_tensor_3d[d0: Int, d1: Int, d2: Int, o: Origin[mut=True]](
    mut view: TensorMutView[3, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2)], o]]:
    comptime TileLayout = LTToTTLayout[Layout.col_major(d0, d1, d2)]
    var proof = try_as_mut_static_f_layout_tensor_3d[d0, d1, d2](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(proof.payload.take())
    )

def try_as_static_f_layout_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, o: Origin](
    view: TensorView[4, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.col_major(d0, d1, d2, d3), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0, "AOT MAX F-order extents must be positive"
    comptime MAXLayout = Layout.col_major(d0, d1, d2, d3)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            proof.status
        )
    var shape = proof.payload.value().shape
    if shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var span_result = as_f_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            span_result.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](span_result.payload.value())
    )

def try_as_static_f_tile_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, o: Origin](
    view: TensorView[4, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3)], o]]:
    comptime TileLayout = LTToTTLayout[Layout.col_major(d0, d1, d2, d3)]
    var proof = try_as_static_f_layout_tensor_4d[d0, d1, d2, d3](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(proof.payload.take())
    )

def try_as_mut_static_f_layout_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, o: Origin[mut=True]](
    mut view: TensorMutView[4, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.col_major(d0, d1, d2, d3), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0, "AOT MAX F-order extents must be positive"
    comptime MAXLayout = Layout.col_major(d0, d1, d2, d3)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            proof.status
        )
    var shape = proof.payload.value().shape
    if shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var span_result = mut_as_f_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            span_result.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](span_result.payload.value())
    )

def try_as_mut_static_f_tile_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, o: Origin[mut=True]](
    mut view: TensorMutView[4, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3)], o]]:
    comptime TileLayout = LTToTTLayout[Layout.col_major(d0, d1, d2, d3)]
    var proof = try_as_mut_static_f_layout_tensor_4d[d0, d1, d2, d3](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(proof.payload.take())
    )

def try_as_static_f_layout_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, o: Origin](
    view: TensorView[5, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.col_major(d0, d1, d2, d3, d4), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0, "AOT MAX F-order extents must be positive"
    comptime MAXLayout = Layout.col_major(d0, d1, d2, d3, d4)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            proof.status
        )
    var shape = proof.payload.value().shape
    if shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var span_result = as_f_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            span_result.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](span_result.payload.value())
    )

def try_as_static_f_tile_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, o: Origin](
    view: TensorView[5, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4)], o]]:
    comptime TileLayout = LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4)]
    var proof = try_as_static_f_layout_tensor_5d[d0, d1, d2, d3, d4](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(proof.payload.take())
    )

def try_as_mut_static_f_layout_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, o: Origin[mut=True]](
    mut view: TensorMutView[5, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.col_major(d0, d1, d2, d3, d4), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0, "AOT MAX F-order extents must be positive"
    comptime MAXLayout = Layout.col_major(d0, d1, d2, d3, d4)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            proof.status
        )
    var shape = proof.payload.value().shape
    if shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var span_result = mut_as_f_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            span_result.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](span_result.payload.value())
    )

def try_as_mut_static_f_tile_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, o: Origin[mut=True]](
    mut view: TensorMutView[5, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4)], o]]:
    comptime TileLayout = LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4)]
    var proof = try_as_mut_static_f_layout_tensor_5d[d0, d1, d2, d3, d4](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(proof.payload.take())
    )

def try_as_static_f_layout_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, o: Origin](
    view: TensorView[6, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.col_major(d0, d1, d2, d3, d4, d5), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0, "AOT MAX F-order extents must be positive"
    comptime MAXLayout = Layout.col_major(d0, d1, d2, d3, d4, d5)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            proof.status
        )
    var shape = proof.payload.value().shape
    if shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(5) != d5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var span_result = as_f_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            span_result.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](span_result.payload.value())
    )

def try_as_static_f_tile_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, o: Origin](
    view: TensorView[6, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5)], o]]:
    comptime TileLayout = LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5)]
    var proof = try_as_static_f_layout_tensor_6d[d0, d1, d2, d3, d4, d5](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(proof.payload.take())
    )

def try_as_mut_static_f_layout_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, o: Origin[mut=True]](
    mut view: TensorMutView[6, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.col_major(d0, d1, d2, d3, d4, d5), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0, "AOT MAX F-order extents must be positive"
    comptime MAXLayout = Layout.col_major(d0, d1, d2, d3, d4, d5)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            proof.status
        )
    var shape = proof.payload.value().shape
    if shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(5) != d5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var span_result = mut_as_f_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            span_result.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](span_result.payload.value())
    )

def try_as_mut_static_f_tile_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, o: Origin[mut=True]](
    mut view: TensorMutView[6, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5)], o]]:
    comptime TileLayout = LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5)]
    var proof = try_as_mut_static_f_layout_tensor_6d[d0, d1, d2, d3, d4, d5](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(proof.payload.take())
    )

def try_as_static_f_layout_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, o: Origin](
    view: TensorView[7, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.col_major(d0, d1, d2, d3, d4, d5, d6), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0, "AOT MAX F-order extents must be positive"
    comptime MAXLayout = Layout.col_major(d0, d1, d2, d3, d4, d5, d6)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            proof.status
        )
    var shape = proof.payload.value().shape
    if shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(5) != d5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(6) != d6:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var span_result = as_f_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            span_result.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](span_result.payload.value())
    )

def try_as_static_f_tile_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, o: Origin](
    view: TensorView[7, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6)], o]]:
    comptime TileLayout = LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6)]
    var proof = try_as_static_f_layout_tensor_7d[d0, d1, d2, d3, d4, d5, d6](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(proof.payload.take())
    )

def try_as_mut_static_f_layout_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, o: Origin[mut=True]](
    mut view: TensorMutView[7, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.col_major(d0, d1, d2, d3, d4, d5, d6), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0, "AOT MAX F-order extents must be positive"
    comptime MAXLayout = Layout.col_major(d0, d1, d2, d3, d4, d5, d6)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            proof.status
        )
    var shape = proof.payload.value().shape
    if shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(5) != d5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(6) != d6:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var span_result = mut_as_f_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            span_result.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](span_result.payload.value())
    )

def try_as_mut_static_f_tile_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, o: Origin[mut=True]](
    mut view: TensorMutView[7, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6)], o]]:
    comptime TileLayout = LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6)]
    var proof = try_as_mut_static_f_layout_tensor_7d[d0, d1, d2, d3, d4, d5, d6](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(proof.payload.take())
    )

def try_as_static_f_layout_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, o: Origin](
    view: TensorView[8, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.col_major(d0, d1, d2, d3, d4, d5, d6, d7), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0 and d7 > 0, "AOT MAX F-order extents must be positive"
    comptime MAXLayout = Layout.col_major(d0, d1, d2, d3, d4, d5, d6, d7)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            proof.status
        )
    var shape = proof.payload.value().shape
    if shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(5) != d5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(6) != d6:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(7) != d7:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var span_result = as_f_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            span_result.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](span_result.payload.value())
    )

def try_as_static_f_tile_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, o: Origin](
    view: TensorView[8, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6, d7)], o]]:
    comptime TileLayout = LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6, d7)]
    var proof = try_as_static_f_layout_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(proof.payload.take())
    )

def try_as_mut_static_f_layout_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, o: Origin[mut=True]](
    mut view: TensorMutView[8, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout.col_major(d0, d1, d2, d3, d4, d5, d6, d7), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0 and d7 > 0, "AOT MAX F-order extents must be positive"
    comptime MAXLayout = Layout.col_major(d0, d1, d2, d3, d4, d5, d6, d7)
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            proof.status
        )
    var shape = proof.payload.value().shape
    if shape.dims.get(0) != d0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(1) != d1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(2) != d2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(3) != d3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(4) != d4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(5) != d5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(6) != d6:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    if shape.dims.get(7) != d7:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            TensorStatus.NOT_REPRESENTABLE
        )
    var span_result = mut_as_f_span(view)
    if not span_result.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
            span_result.status
        )
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](span_result.payload.value())
    )

def try_as_mut_static_f_tile_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, o: Origin[mut=True]](
    mut view: TensorMutView[8, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6, d7)], o]]:
    comptime TileLayout = LTToTTLayout[Layout.col_major(d0, d1, d2, d3, d4, d5, d6, d7)]
    var proof = try_as_mut_static_f_layout_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7](view)
    if not proof.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](
            proof.status
        )
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](
        lt_to_tt(proof.payload.take())
    )


# ============================================================================
# General AOT-constant, positive-stride FP32 zero-copy bridges for ranks 3..8.
# ============================================================================
# Exact source shape/strides and bounded physical envelope are mandatory.
# Read-only LayoutTensor also permits zero strides; mutable bridges require
# an injective owner-validated mapping. TileTensor accepts only positive
# strides and never materializes a new allocation or changes ownership.
# These are specialization-time adapters, not a runtime-dynamic-rank family.

def try_as_static_strided_layout_tensor_3d[d0: Int, d1: Int, d2: Int, s0: Int, s1: Int, s2: Int, o: Origin](
    view: TensorView[3, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2)), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0, "static MAX dimensions must be positive"
    comptime assert s0 >= 0 and s1 >= 0 and s2 >= 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if mapping.shape.dims.get(0) != d0 or mapping.strides.get(0) != s0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1 or mapping.strides.get(1) != s1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2 or mapping.strides.get(2) != s2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    # With nonnegative strides, span_lo is the logical origin.
    # Preserve gaps: MAX receives the complete physical envelope.
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )

def try_as_mut_static_strided_layout_tensor_3d[d0: Int, d1: Int, d2: Int, s0: Int, s1: Int, s2: Int, o: Origin[mut=True]](
    mut view: TensorMutView[3, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2)), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0, "static MAX dimensions must be positive"
    comptime assert s0 >= 0 and s1 >= 0 and s2 >= 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if not mapping.is_proven_unique():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OVERLAPPING)
    if mapping.shape.dims.get(0) != d0 or mapping.strides.get(0) != s0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1 or mapping.strides.get(1) != s1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2 or mapping.strides.get(2) != s2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    # With nonnegative strides, span_lo is the logical origin.
    # Preserve gaps: MAX receives the complete physical envelope.
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )

def try_as_static_strided_tile_tensor_3d[d0: Int, d1: Int, d2: Int, s0: Int, s1: Int, s2: Int, o: Origin](
    view: TensorView[3, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2))], o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0, "static MAX dimensions must be positive"
    comptime assert s0 > 0 and s1 > 0 and s2 > 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_static_strided_layout_tensor_3d[d0, d1, d2, s0, s1, s2](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(lt.payload.take()))

def try_as_mut_static_strided_tile_tensor_3d[d0: Int, d1: Int, d2: Int, s0: Int, s1: Int, s2: Int, o: Origin[mut=True]](
    mut view: TensorMutView[3, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2))], o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0, "static MAX dimensions must be positive"
    comptime assert s0 > 0 and s1 > 0 and s2 > 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2), IntTuple(s0, s1, s2))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_mut_static_strided_layout_tensor_3d[d0, d1, d2, s0, s1, s2](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(lt.payload.take()))

def try_as_static_strided_layout_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, s0: Int, s1: Int, s2: Int, s3: Int, o: Origin](
    view: TensorView[4, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3)), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0, "static MAX dimensions must be positive"
    comptime assert s0 >= 0 and s1 >= 0 and s2 >= 0 and s3 >= 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if mapping.shape.dims.get(0) != d0 or mapping.strides.get(0) != s0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1 or mapping.strides.get(1) != s1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2 or mapping.strides.get(2) != s2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3 or mapping.strides.get(3) != s3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    # With nonnegative strides, span_lo is the logical origin.
    # Preserve gaps: MAX receives the complete physical envelope.
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )

def try_as_mut_static_strided_layout_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, s0: Int, s1: Int, s2: Int, s3: Int, o: Origin[mut=True]](
    mut view: TensorMutView[4, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3)), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0, "static MAX dimensions must be positive"
    comptime assert s0 >= 0 and s1 >= 0 and s2 >= 0 and s3 >= 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if not mapping.is_proven_unique():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OVERLAPPING)
    if mapping.shape.dims.get(0) != d0 or mapping.strides.get(0) != s0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1 or mapping.strides.get(1) != s1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2 or mapping.strides.get(2) != s2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3 or mapping.strides.get(3) != s3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    # With nonnegative strides, span_lo is the logical origin.
    # Preserve gaps: MAX receives the complete physical envelope.
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )

def try_as_static_strided_tile_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, s0: Int, s1: Int, s2: Int, s3: Int, o: Origin](
    view: TensorView[4, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3))], o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0, "static MAX dimensions must be positive"
    comptime assert s0 > 0 and s1 > 0 and s2 > 0 and s3 > 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_static_strided_layout_tensor_4d[d0, d1, d2, d3, s0, s1, s2, s3](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(lt.payload.take()))

def try_as_mut_static_strided_tile_tensor_4d[d0: Int, d1: Int, d2: Int, d3: Int, s0: Int, s1: Int, s2: Int, s3: Int, o: Origin[mut=True]](
    mut view: TensorMutView[4, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3))], o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0, "static MAX dimensions must be positive"
    comptime assert s0 > 0 and s1 > 0 and s2 > 0 and s3 > 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3), IntTuple(s0, s1, s2, s3))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_mut_static_strided_layout_tensor_4d[d0, d1, d2, d3, s0, s1, s2, s3](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(lt.payload.take()))

def try_as_static_strided_layout_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, o: Origin](
    view: TensorView[5, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4)), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0, "static MAX dimensions must be positive"
    comptime assert s0 >= 0 and s1 >= 0 and s2 >= 0 and s3 >= 0 and s4 >= 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if mapping.shape.dims.get(0) != d0 or mapping.strides.get(0) != s0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1 or mapping.strides.get(1) != s1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2 or mapping.strides.get(2) != s2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3 or mapping.strides.get(3) != s3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4 or mapping.strides.get(4) != s4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    # With nonnegative strides, span_lo is the logical origin.
    # Preserve gaps: MAX receives the complete physical envelope.
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )

def try_as_mut_static_strided_layout_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, o: Origin[mut=True]](
    mut view: TensorMutView[5, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4)), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0, "static MAX dimensions must be positive"
    comptime assert s0 >= 0 and s1 >= 0 and s2 >= 0 and s3 >= 0 and s4 >= 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if not mapping.is_proven_unique():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OVERLAPPING)
    if mapping.shape.dims.get(0) != d0 or mapping.strides.get(0) != s0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1 or mapping.strides.get(1) != s1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2 or mapping.strides.get(2) != s2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3 or mapping.strides.get(3) != s3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4 or mapping.strides.get(4) != s4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    # With nonnegative strides, span_lo is the logical origin.
    # Preserve gaps: MAX receives the complete physical envelope.
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )

def try_as_static_strided_tile_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, o: Origin](
    view: TensorView[5, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4))], o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0, "static MAX dimensions must be positive"
    comptime assert s0 > 0 and s1 > 0 and s2 > 0 and s3 > 0 and s4 > 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_static_strided_layout_tensor_5d[d0, d1, d2, d3, d4, s0, s1, s2, s3, s4](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(lt.payload.take()))

def try_as_mut_static_strided_tile_tensor_5d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, o: Origin[mut=True]](
    mut view: TensorMutView[5, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4))], o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0, "static MAX dimensions must be positive"
    comptime assert s0 > 0 and s1 > 0 and s2 > 0 and s3 > 0 and s4 > 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4), IntTuple(s0, s1, s2, s3, s4))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_mut_static_strided_layout_tensor_5d[d0, d1, d2, d3, d4, s0, s1, s2, s3, s4](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(lt.payload.take()))

def try_as_static_strided_layout_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, o: Origin](
    view: TensorView[6, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5)), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0, "static MAX dimensions must be positive"
    comptime assert s0 >= 0 and s1 >= 0 and s2 >= 0 and s3 >= 0 and s4 >= 0 and s5 >= 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if mapping.shape.dims.get(0) != d0 or mapping.strides.get(0) != s0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1 or mapping.strides.get(1) != s1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2 or mapping.strides.get(2) != s2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3 or mapping.strides.get(3) != s3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4 or mapping.strides.get(4) != s4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(5) != d5 or mapping.strides.get(5) != s5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    # With nonnegative strides, span_lo is the logical origin.
    # Preserve gaps: MAX receives the complete physical envelope.
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )

def try_as_mut_static_strided_layout_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, o: Origin[mut=True]](
    mut view: TensorMutView[6, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5)), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0, "static MAX dimensions must be positive"
    comptime assert s0 >= 0 and s1 >= 0 and s2 >= 0 and s3 >= 0 and s4 >= 0 and s5 >= 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if not mapping.is_proven_unique():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OVERLAPPING)
    if mapping.shape.dims.get(0) != d0 or mapping.strides.get(0) != s0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1 or mapping.strides.get(1) != s1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2 or mapping.strides.get(2) != s2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3 or mapping.strides.get(3) != s3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4 or mapping.strides.get(4) != s4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(5) != d5 or mapping.strides.get(5) != s5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    # With nonnegative strides, span_lo is the logical origin.
    # Preserve gaps: MAX receives the complete physical envelope.
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )

def try_as_static_strided_tile_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, o: Origin](
    view: TensorView[6, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5))], o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0, "static MAX dimensions must be positive"
    comptime assert s0 > 0 and s1 > 0 and s2 > 0 and s3 > 0 and s4 > 0 and s5 > 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_static_strided_layout_tensor_6d[d0, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(lt.payload.take()))

def try_as_mut_static_strided_tile_tensor_6d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, o: Origin[mut=True]](
    mut view: TensorMutView[6, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5))], o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0, "static MAX dimensions must be positive"
    comptime assert s0 > 0 and s1 > 0 and s2 > 0 and s3 > 0 and s4 > 0 and s5 > 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4, d5), IntTuple(s0, s1, s2, s3, s4, s5))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_mut_static_strided_layout_tensor_6d[d0, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(lt.payload.take()))

def try_as_static_strided_layout_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int, o: Origin](
    view: TensorView[7, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6)), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0, "static MAX dimensions must be positive"
    comptime assert s0 >= 0 and s1 >= 0 and s2 >= 0 and s3 >= 0 and s4 >= 0 and s5 >= 0 and s6 >= 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if mapping.shape.dims.get(0) != d0 or mapping.strides.get(0) != s0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1 or mapping.strides.get(1) != s1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2 or mapping.strides.get(2) != s2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3 or mapping.strides.get(3) != s3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4 or mapping.strides.get(4) != s4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(5) != d5 or mapping.strides.get(5) != s5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(6) != d6 or mapping.strides.get(6) != s6:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    # With nonnegative strides, span_lo is the logical origin.
    # Preserve gaps: MAX receives the complete physical envelope.
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )

def try_as_mut_static_strided_layout_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int, o: Origin[mut=True]](
    mut view: TensorMutView[7, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6)), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0, "static MAX dimensions must be positive"
    comptime assert s0 >= 0 and s1 >= 0 and s2 >= 0 and s3 >= 0 and s4 >= 0 and s5 >= 0 and s6 >= 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if not mapping.is_proven_unique():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OVERLAPPING)
    if mapping.shape.dims.get(0) != d0 or mapping.strides.get(0) != s0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1 or mapping.strides.get(1) != s1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2 or mapping.strides.get(2) != s2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3 or mapping.strides.get(3) != s3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4 or mapping.strides.get(4) != s4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(5) != d5 or mapping.strides.get(5) != s5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(6) != d6 or mapping.strides.get(6) != s6:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    # With nonnegative strides, span_lo is the logical origin.
    # Preserve gaps: MAX receives the complete physical envelope.
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )

def try_as_static_strided_tile_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int, o: Origin](
    view: TensorView[7, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6))], o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0, "static MAX dimensions must be positive"
    comptime assert s0 > 0 and s1 > 0 and s2 > 0 and s3 > 0 and s4 > 0 and s5 > 0 and s6 > 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_static_strided_layout_tensor_7d[d0, d1, d2, d3, d4, d5, d6, s0, s1, s2, s3, s4, s5, s6](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(lt.payload.take()))

def try_as_mut_static_strided_tile_tensor_7d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int, o: Origin[mut=True]](
    mut view: TensorMutView[7, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6))], o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0, "static MAX dimensions must be positive"
    comptime assert s0 > 0 and s1 > 0 and s2 > 0 and s3 > 0 and s4 > 0 and s5 > 0 and s6 > 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6), IntTuple(s0, s1, s2, s3, s4, s5, s6))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_mut_static_strided_layout_tensor_7d[d0, d1, d2, d3, d4, d5, d6, s0, s1, s2, s3, s4, s5, s6](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(lt.payload.take()))

def try_as_static_strided_layout_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int, s7: Int, o: Origin](
    view: TensorView[8, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7)), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0 and d7 > 0, "static MAX dimensions must be positive"
    comptime assert s0 >= 0 and s1 >= 0 and s2 >= 0 and s3 >= 0 and s4 >= 0 and s5 >= 0 and s6 >= 0 and s7 >= 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if mapping.shape.dims.get(0) != d0 or mapping.strides.get(0) != s0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1 or mapping.strides.get(1) != s1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2 or mapping.strides.get(2) != s2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3 or mapping.strides.get(3) != s3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4 or mapping.strides.get(4) != s4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(5) != d5 or mapping.strides.get(5) != s5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(6) != d6 or mapping.strides.get(6) != s6:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(7) != d7 or mapping.strides.get(7) != s7:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    # With nonnegative strides, span_lo is the logical origin.
    # Preserve gaps: MAX receives the complete physical envelope.
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )

def try_as_mut_static_strided_layout_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int, s7: Int, o: Origin[mut=True]](
    mut view: TensorMutView[8, o],
) -> TensorResult[LayoutTensor[DType.float32, Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7)), o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0 and d7 > 0, "static MAX dimensions must be positive"
    comptime assert s0 >= 0 and s1 >= 0 and s2 >= 0 and s3 >= 0 and s4 >= 0 and s5 >= 0 and s6 >= 0 and s7 >= 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7))
    var proof = view._checked_current_layout()
    if not proof.is_ok():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](proof.status)
    var mapping = proof.payload.value()
    if not mapping.is_proven_unique():
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OVERLAPPING)
    if mapping.shape.dims.get(0) != d0 or mapping.strides.get(0) != s0:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(1) != d1 or mapping.strides.get(1) != s1:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(2) != d2 or mapping.strides.get(2) != s2:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(3) != d3 or mapping.strides.get(3) != s3:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(4) != d4 or mapping.strides.get(4) != s4:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(5) != d5 or mapping.strides.get(5) != s5:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(6) != d6 or mapping.strides.get(6) != s6:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    if mapping.shape.dims.get(7) != d7 or mapping.strides.get(7) != s7:
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.NOT_REPRESENTABLE)
    # With nonnegative strides, span_lo is the logical origin.
    # Preserve gaps: MAX receives the complete physical envelope.
    var low = mapping.span_lo
    var high = mapping.span_hi
    if low < 0 or high < low or high >= len(view.data):
        return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](TensorStatus.OUT_OF_BOUNDS)
    var borrowed = view.data[low:high + 1]
    return TensorResult[LayoutTensor[DType.float32, MAXLayout, o]](
        LayoutTensor[DType.float32, MAXLayout, o](borrowed)
    )

def try_as_static_strided_tile_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int, s7: Int, o: Origin](
    view: TensorView[8, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7))], o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0 and d7 > 0, "static MAX dimensions must be positive"
    comptime assert s0 > 0 and s1 > 0 and s2 > 0 and s3 > 0 and s4 > 0 and s5 > 0 and s6 > 0 and s7 > 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_static_strided_layout_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6, s7](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(lt.payload.take()))

def try_as_mut_static_strided_tile_tensor_8d[d0: Int, d1: Int, d2: Int, d3: Int, d4: Int, d5: Int, d6: Int, d7: Int, s0: Int, s1: Int, s2: Int, s3: Int, s4: Int, s5: Int, s6: Int, s7: Int, o: Origin[mut=True]](
    mut view: TensorMutView[8, o],
) -> TensorResult[TileTensor[DType.float32, LTToTTLayout[Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7))], o]]:
    comptime assert d0 > 0 and d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0 and d5 > 0 and d6 > 0 and d7 > 0, "static MAX dimensions must be positive"
    comptime assert s0 > 0 and s1 > 0 and s2 > 0 and s3 > 0 and s4 > 0 and s5 > 0 and s6 > 0 and s7 > 0, "unsupported static MAX stride"
    comptime MAXLayout = Layout(IntTuple(d0, d1, d2, d3, d4, d5, d6, d7), IntTuple(s0, s1, s2, s3, s4, s5, s6, s7))
    comptime TileLayout = LTToTTLayout[MAXLayout]
    var lt = try_as_mut_static_strided_layout_tensor_8d[d0, d1, d2, d3, d4, d5, d6, d7, s0, s1, s2, s3, s4, s5, s6, s7](view)
    if not lt.is_ok():
        return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt.status)
    return TensorResult[TileTensor[DType.float32, TileLayout, o]](lt_to_tt(lt.payload.take()))
