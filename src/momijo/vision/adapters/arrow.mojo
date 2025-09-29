# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision.adapters
# File: momijo/vision/adapters/arrow.mojo
  
from momijo.vision.tensor import Tensor, packed_hwc_strides, packed_chw_strides
from momijo.vision.dtypes import DType, dtype_bytes
 

# -----------------------------------------------------------------------------
# Core unsafe view: wraps an existing raw buffer as a Tensor without copying.
# Caller must ensure that:
#   - 'ptr' stays alive for the lifetime of the returned Tensor.
#   - 'nbytes' covers the accessible range for the provided shape/strides/dtype.
#   - 's0/s1/s2' are logical strides in ELEMENTS (not bytes).
# -----------------------------------------------------------------------------
fn unsafe_view_from_raw_u8(ptr: UnsafePointer[UInt8], nbytes: Int,
                           h: Int, w: Int, c: Int,
                           s0: Int, s1: Int, s2: Int, dt: DType) -> Tensor:
    # Quick sanity checks (assert-based; remove if you prefer pure unsafe):
    assert(h > 0 and w > 0 and c > 0, "unsafe_view_from_raw_u8: bad shape")
    assert(nbytes > 0, "unsafe_view_from_raw_u8: nbytes must be > 0")
    assert(s0 > 0 and s1 > 0 and s2 > 0, "unsafe_view_from_raw_u8: bad strides")

    # NOTE: We do not validate that s0/s1/s2 match a packed layout; they are trusted as-is.
    return Tensor(ptr, nbytes, h, w, c, s0, s1, s2, dt)

# -----------------------------------------------------------------------------
# HWC helpers (UInt8)
# -----------------------------------------------------------------------------
fn tensor_from_u8_array_hwc(data: UnsafePointer[UInt8], nbytes: Int, h: Int, w: Int, c: Int) -> Tensor:
    # If a raw pointer is already available, return a zero-copy HWC view.
    assert(h > 0 and w > 0 and c > 0, "tensor_from_u8_array_hwc: bad shape")
    var (s0, s1, s2) = packed_hwc_strides(h, w, c)
    # For UInt8 packed HWC, minimal byte footprint is h*w*c.
    var min_bytes = h * w * c
    assert(nbytes >= min_bytes, "tensor_from_u8_array_hwc: nbytes too small for HWC u8")
    return Tensor(data, nbytes, h, w, c, s0, s1, s2, DType.UInt8)

fn array_from_tensor_u8_hwc(t: Tensor) -> UnsafePointer[UInt8]:
    # Returns the underlying pointer (no copy). Caller responsible for size = t.byte_len().
    # This is intended for exporting a packed HWC u8 tensor back to a byte-level consumer.
    return t.data()

# -----------------------------------------------------------------------------
# CHW helpers (general dtype) — useful if your upstream is channel-major
# -----------------------------------------------------------------------------
fn tensor_from_raw_chw(ptr: UnsafePointer[UInt8], nbytes: Int,
                       c: Int, h: Int, w: Int, dt: DType) -> Tensor:
    assert(c > 0 and h > 0 and w > 0, "tensor_from_raw_chw: bad shape")
    var (s0, s1, s2) = packed_chw_strides(c, h, w)
    # Minimal bytes if packed:
    var elems = c * h * w
    var need = elems * dtype_bytes(dt)
    assert(nbytes >= need, "tensor_from_raw_chw: nbytes too small for packed CHW")
    # Note: We store shape as (dim0, dim1, dim2) = (C, H, W) for this logical view.
    return Tensor(ptr, nbytes, c, h, w, s0, s1, s2, dt)

# -----------------------------------------------------------------------------
# Generic helper that accepts strides in ELEMENTS for arbitrary layout
# -----------------------------------------------------------------------------
fn tensor_from_raw_strided(ptr: UnsafePointer[UInt8],
                           nbytes: Int,
                           d0: Int, d1: Int, d2: Int,
                           s0: Int, s1: Int, s2: Int,
                           dt: DType) -> Tensor:
    assert(d0 > 0 and d1 > 0 and d2 > 0, "tensor_from_raw_strided: bad shape")
    assert(s0 > 0 and s1 > 0 and s2 > 0, "tensor_from_raw_strided: bad strides")
    assert(nbytes > 0, "tensor_from_raw_strided: nbytes must be > 0")
    # We trust caller on stride correctness & coverage; pack into a Tensor as-is.
    return Tensor(ptr, nbytes, d0, d1, d2, s0, s1, s2, dt)

 