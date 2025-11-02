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
# Project: momijo.vision.transforms
# File: momijo/vision/transforms/resize.mojo
 
 

from momijo.vision.tensor import Tensor, packed_hwc_strides
from momijo.vision.dtypes import DType 
from momijo.vision.transforms.resize import resize_nearest_u8_hwc

# ---------------------------------------------------------------------------
# Simple RNG (LCG) for deterministic randint
# ---------------------------------------------------------------------------
@fieldwise_init("implicit")
struct _LCG:
    var _state: UInt64

    fn __init__(out self, seed: UInt64):
        self._state = seed

    fn rand01(mut self) -> Float32:
        # 64-bit LCG → take upper bits → [0,1)
        self._state = self._state * 6364136223846793005 + 1
        return Float32((self._state >> 33) & 0xFFFFFFFF) / 4294967296.0

    fn randint(mut self, lo: Int, hi_exclusive: Int) -> Int:
        # Returns integer in [lo, hi_exclusive)
        assert(hi_exclusive > lo, "randint: invalid range")
        var span = hi_exclusive - lo
        var r = self.rand01()
        var v = Int(r * Float32(span))
        if v >= span: v = span - 1
        return lo + v

# ---------------------------------------------------------------------------
# Crop helper (returns packed HWC/u8)
# ---------------------------------------------------------------------------
fn _crop_hwc_u8(src: Tensor, top: Int, left: Int, crop_h: Int, crop_w: Int) -> Tensor:
    assert(src.dtype() == DType.UInt8, "_crop_hwc_u8: only UInt8 supported")
    var h = src.height()
    var w = src.width()
    var c = src.channels()
    assert(top >= 0 and left >= 0 and crop_h > 0 and crop_w > 0, "_crop_hwc_u8: bad args")
    assert(top + crop_h <= h and left + crop_w <= w, "_crop_hwc_u8: ROI out of bounds")

    var s0 = src.stride0()
    var s1 = src.stride1()
    var s2 = src.stride2()
    var sp = src.data()

    var (d0, d1, d2) = packed_hwc_strides(crop_h, crop_w, c)
    var out_len = crop_h * crop_w * c
    var out_buf = UnsafePointer[UInt8].alloc(out_len)
    var out = Tensor(out_buf, out_len, crop_h, crop_w, c, d0, d1, d2, DType.UInt8)

    var y:Int = 0
    while y < crop_h:
        var src_y = top + y
        var x:Int = 0
        while x < crop_w:
            var src_x = left + x
            var base_src = src_y * s0 + src_x * s1
            var base_dst = y * d0 + x * d1
            var ch:Int = 0
            while ch < c:
                out_buf[base_dst + ch * d2] = sp[base_src + ch * s2]
                ch += 1
            x += 1
        y += 1
    return out.copy()

# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------
@fieldwise_init
struct RandomResizedCrop:
    var out_h: Int
    var out_w: Int
    var _rng: _LCG

    fn __init__(out self, out_h: Int, out_w: Int, seed: UInt64 = 0x9E3779B97F4A7C15):
        self.out_h = out_h
        self.out_w = out_w
        self._rng = _LCG(seed)

    # Apply to an HWC/u8 tensor; returns a NEW packed HWC/u8 tensor
    fn __call__(mut self, x_in: Tensor) -> Tensor:
        assert(x_in.dtype() == DType.UInt8, "RandomResizedCrop: only UInt8 supported")
        var h = x_in.height()
        var w = x_in.width()
        var c = x_in.channels()
        assert(h > 0 and w > 0 and c > 0, "RandomResizedCrop: bad input shape")

        var x = x_in
        # Fast path: if input is large enough, random-crop exactly out_h × out_w
        if h >= self.out_h and w >= self.out_w:
            var max_top  = h - self.out_h + 1
            var max_left = w - self.out_w + 1
            var top  = self._rng.randint(0, max_top)
            var left = self._rng.randint(0, max_left)
            return _crop_hwc_u8(x, top, left, self.out_h, self.out_w)

        # Fallback: if smaller, just resize to target (no randomness)
        return resize_nearest_u8_hwc(x, self.out_h, self.out_w)

# Convenience functional wrapper
fn random_resized_crop(x: Tensor, out_h: Int, out_w: Int, seed: UInt64 = 0x9E3779B97F4A7C15) -> Tensor:
    var op = RandomResizedCrop(out_h, out_w, seed)
    return op(x)


# ----------------------------
# Core: nearest neighbor (HWC/u8)
# ----------------------------
fn resize_nearest_u8_hwc(src: Tensor, out_h: Int, out_w: Int) -> Tensor:
    # Preconditions
    assert(src.dtype() == DType.UInt8, "resize_nearest_u8_hwc: only UInt8 supported")
    var h = src.height()
    var w = src.width()
    var c = src.channels()
    assert(h > 0 and w > 0 and c > 0, "resize_nearest_u8_hwc: bad src shape")
    assert(out_h > 0 and out_w > 0, "resize_nearest_u8_hwc: bad output shape")

    # Prepare output (packed HWC)
    var (s0_out, s1_out, s2_out) = packed_hwc_strides(out_h, out_w, c)
    var out_len = out_h * out_w * c
    var out_buf = UnsafePointer[UInt8].alloc(out_len)
    var out = Tensor(out_buf, out_len, out_h, out_w, c, s0_out, s1_out, s2_out, DType.UInt8)

    # Source strides/pointer
    var s0 = src.stride0()
    var s1 = src.stride1()
    var s2 = src.stride2()
    var sp = src.data()

    # Scale factors
    var y_scale = Float64(h) / Float64(out_h)
    var x_scale = Float64(w) / Float64(out_w)

    # Main loop
    var y:Int = 0
    while y < out_h:
        var src_y = _clamp_i(Int(Float64(y) * y_scale), 0, h - 1)
        var x:Int = 0
        while x < out_w:
            var src_x = _clamp_i(Int(Float64(x) * x_scale), 0, w - 1)
            var base_src = src_y * s0 + src_x * s1
            var ch:Int = 0
            while ch < c:
                out_buf[y * s0_out + x * s1_out + ch * s2_out] =
                    sp[base_src + ch * s2]
                ch += 1
            x += 1
        y += 1

    return out.copy()

# ----------------------------
# Generic facade (dtype/layout aware)
# For now only HWC/u8 is implemented.
# ----------------------------
fn resize_nearest(src: Tensor, out_h: Int, out_w: Int) -> Tensor:
    # Extend here for other dtypes/layouts when added.
    assert(src.dtype() == DType.UInt8, "resize_nearest: only UInt8 supported currently")
    return resize_nearest_u8_hwc(src, out_h, out_w)

# ----------------------------
# Backward-compat shim for old 4-arg signature
# (Ignores out_c but validates consistency if provided.)
# ----------------------------
fn resize_nearest_u8_hwc(src: Tensor, out_h: Int, out_w: Int, out_c: Int) -> Tensor:
    # Keep behavior identical to 3-arg version; if channel count matches
    if out_c != src.channels():
        # We don't support channel change here—preserve source C.
        assert(False, "resize_nearest_u8_hwc: changing channels is not supported")
    return resize_nearest_u8_hwc(src, out_h, out_w)
