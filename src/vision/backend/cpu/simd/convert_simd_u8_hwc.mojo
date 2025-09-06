# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision.backend.cpu.simd
# File: src/momijo/vision/backend/cpu/simd/convert_simd_u8_hwc.mojo
#
# "SIMD-friendly" color conversion kernels for uint8 HWC images.
# To remain portable and dependency-light, we implement chunked/unrolled loops
# which are amenable to auto-vectorization by the compiler. No external SIMD types used.
# Style: no 'export', no 'let', no 'inout'.
#
# Implemented ops:
#   - simd_bgr_to_rgb_u8_hwc(h,w, src) -> List[UInt8]
#   - simd_rgb_to_gray_u8_hwc(h,w, src) -> List[UInt8]  (integer luma 77/150/29)
#   - simd_rgba_to_rgb_u8_hwc(h,w, src) -> List[UInt8]  (drop alpha)
#   - simd_bgra_to_bgr_u8_hwc(h,w, src) -> List[UInt8]  (drop alpha)
#   - simd_argmin_channel_u8_hwc(h,w, src) -> List[UInt8] (per-pixel argmin over 3 chans)
#
# Notes:
# - src is flat HWC (UInt8) with channels = 3 (RGB/BGR) or 4 (RGBA/BGRA) as appropriate.
# - Output is a new buffer; functions are alias-safe.

# -------------------------
# Helpers
# -------------------------
@staticmethod
fn _offset(w: Int, c: Int, x: Int, y: Int, ch: Int) -> Int:
    return ((y * w) + x) * c + ch

@staticmethod
fn _alloc_u8(n: Int) -> List[UInt8]:
    var out: List[UInt8] = List[UInt8]()
    var i = 0
    while i < n:
        out.append(0)
        i += 1
    return out

# -------------------------
# BGR -> RGB (3 channels)
# -------------------------
@staticmethod
fn simd_bgr_to_rgb_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    if h <= 0 or w <= 0:
        return List[UInt8]()
    var out = _alloc_u8(h * w * 3)

    var y = 0
    while y < h:
        var x = 0
        # process 4 pixels per iteration to aid auto-vectorization
        while x + 3 < w:
            var base0 = _offset(w, 3, x + 0, y, 0)
            var base1 = base0 + 3
            var base2 = base1 + 3
            var base3 = base2 + 3
            # swap B<->R for each pixel
            out[base0 + 0] = src[base0 + 2]; out[base0 + 1] = src[base0 + 1]; out[base0 + 2] = src[base0 + 0]
            out[base1 + 0] = src[base1 + 2]; out[base1 + 1] = src[base1 + 1]; out[base1 + 2] = src[base1 + 0]
            out[base2 + 0] = src[base2 + 2]; out[base2 + 1] = src[base2 + 1]; out[base2 + 2] = src[base2 + 0]
            out[base3 + 0] = src[base3 + 2]; out[base3 + 1] = src[base3 + 1]; out[base3 + 2] = src[base3 + 0]
            x += 4
        # tail
        while x < w:
            var base = _offset(w, 3, x, y, 0)
            out[base + 0] = src[base + 2]
            out[base + 1] = src[base + 1]
            out[base + 2] = src[base + 0]
            x += 1
        y += 1
    return out

# -------------------------
# RGB -> GRAY (3 channels -> 1)
# -------------------------
@staticmethod
fn simd_rgb_to_gray_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    if h <= 0 or w <= 0:
        return List[UInt8]()
    var out = _alloc_u8(h * w)

    var y = 0
    while y < h:
        var x = 0
        while x + 3 < w:
            var b0 = _offset(w, 3, x + 0, y, 0)
            var b1 = b0 + 3
            var b2 = b1 + 3
            var b3 = b2 + 3

            var r0 = src[b0 + 0]; var g0 = src[b0 + 1]; var b_0 = src[b0 + 2]
            var r1 = src[b1 + 0]; var g1 = src[b1 + 1]; var b_1 = src[b1 + 2]
            var r2 = src[b2 + 0]; var g2 = src[b2 + 1]; var b_2 = src[b2 + 2]
            var r3 = src[b3 + 0]; var g3 = src[b3 + 1]; var b_3 = src[b3 + 2]

            var g0_u16 = UInt16(77) * UInt16(r0) + UInt16(150) * UInt16(g0) + UInt16(29) * UInt16(b_0)
            var g1_u16 = UInt16(77) * UInt16(r1) + UInt16(150) * UInt16(g1) + UInt16(29) * UInt16(b_1)
            var g2_u16 = UInt16(77) * UInt16(r2) + UInt16(150) * UInt16(g2) + UInt16(29) * UInt16(b_2)
            var g3_u16 = UInt16(77) * UInt16(r3) + UInt16(150) * UInt16(g3) + UInt16(29) * UInt16(b_3)

            out[y * w + (x + 0)] = UInt8((g0_u16 >> 8) & UInt16(0xFF))
            out[y * w + (x + 1)] = UInt8((g1_u16 >> 8) & UInt16(0xFF))
            out[y * w + (x + 2)] = UInt8((g2_u16 >> 8) & UInt16(0xFF))
            out[y * w + (x + 3)] = UInt8((g3_u16 >> 8) & UInt16(0xFF))

            x += 4
        # tail
        while x < w:
            var b = _offset(w, 3, x, y, 0)
            var r = src[b + 0]; var g = src[b + 1]; var bb = src[b + 2]
            var gy = UInt16(77) * UInt16(r) + UInt16(150) * UInt16(g) + UInt16(29) * UInt16(bb)
            out[y * w + x] = UInt8((gy >> 8) & UInt16(0xFF))
            x += 1
        y += 1
    return out

# -------------------------
# RGBA -> RGB (drop alpha)
# -------------------------
@staticmethod
fn simd_rgba_to_rgb_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    if h <= 0 or w <= 0:
        return List[UInt8]()
    var out = _alloc_u8(h * w * 3)
    var y = 0
    while y < h:
        var x = 0
        while x + 3 < w:
            var base0 = _offset(w, 4, x + 0, y, 0)
            var base1 = base0 + 4
            var base2 = base1 + 4
            var base3 = base2 + 4
            var d0 = _offset(w, 3, x + 0, y, 0)
            var d1 = d0 + 3
            var d2 = d1 + 3
            var d3 = d2 + 3
            out[d0 + 0] = src[base0 + 0]; out[d0 + 1] = src[base0 + 1]; out[d0 + 2] = src[base0 + 2]
            out[d1 + 0] = src[base1 + 0]; out[d1 + 1] = src[base1 + 1]; out[d1 + 2] = src[base1 + 2]
            out[d2 + 0] = src[base2 + 0]; out[d2 + 1] = src[base2 + 1]; out[d2 + 2] = src[base2 + 2]
            out[d3 + 0] = src[base3 + 0]; out[d3 + 1] = src[base3 + 1]; out[d3 + 2] = src[base3 + 2]
            x += 4
        while x < w:
            var s = _offset(w, 4, x, y, 0)
            var d = _offset(w, 3, x, y, 0)
            out[d + 0] = src[s + 0]
            out[d + 1] = src[s + 1]
            out[d + 2] = src[s + 2]
            x += 1
        y += 1
    return out

# -------------------------
# BGRA -> BGR (drop alpha)
# -------------------------
@staticmethod
fn simd_bgra_to_bgr_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    if h <= 0 or w <= 0:
        return List[UInt8]()
    var out = _alloc_u8(h * w * 3)
    var y = 0
    while y < h:
        var x = 0
        while x + 3 < w:
            var base0 = _offset(w, 4, x + 0, y, 0)
            var base1 = base0 + 4
            var base2 = base1 + 4
            var base3 = base2 + 4
            var d0 = _offset(w, 3, x + 0, y, 0)
            var d1 = d0 + 3
            var d2 = d1 + 3
            var d3 = d2 + 3
            out[d0 + 0] = src[base0 + 0]; out[d0 + 1] = src[base0 + 1]; out[d0 + 2] = src[base0 + 2]
            out[d1 + 0] = src[base1 + 0]; out[d1 + 1] = src[base1 + 1]; out[d1 + 2] = src[base1 + 2]
            out[d2 + 0] = src[base2 + 0]; out[d2 + 1] = src[base2 + 1]; out[d2 + 2] = src[base2 + 2]
            out[d3 + 0] = src[base3 + 0]; out[d3 + 1] = src[base3 + 1]; out[d3 + 2] = src[base3 + 2]
            x += 4
        while x < w:
            var s = _offset(w, 4, x, y, 0)
            var d = _offset(w, 3, x, y, 0)
            out[d + 0] = src[s + 0]
            out[d + 1] = src[s + 1]
            out[d + 2] = src[s + 2]
            x += 1
        y += 1
    return out

# -------------------------
# Per-pixel argmin over 3 channels
# -------------------------
@staticmethod
fn simd_argmin_channel_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    if h <= 0 or w <= 0:
        return List[UInt8]()
    var out = _alloc_u8(h * w)
    var y = 0
    while y < h:
        var x = 0
        while x + 3 < w:
            var b0 = _offset(w, 3, x + 0, y, 0)
            var b1 = b0 + 3
            var b2 = b1 + 3
            var b3 = b2 + 3
            var v0_0 = src[b0 + 0]; var v0_1 = src[b0 + 1]; var v0_2 = src[b0 + 2]
            var v1_0 = src[b1 + 0]; var v1_1 = src[b1 + 1]; var v1_2 = src[b1 + 2]
            var v2_0 = src[b2 + 0]; var v2_1 = src[b2 + 1]; var v2_2 = src[b2 + 2]
            var v3_0 = src[b3 + 0]; var v3_1 = src[b3 + 1]; var v3_2 = src[b3 + 2]

            var a0 = UInt8(0); var m0 = v0_0
            if v0_1 < m0: m0 = v0_1; a0 = UInt8(1)
            if v0_2 < m0: a0 = UInt8(2)
            out[y * w + (x + 0)] = a0

            var a1 = UInt8(0); var m1 = v1_0
            if v1_1 < m1: m1 = v1_1; a1 = UInt8(1)
            if v1_2 < m1: a1 = UInt8(2)
            out[y * w + (x + 1)] = a1

            var a2 = UInt8(0); var m2 = v2_0
            if v2_1 < m2: m2 = v2_1; a2 = UInt8(1)
            if v2_2 < m2: a2 = UInt8(2)
            out[y * w + (x + 2)] = a2

            var a3 = UInt8(0); var m3 = v3_0
            if v3_1 < m3: m3 = v3_1; a3 = UInt8(1)
            if v3_2 < m3: a3 = UInt8(2)
            out[y * w + (x + 3)] = a3

            x += 4
        while x < w:
            var b = _offset(w, 3, x, y, 0)
            var vv0 = src[b + 0]; var vv1 = src[b + 1]; var vv2 = src[b + 2]
            var arg = UInt8(0)
            var minv = vv0
            if vv1 < minv:
                minv = vv1; arg = UInt8(1)
            if vv2 < minv:
                arg = UInt8(2)
            out[y * w + x] = arg
            x += 1
        y += 1
    return out

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    var h = 2; var w = 2
    var rgb: List[UInt8] = List[UInt8]()
    rgb.append(255); rgb.append(0);   rgb.append(0)
    rgb.append(0);   rgb.append(255); rgb.append(0)
    rgb.append(0);   rgb.append(0);   rgb.append(255)
    rgb.append(255); rgb.append(255); rgb.append(255)

    var bgr = simd_bgr_to_rgb_u8_hwc(h, w, rgb)  # treat input as BGR
    if len(bgr) != 12: return False

    var gray = simd_rgb_to_gray_u8_hwc(h, w, rgb)
    if len(gray) != 4: return False

    var argm = simd_argmin_channel_u8_hwc(h, w, rgb)
    if len(argm) != 4: return False

    # RGBA drop alpha
    var rgba: List[UInt8] = List[UInt8]()
    rgba.append(1); rgba.append(2); rgba.append(3); rgba.append(4)
    rgba.append(5); rgba.append(6); rgba.append(7); rgba.append(8)
    rgba.append(9); rgba.append(10); rgba.append(11); rgba.append(12)
    rgba.append(13); rgba.append(14); rgba.append(15); rgba.append(16)
    var rgb_out = simd_rgba_to_rgb_u8_hwc(2, 2, rgba)
    if len(rgb_out) != 12: return False

    return True
