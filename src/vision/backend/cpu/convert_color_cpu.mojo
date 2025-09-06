# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision.backend.cpu
# File: src/momijo/vision/backend/cpu/convert_color_cpu.mojo
#
# CPU color conversion kernels for uint8 HWC images.
# Standalone & dependency-light by design (no external Momijo imports).
# Style: no 'export', no 'let', no 'inout'. Constructors use `fn __init__(out self, ...)`.
#
# Implemented ops:
#   - bgr_to_rgb_u8_hwc
#   - rgb_to_gray_u8_hwc  (integer-approx luma: 77/150/29)
#   - rgba_to_rgb_u8_hwc  (drop alpha)
#   - bgra_to_bgr_u8_hwc  (drop alpha)
#   - argmin_channel_u8_hwc (per-pixel argmin over channels)
#
# All functions return a new buffer to avoid aliasing surprises.

@staticmethod
fn _offset(w: Int, c: Int, x: Int, y: Int, ch: Int) -> Int:
    return ((y * w) + x) * c + ch

@staticmethod
fn bgr_to_rgb_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    var out: List[UInt8] = List[UInt8]()
    var total = h * w * 3
    var i = 0
    while i < total:
        out.append(0)
        i += 1
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var idx_b = _offset(w, 3, x, y, 0)
            var idx_g = idx_b + 1
            var idx_r = idx_b + 2
            var base = idx_b
            out[base + 0] = src[idx_r]
            out[base + 1] = src[idx_g]
            out[base + 2] = src[idx_b]
            x += 1
        y += 1
    return out

@staticmethod
fn rgb_to_gray_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    var out: List[UInt8] = List[UInt8]()
    var total = h * w
    var i = 0
    while i < total:
        out.append(0)
        i += 1
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var idx_r = _offset(w, 3, x, y, 0)
            var idx_g = idx_r + 1
            var idx_b = idx_r + 2
            var r = src[idx_r]
            var g = src[idx_g]
            var b = src[idx_b]
            var gray_u16 = UInt16(77) * UInt16(r) + UInt16(150) * UInt16(g) + UInt16(29) * UInt16(b)
            out[y * w + x] = UInt8((gray_u16 >> 8) & UInt16(0xFF))
            x += 1
        y += 1
    return out

@staticmethod
fn rgba_to_rgb_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    var out: List[UInt8] = List[UInt8]()
    var total = h * w * 3
    var i = 0
    while i < total:
        out.append(0)
        i += 1
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var idx_r = _offset(w, 4, x, y, 0)
            var idx_g = idx_r + 1
            var idx_b = idx_r + 2
            var base = _offset(w, 3, x, y, 0)
            out[base + 0] = src[idx_r]
            out[base + 1] = src[idx_g]
            out[base + 2] = src[idx_b]
            x += 1
        y += 1
    return out

@staticmethod
fn bgra_to_bgr_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    var out: List[UInt8] = List[UInt8]()
    var total = h * w * 3
    var i = 0
    while i < total:
        out.append(0)
        i += 1
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var idx_b = _offset(w, 4, x, y, 0)
            var idx_g = idx_b + 1
            var idx_r = idx_b + 2
            var base = _offset(w, 3, x, y, 0)
            out[base + 0] = src[idx_b]
            out[base + 1] = src[idx_g]
            out[base + 2] = src[idx_r]
            x += 1
        y += 1
    return out

@staticmethod
fn argmin_channel_u8_hwc(h: Int, w: Int, src: List[UInt8]) -> List[UInt8]:
    var out: List[UInt8] = List[UInt8]()
    var total = h * w
    var i = 0
    while i < total:
        out.append(0)
        i += 1
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var idx0 = _offset(w, 3, x, y, 0)
            var v0 = src[idx0]
            var v1 = src[idx0 + 1]
            var v2 = src[idx0 + 2]
            var arg = UInt8(0)
            var minv = v0
            if v1 < minv:
                minv = v1
                arg = UInt8(1)
            if v2 < minv:
                arg = UInt8(2)
            out[y * w + x] = arg
            x += 1
        y += 1
    return out

@staticmethod
fn __self_test__() -> Bool:
    var rgb: List[UInt8] = List[UInt8]()
    rgb.append(255); rgb.append(0);   rgb.append(0)
    rgb.append(0);   rgb.append(255); rgb.append(0)
    rgb.append(0);   rgb.append(0);   rgb.append(255)
    rgb.append(255); rgb.append(255); rgb.append(255)
    var g = rgb_to_gray_u8_hwc(2, 2, rgb)
    if len(g) != 4: return False
    var bgr = bgr_to_rgb_u8_hwc(2, 2, rgb)
    if len(bgr) != 12: return False
    var argm = argmin_channel_u8_hwc(2, 2, rgb)
    if len(argm) != 4: return False
    return True
