# Project:      Momijo
# Module:       src.momijo.vision.backend.cpu.convert_color_cpu
# File:         convert_color_cpu.mojo
# Path:         src/momijo/vision/backend/cpu/convert_color_cpu.mojo
#
# Description:  src.momijo.vision.backend.cpu.convert_color_cpu — focused Momijo functionality with a stable public API.
#               Composable building blocks intended for reuse.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Key functions: _offset, bgr_to_rgb_u8_hwc, rgb_to_gray_u8_hwc, rgba_to_rgb_u8_hwc, bgra_to_bgr_u8_hwc, argmin_channel_u8_hwc, __self_test__
#   - Static methods present.


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
            out[y * w + x] = UInt8((gray_u16 >> UInt8(8)) & UInt16(0xFF))
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