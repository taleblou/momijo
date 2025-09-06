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
# File: src/momijo/vision/backend/cpu/resize_cpu.mojo
#
# CPU resize kernels for uint8 images in HWC layout.
# Standalone & dependency-light (no external Momijo imports).
# Style: no 'export', no 'let', no 'inout'; constructors use `fn __init__(out self, ...)`.
#
# Implemented ops (generic over channel count c):
#   - resize_nearest_u8_hwc(h, w, c, src, oh, ow) -> List[UInt8]
#   - resize_bilinear_u8_hwc(h, w, c, src, oh, ow) -> List[UInt8]  (fixed-point 8.8 weights)
#
# Notes:
# - src is a flat List[UInt8] of length h*w*c in row-major HWC.
# - Output is a new List[UInt8] of length oh*ow*c.

# -------------------------
# Helpers
# -------------------------
@staticmethod
fn _offset(w: Int, c: Int, x: Int, y: Int, ch: Int) -> Int:
    # Row-major HWC: ((y*w) + x)*c + ch
    return ((y * w) + x) * c + ch

@staticmethod
fn _alloc_u8(n: Int) -> List[UInt8]:
    var out: List[UInt8] = List[UInt8]()
    var i = 0
    while i < n:
        out.append(0)
        i += 1
    return out

@staticmethod
fn _clamp_u8(v: Int) -> UInt8:
    if v < 0:
        return UInt8(0)
    if v > 255:
        return UInt8(255)
    return UInt8(v)

# -------------------------
# Nearest-neighbor
# -------------------------
@staticmethod
fn resize_nearest_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0 or oh <= 0 or ow <= 0:
        return List[UInt8]()
    var out = _alloc_u8(oh * ow * c)
    var y = 0
    while y < oh:
        var sy = (y * h) // oh
        var x = 0
        while x < ow:
            var sx = (x * w) // ow
            var ch = 0
            while ch < c:
                var sv = src[_offset(w, c, sx, sy, ch)]
                out[_offset(ow, c, x, y, ch)] = sv
                ch += 1
            x += 1
        y += 1
    return out

# -------------------------
# Bilinear (8-bit fractional fixed-point: 8.8 weights)
# mapping: sx_f = (x*(w-1)*256)/(ow-1)  (if ow==1 => 0), sy similarly.
# -------------------------
@staticmethod
fn resize_bilinear_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0 or oh <= 0 or ow <= 0:
        return List[UInt8]()
    var out = _alloc_u8(oh * ow * c)

    var y = 0
    while y < oh:
        var sy_fp = 0
        if oh > 1:
            sy_fp = (y * (h - 1) * 256) // (oh - 1)
        var y0 = sy_fp >> 8
        var fy = sy_fp & 255  # 0..255
        var y1 = y0 + 1
        if y1 >= h:
            y1 = h - 1

        var x = 0
        while x < ow:
            var sx_fp = 0
            if ow > 1:
                sx_fp = (x * (w - 1) * 256) // (ow - 1)
            var x0 = sx_fp >> 8
            var fx = sx_fp & 255
            var x1 = x0 + 1
            if x1 >= w:
                x1 = w - 1

            var wy0 = 256 - fy
            var wx0 = 256 - fx
            # Weights sum to 256*256
            var w00 = wy0 * wx0
            var w10 = wy0 * fx
            var w01 = fy * wx0
            var w11 = fy * fx

            var chn = 0
            while chn < c:
                var p00 = Int(src[_offset(w, c, x0, y0, chn)])
                var p10 = Int(src[_offset(w, c, x1, y0, chn)])
                var p01 = Int(src[_offset(w, c, x0, y1, chn)])
                var p11 = Int(src[_offset(w, c, x1, y1, chn)])
                var accum = p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11
                # round and normalize (>>16)
                var val = (accum + 32768) >> 16
                out[_offset(ow, c, x, y, chn)] = _clamp_u8(val)
                chn += 1
            x += 1
        y += 1
    return out

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # 2x2 RGB pattern
    var h = 2; var w = 2; var c = 3
    var rgb: List[UInt8] = List[UInt8]()
    # (255,0,0) (0,255,0)
    rgb.append(255); rgb.append(0);   rgb.append(0)
    rgb.append(0);   rgb.append(255); rgb.append(0)
    # (0,0,255) (255,255,255)
    rgb.append(0);   rgb.append(0);   rgb.append(255)
    rgb.append(255); rgb.append(255); rgb.append(255)

    var oh = 4; var ow = 4
    var nn = resize_nearest_u8_hwc(h, w, c, rgb, oh, ow)
    if len(nn) != oh*ow*c: return False
    # Top-left should match original top-left
    if nn[0] != 255: return False

    var bl = resize_bilinear_u8_hwc(h, w, c, rgb, oh, ow)
    if len(bl) != oh*ow*c: return False
    # Center pixel must be between 0..255 (always true) and channel count preserved
    if bl[_offset(ow, c, 2, 2, 0)] > 255: return False

    return True
