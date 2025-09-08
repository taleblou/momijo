# Project:      Momijo
# Module:       src.momijo.vision.backend.cpu.resize_tiled_cpu
# File:         resize_tiled_cpu.mojo
# Path:         src/momijo/vision/backend/cpu/resize_tiled_cpu.mojo
#
# Description:  src.momijo.vision.backend.cpu.resize_tiled_cpu — focused Momijo functionality with a stable public API.
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
#   - Key functions: _offset, _alloc_u8, _min, _clamp_u8, resize_nearest_tiled_u8_hwc, resize_bilinear_tiled_u8_hwc, __self_test__
#   - Static methods present.
#   - Uses generic functions/types with explicit trait bounds.


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

@staticmethod
fn _min(a: Int, b: Int) -> Int:
    if a < b: return a
    return b

@staticmethod
fn _clamp_u8(v: Int) -> UInt8:
    if v < 0: return UInt8(0)
    if v > 255: return UInt8(255)
    return UInt8(v)

# -------------------------
# Nearest neighbor (tiled)
# -------------------------
@staticmethod
fn resize_nearest_tiled_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int, tile_h: Int, tile_w: Int) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0 or oh <= 0 or ow <= 0:
        return List[UInt8]()
    if tile_h <= 0: tile_h = 32
    if tile_w <= 0: tile_w = 64

    var out = _alloc_u8(oh * ow * c)

    var y0 = 0
    while y0 < oh:
        var y1 = _min(y0 + tile_h, oh)
        var x0 = 0
        while x0 < ow:
            var x1 = _min(x0 + tile_w, ow)

            var y = y0
            while y < y1:
                var sy = (y * h) // oh
                var x = x0
                while x < x1:
                    var sx = (x * w) // ow
                    var ch = 0
                    while ch < c:
                        out[_offset(ow, c, x, y, ch)] = src[_offset(w, c, sx, sy, ch)]
                        ch += 1
                    x += 1
                y += 1

            x0 += tile_w
        y0 += tile_h

    return out

# -------------------------
# Bilinear (tiled; fixed-point 8.8 weights)
# -------------------------
@staticmethod
fn resize_bilinear_tiled_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int, tile_h: Int, tile_w: Int) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0 or oh <= 0 or ow <= 0:
        return List[UInt8]()
    if tile_h <= 0: tile_h = 32
    if tile_w <= 0: tile_w = 64

    var out = _alloc_u8(oh * ow * c)

    var y0 = 0
    while y0 < oh:
        var y1 = _min(y0 + tile_h, oh)
        var x0 = 0
        while x0 < ow:
            var x1 = _min(x0 + tile_w, ow)

            var y = y0
            while y < y1:
                var sy_fp = 0
                if oh > 1:
                    sy_fp = (y * (h - 1) * 256) // (oh - 1)
                var yb = sy_fp >> UInt8(8)
                var fy = sy_fp & UInt8(255)
                var yb1 = yb + 1
                if yb1 >= h:
                    yb1 = h - 1

                var x = x0
                while x < x1:
                    var sx_fp = 0
                    if ow > 1:
                        sx_fp = (x * (w - 1) * 256) // (ow - 1)
                    var xb = sx_fp >> UInt8(8)
                    var fx = sx_fp & UInt8(255)
                    var xb1 = xb + 1
                    if xb1 >= w:
                        xb1 = w - 1

                    var wy0 = 256 - fy
                    var wx0 = 256 - fx
                    var w00 = wy0 * wx0
                    var w10 = wy0 * fx
                    var w01 = fy * wx0
                    var w11 = fy * fx

                    var chn = 0
                    while chn < c:
                        var p00 = Int(src[_offset(w, c, xb,  yb,  chn)])
                        var p10 = Int(src[_offset(w, c, xb1, yb,  chn)])
                        var p01 = Int(src[_offset(w, c, xb,  yb1, chn)])
                        var p11 = Int(src[_offset(w, c, xb1, yb1, chn)])
                        var accum = p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11
                        var val = (accum + 32768) >> 16
                        out[_offset(ow, c, x, y, chn)] = _clamp_u8(val)
                        chn += 1
                    x += 1
                y += 1

            x0 += tile_w
        y0 += tile_h

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

    var oh = 5; var ow = 7
    var tile_h = 3; var tile_w = 4

    var nn = resize_nearest_tiled_u8_hwc(h, w, c, rgb, oh, ow, tile_h, tile_w)
    if len(nn) != oh*ow*c: return False

    var bl = resize_bilinear_tiled_u8_hwc(h, w, c, rgb, oh, ow, tile_h, tile_w)
    if len(bl) != oh*ow*c: return False

    # Edge cases: degenerate dims & tiles
    var nn2 = resize_nearest_tiled_u8_hwc(h, w, c, rgb, 1, 1, 0, 0)
    if len(nn2) != 1*c: return False

    return True