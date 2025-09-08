# Project:      Momijo
# Module:       src.momijo.vision.backend.cpu.simd.resize_simd_u8_hwc
# File:         resize_simd_u8_hwc.mojo
# Path:         src/momijo/vision/backend/cpu/simd/resize_simd_u8_hwc.mojo
#
# Description:  src.momijo.vision.backend.cpu.simd.resize_simd_u8_hwc — focused Momijo functionality with a stable public API.
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
#   - Key functions: _offset, _alloc_u8, _clamp_u8, simd_resize_nearest_u8_hwc, simd_resize_bilinear_u8_hwc, __self_test__
#   - Static methods present.
#   - Uses generic functions/types with explicit trait bounds.


@staticmethod
fn _offset(w: Int, c: Int, x: Int, y: Int, ch: Int) -> Int:
    # Row-major HWC
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
    if v < 0: return UInt8(0)
    if v > 255: return UInt8(255)
    return UInt8(v)

# -------------------------
# Nearest-neighbor (chunked over x for better vectorization)
# -------------------------
@staticmethod
fn simd_resize_nearest_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0 or oh <= 0 or ow <= 0:
        return List[UInt8]()
    var out = _alloc_u8(oh * ow * c)

    var y = 0
    while y < oh:
        var sy = (y * h) // oh
        var x = 0
        # process 8 pixels per iteration to encourage auto-vectorization
        while x + 7 < ow:
            var sx0 = ((x + 0) * w) // ow
            var sx1 = ((x + 1) * w) // ow
            var sx2 = ((x + 2) * w) // ow
            var sx3 = ((x + 3) * w) // ow
            var sx4 = ((x + 4) * w) // ow
            var sx5 = ((x + 5) * w) // ow
            var sx6 = ((x + 6) * w) // ow
            var sx7 = ((x + 7) * w) // ow

            var chn = 0
            while chn < c:
                var sv0 = src[_offset(w, c, sx0, sy, chn)]
                var sv1 = src[_offset(w, c, sx1, sy, chn)]
                var sv2 = src[_offset(w, c, sx2, sy, chn)]
                var sv3 = src[_offset(w, c, sx3, sy, chn)]
                var sv4 = src[_offset(w, c, sx4, sy, chn)]
                var sv5 = src[_offset(w, c, sx5, sy, chn)]
                var sv6 = src[_offset(w, c, sx6, sy, chn)]
                var sv7 = src[_offset(w, c, sx7, sy, chn)]

                out[_offset(ow, c, x + 0, y, chn)] = sv0
                out[_offset(ow, c, x + 1, y, chn)] = sv1
                out[_offset(ow, c, x + 2, y, chn)] = sv2
                out[_offset(ow, c, x + 3, y, chn)] = sv3
                out[_offset(ow, c, x + 4, y, chn)] = sv4
                out[_offset(ow, c, x + 5, y, chn)] = sv5
                out[_offset(ow, c, x + 6, y, chn)] = sv6
                out[_offset(ow, c, x + 7, y, chn)] = sv7

                chn += 1
            x += 8
        # tail
        while x < ow:
            var sx = (x * w) // ow
            var ch2 = 0
            while ch2 < c:
                out[_offset(ow, c, x, y, ch2)] = src[_offset(w, c, sx, sy, ch2)]
                ch2 += 1
            x += 1
        y += 1
    return out

# -------------------------
# Bilinear (chunked 4 pixels per loop; fixed-point 8.8)
# -------------------------
@staticmethod
fn simd_resize_bilinear_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], oh: Int, ow: Int) -> List[UInt8]:
    if h <= 0 or w <= 0 or c <= 0 or oh <= 0 or ow <= 0:
        return List[UInt8]()
    var out = _alloc_u8(oh * ow * c)

    var y = 0
    while y < oh:
        var sy_fp = 0
        if oh > 1:
            sy_fp = (y * (h - 1) * 256) // (oh - 1)
        var y0 = sy_fp >> UInt8(8)
        var fy = sy_fp & UInt8(255)
        var y1 = y0 + 1
        if y1 >= h: y1 = h - 1

        var wy0 = 256 - fy

        var x = 0
        while x + 3 < ow:
            var sx0_fp = 0; var sx1_fp = 0; var sx2_fp = 0; var sx3_fp = 0
            if ow > 1:
                sx0_fp = ((x + 0) * (w - 1) * 256) // (ow - 1)
                sx1_fp = ((x + 1) * (w - 1) * 256) // (ow - 1)
                sx2_fp = ((x + 2) * (w - 1) * 256) // (ow - 1)
                sx3_fp = ((x + 3) * (w - 1) * 256) // (ow - 1)

            var x0_0 = sx0_fp >> UInt8(8); var fx0 = sx0_fp & UInt8(255); var x1_0 = x0_0 + 1; if x1_0 >= w: x1_0 = w - 1
            var x0_1 = sx1_fp >> UInt8(8); var fx1 = sx1_fp & UInt8(255); var x1_1 = x0_1 + 1; if x1_1 >= w: x1_1 = w - 1
            var x0_2 = sx2_fp >> UInt8(8); var fx2 = sx2_fp & UInt8(255); var x1_2 = x0_2 + 1; if x1_2 >= w: x1_2 = w - 1
            var x0_3 = sx3_fp >> UInt8(8); var fx3 = sx3_fp & UInt8(255); var x1_3 = x0_3 + 1; if x1_3 >= w: x1_3 = w - 1

            var wx0_0 = 256 - fx0; var wx0_1 = 256 - fx1; var wx0_2 = 256 - fx2; var wx0_3 = 256 - fx3

            var w00_0 = wy0 * wx0_0; var w10_0 = wy0 * fx0; var w01_0 = fy * wx0_0; var w11_0 = fy * fx0
            var w00_1 = wy0 * wx0_1; var w10_1 = wy0 * fx1; var w01_1 = fy * wx0_1; var w11_1 = fy * fx1
            var w00_2 = wy0 * wx0_2; var w10_2 = wy0 * fx2; var w01_2 = fy * wx0_2; var w11_2 = fy * fx2
            var w00_3 = wy0 * wx0_3; var w10_3 = wy0 * fx3; var w01_3 = fy * wx0_3; var w11_3 = fy * fx3

            var chn = 0
            while chn < c:
                var p00_0 = Int(src[_offset(w, c, x0_0, y0, chn)])
                var p10_0 = Int(src[_offset(w, c, x1_0, y0, chn)])
                var p01_0 = Int(src[_offset(w, c, x0_0, y1, chn)])
                var p11_0 = Int(src[_offset(w, c, x1_0, y1, chn)])

                var p00_1 = Int(src[_offset(w, c, x0_1, y0, chn)])
                var p10_1 = Int(src[_offset(w, c, x1_1, y0, chn)])
                var p01_1 = Int(src[_offset(w, c, x0_1, y1, chn)])
                var p11_1 = Int(src[_offset(w, c, x1_1, y1, chn)])

                var p00_2 = Int(src[_offset(w, c, x0_2, y0, chn)])
                var p10_2 = Int(src[_offset(w, c, x1_2, y0, chn)])
                var p01_2 = Int(src[_offset(w, c, x0_2, y1, chn)])
                var p11_2 = Int(src[_offset(w, c, x1_2, y1, chn)])

                var p00_3 = Int(src[_offset(w, c, x0_3, y0, chn)])
                var p10_3 = Int(src[_offset(w, c, x1_3, y0, chn)])
                var p01_3 = Int(src[_offset(w, c, x0_3, y1, chn)])
                var p11_3 = Int(src[_offset(w, c, x1_3, y1, chn)])

                var acc0 = p00_0 * w00_0 + p10_0 * w10_0 + p01_0 * w01_0 + p11_0 * w11_0
                var acc1 = p00_1 * w00_1 + p10_1 * w10_1 + p01_1 * w01_1 + p11_1 * w11_1
                var acc2 = p00_2 * w00_2 + p10_2 * w10_2 + p01_2 * w01_2 + p11_2 * w11_2
                var acc3 = p00_3 * w00_3 + p10_3 * w10_3 + p01_3 * w01_3 + p11_3 * w11_3

                out[_offset(ow, c, x + 0, y, chn)] = _clamp_u8((acc0 + 32768) >> 16)
                out[_offset(ow, c, x + 1, y, chn)] = _clamp_u8((acc1 + 32768) >> 16)
                out[_offset(ow, c, x + 2, y, chn)] = _clamp_u8((acc2 + 32768) >> 16)
                out[_offset(ow, c, x + 3, y, chn)] = _clamp_u8((acc3 + 32768) >> 16)

                chn += 1
            x += 4

        # tail
        while x < ow:
            var sx_fp = 0
            if ow > 1:
                sx_fp = (x * (w - 1) * 256) // (ow - 1)
            var x0 = sx_fp >> UInt8(8)
            var fx = sx_fp & UInt8(255)
            var x1 = x0 + 1
            if x1 >= w: x1 = w - 1

            var wx0 = 256 - fx
            var w00 = wy0 * wx0
            var w10 = wy0 * fx
            var w01 = fy * wx0
            var w11 = fy * fx

            var ch2 = 0
            while ch2 < c:
                var p00 = Int(src[_offset(w, c, x0, y0, ch2)])
                var p10 = Int(src[_offset(w, c, x1, y0, ch2)])
                var p01 = Int(src[_offset(w, c, x0, y1, ch2)])
                var p11 = Int(src[_offset(w, c, x1, y1, ch2)])
                var acc = p00 * w00 + p10 * w10 + p01 * w01 + p11 * w11
                out[_offset(ow, c, x, y, ch2)] = _clamp_u8((acc + 32768) >> 16)
                ch2 += 1
            x += 1
        y += 1
    return out

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # 2x2 RGB test pattern
    var h = 2; var w = 2; var c = 3
    var rgb: List[UInt8] = List[UInt8]()
    # (255,0,0) (0,255,0)
    rgb.append(255); rgb.append(0);   rgb.append(0)
    rgb.append(0);   rgb.append(255); rgb.append(0)
    # (0,0,255) (255,255,255)
    rgb.append(0);   rgb.append(0);   rgb.append(255)
    rgb.append(255); rgb.append(255); rgb.append(255)

    var oh = 5; var ow = 7
    var nn = simd_resize_nearest_u8_hwc(h,w,c, rgb, oh,ow)
    if len(nn) != oh*ow*c: return False

    var bl = simd_resize_bilinear_u8_hwc(h,w,c, rgb, oh,ow)
    if len(bl) != oh*ow*c: return False

    # Edge: degenerate dims
    var nn2 = simd_resize_nearest_u8_hwc(h,w,c, rgb, 1,1)
    if len(nn2) != 1*c: return False

    return True