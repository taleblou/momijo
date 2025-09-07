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
# Project: momijo.visual.render.raster
# File: src/momijo/visual/render/raster/simd.mojo

from momijo.visual.render.raster.raster_buffer import Raster

fn _clipi(v: Int, lo: Int, hi: Int) -> Int:
    if v < lo: return lo
    if v > hi: return hi
    return v

# Fast memset-like for 32-bit pixels in List[Int] with loop unrolling by 8.
# Writes 'value' to data[start : start+n).
fn _memset32_line(mut data: List[Int], start: Int, n: Int, value: Int) -> None:
    if n <= 0: return
    var i = 0
    var idx = start
    # Write blocks of 8
    var n8 = n >> UInt8(3)
    while i < n8:
        data[idx + 0] = value
        data[idx + 1] = value
        data[idx + 2] = value
        data[idx + 3] = value
        data[idx + 4] = value
        data[idx + 5] = value
        data[idx + 6] = value
        data[idx + 7] = value
        idx += 8
        i += 1
    # Remainder
    var rem = n & UInt8(7)
    var k = 0
    while k < rem:
        data[idx + k] = value
        k += 1

# --- Clipped horizontal span (fast) ------------------------------------------
fn hspan_simd(mut img: Raster, x: Int, y: Int, w: Int, rgb: Int) -> None:
    if w <= 0 or y < 0 or y >= img.height: return
    var x0 = x
    var x1 = x + w - 1
    if x1 < 0 or x0 >= img.width: return
    if x0 < 0: x0 = 0
    if x1 >= img.width: x1 = img.width - 1
    var n = x1 - x0 + 1
    var start = y * img.width + x0
    _memset32_line(img.data, start, n, rgb)

# --- Rect fill with span writer (SIMD-ish) -----------------------------------
fn fill_rect_simd(mut img: Raster, x: Int, y: Int, w: Int, h: Int, rgb: Int) -> None:
    if w <= 0 or h <= 0: return
    var y0 = y
    var y1 = y + h - 1
    if y1 < 0 or y0 >= img.height: return
    if y0 < 0: y0 = 0
    if y1 >= img.height: y1 = img.height - 1
    var yy = y0
    while yy <= y1:
        hspan_simd(img, x, yy, w, rgb)
        yy += 1

# --- Alpha blit (RGBA source -> RGB dest), loop-unrolled by 8 ----------------
# src_rgba: packed bytes (row_stride bytes per row). Each pixel: R,G,B,A
fn blit_rgba8_over_simd(mut dst: Raster, dx: Int, dy: Int,
                        src_rgba: List[UInt8], src_w: Int, src_h: Int, src_stride: Int) -> None:
    if src_w <= 0 or src_h <= 0: return
    var y = 0
    while y < src_h:
        var x = 0
        var base = y * src_stride
        # Process in groups of 8 pixels
        var x_end = src_w & ~7  # floor to multiple of 8
        while x < x_end:
            var k = 0
            while k < 8:
                var sx = x + k
                var dst_x = dx + sx
                var dst_y = dy + y
                if dst_x >= 0 and dst_y >= 0 and dst_x < dst.width and dst_y < dst.height:
                    var o = base + (sx << UInt8(2))
                    var sr = Int(src_rgba[o + 0])
                    var sg = Int(src_rgba[o + 1])
                    var sb = Int(src_rgba[o + 2])
                    var sa = Int(src_rgba[o + 3])
                    if sa >= 255:
                        dst.data[dst_y * dst.width + dst_x] = (sr << UInt8(16)) | (sg << UInt8(8)) | sb
                    elif sa > 0:
                        var d = dst.data[dst_y * dst.width + dst_x]
                        var dr = (d >> UInt8(16)) & 255
                        var dg = (d >> UInt8(8)) & 255
                        var db = d & UInt8(255)
                        var ia = 255 - sa
                        var rr = (sr * sa + dr * ia) / 255
                        var gg = (sg * sa + dg * ia) / 255
                        var bb = (sb * sa + db * ia) / 255
                        dst.data[dst_y * dst.width + dst_x] = (rr << UInt8(16)) | (gg << UInt8(8)) | bb
                k += 1
            x += 8
        # Remainder
        while x < src_w:
            var dst_x = dx + x
            var dst_y = dy + y
            if dst_x >= 0 and dst_y >= 0 and dst_x < dst.width and dst_y < dst.height:
                var o = base + (x << UInt8(2))
                var sr = Int(src_rgba[o + 0])
                var sg = Int(src_rgba[o + 1])
                var sb = Int(src_rgba[o + 2])
                var sa = Int(src_rgba[o + 3])
                if sa >= 255:
                    dst.data[dst_y * dst.width + dst_x] = (sr << UInt8(16)) | (sg << UInt8(8)) | sb
                elif sa > 0:
                    var d = dst.data[dst_y * dst.width + dst_x]
                    var dr = (d >> UInt8(16)) & 255
                    var dg = (d >> UInt8(8)) & 255
                    var db = d & UInt8(255)
                    var ia = 255 - sa
                    var rr = (sr * sa + dr * ia) / 255
                    var gg = (sg * sa + dg * ia) / 255
                    var bb = (sb * sa + db * ia) / 255
                    dst.data[dst_y * dst.width + dst_x] = (rr << UInt8(16)) | (gg << UInt8(8)) | bb
            x += 1
        y += 1

# --- Raster copy (rectangle), clipped, unrolled by 8 -------------------------
fn blit_raster_simd(mut dst: Raster, dx: Int, dy: Int, src: Raster, sx: Int, sy: Int, w: Int, h: Int) -> None:
    if w <= 0 or h <= 0: return
    # Clip source/dest rectangles
    var x0 = 0; var y0 = 0
    var x1 = w; var y1 = h
    # Left clip
    if dx < 0:
        x0 = -dx
    if dy < 0:
        y0 = -dy
    # Right/Bottom clip
    if dx + w > dst.width:
        x1 = dst.width - dx
    if dy + h > dst.height:
        y1 = dst.height - dy
    # Source bounds clip
    if sx + x1 > src.width:
        x1 = src.width - sx
    if sy + y1 > src.height:
        y1 = src.height - sy
    if x1 <= x0 or y1 <= y0: return

    var yy = y0
    while yy < y1:
        var xx = x0
        var n = x1 - x0
        var sbase = (sy + yy) * src.width + (sx + x0)
        var dbase = (dy + yy) * dst.width + (dx + x0)
        # Blocks of 8
        var n8 = n >> UInt8(3)
        var i = 0
        while i < n8:
            dst.data[dbase + 0] = src.data[sbase + 0]
            dst.data[dbase + 1] = src.data[sbase + 1]
            dst.data[dbase + 2] = src.data[sbase + 2]
            dst.data[dbase + 3] = src.data[sbase + 3]
            dst.data[dbase + 4] = src.data[sbase + 4]
            dst.data[dbase + 5] = src.data[sbase + 5]
            dst.data[dbase + 6] = src.data[sbase + 6]
            dst.data[dbase + 7] = src.data[sbase + 7]
            dbase += 8; sbase += 8
            i += 1
        # Remainder
        var rem = n & UInt8(7)
        var k = 0
        while k < rem:
            dst.data[dbase + k] = src.data[sbase + k]
            k += 1
        yy += 1

# --- Tiled clear (cache-friendly) --------------------------------------------
fn clear_tiled(mut img: Raster, rgb: Int, tile_h: Int = 64) -> None:
    var H = img.height
    var W = img.width
    if H <= 0 or W <= 0: return
    var y0 = 0
    while y0 < H:
        var yend = y0 + tile_h
        if yend > H: yend = H
        var yy = y0
        while yy < yend:
            hspan_simd(img, 0, yy, W, rgb)
            yy += 1
        y0 += tile_h

# --- Self test ---------------------------------------------------------------
fn _self_test() -> Bool:
    var img = Raster(32, 16, 0x000000)
    fill_rect_simd(img, 2, 3, 10, 4, 0x112233)
    # Basic sanity checks on edges of the filled rect
    return (img.data[3 * img.width + 2] == 0x112233) and (img.data[(3+3) * img.width + (2+9)] == 0x112233)