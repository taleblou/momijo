# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo
# SPDX-License-Identifier: MIT
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.vision
# File: src/momijo/vision/tile.mojo
#
# Minimal, dependency-light tiling utilities for Momijo Vision.
# Focused on UInt8 buffers in HWC layout. No external imports.
# Style: no 'export', no 'let', no 'inout'.
#
# Implemented:
#   - Tile struct (x,y,w,h) + helpers
#   - tile_grid(h,w, tile_h,tile_w, overlap_h,overlap_w) -> List[Tile]
#   - extract_tile_u8_hwc(h,w,c, src, tile) -> List[UInt8]
#   - split_tiles_u8_hwc(h,w,c, src, tile_h,tile_w, overlap_h,overlap_w) -> (List[Tile], List[List[UInt8]])
#   - reassemble_overwrite_u8_hwc(h,w,c, tiles, bufs) -> List[UInt8]
#   - reassemble_average_u8_hwc(h,w,c, tiles, bufs) -> List[UInt8]   (average over overlaps)
#   - Convenience ImageU8 wrappers
#
# Notes:
# - Overlap is handled by stride = tile_dim - overlap_dim, clipped to >=1.
# - Grid clamps last tiles so that coverage reaches the right/bottom edges.
# - 'average' reassembly uses per-pixel counts to avoid seams in overlaps.

# -------------------------
# Tile & helpers
# -------------------------
struct Tile(Copyable, Movable):
    var x: Int
    var y: Int
    var w: Int
    var h: Int
    fn __init__(out self, x: Int, y: Int, w: Int, h: Int):
        self.x = x; self.y = y; self.w = w; self.h = h
    fn to_string(self) -> String:
        return String("Tile(") + String(self.x) + String(",") + String(self.y) + String(",") + String(self.w) + String("x") + String(self.h) + String(")")

@staticmethod
fn _min(a: Int, b: Int) -> Int:
    if a < b: return a
    return b

@staticmethod
fn _max(a: Int, b: Int) -> Int:
    if a > b: return a
    return b

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
fn _alloc_i32(n: Int) -> List[Int]:
    var out: List[Int] = List[Int]()
    var i = 0
    while i < n:
        out.append(0)
        i += 1
    return out

# -------------------------
# Grid builder
# -------------------------
@staticmethod
fn tile_grid(h: Int, w: Int, tile_h: Int, tile_w: Int, overlap_h: Int, overlap_w: Int) -> List[Tile]:
    var tiles: List[Tile] = List[Tile]()
    if h <= 0 or w <= 0 or tile_h <= 0 or tile_w <= 0:
        return tiles

    var ov_h = overlap_h; if ov_h < 0: ov_h = 0; if ov_h >= tile_h: ov_h = tile_h - 1
    var ov_w = overlap_w; if ov_w < 0: ov_w = 0; if ov_w >= tile_w: ov_w = tile_w - 1

    var stride_h = tile_h - ov_h; if stride_h < 1: stride_h = 1
    var stride_w = tile_w - ov_w; if stride_w < 1: stride_w = 1

    var y = 0
    while True:
        var th = tile_h
        if y + th > h:
            th = h - y
        var x = 0
        while True:
            var tw = tile_w
            if x + tw > w:
                tw = w - x
            tiles.append(Tile(x, y, tw, th))
            if x + tile_w >= w:
                break
            x = x + stride_w
            if x + tile_w > w:
                x = _max(0, w - tile_w)
        if y + tile_h >= h:
            break
        y = y + stride_h
        if y + tile_h > h:
            y = _max(0, h - tile_h)
    return tiles

# -------------------------
# Extract / Split
# -------------------------
@staticmethod
fn extract_tile_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8], t: Tile) -> List[UInt8]:
    var out = _alloc_u8(t.w * t.h * c)
    var yy = 0
    while yy < t.h:
        var x = 0
        while x < t.w:
            var chn = 0
            while chn < c:
                out[_offset(t.w, c, x, yy, chn)] = src[_offset(w, c, t.x + x, t.y + yy, chn)]
                chn += 1
            x += 1
        yy += 1
    return out

@staticmethod
fn split_tiles_u8_hwc(h: Int, w: Int, c: Int, src: List[UInt8],
                      tile_h: Int, tile_w: Int, overlap_h: Int, overlap_w: Int) -> (List[Tile], List[List[UInt8]]):
    var ts = tile_grid(h, w, tile_h, tile_w, overlap_h, overlap_w)
    var bufs: List[List[UInt8]] = List[List[UInt8]]()
    var i = 0
    while i < len(ts):
        bufs.append(extract_tile_u8_hwc(h, w, c, src, ts[i]))
        i += 1
    return (ts, bufs)

# -------------------------
# Reassemble
# -------------------------
@staticmethod
fn reassemble_overwrite_u8_hwc(h: Int, w: Int, c: Int, tiles: List[Tile], bufs: List[List[UInt8]]) -> List[UInt8]:
    var out = _alloc_u8(h * w * c)
    var i = 0
    while i < len(tiles) and i < len(bufs):
        var t = tiles[i]
        var b = bufs[i]
        var yy = 0
        while yy < t.h:
            var x = 0
            while x < t.w:
                var chn = 0
                while chn < c:
                    out[_offset(w, c, t.x + x, t.y + yy, chn)] = b[_offset(t.w, c, x, yy, chn)]
                    chn += 1
                x += 1
            yy += 1
        i += 1
    return out

@staticmethod
fn reassemble_average_u8_hwc(h: Int, w: Int, c: Int, tiles: List[Tile], bufs: List[List[UInt8]]) -> List[UInt8]:
    var accum = _alloc_i32(h * w * c)
    var counts = _alloc_i32(h * w)  # per pixel (not per channel)
    var i = 0
    while i < len(tiles) and i < len(bufs):
        var t = tiles[i]
        var b = bufs[i]
        var yy = 0
        while yy < t.h:
            var x = 0
            while x < t.w:
                # count per target pixel
                counts[(t.y + yy) * w + (t.x + x)] = counts[(t.y + yy) * w + (t.x + x)] + 1
                var chn = 0
                while chn < c:
                    var idx_out = _offset(w, c, t.x + x, t.y + yy, chn)
                    var idx_in  = _offset(t.w, c, x, yy, chn)
                    accum[idx_out] = accum[idx_out] + Int(b[idx_in])
                    chn += 1
                x += 1
            yy += 1
        i += 1

    var out = _alloc_u8(h * w * c)
    var p = 0
    while p < h * w:
        var cnt = counts[p]
        if cnt < 1: cnt = 1
        var chn = 0
        while chn < c:
            var idx = p * c + chn
            var val = (accum[idx] + (cnt // 2)) // cnt
            if val < 0: val = 0
            if val > 255: val = 255
            out[idx] = UInt8(val)
            chn += 1
        p += 1
    return out

# -------------------------
# Lightweight Image wrapper (optional)
# -------------------------
struct ImageU8(Copyable, Movable):
    var h: Int
    var w: Int
    var c: Int
    var data: List[UInt8]
    fn __init__(out self, h: Int, w: Int, c: Int, data: List[UInt8]):
        self.h = h; self.w = w; self.c = c
        var expected = h * w * c
        if len(data) != expected:
            var buf: List[UInt8] = List[UInt8]()
            var i = 0
            while i < expected:
                buf.append(0); i += 1
            self.data = buf
        else:
            self.data = data

@staticmethod
fn split_image_tiles(img: ImageU8, tile_h: Int, tile_w: Int, overlap_h: Int, overlap_w: Int) -> (List[Tile], List[List[UInt8]]):
    return split_tiles_u8_hwc(img.h, img.w, img.c, img.data, tile_h, tile_w, overlap_h, overlap_w)

@staticmethod
fn reassemble_overwrite_image(img_h: Int, img_w: Int, c: Int, tiles: List[Tile], bufs: List[List[UInt8]]) -> ImageU8:
    var out = reassemble_overwrite_u8_hwc(img_h, img_w, c, tiles, bufs)
    return ImageU8(img_h, img_w, c, out)

@staticmethod
fn reassemble_average_image(img_h: Int, img_w: Int, c: Int, tiles: List[Tile], bufs: List[List[UInt8]]) -> ImageU8:
    var out = reassemble_average_u8_hwc(img_h, img_w, c, tiles, bufs)
    return ImageU8(img_h, img_w, c, out)

# -------------------------
# Minimal smoke test
# -------------------------
@staticmethod
fn __self_test__() -> Bool:
    # Build 5x7 RGB gradient
    var h = 5; var w = 7; var c = 3
    var src: List[UInt8] = List[UInt8]()
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            # r = x*10, g = y*20, b = (x+y)*5
            src.append(UInt8((x * 10) & 255))
            src.append(UInt8((y * 20) & 255))
            src.append(UInt8(((x + y) * 5) & 255))
            x += 1
        y += 1

    # 1) Split with no overlap; reassemble overwrite == original
    var tiles0: List[Tile] = List[Tile](); var bufs0: List[List[UInt8]] = List[List[UInt8]]()
    (tiles0, bufs0) = split_tiles_u8_hwc(h, w, c, src, 2, 3, 0, 0)
    var out0 = reassemble_overwrite_u8_hwc(h, w, c, tiles0, bufs0)
    if len(out0) != len(src): return False
    var i = 0
    while i < len(src):
        if out0[i] != src[i]: return False
        i += 1

    # 2) Split with overlap; average reassembly should equal original if we don't modify tiles
    var tiles1: List[Tile] = List[Tile](); var bufs1: List[List[UInt8]] = List[List[UInt8]]()
    (tiles1, bufs1) = split_tiles_u8_hwc(h, w, c, src, 3, 4, 1, 1)
    var out1 = reassemble_average_u8_hwc(h, w, c, tiles1, bufs1)
    if len(out1) != len(src): return False
    i = 0
    while i < len(src):
        if out1[i] != src[i]: return False
        i += 1

    # 3) ImageU8 wrappers
    var img = ImageU8(h, w, c, src)
    var tiles2: List[Tile] = List[Tile](); var bufs2: List[List[UInt8]] = List[List[UInt8]]()
    (tiles2, bufs2) = split_image_tiles(img, 3, 3, 1, 2)
    var img_rec = reassemble_average_image(h, w, c, tiles2, bufs2)
    if len(img_rec.data) != h*w*c: return False

    return True
