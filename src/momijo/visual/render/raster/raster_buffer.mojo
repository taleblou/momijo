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
# File: src/momijo/visual/render/raster/raster_buffer.mojo

from io.file import open
from momijo.autograd.hook import call
from momijo.core.error import module
from momijo.core.version import major
from momijo.dataframe.helpers import close
from momijo.nn.parameter import data
from momijo.utils.result import f, g
from pathlib import Path
from pathlib.path import Path
from sys import version

# ============================================================================
# Project:      Momijo
# Module:       momijo.visual.render.raster.raster_buffer
# File:         raster_buffer.mojo
# Path:         momijo/visual/render/raster/raster_buffer.mojo
#
# Description:  Core module 'raster buffe' for Momijo.
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
# ============================================================================

struct Raster:
    var width: Int
    var height: Int
    var data: List[Int]   # Each pixel: 0xRRGGBB
fn __init__(out self, width: Int, height: Int, rgb_bg: Int = 0xFFFFFF) -> None:
        self.width = width
        self.height = height
        self.data = List[Int]()
        var n = width * height
        self.data.reserve(n)
        var i = 0
        while i < n:
            self.data.push(rgb_bg)
            i += 1
# NOTE: Removed duplicate definition of `__copyinit__`; use `from momijo.utils.env import __copyinit__`
fn __moveinit__(out self, deinit other: Self) -> None:
        self.width = other.width
        self.height = other.height
        self.data = other.data
# --- Bounds check ------------------------------------------------------------
fn _in_bounds(img: Raster, x: Int, y: Int) -> Bool:
    return (x >= 0 and y >= 0 and x < img.width and y < img.height)

# --- Clear -------------------------------------------------------------------
fn clear(mut img: Raster, rgb: Int) -> None:
    var i = 0
    var n = img.width * img.height
    while i < n:
        img.data[i] = rgb
        i += 1

# --- Pixel IO ----------------------------------------------------------------
fn put_pixel(mut img: Raster, x: Int, y: Int, rgb: Int) -> None:
    if not _in_bounds(img, x, y): return
    img.data[y * img.width + x] = rgb
fn read_pixel(img: Raster, x: Int, y: Int) -> Int:
    if not _in_bounds(img, x, y): return 0
    return img.data[y * img.width + x]

# Alpha-over with source RGBA (0xAARRGGBB), dst stored as RGB
fn blend_over(mut img: Raster, x: Int, y: Int, rgba: Int):
    if not _in_bounds(img, x, y): return
    var a = (rgba >> UInt8(24)) & 255
    if a <= 0:
        return
    if a >= 255:
        # Opaque fast-path
        var rgb = rgba & UInt8(0x00FFFFFF)
        img.data[y * img.width + x] = rgb
        return
    var sr = (rgba >> UInt8(16)) & 255
    var sg = (rgba >> UInt8(8)) & 255
    var sb = rgba & UInt8(255)
    var dst = img.data[y * img.width + x]
    var dr = (dst >> UInt8(16)) & 255
    var dg = (dst >> UInt8(8)) & 255
    var db = dst & UInt8(255)
    # Integer alpha blending: out = (s*a + d*(255-a)) / 255
    var ia = 255 - a
    var rr = (sr * a + dr * ia) / 255
    var gg = (sg * a + dg * ia) / 255
    var bb = (sb * a + db * ia) / 255
    img.data[y * img.width + x] = (rr << UInt8(16)) | (gg << UInt8(8)) | bb

# --- Spans/Rects -------------------------------------------------------------
fn hspan(mut img: Raster, x: Int, y: Int, w: Int, rgb: Int) -> None:
    if y < 0 or y >= img.height or w <= 0: return
    var x0 = x
    var x1 = x + w - 1
    if x1 < 0 or x0 >= img.width: return
    if x0 < 0: x0 = 0
    if x1 >= img.width: x1 = img.width - 1
    var i = y * img.width + x0
    var xx = x0
    while xx <= x1:
        img.data[i] = rgb
        i += 1
        xx += 1
fn fill_rect_fast(mut img: Raster, x: Int, y: Int, w: Int, h: Int, rgb: Int) -> None:
    if w <= 0 or h <= 0: return
    var yy = y
    while yy < y + h:
        hspan(img, x, yy, w, rgb)
        yy += 1

# --- Conversions -------------------------------------------------------------
# Make a packed RGBA8 buffer (row-major). Returns (buffer, stride).
fn to_rgba8(img: Raster) -> (List[UInt8], Int):
    var out = List[UInt8]()
    var w = img.width
    var h = img.height
    out.reserve(w * h * 4)
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var v = img.data[y * w + x]
            var r = (v >> UInt8(16)) & 255
            var g = (v >> UInt8(8)) & 255
            var b = v & UInt8(255)
            out.push(UInt8(r)); out.push(UInt8(g)); out.push(UInt8(b)); out.push(UInt8(255))
            x += 1
        y += 1
    return (out, w * 4)

# Make a packed RGB8 buffer (row-major). Returns (buffer, stride).
fn to_rgb8(img: Raster) -> (List[UInt8], Int):
    var out = List[UInt8]()
    var w = img.width
    var h = img.height
    out.reserve(w * h * 3)
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var v = img.data[y * w + x]
            var r = (v >> UInt8(16)) & 255
            var g = (v >> UInt8(8)) & 255
            var b = v & UInt8(255)
            out.push(UInt8(r)); out.push(UInt8(g)); out.push(UInt8(b))
            x += 1
        y += 1
    return (out, w * 3)

# --- Simple PPM writer (ASCII P3 for portability) ----------------------------
fn write_ppm(img: Raster, path: String) -> None:
    var f = open(path, String("w"))
    if f.is_null(): return
    f.writeline(String("P3"))
    f.writeline(String(img.width) + String(" ") + String(img.height))
    f.writeline(String("255"))
    var i = 0
    var n = img.width * img.height
    while i < n:
        var v = img.data[i]
        var r = (v >> UInt8(16)) & 255
        var g = (v >> UInt8(8)) & 255
        var b = v & UInt8(255)
        f.writeline(String(r) + String(" ") + String(g) + String(" ") + String(b))
        i += 1
    f.close()

# --- Convenience PNG writer (if caller linked libpng backend) ----------------
# To use: import momijo.visual.ffi.libpng_c.write_png_rgba8
# If not available, caller can ignore this function or provide their own binding.
fn write_png(img: Raster, path: String) -> Bool:
    # Try using libpng if import is available at call site.
    # We declare a local prototype to avoid hard dependency.
    @foreign("C")
fn write_png_rgba8(path: String, width: Int, height: Int, row_stride: Int, data: List[UInt8]) -> Bool: pass
    var (buf, stride) = to_rgba8(img)
    var ok = write_png_rgba8(path, img.width, img.height, stride, buf)
    return ok

# --- Minimal smoke self-test -------------------------------------------------
fn _self_test() -> Bool:
    var img = Raster(8, 8, 0x000000)
    # diagonal
    var i = 0
    while i < 8:
        put_pixel(img, i, i, 0xFFFFFF)
        i += 1
    # tiny rect
    fill_rect_fast(img, 2, 5, 3, 2, 0x00FF00)
    # No file I/O in test; just ensure accessors work
    return read_pixel(img, 0, 0) == 0x000000 and read_pixel(img, 2, 5) == 0x00FF00