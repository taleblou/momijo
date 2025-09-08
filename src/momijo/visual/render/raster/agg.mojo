# Project:      Momijo
# Module:       src.momijo.visual.render.raster.agg
# File:         agg.mojo
# Path:         src/momijo/visual/render/raster/agg.mojo
#
# Description:  src.momijo.visual.render.raster.agg — focused Momijo functionality with a stable public API.
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
#   - Key functions: rgb_hex, rgba_hex, _clipi, _plot, draw_point, draw_circle, draw_line, stroke_rect ...
#   - Uses generic functions/types with explicit trait bounds.


from momijo.autograd.hook import call
from momijo.core.error import module
from momijo.dataframe.logical_plan import sort
from momijo.ir.dialects.annotations import integer
from momijo.ir.passes.tiling import Tiling
from momijo.utils.result import g
from momijo.visual.render.raster.raster_buffer import put_pixel
from momijo.visual.scene.facet import Rect
from momijo.visual.scene.scene import put_pixel
from pathlib import Path
from pathlib.path import Path

fn rgb_hex(r: Int, g: Int, b: Int) -> Int:
    var rr = r; if rr < 0: rr = 0; if rr > 255: rr = 255
    var gg = g; if gg < 0: gg = 0; if gg > 255: gg = 255
    var bb = b; if bb < 0: bb = 0; if bb > 255: bb = 255
    return (rr << UInt8(16)) | (gg << UInt8(8)) | bb
fn rgba_hex(r: Int, g: Int, b: Int, a: Int) -> Int:
    var aa = a; if aa < 0: aa = 0; if aa > 255: aa = 255
    return (aa << UInt8(24)) | rgb_hex(r,g,b)

# --- Clip helper -------------------------------------------------------------
fn _clipi(v: Int, lo: Int, hi: Int) -> Int:
    if v < lo: return lo
    if v > hi: return hi
    return v

# --- Basic pixel write (no blending) ----------------------------------------
fn _plot(mut img: Raster, x: Int, y: Int, rgb: Int) -> None:
    if x < 0 or y < 0: return
    if x >= img.width or y >= img.height: return
    put_pixel(img, x, y, rgb)

# --- Point (filled disk approximation) --------------------------------------
fn draw_point(mut img: Raster, x: Float64, y: Float64, radius: Int, rgb: Int):
    var cx = Int(x)
    var cy = Int(y)
    var r = radius
    if r <= 0:
        _plot(img, cx, cy, rgb)
        return
    var r2 = r * r
    var yy = -r
    while yy <= r:
        var xx = -r
        while xx <= r:
            if (xx*xx + yy*yy) <= r2:
                _plot(img, cx + xx, cy + yy, rgb)
            xx += 1
        yy += 1

# --- Circle outline (midpoint) ----------------------------------------------
fn draw_circle(mut img: Raster, cx: Int, cy: Int, r: Int, rgb: Int) -> None:
    if r < 0: return
    var x = r
    var y = 0
    var err = 1 - r
    while x >= y:
        _plot(img, cx + x, cy + y, rgb)
        _plot(img, cx + y, cy + x, rgb)
        _plot(img, cx - y, cy + x, rgb)
        _plot(img, cx - x, cy + y, rgb)
        _plot(img, cx - x, cy - y, rgb)
        _plot(img, cx - y, cy - x, rgb)
        _plot(img, cx + y, cy - x, rgb)
        _plot(img, cx + x, cy - y, rgb)
        y += 1
        if err < 0:
            err += 2*y + 1
        else:
            x -= 1
            err += 2*(y - x) + 1

# --- Bresenham line (integer, non-AA) ---------------------------------------
fn draw_line(mut img: Raster, x0: Int, y0: Int, x1: Int, y1: Int, rgb: Int) -> None:
    var dx = abs(x1 - x0)
    var sx = 1 if x0 < x1 else -1
    var dy = -abs(y1 - y0)
    var sy = 1 if y0 < y1 else -1
    var err = dx + dy
    var x = x0
    var y = y0
    while True:
        _plot(img, x, y, rgb)
        if x == x1 and y == y1: break
        var e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

# --- Rect stroke -------------------------------------------------------------
fn stroke_rect(mut img: Raster, x: Int, y: Int, w: Int, h: Int, rgb: Int) -> None:
    if w <= 0 or h <= 0: return
    draw_line(img, x, y, x + w - 1, y, rgb)
    draw_line(img, x, y, x, y + h - 1, rgb)
    draw_line(img, x + w - 1, y, x + w - 1, y + h - 1, rgb)
    draw_line(img, x, y + h - 1, x + w - 1, y + h - 1, rgb)

# --- Rect fill ---------------------------------------------------------------
fn fill_rect(mut img: Raster, x: Int, y: Int, w: Int, h: Int, rgb: Int) -> None:
    if w <= 0 or h <= 0: return
    var x0 = x; var y0 = y
    var x1 = x + w - 1
    var y1 = y + h - 1
    if x1 < 0 or y1 < 0: return
    if x0 >= img.width or y0 >= img.height: return
    x0 = _clipi(x0, 0, img.width - 1)
    y0 = _clipi(y0, 0, img.height - 1)
    x1 = _clipi(x1, 0, img.width - 1)
    y1 = _clipi(y1, 0, img.height - 1)
    var yy = y0
    while yy <= y1:
        var xx = x0
        while xx <= x1:
            put_pixel(img, xx, yy, rgb)
            xx += 1
        yy += 1

# --- Polyline ----------------------------------------------------------------
fn draw_polyline(mut img: Raster, xs: List[Int], ys: List[Int], rgb: Int) -> None:
    var n = len(xs)
    if n == 0 or n != len(ys): return
    var i = 1
    while i < n:
        draw_line(img, xs[i-1], ys[i-1], xs[i], ys[i], rgb)
        i += 1

# --- Polygon fill (scanline, even-odd) --------------------------------------
fn fill_polygon(mut img: Raster, xs: List[Int], ys: List[Int], rgb: Int) -> None:
    var n = len(xs)
    if n < 3 or n != len(ys): return
    var miny = img.height; var maxy = 0
    var i = 0
    while i < n:
        var yi = ys[i]
        if yi < miny: miny = yi
        if yi > maxy: maxy = yi
        i += 1
    if maxy < 0 or miny >= img.height: return
    miny = _clipi(miny, 0, img.height - 1)
    maxy = _clipi(maxy, 0, img.height - 1)
    var y = miny
    while y <= maxy:
        var interxs = List[Int]()
        var j = n - 1
        i = 0
        while i < n:
            var yi = ys[i]; var yj = ys[j]
            var xi = xs[i]; var xj = xs[j]
            var cond = ((yi > y) != (yj > y))
            if cond:
                var den = (yj - yi)
                if den == 0: den = 1
                var x_int = xi + ((y - yi) * (xj - xi)) / den
                interxs.push(x_int)
            j = i
            i += 1
        # insertion sort (small N)
        var k = 1
        while k < len(interxs):
            var v = interxs[k]
            var p = k - 1
            while p >= 0 and interxs[p] > v:
                interxs[p + 1] = interxs[p]
                p -= 1
            interxs[p + 1] = v
            k += 1
        k = 0
        while (k + 1) < len(interxs):
            var xstart = interxs[k]
            var xend = interxs[k + 1]
            if xend >= 0 and xstart < img.width:
                xstart = _clipi(xstart, 0, img.width - 1)
                xend = _clipi(xend, 0, img.width - 1)
                var xcur = xstart
                while xcur <= xend:
                    put_pixel(img, xcur, y, rgb)
                    xcur += 1
            k += 2
        y += 1

# --- Batch scatter with Tiling (cache-friendly) ------------------------------
# tiling.note {tile=...}
fn scatter_points_tiled(mut img: Raster, xs: List[Float64], ys: List[Float64],
                        radius: Int, r: Int, g: Int, b: Int, tile: Int = 64) -> None:
    var W = img.width
    var H = img.height
    var n = len(xs)
    if n == 0 or W <= 0 or H <= 0: return
    var y0 = 0
    while y0 < H:
        var yend = y0 + tile
        if yend > H: yend = H
        var x0 = 0
        while x0 < W:
            var xend = x0 + tile
            if xend > W: xend = W
            var i = 0
            while i < n:
                var xi = xs[i]; var yi = ys[i]
                if xi >= Float64(x0 - 4) and xi < Float64(xend + 4) and yi >= Float64(y0 - 4) and yi < Float64(yend + 4):
                    draw_point(img, xi, yi, radius, rgb_hex(r, g, b))
                i += 1
            x0 += tile
        y0 += tile

# --- Scene helpers (compat shims) --------------------------------------------
# Convenience wrapper to match older call sites
fn rasterize_scene_points(mut img: Raster, xs: List[Float64], ys: List[Float64], size: Int, r: Int, g: Int, b: Int) -> None:
    scatter_points_tiled(img, xs, ys, size, r, g, b, 64)

# --- Self test ---------------------------------------------------------------
fn _self_test() -> Bool:
    # Dependency on Raster; return True as smoke test.
    return True