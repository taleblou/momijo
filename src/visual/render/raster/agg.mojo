# ============================================================================
#  Momijo Visualization - render/raster/agg.mojo
#  Copyright (c) 2025  Morteza Talebou
#  MIT License - https://taleblou.ir/
#  Mojo checklist: no global/export; __init__(out self,...); var only; explicit imports.
# ============================================================================

from momijo.visual.render.raster.raster_buffer import Raster, put_pixel

fn _clip(v: Int, lo: Int, hi: Int) -> Int:
    if v < lo: return lo
    if v > hi: return hi
    return v

fn draw_point(mut img: Raster, x: Float64, y: Float64, radius: Int, rgb: Int):
    var cx = Int(x); var cy = Int(y)
    var r2 = radius * radius
    var dy = -radius
    while dy <= radius:
        var dx = -radius
        while dx <= radius:
            if dx*dx + dy*dy <= r2:
                put_pixel(img, cx + dx, cy + dy, rgb)
            dx += 1
        dy += 1

fn draw_line(mut img: Raster, x0: Float64, y0: Float64, x1: Float64, y1: Float64, rgb: Int):
    # Bresenham-like
    var xi0 = Int(x0); var yi0 = Int(y0); var xi1 = Int(x1); var yi1 = Int(y1)
    var dx = abs(xi1 - xi0); var sx = 1 if xi0 < xi1 else -1
    var dy = -abs(yi1 - yi0); var sy = 1 if yi0 < yi1 else -1
    var err = dx + dy
    var x = xi0; var y = yi0
    while True:
        put_pixel(img, x, y, rgb)
        if x == xi1 and y == yi1: break
        var e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

fn draw_rect(mut img: Raster, x: Float64, y: Float64, w: Int, h: Int, rgb: Int):
    var xi = Int(x); var yi = Int(y)
    var j = 0
    while j < h:
        var i = 0
        while i < w:
            put_pixel(img, xi + i, yi + j, rgb)
            i += 1
        j += 1

fn rgb_hex(r: Int, g: Int, b: Int) -> Int:
    return ((r & 255) << 16) | ((g & 255) << 8) | (b & 255)

fn rasterize_scene_points(mut img: Raster, xs: List[Float64], ys: List[Float64], radius: Int, r: Int, g: Int, b: Int):
    var tile = 64
    var H = img.height
    var W = img.width
    var n = len(xs)
    var y0 = 0
    while y0 < H:
        var yend = y0 + tile
        if yend > H: yend = H
        var x0 = 0
        while x0 < W:
            var xend = x0 + tile
            if xend > W: xend = W
            # process all points (no spatial index in MVP)
            var i = 0
            while i < n:
                var xi = xs[i]; var yi = ys[i]
                if xi >= x0-4 and xi < xend+4 and yi >= y0-4 and yi < yend+4:
                    draw_point(img, xi, yi, radius, rgb_hex(r,g,b))
                i += 1
            x0 += tile
        y0 += tile
