# ============================================================================
#  Momijo Visualization - render/raster/simd.mojo
#  (c) 2025  MIT
#  NOTE: Mojo scaffolding for FFI/SIMD; adapt to your toolchain.
# ============================================================================

from momijo.visual.render.raster.raster_buffer import Raster

fn fill_rect_simd(mut img: Raster, x: Int, y: Int, w: Int, h: Int, rgb: Int):
    var yy = 0
    while yy < h:
        var base = (y + yy) * img.width + x
        var i = 0
        # chunked write of 8 pixels per iteration
        while i + 8 <= w:
            img.data[base + i + 0] = rgb
            img.data[base + i + 1] = rgb
            img.data[base + i + 2] = rgb
            img.data[base + i + 3] = rgb
            img.data[base + i + 4] = rgb
            img.data[base + i + 5] = rgb
            img.data[base + i + 6] = rgb
            img.data[base + i + 7] = rgb
            i += 8
        while i < w:
            img.data[base + i] = rgb
            i += 1
        yy += 1
