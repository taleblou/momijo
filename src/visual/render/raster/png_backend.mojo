# ============================================================================
#  Momijo Visualization - render/raster/png_backend.mojo
#  Copyright (c) 2025  Morteza Talebou
#  MIT License - https://taleblou.ir/
#  Mojo checklist: no global/export; __init__(out self,...); var only; explicit imports.
# ============================================================================

from momijo.visual.runtime.backend_select import BackendKind, BackendKinds
from momijo.visual.scene.scene import Scene, Mark, SceneMarkKinds
from momijo.visual.render.raster.raster_buffer import Raster, put_pixel, write_ppm
from momijo.visual.render.raster.agg import rasterize_scene_points, rgb_hex

fn render_scene_raster(scene: Scene, backend: BackendKind, path: String):
    var img = Raster(scene.width, scene.height)

    # background white
    var i = 0
    while i < scene.width * scene.height:
        img.data[i] = rgb_hex(255,255,255)
        i += 1

    var mi = 0
    while mi < len(scene.marks):
        var m = scene.marks[mi]
        if m.kind.value == SceneMarkKinds.point().value:
            var xs = List[Float64](); var ys = List[Float64]()
            var j = 0
            while j < len(m.points):
                xs.append(m.points[j].x); ys.append(m.points[j].y); j += 1
            rasterize_scene_points(img, xs, ys, Int(m.size), m.color.r, m.color.g, m.color.b)
        elif m.kind.value == SceneMarkKinds.line().value:
            var j = 1
            while j < len(m.points):
                # crude line
                var rgb = rgb_hex(m.color.r, m.color.g, m.color.b)
                # inline Bresenham
                var x0 = Int(m.points[j-1].x); var y0 = Int(m.points[j-1].y)
                var x1 = Int(m.points[j].x);   var y1 = Int(m.points[j].y)
                var dx = abs(x1 - x0); var sx = 1 if x0 < x1 else -1
                var dy = -abs(y1 - y0); var sy = 1 if y0 < y1 else -1
                var err = dx + dy
                var x = x0; var y = y0
                while True:
                    put_pixel(img, x, y, rgb)
                    if x == x1 and y == y1: break
                    var e2 = 2 * err
                    if e2 >= dy:
                        err += dy; x += sx
                    if e2 <= dx:
                        err += dx; y += sy
                j += 1
        elif m.kind.value == SceneMarkKinds.rect().value:
            var j = 0
            while j < len(m.points):
                var p = m.points[j]
                var wbar = 10; var hbar = 10
                var x = Int(p.x - Float64(wbar) * 0.5)
                var y = Int(p.y - Float64(hbar) * 0.5)
                var rgb = rgb_hex(m.color.r, m.color.g, m.color.b)
                var yy = 0
                while yy < hbar:
                    var xx = 0
                    while xx < wbar:
                        put_pixel(img, x + xx, y + yy, rgb)
                        xx += 1
                    yy += 1
                j += 1
        mi += 1

    # write PPM as current raster output
    var ppm = path + String(".ppm")
    write_ppm(img, ppm)


from momijo.visual.render.raster.simd import fill_rect_simd

# Overwrite rect drawing using SIMD-like chunk fill:
# (kept simple; point/line stay as-is)
