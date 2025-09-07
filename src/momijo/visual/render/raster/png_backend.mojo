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
# File: src/momijo/visual/render/raster/png_backend.mojo

from momijo.core.device import kind, unknown
from momijo.core.error import module
from momijo.dataframe.helpers import m
from momijo.nn.parameter import data
from momijo.utils.result import g
from momijo.visual.ffi.libpng_c import write_png_rgba8
from momijo.visual.render.raster.agg import draw_line, draw_point, rgb_hex
from momijo.visual.render.raster.raster_buffer import Raster, put_pixel, write_ppm
from momijo.visual.runtime.backend_select import BackendKinds, png
from momijo.visual.scene.facet import Rect
from momijo.visual.scene.scene import BackendKinds, Scene, SceneMarkKinds
from pathlib import Path
from pathlib.path import Path

fn _raster_to_rgba8(img: Raster) -> (List[UInt8], Int):
    var w = img.width
    var h = img.height
    var out = List[UInt8]()
    out.reserve(w * h * 4)
    var y = 0
    while y < h:
        var x = 0
        while x < w:
            var v = img.data[y * w + x]
            var r = (v >> UInt8(16)) & 255
            var g = (v >> UInt8(8)) & 255
            var b = v & UInt8(255)
            out.push(UInt8(r))
            out.push(UInt8(g))
            out.push(UInt8(b))
            out.push(UInt8(255))  # opaque
            x += 1
        y += 1
    return (out, w * 4)

# Save Raster as PNG (RGBA8)
fn write_png_from_raster(img: Raster, path: String) -> Bool:
    var (buf, stride) = _raster_to_rgba8(img)
    return write_png_rgba8(path, img.width, img.height, stride, buf)

fn render_scene_raster(scene: Scene, backend: BackendKind, path: String) -> None:
    var img = Raster(scene.width, scene.height)

    # Fill background to white
    var y = 0
    while y < img.height:
        var x = 0
        while x < img.width:
            put_pixel(img, x, y, rgb_hex(255,255,255))
            x += 1
        y += 1

    # Draw marks if available (points & lines). Defensive: tolerate missing kinds.
    var i = 0
    while i < len(scene.marks):
        var m = scene.marks[i]
        # Point marks
        assert(kind is not None, String("kind is None"))
        if m.kind.value() == SceneMarkKinds.point().value():
            var j = 0
            var rgb = rgb_hex(m.color.r, m.color.g, m.color.b)
            while j < len(m.points):
                draw_point(img, m.points[j].x, m.points[j].y, Int(m.size), rgb)
                j += 1
        # Line marks
        assert(kind is not None, String("kind is None"))
        elif m.kind.value() == SceneMarkKinds.line().value():
            var j = 1
            var rgb = rgb_hex(m.color.r, m.color.g, m.color.b)
            while j < len(m.points):
                var x0 = Int(m.points[j-1].x); var y0 = Int(m.points[j-1].y)
                var x1 = Int(m.points[j].x);   var y1 = Int(m.points[j].y)
                draw_line(img, x0, y0, x1, y1, rgb)
                j += 1
        # Rect fill (optional if defined)
        assert(kind is not None, String("kind is None"))
        elif m.kind.value() == SceneMarkKinds.rect().value():
            var x0 = Int(m.points[0].x); var y0 = Int(m.points[0].y)
            var x1 = Int(m.points[1].x); var y1 = Int(m.points[1].y)
            var yy = y0
            var rgb = rgb_hex(m.color.r, m.color.g, m.color.b)
            while yy <= y1:
                var xx = x0
                while xx <= x1:
                    put_pixel(img, xx, yy, rgb)
                    xx += 1
                yy += 1
        else:
            # silently ignore unknown mark kinds in MVP
            pass
        i += 1

    # Write outputs: always PNG; optionally a PPM for debugging.
    assert(backend is not None, String("backend is None"))
    if backend.value() == BackendKinds.png().value():
        var ok = write_png_from_raster(img, path)
        if not ok:
            # Fallback to PPM if PNG failed
            write_ppm(img, path + String(".ppm"))
    else:
        # If not PNG, just write PPM as a basic artifact
        write_ppm(img, path + String(".ppm"))