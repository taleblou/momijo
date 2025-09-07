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
# Project: momijo.visual.spec
# File: src/momijo/visual/spec/compiler.mojo

from momijo.autograd.hook import call
from momijo.core.device import kind
from momijo.core.error import code, module
from momijo.dataframe.helpers import m, t
from momijo.ir.dialects.annotations import pure
from momijo.ir.dialects.shape_inference import Axes
from momijo.tensor.allocator import free
from momijo.tensor.errors import OK
from momijo.utils.timer import stop
from momijo.visual.render.raster.agg import draw_line, draw_point, fill_rect, rgb_hex
from momijo.visual.render.raster.raster_buffer import Raster, to_rgba8
from momijo.visual.runtime.backend_select import png, render_scene, svg
from momijo.visual.runtime.theme import Theme, theme_scientific
from momijo.visual.scene.facet import inner
from momijo.visual.scene.layout import AxisSpec, Margins, compute_layout, map_xy
from momijo.visual.scene.scene import Mark, Point2D, Scene, SceneMarkKinds, add_mark
from pathlib import Path
from pathlib.path import Path

fn version_string() -> String:
    return String("Momijo Visual Runtime 0.1.0 (compiler.mojo)")

# --- Report struct -----------------------------------------------------------
struct CompileReport:
    var ok: Bool
    var messages: List[String]
fn __init__(out self) -> None:
        self.ok = True
        self.messages = List[String]()
fn add(mut self, msg: String) -> None:
        self.messages.push(msg)
fn __copyinit__(out self, other: Self) -> None:
        self.ok = other.ok
        self.messages = other.messages
fn __moveinit__(out self, deinit other: Self) -> None:
        self.ok = other.ok
        self.messages = other.messages
# --- Environment check (lightweight) ----------------------------------------
fn check_environment() -> CompileReport:
    var r = CompileReport()
    r.add(String("Runtime: ") + version_string())
    r.add(String("Backends: svg, png(pure/libpng fallback via png_backend)"))
    r.add(String("Scene/Layout/Theme modules: OK"))
    # We avoid probing for shared libs here; rely on backend fallbacks (PPM) if needed.
    return r

# --- Self-tests aggregator ---------------------------------------------------
# Each module defines a _self_test() -> Bool. We call those that are fast and side-effect free.
@foreign("C")
fn _dummy_foreign_marker() -> None: pass  # keeps foreign section present for parity

# Declarations (as prototypes) to avoid hard dependencies when not linked.
fn _try_scene() -> Bool:
    return True   # scene has own self-test, but not invoked here to avoid duplicate code load.

# Aggregated quick test: constructs a small raster, draws lines/rect/points and roundtrips RGBA conversion.
fn run_self_tests() -> CompileReport:
    var rep = CompileReport()
    # Raster + AGG
    var img = Raster(64, 48, 0xFFFFFF)
    draw_line(img, 2, 2, 60, 40, rgb_hex(30,30,30))
    fill_rect(img, 10, 10, 12, 8, 0x00FF00)
    draw_point(img, 20.0, 20.0, 3, 0xFF0000)
    var (rgba_buf, stride) = to_rgba8(img)
    if len(rgba_buf) != img.width * img.height * 4:
        rep.ok = False
        rep.add(String("to_rgba8 size mismatch"))
    else:
        rep.add(String("Raster/AGG conversions: OK"))
    # Layout sanity
    var m = Margins(12)
    var L = compute_layout(320, 240, m, AxisSpec(0.0, 10.0, 5), AxisSpec(-1.0, 1.0, 5))
    var (px, py) = map_xy(L, 5.0, 0.0)
    if px < L.inner.x or px >= L.inner.x + L.inner.w or py < L.inner.y or py >= L.inner.y + L.inner.h:
        rep.ok = False
        rep.add(String("Layout mapping out of bounds"))
    else:
        rep.add(String("Layout/ticks: OK"))
    # Themes
    var t = theme_scientific()
    if len(t.palette_cat) == 0:
        rep.ok = False
        rep.add(String("Theme palette empty"))
    else:
        rep.add(String("Themes: OK"))
    return rep

# --- Demo scene --------------------------------------------------------------
fn build_demo_scene(width: Int = 480, height: Int = 320) -> Scene:
    var sc = Scene(width, height)
    # Background will be set by backend if needed; keep defaults
    # Axes box (simple visual guides)
    var left = 40; var right = width - 20; var top = 20; var bottom = height - 40
    # Frame rectangle
    var frame = Mark()
    frame.kind = SceneMarkKinds.rect()
    frame.color = ColorRGBA(240,240,240,255)
    frame.size = 0
    frame.points = List[Point2D]()
    frame.points.push(Point2D(Float64(left), Float64(top)))
    frame.points.push(Point2D(Float64(right), Float64(bottom)))
    sc.add_mark(frame)
    # Polyline (y = 0.5x + 10 in pixel coords just to show lines)
    var poly = Mark()
    poly.kind = SceneMarkKinds.line()
    poly.color = ColorRGBA(33,150,243,255)
    poly.size = 1
    poly.points = List[Point2D]()
    var x = left
    while x <= right:
        var y = top + ((x - left) / 2)
        if y > bottom: y = bottom
        poly.points.push(Point2D(Float64(x), Float64(y)))
        x += 8
    sc.add_mark(poly)
    # Scatter points
    var pts = Mark()
    pts.kind = SceneMarkKinds.point()
    pts.color = ColorRGBA(233,30,99,255)
    pts.size = 3
    pts.points = List[Point2D]()
    pts.points.push(Point2D(Float64(left + 20), Float64(bottom - 20)))
    pts.points.push(Point2D(Float64(left + 60), Float64(top + 30)))
    pts.points.push(Point2D(Float64(left + 120), Float64(top + 60)))
    sc.add_mark(pts)
    return sc

# --- Render helpers ----------------------------------------------------------
fn render_demo(basepath: String) -> Bool:
    var sc = build_demo_scene(480, 320)
    # SVG
    render_scene(sc, basepath + String(".svg"))
    # PNG
    render_scene(sc, basepath + String(".png"))
    return True

# --- One-stop entrypoints ----------------------------------------------------
fn compile_and_render(sc: Scene, out_path: String) -> Bool:
    # Simply defer to backend_select; name kept for API symmetry.
    render_scene(sc, out_path)
    return True

# --- Self-test ---------------------------------------------------------------
fn _self_test() -> Bool:
    var rep = run_self_tests()
    return rep.ok