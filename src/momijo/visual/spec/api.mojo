# Project:      Momijo
# Module:       src.momijo.visual.spec.api
# File:         api.mojo
# Path:         src/momijo/visual/spec/api.mojo
#
# Description:  src.momijo.visual.spec.api — focused Momijo functionality with a stable public API.
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
#   - Structs: Figure
#   - Key functions: _build_points, _add_grid, __init__, scatter, line, rect, save, __copyinit__ ...
#   - Uses generic functions/types with explicit trait bounds.
#   - GPU/device utilities present; validate backend assumptions.


from momijo.core.device import kind
from momijo.core.error import module
from momijo.dataframe.helpers import m
from momijo.nn.parameter import data
from momijo.utils.result import g
from momijo.visual.runtime.backend_select import png, render_scene, svg
from momijo.visual.runtime.theme import Theme, theme_color_at, theme_dark, theme_publisher, theme_scientific
from momijo.visual.scene.facet import inner
from momijo.visual.scene.layout import AxisSpec, Margins, PlotLayout, compute_layout, grid_x_pixels, grid_y_pixels, map_xy
from momijo.visual.scene.scene import Mark, Point2D, Scene, SceneMarkKinds, add_mark
from pathlib import Path
from pathlib.path import Path

fn _build_points(L: PlotLayout, xs: List[Float64], ys: List[Float64]) -> List[Point2D]:
    var out = List[Point2D]()
    var n = len(xs)
    if n == 0 or n != len(ys): return out
    var i = 0
    while i < n:
        var (px, py) = map_xy(L, xs[i], ys[i])
        out.push(Point2D(Float64(px), Float64(py)))
        i += 1
    return out

# --- Grid as scene marks (optional) ------------------------------------------
# Adds light grid lines inside the inner plot rect based on layout ticks.
fn _add_grid(mut sc: Scene, L: PlotLayout, color: ColorRGBA) -> None:
    # vertical lines
    var xs = grid_x_pixels(L)
    var i = 0
    while i < len(xs):
        var m = Mark()
        m.kind = SceneMarkKinds.line()
        m.color = color
        m.size = 1
        m.points = List[Point2D]()
        m.points.push(Point2D(Float64(xs[i]), Float64(L.inner.y)))
        m.points.push(Point2D(Float64(xs[i]), Float64(L.inner.y + L.inner.h - 1)))
        sc.add_mark(m)
        i += 1
    # horizontal lines
    var ys = grid_y_pixels(L)
    i = 0
    while i < len(ys):
        var m2 = Mark()
        m2.kind = SceneMarkKinds.line()
        m2.color = color
        m2.size = 1
        m2.points = List[Point2D]()
        m2.points.push(Point2D(Float64(L.inner.x), Float64(ys[i])))
        m2.points.push(Point2D(Float64(L.inner.x + L.inner.w - 1), Float64(ys[i])))
        sc.add_mark(m2)
        i += 1

# --- Figure ------------------------------------------------------------------
struct Figure:
    var scene: Scene
    var layout: PlotLayout
    var theme: Theme
fn __init__(out self,
                width: Int = 640, height: Int = 480,
                x_min: Float64 = 0.0, x_max: Float64 = 1.0,
                y_min: Float64 = 0.0, y_max: Float64 = 1.0,
                desired_xticks: Int = 5, desired_yticks: Int = 5,
                margins_px: Int = 40,
                theme_name: Int = 0,    # 0: scientific, 1: dark, 2: publisher
                draw_grid: Bool = True) -> None:
        # Theme
        var th = theme_scientific()
        if theme_name == 1: th = theme_dark()
        elif theme_name == 2: th = theme_publisher()
        self.theme = th

        # Scene
        self.scene = Scene(width, height)
        self.scene.background = th.background

        # Layout
        var m = Margins(margins_px)
        var xspec = AxisSpec(x_min, x_max, desired_xticks)
        var yspec = AxisSpec(y_min, y_max, desired_yticks)
        self.layout = compute_layout(width, height, m, xspec, yspec)

        # Optional grid (behind data): light alpha from theme.grid
        if draw_grid:
            var gcol = ColorRGBA(th.grid.r, th.grid.g, th.grid.b, th.grid.a)
            _add_grid(self.scene, self.layout, gcol)

    # Scatter series (data coords)
fn scatter(mut self, xs: List[Float64], ys: List[Float64], size_px: Int = 2, series_idx: Int = 0) -> None:
        var pts = _build_points(self.layout, xs, ys)
        var m = Mark()
        m.kind = SceneMarkKinds.point()
        m.size = size_px
        m.color = theme_color_at(self.theme, series_idx)
        m.points = pts
        self.scene.add_mark(m)

    # Line series (data coords, connected in order)
fn line(mut self, xs: List[Float64], ys: List[Float64], stroke_px: Int = 1, series_idx: Int = 0) -> None:
        var pts = _build_points(self.layout, xs, ys)
        var m = Mark()
        m.kind = SceneMarkKinds.line()
        m.size = stroke_px
        m.color = theme_color_at(self.theme, series_idx)
        m.points = pts
        self.scene.add_mark(m)

    # Rectangle in data space (corners)
fn rect(mut self, x0: Float64, y0: Float64, x1: Float64, y1: Float64, series_idx: Int = 0) -> None:
        var (px0, py0) = map_xy(self.layout, x0, y0)
        var (px1, py1) = map_xy(self.layout, x1, y1)
        var m = Mark()
        m.kind = SceneMarkKinds.rect()
        m.color = theme_color_at(self.theme, series_idx)
        m.size = 0
        m.points = List[Point2D]()
        m.points.push(Point2D(Float64(px0), Float64(py0)))
        m.points.push(Point2D(Float64(px1), Float64(py1)))
        self.scene.add_mark(m)

    # Save via backend selection (by extension: .svg / .png)
fn save(self, path: String) -> None:
        render_scene(self.scene, path)
fn __copyinit__(out self, other: Self) -> None:
        self.scene = other.scene
        self.layout = other.layout
        self.theme = other.theme
fn __moveinit__(out self, deinit other: Self) -> None:
        self.scene = other.scene
        self.layout = other.layout
        self.theme = other.theme
# --- One-shot convenience ----------------------------------------------------
# Make a quick scatter plot and save. Returns True on success (best-effort).
fn quick_scatter(xs: List[Float64], ys: List[Float64], path: String,
                 width: Int = 640, height: Int = 480,
                 x_min: Float64 = 0.0, x_max: Float64 = 1.0,
                 y_min: Float64 = 0.0, y_max: Float64 = 1.0,
                 theme_name: Int = 0) -> Bool:
    var fig = Figure(width, height, x_min, x_max, y_min, y_max, 5, 5, 40, theme_name, True)
    fig.scatter(xs, ys, 2, 0)
    fig.save(path)
    return True

# --- Self-test ---------------------------------------------------------------
fn _self_test() -> Bool:
    var xs = List[Float64](); var ys = List[Float64]()
    xs.push(0.0); ys.push(0.0)
    xs.push(0.5); ys.push(0.5)
    xs.push(1.0); ys.push(1.0)
    var fig = Figure(320, 240, 0.0, 1.0, 0.0, 1.0, 3, 3, 24, 0, True)
    fig.line(xs, ys, 1, 0)
    # Not writing files in self-test; just verify marks exist
    return len(fig.scene.marks) >= 1