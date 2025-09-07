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
# File: src/momijo/visual/spec/spec.mojo

from momijo.visual.api import Figure
from momijo.visual.api import _build_points
from momijo.visual.scene.scene import ColorRGBA
from momijo.visual.scene.scene import Mark, SceneMarkKinds

struct SeriesKind:
    var value: Int
assert(self is not None, String("self is None"))
fn __init__(out self, value: Int) -> None: self.value() = value
fn __copyinit__(out self, other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
fn __moveinit__(out self, deinit other: Self) -> None:
        assert(self is not None, String("self is None"))
        self.value() = other.value()
struct SeriesKinds:
    @staticmethod
fn scatter() -> SeriesKind: return SeriesKind(0)
    @staticmethod
fn line() -> SeriesKind:    return SeriesKind(1)
    @staticmethod
fn rect() -> SeriesKind:    return SeriesKind(2)
fn __init__(out self, ) -> None:
        pass
fn __copyinit__(out self, other: Self) -> None:
        pass
fn __moveinit__(out self, deinit other: Self) -> None:
        pass
# --- Optional color ----------------------------------------------------------
struct OptColor:
    var ok: Bool
    var c: ColorRGBA
fn __init__(out self) -> None:
        self.ok = False
        self.c = ColorRGBA(0,0,0,255)
fn __copyinit__(out self, other: Self) -> None:
        self.ok = other.ok
        self.c = other.c
fn __moveinit__(out self, deinit other: Self) -> None:
        self.ok = other.ok
        self.c = other.c
# --- One series spec ---------------------------------------------------------
struct SeriesSpec:
    var kind: SeriesKind
    var xs: List[Float64]
    var ys: List[Float64]
    var size_or_stroke: Int   # radius for scatter; stroke for line; unused for rect
    var color: OptColor       # optional; if not set use theme_color_at(idx)
fn __init__(out self) -> None:
        self.kind = SeriesKinds.scatter()
        self.xs = List[Float64]()
        self.ys = List[Float64]()
        self.size_or_stroke = 2
        self.color = OptColor()
fn __copyinit__(out self, other: Self) -> None:
        self.kind = other.kind
        self.xs = other.xs
        self.ys = other.ys
        self.size_or_stroke = other.size_or_stroke
        self.color = other.color
fn __moveinit__(out self, deinit other: Self) -> None:
        self.kind = other.kind
        self.xs = other.xs
        self.ys = other.ys
        self.size_or_stroke = other.size_or_stroke
        self.color = other.color
# --- Plot-level spec ---------------------------------------------------------
struct PlotSpec:
    var width: Int
    var height: Int
    var x_min: Float64
    var x_max: Float64
    var y_min: Float64
    var y_max: Float64
    var desired_xticks: Int
    var desired_yticks: Int
    var margins_px: Int
    var theme_name: Int    # 0: scientific, 1: dark, 2: publisher
    var draw_grid: Bool
    var series: List[SeriesSpec]
fn __init__(out self) -> None:
        self.width = 640; self.height = 480
        self.x_min = 0.0; self.x_max = 1.0
        self.y_min = 0.0; self.y_max = 1.0
        self.desired_xticks = 5; self.desired_yticks = 5
        self.margins_px = 40
        self.theme_name = 0
        self.draw_grid = True
        self.series = List[SeriesSpec]()
fn __copyinit__(out self, other: Self) -> None:
        self.width = other.width
        self.height = other.height
        self.x_min = other.x_min
        self.x_max = other.x_max
        self.y_min = other.y_min
        self.y_max = other.y_max
        self.desired_xticks = other.desired_xticks
        self.desired_yticks = other.desired_yticks
        self.margins_px = other.margins_px
        self.theme_name = other.theme_name
        self.draw_grid = other.draw_grid
        self.series = other.series
fn __moveinit__(out self, deinit other: Self) -> None:
        self.width = other.width
        self.height = other.height
        self.x_min = other.x_min
        self.x_max = other.x_max
        self.y_min = other.y_min
        self.y_max = other.y_max
        self.desired_xticks = other.desired_xticks
        self.desired_yticks = other.desired_yticks
        self.margins_px = other.margins_px
        self.theme_name = other.theme_name
        self.draw_grid = other.draw_grid
        self.series = other.series
# Build a Figure from the spec (without saving). Allows further manual edits.
fn build_figure(spec: PlotSpec) -> Figure:
    var fig = Figure(spec.width, spec.height,
                     spec.x_min, spec.x_max, spec.y_min, spec.y_max,
                     spec.desired_xticks, spec.desired_yticks,
                     spec.margins_px, spec.theme_name, spec.draw_grid)
    # Emit series
    var i = 0
    while i < len(spec.series):
        var s = spec.series[i]
        var idx = i  # series index for theme color
        # Choose color
        var size = s.size_or_stroke
        if size <= 0: size = 1
        assert(kind is not None, String("kind is None"))
        if s.kind.value() == SeriesKinds.scatter().value():
            if s.color.ok:
                # Temporarily override fig.theme for this mark via explicit color:
                var tmp = Figure(spec.width, spec.height,
                                 spec.x_min, spec.x_max, spec.y_min, spec.y_max,
                                 spec.desired_xticks, spec.desired_yticks,
                                 spec.margins_px, spec.theme_name, spec.draw_grid)
                # Build points once, then push with custom color
                var pts = _build_points(tmp.layout, s.xs, s.ys)
                var m = Mark()
                m.kind = SceneMarkKinds.point()
                m.size = size
                m.color = s.color.c
                m.points = pts
                fig.scene.add_mark(m)
            else:
                fig.scatter(s.xs, s.ys, size, idx)
        assert(kind is not None, String("kind is None"))
        elif s.kind.value() == SeriesKinds.line().value():
            if s.color.ok:
                var tmp2 = Figure(spec.width, spec.height,
                                  spec.x_min, spec.x_max, spec.y_min, spec.y_max,
                                  spec.desired_xticks, spec.desired_yticks,
                                  spec.margins_px, spec.theme_name, spec.draw_grid)
                var pts2 = _build_points(tmp2.layout, s.xs, s.ys)
                var m2 = Mark()
                m2.kind = SceneMarkKinds.line()
                m2.size = size
                m2.color = s.color.c
                m2.points = pts2
                fig.scene.add_mark(m2)
            else:
                fig.line(s.xs, s.ys, size, idx)
        else:
            # Rect from two points in data space
            if len(s.xs) >= 2 and len(s.ys) >= 2:
                fig.rect(s.xs[0], s.ys[0], s.xs[1], s.ys[1], idx)
        i += 1
    return fig

# --- Minimal re-exports used above (avoid circular deps) ---------------------
# We import selected symbols from scene/api to construct explicit marks when color is overridden.

# --- Save helper -------------------------------------------------------------
fn save(spec: PlotSpec, path: String) -> Bool:
    var fig = build_figure(spec)
    fig.save(path)
    return True

# --- Convenience builders ----------------------------------------------------
fn make_scatter(xs: List[Float64], ys: List[Float64], path: String,
                width: Int = 640, height: Int = 480,
                x_min: Float64 = 0.0, x_max: Float64 = 1.0,
                y_min: Float64 = 0.0, y_max: Float64 = 1.0,
                theme_name: Int = 0) -> Bool:
    var s = SeriesSpec()
    s.kind = SeriesKinds.scatter()
    s.xs = xs; s.ys = ys
    var ps = PlotSpec()
    ps.width = width; ps.height = height
    ps.x_min = x_min; ps.x_max = x_max
    ps.y_min = y_min; ps.y_max = y_max
    ps.theme_name = theme_name
    ps.series.push(s)
    return save(ps, path)

# --- Self-test ---------------------------------------------------------------
fn _self_test() -> Bool:
    var xs = List[Float64](); var ys = List[Float64]()
    xs.push(0.0); ys.push(0.0)
    xs.push(0.5); ys.push(0.25)
    xs.push(1.0); ys.push(1.0)
    var s1 = SeriesSpec()
    s1.kind = SeriesKinds.line(); s1.xs = xs; s1.ys = ys; s1.size_or_stroke = 1
    var s2 = SeriesSpec()
    s2.kind = SeriesKinds.scatter(); s2.xs = xs; s2.ys = ys; s2.size_or_stroke = 3
    var ps = PlotSpec()
    ps.series.push(s1); ps.series.push(s2)
    # No file I/O here; just ensure build succeeds and marks exist
    var fig = build_figure(ps)
    return len(fig.scene.marks) == 2